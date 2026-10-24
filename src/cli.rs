use crate::gpu::{Gpu, list_adapters};
use crate::matrix::{evaluate, result_text, save_result, split_input};
use anyhow::{Context, Result, bail};
use clap::Parser;
use std::io::Read;

const EXAMPLES: &str = "\
Without --input/--expression the interactive UI starts.

Input file (definitions), one per line:
  A = [1 2; 3 4]
  B = [0 1; 1 0]
  include other.txt          load matrices from another file
  # expr: A * B + A'         optional default expression

Expression file: + - * ( ) ' (transpose); a number scales, e.g. (A + B) * C' - 2 * A
Linear transformations (degrees, applied to column vectors): rot(a) rotx(a) roty(a) rotz(a)
  scale(sx, sy[, sz]) shearx(k) sheary(k) identity(n) reflect(a) proj(a), e.g. rot(90) * v
Use '-' for stdin. The result is printed as 'result = [..]' (or written to --output),
so it can be piped into another run:

  gtu --list-gpus                    # index: name (backend, type)
  gtu -g 1 -i matrices.txt -e expression.txt   # run on GPU 1 (also selects the GPU for the UI)
  gtu -i matrices.txt -e expression.txt -o result.txt
  gtu -i matrices.txt -e expression.txt | gtu -i - -e next.txt
  echo \"A * B\" | gtu -i matrices.txt -e -
  cat matrices.txt | gtu -e expression.txt";

/// GPU matrix calculator: run without options for the interactive UI.
#[derive(Parser)]
#[command(name = "gtu", version, after_help = EXAMPLES)]
pub struct Cli {
    /// File with matrix definitions ('-' = stdin; default when only --expression is given)
    #[arg(short, long, value_name = "FILE")]
    input: Option<String>,

    /// File with the expression ('-' = stdin; default: the '# expr:' line of the input)
    #[arg(short, long, value_name = "FILE")]
    expression: Option<String>,

    /// File to write the result to (default: print to stdout)
    #[arg(short, long, value_name = "FILE")]
    output: Option<String>,

    /// GPU to use, by index from --list-gpus (default: automatic high-performance choice)
    #[arg(short, long, value_name = "INDEX")]
    pub gpu: Option<usize>,

    /// List the GPUs visible to gtu and exit
    #[arg(short, long)]
    list_gpus: bool,
}

impl Cli {
    pub fn is_batch(&self) -> bool {
        self.input.is_some() || self.expression.is_some() || self.output.is_some() || self.list_gpus
    }
}

fn read(path: &str) -> Result<String> {
    if path == "-" {
        let mut s = String::new();
        std::io::stdin()
            .read_to_string(&mut s)
            .context("cannot read stdin")?;
        Ok(s)
    } else {
        std::fs::read_to_string(path).with_context(|| format!("cannot read '{path}'"))
    }
}

/// Joins the expression lines, skipping blank lines and `#` comments.
fn clean_expression(text: &str) -> String {
    text.lines()
        .map(str::trim)
        .filter(|l| !l.is_empty() && !l.starts_with('#'))
        .collect::<Vec<_>>()
        .join(" ")
}

pub fn open_gpu(index: Option<usize>) -> Result<Gpu> {
    match index {
        None => Gpu::new(),
        Some(i) => Gpu::with_adapter(i)
            .with_context(|| format!("GPU {i} not available; see `gtu --list-gpus`")),
    }
}

fn list_gpus() {
    let adapters = list_adapters();
    if adapters.is_empty() {
        println!("no GPUs found");
    }
    for (i, a) in adapters.iter().enumerate() {
        println!("{i}: {} ({:?}, {:?})", a.name, a.backend, a.device_type);
    }
}

pub fn run(cli: &Cli) -> Result<()> {
    if cli.list_gpus {
        list_gpus();
        return Ok(());
    }
    let input = cli.input.as_deref().unwrap_or("-");
    if input == "-" && cli.expression.as_deref() == Some("-") {
        bail!("--input and --expression cannot both read from stdin");
    }
    let (defs, default_expr) = split_input(&read(input)?);
    let expr = match &cli.expression {
        Some(p) => clean_expression(&read(p)?),
        None => default_expr.context("no expression: pass --expression or add a '# expr:' line")?,
    };
    if expr.is_empty() {
        bail!("expression is empty");
    }
    let gpu = open_gpu(cli.gpu)?;
    let result = evaluate(&defs, &expr, &gpu)?;
    match &cli.output {
        Some(path) => save_result(path, &result),
        None => {
            print!("{}", result_text(&result));
            Ok(())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn expression_skips_comments_and_joins_lines() {
        assert_eq!(clean_expression("# note\nA *\n  B\n\n"), "A * B");
    }

    #[test]
    fn gpu_option_parses_and_list_is_batch() {
        let cli = Cli::parse_from(["gtu", "-g", "2"]);
        assert_eq!(cli.gpu, Some(2));
        assert!(!cli.is_batch());
        assert!(Cli::parse_from(["gtu", "--list-gpus"]).is_batch());
        assert!(open_gpu(Some(9999)).is_err());
    }

    #[test]
    fn batch_mode_needs_an_option() {
        assert!(!Cli::parse_from(["gtu"]).is_batch());
        assert!(Cli::parse_from(["gtu", "-i", "m.txt"]).is_batch());
    }

    #[test]
    fn runs_from_files_to_output_file() {
        if Gpu::new().is_err() {
            return;
        }
        let dir = std::env::temp_dir().join(format!("gte-cli-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let p = |n: &str| dir.join(n).to_str().unwrap().to_string();
        std::fs::write(p("in.txt"), "A = [1 2; 3 4]\nB = [0 1; 1 0]\n").unwrap();
        std::fs::write(p("e.txt"), "A * B\n").unwrap();
        let cli = Cli::parse_from([
            "gtu",
            "-i",
            &p("in.txt"),
            "-e",
            &p("e.txt"),
            "-o",
            &p("out.txt"),
        ]);
        run(&cli).unwrap();
        assert_eq!(
            std::fs::read_to_string(p("out.txt")).unwrap(),
            "result = [2 1; 4 3]\n"
        );
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
