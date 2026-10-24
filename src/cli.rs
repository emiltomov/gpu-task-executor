use crate::examples::EXAMPLES as BUNDLED;
use crate::gpu::{Gpu, list_adapters};
use crate::matrix::{clean_expression, evaluate, result_text, save_result, split_input};
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
  cat matrices.txt | gtu -e expression.txt

Piping: every run prints 'result = [..]', which is itself a valid input file, so runs chain:
  gtu -i a.txt -e step1.txt | gtu -i - -e step2.txt | gtu -i - -e step3.txt -o final.txt
  (the result is named 'result'; refer to it as 'result' in the next expression)
Only one of --input/--expression can read stdin. Errors go to stderr with exit code 1,
so `gtu ... && next` and `set -o pipefail` work in scripts.

Bundled algorithms (Markov chains, graph search, lighting/photo):
  gtu --list-examples                 # number: title
  gtu --example 2                     # run example 2 (number or part of the title)
  gtu --example 2 --show-example      # print its input file, to edit and re-run via -i -
  gtu --example 'steady' -o out.txt
  gtu --example 3 | gtu -i - -e next.txt";

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

    /// Run a bundled algorithm example by number or part of its title
    #[arg(long, value_name = "N|TITLE")]
    example: Option<String>,

    /// With --example: print the example as an input file instead of running it
    #[arg(long, requires = "example")]
    show_example: bool,

    /// List the bundled algorithm examples and exit
    #[arg(long)]
    list_examples: bool,

    /// List the GPUs visible to gtu and exit
    #[arg(short, long)]
    list_gpus: bool,
}

impl Cli {
    pub fn is_batch(&self) -> bool {
        self.input.is_some()
            || self.expression.is_some()
            || self.output.is_some()
            || self.list_gpus
            || self.example.is_some()
            || self.list_examples
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

fn find_example(key: &str) -> Result<&'static crate::examples::Example> {
    if let Ok(n) = key.parse::<usize>() {
        return BUNDLED
            .get(n.wrapping_sub(1))
            .with_context(|| format!("no example {n}; see `gtu --list-examples`"));
    }
    let key = key.to_lowercase();
    let mut hits = BUNDLED
        .iter()
        .filter(|e| e.title.to_lowercase().contains(&key));
    match (hits.next(), hits.next()) {
        (Some(e), None) => Ok(e),
        (None, _) => bail!("no example matches '{key}'; see `gtu --list-examples`"),
        _ => bail!("'{key}' matches several examples; use its number"),
    }
}

pub fn run(cli: &Cli) -> Result<()> {
    if cli.list_examples {
        for (i, e) in BUNDLED.iter().enumerate() {
            println!("{}: {}", i + 1, e.title);
        }
        return Ok(());
    }
    if cli.list_gpus {
        list_gpus();
        return Ok(());
    }
    if let Some(key) = &cli.example {
        let (defs, expr) = find_example(key)?.load();
        if cli.show_example {
            print!("{}\n# expr: {expr}\n", defs.trim_end());
            return Ok(());
        }
        return finish(cli, &defs, &expr);
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
    finish(cli, &defs, &expr)
}

fn finish(cli: &Cli, defs: &str, expr: &str) -> Result<()> {
    if expr.is_empty() {
        bail!("expression is empty");
    }
    let gpu = open_gpu(cli.gpu)?;
    let result = evaluate(defs, expr, &gpu)?;
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
    fn gpu_option_parses_and_list_is_batch() {
        let cli = Cli::parse_from(["gtu", "-g", "2"]);
        assert_eq!(cli.gpu, Some(2));
        assert!(!cli.is_batch());
        assert!(Cli::parse_from(["gtu", "--list-gpus"]).is_batch());
        assert!(open_gpu(Some(9999)).is_err());
    }

    #[test]
    fn examples_are_found_by_number_or_title() {
        assert_eq!(find_example("1").unwrap().title, BUNDLED[0].title);
        assert!(find_example("steady").is_ok());
        assert!(find_example("0").is_err());
        assert!(find_example("nonexistent").is_err());
        assert!(Cli::parse_from(["gtu", "--list-examples"]).is_batch());
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
