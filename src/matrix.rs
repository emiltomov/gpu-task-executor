use crate::gpu::Gpu;
use anyhow::{Context, Result, bail};
use std::collections::HashMap;
use wgpu::util::DeviceExt;

const MAX_ELEMENTS: usize = 64 * 65535;

#[derive(Clone, Debug, PartialEq)]
pub struct Matrix {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<f32>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Op {
    Add,
    Sub,
    Mul,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Params {
    op: u32,
    rows: u32,
    cols: u32,
    inner: u32,
}

impl Matrix {
    fn scalar(v: f32) -> Self {
        Self {
            rows: 1,
            cols: 1,
            data: vec![v],
        }
    }

    fn is_scalar(&self) -> bool {
        self.rows == 1 && self.cols == 1
    }

    /// Parses `[1 2; 3 4]` (brackets optional): rows separated by `;`, values by spaces or commas.
    pub fn parse(text: &str) -> Result<Self> {
        let text = text.trim();
        let text = match (text.strip_prefix('['), text.ends_with(']')) {
            (Some(t), true) => &t[..t.len() - 1],
            (None, false) => text,
            _ => bail!("unbalanced '[' ']'"),
        };
        let mut data = Vec::new();
        let mut cols = 0;
        let mut rows = 0;
        for row in text.split(';').filter(|r| !r.trim().is_empty()) {
            let values = row
                .split(|c: char| c.is_whitespace() || c == ',')
                .filter(|v| !v.is_empty())
                .map(|v| {
                    v.parse::<f32>()
                        .with_context(|| format!("bad number '{v}'"))
                })
                .collect::<Result<Vec<_>>>()?;
            if rows > 0 && values.len() != cols {
                bail!(
                    "row {} has {} values, expected {cols}",
                    rows + 1,
                    values.len()
                );
            }
            cols = values.len();
            rows += 1;
            data.extend(values);
        }
        if rows == 0 {
            bail!("empty matrix");
        }
        Ok(Self { rows, cols, data })
    }

    /// Full-precision `[1 2; 3 4]` literal, readable by [`Matrix::parse`].
    pub fn to_literal(&self) -> String {
        let rows: Vec<String> = self
            .data
            .chunks(self.cols)
            .map(|r| r.iter().map(f32::to_string).collect::<Vec<_>>().join(" "))
            .collect();
        format!("[{}]", rows.join("; "))
    }

    pub fn row_strings(&self) -> Vec<String> {
        let cells: Vec<String> = self.data.iter().map(|&v| fmt_num(v)).collect();
        let width = cells.iter().map(|c| c.len()).max().unwrap_or(0);
        cells
            .chunks(self.cols)
            .map(|r| {
                r.iter()
                    .map(|c| format!("{c:>width$}"))
                    .collect::<Vec<_>>()
                    .join("  ")
            })
            .collect()
    }
}

fn fmt_num(v: f32) -> String {
    let s = format!("{v:.4}");
    let s = s.trim_end_matches('0').trim_end_matches('.');
    if s == "-0" || s.is_empty() {
        "0".into()
    } else {
        s.into()
    }
}

const EXPR_PREFIX: &str = "# expr:";
const MAX_INCLUDE_DEPTH: u32 = 8;

/// Parses lines like `A = [1 2; 3 4]` (blank lines and `#` comments are skipped).
/// `include path` pulls in the matrices of another file.
pub fn parse_definitions(text: &str) -> Result<HashMap<String, Matrix>> {
    let mut vars = HashMap::new();
    parse_into(text, &mut vars, 0)?;
    Ok(vars)
}

fn parse_into(text: &str, vars: &mut HashMap<String, Matrix>, depth: u32) -> Result<()> {
    for (n, line) in text.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        if line.starts_with("output ") {
            continue;
        }
        if let Some(path) = line.strip_prefix("include ") {
            let path = path.trim();
            if depth >= MAX_INCLUDE_DEPTH {
                bail!("line {}: includes nested too deeply", n + 1);
            }
            let content = std::fs::read_to_string(path)
                .with_context(|| format!("line {}: cannot read '{path}'", n + 1))?;
            parse_into(&content, vars, depth + 1)
                .with_context(|| format!("in included file '{path}'"))?;
            continue;
        }
        let (name, body) = line
            .split_once('=')
            .with_context(|| format!("line {}: expected NAME = [values]", n + 1))?;
        let name = name.trim();
        if name.is_empty() || !name.chars().all(|c| c.is_alphanumeric() || c == '_') {
            bail!("line {}: bad name '{name}'", n + 1);
        }
        let m = Matrix::parse(body).with_context(|| format!("line {}", n + 1))?;
        vars.insert(name.to_string(), m);
    }
    Ok(())
}

/// Path from the first `output path` line of the definitions, if any.
pub fn output_path(defs: &str) -> Option<String> {
    defs.lines()
        .find_map(|l| l.trim().strip_prefix("output "))
        .map(|p| p.trim().to_string())
        .filter(|p| !p.is_empty())
}

/// Saves the definitions plus the expression (as a `# expr:` comment, so the file stays includable).
pub fn save_input(path: &str, defs: &str, expr: &str) -> Result<()> {
    let text = format!("{}\n{EXPR_PREFIX} {expr}\n", defs.trim_end());
    std::fs::write(path, text).with_context(|| format!("cannot write '{path}'"))
}

pub fn result_text(m: &Matrix) -> String {
    format!("result = {}\n", m.to_literal())
}

/// Saves a result as `result = [..]`, so the file can be opened or included later.
pub fn save_result(path: &str, m: &Matrix) -> Result<()> {
    std::fs::write(path, result_text(m)).with_context(|| format!("cannot write '{path}'"))
}

/// Returns `(definitions, expression)` from a file written by [`save_input`] (or any definitions file).
pub fn load_input(path: &str) -> Result<(String, Option<String>)> {
    let text = std::fs::read_to_string(path).with_context(|| format!("cannot read '{path}'"))?;
    Ok(split_input(&text))
}

/// Separates the `# expr:` line from the definitions.
pub fn split_input(text: &str) -> (String, Option<String>) {
    let mut expr = None;
    let mut defs = Vec::new();
    for line in text.lines() {
        match line.trim().strip_prefix(EXPR_PREFIX) {
            Some(e) => expr = Some(e.trim().to_string()),
            None => defs.push(line),
        }
    }
    (defs.join("\n").trim_end().to_string(), expr)
}

#[derive(Debug, Clone, PartialEq)]
enum Tok {
    Num(f32),
    Id(String),
    Sym(char),
}

fn tokenize(s: &str) -> Result<Vec<Tok>> {
    let chars: Vec<char> = s.chars().collect();
    let mut toks = Vec::new();
    let mut i = 0;
    while i < chars.len() {
        let c = chars[i];
        if c.is_whitespace() {
            i += 1;
        } else if c.is_ascii_digit() || c == '.' {
            let start = i;
            while i < chars.len() && (chars[i].is_ascii_digit() || chars[i] == '.') {
                i += 1;
            }
            let t: String = chars[start..i].iter().collect();
            toks.push(Tok::Num(
                t.parse().with_context(|| format!("bad number '{t}'"))?,
            ));
        } else if c.is_alphabetic() || c == '_' {
            let start = i;
            while i < chars.len() && (chars[i].is_alphanumeric() || chars[i] == '_') {
                i += 1;
            }
            toks.push(Tok::Id(chars[start..i].iter().collect()));
        } else if "+-*()'".contains(c) {
            toks.push(Tok::Sym(c));
            i += 1;
        } else {
            bail!("unexpected '{c}'");
        }
    }
    Ok(toks)
}

#[derive(Debug)]
enum Expr {
    Num(f32),
    Var(String),
    Neg(Box<Expr>),
    T(Box<Expr>),
    Bin(Op, Box<Expr>, Box<Expr>),
}

struct Parser {
    toks: Vec<Tok>,
    pos: usize,
}

impl Parser {
    fn peek(&self) -> Option<&Tok> {
        self.toks.get(self.pos)
    }

    fn eat(&mut self, c: char) -> bool {
        if self.peek() == Some(&Tok::Sym(c)) {
            self.pos += 1;
            true
        } else {
            false
        }
    }

    fn expr(&mut self) -> Result<Expr> {
        let mut l = self.term()?;
        loop {
            let op = if self.eat('+') {
                Op::Add
            } else if self.eat('-') {
                Op::Sub
            } else {
                return Ok(l);
            };
            l = Expr::Bin(op, Box::new(l), Box::new(self.term()?));
        }
    }

    fn term(&mut self) -> Result<Expr> {
        let mut l = self.unary()?;
        while self.eat('*') {
            l = Expr::Bin(Op::Mul, Box::new(l), Box::new(self.unary()?));
        }
        Ok(l)
    }

    fn unary(&mut self) -> Result<Expr> {
        if self.eat('-') {
            return Ok(Expr::Neg(Box::new(self.unary()?)));
        }
        let mut e = self.atom()?;
        while self.eat('\'') {
            e = Expr::T(Box::new(e));
        }
        Ok(e)
    }

    fn atom(&mut self) -> Result<Expr> {
        let t = self.peek().cloned();
        self.pos += 1;
        match t {
            Some(Tok::Num(n)) => Ok(Expr::Num(n)),
            Some(Tok::Id(n)) => Ok(Expr::Var(n)),
            Some(Tok::Sym('(')) => {
                let e = self.expr()?;
                if !self.eat(')') {
                    bail!("missing ')'");
                }
                Ok(e)
            }
            Some(Tok::Sym(c)) => bail!("unexpected '{c}'"),
            None => bail!("unexpected end of expression"),
        }
    }
}

/// Backend that executes one matrix operation (`None` second operand = transpose).
pub trait Backend {
    fn binary(&self, op: Op, a: &Matrix, b: &Matrix) -> Result<Matrix>;
    fn transpose(&self, a: &Matrix) -> Result<Matrix>;
}

fn eval(e: &Expr, vars: &HashMap<String, Matrix>, be: &dyn Backend) -> Result<Matrix> {
    Ok(match e {
        Expr::Num(n) => Matrix::scalar(*n),
        Expr::Var(n) => vars
            .get(n)
            .with_context(|| format!("unknown matrix '{n}'"))?
            .clone(),
        Expr::Neg(x) => be.binary(Op::Mul, &eval(x, vars, be)?, &Matrix::scalar(-1.0))?,
        Expr::T(x) => be.transpose(&eval(x, vars, be)?)?,
        Expr::Bin(op, l, r) => be.binary(*op, &eval(l, vars, be)?, &eval(r, vars, be)?)?,
    })
}

/// Evaluates e.g. `(A + B) * C'` against the definitions, composing operations.
pub fn evaluate(defs: &str, expr: &str, be: &dyn Backend) -> Result<Matrix> {
    let vars = parse_definitions(defs)?;
    let mut p = Parser {
        toks: tokenize(expr)?,
        pos: 0,
    };
    let e = p.expr()?;
    if p.pos < p.toks.len() {
        bail!("unexpected trailing input");
    }
    eval(&e, &vars, be)
}

impl Backend for Gpu {
    fn binary(&self, op: Op, a: &Matrix, b: &Matrix) -> Result<Matrix> {
        match op {
            Op::Add | Op::Sub => {
                if (a.rows, a.cols) != (b.rows, b.cols) {
                    bail!(
                        "shape mismatch: {}x{} {} {}x{}",
                        a.rows,
                        a.cols,
                        if op == Op::Add { "+" } else { "-" },
                        b.rows,
                        b.cols
                    );
                }
                self.dispatch(if op == Op::Add { 0 } else { 1 }, a, b, a.rows, a.cols, 0)
            }
            Op::Mul if b.is_scalar() => self.dispatch(4, a, b, a.rows, a.cols, 0),
            Op::Mul if a.is_scalar() => self.dispatch(4, b, a, b.rows, b.cols, 0),
            Op::Mul => {
                if a.cols != b.rows {
                    bail!(
                        "shape mismatch: {}x{} * {}x{}",
                        a.rows,
                        a.cols,
                        b.rows,
                        b.cols
                    );
                }
                self.dispatch(2, a, b, a.rows, b.cols, a.cols)
            }
        }
    }

    fn transpose(&self, a: &Matrix) -> Result<Matrix> {
        self.dispatch(3, a, &Matrix::scalar(0.0), a.cols, a.rows, 0)
    }
}

fn entry(binding: u32, buf: &wgpu::Buffer) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry {
        binding,
        resource: buf.as_entire_binding(),
    }
}

impl Gpu {
    fn dispatch(
        &self,
        op: u32,
        a: &Matrix,
        b: &Matrix,
        rows: usize,
        cols: usize,
        inner: usize,
    ) -> Result<Matrix> {
        let n = rows * cols;
        if n > MAX_ELEMENTS {
            bail!("result too large ({rows}x{cols})");
        }
        let storage = |data: &[f32]| {
            self.device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: None,
                    contents: bytemuck::cast_slice(data),
                    usage: wgpu::BufferUsages::STORAGE,
                })
        };
        let (buf_a, buf_b) = (storage(&a.data), storage(&b.data));
        let bytes = (n * 4) as u64;
        let out = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let readback = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let params = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytemuck::bytes_of(&Params {
                    op,
                    rows: rows as u32,
                    cols: cols as u32,
                    inner: inner as u32,
                }),
                usage: wgpu::BufferUsages::UNIFORM,
            });
        let module = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("matrix"),
                source: wgpu::ShaderSource::Wgsl(include_str!("matrix.wgsl").into()),
            });
        let pipeline = self
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("matrix"),
                layout: None,
                module: &module,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            });
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                entry(0, &buf_a),
                entry(1, &buf_b),
                entry(2, &out),
                entry(3, &params),
            ],
        });
        let mut encoder = self.device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups((n as u32).div_ceil(64), 1, 1);
        }
        encoder.copy_buffer_to_buffer(&out, 0, &readback, 0, bytes);
        self.queue.submit([encoder.finish()]);
        let (tx, rx) = std::sync::mpsc::channel();
        readback.slice(..).map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        self.device.poll(wgpu::PollType::wait_indefinitely())?;
        rx.recv()??;
        let data = {
            let view = readback.slice(..).get_mapped_range()?;
            bytemuck::cast_slice(&view).to_vec()
        };
        readback.unmap();
        Ok(Matrix { rows, cols, data })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Cpu;

    impl Backend for Cpu {
        fn binary(&self, op: Op, a: &Matrix, b: &Matrix) -> Result<Matrix> {
            let mut m = Matrix {
                rows: a.rows,
                cols: a.cols,
                data: vec![],
            };
            match op {
                Op::Add | Op::Sub => {
                    let s = if op == Op::Add { 1.0 } else { -1.0 };
                    m.data = a.data.iter().zip(&b.data).map(|(x, y)| x + s * y).collect();
                }
                Op::Mul if b.is_scalar() => m.data = a.data.iter().map(|x| x * b.data[0]).collect(),
                Op::Mul if a.is_scalar() => {
                    m = Matrix {
                        rows: b.rows,
                        cols: b.cols,
                        data: b.data.iter().map(|x| x * a.data[0]).collect(),
                    }
                }
                Op::Mul => {
                    m = Matrix {
                        rows: a.rows,
                        cols: b.cols,
                        data: vec![0.0; a.rows * b.cols],
                    };
                    for r in 0..a.rows {
                        for c in 0..b.cols {
                            for k in 0..a.cols {
                                m.data[r * b.cols + c] +=
                                    a.data[r * a.cols + k] * b.data[k * b.cols + c];
                            }
                        }
                    }
                }
            }
            Ok(m)
        }

        fn transpose(&self, a: &Matrix) -> Result<Matrix> {
            let mut data = vec![0.0; a.data.len()];
            for r in 0..a.rows {
                for c in 0..a.cols {
                    data[c * a.rows + r] = a.data[r * a.cols + c];
                }
            }
            Ok(Matrix {
                rows: a.cols,
                cols: a.rows,
                data,
            })
        }
    }

    const DEFS: &str = "A = [1 2; 3 4]\nB = [0 1; 1 0]\nC = [1 2 3]";

    #[test]
    fn composes_operations_with_precedence() {
        // A*B = [2 1; 4 3]; A' = [1 3; 2 4]
        let m = evaluate(DEFS, "A * B + A'", &Cpu).unwrap();
        assert_eq!(m.data, vec![3.0, 4.0, 6.0, 7.0]);
        let m = evaluate(DEFS, "2 * (A - B)'", &Cpu).unwrap();
        assert_eq!(m.data, vec![2.0, 4.0, 2.0, 8.0]);
        assert_eq!(
            evaluate(DEFS, "-C", &Cpu).unwrap().data,
            vec![-1.0, -2.0, -3.0]
        );
    }

    #[test]
    fn saves_loads_and_includes_files() {
        let dir = std::env::temp_dir().join(format!("gte-matrix-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let file = dir.join("in.txt");
        let path = file.to_str().unwrap();
        save_input(path, "A = [1 2; 3 4]", "A'").unwrap();
        let (defs, expr) = load_input(path).unwrap();
        assert_eq!(defs, "A = [1 2; 3 4]");
        assert_eq!(expr.as_deref(), Some("A'"));

        let m = evaluate(&format!("include {path}\nB = [1 0; 0 1]"), "A * B", &Cpu).unwrap();
        assert_eq!(m.data, vec![1.0, 2.0, 3.0, 4.0]);
        let out = dir.join("out.txt");
        let out = out.to_str().unwrap();
        save_result(out, &m).unwrap();
        let back = evaluate(&format!("include {out}"), "result", &Cpu).unwrap();
        assert_eq!(back, m);
        assert_eq!(
            output_path("A = [1]\n output  r.txt "),
            Some("r.txt".into())
        );
        assert_eq!(output_path("A = [1]"), None);
        assert!(evaluate("output r.txt\nA = [1]", "A", &Cpu).is_ok());
        assert!(evaluate("include /nonexistent/x.txt", "1", &Cpu).is_err());
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn reports_errors() {
        assert!(Matrix::parse("1 2; 3").is_err());
        assert!(Matrix::parse("[1 2").is_err());
        assert_eq!(
            Matrix::parse("[1,2;3,4]").unwrap(),
            Matrix::parse("1 2; 3 4").unwrap()
        );
        assert!(evaluate(DEFS, "A + Z", &Cpu).is_err());
        assert!(evaluate(DEFS, "(A", &Cpu).is_err());
    }

    #[test]
    fn gpu_matches_cpu() {
        let Ok(gpu) = Gpu::new() else { return };
        for e in ["A * B + A'", "2 * (A - B)'", "C * C' - 3", "A * (B + A)"] {
            let g = evaluate(DEFS, e, &gpu);
            let c = evaluate(DEFS, e, &Cpu);
            assert_eq!(g.is_ok(), c.is_ok(), "{e}");
            if let (Ok(g), Ok(c)) = (g, c) {
                assert_eq!(g, c, "{e}");
            }
        }
        assert!(evaluate(DEFS, "A + C", &gpu).is_err());
    }
}
