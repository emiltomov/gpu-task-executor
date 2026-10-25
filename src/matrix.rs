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

pub fn fmt_num(v: f32) -> String {
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
        if line.starts_with("output ") || line.starts_with("graph ") {
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
        let m = match body.trim().strip_prefix("image ") {
            Some(spec) => load_image(spec.trim()),
            None => Matrix::parse(body),
        }
        .with_context(|| format!("line {}", n + 1))?;
        vars.insert(name.to_string(), m);
    }
    Ok(())
}

const MAX_IMAGE_SIDE: u32 = 128;
const EARTH_PNG: &[u8] = include_bytes!("../examples/lighting/earth.png");

/// Loads a grayscale image (0..1) as a matrix; `builtin:earth` is a bundled NASA photo.
fn load_image(spec: &str) -> Result<Matrix> {
    let img = if spec == "builtin:earth" {
        image::load_from_memory(EARTH_PNG)?
    } else {
        image::open(spec).with_context(|| format!("cannot read image '{spec}'"))?
    };
    let img = if img.width().max(img.height()) > MAX_IMAGE_SIDE {
        img.thumbnail(MAX_IMAGE_SIDE, MAX_IMAGE_SIDE)
    } else {
        img
    };
    let gray = img.to_luma8();
    Ok(Matrix {
        rows: gray.height() as usize,
        cols: gray.width() as usize,
        data: gray.pixels().map(|p| p.0[0] as f32 / 255.0).collect(),
    })
}

/// Matrix name from the first `graph NAME` line of the definitions, if any.
pub fn graph_name(defs: &str) -> Option<String> {
    defs.lines()
        .find_map(|l| l.trim().strip_prefix("graph "))
        .map(|p| p.trim().to_string())
        .filter(|p| !p.is_empty())
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

/// Joins the expression lines, skipping blank lines and `#` comments.
pub fn clean_expression(text: &str) -> String {
    text.lines()
        .map(str::trim)
        .filter(|l| !l.is_empty() && !l.starts_with('#'))
        .collect::<Vec<_>>()
        .join(" ")
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
        } else if "+-*(),'".contains(c) {
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
    Call(String, Vec<Expr>),
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
            Some(Tok::Id(n)) if self.eat('(') => {
                let mut args = Vec::new();
                if !self.eat(')') {
                    loop {
                        args.push(self.expr()?);
                        if self.eat(')') {
                            break;
                        }
                        if !self.eat(',') {
                            bail!("expected ',' or ')' in call to '{n}'");
                        }
                    }
                }
                Ok(Expr::Call(n, args))
            }
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
        Expr::Call(name, args) if name == "det" || name == "trace" => {
            if args.len() != 1 {
                bail!("{name}() takes 1 argument");
            }
            let m = eval(&args[0], vars, be)?;
            if m.rows != m.cols {
                bail!("{name}() needs a square matrix, got {}x{}", m.rows, m.cols);
            }
            Matrix::scalar(if name == "det" {
                determinant(&m)
            } else {
                trace(&m)
            })
        }
        Expr::Call(name, args) => {
            let mut values = Vec::new();
            for a in args {
                let m = eval(a, vars, be)?;
                // A single vector argument expands into several, e.g. scale(v).
                if args.len() == 1 && (m.rows == 1 || m.cols == 1) {
                    values.extend(&m.data);
                } else if m.rows == 1 && m.cols == 1 {
                    values.push(m.data[0]);
                } else {
                    bail!(
                        "arguments of {name}() must be numbers, got {}x{} (reduce with det() or trace())",
                        m.rows,
                        m.cols
                    );
                }
            }
            transformation(name, &values)?
        }
        Expr::Neg(x) => be.binary(Op::Mul, &eval(x, vars, be)?, &Matrix::scalar(-1.0))?,
        Expr::T(x) => be.transpose(&eval(x, vars, be)?)?,
        Expr::Bin(op, l, r) => {
            let (a, b) = (eval(l, vars, be)?, eval(r, vars, be)?);
            // A number added to or subtracted from a matrix applies to every element.
            let (a, b) = match op {
                Op::Add | Op::Sub if b.is_scalar() && !a.is_scalar() => {
                    let f = Matrix::filled(a.rows, a.cols, b.data[0]);
                    (a, f)
                }
                Op::Add | Op::Sub if a.is_scalar() && !b.is_scalar() => {
                    (Matrix::filled(b.rows, b.cols, a.data[0]), b)
                }
                _ => (a, b),
            };
            be.binary(*op, &a, &b)?
        }
    })
}

impl Matrix {
    fn filled(rows: usize, cols: usize, v: f32) -> Matrix {
        Matrix {
            rows,
            cols,
            data: vec![v; rows * cols],
        }
    }
}

fn trace(m: &Matrix) -> f32 {
    (0..m.rows).map(|i| m.data[i * m.cols + i]).sum()
}

/// Determinant by Gaussian elimination with partial pivoting.
fn determinant(m: &Matrix) -> f32 {
    let n = m.rows;
    let mut a: Vec<f64> = m.data.iter().map(|&v| v as f64).collect();
    let mut det = 1.0;
    for c in 0..n {
        let p = (c..n)
            .max_by(|&x, &y| a[x * n + c].abs().total_cmp(&a[y * n + c].abs()))
            .unwrap();
        if a[p * n + c] == 0.0 {
            return 0.0;
        }
        if p != c {
            for k in 0..n {
                a.swap(p * n + k, c * n + k);
            }
            det = -det;
        }
        det *= a[c * n + c];
        for r in c + 1..n {
            let f = a[r * n + c] / a[c * n + c];
            for k in c..n {
                a[r * n + k] -= f * a[c * n + k];
            }
        }
    }
    clean(det as f32)
}

fn square(n: usize, f: impl Fn(usize, usize) -> f32) -> Matrix {
    let data = (0..n * n).map(|i| f(i / n, i % n)).collect();
    Matrix {
        rows: n,
        cols: n,
        data,
    }
}

fn matrix_of(rows: &[&[f32]]) -> Matrix {
    Matrix {
        rows: rows.len(),
        cols: rows[0].len(),
        data: rows.iter().flat_map(|r| r.iter().copied()).collect(),
    }
}

fn clean(v: f32) -> f32 {
    if v.abs() < 1e-6 { 0.0 } else { v }
}

fn sin_cos_deg(deg: f32) -> (f32, f32) {
    let (s, c) = deg.to_radians().sin_cos();
    (clean(s), clean(c))
}

/// Builds a linear transformation matrix; angles are in degrees and
/// transformations apply to column vectors (`rot(90) * v`).
fn transformation(name: &str, a: &[f32]) -> Result<Matrix> {
    let argc = |n: usize| -> Result<()> {
        if a.len() != n {
            bail!("{name}() takes {n} argument(s), got {}", a.len());
        }
        Ok(())
    };
    let (sin, cos) = sin_cos_deg(a.first().copied().unwrap_or(0.0));
    let m = match name {
        "identity" => {
            argc(1)?;
            let n = a[0];
            if n.fract() != 0.0 || !(1.0..=1024.0).contains(&n) {
                bail!("identity() needs a size between 1 and 1024");
            }
            square(n as usize, |r, c| if r == c { 1.0 } else { 0.0 })
        }
        "ones" => {
            argc(2)?;
            let dim = |v: f32| -> Result<usize> {
                if v.fract() != 0.0 || !(1.0..=1024.0).contains(&v) {
                    bail!("ones() needs sizes between 1 and 1024");
                }
                Ok(v as usize)
            };
            Matrix::filled(dim(a[0])?, dim(a[1])?, 1.0)
        }
        // n x n blur kernel: each row averages a pixel with its neighbours (1 2 1) / 4.
        "smooth" => {
            argc(1)?;
            let n = a[0];
            if n.fract() != 0.0 || !(2.0..=1024.0).contains(&n) {
                bail!("smooth() needs a size between 2 and 1024");
            }
            let n = n as usize;
            square(n, |r, c| {
                let edge = r == 0 || r == n - 1;
                match r.abs_diff(c) {
                    0 if edge => 0.75,
                    0 => 0.5,
                    1 => 0.25,
                    _ => 0.0,
                }
            })
        }
        "scale" => {
            if a.is_empty() {
                bail!("scale() needs at least one factor");
            }
            square(a.len(), |r, c| if r == c { a[r] } else { 0.0 })
        }
        "rot" => {
            argc(1)?;
            matrix_of(&[&[cos, -sin], &[sin, cos]])
        }
        "rotx" => {
            argc(1)?;
            matrix_of(&[&[1.0, 0.0, 0.0], &[0.0, cos, -sin], &[0.0, sin, cos]])
        }
        "roty" => {
            argc(1)?;
            matrix_of(&[&[cos, 0.0, sin], &[0.0, 1.0, 0.0], &[-sin, 0.0, cos]])
        }
        "rotz" => {
            argc(1)?;
            matrix_of(&[&[cos, -sin, 0.0], &[sin, cos, 0.0], &[0.0, 0.0, 1.0]])
        }
        "shearx" => {
            argc(1)?;
            matrix_of(&[&[1.0, a[0]], &[0.0, 1.0]])
        }
        "sheary" => {
            argc(1)?;
            matrix_of(&[&[1.0, 0.0], &[a[0], 1.0]])
        }
        // Mirror across the line through the origin at the given angle.
        "reflect" => {
            argc(1)?;
            let (s2, c2) = sin_cos_deg(2.0 * a[0]);
            matrix_of(&[&[c2, s2], &[s2, -c2]])
        }
        // Orthogonal projection onto the line through the origin at the given angle.
        "proj" => {
            argc(1)?;
            matrix_of(&[&[cos * cos, cos * sin], &[cos * sin, sin * sin]])
        }
        _ => bail!("unknown function '{name}'"),
    };
    Ok(m)
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
    fn linear_transformations() {
        let defs = "v = [1; 0]\nP = [1 2; 3 4]";
        let near = |m: Matrix, want: &[f32]| {
            assert_eq!(m.data.len(), want.len());
            assert!(
                m.data.iter().zip(want).all(|(a, b)| (a - b).abs() < 1e-5),
                "{m:?}"
            );
        };
        near(evaluate(defs, "rot(90) * v", &Cpu).unwrap(), &[0.0, 1.0]);
        near(
            evaluate(defs, "scale(2, 3) * P", &Cpu).unwrap(),
            &[2.0, 4.0, 9.0, 12.0],
        );
        near(
            evaluate(defs, "shearx(2) * P", &Cpu).unwrap(),
            &[7.0, 10.0, 3.0, 4.0],
        );
        near(
            evaluate(defs, "reflect(0) * P", &Cpu).unwrap(),
            &[1.0, 2.0, -3.0, -4.0],
        );
        near(
            evaluate(defs, "proj(0) * P", &Cpu).unwrap(),
            &[1.0, 2.0, 0.0, 0.0],
        );
        near(
            evaluate(defs, "rot(30) * rot(60) - rot(90)", &Cpu).unwrap(),
            &[0.0; 4],
        );
        near(
            evaluate(defs, "rotz(90) * scale(1, 2, 3)", &Cpu).unwrap(),
            &[0.0, -2.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 3.0],
        );
        near(
            evaluate(defs, "identity(2) * P", &Cpu).unwrap(),
            &[1.0, 2.0, 3.0, 4.0],
        );
        assert!(evaluate(defs, "rot(1, 2)", &Cpu).is_err());
        assert!(evaluate(defs, "rot(P)", &Cpu).is_err());
        assert!(evaluate(defs, "nope(1)", &Cpu).is_err());
        assert!(evaluate(defs, "rot(30", &Cpu).is_err());
    }

    #[test]
    fn expressions_compose_inside_function_arguments() {
        let defs = "A = [30]\nB = [15]\nM = [2 0; 0 3]\nv = [1 2 3]";
        let a = evaluate(defs, "rot(A * B + A')", &Cpu).unwrap();
        let b = evaluate(defs, "rot(480)", &Cpu).unwrap();
        assert_eq!(a.data.len(), 4);
        assert!((a.data[0] - b.data[0]).abs() < 1e-3);
        assert_eq!(evaluate(defs, "det(M)", &Cpu).unwrap().data, vec![6.0]);
        assert_eq!(
            evaluate(defs, "trace(M * M')", &Cpu).unwrap().data,
            vec![13.0]
        );
        let r = evaluate(defs, "rot(det(M) * 15)", &Cpu).unwrap();
        assert!(r.data[0].abs() < 1e-5 && (r.data[1] + 1.0).abs() < 1e-5);
        assert_eq!(evaluate(defs, "scale(v)", &Cpu).unwrap().rows, 3);
        assert!(evaluate(defs, "rot(M)", &Cpu).is_err());
    }

    #[test]
    fn graph_directive_and_scalar_broadcast() {
        assert_eq!(graph_name("A = [1]\ngraph A\n"), Some("A".into()));
        assert_eq!(graph_name("A = [1]"), None);
        let r = evaluate("A = [1 2; 3 4]", "A - 1 + ones(2, 2)", &Cpu).unwrap();
        assert_eq!(r.data, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn bundled_examples_evaluate() {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("examples");
        let read = |p: &str| std::fs::read_to_string(root.join(p)).unwrap();
        let run = |input: &str, expr: &str| {
            let (defs, _) = split_input(&read(input));
            let expr: Vec<String> = read(expr)
                .lines()
                .filter(|l| !l.trim().starts_with('#') && !l.trim().is_empty())
                .map(str::to_string)
                .collect();
            evaluate(&defs, &expr.join(" "), &Cpu).unwrap()
        };
        for (dir, input) in [
            ("markov", "chain"),
            ("graph", "graph"),
            ("lighting", "scene"),
        ] {
            for entry in std::fs::read_dir(root.join(dir)).unwrap() {
                let name = entry.unwrap().file_name().into_string().unwrap();
                if name != format!("{input}.txt")
                    && name.ends_with(".txt")
                    && !name.starts_with("photo")
                {
                    run(&format!("{dir}/{input}.txt"), &format!("{dir}/{name}"));
                }
            }
        }
        let steady = run("markov/chain.txt", "markov/steady.txt");
        assert!((steady.data[0] - 5.0 / 6.0).abs() < 1e-3);
        assert_eq!(
            run("graph/graph.txt", "graph/degrees.txt").data,
            vec![2.0, 2.0, 3.0, 2.0, 1.0]
        );
        assert_eq!(
            run("graph/graph.txt", "graph/neighbors.txt").data,
            vec![1.0, 1.0, 0.0, 1.0, 0.0]
        );
        assert!(
            run("graph/graph.txt", "graph/reach3.txt")
                .data
                .iter()
                .all(|&v| v > 0.0)
        );
    }

    #[test]
    fn expression_skips_comments_and_joins_lines() {
        assert_eq!(clean_expression("# note\nA *\n  B\n\n"), "A * B");
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
