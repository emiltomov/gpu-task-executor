use crate::editor::Editor;
use crate::examples::EXAMPLES;
use crate::gpu::Gpu;
use crate::graph::{self, Graph, View};
use crate::matrix::{
    Matrix, evaluate_vars, graph_spec, load_input, output_path, parse_definitions, save_input,
    save_result, size_value,
};
use ratatui::{
    Frame,
    crossterm::event::{KeyCode, KeyEvent, KeyModifiers},
    layout::{Constraint, Layout, Rect},
    style::{Color, Style},
    text::{Line, Span},
    widgets::{Block, Clear, List, ListState, Paragraph},
};

#[derive(Clone, Copy, PartialEq)]
enum Focus {
    Defs,
    Expr,
    Back,
}

#[derive(Clone, Copy, PartialEq, Debug)]
enum ExportKind {
    Maze,
    Image,
}

impl ExportKind {
    fn default_name(self) -> &'static str {
        match self {
            ExportKind::Maze => "maze.zip",
            ExportKind::Image => "image.png",
        }
    }
}

const FEED_UNAVAILABLE: &str =
    "Ctrl+N/Ctrl+P need a picture result: calculate an image example first";

const EXPORT_UNAVAILABLE: &str =
    "Ctrl+E exports only maze (graph ... grid), image and raycast examples";

#[derive(Clone, Copy, PartialEq)]
enum Prompt {
    Save,
    SaveResult,
    Export(ExportKind),
    Open,
}

/// System clipboard with an in-process fallback when none is available.
#[derive(Default)]
struct Clip {
    system: Option<arboard::Clipboard>,
    local: String,
}

impl Clip {
    fn system(&mut self) -> Option<&mut arboard::Clipboard> {
        if self.system.is_none() {
            self.system = arboard::Clipboard::new().ok();
        }
        self.system.as_mut()
    }

    fn set(&mut self, text: String) {
        if let Some(c) = self.system() {
            let _ = c.set_text(text.clone());
        }
        self.local = text;
    }

    fn get(&mut self) -> String {
        self.system()
            .and_then(|c| c.get_text().ok())
            .unwrap_or_else(|| self.local.clone())
    }
}

pub struct MatrixUi {
    defs: Editor,
    expr: Editor,
    /// Inverse expression run by Ctrl+P (image examples).
    back: Editor,
    clipboard: Clip,
    /// File path prompt shown above the result while saving/opening.
    prompt: Option<(Prompt, Editor)>,
    status: Option<Result<String, String>>,
    /// Example picker, shown over the whole screen while open.
    examples: Option<ListState>,
    /// Graph drawn next to the result when the definitions contain `graph NAME`.
    graph: Option<(String, Graph)>,
    graph_view: View,
    /// While set, arrows pan and +/- zoom the graph instead of editing.
    graph_focus: bool,
    /// Legend popup for the graph symbols and colors.
    legend: bool,
    /// Name of the loaded example or file, shown in the result title.
    source: Option<String>,
    /// Input picture matching a large (image-sized) result, drawn next to it.
    before: Option<Matrix>,
    before_name: Option<String>,
    /// Result fed back as the input picture: (definitions it applies to, variable, picture).
    feed: Option<(String, String, Matrix)>,
    /// Result shown before the last Ctrl+N, so Ctrl+P can restore it exactly.
    prev_input: Option<Matrix>,
    back_default: String,
    example_name: Option<String>,
    expr_help: bool,
    /// First visible line of the definitions editor.
    defs_scroll: usize,
    /// Scroll position (row, column) of the numeric result.
    result_scroll: (usize, usize),
    focus: Focus,
    result: Option<Result<Matrix, String>>,
}

pub const EXPR_HELP: (&str, &[&str]) = (
    "Expression operations",
    &[
        "A + B   A - B      element-wise; a number also works: A - 1, 2 + A",
        "A * B              matrix product; a number scales: 2 * A",
        "A'                 transpose        -A  negate        ( )  grouping",
        "Compose freely:    (A + B) * C' - 2 * A",
        "Linear transformations (degrees, apply to column vectors):",
        "  rot(a)                 2D rotation",
        "  rotx(a) roty(a) rotz(a)  3D rotations",
        "  scale(sx, sy[, sz])    scaling",
        "  shearx(k) sheary(k)    shear",
        "  reflect(a)             mirror across the line at angle a",
        "  proj(a)                project onto the line at angle a",
        "  identity(n)            n x n identity",
        "  det(M)  trace(M)       matrix -> number, usable inside any argument",
        "Arguments may be any expression giving a number (1x1), or one vector:",
        "  rot(det(A * B) + 10)   scale(v)",
        "  ones(r, c)             matrix of ones",
        "Graph search on an adjacency matrix (result is a node vector):",
        "  bfs(A, s)              hops from node s + 1 for every reachable node",
        "  raycast(I, a)          image I seen by a camera orbited a degrees around y (one ray per pixel)",
        "  astar(A, s, g[, w])    shortest path s -> g as steps 1, 2, ...; w = grid",
        "                         width enables the Manhattan heuristic",
        "  smooth(n)              n x n blur kernel (1 2 1)/4",
        "  rows(M), cols(M)       size of a matrix, e.g. flip(rows(I)), resize(I, 2 * rows(I), 2 * cols(I))",
        "  flip(n)                exchange matrix: flip(n) * I mirrors rows, I * flip(n) columns",
        "  resize(I, rows, cols)  scale an image up or down (shrinking averages pixels)",
        "e.g.  rot(90) * v     rotz(30) * scale(1, 2, 3) * P",
        "      smooth(rows(I)) * I * smooth(cols(I))'",
        "Esc / F3 close",
    ],
);

pub const HELP: (&str, &[&str]) = (
    "Matrices",
    &[
        "Declare:  A = [1 2; 3 4]   (one per line)",
        "Include:  include path/to/file.txt",
        "Output:   output path/to/result.txt  (definitions line; the result",
        "          is written there on every calculation)",
        "Expr:     operators and functions: press F3",
        "PgUp/PgDn, Ctrl+arrows  scroll a large result",
        "Tab            switch definitions / expression",
        "Enter          calculate (in expression), F5 anywhere",
        "Arrows, Home/End, Shift  move / select",
        "Ctrl+C/X/V/A   copy / cut / paste / select all",
        "Ctrl+S         save input    Ctrl+O  open .txt / maze .zip / image",
        "Ctrl+R         save result once (asks for a path)",
        "Ctrl+E         export a `graph NAME grid W` maze + A*/BFS solutions as zip;",
        "               image and raycast results as PNG (only in those cases)",
        "Ctrl+N         image examples: use the latest result as the input picture and",
        "               calculate again (repeat to chain; reloading/editing definitions resets)",
        "Ctrl+P         apply the back expression (box below the expression, from the",
        "               example's `# back:` line, editable) to the latest result = undo Ctrl+N",
        "F2             load an example (also in the main menu)",
        "graph NAME [grid W]  (in definitions) draw matrix NAME as a graph (grid: node i at column i % W)",
        "M = maze W H [seed]  random maze; W, H, seed: numbers or variables (W = 21)",
        "F3             expression help: all operations",
        "F4             graph view: arrows pan, +/- zoom, 0 reset, F4/Esc back",
        "F6             legend of graph symbols and colors (also L in graph view)",
        "F1             this help",
        "Esc            back",
    ],
);

impl MatrixUi {
    pub fn new() -> Self {
        Self {
            defs: Editor::new("A = [1 2; 3 4]\nB = [0 1; 1 0]"),
            expr: Editor::new("A * B + A'"),
            back: Editor::new(""),
            clipboard: Clip::default(),
            prompt: None,
            status: None,
            examples: None,
            graph: None,
            graph_view: View::default(),
            graph_focus: false,
            legend: false,
            source: None,
            before: None,
            before_name: None,
            feed: None,
            prev_input: None,
            back_default: String::new(),
            example_name: None,
            expr_help: false,
            defs_scroll: 0,
            result_scroll: (0, 0),
            focus: Focus::Expr,
            result: None,
        }
    }

    /// Starts with the example picker open.
    pub fn with_examples() -> Self {
        let mut ui = Self::new();
        ui.examples = Some(ListState::default().with_selected(Some(0)));
        ui
    }

    fn editor(&mut self) -> &mut Editor {
        if let Some((_, ed)) = &mut self.prompt {
            return ed;
        }
        match self.focus {
            Focus::Defs => &mut self.defs,
            Focus::Expr => &mut self.expr,
            Focus::Back => &mut self.back,
        }
    }

    fn calculate(&mut self, gpu: &Gpu) {
        let expr = self.expr.text();
        self.calculate_expr(gpu, &expr);
    }

    fn calculate_expr(&mut self, gpu: &Gpu, expr: &str) {
        self.status = None;
        self.result_scroll = (0, 0);
        let defs = self.defs.text();
        let mut vars = parse_definitions(&defs);
        if let (Ok(vars), Some((fed_defs, name, m))) = (&mut vars, &self.feed) {
            if *fed_defs == defs && vars.contains_key(name) {
                vars.insert(name.clone(), m.clone());
            }
        }
        self.result = Some(
            vars.as_ref()
                .map_err(|e| format!("{e:#}"))
                .and_then(|v| evaluate_vars(v, expr, gpu).map_err(|e| format!("{e:#}"))),
        );
        if let (Some(Ok(m)), Some(path)) = (&self.result, output_path(&defs)) {
            self.status = Some(
                save_result(&path, m)
                    .map(|_| format!("Result saved to {path}"))
                    .map_err(|e| format!("{e:#}")),
            );
        }
        self.graph = None;
        (self.before, self.before_name) = match (&self.result, &vars) {
            (Some(Ok(r)), Ok(vars)) if is_picture(r) => {
                let same = |m: &Matrix| (m.rows, m.cols) == (r.rows, r.cols);
                vars.get_key_value("I")
                    .filter(|(_, m)| same(m))
                    .or_else(|| vars.iter().find(|(_, m)| same(m)))
                    .or_else(|| vars.get_key_value("I").filter(|(_, m)| is_picture(m)))
                    .map_or((None, None), |(k, m)| (Some(m.clone()), Some(k.clone())))
            }
            _ => (None, None),
        };
        if let Some((name, grid)) = graph_spec(&defs) {
            let built = parse_definitions(&defs)
                .ok()
                .and_then(|vars| {
                    let grid = grid.and_then(|w| size_value(&w, &vars));
                    vars.get(&name).cloned().map(|m| (m, grid))
                })
                .ok_or_else(|| format!("graph: unknown matrix '{name}'"))
                .and_then(|(m, grid)| {
                    Graph::from_matrix(&m, grid).map_err(|e| format!("graph: {e:#}"))
                });
            match built {
                Ok(g) => {
                    self.graph = Some((name, g));
                    self.graph_view = View::default();
                }
                Err(e) => self.status = Some(Err(e)),
            }
        }
        if self.graph.is_none() {
            self.graph_focus = false;
        }
    }

    /// Returns `true` when the user wants to leave the screen.
    pub fn handle_key(&mut self, key: KeyEvent, gpu: &Gpu) -> bool {
        if self.expr_help {
            if matches!(key.code, KeyCode::Esc | KeyCode::F(3)) {
                self.expr_help = false;
            }
            return false;
        }
        if let Some(list) = &mut self.examples {
            match key.code {
                KeyCode::Up | KeyCode::Char('k') => list.select_previous(),
                KeyCode::Down | KeyCode::Char('j') => list.select_next(),
                KeyCode::Esc | KeyCode::Char('q') => self.examples = None,
                KeyCode::Enter => {
                    if let Some(ex) = list.selected().and_then(|i| EXAMPLES.get(i)) {
                        let (defs, expr) = ex.load();
                        self.defs = Editor::new(&defs);
                        self.feed = None;
                        self.prev_input = None;
                        self.expr = Editor::new(&expr);
                        self.back = Editor::new(&ex.back());
                        self.back_default = ex.back();
                        self.example_name = Some(ex.title.to_string());
                        self.source = Some(ex.title.to_string());
                        self.focus = Focus::Expr;
                        self.examples = None;
                        self.calculate(gpu);
                    }
                }
                _ => {}
            }
            return false;
        }
        if self.legend {
            if matches!(key.code, KeyCode::Esc | KeyCode::F(6) | KeyCode::Char('l')) {
                self.legend = false;
            }
            return false;
        }
        if self.graph_focus && key.code == KeyCode::Char('l') {
            self.legend = true;
            return false;
        }
        if self.graph_focus {
            match key.code {
                KeyCode::F(6) => self.legend = true,
                KeyCode::Esc | KeyCode::F(4) => self.graph_focus = false,
                KeyCode::Left => self.graph_view.pan(-0.5, 0.0),
                KeyCode::Right => self.graph_view.pan(0.5, 0.0),
                KeyCode::Up => self.graph_view.pan(0.0, 0.5),
                KeyCode::Down => self.graph_view.pan(0.0, -0.5),
                KeyCode::Char('+' | '=') | KeyCode::PageUp => self.graph_view.zoom_by(1.5),
                KeyCode::Char('-' | '_') | KeyCode::PageDown => self.graph_view.zoom_by(1.0 / 1.5),
                KeyCode::Char('0') | KeyCode::Home => self.graph_view = View::default(),
                _ => {}
            }
            return false;
        }
        if key.code == KeyCode::F(6) && self.prompt.is_none() {
            if self.graph.is_some() {
                self.legend = true;
            } else {
                self.status = Some(Err("No graph: add a `graph NAME` line and calculate".into()));
            }
            return false;
        }
        if key.code == KeyCode::F(4) && self.prompt.is_none() {
            if self.graph.is_some() {
                self.graph_focus = true;
            } else {
                self.status = Some(Err("No graph: add a `graph NAME` line and calculate".into()));
            }
            return false;
        }
        let shift = key.modifiers.contains(KeyModifiers::SHIFT);
        let ctrl = key.modifiers.contains(KeyModifiers::CONTROL);
        let multiline = self.prompt.is_none() && self.focus == Focus::Defs;
        if self.prompt.is_none() {
            let (row, col) = &mut self.result_scroll;
            match key.code {
                KeyCode::PageUp => *row = row.saturating_sub(10),
                KeyCode::PageDown => *row += 10,
                KeyCode::Up if ctrl => *row = row.saturating_sub(1),
                KeyCode::Down if ctrl => *row += 1,
                KeyCode::Left if ctrl => *col = col.saturating_sub(4),
                KeyCode::Right if ctrl => *col += 4,
                _ => {}
            }
            if matches!(key.code, KeyCode::PageUp | KeyCode::PageDown)
                || (ctrl
                    && matches!(
                        key.code,
                        KeyCode::Up | KeyCode::Down | KeyCode::Left | KeyCode::Right
                    ))
            {
                return false;
            }
        }
        if ctrl {
            if let KeyCode::Char(c) = key.code {
                match c.to_ascii_lowercase() {
                    's' | 'o' | 'r' | 'e' if self.prompt.is_none() => {
                        let (kind, name) = match c.to_ascii_lowercase() {
                            's' => (Prompt::Save, "matrices.txt"),
                            'o' => (
                                Prompt::Open,
                                self.export_kind()
                                    .map_or("matrices.txt", ExportKind::default_name),
                            ),
                            'e' => match self.export_kind() {
                                Some(k) => (Prompt::Export(k), k.default_name()),
                                None => {
                                    self.status = Some(Err(EXPORT_UNAVAILABLE.into()));
                                    return false;
                                }
                            },
                            _ => (Prompt::SaveResult, "result.txt"),
                        };
                        if kind != Prompt::SaveResult || matches!(self.result, Some(Ok(_))) {
                            self.prompt = Some((kind, Editor::new(name)));
                        } else {
                            self.status =
                                Some(Err("Nothing to save: calculate a result first".into()));
                        }
                    }
                    'n' if self.prompt.is_none() => self.feed_result_back(gpu, false),
                    'p' if self.prompt.is_none() => self.feed_result_back(gpu, true),
                    'c' => {
                        if let Some(t) = self.editor().selected_text() {
                            self.clipboard.set(t);
                        }
                    }
                    'x' => {
                        if let Some(t) = self.editor().cut() {
                            self.clipboard.set(t);
                        }
                    }
                    'v' => {
                        let clip = self.clipboard.get();
                        self.editor().insert_str(&clip);
                    }
                    'a' => self.editor().select_all(),
                    _ => {}
                }
                return false;
            }
        }
        if let Some((kind, ed)) = &self.prompt {
            let (kind, path) = (*kind, ed.text());
            match key.code {
                KeyCode::Esc => self.prompt = None,
                KeyCode::Enter => {
                    self.prompt = None;
                    self.status = Some(self.file_action(kind, path.trim()));
                }
                _ => {}
            }
            if self.prompt.is_none() {
                return false;
            }
        }
        match key.code {
            KeyCode::Esc => return true,
            KeyCode::Tab | KeyCode::BackTab if self.prompt.is_none() => {
                let back = self.before_name.is_some();
                self.focus = match self.focus {
                    Focus::Defs => Focus::Expr,
                    Focus::Expr if back => Focus::Back,
                    _ => Focus::Defs,
                }
            }
            KeyCode::F(5) => self.calculate(gpu),
            KeyCode::F(3) if self.prompt.is_none() => self.expr_help = true,
            KeyCode::F(2) if self.prompt.is_none() => {
                self.examples = Some(ListState::default().with_selected(Some(0)));
            }
            KeyCode::Enter if self.prompt.is_some() => {}
            KeyCode::Enter if !multiline && self.focus == Focus::Back => {
                self.feed_result_back(gpu, true)
            }
            KeyCode::Enter if !multiline => self.calculate(gpu),
            KeyCode::Enter => self.editor().insert_str("\n"),
            KeyCode::Backspace => self.editor().backspace(),
            KeyCode::Delete => self.editor().delete(),
            KeyCode::Left => self.editor().left(shift),
            KeyCode::Right => self.editor().right(shift),
            KeyCode::Up if multiline => self.editor().up(shift),
            KeyCode::Down if multiline => self.editor().down(shift),
            KeyCode::Up | KeyCode::Down => {}
            KeyCode::Home => self.editor().home(shift),
            KeyCode::End => self.editor().end(shift),
            KeyCode::Char(c) => self.editor().insert_str(c.encode_utf8(&mut [0; 4])),
            _ => {}
        }
        false
    }

    fn file_action(&mut self, kind: Prompt, path: &str) -> Result<String, String> {
        let done = match kind {
            Prompt::Save => save_input(path, &self.defs.text(), &self.expr.text())
                .map(|_| format!("Saved to {path}")),
            Prompt::SaveResult => match &self.result {
                Some(Ok(m)) => save_result(path, m).map(|_| format!("Result saved to {path}")),
                _ => return Err("Nothing to save: calculate a result first".into()),
            },
            Prompt::Export(kind) => self.export(kind, path),
            Prompt::Open => load_input(path).map(|(defs, expr)| {
                let old = crate::matrix::main_size(&self.defs.text());
                let new = crate::matrix::main_size(&defs);
                self.defs = Editor::new(&defs);
                self.feed = None;
                self.prev_input = None;
                match expr {
                    Some(e) => self.expr = Editor::new(&e),
                    // Keep the current expressions but fit their sizes to the opened matrix.
                    None => {
                        if let (Some(old), Some(new)) = (old, new) {
                            for ed in [&mut self.expr, &mut self.back] {
                                *ed =
                                    Editor::new(&crate::matrix::remap_sizes(&ed.text(), old, new));
                            }
                        }
                    }
                }
                self.result = None;
                self.source = Some(match &self.example_name {
                    Some(ex) => format!("{ex} | {path}"),
                    None => path.to_string(),
                });
                format!("Loaded {path}")
            }),
        };
        done.map_err(|e| format!("{e:#}"))
    }

    /// Ctrl+N: the latest picture result becomes the input picture, then the expression runs again.
    /// Ctrl+N (forward expression) / Ctrl+P (back expression): the latest picture result becomes
    /// the input picture, then the chosen expression is calculated.
    fn feed_result_back(&mut self, gpu: &Gpu, back: bool) {
        let expr = if back { &self.back } else { &self.expr }.text();
        if back && expr.trim().is_empty() {
            self.status = Some(Err("Ctrl+P needs a back expression (the box below)".into()));
            return;
        }
        match (&self.before_name, &self.result) {
            (Some(name), Some(Ok(r))) => {
                let (name, r) = (name.clone(), r.clone());
                let prev = Some(r.clone());
                let restore = back && expr.trim() == self.back_default.trim();
                self.feed = Some((self.defs.text(), name, r));
                self.calculate_expr(gpu, &expr);
                if back {
                    // The default back expression returns to the exact picture before Ctrl+N.
                    let exact = self.prev_input.take();
                    if let (true, Some(Ok(res)), Some(p)) = (restore, &self.result, exact) {
                        if (res.rows, res.cols) == (p.rows, p.cols) {
                            self.result = Some(Ok(p));
                        }
                    }
                } else {
                    self.prev_input = prev;
                }
            }
            _ => self.status = Some(Err(FEED_UNAVAILABLE.into())),
        }
    }

    /// What Ctrl+E can export for the current definitions/result, if anything.
    fn export_kind(&self) -> Option<ExportKind> {
        if matches!(graph_spec(&self.defs.text()), Some((_, Some(_)))) {
            Some(ExportKind::Maze)
        } else if self.before.is_some() {
            Some(ExportKind::Image)
        } else {
            None
        }
    }

    fn export(&self, kind: ExportKind, path: &str) -> anyhow::Result<String> {
        match kind {
            ExportKind::Maze => self.export_maze(path),
            ExportKind::Image => match &self.result {
                Some(Ok(m)) => {
                    crate::matrix::save_png(path, m)?;
                    Ok(format!("Image saved to {path}"))
                }
                _ => anyhow::bail!("Nothing to export: calculate a result first"),
            },
        }
    }

    /// Zips the grid graph (`graph NAME grid W`) as a maze with A*/BFS solutions and the inputs.
    fn export_maze(&self, path: &str) -> anyhow::Result<String> {
        use anyhow::Context;
        let defs = self.defs.text();
        let Some((name, Some(width))) = graph_spec(&defs) else {
            anyhow::bail!("export needs a `graph NAME grid W` line in the definitions");
        };
        let mut vars = parse_definitions(&defs)?;
        let width =
            size_value(&width, &vars).with_context(|| format!("bad grid width '{width}'"))?;
        let m = vars
            .remove(&name)
            .with_context(|| format!("unknown matrix '{name}'"))?;
        let input = format!("{}\n# expr: {}\n", defs.trim_end(), self.expr.text());
        let result = match &self.result {
            Some(Ok(r)) => Some(r),
            _ => None,
        };
        crate::search::save_maze_bundle(path, &m, width, &input, result)?;
        Ok(format!("Maze and solutions saved to {path}"))
    }

    pub fn draw(&mut self, frame: &mut Frame, area: Rect) {
        if let Some(state) = &mut self.examples {
            let items: Vec<&str> = EXAMPLES.iter().map(|e| e.title).collect();
            let list = List::new(items)
                .block(
                    Block::bordered()
                        .title(" Matrix examples ")
                        .title_bottom(" up/down: select  Enter: load and calculate  Esc: back "),
                )
                .highlight_symbol("> ")
                .highlight_style(Style::new().add_modifier(ratatui::style::Modifier::REVERSED));
            frame.render_widget(Clear, area);
            frame.render_stateful_widget(list, area, state);
            return;
        }
        if self.expr_help {
            let lines: Vec<Line> = EXPR_HELP.1.iter().map(|l| Line::from(*l)).collect();
            frame.render_widget(Clear, area);
            frame.render_widget(
                Paragraph::new(lines).block(
                    Block::bordered()
                        .title(format!(" {} ", EXPR_HELP.0))
                        .title_bottom(" Esc/F3: close "),
                ),
                area,
            );
            return;
        }
        let max_defs = (area.height as usize * 35 / 100).max(3);
        let defs_height = (self.defs.line_count() + 2).min(max_defs);
        self.defs_scroll = scroll_offset(
            self.defs_scroll,
            self.defs.cursor_line(),
            defs_height - 2,
            self.defs.line_count(),
        );
        let show_back = self.before_name.is_some();
        if !show_back && self.focus == Focus::Back {
            self.focus = Focus::Expr;
        }
        let [defs, expr, back, prompt, result] = Layout::vertical([
            Constraint::Length(defs_height as u16),
            Constraint::Length(3),
            Constraint::Length(if show_back { 3 } else { 0 }),
            Constraint::Length(if self.prompt.is_some() { 3 } else { 0 }),
            Constraint::Min(3),
        ])
        .areas(area);
        if let Some((kind, ed)) = &self.prompt {
            let title = match kind {
                Prompt::Save => " Save to file (Enter: save, Esc: cancel) ",
                Prompt::SaveResult => " Save result to file (Enter: save, Esc: cancel) ",
                Prompt::Export(ExportKind::Maze) => {
                    " Export maze + solutions as zip (Enter: save, Esc: cancel) "
                }
                Prompt::Export(_) => " Export result as PNG image (Enter: save, Esc: cancel) ",
                Prompt::Open => {
                    " Open matrices .txt, maze .zip or image file (Enter: load, Esc: cancel) "
                }
            };
            frame.render_widget(
                Paragraph::new(ed.lines(true)).block(
                    Block::bordered()
                        .title(title)
                        .border_style(Style::new().fg(Color::Cyan)),
                ),
                prompt,
            );
        }
        let focused = |f| {
            if self.focus == f {
                Style::new().fg(Color::Yellow)
            } else {
                Style::new()
            }
        };
        frame.render_widget(
            Paragraph::new(
                self.defs
                    .lines(self.prompt.is_none() && self.focus == Focus::Defs),
            )
            .scroll((self.defs_scroll as u16, 0))
            .block(
                Block::bordered()
                    .title(if self.defs.line_count() > defs_height - 2 {
                        format!(
                            " Matrices (line {}/{}) ",
                            self.defs.cursor_line() + 1,
                            self.defs.line_count()
                        )
                    } else {
                        " Matrices (NAME = [row; row; ...]) ".to_string()
                    })
                    .border_style(focused(Focus::Defs)),
            ),
            defs,
        );
        frame.render_widget(
            Paragraph::new(
                self.expr
                    .lines(self.prompt.is_none() && self.focus == Focus::Expr),
            )
            .block(
                Block::bordered()
                    .title(" Expr: + - * ' ( )  rot rotx roty rotz scale shearx sheary reflect proj identity ones smooth flip det trace  (F3: help) ")
                    .border_style(focused(Focus::Expr)),
            ),
            expr,
        );
        if show_back {
            frame.render_widget(
                Paragraph::new(
                    self.back
                        .lines(self.prompt.is_none() && self.focus == Focus::Back),
                )
                .block(
                    Block::bordered()
                        .title(" Back expr: Ctrl+P / Enter here applies it to the latest result (undoes the expression) ")
                        .border_style(focused(Focus::Back)),
                ),
                back,
            );
        }
        let (lines, style) = match &self.result {
            None => (
                vec![Line::from(
                    "Press Enter in the expression (or F5) to calculate",
                )],
                Style::new(),
            ),
            Some(Ok(m)) => (
                m.row_strings().into_iter().map(Line::from).collect(),
                Style::new().fg(Color::Green),
            ),
            Some(Err(e)) => (vec![Line::from(e.as_str())], Style::new().fg(Color::Red)),
        };
        let mut lines = lines;
        if let Some(status) = &self.status {
            let (msg, color) = match status {
                Ok(m) => (m, Color::Cyan),
                Err(e) => (e, Color::Red),
            };
            lines.insert(0, Line::styled(msg.as_str(), Style::new().fg(color)));
        }
        let mode = if self.graph_focus {
            " | graph view"
        } else {
            ""
        };
        let title = match &self.source {
            Some(name) => format!(" Result: {name}{mode} "),
            None => format!(" Result{mode} "),
        };
        let hints: Vec<String> = if self.graph_focus {
            vec![
                " arrows: pan  +/- PgUp/PgDn: zoom  0/Home: reset  F6/L: legend  F4/Esc: back "
                    .into(),
                " arrows pan  +/- zoom  0 reset  F6 legend  Esc back ".into(),
                " arrows +/- 0 F6 Esc ".into(),
            ]
        } else {
            let (long, short) = match self.export_kind() {
                Some(ExportKind::Maze) => ("  Ctrl+E maze zip", " ^E zip"),
                Some(_) => ("  Ctrl+E export PNG", " ^E png"),
                None => ("", ""),
            };
            vec![
                format!(
                    " Tab: switch  Enter: calc  Ctrl+S save  Ctrl+R result{long}  Ctrl+O open  F2: examples  F3: operations list  F4: graph  F6: legend  F1: help  Esc: back "
                ),
                format!(
                    " Tab  Enter: calc  ^S save  ^R result{short}  ^O open  F1: all keys  Esc "
                ),
                " Enter: calc  F1: all keys  Esc ".into(),
            ]
        };
        let room = result.width.saturating_sub(2) as usize;
        let hint = hints
            .iter()
            .find(|h| h.chars().count() <= room)
            .unwrap_or(&hints[hints.len() - 1]);
        let outer = Block::bordered().title(title).title_bottom(hint.as_str());
        let inner = outer.inner(result);
        frame.render_widget(outer, result);
        if let (Some(before), Some(Ok(after))) = (&self.before, &self.result) {
            let [msg, pics] =
                Layout::vertical([Constraint::Length(1), Constraint::Min(1)]).areas(inner);
            if let Some(status) = &self.status {
                let color = if status.is_ok() {
                    Color::Cyan
                } else {
                    Color::Red
                };
                let text = status.as_ref().unwrap_or_else(|e| e);
                frame.render_widget(
                    Paragraph::new(text.as_str()).style(Style::new().fg(color)),
                    msg,
                );
            }
            let [l, r] = Layout::horizontal([Constraint::Percentage(50); 2]).areas(pics);
            draw_picture(frame, l, " Before ", before);
            draw_picture(frame, r, " After ", after);
            return;
        }
        let (text_area, graph_area) = if self.graph.is_some() {
            let [l, r] =
                Layout::horizontal([Constraint::Percentage(40), Constraint::Min(10)]).areas(inner);
            (l, Some(r))
        } else {
            (inner, None)
        };
        let max_row = lines.len().saturating_sub(text_area.height as usize);
        let widest = lines.iter().map(Line::width).max().unwrap_or(0);
        let max_col = widest.saturating_sub(text_area.width as usize);
        self.result_scroll = (
            self.result_scroll.0.min(max_row),
            self.result_scroll.1.min(max_col),
        );
        if max_row > 0 || max_col > 0 {
            let first = self.result_scroll.0 + 1;
            let last = (self.result_scroll.0 + text_area.height as usize).min(lines.len());
            let info = format!(
                " lines {first}-{last}/{} (PgUp/PgDn, Ctrl+arrows) ",
                lines.len()
            );
            let top = Rect::new(result.x + 1, result.y, result.width.saturating_sub(2), 1);
            frame.render_widget(
                Paragraph::new(info)
                    .alignment(ratatui::layout::Alignment::Right)
                    .style(Style::new().fg(Color::Cyan)),
                top,
            );
        }
        frame.render_widget(
            Paragraph::new(lines)
                .style(style)
                .scroll((self.result_scroll.0 as u16, self.result_scroll.1 as u16)),
            text_area,
        );
        if let (Some(area), Some((name, g))) = (graph_area, &self.graph) {
            let values = match &self.result {
                Some(Ok(m)) if m.data.len() == g.n && (m.rows == 1 || m.cols == 1) => {
                    Some(m.data.as_slice())
                }
                _ => None,
            };
            let title = format!(
                " Graph {name}  zoom {:.1}x{} ",
                self.graph_view.zoom,
                if self.graph_focus {
                    ""
                } else {
                    "  (F4: zoom/pan)"
                }
            );
            graph::draw(
                frame,
                area,
                &title,
                g,
                values,
                self.graph_view,
                self.graph_focus,
            );
        }
        if let (true, Some((_, g))) = (self.legend, &self.graph) {
            draw_legend(frame, area, g.maze.is_some());
        }
    }
}

/// Popup explaining the symbols and colors of the graph drawing.
fn draw_legend(frame: &mut Frame, area: Rect, maze: bool) {
    let item = |glyph: &'static str, color: Color, text: &'static str| {
        Line::from(vec![
            Span::styled(format!(" {glyph} "), Style::new().fg(color)),
            Span::raw(text),
        ])
    };
    let mut lines = if maze {
        vec![
            item("●", Color::Cyan, "cell (node)"),
            item("●", Color::Red, "dead end (a single passage)"),
            item(
                "●",
                Color::Yellow,
                "cell with a value > 0 in the result (A* path, BFS)",
            ),
            item("━", Color::Green, "passage on the longest path"),
            item("━", Color::DarkGray, "other passage"),
            item("┃", Color::Magenta, "wall"),
        ]
    } else {
        vec![
            item("○", Color::Cyan, "node"),
            item("○", Color::Green, "node with a value > 0 in the result"),
            item(
                "─",
                Color::DarkGray,
                "edge (an arrowhead marks one-way edges)",
            ),
            item(
                "1",
                Color::Yellow,
                "edge weight (weighted graphs, when zoomed in)",
            ),
        ]
    };
    lines.push(item(
        "i=v",
        Color::Cyan,
        "node i with result value v (when zoomed in)",
    ));
    lines.push(Line::raw(""));
    lines.push(Line::raw(" F4 focus: arrows pan, +/- zoom, 0 reset"));
    let width = (lines.iter().map(Line::width).max().unwrap_or(0) as u16 + 4).min(area.width);
    let height = (lines.len() as u16 + 2).min(area.height);
    let popup = Rect::new(
        area.x + area.width.saturating_sub(width) / 2,
        area.y + area.height.saturating_sub(height) / 2,
        width,
        height,
    );
    frame.render_widget(Clear, popup);
    frame.render_widget(
        Paragraph::new(lines).block(
            Block::bordered()
                .title(" Legend ")
                .title_bottom(" F6/Esc: close "),
        ),
        popup,
    );
}
/// Keeps the cursor line inside a window of `visible` lines, moving it as little as possible.
fn scroll_offset(offset: usize, cursor: usize, visible: usize, total: usize) -> usize {
    let offset = if cursor < offset {
        cursor
    } else if cursor >= offset + visible {
        cursor + 1 - visible
    } else {
        offset
    };
    offset.min(total.saturating_sub(visible))
}

/// Results this big are pictures rather than numbers worth reading.
fn is_picture(m: &Matrix) -> bool {
    m.rows >= 8 && m.cols >= 8
}

/// Grayscale picture (values clamped to 0..1) using half blocks: 1 column x 2 pixels per cell.
fn draw_picture(frame: &mut Frame, area: Rect, title: &str, m: &Matrix) {
    let block = Block::bordered().title(title);
    let inner = block.inner(area);
    frame.render_widget(block, area);
    if inner.width == 0 || inner.height == 0 {
        return;
    }
    // Keep the aspect ratio (a character cell is about twice as tall as wide).
    let scale = (inner.width as f32 / m.cols as f32).min(inner.height as f32 * 2.0 / m.rows as f32);
    let (w, h) = (
        ((m.cols as f32 * scale) as u16).max(1),
        (((m.rows as f32 * scale) / 2.0) as u16).max(1),
    );
    let gray = |px: u16, py: u16| {
        let x = (px as usize * m.cols / w as usize).min(m.cols - 1);
        let y = (py as usize * m.rows / (h as usize * 2)).min(m.rows - 1);
        let v = (m.data[y * m.cols + x].clamp(0.0, 1.0) * 255.0) as u8;
        Color::Rgb(v, v, v)
    };
    let buf = frame.buffer_mut();
    for cy in 0..h {
        for cx in 0..w {
            if let Some(cell) = buf.cell_mut((inner.x + cx, inner.y + cy)) {
                cell.set_symbol("▀")
                    .set_fg(gray(cx, cy * 2))
                    .set_bg(gray(cx, cy * 2 + 1));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn export_kind_follows_example() {
        let Ok(gpu) = Gpu::new() else { return };
        let mut ui = MatrixUi::with_examples();
        let mut kind = |title: &str| {
            let (defs, expr) = EXAMPLES.iter().find(|e| e.title == title).unwrap().load();
            ui.defs = Editor::new(&defs);
            ui.expr = Editor::new(&expr);
            ui.calculate(&gpu);
            ui.export_kind()
        };
        assert_eq!(
            kind("Maze: A* path through 504 cells"),
            Some(ExportKind::Maze)
        );
        assert_eq!(kind("Photo: blur"), Some(ExportKind::Image));
        assert_eq!(kind("Graph: node degrees"), None);
        assert_eq!(kind("Lighting: Lambert shading"), None);
        assert_eq!(
            kind("Photo: raycast (camera rotated 40 degrees)"),
            Some(ExportKind::Image)
        );
        assert_eq!(kind("Lighting: moved light"), None);
    }

    #[test]
    fn ctrl_n_feeds_result_back_as_input() {
        let Ok(gpu) = Gpu::new() else { return };
        let mut ui = MatrixUi::with_examples();
        ui.defs = Editor::new("I = image builtin:earth");
        ui.expr = Editor::new("0.5 * I");
        ui.calculate(&gpu);
        let (first, input) = (
            ui.result.clone().unwrap().unwrap(),
            ui.before.clone().unwrap(),
        );
        ui.feed_result_back(&gpu, false);
        let second = ui.result.clone().unwrap().unwrap();
        assert_eq!(ui.before.as_ref().unwrap().data, first.data);
        assert!((second.data[1000] - 0.25 * input.data[1000]).abs() < 1e-5);
        ui.defs = Editor::new("I = image builtin:earth\n# changed");
        ui.calculate(&gpu);
        assert!((ui.result.unwrap().unwrap().data[1000] - 0.5 * input.data[1000]).abs() < 1e-5);
    }

    #[test]
    fn photo_examples_work_on_non_square_images() {
        let Ok(gpu) = Gpu::new() else { return };
        let img = Matrix {
            rows: 106,
            cols: 128,
            data: vec![0.5; 106 * 128],
        };
        for ex in EXAMPLES.iter().filter(|e| e.title.starts_with("Photo")) {
            let (_, expr) = ex.load();
            let vars = std::collections::HashMap::from([("I".to_string(), img.clone())]);
            evaluate_vars(&vars, &expr, &gpu).unwrap_or_else(|e| panic!("{}: {e:#}", ex.title));
            let back = ex.back();
            if !back.is_empty() {
                evaluate_vars(&vars, &back, &gpu)
                    .unwrap_or_else(|e| panic!("{} (back): {e:#}", ex.title));
            }
        }
    }

    #[test]
    fn ctrl_p_restores_raycast_exactly() {
        let Ok(gpu) = Gpu::new() else { return };
        let mut ui = MatrixUi::with_examples();
        let ex = EXAMPLES
            .iter()
            .find(|e| e.title.contains("aycast"))
            .unwrap();
        let (defs, expr) = ex.load();
        ui.defs = Editor::new(&defs);
        ui.expr = Editor::new(&expr);
        ui.back = Editor::new(&ex.back());
        ui.back_default = ex.back();
        ui.calculate(&gpu);
        let first = ui.result.clone().unwrap().unwrap();
        ui.feed_result_back(&gpu, false);
        assert_ne!(ui.result.clone().unwrap().unwrap().data, first.data);
        ui.feed_result_back(&gpu, true);
        assert_eq!(ui.result.clone().unwrap().unwrap().data, first.data);
    }

    #[test]
    fn ctrl_p_applies_back_expression_to_latest_result() {
        let Ok(gpu) = Gpu::new() else { return };
        let mut ui = MatrixUi::with_examples();
        let (defs, expr) = EXAMPLES
            .iter()
            .find(|e| e.title == "Photo: enhance contrast/brightness")
            .unwrap()
            .load();
        ui.defs = Editor::new(&defs);
        ui.expr = Editor::new(&expr);
        ui.back = Editor::new(
            &EXAMPLES
                .iter()
                .find(|e| e.title == "Photo: enhance contrast/brightness")
                .unwrap()
                .back(),
        );
        ui.calculate(&gpu);
        let first = ui.result.clone().unwrap().unwrap();
        ui.feed_result_back(&gpu, false);
        assert_ne!(ui.result.clone().unwrap().unwrap().data, first.data);
        ui.feed_result_back(&gpu, true);
        let back = ui.result.clone().unwrap().unwrap();
        assert!(
            back.data
                .iter()
                .zip(&first.data)
                .all(|(a, b)| (a - b).abs() < 1e-4)
        );
        ui.back = Editor::new("");
        ui.feed_result_back(&gpu, true);
        assert!(matches!(ui.status, Some(Err(_))));
    }

    #[test]
    fn scale_down_example_shows_original_and_scaled() {
        let Ok(gpu) = Gpu::new() else { return };
        let mut ui = MatrixUi::with_examples();
        let ex = EXAMPLES
            .iter()
            .find(|e| e.title == "Photo: scale down (half size)")
            .unwrap();
        let (defs, expr) = ex.load();
        ui.defs = Editor::new(&defs);
        ui.expr = Editor::new(&expr);
        ui.back = Editor::new(&ex.back());
        ui.calculate(&gpu);
        let r = ui.result.clone().unwrap().unwrap();
        assert_eq!((r.rows, r.cols), (48, 48));
        assert_eq!(ui.before.as_ref().map(|b| b.rows), Some(96));
        ui.feed_result_back(&gpu, true);
        let up = ui.result.clone().unwrap().unwrap();
        assert_eq!((up.rows, up.cols), (96, 96));
    }

    #[test]
    fn photo_example_shows_before_and_after() {
        let Ok(gpu) = Gpu::new() else { return };
        let mut ui = MatrixUi::with_examples();
        let (defs, expr) = EXAMPLES
            .iter()
            .find(|e| e.title == "Photo: enhance contrast/brightness")
            .unwrap()
            .load();
        ui.defs = Editor::new(&defs);
        ui.expr = Editor::new(&expr);
        ui.calculate(&gpu);
        let before = ui.before.expect("before picture");
        assert_eq!((before.rows, before.cols), (96, 96));
    }

    #[test]
    fn graph_view_zooms_pans_and_draws_huge_graph() {
        let Ok(gpu) = Gpu::new() else { return };
        let mut ui = MatrixUi::with_examples();
        let key = |ui: &mut MatrixUi, code| ui.handle_key(KeyEvent::from(code), &gpu);
        key(&mut ui, KeyCode::F(4));
        assert!(!ui.graph_focus, "no graph yet");
        let ex = EXAMPLES
            .iter()
            .position(|e| e.title.starts_with("Maze: A*"))
            .unwrap();
        ui.examples.as_mut().unwrap().select(Some(ex));
        key(&mut ui, KeyCode::Enter);
        assert_eq!(ui.source.as_deref(), Some(EXAMPLES[ex].title));

        key(&mut ui, KeyCode::F(4));
        assert!(ui.graph_focus);
        key(&mut ui, KeyCode::Char('+'));
        key(&mut ui, KeyCode::Right);
        assert!(ui.graph_view.zoom > 1.0 && ui.graph_view.cx > 0.0);
        key(&mut ui, KeyCode::Char('l'));
        assert!(ui.legend);
        let mut terminal =
            ratatui::Terminal::new(ratatui::backend::TestBackend::new(120, 40)).unwrap();
        terminal.draw(|f| ui.draw(f, f.area())).unwrap();
        assert!(format!("{}", terminal.backend()).contains("dead end"));
        key(&mut ui, KeyCode::Esc);
        assert!(!ui.legend && ui.graph_focus, "Esc closes only the legend");
        key(&mut ui, KeyCode::F(6));
        assert!(ui.legend);
        key(&mut ui, KeyCode::F(6));
        assert!(!key(&mut ui, KeyCode::Esc) && !ui.graph_focus);

        for zoom in [1.0, 8.0] {
            ui.graph_view.zoom = zoom;
            let mut terminal =
                ratatui::Terminal::new(ratatui::backend::TestBackend::new(120, 40)).unwrap();
            terminal.draw(|f| ui.draw(f, f.area())).unwrap();
            let screen = format!("{}", terminal.backend());
            assert!(screen.contains(EXAMPLES[ex].title) && screen.contains("Graph M"));
        }
    }

    #[test]
    fn scroll_follows_cursor() {
        assert_eq!(scroll_offset(0, 3, 5, 20), 0);
        assert_eq!(scroll_offset(0, 7, 5, 20), 3);
        assert_eq!(scroll_offset(3, 4, 5, 20), 3);
        assert_eq!(scroll_offset(3, 1, 5, 20), 1);
        assert_eq!(scroll_offset(9, 4, 5, 6), 1);
    }

    #[test]
    fn page_keys_scroll_result_without_editing() {
        let Ok(gpu) = Gpu::new() else { return };
        let mut ui = MatrixUi::new();
        let text = ui.expr.text();
        ui.handle_key(KeyEvent::from(KeyCode::PageDown), &gpu);
        assert_eq!(ui.result_scroll.0, 10);
        ui.handle_key(KeyEvent::new(KeyCode::Right, KeyModifiers::CONTROL), &gpu);
        assert_eq!(ui.result_scroll.1, 4);
        ui.handle_key(KeyEvent::from(KeyCode::PageUp), &gpu);
        assert_eq!(ui.result_scroll.0, 0);
        assert_eq!(ui.expr.text(), text);
    }

    #[test]
    fn f3_toggles_expression_help_without_typing() {
        let Ok(gpu) = Gpu::new() else { return };
        let mut ui = MatrixUi::new();
        let before = ui.expr.text();
        ui.handle_key(KeyEvent::from(KeyCode::F(3)), &gpu);
        assert!(ui.expr_help);
        ui.handle_key(KeyEvent::from(KeyCode::Char('x')), &gpu);
        assert_eq!(ui.expr.text(), before);
        ui.handle_key(KeyEvent::from(KeyCode::Esc), &gpu);
        assert!(!ui.expr_help);
    }

    #[test]
    fn clipboard_round_trips() {
        let mut c = Clip::default();
        c.set("A = [1 2]".into());
        assert_eq!(c.get(), "A = [1 2]");
    }
}
