use crate::editor::Editor;
use crate::examples::EXAMPLES;
use crate::gpu::Gpu;
use crate::graph::{self, Graph};
use crate::matrix::{
    Matrix, evaluate, graph_name, load_input, output_path, parse_definitions, save_input,
    save_result,
};
use ratatui::{
    Frame,
    crossterm::event::{KeyCode, KeyEvent, KeyModifiers},
    layout::{Constraint, Layout, Rect},
    style::{Color, Style},
    text::Line,
    widgets::{Block, Clear, List, ListState, Paragraph},
};

#[derive(Clone, Copy, PartialEq)]
enum Focus {
    Defs,
    Expr,
}

#[derive(Clone, Copy, PartialEq)]
enum Prompt {
    Save,
    SaveResult,
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
    clipboard: Clip,
    /// File path prompt shown above the result while saving/opening.
    prompt: Option<(Prompt, Editor)>,
    status: Option<Result<String, String>>,
    /// Example picker, shown over the whole screen while open.
    examples: Option<ListState>,
    /// Graph drawn next to the result when the definitions contain `graph NAME`.
    graph: Option<(String, Graph)>,
    /// Input picture matching a large (image-sized) result, drawn next to it.
    before: Option<Matrix>,
    expr_help: bool,
    /// First visible line of the definitions editor.
    defs_scroll: usize,
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
        "  ones(r, c)             matrix of ones",
        "  smooth(n)              n x n blur kernel (1 2 1)/4",
        "e.g.  rot(90) * v     rotz(30) * scale(1, 2, 3) * P",
        "      smooth(96) * I * smooth(96)'",
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
        "Tab            switch definitions / expression",
        "Enter          calculate (in expression), F5 anywhere",
        "Arrows, Home/End, Shift  move / select",
        "Ctrl+C/X/V/A   copy / cut / paste / select all",
        "Ctrl+S         save input    Ctrl+O  open file",
        "Ctrl+R         save result once (asks for a path)",
        "F2             load an example (also in the main menu)",
        "graph NAME    (in definitions) draw matrix NAME as a 2D graph",
        "F3             expression help: all operations",
        "F1             this help",
        "Esc            back",
    ],
);

impl MatrixUi {
    pub fn new() -> Self {
        Self {
            defs: Editor::new("A = [1 2; 3 4]\nB = [0 1; 1 0]"),
            expr: Editor::new("A * B + A'"),
            clipboard: Clip::default(),
            prompt: None,
            status: None,
            examples: None,
            graph: None,
            before: None,
            expr_help: false,
            defs_scroll: 0,
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
        }
    }

    fn calculate(&mut self, gpu: &Gpu) {
        self.status = None;
        self.result =
            Some(evaluate(&self.defs.text(), &self.expr.text(), gpu).map_err(|e| format!("{e:#}")));
        if let (Some(Ok(m)), Some(path)) = (&self.result, output_path(&self.defs.text())) {
            self.status = Some(
                save_result(&path, m)
                    .map(|_| format!("Result saved to {path}"))
                    .map_err(|e| format!("{e:#}")),
            );
        }
        self.graph = None;
        let defs = self.defs.text();
        self.before = match (&self.result, parse_definitions(&defs)) {
            (Some(Ok(r)), Ok(vars)) if is_picture(r) => vars
                .get("I")
                .filter(|m| (m.rows, m.cols) == (r.rows, r.cols))
                .or_else(|| vars.values().find(|m| (m.rows, m.cols) == (r.rows, r.cols)))
                .cloned(),
            _ => None,
        };
        if let Some(name) = graph_name(&defs) {
            let built = parse_definitions(&defs)
                .ok()
                .and_then(|vars| vars.get(&name).cloned())
                .ok_or_else(|| format!("graph: unknown matrix '{name}'"))
                .and_then(|m| Graph::from_matrix(&m).map_err(|e| format!("graph: {e:#}")));
            match built {
                Ok(g) => self.graph = Some((name, g)),
                Err(e) => self.status = Some(Err(e)),
            }
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
                        self.expr = Editor::new(&expr);
                        self.focus = Focus::Expr;
                        self.examples = None;
                        self.calculate(gpu);
                    }
                }
                _ => {}
            }
            return false;
        }
        let shift = key.modifiers.contains(KeyModifiers::SHIFT);
        let ctrl = key.modifiers.contains(KeyModifiers::CONTROL);
        let multiline = self.prompt.is_none() && self.focus == Focus::Defs;
        if ctrl {
            if let KeyCode::Char(c) = key.code {
                match c.to_ascii_lowercase() {
                    's' | 'o' | 'r' if self.prompt.is_none() => {
                        let (kind, name) = match c.to_ascii_lowercase() {
                            's' => (Prompt::Save, "matrices.txt"),
                            'o' => (Prompt::Open, "matrices.txt"),
                            _ => (Prompt::SaveResult, "result.txt"),
                        };
                        if kind != Prompt::SaveResult || matches!(self.result, Some(Ok(_))) {
                            self.prompt = Some((kind, Editor::new(name)));
                        } else {
                            self.status =
                                Some(Err("Nothing to save: calculate a result first".into()));
                        }
                    }
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
                self.focus = if self.focus == Focus::Defs {
                    Focus::Expr
                } else {
                    Focus::Defs
                }
            }
            KeyCode::F(5) => self.calculate(gpu),
            KeyCode::F(3) if self.prompt.is_none() => self.expr_help = true,
            KeyCode::F(2) if self.prompt.is_none() => {
                self.examples = Some(ListState::default().with_selected(Some(0)));
            }
            KeyCode::Enter if self.prompt.is_some() => {}
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
            Prompt::Open => load_input(path).map(|(defs, expr)| {
                self.defs = Editor::new(&defs);
                if let Some(e) = expr {
                    self.expr = Editor::new(&e);
                }
                self.result = None;
                format!("Loaded {path}")
            }),
        };
        done.map_err(|e| format!("{e:#}"))
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
        let [defs, expr, prompt, result] = Layout::vertical([
            Constraint::Length(defs_height as u16),
            Constraint::Length(3),
            Constraint::Length(if self.prompt.is_some() { 3 } else { 0 }),
            Constraint::Min(3),
        ])
        .areas(area);
        if let Some((kind, ed)) = &self.prompt {
            let title = match kind {
                Prompt::Save => " Save to file (Enter: save, Esc: cancel) ",
                Prompt::SaveResult => " Save result to file (Enter: save, Esc: cancel) ",
                Prompt::Open => " Open file (Enter: load, Esc: cancel) ",
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
                    .title(" Expr: + - * ' ( )  rot rotx roty rotz scale shearx sheary reflect proj identity ones smooth  (F3: help) ")
                    .border_style(focused(Focus::Expr)),
            ),
            expr,
        );
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
        let outer = Block::bordered().title(" Result ")
            .title_bottom(" Tab: switch  Enter: calculate  arrows/Shift: move/select  Ctrl+C/X/V/A  Ctrl+S save  Ctrl+R save result  Ctrl+O open  F2: examples  F3: expr help  F1: help  Esc: back ");
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
        frame.render_widget(Paragraph::new(lines).style(style), text_area);
        if let (Some(area), Some((name, g))) = (graph_area, &self.graph) {
            let values = match &self.result {
                Some(Ok(m)) if m.data.len() == g.n && (m.rows == 1 || m.cols == 1) => {
                    Some(m.data.as_slice())
                }
                _ => None,
            };
            graph::draw(frame, area, &format!(" Graph {name} "), g, values);
        }
    }
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
    fn scroll_follows_cursor() {
        assert_eq!(scroll_offset(0, 3, 5, 20), 0);
        assert_eq!(scroll_offset(0, 7, 5, 20), 3);
        assert_eq!(scroll_offset(3, 4, 5, 20), 3);
        assert_eq!(scroll_offset(3, 1, 5, 20), 1);
        assert_eq!(scroll_offset(9, 4, 5, 6), 1);
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
