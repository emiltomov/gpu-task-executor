use crate::editor::Editor;
use crate::gpu::Gpu;
use crate::matrix::{Matrix, evaluate, load_input, output_path, save_input, save_result};
use ratatui::{
    Frame,
    crossterm::event::{KeyCode, KeyEvent, KeyModifiers},
    layout::{Constraint, Layout, Rect},
    style::{Color, Style},
    text::Line,
    widgets::{Block, Paragraph},
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
    focus: Focus,
    result: Option<Result<Matrix, String>>,
}

pub const HELP: (&str, &[&str]) = (
    "Matrices",
    &[
        "Declare:  A = [1 2; 3 4]   (one per line)",
        "Include:  include path/to/file.txt",
        "Output:   output path/to/result.txt  (definitions line; the result",
        "          is written there on every calculation)",
        "Expr:     + - * ( ) ' (transpose); a number scales",
        "          e.g. (A + B) * C' - 2 * A",
        "Linear transformations (angles in degrees, apply to column vectors):",
        "          rot(a) rotx(a) roty(a) rotz(a)  rotations",
        "          scale(sx, sy[, sz])  shearx(k) sheary(k)  identity(n)",
        "          reflect(a) mirror across line at angle a; proj(a) project onto it",
        "          e.g. rot(90) * v      rotz(30) * scale(1, 2, 3) * P",
        "Tab            switch definitions / expression",
        "Enter          calculate (in expression), F5 anywhere",
        "Arrows, Home/End, Shift  move / select",
        "Ctrl+C/X/V/A   copy / cut / paste / select all",
        "Ctrl+S         save input    Ctrl+O  open file",
        "Ctrl+R         save result once (asks for a path)",
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
            focus: Focus::Expr,
            result: None,
        }
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
    }

    /// Returns `true` when the user wants to leave the screen.
    pub fn handle_key(&mut self, key: KeyEvent, gpu: &Gpu) -> bool {
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

    pub fn draw(&self, frame: &mut Frame, area: Rect) {
        let [defs, expr, prompt, result] = Layout::vertical([
            Constraint::Length(self.defs.line_count() as u16 + 2),
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
            .block(
                Block::bordered()
                    .title(" Matrices (NAME = [row; row; ...]) ")
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
                    .title(" Expression: + - * ( ) ' (transpose), numbers scale ")
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
        frame.render_widget(
            Paragraph::new(lines).style(style).block(
                Block::bordered()
                    .title(" Result ")
                    .title_bottom(" Tab: switch  Enter: calculate  arrows/Shift: move/select  Ctrl+C/X/V/A  Ctrl+S save  Ctrl+R save result  Ctrl+O open  F1: help  Esc: back "),
            ),
            result,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clipboard_round_trips() {
        let mut c = Clip::default();
        c.set("A = [1 2]".into());
        assert_eq!(c.get(), "A = [1 2]");
    }
}
