use crate::gpu::Gpu;
use crate::matrix::{Matrix, evaluate};
use ratatui::{
    Frame,
    crossterm::event::KeyCode,
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

pub struct MatrixUi {
    defs: String,
    expr: String,
    focus: Focus,
    result: Option<Result<Matrix, String>>,
}

impl MatrixUi {
    pub fn new() -> Self {
        Self {
            defs: "A = 1 2; 3 4\nB = 0 1; 1 0".into(),
            expr: "A * B + A'".into(),
            focus: Focus::Expr,
            result: None,
        }
    }

    fn text(&mut self) -> &mut String {
        match self.focus {
            Focus::Defs => &mut self.defs,
            Focus::Expr => &mut self.expr,
        }
    }

    fn calculate(&mut self, gpu: &Gpu) {
        self.result = Some(evaluate(&self.defs, &self.expr, gpu).map_err(|e| format!("{e:#}")));
    }

    /// Returns `true` when the user wants to leave the screen.
    pub fn handle_key(&mut self, key: KeyCode, gpu: &Gpu) -> bool {
        match key {
            KeyCode::Esc => return true,
            KeyCode::Tab | KeyCode::BackTab => {
                self.focus = if self.focus == Focus::Defs {
                    Focus::Expr
                } else {
                    Focus::Defs
                }
            }
            KeyCode::F(5) => self.calculate(gpu),
            KeyCode::Enter if self.focus == Focus::Expr => self.calculate(gpu),
            KeyCode::Enter => self.defs.push('\n'),
            KeyCode::Backspace => {
                self.text().pop();
            }
            KeyCode::Char(c) => self.text().push(c),
            _ => {}
        }
        false
    }

    pub fn draw(&self, frame: &mut Frame, area: Rect) {
        let [defs, expr, result] = Layout::vertical([
            Constraint::Length(self.defs.lines().count().max(1) as u16 + 3),
            Constraint::Length(3),
            Constraint::Min(3),
        ])
        .areas(area);
        let focused = |f| {
            if self.focus == f {
                Style::new().fg(Color::Yellow)
            } else {
                Style::new()
            }
        };
        let cursor = |f, s: &str| format!("{s}{}", if self.focus == f { "_" } else { "" });
        frame.render_widget(
            Paragraph::new(cursor(Focus::Defs, &self.defs)).block(
                Block::bordered()
                    .title(" Matrices (NAME = row; row; ...) ")
                    .border_style(focused(Focus::Defs)),
            ),
            defs,
        );
        frame.render_widget(
            Paragraph::new(cursor(Focus::Expr, &self.expr)).block(
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
        frame.render_widget(
            Paragraph::new(lines).style(style).block(
                Block::bordered()
                    .title(" Result ")
                    .title_bottom(" Tab: switch field  Enter: calculate  Esc: back "),
            ),
            result,
        );
    }
}
