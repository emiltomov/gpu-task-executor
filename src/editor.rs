use ratatui::{
    style::{Color, Modifier, Style},
    text::{Line, Span},
};

/// Minimal multi-line text buffer with cursor and selection.
pub struct Editor {
    chars: Vec<char>,
    cur: usize,
    anchor: Option<usize>,
}

impl Editor {
    pub fn new(text: &str) -> Self {
        let chars: Vec<char> = text.chars().collect();
        Self {
            cur: chars.len(),
            chars,
            anchor: None,
        }
    }

    pub fn text(&self) -> String {
        self.chars.iter().collect()
    }

    pub fn line_count(&self) -> usize {
        self.chars.iter().filter(|&&c| c == '\n').count() + 1
    }

    /// Zero-based line the cursor is on.
    pub fn cursor_line(&self) -> usize {
        self.chars[..self.cur]
            .iter()
            .filter(|&&c| c == '\n')
            .count()
    }

    fn selection(&self) -> Option<(usize, usize)> {
        let a = self.anchor.filter(|&a| a != self.cur)?;
        Some((a.min(self.cur), a.max(self.cur)))
    }

    pub fn selected_text(&self) -> Option<String> {
        self.selection()
            .map(|(s, e)| self.chars[s..e].iter().collect())
    }

    fn delete_selection(&mut self) -> bool {
        let Some((s, e)) = self.selection() else {
            self.anchor = None;
            return false;
        };
        self.chars.drain(s..e);
        self.cur = s;
        self.anchor = None;
        true
    }

    pub fn cut(&mut self) -> Option<String> {
        let t = self.selected_text();
        self.delete_selection();
        t
    }

    pub fn insert_str(&mut self, s: &str) {
        self.delete_selection();
        let new: Vec<char> = s.chars().collect();
        let n = new.len();
        self.chars.splice(self.cur..self.cur, new);
        self.cur += n;
    }

    pub fn backspace(&mut self) {
        if !self.delete_selection() && self.cur > 0 {
            self.cur -= 1;
            self.chars.remove(self.cur);
        }
    }

    pub fn delete(&mut self) {
        if !self.delete_selection() && self.cur < self.chars.len() {
            self.chars.remove(self.cur);
        }
    }

    fn move_to(&mut self, pos: usize, select: bool) {
        if select {
            self.anchor.get_or_insert(self.cur);
        } else {
            self.anchor = None;
        }
        self.cur = pos;
    }

    fn line_start(&self, i: usize) -> usize {
        self.chars[..i]
            .iter()
            .rposition(|&c| c == '\n')
            .map_or(0, |p| p + 1)
    }

    fn line_end(&self, i: usize) -> usize {
        self.chars[i..]
            .iter()
            .position(|&c| c == '\n')
            .map_or(self.chars.len(), |p| i + p)
    }

    pub fn left(&mut self, select: bool) {
        self.move_to(self.cur.saturating_sub(1), select);
    }

    pub fn right(&mut self, select: bool) {
        self.move_to((self.cur + 1).min(self.chars.len()), select);
    }

    pub fn up(&mut self, select: bool) {
        let start = self.line_start(self.cur);
        let col = self.cur - start;
        let target = if start == 0 {
            0
        } else {
            let prev = self.line_start(start - 1);
            prev + col.min(start - 1 - prev)
        };
        self.move_to(target, select);
    }

    pub fn down(&mut self, select: bool) {
        let col = self.cur - self.line_start(self.cur);
        let end = self.line_end(self.cur);
        let target = if end == self.chars.len() {
            end
        } else {
            let next = end + 1;
            next + col.min(self.line_end(next) - next)
        };
        self.move_to(target, select);
    }

    pub fn home(&mut self, select: bool) {
        self.move_to(self.line_start(self.cur), select);
    }

    pub fn end(&mut self, select: bool) {
        self.move_to(self.line_end(self.cur), select);
    }

    pub fn select_all(&mut self) {
        self.anchor = Some(0);
        self.cur = self.chars.len();
    }

    pub fn lines(&self, focused: bool) -> Vec<Line<'static>> {
        let sel = self.selection();
        let mut lines = Vec::new();
        let mut spans: Vec<Span<'static>> = Vec::new();
        let cursor_style = Style::new().add_modifier(Modifier::REVERSED);
        for (i, &c) in self.chars.iter().enumerate() {
            let at_cursor = focused && i == self.cur && sel.is_none();
            let selected = sel.is_some_and(|(s, e)| i >= s && i < e);
            let style = if selected {
                Style::new().bg(Color::Blue)
            } else if at_cursor {
                cursor_style
            } else {
                Style::new()
            };
            if c == '\n' {
                if at_cursor {
                    spans.push(Span::styled(" ", cursor_style));
                }
                lines.push(Line::from(std::mem::take(&mut spans)));
            } else {
                spans.push(Span::styled(c.to_string(), style));
            }
        }
        if focused && self.cur == self.chars.len() && sel.is_none() {
            spans.push(Span::styled(" ", cursor_style));
        }
        lines.push(Line::from(spans));
        lines
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn edits_at_cursor() {
        let mut e = Editor::new("ab\ncd");
        e.left(false);
        e.insert_str("X");
        assert_eq!(e.text(), "ab\ncXd");
        e.up(false);
        e.end(false);
        e.delete();
        assert_eq!(e.text(), "abcXd");
        e.home(false);
        e.delete();
        e.backspace();
        assert_eq!(e.text(), "bcXd");
    }

    #[test]
    fn cut_and_paste_selection() {
        let mut e = Editor::new("hello world");
        e.home(false);
        for _ in 0..5 {
            e.right(true);
        }
        assert_eq!(e.selected_text().as_deref(), Some("hello"));
        let clip = e.cut().unwrap();
        assert_eq!(e.text(), " world");
        e.end(false);
        e.insert_str(&clip);
        assert_eq!(e.text(), " worldhello");
        e.select_all();
        e.insert_str("x");
        assert_eq!(e.text(), "x");
    }

    #[test]
    fn vertical_moves_keep_column_and_clamp() {
        let mut e = Editor::new("abcd\nx\nabcd");
        e.up(false);
        assert_eq!(e.cur, 6);
        e.up(false);
        assert_eq!(e.cur, 1);
        e.down(false);
        e.down(false);
        assert_eq!(e.cur, 8);
    }
}
