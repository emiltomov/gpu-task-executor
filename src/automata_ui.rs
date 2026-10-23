use crate::automata::{Automata, Kind, Settings};
use crate::gpu::Gpu;
use ratatui::{
    Frame,
    crossterm::event::KeyCode,
    layout::Rect,
    style::{Color, Modifier, Style},
    widgets::{Block, List, ListState},
};

#[derive(Clone, Copy, PartialEq)]
enum Field {
    Pattern,
    Density,
    Radius,
    Mu,
    Sigma,
    Dt,
    Generations,
    Speed,
}

impl Field {
    fn label(self) -> &'static str {
        match self {
            Field::Pattern => "Start pattern",
            Field::Density => "Random density",
            Field::Radius => "Kernel radius (R)",
            Field::Mu => "Growth centre (mu)",
            Field::Sigma => "Growth width (sigma)",
            Field::Dt => "Time step (dt)",
            Field::Generations => "Generations (0 = endless)",
            Field::Speed => "Steps per frame",
        }
    }
}

fn fields(kind: Kind) -> &'static [Field] {
    match kind {
        Kind::Life => &[
            Field::Pattern,
            Field::Density,
            Field::Generations,
            Field::Speed,
        ],
        Kind::Lenia => &[
            Field::Pattern,
            Field::Density,
            Field::Radius,
            Field::Mu,
            Field::Sigma,
            Field::Dt,
            Field::Generations,
            Field::Speed,
        ],
    }
}

pub enum SetupAction {
    None,
    Back,
    Start,
}

pub struct Setup {
    pub settings: Settings,
    list: ListState,
}

impl Setup {
    pub fn new(kind: Kind) -> Self {
        Self {
            settings: Settings::new(kind),
            list: ListState::default().with_selected(Some(0)),
        }
    }

    fn value(&self, f: Field) -> String {
        let s = &self.settings;
        match f {
            Field::Pattern => s.kind.patterns()[s.pattern].to_string(),
            Field::Density => format!("{:.0}%", s.density * 100.0),
            Field::Radius => s.lenia.radius.to_string(),
            Field::Mu => format!("{:.3}", s.lenia.mu),
            Field::Sigma => format!("{:.4}", s.lenia.sigma),
            Field::Dt => format!("{:.3}", s.lenia.dt),
            Field::Generations if s.generations == 0 => "endless".into(),
            Field::Generations => s.generations.to_string(),
            Field::Speed => s.steps_per_frame.to_string(),
        }
    }

    fn adjust(&mut self, f: Field, dir: i32) {
        let s = &mut self.settings;
        let d = dir as f32;
        match f {
            Field::Pattern => {
                let n = s.kind.patterns().len() as i32;
                s.pattern = (s.pattern as i32 + dir).rem_euclid(n) as usize;
            }
            Field::Density => s.density = (s.density + 0.05 * d).clamp(0.05, 1.0),
            Field::Radius => s.lenia.radius = (s.lenia.radius as i32 + dir).clamp(3, 40) as u32,
            Field::Mu => s.lenia.mu = (s.lenia.mu + 0.005 * d).clamp(0.01, 0.6),
            Field::Sigma => s.lenia.sigma = (s.lenia.sigma + 0.001 * d).clamp(0.002, 0.2),
            Field::Dt => s.lenia.dt = (s.lenia.dt + 0.01 * d).clamp(0.01, 1.0),
            Field::Generations => {
                s.generations = (s.generations as i64 + 100 * dir as i64).max(0) as u64
            }
            Field::Speed => {
                s.steps_per_frame = if dir > 0 {
                    (s.steps_per_frame * 2).min(256)
                } else {
                    (s.steps_per_frame / 2).max(1)
                }
            }
        }
    }

    pub fn handle_key(&mut self, key: KeyCode) -> SetupAction {
        let list = fields(self.settings.kind);
        let sel = self.list.selected().unwrap_or(0).min(list.len() - 1);
        match key {
            KeyCode::Up | KeyCode::Char('k') => {
                self.list.select(Some((sel + list.len() - 1) % list.len()))
            }
            KeyCode::Down | KeyCode::Char('j') => self.list.select(Some((sel + 1) % list.len())),
            KeyCode::Left | KeyCode::Char('h') => self.adjust(list[sel], -1),
            KeyCode::Right | KeyCode::Char('l') => self.adjust(list[sel], 1),
            KeyCode::Enter => return SetupAction::Start,
            KeyCode::Char('q') | KeyCode::Esc => return SetupAction::Back,
            _ => {}
        }
        SetupAction::None
    }

    pub fn draw(&mut self, frame: &mut Frame, area: Rect) {
        let items: Vec<String> = fields(self.settings.kind)
            .iter()
            .map(|&f| format!("{:<28} < {} >", f.label(), self.value(f)))
            .collect();
        let list = List::new(items)
            .block(
                Block::bordered()
                    .title(format!(" {} setup ", self.settings.kind.title()))
                    .title_bottom(
                        " up/down: select  left/right: change  Enter: start  q/Esc: back ",
                    ),
            )
            .highlight_symbol("> ")
            .highlight_style(Style::new().add_modifier(Modifier::REVERSED));
        frame.render_stateful_widget(list, area, &mut self.list);
    }
}

pub struct Run {
    settings: Settings,
    sim: Option<Automata>,
    paused: bool,
    /// Set by the main loop when a frame interval elapsed.
    pub due: bool,
    step_once: bool,
    /// Index into the Life patterns (never 0 = Random) used for spawning.
    spawn_pattern: usize,
    /// Grid area on screen, used to map mouse clicks to cells.
    inner: Rect,
    pending_spawn: Option<(i64, i64)>,
}

impl Run {
    pub fn new(settings: Settings) -> Self {
        Self {
            settings,
            sim: None,
            paused: false,
            due: true,
            step_once: false,
            spawn_pattern: 1,
            inner: Rect::default(),
            pending_spawn: None,
        }
    }

    fn life(&self) -> bool {
        self.settings.kind == Kind::Life
    }

    /// Spawns the selected pattern at the clicked cell (Life only).
    pub fn handle_click(&mut self, col: u16, row: u16) {
        let i = self.inner;
        if self.life() && col >= i.x && row >= i.y && col < i.x + i.width && row < i.y + i.height {
            self.pending_spawn = Some(((col - i.x) as i64 * 2 + 1, (row - i.y) as i64 * 4 + 2));
        }
    }

    fn finished(&self) -> bool {
        let g = self.settings.generations;
        g > 0 && self.sim.as_ref().is_some_and(|s| s.generation >= g)
    }

    /// Returns `true` when the user wants to leave the screen.
    pub fn handle_key(&mut self, key: KeyCode) -> bool {
        match key {
            KeyCode::Char('q') | KeyCode::Esc => return true,
            KeyCode::Char(' ') => self.paused = !self.paused,
            KeyCode::Char('n') => self.step_once = true,
            KeyCode::Char('r') => self.sim = None,
            KeyCode::Char('p') if self.life() => {
                let n = Kind::Life.patterns().len();
                self.spawn_pattern = self.spawn_pattern % (n - 1) + 1;
            }
            KeyCode::Char('s') if self.life() => {
                if let Some(sim) = &self.sim {
                    self.pending_spawn = Some((
                        rand::random_range(0..sim.width as i64),
                        rand::random_range(0..sim.height as i64),
                    ));
                }
            }
            KeyCode::Char('+') | KeyCode::Char('=') => {
                self.settings.steps_per_frame = (self.settings.steps_per_frame * 2).min(256)
            }
            KeyCode::Char('-') => {
                self.settings.steps_per_frame = (self.settings.steps_per_frame / 2).max(1)
            }
            _ => {}
        }
        false
    }

    pub fn draw(&mut self, frame: &mut Frame, area: Rect, gpu: &Gpu, error: &mut Option<String>) {
        let kind = self.settings.kind;
        let block = Block::bordered();
        let inner = block.inner(area);
        self.inner = inner;
        if inner.width == 0 || inner.height == 0 {
            frame.render_widget(block, area);
            return;
        }
        // Braille packs 2x4 dots per cell, half blocks 1x2 pixels per cell.
        let (w, h) = match kind {
            Kind::Life => (inner.width as u32 * 2, inner.height as u32 * 4),
            Kind::Lenia => (inner.width as u32, inner.height as u32 * 2),
        };
        if self
            .sim
            .as_ref()
            .is_none_or(|s| (s.width, s.height) != (w, h))
        {
            self.sim = Some(Automata::new(gpu, &self.settings, w, h));
        }
        let finished = self.finished();
        let sim = self.sim.as_mut().unwrap();
        if let Some((x, y)) = self.pending_spawn.take() {
            sim.spawn(gpu, self.spawn_pattern, x, y);
        }
        if self.due && !self.paused && !finished || self.step_once {
            let remaining = match self.settings.generations {
                0 => u64::MAX,
                g => g.saturating_sub(sim.generation),
            };
            let n = (self.settings.steps_per_frame as u64)
                .min(remaining)
                .max(self.step_once as u64);
            sim.step(gpu, n as u32);
        }
        self.due = false;
        self.step_once = false;

        let gen_limit = self.settings.generations;
        let status = if gen_limit > 0 && sim.generation >= gen_limit {
            "done"
        } else if self.paused {
            "paused"
        } else {
            "running"
        };
        let gens = match self.settings.generations {
            0 => sim.generation.to_string(),
            g => format!("{}/{g}", sim.generation),
        };
        let block = block
            .title(format!(
                " {} | {w}x{h} cells | gen {gens} | {}x/frame | {status} ",
                kind.title(),
                self.settings.steps_per_frame
            ))
            .title_bottom(if kind == Kind::Life {
                format!(
                    " click/s: spawn {}  p: pattern  space: pause  n: step  +/-: speed  r: restart  q: back ",
                    Kind::Life.patterns()[self.spawn_pattern]
                )
            } else {
                " space: pause  n: step  +/-: speed  r: restart  q/Esc: back ".to_string()
            });
        frame.render_widget(block, area);

        match sim.read(gpu) {
            Ok(data) => match kind {
                Kind::Life => render_braille(frame, inner, &data, w as usize),
                Kind::Lenia => render_half_blocks(frame, inner, &data, w as usize),
            },
            Err(e) => *error = Some(e.to_string()),
        }
    }
}

fn render_braille(frame: &mut Frame, inner: Rect, data: &[f32], w: usize) {
    // Dot bit for (column, row) inside a braille cell.
    const BITS: [[u32; 4]; 2] = [[0x01, 0x02, 0x04, 0x40], [0x08, 0x10, 0x20, 0x80]];
    let buf = frame.buffer_mut();
    for cy in 0..inner.height as usize {
        for cx in 0..inner.width as usize {
            let mut bits = 0;
            for (dx, col) in BITS.iter().enumerate() {
                for (dy, bit) in col.iter().enumerate() {
                    if data[(cy * 4 + dy) * w + cx * 2 + dx] > 0.5 {
                        bits |= bit;
                    }
                }
            }
            if bits != 0 {
                buf[(inner.x + cx as u16, inner.y + cy as u16)]
                    .set_char(char::from_u32(0x2800 + bits).unwrap())
                    .set_fg(Color::LightGreen);
            }
        }
    }
}

fn palette(v: f32) -> Color {
    let v = v.clamp(0.0, 1.0);
    let r = 255.0 * v * v;
    let g = 220.0 * v;
    let b = 40.0 + 360.0 * v * (1.0 - v);
    Color::Rgb(r as u8, g as u8, b.min(255.0) as u8)
}

fn render_half_blocks(frame: &mut Frame, inner: Rect, data: &[f32], w: usize) {
    let buf = frame.buffer_mut();
    for cy in 0..inner.height as usize {
        for cx in 0..inner.width as usize {
            buf[(inner.x + cx as u16, inner.y + cy as u16)]
                .set_char('▀')
                .set_fg(palette(data[cy * 2 * w + cx]))
                .set_bg(palette(data[(cy * 2 + 1) * w + cx]));
        }
    }
}
