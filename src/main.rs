#[allow(dead_code)]
mod gpu;
mod mandelbrot;

use anyhow::Result;
use gpu::Gpu;
use mandelbrot::{Mandelbrot, View};
use ratatui::{
    DefaultTerminal, Frame,
    crossterm::event::{self, Event, KeyCode, KeyEventKind},
    layout::{Constraint, Layout},
    style::{Color, Modifier, Style},
    widgets::{Block, List, ListState, Paragraph},
};

const MENU: [&str; 2] = ["Mandelbrot", "Exit"];
const DEFAULT_SCALE: f32 = 0.05;

struct Renderer {
    gpu: Gpu,
    mandelbrot: Mandelbrot,
}

enum Screen {
    Menu,
    Mandelbrot(View),
}

fn color(iter: u32, max_iter: u32) -> Color {
    if iter >= max_iter {
        return Color::Black;
    }
    let t = iter as f32 * 0.1;
    let ch = |phase: f32| (127.5 + 127.5 * (t + phase).sin()) as u8;
    Color::Rgb(ch(0.0), ch(2.1), ch(4.2))
}

fn draw_menu(frame: &mut Frame, state: &mut ListState, error: Option<&str>) {
    let [list_area, status] =
        Layout::vertical([Constraint::Min(0), Constraint::Length(1)]).areas(frame.area());
    let list = List::new(MENU)
        .block(Block::bordered().title(" GPU Task Executor "))
        .highlight_symbol("> ")
        .highlight_style(Style::new().add_modifier(Modifier::REVERSED));
    frame.render_stateful_widget(list, list_area, state);
    if let Some(e) = error {
        frame.render_widget(Paragraph::new(e).style(Style::new().fg(Color::Red)), status);
    }
}

fn draw_mandelbrot(frame: &mut Frame, r: &Renderer, view: View, error: &mut Option<String>) {
    let block = Block::bordered()
        .title(" Mandelbrot (GPU) ")
        .title_bottom(" arrows: pan  +/-: zoom  r: reset  q/Esc: back ");
    let inner = block.inner(frame.area());
    frame.render_widget(block, frame.area());
    if inner.width == 0 || inner.height == 0 {
        return;
    }
    // Each cell shows two vertically stacked pixels via the upper-half block.
    let (w, h) = (inner.width as u32, inner.height as u32 * 2);
    match r.mandelbrot.render(&r.gpu, w, h, view) {
        Ok(data) => {
            let buf = frame.buffer_mut();
            for cy in 0..inner.height {
                for cx in 0..inner.width {
                    let px = |row: u32| data[(row * w + cx as u32) as usize];
                    let top = color(px(cy as u32 * 2), view.max_iter);
                    let bottom = color(px(cy as u32 * 2 + 1), view.max_iter);
                    buf[(inner.x + cx, inner.y + cy)]
                        .set_symbol("▀")
                        .set_fg(top)
                        .set_bg(bottom);
                }
            }
        }
        Err(e) => *error = Some(e.to_string()),
    }
}

fn zoomed(view: View, factor: f32) -> View {
    let scale = (view.scale * factor).clamp(1e-6, 1.0);
    let max_iter = (200.0 + 40.0 * (DEFAULT_SCALE / scale).log2().max(0.0)) as u32;
    View { scale, max_iter, ..view }
}

fn app(terminal: &mut DefaultTerminal) -> Result<()> {
    let mut menu = ListState::default().with_selected(Some(0));
    let mut screen = Screen::Menu;
    let mut renderer: Option<Renderer> = None;
    let mut error: Option<String> = None;

    loop {
        terminal.draw(|f| match (&screen, &renderer) {
            (Screen::Mandelbrot(view), Some(r)) => draw_mandelbrot(f, r, *view, &mut error),
            _ => draw_menu(f, &mut menu, error.as_deref()),
        })?;
        if error.is_some() {
            screen = Screen::Menu;
        }

        let Event::Key(key) = event::read()? else { continue };
        if key.kind != KeyEventKind::Press {
            continue;
        }
        match &mut screen {
            Screen::Menu => match key.code {
                KeyCode::Up | KeyCode::Char('k') => menu.select_previous(),
                KeyCode::Down | KeyCode::Char('j') => menu.select_next(),
                KeyCode::Char('q') | KeyCode::Esc => return Ok(()),
                KeyCode::Enter => match menu.selected() {
                    Some(0) => {
                        error = None;
                        if renderer.is_none() {
                            match Gpu::new() {
                                Ok(gpu) => {
                                    let mandelbrot = Mandelbrot::new(&gpu);
                                    renderer = Some(Renderer { gpu, mandelbrot });
                                }
                                Err(e) => {
                                    error = Some(format!("GPU init failed: {e:#}"));
                                    continue;
                                }
                            }
                        }
                        screen = Screen::Mandelbrot(View::default());
                    }
                    Some(1) => return Ok(()),
                    _ => {}
                },
                _ => {}
            },
            Screen::Mandelbrot(view) => {
                let step = view.scale * 8.0;
                match key.code {
                    KeyCode::Left => view.center_x -= step,
                    KeyCode::Right => view.center_x += step,
                    KeyCode::Up => view.center_y -= step,
                    KeyCode::Down => view.center_y += step,
                    KeyCode::Char('+') | KeyCode::Char('=') => *view = zoomed(*view, 0.7),
                    KeyCode::Char('-') => *view = zoomed(*view, 1.0 / 0.7),
                    KeyCode::Char('r') => *view = View::default(),
                    KeyCode::Char('q') | KeyCode::Esc => screen = Screen::Menu,
                    _ => {}
                }
            }
        }
    }
}

fn main() -> Result<()> {
    ratatui::run(app)
}
