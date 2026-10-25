mod automata;
mod automata_ui;
mod cli;
mod editor;
mod examples;
#[allow(dead_code)]
mod gpu;
mod gpuinfo;
mod graph;
mod mandelbrot;
mod matrix;
mod matrix_ui;
mod search;

use anyhow::Result;
use automata::Kind;
use automata_ui::{Run, Setup, SetupAction};
use clap::Parser;
use gpu::Gpu;
use gpuinfo::{Monitor, Telemetry};
use mandelbrot::{Mandelbrot, View};
use matrix_ui::MatrixUi;
use std::time::{Duration, Instant};

use ratatui::{
    DefaultTerminal, Frame,
    crossterm::{
        event::{
            self, DisableMouseCapture, EnableMouseCapture, Event, KeyCode, KeyEventKind,
            MouseButton, MouseEventKind,
        },
        execute,
    },
    layout::{Constraint, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Clear, List, ListState, Paragraph},
};

const MENU: [&str; 7] = [
    "Mandelbrot",
    "Game of Life",
    "Lenia",
    "Matrices",
    "Matrix examples",
    "Select GPU",
    "Exit",
];
const FRAME: Duration = Duration::from_millis(33);
const DEFAULT_SCALE: f32 = 0.05;

struct Renderer {
    gpu: Gpu,
    mandelbrot: Mandelbrot,
    cache: Option<((u32, u32, View), Vec<u32>)>,
}

fn gib(bytes: u64) -> f64 {
    bytes as f64 / (1u64 << 30) as f64
}

fn usage_color(pct: u32) -> Color {
    match pct {
        0..=50 => Color::Rgb(0, 130, 0),
        51..=85 => Color::Rgb(180, 100, 0),
        _ => Color::Red,
    }
}

fn draw_header(
    frame: &mut Frame,
    area: Rect,
    r: Option<&Renderer>,
    telemetry: &Result<Telemetry, String>,
) {
    let dim = Style::new().fg(Color::Rgb(90, 90, 100));
    let mut lines = Vec::new();
    match r {
        Some(r) => {
            let i = &r.gpu.adapter_info;
            lines.push(Line::from(vec![
                Span::styled("Adapter: ", dim),
                Span::raw(format!(
                    "{} | {:?} | {:?}",
                    i.name, i.backend, i.device_type
                )),
            ]));
            if !i.driver.is_empty() {
                lines.push(Line::from(vec![
                    Span::styled("Driver:  ", dim),
                    Span::raw(format!("{} {}", i.driver, i.driver_info)),
                ]));
            }
        }
        None => lines.push(Line::styled(
            "No GPU adapter available",
            Style::new().fg(Color::Red),
        )),
    }
    match telemetry {
        Ok(t) => {
            let vram_pct = (t.vram_used as f64 / t.vram_total.max(1) as f64 * 100.0) as u32;
            let mut spans = vec![
                Span::styled("NVIDIA:  ", dim),
                Span::raw(format!("{} | ", t.name)),
                Span::raw(format!(
                    "Cores {} | ",
                    t.cores.map_or("n/a".into(), |c| c.to_string())
                )),
                Span::raw("Usage "),
                Span::styled(
                    format!("{}%", t.gpu_util),
                    Style::new().fg(usage_color(t.gpu_util)),
                ),
                Span::raw(" | VRAM "),
                Span::styled(
                    format!(
                        "{:.1}/{:.1} GiB ({vram_pct}%)",
                        gib(t.vram_used),
                        gib(t.vram_total)
                    ),
                    Style::new().fg(usage_color(vram_pct)),
                ),
            ];
            if let Some(c) = t.temp_c {
                spans.push(Span::raw(format!(" | {c}°C")));
            }
            if let Some(w) = t.power_w {
                spans.push(Span::raw(format!(" | {w:.0} W")));
            }
            lines.push(Line::from(spans));
        }
        Err(e) => lines.push(Line::from(vec![
            Span::styled("Telemetry: ", dim),
            Span::raw(format!("unavailable ({e})")),
        ])),
    }
    frame.render_widget(
        Paragraph::new(lines)
            .style(Style::new().bg(Color::Rgb(230, 230, 235)).fg(Color::Black))
            .block(Block::bordered().title(" GPU ")),
        area,
    );
}

enum Screen {
    Menu,
    Mandelbrot(View),
    Setup(Setup),
    Run(Run),
    Gpus(Vec<wgpu::AdapterInfo>, ListState),
    Matrix(MatrixUi),
}

fn make_renderer(gpu: Gpu) -> Renderer {
    let mandelbrot = Mandelbrot::new(&gpu);
    Renderer {
        gpu,
        mandelbrot,
        cache: None,
    }
}

fn draw_gpus(frame: &mut Frame, area: Rect, infos: &[wgpu::AdapterInfo], state: &mut ListState) {
    let items = infos
        .iter()
        .map(|i| format!("{} | {:?} | {:?}", i.name, i.backend, i.device_type));
    let list = List::new(items)
        .block(
            Block::bordered()
                .title(" Select GPU ")
                .title_bottom(" up/down: move  Enter: select  F1: help  q/Esc: back "),
        )
        .highlight_symbol("> ")
        .highlight_style(Style::new().add_modifier(Modifier::REVERSED));
    frame.render_stateful_widget(list, area, state);
}

fn color(iter: u32, max_iter: u32) -> Color {
    if iter >= max_iter {
        return Color::Black;
    }
    let t = iter as f32 * 0.1;
    let ch = |phase: f32| (127.5 + 127.5 * (t + phase).sin()) as u8;
    Color::Rgb(ch(0.0), ch(2.1), ch(4.2))
}

fn help_text(screen: &Screen) -> (&'static str, &'static [&'static str]) {
    match screen {
        Screen::Menu | Screen::Gpus(..) | Screen::Setup(_) => (
            "Menu / lists",
            &[
                "Up/k, Down/j   select",
                "Left/Right     change a setup value",
                "Enter          open / confirm",
                "q, Esc         back / quit",
            ],
        ),
        Screen::Mandelbrot(_) => (
            "Mandelbrot",
            &[
                "Arrows   pan",
                "+ / -    zoom",
                "r        reset view",
                "q, Esc   back",
            ],
        ),
        Screen::Run(_) => (
            "Simulation",
            &[
                "space    pause / resume",
                "n        single step",
                "+ / -    change speed",
                "r        restart",
                "Life: click or s spawn the pattern at the mouse pointer",
                "      (s spawns randomly if the pointer is outside the grid)",
                "Life: p  next pattern",
                "q, Esc   back",
            ],
        ),
        Screen::Matrix(_) => matrix_ui::HELP,
    }
}

fn draw_help(frame: &mut Frame, area: Rect, screen: &Screen) {
    let (title, lines) = help_text(screen);
    let width = lines
        .iter()
        .map(|l| l.len())
        .max()
        .unwrap_or(0)
        .max(title.len()) as u16
        + 4;
    let height = lines.len() as u16 + 4;
    let [_, row, _] = Layout::vertical([
        Constraint::Min(0),
        Constraint::Length(height.min(area.height)),
        Constraint::Min(0),
    ])
    .flex(ratatui::layout::Flex::Center)
    .areas(area);
    let [_, popup, _] = Layout::horizontal([
        Constraint::Min(0),
        Constraint::Length(width.min(area.width)),
        Constraint::Min(0),
    ])
    .flex(ratatui::layout::Flex::Center)
    .areas(row);
    frame.render_widget(Clear, popup);
    frame.render_widget(
        Paragraph::new(lines.iter().map(|l| Line::from(*l)).collect::<Vec<_>>()).block(
            Block::bordered()
                .title(format!(" Help: {title} "))
                .title_bottom(" any key: close ")
                .border_style(Style::new().fg(Color::Cyan)),
        ),
        popup,
    );
}

fn draw_menu(frame: &mut Frame, area: Rect, state: &mut ListState, error: Option<&str>) {
    let [list_area, status] =
        Layout::vertical([Constraint::Min(0), Constraint::Length(1)]).areas(area);
    let list = List::new(MENU)
        .block(Block::bordered().title(" GPU Task Executor "))
        .highlight_symbol("> ")
        .highlight_style(Style::new().add_modifier(Modifier::REVERSED));
    frame.render_stateful_widget(list, list_area, state);
    if let Some(e) = error {
        frame.render_widget(Paragraph::new(e).style(Style::new().fg(Color::Red)), status);
    }
}

fn draw_mandelbrot(
    frame: &mut Frame,
    area: Rect,
    r: &mut Renderer,
    view: View,
    error: &mut Option<String>,
) {
    let block = Block::bordered()
        .title(" Mandelbrot (GPU) ")
        .title_bottom(" arrows: pan  +/-: zoom  r: reset  F1: help  q/Esc: back ");
    let inner = block.inner(area);
    frame.render_widget(block, area);
    if inner.width == 0 || inner.height == 0 {
        return;
    }
    // Each cell shows two vertically stacked pixels via the upper-half block.
    let (w, h) = (inner.width as u32, inner.height as u32 * 2);
    let key = (w, h, view);
    if r.cache.as_ref().is_none_or(|(k, _)| *k != key) {
        match r.mandelbrot.render(&r.gpu, w, h, view) {
            Ok(data) => r.cache = Some((key, data)),
            Err(e) => *error = Some(e.to_string()),
        }
    }
    match &r.cache {
        Some((_, data)) => {
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
        None => {}
    }
}

fn zoomed(view: View, factor: f32) -> View {
    let scale = (view.scale * factor).clamp(1e-6, 1.0);
    let max_iter = (200.0 + 40.0 * (DEFAULT_SCALE / scale).log2().max(0.0)) as u32;
    View {
        scale,
        max_iter,
        ..view
    }
}

fn app(terminal: &mut DefaultTerminal, gpu_index: Option<usize>) -> Result<()> {
    let mut menu = ListState::default().with_selected(Some(0));
    let mut screen = Screen::Menu;
    let monitor = Monitor::new();
    let mut error: Option<String> = None;
    let mut renderer = match cli::open_gpu(gpu_index) {
        Ok(gpu) => Some(make_renderer(gpu)),
        Err(e) => {
            error = Some(format!("GPU init failed: {e:#}"));
            None
        }
    };

    let mut help = false;
    let mut telemetry = monitor.sample();
    let mut sampled = Instant::now();

    loop {
        if sampled.elapsed() >= Duration::from_secs(1) {
            telemetry = monitor.sample();
            sampled = Instant::now();
        }
        terminal.draw(|f| {
            let [header, body] =
                Layout::vertical([Constraint::Length(5), Constraint::Min(0)]).areas(f.area());
            draw_header(f, header, renderer.as_ref(), &telemetry);
            match (&mut screen, renderer.as_mut()) {
                (Screen::Mandelbrot(view), Some(r)) => {
                    draw_mandelbrot(f, body, r, *view, &mut error)
                }
                (Screen::Run(run), Some(r)) => run.draw(f, body, &r.gpu, &mut error),
                (Screen::Setup(setup), _) => setup.draw(f, body),
                (Screen::Matrix(m), _) => m.draw(f, body),
                (Screen::Gpus(infos, state), _) => draw_gpus(f, body, infos, state),
                _ => draw_menu(f, body, &mut menu, error.as_deref()),
            }
            if help {
                draw_help(f, body, &screen);
            }
        })?;
        if error.is_some() {
            screen = Screen::Menu;
        }

        let timeout = if matches!(screen, Screen::Run(_)) {
            FRAME
        } else {
            Duration::from_secs(1)
        };
        if !event::poll(timeout)? {
            if let Screen::Run(run) = &mut screen {
                run.due = true;
            }
            continue;
        }
        let key = match event::read()? {
            Event::Key(key) => key,
            Event::Mouse(m) => {
                if let Screen::Run(run) = &mut screen {
                    match m.kind {
                        MouseEventKind::Down(MouseButton::Left) => {
                            run.handle_click(m.column, m.row)
                        }
                        _ => run.set_mouse(m.column, m.row),
                    }
                }
                continue;
            }
            _ => continue,
        };
        if key.kind != KeyEventKind::Press {
            continue;
        }
        if help {
            help = false;
            continue;
        }
        // The matrix editor types text, so it uses F1 only.
        let typing = matches!(screen, Screen::Matrix(_));
        if key.code == KeyCode::F(1) || (key.code == KeyCode::Char('h') && !typing) {
            help = true;
            continue;
        }
        match &mut screen {
            Screen::Menu => match key.code {
                KeyCode::Up | KeyCode::Char('k') => menu.select_previous(),
                KeyCode::Down | KeyCode::Char('j') => menu.select_next(),
                KeyCode::Char('q') | KeyCode::Esc => return Ok(()),
                KeyCode::Enter => match menu.selected() {
                    Some(0) if renderer.is_some() => {
                        error = None;
                        screen = Screen::Mandelbrot(View::default());
                    }
                    Some(i @ (1 | 2)) if renderer.is_some() => {
                        error = None;
                        screen = Screen::Setup(Setup::new(if i == 1 {
                            Kind::Life
                        } else {
                            Kind::Lenia
                        }));
                    }
                    Some(3) if renderer.is_some() => {
                        error = None;
                        screen = Screen::Matrix(MatrixUi::new());
                    }
                    Some(4) if renderer.is_some() => {
                        error = None;
                        screen = Screen::Matrix(MatrixUi::with_examples());
                    }
                    Some(5) => {
                        let infos = gpu::list_adapters();
                        let current = renderer.as_ref().and_then(|r| {
                            infos.iter().position(|i| {
                                i.name == r.gpu.adapter_info.name
                                    && i.backend == r.gpu.adapter_info.backend
                            })
                        });
                        let state = ListState::default().with_selected(current.or(Some(0)));
                        screen = Screen::Gpus(infos, state);
                    }
                    Some(6) => return Ok(()),
                    _ => {}
                },
                _ => {}
            },
            Screen::Setup(setup) => match setup.handle_key(key.code) {
                SetupAction::Back => screen = Screen::Menu,
                SetupAction::Start => screen = Screen::Run(Run::new(setup.settings.clone())),
                SetupAction::None => {}
            },
            Screen::Gpus(infos, state) => match key.code {
                KeyCode::Up | KeyCode::Char('k') => state.select_previous(),
                KeyCode::Down | KeyCode::Char('j') => state.select_next(),
                KeyCode::Char('q') | KeyCode::Esc => screen = Screen::Menu,
                KeyCode::Enter => {
                    if let Some(i) = state.selected().filter(|&i| i < infos.len()) {
                        match Gpu::with_adapter(i) {
                            Ok(gpu) => {
                                renderer = Some(make_renderer(gpu));
                                error = None;
                            }
                            Err(e) => error = Some(format!("GPU init failed: {e:#}")),
                        }
                        screen = Screen::Menu;
                    }
                }
                _ => {}
            },
            Screen::Matrix(m) => {
                if let Some(r) = renderer.as_ref()
                    && m.handle_key(key, &r.gpu)
                {
                    screen = Screen::Menu;
                }
            }
            Screen::Run(run) => {
                if run.handle_key(key.code) {
                    screen = Screen::Menu;
                }
            }
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
    let args = cli::Cli::parse();
    if args.is_batch() {
        if let Err(e) = cli::run(&args) {
            eprintln!("error: {e:#}");
            std::process::exit(1);
        }
        return Ok(());
    }
    ratatui::run(|terminal| {
        execute!(std::io::stdout(), EnableMouseCapture)?;
        let result = app(terminal, args.gpu);
        execute!(std::io::stdout(), DisableMouseCapture)?;
        result
    })
}
