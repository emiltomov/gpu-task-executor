use crate::matrix::{Matrix, fmt_num};
use anyhow::{Result, bail};
use ratatui::{
    Frame,
    layout::Rect,
    style::{Color, Style},
    symbols::Marker,
    text::Span,
    widgets::{
        Block,
        canvas::{Canvas, Circle, Line},
    },
};

const MAX_NODES: usize = 200;
const NODE_RADIUS: f64 = 0.06;

/// Directed weighted graph read from an adjacency matrix (`m[i][j] != 0` is an edge i -> j).
pub struct Graph {
    pub n: usize,
    pub edges: Vec<(usize, usize, f32)>,
    pub pos: Vec<(f64, f64)>,
}

impl Graph {
    pub fn from_matrix(m: &Matrix) -> Result<Self> {
        if m.rows != m.cols {
            bail!("a graph needs a square matrix, got {}x{}", m.rows, m.cols);
        }
        if m.rows > MAX_NODES {
            bail!(
                "graph too large to draw ({} nodes, max {MAX_NODES})",
                m.rows
            );
        }
        let n = m.rows;
        let edges = (0..n * n)
            .filter(|&i| m.data[i] != 0.0)
            .map(|i| (i / n, i % n, m.data[i]))
            .collect::<Vec<_>>();
        let pos = layout(n, &edges);
        Ok(Self { n, edges, pos })
    }

    fn weighted(&self) -> bool {
        self.edges.iter().any(|e| e.2 != 1.0)
    }

    fn has_edge(&self, a: usize, b: usize) -> bool {
        self.edges.iter().any(|e| e.0 == a && e.1 == b)
    }
}

/// Force-directed (Fruchterman-Reingold) layout starting from a circle, scaled into [-1, 1].
fn layout(n: usize, edges: &[(usize, usize, f32)]) -> Vec<(f64, f64)> {
    let mut pos: Vec<(f64, f64)> = (0..n)
        .map(|i| {
            let a = std::f64::consts::TAU * i as f64 / n.max(1) as f64;
            (a.cos(), a.sin())
        })
        .collect();
    if n < 3 {
        return pos;
    }
    let k = (4.0 / n as f64).sqrt();
    let iterations = 300;
    for it in 0..iterations {
        let temp = 0.2 * (1.0 - it as f64 / iterations as f64) + 0.002;
        let mut disp = vec![(0.0f64, 0.0f64); n];
        for i in 0..n {
            for j in (i + 1)..n {
                let (dx, dy) = (pos[i].0 - pos[j].0, pos[i].1 - pos[j].1);
                let d = (dx * dx + dy * dy).sqrt().max(1e-6);
                let f = k * k / d;
                disp[i].0 += dx / d * f;
                disp[i].1 += dy / d * f;
                disp[j].0 -= dx / d * f;
                disp[j].1 -= dy / d * f;
            }
        }
        for &(a, b, _) in edges {
            if a == b {
                continue;
            }
            let (dx, dy) = (pos[a].0 - pos[b].0, pos[a].1 - pos[b].1);
            let d = (dx * dx + dy * dy).sqrt().max(1e-6);
            let f = d * d / k;
            disp[a].0 -= dx / d * f;
            disp[a].1 -= dy / d * f;
            disp[b].0 += dx / d * f;
            disp[b].1 += dy / d * f;
        }
        for i in 0..n {
            let d = (disp[i].0.powi(2) + disp[i].1.powi(2)).sqrt().max(1e-6);
            let step = d.min(temp);
            pos[i].0 += disp[i].0 / d * step;
            pos[i].1 += disp[i].1 / d * step;
        }
    }
    let max = pos
        .iter()
        .flat_map(|p| [p.0.abs(), p.1.abs()])
        .fold(1e-6, f64::max);
    pos.iter()
        .map(|p| (p.0 / max * 0.85, p.1 / max * 0.85))
        .collect()
}

/// Draws the graph; `values` (one per node) colors positive nodes green and prints the value.
pub fn draw(frame: &mut Frame, area: Rect, title: &str, g: &Graph, values: Option<&[f32]>) {
    let block = Block::bordered().title(title.to_string());
    let inner = block.inner(area);
    // Braille gives 2x4 dots per cell; widen the x range so circles stay round.
    let aspect = (inner.width.max(1) as f64 * 2.0) / (inner.height.max(1) as f64 * 4.0);
    let canvas = Canvas::default()
        .block(block)
        .marker(Marker::Braille)
        .x_bounds([-aspect, aspect])
        .y_bounds([-1.0, 1.0])
        .paint(|ctx| {
            let p = |i: usize| (g.pos[i].0 * aspect, g.pos[i].1);
            for &(a, b, w) in &g.edges {
                if a == b {
                    continue;
                }
                let ((x1, y1), (x2, y2)) = (p(a), p(b));
                ctx.draw(&Line {
                    x1,
                    y1,
                    x2,
                    y2,
                    color: Color::DarkGray,
                });
                if !g.has_edge(b, a) {
                    arrowhead(ctx, (x1, y1), (x2, y2));
                }
                if g.weighted() {
                    ctx.print(
                        (x1 + x2) / 2.0,
                        (y1 + y2) / 2.0,
                        Span::styled(fmt_num(w), Style::new().fg(Color::Yellow)),
                    );
                }
            }
            for i in 0..g.n {
                let (x, y) = p(i);
                let lit = values.is_some_and(|v| v.get(i).is_some_and(|&x| x > 0.0));
                let color = if lit { Color::Green } else { Color::Cyan };
                ctx.draw(&Circle {
                    x,
                    y,
                    radius: NODE_RADIUS,
                    color,
                });
                let label = match values.and_then(|v| v.get(i)) {
                    Some(v) => format!("{i}={}", fmt_num(*v)),
                    None => i.to_string(),
                };
                ctx.print(x, y, Span::styled(label, Style::new().fg(color)));
            }
        });
    frame.render_widget(canvas, area);
}

fn arrowhead(ctx: &mut ratatui::widgets::canvas::Context, from: (f64, f64), to: (f64, f64)) {
    let (dx, dy) = (to.0 - from.0, to.1 - from.1);
    let d = (dx * dx + dy * dy).sqrt().max(1e-6);
    let (ux, uy) = (dx / d, dy / d);
    let tip = (to.0 - ux * NODE_RADIUS, to.1 - uy * NODE_RADIUS);
    for sign in [-1.0, 1.0] {
        let a: f64 = 0.45 * sign;
        let (c, s) = (a.cos(), a.sin());
        let (wx, wy) = (-(ux * c - uy * s), -(ux * s + uy * c));
        ctx.draw(&Line {
            x1: tip.0,
            y1: tip.1,
            x2: tip.0 + wx * 0.07,
            y2: tip.1 + wy * 0.07,
            color: Color::Gray,
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn graph(text: &str) -> Result<Graph> {
        Graph::from_matrix(&Matrix::parse(text)?)
    }

    #[test]
    fn reads_edges_and_layout_fits_unit_box() {
        let g = graph("[0 1 1 0 0; 1 0 1 0 0; 1 1 0 1 0; 0 0 1 0 1; 0 0 0 1 0]").unwrap();
        assert_eq!((g.n, g.edges.len()), (5, 10));
        assert_eq!(g.pos.len(), 5);
        assert!(g.pos.iter().all(|p| p.0.abs() <= 1.0 && p.1.abs() <= 1.0));
        // Connected nodes end up closer than the most distant pair.
        let d = |a: usize, b: usize| {
            ((g.pos[a].0 - g.pos[b].0).powi(2) + (g.pos[a].1 - g.pos[b].1).powi(2)).sqrt()
        };
        assert!(d(3, 4) < d(0, 4));
    }

    #[test]
    fn rejects_non_square_and_handles_tiny_graphs() {
        assert!(graph("[1 2 3]").is_err());
        let g = graph("[0 1; 0 0]").unwrap();
        assert_eq!(g.edges, vec![(0, 1, 1.0)]);
        assert_eq!(graph("[1]").unwrap().pos.len(), 1);
    }
}
