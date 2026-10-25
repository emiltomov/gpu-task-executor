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
use std::collections::HashSet;

const MAX_NODES: usize = 1000;
/// Above this many nodes the picture drops circles and labels so edges stay readable.
const DETAILED_NODES: usize = 60;
const NODE_RADIUS: f64 = 0.06;

/// Directed weighted graph read from an adjacency matrix (`m[i][j] != 0` is an edge i -> j).
pub struct Graph {
    pub n: usize,
    pub edges: Vec<(usize, usize, f32)>,
    pub pos: Vec<(f64, f64)>,
    /// Set for grid graphs (mazes): longest path and dead ends are highlighted.
    pub maze: Option<MazeInfo>,
}

pub struct MazeInfo {
    /// Undirected edges (smaller index first) along the longest path.
    pub longest: HashSet<(usize, usize)>,
    pub dead_ends: Vec<usize>,
}

impl Graph {
    /// `grid_width` places node `i` at column `i % w`, row `i / w` instead of force-directed.
    pub fn from_matrix(m: &Matrix, grid_width: Option<usize>) -> Result<Self> {
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
        let grid = grid_width.filter(|&w| w > 0);
        let pos = match grid {
            Some(w) => grid_layout(n, w),
            None => layout(n, &edges),
        };
        let maze = grid.map(|_| maze_info(n, &edges));
        Ok(Self {
            n,
            edges,
            pos,
            maze,
        })
    }

    fn weighted(&self) -> bool {
        self.edges.iter().any(|e| e.2 != 1.0)
    }

    fn has_edge(&self, a: usize, b: usize) -> bool {
        self.edges.iter().any(|e| e.0 == a && e.1 == b)
    }
}

/// Longest path (tree diameter by double BFS, exact for mazes) and dead ends (one neighbor).
fn maze_info(n: usize, edges: &[(usize, usize, f32)]) -> MazeInfo {
    let mut adj = vec![Vec::new(); n];
    for &(a, b, _) in edges.iter().filter(|e| e.0 != e.1) {
        adj[a].push(b);
        adj[b].push(a);
    }
    for list in &mut adj {
        list.sort_unstable();
        list.dedup();
    }
    let bfs = |from: usize| {
        let mut prev = vec![usize::MAX; n];
        let mut order = vec![from];
        prev[from] = from;
        let mut head = 0;
        while head < order.len() {
            let i = order[head];
            head += 1;
            for &j in &adj[i] {
                if prev[j] == usize::MAX {
                    prev[j] = i;
                    order.push(j);
                }
            }
        }
        (*order.last().unwrap(), prev)
    };
    let mut longest = HashSet::new();
    if n > 0 {
        let (a, _) = bfs(0);
        let (b, prev) = bfs(a);
        let mut cur = b;
        while cur != a {
            longest.insert((cur.min(prev[cur]), cur.max(prev[cur])));
            cur = prev[cur];
        }
    }
    let dead_ends = (0..n).filter(|&i| adj[i].len() == 1).collect();
    MazeInfo { longest, dead_ends }
}

fn grid_layout(n: usize, w: usize) -> Vec<(f64, f64)> {
    let h = n.div_ceil(w);
    let scale = |i: usize, len: usize| {
        if len < 2 {
            0.0
        } else {
            (i as f64 / (len - 1) as f64 * 2.0 - 1.0) * 0.9
        }
    };
    (0..n)
        .map(|i| (scale(i % w, w), -scale(i / w, h)))
        .collect()
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
                let on_longest = g
                    .maze
                    .as_ref()
                    .is_some_and(|m| m.longest.contains(&(a.min(b), a.max(b))));
                ctx.draw(&Line {
                    x1,
                    y1,
                    x2,
                    y2,
                    color: if on_longest {
                        Color::Green
                    } else {
                        Color::DarkGray
                    },
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
            if let Some(m) = &g.maze {
                for &i in &m.dead_ends {
                    let (x, y) = p(i);
                    ctx.print(x, y, Span::styled("█", Style::new().fg(Color::Red)));
                }
            }
            let detailed = g.n <= DETAILED_NODES;
            for i in 0..g.n {
                let (x, y) = p(i);
                let lit = values.is_some_and(|v| v.get(i).is_some_and(|&x| x > 0.0));
                // Mazes keep green for the longest path; result highlights turn yellow.
                let lit_color = if g.maze.is_some() {
                    Color::Yellow
                } else {
                    Color::Green
                };
                let color = if lit { lit_color } else { Color::Cyan };
                if !detailed {
                    if lit {
                        ctx.print(x, y, Span::styled("█", Style::new().fg(color)));
                    }
                    continue;
                }
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
        Graph::from_matrix(&Matrix::parse(text)?, None)
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
    fn grid_layout_places_cells_by_index() {
        let m = Matrix::parse("[0 1 0 0; 1 0 0 0; 0 0 0 1; 0 0 1 0]").unwrap();
        let g = Graph::from_matrix(&m, Some(2)).unwrap();
        assert!(g.pos[0].0 < g.pos[1].0 && g.pos[0].1 > g.pos[2].1);
        assert_eq!(g.pos[0].0, g.pos[2].0);
    }

    #[test]
    fn grid_graph_finds_longest_path_and_dead_ends() {
        // Path 0-1-2 plus a branch 1-3: dead ends are 0, 2, 3; longest path has 2 edges.
        let m = Matrix::parse("[0 1 0 0; 1 0 1 1; 0 1 0 0; 0 1 0 0]").unwrap();
        let g = Graph::from_matrix(&m, Some(2)).unwrap();
        let info = g.maze.unwrap();
        assert_eq!(info.dead_ends, vec![0, 2, 3]);
        assert_eq!(info.longest.len(), 2);
        assert!(Graph::from_matrix(&m, None).unwrap().maze.is_none());
    }

    #[test]
    fn rejects_non_square_and_handles_tiny_graphs() {
        assert!(graph("[1 2 3]").is_err());
        let g = graph("[0 1; 0 0]").unwrap();
        assert_eq!(g.edges, vec![(0, 1, 1.0)]);
        assert_eq!(graph("[1]").unwrap().pos.len(), 1);
    }
}
