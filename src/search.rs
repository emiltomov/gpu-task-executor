use crate::matrix::Matrix;
use anyhow::{Context, Result, bail};
use std::cmp::Reverse;
use std::collections::{BinaryHeap, VecDeque};
use std::io::Write;

const MAX_MAZE_CELLS: usize = 1024;

/// Adjacency matrix of a random perfect maze (a spanning tree of a `w` x `h` grid, cell
/// `y * w + x` is node `y * w + x`) carved by randomized depth-first search.
pub fn maze(w: usize, h: usize, seed: u64) -> Result<Matrix> {
    let n = w * h;
    if w == 0 || h == 0 || n > MAX_MAZE_CELLS {
        bail!("maze needs W x H between 1 and {MAX_MAZE_CELLS} cells, got {w}x{h}");
    }
    let mut rng = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    let mut next = move || {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        rng
    };
    let mut data = vec![0.0; n * n];
    let mut seen = vec![false; n];
    let mut stack = vec![0];
    seen[0] = true;
    while let Some(&c) = stack.last() {
        let (x, y) = (c % w, c / w);
        let mut free = Vec::with_capacity(4);
        if x > 0 {
            free.push(c - 1);
        }
        if x + 1 < w {
            free.push(c + 1);
        }
        if y > 0 {
            free.push(c - w);
        }
        if y + 1 < h {
            free.push(c + w);
        }
        free.retain(|&o| !seen[o]);
        if free.is_empty() {
            stack.pop();
            continue;
        }
        let o = free[next() as usize % free.len()];
        seen[o] = true;
        data[c * n + o] = 1.0;
        data[o * n + c] = 1.0;
        stack.push(o);
    }
    Ok(Matrix {
        rows: n,
        cols: n,
        data,
    })
}

fn neighbors(m: &Matrix, i: usize) -> impl Iterator<Item = (usize, f32)> + '_ {
    m.data[i * m.cols..(i + 1) * m.cols]
        .iter()
        .enumerate()
        .filter(|(_, w)| **w != 0.0)
        .map(|(j, &w)| (j, w.abs()))
}

fn check(m: &Matrix, nodes: &[usize]) -> Result<()> {
    if m.rows != m.cols {
        bail!(
            "graph search needs a square matrix, got {}x{}",
            m.rows,
            m.cols
        );
    }
    if let Some(&bad) = nodes.iter().find(|&&v| v >= m.rows) {
        bail!("node {bad} is outside the graph (0..{})", m.rows);
    }
    Ok(())
}

/// Breadth-first discovery: every node reachable from `start` gets `hops + 1`, others 0.
pub fn bfs(m: &Matrix, start: usize) -> Result<Matrix> {
    check(m, &[start])?;
    let mut dist = vec![0.0f32; m.rows];
    dist[start] = 1.0;
    let mut queue = VecDeque::from([start]);
    while let Some(i) = queue.pop_front() {
        for (j, _) in neighbors(m, i) {
            if dist[j] == 0.0 {
                dist[j] = dist[i] + 1.0;
                queue.push_back(j);
            }
        }
    }
    Ok(column(dist))
}

/// A* shortest path (edge weights are costs). With a grid `width` the Manhattan distance
/// between grid cells is the heuristic, otherwise it degrades to Dijkstra. Nodes on the path
/// get their 1-based step number, others 0.
pub fn astar(m: &Matrix, start: usize, goal: usize, width: Option<usize>) -> Result<Matrix> {
    check(m, &[start, goal])?;
    let min_cost = m
        .data
        .iter()
        .filter(|w| **w != 0.0)
        .map(|w| w.abs())
        .fold(f32::INFINITY, f32::min);
    let h = |i: usize| match width {
        Some(w) if w > 0 => {
            let (dx, dy) = ((i % w).abs_diff(goal % w), (i / w).abs_diff(goal / w));
            (dx + dy) as f32 * min_cost
        }
        _ => 0.0,
    };
    let n = m.rows;
    let mut g = vec![f32::INFINITY; n];
    let mut prev = vec![usize::MAX; n];
    g[start] = 0.0;
    // Costs are non-negative, so their bit patterns order like the floats themselves.
    let mut open = BinaryHeap::from([Reverse((h(start).to_bits(), start))]);
    while let Some(Reverse((f, i))) = open.pop() {
        if i == goal {
            break;
        }
        if f > (g[i] + h(i)).to_bits() {
            continue;
        }
        for (j, cost) in neighbors(m, i) {
            let cand = g[i] + cost;
            if cand < g[j] {
                g[j] = cand;
                prev[j] = i;
                open.push(Reverse(((cand + h(j)).to_bits(), j)));
            }
        }
    }
    if g[goal].is_infinite() {
        bail!("no path from {start} to {goal}");
    }
    let mut path = vec![goal];
    while *path.last().unwrap() != start {
        path.push(prev[*path.last().unwrap()]);
    }
    let mut out = vec![0.0; n];
    for (step, &node) in path.iter().rev().enumerate() {
        out[node] = step as f32 + 1.0;
    }
    Ok(column(out))
}

/// Draws a grid maze (node `i` at column `i % width`) as text: `#` walls, open passages blank.
/// Cells (and the passages between two cells) with a positive `marks` value show `mark(value)`.
pub fn maze_art(
    m: &Matrix,
    width: usize,
    marks: Option<&Matrix>,
    mark: impl Fn(f32) -> char,
) -> String {
    let height = m.rows.div_ceil(width);
    let mut grid = vec![vec!['#'; 2 * width + 1]; 2 * height + 1];
    let value = |i: usize| marks.map_or(0.0, |k| k.data[i]);
    for i in 0..m.rows {
        let (x, y) = (i % width, i / width);
        grid[2 * y + 1][2 * x + 1] = if value(i) > 0.0 { mark(value(i)) } else { ' ' };
        for (j, _) in neighbors(m, i) {
            let (jx, jy) = (j % width, j / width);
            grid[y + jy + 1][x + jx + 1] = if value(i) > 0.0 && value(j) > 0.0 {
                mark(value(i).max(value(j)))
            } else {
                ' '
            };
        }
    }
    grid.iter()
        .map(|r| r.iter().collect::<String>() + "\n")
        .collect()
}

/// Writes a zip with the maze, its adjacency matrix and A*/BFS solutions from the first to the
/// last cell, plus the user's input and current result.
pub fn save_maze_bundle(
    path: &str,
    m: &Matrix,
    width: usize,
    input: &str,
    result: Option<&Matrix>,
) -> Result<()> {
    if width == 0 || m.rows % width != 0 {
        bail!("grid width {width} does not fit {} nodes", m.rows);
    }
    let (start, goal) = (0, m.rows - 1);
    let route = astar(m, start, goal, Some(width))?;
    let dist = bfs(m, start)?;
    let mut files = vec![
        ("input.txt".to_string(), input.to_string()),
        ("maze.txt".to_string(), maze_art(m, width, None, |_| ' ')),
        (
            format!("solution_astar_{start}_to_{goal}.txt"),
            maze_art(m, width, Some(&route), |_| '*'),
        ),
        (
            format!("discovered_bfs_from_{start}.txt"),
            maze_art(m, width, Some(&dist), |v| {
                char::from_digit((v as u32 - 1) % 10, 10).unwrap_or('.')
            }),
        ),
        (
            "adjacency.txt".to_string(),
            format!("M = {}\n", m.to_literal()),
        ),
        (
            "astar_path.txt".to_string(),
            format!("path = {}\n", route.to_literal()),
        ),
        (
            "bfs_distances.txt".to_string(),
            format!("distances = {}\n", dist.to_literal()),
        ),
    ];
    if let Some(r) = result {
        files.push((
            "result.txt".to_string(),
            format!("result = {}\n", r.to_literal()),
        ));
    }
    let file = std::fs::File::create(crate::matrix::expand_tilde(path)).with_context(|| format!("cannot write '{path}'"))?;
    let mut zip = zip::ZipWriter::new(file);
    let options = zip::write::SimpleFileOptions::default();
    for (name, text) in files {
        zip.start_file(name, options)?;
        zip.write_all(text.as_bytes())?;
    }
    zip.finish()?;
    Ok(())
}

fn column(data: Vec<f32>) -> Matrix {
    Matrix {
        rows: data.len(),
        cols: 1,
        data,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn maze_is_a_connected_tree() {
        let m = maze(21, 24, 7).unwrap();
        assert_eq!(m.rows, 504);
        // A spanning tree has n - 1 undirected edges.
        assert_eq!(m.data.iter().filter(|v| **v != 0.0).count(), 2 * 503);
        assert!(bfs(&m, 0).unwrap().data.iter().all(|&d| d > 0.0));
        assert!(maze(0, 3, 1).is_err() && maze(40, 40, 1).is_err());
    }

    #[test]
    fn astar_finds_the_unique_maze_path() {
        let m = maze(21, 24, 7).unwrap();
        let dist = bfs(&m, 0).unwrap();
        for width in [Some(21), None] {
            let path = astar(&m, 0, 503, width).unwrap();
            assert_eq!(
                path.data.iter().cloned().fold(0.0, f32::max),
                dist.data[503]
            );
            assert_eq!((path.data[0], path.data[503]), (1.0, dist.data[503]));
        }
    }

    #[test]
    fn bundle_zip_has_maze_and_solutions() {
        let m = maze(6, 5, 3).unwrap();
        let art = maze_art(&m, 6, None, |_| ' ');
        assert_eq!(art.lines().count(), 11);
        assert!(art.lines().all(|l| l.len() == 13));
        let route = astar(&m, 0, 29, Some(6)).unwrap();
        let solved = maze_art(&m, 6, Some(&route), |_| '*');
        // Path of k cells shows k cells and k - 1 passages.
        assert_eq!(
            solved.matches('*').count() as f32,
            route.data[29] * 2.0 - 1.0
        );

        let file = std::env::temp_dir().join(format!("gtu-maze-{}.zip", std::process::id()));
        save_maze_bundle(file.to_str().unwrap(), &m, 6, "M = maze 6 5 3\n", None).unwrap();
        let mut zip = zip::ZipArchive::new(std::fs::File::open(&file).unwrap()).unwrap();
        let names: Vec<String> = zip.file_names().map(|n| n.unwrap().to_string()).collect();
        std::fs::remove_file(&file).unwrap();
        for want in [
            "maze.txt",
            "adjacency.txt",
            "solution_astar_0_to_29.txt",
            "discovered_bfs_from_0.txt",
        ] {
            assert!(
                names.iter().any(|n| n == want),
                "{want} missing in {names:?}"
            );
        }
        assert!(zip.by_name("input.txt").is_ok());
        assert!(save_maze_bundle("x.zip", &m, 4, "", None).is_err());
    }

    #[test]
    fn astar_prefers_cheaper_weighted_route() {
        // 0 -> 2 directly costs 5, via 1 costs 1 + 1.
        let m = Matrix::parse("[0 1 5; 0 0 1; 0 0 0]").unwrap();
        assert_eq!(astar(&m, 0, 2, None).unwrap().data, vec![1.0, 2.0, 3.0]);
        assert!(astar(&m, 2, 0, None).is_err() && astar(&m, 0, 9, None).is_err());
    }
}
