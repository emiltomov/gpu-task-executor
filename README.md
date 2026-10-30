# gpu-task-executor

A Rust terminal UI (`ratatui`) that runs compute workloads on the GPU via `wgpu` (WGSL shaders).

<div align="center">
  <img src="docs/GTU-Demo-Session.gif" alt="gtu demo session: matrix calculator, examples, graph and photo before/after" width="800">
  <br>
  <sub>Demo session: matrix calculator, bundled examples, graph view and photo before/after.</sub>
</div>

## Features

- **Mandelbrot** – per-pixel iteration counts computed on the GPU; pan and zoom.
- **Game of Life** – rendered with braille characters (2x4 cells per character).
- **Lenia** – continuous cellular automaton rendered with half blocks (1x2).
- **Matrices** – open **Matrix examples** in the main menu (or `F2` in the Matrices screen) to load and run the bundled examples from `examples/` (Markov chains, graph search/adjacency, lighting and raycast, real-photo enhance/blur with before/after). Define matrices (`A = [1 2; 3 4]`) and compose operations in one expression, e.g. `(A + B) * C' - 2 * A`. Supports `+ - *` (matrix product; a number scales), `'` transpose, unary `-`, parentheses, and linear transformations as functions (angles in degrees, applied to column vectors): `rot(a)`, `rotx/roty/rotz(a)`, `scale(sx, sy[, sz])`, `shearx(k)`, `sheary(k)`, `identity(n)`, `reflect(a)` (mirror across the line at angle `a`), `proj(a)` (projection onto it), e.g. `rot(90) * v`; `det(M)`/`trace(M)` give numbers, `flip(n)` (exchange matrix: `flip(n) * I` mirrors rows, `I * flip(n)` columns), `rows(M)`/`cols(M)` (size of any matrix, so examples fit any opened image), `resize(I, rows, cols)` scales an image up or down, `raycast(I, a)` casts one ray per pixel from a camera orbited `a` degrees around the y axis and resamples image `I` (black where rays miss), and arguments may be any expression, e.g. `roty(det(A * B) + 10)`. Arithmetic and transpose run on the GPU; reductions, transformation construction, image resampling and graph searches run on the CPU.
- GPU info header on every screen, with NVML telemetry (sampled at most once per second; shows "unavailable" if NVML is missing).

Automata state stays on the GPU (ping-pong storage buffers) and is read back only for drawing. Grid size follows the terminal; resizing restarts the simulation.

## Requirements

- Rust (edition 2024 toolchain)
- A Vulkan/Metal/DX12-capable GPU and drivers
- Optional: NVIDIA driver for telemetry

## Install dependencies

1. Install Rust via [rustup](https://rustup.rs):
   ```sh
   curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
   rustup update stable
   ```
2. Install GPU drivers (Linux example, Debian/Ubuntu):
   ```sh
   sudo apt install build-essential pkg-config mesa-vulkan-drivers vulkan-tools
   vulkaninfo --summary   # verify a GPU is visible
   ```
   NVIDIA users should install the proprietary driver (also provides NVML for telemetry).

   **macOS** (Apple Silicon or Intel; wgpu uses Metal, so no GPU driver install is needed):
   ```sh
   xcode-select --install      # Xcode command line tools (compiler, linker)
   brew install pkg-config     # optional; Homebrew from https://brew.sh
   ```
   Run `gtu` in a terminal that supports truecolor/Unicode (Terminal.app, iTerm2). NVML telemetry is
   NVIDIA-only, so the GPU header shows "unavailable" on macOS. The system clipboard uses the native
   pasteboard via `arboard`; no extra packages are required.
3. Cargo fetches the crate dependencies automatically on first build.

## Build

```sh
git clone <repo-url> && cd gpu-task-executor
cargo build            # debug
cargo build --release  # optimized
```

## Usage

```sh
cargo run --release   # needs a real terminal
```

Keys (`F1` opens a help window on every screen):

| Screen | Keys |
|--------|------|
| Menu / Setup | `↑`/`k`, `↓`/`j`, `Enter`, `q`/`Esc` |
| Mandelbrot | arrows pan, `+`/`-` zoom, `r` reset, `q`/`Esc` back |
| Matrices | `Tab` switch field, arrows/Home/End move (Shift selects), `Ctrl+C`/`X`/`V`/`A` copy/cut/paste (system clipboard, falling back to an internal one)/select all, `Enter` (in expression) or `F5` calculate, `Ctrl+S` save input, `Ctrl+E` export a `graph NAME grid W` maze as a zip (maze/solution/discovery drawings, adjacency, A*/BFS vectors, input, result), `Ctrl+E` also exports image results as PNG, `Ctrl+N` (image examples) uses the latest result as the input picture and calculates again, so repeated presses chain the expression (`Ctrl+P` applies the editable back expression, shown below the expression, to the latest result to undo a step), `Ctrl+R` save result (as `result = [...]`), or put a line `output path` in the definitions to save the result there on every calculation, `Ctrl+O` open a matrices file, a maze `.zip` (its input) or an image (`I = image path`; size numbers in your expressions are fitted to the opened matrix), `PgUp`/`PgDn` and `Ctrl`+arrows scroll a large result, `F4` focuses the graph (arrows pan, `+`/`-` or `PgUp`/`PgDn` zoom, `0` reset, `F4`/`Esc` back; labels appear when zoomed in enough), `F3` opens the expression help (all operations; the expression box also lists them), `Esc` back. `I = image path.png` loads a photo as grayscale (`builtin:earth` is bundled); large results are shown as before/after pictures. A line `graph NAME` draws that square matrix as a 2D graph beside the result (force-directed layout, or `graph NAME grid W` to place node `i` at column `i % W`; vector results highlight nodes; grid graphs are drawn as mazes (magenta walls, nodes as dots) and show the longest path in green and dead ends in red, with result highlights in yellow; up to 1000 nodes). `M = maze W H [seed]` generates (W, H and seed may be numbers or 1x1 variables like `W = 21`, as may the grid width) a random maze adjacency matrix, and `astar(A, start, goal[, gridWidth])` / `bfs(A, start)` search a graph and return a node vector. A definitions line `include path` loads the matrices of another file; saved files keep the expression in a `# expr:` comment |
| Game of Life | left click spawns the selected pattern, `s` spawns it under the mouse pointer (random spot if outside the grid), `p` cycles patterns, `space` pause, `n` step, `+`/`-` speed, `r` restart |

## Command line (no UI)

The binary is `gtu`. Giving `--input`, `--expression` or `--output` evaluates a matrix expression on the GPU and exits; with no options the interactive UI starts. `gtu -h` shows the help.

```sh
gtu --list-gpus                                       # index: name (backend, type)
gtu -e 'identity(3)'                                    # inline: a value that is not a file is used as text
gtu -i 'A=1 2; 3 5' -e 'A * identity(2)'               # several matrices: separate with \n
gtu --list-examples                                      # bundled algorithms: number: title
gtu --example 2 [-o out.txt]                           # run one (number or title part); --show-example prints its input file
gtu -i a.txt -e s1.txt | gtu -i - -e s2.txt             # piping: output is "result = [..]", usable as next input
gtu -g 1 -i matrices.txt -e expression.txt            # run on GPU 1 (-g also selects the GPU for the UI)
gtu -i matrices.txt -e expression.txt -o result.txt   # write the result to a file
gtu -i matrices.txt -e expression.txt                 # print "result = [..]" to stdout
gtu -i matrices.txt -e expression.txt | gtu -i - -e next.txt   # pipe: use `result` in next.txt
echo "A * B" | gtu -i matrices.txt -e -               # expression from stdin
cat matrices.txt | gtu -e expression.txt              # definitions from stdin
```

If `--expression` is omitted, the `# expr:` line of the input file is used. Errors go to stderr with exit code 1.

## Bundled examples

All 21 examples are embedded in `gtu`: choose **Matrix examples** in the main menu or press `F2` in Matrices. From a checkout:

```sh
cargo run --release -- --list-examples
cargo run --release -- --example 'Maze: A*'
cargo run --release -- --example 'Photo: raycast'
```

With the binary on your `PATH`, `gtu --example NUMBER` or `gtu --example 'unique title fragment'` runs the same examples. `--show-example` prints the selected example's definitions and `# expr:` line instead of evaluating it:

```sh
gtu --example 'Photo: blur' --show-example > blur.txt
gtu -i blur.txt
gtu -i examples/maze/maze.txt -e examples/maze/discover.txt
```

### Markov chains

Definitions: [`examples/markov/chain.txt`](examples/markov/chain.txt), a two-state weather transition matrix and initial distribution.

| Expression file | Demonstrates |
|-----------------|--------------|
| [two_days.txt](examples/markov/two_days.txt) | Weather distribution after two transitions (`s * P * P`). |
| [steady.txt](examples/markov/steady.txt) | Sixteen transitions approaching the steady distribution `[5/6 1/6]`. |

### Graph adjacency and walks

Definitions: [`examples/graph/graph.txt`](examples/graph/graph.txt), a five-node adjacency matrix, vectors and `graph A` visualization.

| Expression file | Demonstrates |
|-----------------|--------------|
| [degrees.txt](examples/graph/degrees.txt) | Node degrees from `A * ones`. |
| [neighbors.txt](examples/graph/neighbors.txt) | Nodes adjacent to node 2 from `A * e2` (zero-based node indices). |
| [walks3.txt](examples/graph/walks3.txt) | Number of length-three walks between each pair of nodes (`A * A * A`). |
| [reach3.txt](examples/graph/reach3.txt) | Reachability from node 0 within three hops, using repeated multiplication by `identity(5) + A`. |

Use `F4` to focus the graph, arrows to pan, `+`/`-` to zoom, and `0` to reset. `F6` opens the symbol/color legend (`L` also works while the graph is focused).

### Mazes, A* and BFS

Definitions: [`examples/maze/maze.txt`](examples/maze/maze.txt), a seeded **21 x 24 perfect maze** (504 cells). `M = maze W H 7` generates its adjacency matrix; `graph M grid W` places each node at its maze cell. Edit `W`, `H` or the seed to generate another maze.

| Expression file | Demonstrates |
|-----------------|--------------|
| [astar.txt](examples/maze/astar.txt) | `astar(M, 0, W * H - 1, W)`: shortest path from the first to the last cell; path entries contain one-based step numbers, other entries are zero. |
| [discover.txt](examples/maze/discover.txt) | `bfs(M, 0)`: discovery from cell 0; each reachable cell contains its hop distance plus one, and unreachable cells are zero. |

In the TUI, maze walls are magenta, the longest path is green, dead ends are red, and result highlights are yellow. `Ctrl+E` exports a ZIP containing the maze drawing, A*/BFS drawings and vectors, adjacency matrix, input and current result (if calculated). Reopen the ZIP with `Ctrl+O` to restore its input. Maze generation supports up to 1024 cells; the graph view draws up to 1000 nodes.

### Lighting

Definitions: [`examples/lighting/scene.txt`](examples/lighting/scene.txt), facet normals and a light direction.

| Expression file | Demonstrates |
|-----------------|--------------|
| [shading.txt](examples/lighting/shading.txt) | Lambert shading from normal/light dot products (`N * L`). |
| [moved_light.txt](examples/lighting/moved_light.txt) | Rotate the light 45 degrees around the y axis before shading (`N * (roty(45) * L)`). |

### Images and raycasting

Definitions: [`examples/lighting/photo.txt`](examples/lighting/photo.txt), the bundled NASA Apollo 17 Earth photo (`I = image builtin:earth`). PNG/JPEG input becomes a grayscale matrix with values in `0..1`, resized to at most 128 pixels per side.

| Expression file | Demonstrates |
|-----------------|--------------|
| [photo_raycast.txt](examples/lighting/photo_raycast.txt) | Orbit the camera 40 degrees around the y axis and resample the photo, with black where rays miss. |
| [photo_enhance.txt](examples/lighting/photo_enhance.txt) | Increase contrast and adjust brightness (`1.6 * I - 0.25`). |
| [photo_blur.txt](examples/lighting/photo_blur.txt) | Separable blur using `smooth(rows(I)) * I * smooth(cols(I))'`. |
| [photo_lift.txt](examples/lighting/photo_lift.txt) | Lift shadows and compress contrast (`0.5 * I + 0.4`). |
| [photo_sharpen.txt](examples/lighting/photo_sharpen.txt) | Sharpen by subtracting a blurred image from `2 * I`. |
| [photo_edges.txt](examples/lighting/photo_edges.txt) | Extract image-minus-blur edges, centered on mid-gray. |
| [photo_invert.txt](examples/lighting/photo_invert.txt) | Negative image (`-1 * I + 1`). |
| [photo_scale_down.txt](examples/lighting/photo_scale_down.txt) | Halve both dimensions with `resize`. |
| [photo_scale_up.txt](examples/lighting/photo_scale_up.txt) | Double both dimensions with `resize`. |
| [photo_flip.txt](examples/lighting/photo_flip.txt) | Mirror left/right (`I * flip(cols(I))`). |
| [photo_rotate.txt](examples/lighting/photo_rotate.txt) | Rotate 90 degrees (`I' * flip(rows(I))`). |

The TUI shows before/after pictures. Load an image example, then use `Ctrl+O` to try your own PNG/JPEG; the current expression is retained and its size literals are fitted to the new input. `Ctrl+N` feeds the latest result back into the forward expression. `Ctrl+P` applies the editable back expression from the example's `# back:` line; when using the unchanged default back expression, it restores the exact picture from before the last forward step. `Ctrl+E` exports the result as a grayscale PNG (values clamped to `0..1`).

The CLI can use the same expression files with your own image definition:

```sh
gtu -i 'I = image ./photo.png' -e examples/lighting/photo_blur.txt -o result.txt
```

CLI output is matrix text; PNG and maze ZIP exports use the TUI. See [`examples/README.md`](examples/README.md) for the folder overview and [`arch-readme.md`](arch-readme.md) for the implementation.

## Tests

```sh
cargo test
cargo test blinker_oscillates_on_gpu   # single test
```

GPU tests pass early when no adapter is available.

## Layout

- `src/main.rs` – CLI/TUI dispatch, event loop and screens (Menu, Mandelbrot, Setup, Run, Gpus, Matrix)
- `src/cli.rs` – batch matrix evaluation, GPU selection and bundled-example commands
- `src/gpu.rs` – shared device, queue and adapter info
- `src/mandelbrot.rs`, `mandelbrot.wgsl` – Mandelbrot compute
- `src/automata.rs`, `automata.wgsl` – Life and Lenia
- `src/matrix.rs`, `matrix.wgsl`, `matrix_ui.rs` – matrix expression parser, GPU ops, screen
- `src/examples.rs` – embedded example registry shared by the CLI and TUI
- `src/search.rs`, `src/graph.rs` – CPU maze/search/export helpers and graph/maze visualization
- `src/automata_ui.rs` – setup form and run screen
