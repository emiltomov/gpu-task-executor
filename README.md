# gpu-task-executor

A Rust terminal UI (`ratatui`) that runs compute workloads on the GPU via `wgpu` (WGSL shaders).

## Features

- **Mandelbrot** – per-pixel iteration counts computed on the GPU; pan and zoom.
- **Game of Life** – rendered with braille characters (2x4 cells per character).
- **Lenia** – continuous cellular automaton rendered with half blocks (1x2).
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

Keys:

| Screen | Keys |
|--------|------|
| Menu / Setup | `↑`/`k`, `↓`/`j`, `Enter`, `q`/`Esc` |
| Mandelbrot | arrows pan, `+`/`-` zoom, `r` reset, `q`/`Esc` back |
| Game of Life | left click spawns the selected pattern, `s` spawns it at a random spot, `p` cycles patterns, `space` pause, `n` step, `+`/`-` speed, `r` restart |

## Tests

```sh
cargo test
cargo test blinker_oscillates_on_gpu   # single test
```

GPU tests pass early when no adapter is available.

## Layout

- `src/main.rs` – event loop and screens (Menu, Mandelbrot, Setup, Run)
- `src/gpu.rs` – shared device, queue and adapter info
- `src/mandelbrot.rs`, `mandelbrot.wgsl` – Mandelbrot compute
- `src/automata.rs`, `automata.wgsl` – Life and Lenia
- `src/automata_ui.rs` – setup form and run screen
