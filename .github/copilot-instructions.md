# Copilot Instructions

Rust TUI app (`ratatui`) that runs compute workloads on the GPU via `wgpu` (WGSL shaders).

## Commands

- Build/run: `cargo build`, `cargo run` (needs a real terminal); the binary is named `gtu`. `cargo run -- -i m.txt -e e.txt` runs the matrix CLI without UI (`cli.rs`).
- Test: `cargo test`; single test: `cargo test blinker_oscillates_on_gpu`
- GPU tests return early (pass) when no adapter is available.

## Architecture

- `main.rs`: event loop and `Screen` enum (Menu, Mandelbrot, Setup, Run, Gpus, Matrix); the "Select GPU" menu item lists adapters and rebuilds the `Renderer` (Gpu + Mandelbrot). A GPU info header is drawn above every screen; NVML telemetry is sampled at most once per second and falls back to "unavailable".
- `gpuinfo.rs`: NVML `Monitor`/`Telemetry` (NVIDIA only).
- `gpu.rs`: `Gpu` (device, queue, adapter info) created once at startup and shared by all screens.
- `mandelbrot.rs` + `mandelbrot.wgsl`: per-pixel iteration counts, cached by (width, height, view).
- `automata.rs` + `automata.wgsl`: Game of Life and Lenia on ping-pong storage buffers; state stays on the GPU and is read back only for drawing. Both entry points share one bind-group layout, so `life` must reference the `kernel` binding.
- `matrix.rs` + `matrix.wgsl` + `matrix_ui.rs`: matrix expressions (`Matrix`, tokenizer/parser, `Backend` trait implemented by `Gpu`; each operation is one GPU dispatch with readback, so composed expressions chain GPU ops). Tests compare against a CPU backend.
- `automata_ui.rs`: setup form and run screen. Grid size follows the terminal (Life: braille, 2x4 cells per character; Lenia: half blocks, 1x2); a resize restarts the simulation.
- `gpu.rs` `Gpu::run` (vecadd/saxpy/square, uses `shader.wgsl`) is currently unused.

## Conventions

- Do not overengineer; never commit or push.
- After each task, show a commit message (max 10 lines) prefixed with the current branch name.
- Create or update tests for relevant changes.
- If credentials are needed, stop and ask the user.
