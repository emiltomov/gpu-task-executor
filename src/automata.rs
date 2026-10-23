use crate::gpu::Gpu;
use anyhow::Result;
use wgpu::util::DeviceExt;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Kind {
    Life,
    Lenia,
}

impl Kind {
    pub fn title(self) -> &'static str {
        match self {
            Kind::Life => "Game of Life",
            Kind::Lenia => "Lenia",
        }
    }

    pub fn patterns(self) -> Vec<&'static str> {
        match self {
            Kind::Life => LIFE_PATTERNS.iter().map(|p| p.0).collect(),
            Kind::Lenia => vec!["Noise blob", "Noise field", "Two blobs"],
        }
    }
}

const LIFE_PATTERNS: [(&str, &str); 8] = [
    ("Random", ""),
    ("Glider", ".O.\n..O\nOOO"),
    ("R-pentomino", ".OO\nOO.\n.O."),
    ("Acorn", ".O.....\n...O...\nOO..OOO"),
    ("Lightweight spaceship", ".O..O\nO....\nO...O\nOOOO."),
    ("Pentadecathlon", "..O....O..\nOO.OOOO.OO\n..O....O.."),
    (
        "Gosper glider gun",
        "........................O...........\n\
         ......................O.O...........\n\
         ............OO......OO............OO\n\
         ...........O...O....OO............OO\n\
         OO........O.....O...OO..............\n\
         OO........O...O.OO....O.O...........\n\
         ..........O.....O.......O...........\n\
         ...........O...O....................\n\
         ............OO......................",
    ),
    ("Blinker", "OOO"),
];

#[derive(Clone, Copy, PartialEq, Debug)]
pub struct LeniaParams {
    pub radius: u32,
    pub mu: f32,
    pub sigma: f32,
    pub dt: f32,
}

impl Default for LeniaParams {
    fn default() -> Self {
        Self {
            radius: 13,
            mu: 0.14,
            sigma: 0.02,
            dt: 0.1,
        }
    }
}

#[derive(Clone, PartialEq, Debug)]
pub struct Settings {
    pub kind: Kind,
    pub pattern: usize,
    /// Fill ratio for random patterns, 0..=1.
    pub density: f32,
    /// Stop after this many generations; 0 runs forever.
    pub generations: u64,
    pub steps_per_frame: u32,
    pub lenia: LeniaParams,
}

impl Settings {
    pub fn new(kind: Kind) -> Self {
        Self {
            kind,
            pattern: 0,
            density: if kind == Kind::Life { 0.3 } else { 0.5 },
            generations: 0,
            steps_per_frame: 1,
            lenia: LeniaParams::default(),
        }
    }
}

/// Builds the initial `width * height` grid (row-major) for the chosen pattern.
pub fn initial_grid(s: &Settings, width: usize, height: usize) -> Vec<f32> {
    let mut grid = vec![0.0; width * height];
    match s.kind {
        Kind::Life if s.pattern == 0 => {
            for c in grid.iter_mut() {
                *c = (rand::random::<f32>() < s.density) as u8 as f32;
            }
        }
        Kind::Life => stamp(&mut grid, width, height, LIFE_PATTERNS[s.pattern].1),
        Kind::Lenia => {
            let r = s.lenia.radius as f32;
            let blobs: &[(f32, f32)] = match s.pattern {
                0 => &[(0.5, 0.5)],
                2 => &[(0.3, 0.5), (0.7, 0.5)],
                _ => &[],
            };
            if blobs.is_empty() {
                for c in grid.iter_mut() {
                    *c = rand::random::<f32>() * s.density;
                }
            }
            for &(fx, fy) in blobs {
                let (cx, cy) = (fx * width as f32, fy * height as f32);
                let reach = r * 1.5;
                for y in 0..height {
                    for x in 0..width {
                        let d = ((x as f32 - cx).powi(2) + (y as f32 - cy).powi(2)).sqrt();
                        if d < reach {
                            grid[y * width + x] = rand::random::<f32>() * s.density;
                        }
                    }
                }
            }
        }
    }
    grid
}

/// Places an ASCII pattern (`O` = alive) at the centre of the grid.
fn stamp(grid: &mut [f32], width: usize, height: usize, pattern: &str) {
    let rows: Vec<&str> = pattern.lines().collect();
    let pw = rows.iter().map(|r| r.len()).max().unwrap_or(0);
    let (ox, oy) = (
        width.saturating_sub(pw) / 2,
        height.saturating_sub(rows.len()) / 2,
    );
    for (y, row) in rows.iter().enumerate() {
        for (x, ch) in row.chars().enumerate() {
            if ch == 'O' && ox + x < width && oy + y < height {
                grid[(oy + y) * width + ox + x] = 1.0;
            }
        }
    }
}

/// Normalised ring kernel `exp(4 - 1 / (r (1 - r)))` over `(2R+1)^2` cells.
fn lenia_kernel(radius: u32) -> Vec<f32> {
    let side = 2 * radius as i32 + 1;
    let mut k = Vec::with_capacity((side * side) as usize);
    for dy in -(radius as i32)..=radius as i32 {
        for dx in -(radius as i32)..=radius as i32 {
            let r = ((dx * dx + dy * dy) as f32).sqrt() / radius as f32;
            k.push(if r > 0.0 && r < 1.0 {
                (4.0 - 1.0 / (r * (1.0 - r))).exp()
            } else {
                0.0
            });
        }
    }
    let sum: f32 = k.iter().sum();
    k.iter_mut().for_each(|v| *v /= sum);
    k
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Params {
    width: u32,
    height: u32,
    radius: i32,
    mu: f32,
    sigma: f32,
    dt: f32,
    _pad: [u32; 2],
}

/// A cellular automaton that lives entirely on the GPU (ping-pong buffers).
pub struct Automata {
    pub width: u32,
    pub height: u32,
    pub generation: u64,
    pipeline: wgpu::ComputePipeline,
    cells: [wgpu::Buffer; 2],
    bind_groups: [wgpu::BindGroup; 2],
    readback: wgpu::Buffer,
    current: usize,
}

impl Automata {
    pub fn new(gpu: &Gpu, s: &Settings, width: u32, height: u32) -> Self {
        let init = initial_grid(s, width as usize, height as usize);
        let bytes = init.len() as u64 * 4;
        let dev = &gpu.device;

        let make_cells = |label: &str, contents: &[u8]| {
            dev.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
            })
        };
        let cells = [
            make_cells("cells a", bytemuck::cast_slice(&init)),
            make_cells("cells b", bytemuck::cast_slice(&init)),
        ];
        let kernel = match s.kind {
            Kind::Life => vec![0.0; 1],
            Kind::Lenia => lenia_kernel(s.lenia.radius),
        };
        let kernel_buf = dev.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("kernel"),
            contents: bytemuck::cast_slice(&kernel),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let params = dev.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("automata params"),
            contents: bytemuck::bytes_of(&Params {
                width,
                height,
                radius: s.lenia.radius as i32,
                mu: s.lenia.mu,
                sigma: s.lenia.sigma,
                dt: s.lenia.dt,
                _pad: [0; 2],
            }),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let readback = dev.create_buffer(&wgpu::BufferDescriptor {
            label: Some("automata readback"),
            size: bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let module = dev.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("automata"),
            source: wgpu::ShaderSource::Wgsl(include_str!("automata.wgsl").into()),
        });
        let entry = match s.kind {
            Kind::Life => "life",
            Kind::Lenia => "lenia",
        };
        let pipeline = dev.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(entry),
            layout: None,
            module: &module,
            entry_point: Some(entry),
            compilation_options: Default::default(),
            cache: None,
        });
        let layout = pipeline.get_bind_group_layout(0);
        let bind = |src: usize| {
            dev.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: cells[src].as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: cells[1 - src].as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: params.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: kernel_buf.as_entire_binding(),
                    },
                ],
            })
        };
        let bind_groups = [bind(0), bind(1)];

        Self {
            width,
            height,
            generation: 0,
            pipeline,
            cells,
            bind_groups,
            readback,
            current: 0,
        }
    }

    pub fn step(&mut self, gpu: &Gpu, n: u32) {
        let mut encoder = gpu.device.create_command_encoder(&Default::default());
        for _ in 0..n {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &self.bind_groups[self.current], &[]);
            pass.dispatch_workgroups(self.width.div_ceil(16), self.height.div_ceil(16), 1);
            drop(pass);
            self.current = 1 - self.current;
            self.generation += 1;
        }
        gpu.queue.submit([encoder.finish()]);
    }

    pub fn read(&self, gpu: &Gpu) -> Result<Vec<f32>> {
        let bytes = self.width as u64 * self.height as u64 * 4;
        let mut encoder = gpu.device.create_command_encoder(&Default::default());
        encoder.copy_buffer_to_buffer(&self.cells[self.current], 0, &self.readback, 0, bytes);
        gpu.queue.submit([encoder.finish()]);

        let (tx, rx) = std::sync::mpsc::channel();
        self.readback
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |r| {
                let _ = tx.send(r);
            });
        gpu.device.poll(wgpu::PollType::wait_indefinitely())?;
        rx.recv()??;
        let data = {
            let view = self.readback.slice(..).get_mapped_range()?;
            bytemuck::cast_slice(&view).to_vec()
        };
        self.readback.unmap();
        Ok(data)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stamp_centres_pattern() {
        let mut g = vec![0.0; 7 * 7];
        stamp(&mut g, 7, 7, "OOO");
        assert_eq!(g.iter().filter(|&&c| c == 1.0).count(), 3);
        assert_eq!(&g[3 * 7 + 2..3 * 7 + 5], &[1.0, 1.0, 1.0]);
    }

    #[test]
    fn kernel_is_normalised() {
        let k = lenia_kernel(7);
        assert_eq!(k.len(), 15 * 15);
        assert!((k.iter().sum::<f32>() - 1.0).abs() < 1e-4);
    }

    #[test]
    fn blinker_oscillates_on_gpu() {
        let Ok(gpu) = Gpu::new() else { return };
        let mut s = Settings::new(Kind::Life);
        s.pattern = LIFE_PATTERNS.iter().position(|p| p.0 == "Blinker").unwrap();
        let mut a = Automata::new(&gpu, &s, 9, 9);
        let start = a.read(&gpu).unwrap();
        a.step(&gpu, 1);
        let mid = a.read(&gpu).unwrap();
        a.step(&gpu, 1);
        assert_ne!(start, mid);
        assert_eq!(start, a.read(&gpu).unwrap());
        assert_eq!(a.generation, 2);
    }

    #[test]
    fn lenia_stays_in_unit_range() {
        let Ok(gpu) = Gpu::new() else { return };
        let s = Settings::new(Kind::Lenia);
        let mut a = Automata::new(&gpu, &s, 64, 64);
        a.step(&gpu, 20);
        assert!(
            a.read(&gpu)
                .unwrap()
                .iter()
                .all(|v| (0.0..=1.0).contains(v))
        );
    }
}
