use anyhow::{Context, Result, anyhow};
use std::time::{Duration, Instant};
use wgpu::util::DeviceExt;

const SHADER: &str = include_str!("shader.wgsl");
const WORKGROUP_SIZE: u32 = 256;
const MAX_DIM: u32 = 65535;

#[derive(Clone, Copy, Debug, clap::ValueEnum)]
pub enum Task {
    Vecadd,
    Saxpy,
    Square,
}

impl Task {
    fn entry_point(self) -> &'static str {
        match self {
            Task::Vecadd => "vecadd",
            Task::Saxpy => "saxpy",
            Task::Square => "square",
        }
    }

    pub fn cpu(self, a: f32, b: f32, alpha: f32) -> f32 {
        match self {
            Task::Vecadd => a + b,
            Task::Saxpy => alpha * a + b,
            Task::Square => a * a,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Params {
    n: u32,
    alpha: f32,
    _pad: [u32; 2],
}

pub struct Gpu {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub adapter_info: wgpu::AdapterInfo,
}

pub struct RunResult {
    pub output: Vec<f32>,
    pub gpu_time: Duration,
}

pub fn list_adapters() -> Vec<wgpu::AdapterInfo> {
    let instance = wgpu::Instance::default();
    pollster::block_on(instance.enumerate_adapters(wgpu::Backends::all()))
        .iter()
        .map(|a| a.get_info())
        .collect()
}

impl Gpu {
    pub fn new() -> Result<Self> {
        let instance = wgpu::Instance::default();
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            ..Default::default()
        }))
        .context("no suitable GPU adapter found")?;
        let adapter_info = adapter.get_info();
        let (device, queue) =
            pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default()))
                .context("failed to create GPU device")?;
        Ok(Self {
            device,
            queue,
            adapter_info,
        })
    }

    pub fn with_adapter(index: usize) -> Result<Self> {
        let instance = wgpu::Instance::default();
        let adapter = pollster::block_on(instance.enumerate_adapters(wgpu::Backends::all()))
            .into_iter()
            .nth(index)
            .context("GPU adapter not found")?;
        let adapter_info = adapter.get_info();
        let (device, queue) =
            pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default()))
                .context("failed to create GPU device")?;
        Ok(Self {
            device,
            queue,
            adapter_info,
        })
    }

    pub fn run(&self, task: Task, a: &[f32], b: &[f32], alpha: f32) -> Result<RunResult> {
        let n = a.len();
        let groups = (n as u32).div_ceil(WORKGROUP_SIZE);
        if groups > MAX_DIM {
            return Err(anyhow!(
                "size {n} exceeds the max of {}",
                MAX_DIM * WORKGROUP_SIZE
            ));
        }
        let bytes = (n * 4) as u64;
        let storage = |label, data: &[f32]| {
            self.device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some(label),
                    contents: bytemuck::cast_slice(data),
                    usage: wgpu::BufferUsages::STORAGE,
                })
        };
        let buf_a = storage("a", a);
        let buf_b = storage("b", b);
        let buf_out = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("out"),
            size: bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let readback = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback"),
            size: bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let params = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("params"),
                contents: bytemuck::bytes_of(&Params {
                    n: n as u32,
                    alpha,
                    _pad: [0; 2],
                }),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        let module = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("shader"),
                source: wgpu::ShaderSource::Wgsl(SHADER.into()),
            });
        let pipeline = self
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(task.entry_point()),
                layout: None,
                module: &module,
                entry_point: Some(task.entry_point()),
                compilation_options: Default::default(),
                cache: None,
            });
        let layout = pipeline.get_bind_group_layout(0);
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buf_a.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: buf_b.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: buf_out.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: params.as_entire_binding(),
                },
            ],
        });

        let mut encoder = self.device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(groups, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&buf_out, 0, &readback, 0, bytes);

        let start = Instant::now();
        self.queue.submit([encoder.finish()]);
        let (tx, rx) = std::sync::mpsc::channel();
        readback.slice(..).map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        self.device.poll(wgpu::PollType::wait_indefinitely())?;
        rx.recv()??;
        let gpu_time = start.elapsed();

        let output = {
            let view = readback.slice(..).get_mapped_range()?;
            bytemuck::cast_slice(&view).to_vec()
        };
        readback.unmap();
        Ok(RunResult { output, gpu_time })
    }
}
