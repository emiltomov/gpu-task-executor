use crate::gpu::Gpu;
use anyhow::Result;
use wgpu::util::DeviceExt;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Params {
    width: u32,
    height: u32,
    max_iter: u32,
    scale: f32,
    center_x: f32,
    center_y: f32,
    _pad: [u32; 2],
}

#[derive(Clone, Copy)]
pub struct View {
    pub center_x: f32,
    pub center_y: f32,
    /// Complex-plane units per pixel.
    pub scale: f32,
    pub max_iter: u32,
}

impl Default for View {
    fn default() -> Self {
        Self { center_x: -0.5, center_y: 0.0, scale: 0.05, max_iter: 200 }
    }
}

pub struct Mandelbrot {
    pipeline: wgpu::ComputePipeline,
}

impl Mandelbrot {
    pub fn new(gpu: &Gpu) -> Self {
        let module = gpu.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("mandelbrot"),
            source: wgpu::ShaderSource::Wgsl(include_str!("mandelbrot.wgsl").into()),
        });
        let pipeline = gpu.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("mandelbrot"),
            layout: None,
            module: &module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
        Self { pipeline }
    }

    /// Returns `width * height` iteration counts (row-major); `max_iter` means inside the set.
    pub fn render(&self, gpu: &Gpu, width: u32, height: u32, view: View) -> Result<Vec<u32>> {
        let bytes = (width as u64) * (height as u64) * 4;
        let out = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("mandelbrot out"),
            size: bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let readback = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("mandelbrot readback"),
            size: bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let params = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("mandelbrot params"),
            contents: bytemuck::bytes_of(&Params {
                width,
                height,
                max_iter: view.max_iter,
                scale: view.scale,
                center_x: view.center_x,
                center_y: view.center_y,
                _pad: [0; 2],
            }),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let bind_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: out.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: params.as_entire_binding() },
            ],
        });

        let mut encoder = gpu.device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(width.div_ceil(16), height.div_ceil(16), 1);
        }
        encoder.copy_buffer_to_buffer(&out, 0, &readback, 0, bytes);
        gpu.queue.submit([encoder.finish()]);

        let (tx, rx) = std::sync::mpsc::channel();
        readback.slice(..).map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        gpu.device.poll(wgpu::PollType::wait_indefinitely())?;
        rx.recv()??;
        let data = {
            let view = readback.slice(..).get_mapped_range()?;
            bytemuck::cast_slice(&view).to_vec()
        };
        readback.unmap();
        Ok(data)
    }
}
