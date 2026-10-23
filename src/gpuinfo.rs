use nvml_wrapper::{Nvml, enum_wrappers::device::TemperatureSensor};

pub struct Telemetry {
    pub name: String,
    pub vram_used: u64,
    pub vram_total: u64,
    pub cores: Option<u32>,
    pub gpu_util: u32,
    pub temp_c: Option<u32>,
    pub power_w: Option<f32>,
}

/// Live NVIDIA telemetry via NVML; `Err` holds why it is unavailable.
pub struct Monitor {
    nvml: Result<Nvml, String>,
}

impl Monitor {
    pub fn new() -> Self {
        Self {
            nvml: Nvml::init().map_err(|e| e.to_string()),
        }
    }

    pub fn sample(&self) -> Result<Telemetry, String> {
        let nvml = self.nvml.as_ref().map_err(|e| e.clone())?;
        let dev = nvml.device_by_index(0).map_err(|e| e.to_string())?;
        let mem = dev.memory_info().map_err(|e| e.to_string())?;
        Ok(Telemetry {
            name: dev.name().map_err(|e| e.to_string())?,
            vram_used: mem.used,
            vram_total: mem.total,
            cores: dev.num_cores().ok(),
            gpu_util: dev.utilization_rates().map_err(|e| e.to_string())?.gpu,
            temp_c: dev.temperature(TemperatureSensor::Gpu).ok(),
            power_w: dev.power_usage().ok().map(|mw| mw as f32 / 1000.0),
        })
    }
}
