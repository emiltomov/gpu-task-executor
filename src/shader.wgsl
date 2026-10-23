struct Params { n: u32, alpha: f32 }

@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(256)
fn vecadd(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x < params.n) { out[id.x] = a[id.x] + b[id.x]; }
}

@compute @workgroup_size(256)
fn saxpy(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x < params.n) { out[id.x] = params.alpha * a[id.x] + b[id.x]; }
}

@compute @workgroup_size(256)
fn square(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x < params.n) { out[id.x] = a[id.x] * a[id.x]; }
}
