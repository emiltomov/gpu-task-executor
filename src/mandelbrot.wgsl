struct Params {
    width: u32,
    height: u32,
    max_iter: u32,
    scale: f32,
    center_x: f32,
    center_y: f32,
}

@group(0) @binding(0) var<storage, read_write> out: array<u32>;
@group(0) @binding(1) var<uniform> p: Params;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= p.width || id.y >= p.height) { return; }
    let c = vec2<f32>(
        p.center_x + (f32(id.x) - f32(p.width) * 0.5) * p.scale,
        p.center_y + (f32(id.y) - f32(p.height) * 0.5) * p.scale,
    );
    var z = vec2<f32>(0.0, 0.0);
    var i = 0u;
    loop {
        if (i >= p.max_iter || dot(z, z) > 4.0) { break; }
        z = vec2<f32>(z.x * z.x - z.y * z.y + c.x, 2.0 * z.x * z.y + c.y);
        i = i + 1u;
    }
    out[id.y * p.width + id.x] = i;
}
