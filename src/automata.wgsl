struct Params {
    width: u32,
    height: u32,
    radius: i32,
    mu: f32,
    sigma: f32,
    dt: f32,
}

@group(0) @binding(0) var<storage, read> src: array<f32>;
@group(0) @binding(1) var<storage, read_write> dst: array<f32>;
@group(0) @binding(2) var<uniform> p: Params;
@group(0) @binding(3) var<storage, read> kernel: array<f32>;

fn index(x: i32, y: i32) -> u32 {
    let w = i32(p.width);
    let h = i32(p.height);
    return u32(((y % h) + h) % h) * p.width + u32(((x % w) + w) % w);
}

@compute @workgroup_size(16, 16)
fn life(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= p.width || id.y >= p.height) { return; }
    _ = kernel[0]; // keeps the shared bind group layout identical for both entry points
    let x = i32(id.x);
    let y = i32(id.y);
    var n = 0u;
    for (var dy = -1; dy <= 1; dy++) {
        for (var dx = -1; dx <= 1; dx++) {
            if ((dx != 0 || dy != 0) && src[index(x + dx, y + dy)] > 0.5) { n++; }
        }
    }
    let alive = src[index(x, y)] > 0.5;
    let next = (alive && (n == 2u || n == 3u)) || (!alive && n == 3u);
    dst[index(x, y)] = select(0.0, 1.0, next);
}

@compute @workgroup_size(16, 16)
fn lenia(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= p.width || id.y >= p.height) { return; }
    let x = i32(id.x);
    let y = i32(id.y);
    let side = 2 * p.radius + 1;
    var u = 0.0;
    for (var dy = -p.radius; dy <= p.radius; dy++) {
        for (var dx = -p.radius; dx <= p.radius; dx++) {
            u += kernel[(dy + p.radius) * side + dx + p.radius] * src[index(x + dx, y + dy)];
        }
    }
    let d = u - p.mu;
    let growth = 2.0 * exp(-(d * d) / (2.0 * p.sigma * p.sigma)) - 1.0;
    dst[index(x, y)] = clamp(src[index(x, y)] + p.dt * growth, 0.0, 1.0);
}
