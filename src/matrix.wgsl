struct Params {
    op: u32,
    rows: u32,
    cols: u32,
    inner: u32,
}

@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@group(0) @binding(3) var<uniform> p: Params;

// rows/cols describe the output. op: 0 add, 1 sub, 2 matmul, 3 transpose, 4 scale by b[0].
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if (i >= p.rows * p.cols) { return; }
    let r = i / p.cols;
    let c = i % p.cols;
    var v = 0.0;
    switch p.op {
        case 0u: { v = a[i] + b[i]; }
        case 1u: { v = a[i] - b[i]; }
        case 2u: {
            for (var k = 0u; k < p.inner; k = k + 1u) {
                v = v + a[r * p.inner + k] * b[k * p.cols + c];
            }
        }
        case 3u: { v = a[c * p.rows + r]; }
        default: { v = a[i] * b[0]; }
    }
    out[i] = v;
}
