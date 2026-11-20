// Portable F32 x F32 -> F32 GEMM for adapters without ShaderF16.
// 8x8 output tile, K tile 16, F32 workgroup storage only.
struct MulMatParams {
    offset_src0: u32,
    offset_src1: u32,
    offset_dst: u32,
    m: u32,
    n: u32,
    k: u32,
    stride_01: u32,
    stride_11: u32,
    stride_02: u32,
    stride_12: u32,
    stride_03: u32,
    stride_13: u32,
    bs02: u32,
    bs03: u32,
    broadcast2: u32,
    broadcast3: u32
};

@group(0) @binding(0) var<storage, read_write> src0: array<f32>;
@group(0) @binding(1) var<storage, read_write> src1: array<f32>;
@group(0) @binding(2) var<storage, read_write> dst: array<f32>;
@group(0) @binding(3) var<uniform> params: MulMatParams;

const WG_SIZE_M = 8u;
const WG_SIZE_N = 8u;
const TILE_K = 16u;
const TILE_A_SIZE = WG_SIZE_M * TILE_K;
const TILE_B_SIZE = WG_SIZE_N * TILE_K;

var<workgroup> tile_a: array<f32, 128>;
var<workgroup> tile_b: array<f32, 128>;

@compute @workgroup_size(64)
fn main(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(num_workgroups) num_wg: vec3<u32>
) {
    let thread_id = local_id.x;
    let local_m = thread_id % WG_SIZE_M;
    let local_n = thread_id / WG_SIZE_M;

    let wg_m_count = (params.m + WG_SIZE_M - 1u) / WG_SIZE_M;
    let wg_n_count = (params.n + WG_SIZE_N - 1u) / WG_SIZE_N;
    let wg_per_matrix = wg_m_count * wg_n_count;
    let wg_linear = wg_id.y * num_wg.x + wg_id.x;
    let batch_idx = wg_linear / wg_per_matrix;

    let total_batches = params.bs02 * params.broadcast2 * params.bs03 * params.broadcast3;
    if (batch_idx >= total_batches) {
        return;
    }

    let wg_in_batch = wg_linear % wg_per_matrix;
    let wg_m = wg_in_batch % wg_m_count;
    let wg_n = wg_in_batch / wg_m_count;
    let output_row_base = wg_m * WG_SIZE_M + local_m;
    let output_col_base = wg_n * WG_SIZE_N + local_n;

    let dst2_stride = params.m * params.n;
    let dst3_stride = dst2_stride * params.bs02 * params.broadcast2;
    let dst3_idx = batch_idx / (params.bs02 * params.broadcast2);
    let src03_idx = dst3_idx / params.broadcast3;
    let src13_idx = dst3_idx;
    let dst2_idx = batch_idx % (params.bs02 * params.broadcast2);
    let src02_idx = dst2_idx / params.broadcast2;
    let src12_idx = dst2_idx;

    let src0_batch_offset = params.offset_src0 + src03_idx * params.stride_03 + src02_idx * params.stride_02;
    let src1_batch_offset = params.offset_src1 + src13_idx * params.stride_13 + src12_idx * params.stride_12;

    var acc = 0.0f;
    for (var k_outer = 0u; k_outer < params.k; k_outer += TILE_K) {
        for (var elem_idx = thread_id; elem_idx < TILE_A_SIZE; elem_idx += 64u) {
            let tile_m = elem_idx / TILE_K;
            let tile_k = elem_idx % TILE_K;
            let global_m = wg_m * WG_SIZE_M + tile_m;
            let global_k = k_outer + tile_k;
            var value = 0.0f;
            if (global_m < params.m && global_k < params.k) {
                value = src0[src0_batch_offset + global_m * params.stride_01 + global_k];
            }
            tile_a[elem_idx] = value;
        }

        for (var elem_idx = thread_id; elem_idx < TILE_B_SIZE; elem_idx += 64u) {
            let tile_n = elem_idx / TILE_K;
            let tile_k = elem_idx % TILE_K;
            let global_n = wg_n * WG_SIZE_N + tile_n;
            let global_k = k_outer + tile_k;
            var value = 0.0f;
            if (global_n < params.n && global_k < params.k) {
                value = src1[src1_batch_offset + global_n * params.stride_11 + global_k];
            }
            tile_b[elem_idx] = value;
        }

        workgroupBarrier();
        for (var k_inner = 0u; k_inner < TILE_K; k_inner++) {
            acc += tile_a[local_m * TILE_K + k_inner] * tile_b[local_n * TILE_K + k_inner];
        }
        workgroupBarrier();
    }

    if (output_row_base < params.m && output_col_base < params.n) {
        let dst_batch_offset = params.offset_dst + dst3_idx * dst3_stride + dst2_idx * dst2_stride;
        dst[dst_batch_offset + output_col_base * params.m + output_row_base] = acc;
    }
}

// AI_CHAT_NO_SHADER_F16_F32_GEMM_SHADER_V1
