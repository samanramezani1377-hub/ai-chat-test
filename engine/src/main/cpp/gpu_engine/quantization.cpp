#include "quantization.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <utility>
#include <string>
#include <vector>

namespace ai_chat::gpu_engine {
namespace {




constexpr char kQ5_0DequantizeWgsl[] = R"wgsl(
@group(0) @binding(0) var<storage, read> encoded: array<u32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
const ELEMENT_COUNT: u32 = 32u;
fn read_u8(byte_offset: u32) -> u32 {
    let word = encoded[byte_offset >> 2u];
    let shift = (byte_offset & 3u) * 8u;
    return (word >> shift) & 255u;
}
fn read_u16(byte_offset: u32) -> u32 {
    let word = encoded[byte_offset >> 2u];
    let shift = (byte_offset & 3u) * 8u;
    return (word >> shift) & 65535u;
}
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= ELEMENT_COUNT) { return; }
    let block_index = gid.x / 32u;
    let within = gid.x % 32u;
    let offset = block_index * 22u;
    let scale = unpack2x16float(vec2<u32>(read_u16(offset), 0u)).x;
    let high_bits = read_u8(offset + 2u) | (read_u8(offset + 3u) << 8u) |
                    (read_u8(offset + 4u) << 16u) | (read_u8(offset + 5u) << 24u);
    let packed = read_u8(offset + 6u + (within % 16u));
    let nibble = select(packed & 15u, packed >> 4u, within >= 16u);
    let high = (high_bits >> within) & 1u;
    output[gid.x] = scale * f32(i32(nibble | (high << 4u)) - 16);
}
)wgsl";

constexpr char kQ5_1DequantizeWgsl[] = R"wgsl(
@group(0) @binding(0) var<storage, read> encoded: array<u32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
const ELEMENT_COUNT: u32 = 32u;
fn read_u8(byte_offset: u32) -> u32 {
    let word = encoded[byte_offset >> 2u];
    let shift = (byte_offset & 3u) * 8u;
    return (word >> shift) & 255u;
}
fn read_u16(byte_offset: u32) -> u32 {
    let word = encoded[byte_offset >> 2u];
    let shift = (byte_offset & 3u) * 8u;
    return (word >> shift) & 65535u;
}
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= ELEMENT_COUNT) { return; }
    let block_index = gid.x / 32u;
    let within = gid.x % 32u;
    let offset = block_index * 24u;
    let scale = unpack2x16float(vec2<u32>(read_u16(offset), 0u)).x;
    let minimum = unpack2x16float(vec2<u32>(read_u16(offset + 2u), 0u)).x;
    let high_bits = read_u8(offset + 4u) | (read_u8(offset + 5u) << 8u) |
                    (read_u8(offset + 6u) << 16u) | (read_u8(offset + 7u) << 24u);
    let packed = read_u8(offset + 8u + (within % 16u));
    let nibble = select(packed & 15u, packed >> 4u, within >= 16u);
    let high = (high_bits >> within) & 1u;
    output[gid.x] = scale * f32(nibble | (high << 4u)) + minimum;
}
)wgsl";



constexpr char kQ4_0DequantizeGlsl[] = R"glsl(#version 310 es
precision highp float;
precision highp int;
layout(local_size_x = 64) in;
layout(std430, binding = 0) readonly buffer Encoded { uint encoded[]; };
layout(std430, binding = 1) writeonly buffer Output { float output_values[]; };
const uint ELEMENT_COUNT = 32u;
uint read_u8(uint byte_offset) {
    uint word = encoded[byte_offset >> 2u];
    return (word >> ((byte_offset & 3u) * 8u)) & 255u;
}
uint read_u16(uint byte_offset) {
    uint word = encoded[byte_offset >> 2u];
    return (word >> ((byte_offset & 3u) * 8u)) & 65535u;
}
void main() {
    uint index = gl_GlobalInvocationID.x;
    if (index >= ELEMENT_COUNT) return;
    uint within = index % 32u;
    uint offset = (index / 32u) * 18u;
    float scale = unpackHalf2x16(read_u16(offset)).x;
    uint packed = read_u8(offset + 2u + (within % 16u));
    uint nibble = (within >= 16u) ? (packed >> 4u) : (packed & 15u);
    output_values[index] = scale * float(int(nibble) - 8);
}
)glsl";

constexpr char kQ4_1DequantizeGlsl[] = R"glsl(#version 310 es
precision highp float;
precision highp int;
layout(local_size_x = 64) in;
layout(std430, binding = 0) readonly buffer Encoded { uint encoded[]; };
layout(std430, binding = 1) writeonly buffer Output { float output_values[]; };
const uint ELEMENT_COUNT = 32u;
uint read_u8(uint byte_offset) {
    uint word = encoded[byte_offset >> 2u];
    return (word >> ((byte_offset & 3u) * 8u)) & 255u;
}
uint read_u16(uint byte_offset) {
    uint word = encoded[byte_offset >> 2u];
    return (word >> ((byte_offset & 3u) * 8u)) & 65535u;
}
void main() {
    uint index = gl_GlobalInvocationID.x;
    if (index >= ELEMENT_COUNT) return;
    uint within = index % 32u;
    uint offset = (index / 32u) * 20u;
    float scale = unpackHalf2x16(read_u16(offset)).x;
    float minimum = unpackHalf2x16(read_u16(offset + 2u)).x;
    uint packed = read_u8(offset + 4u + (within % 16u));
    uint nibble = (within >= 16u) ? (packed >> 4u) : (packed & 15u);
    output_values[index] = scale * float(nibble) + minimum;
}
)glsl";

constexpr char kQ8_0DequantizeGlsl[] = R"glsl(#version 310 es
precision highp float;
precision highp int;
layout(local_size_x = 64) in;
layout(std430, binding = 0) readonly buffer Encoded { uint encoded[]; };
layout(std430, binding = 1) writeonly buffer Output { float output_values[]; };
const uint ELEMENT_COUNT = 32u;
uint read_u8(uint byte_offset) {
    uint word = encoded[byte_offset >> 2u];
    return (word >> ((byte_offset & 3u) * 8u)) & 255u;
}
uint read_u16(uint byte_offset) {
    uint word = encoded[byte_offset >> 2u];
    return (word >> ((byte_offset & 3u) * 8u)) & 65535u;
}
void main() {
    uint index = gl_GlobalInvocationID.x;
    if (index >= ELEMENT_COUNT) return;
    uint offset = (index / 32u) * 34u;
    float scale = unpackHalf2x16(read_u16(offset)).x;
    int quantized = int(read_u8(offset + 2u + (index % 32u)));
    if (quantized >= 128) quantized -= 256;
    output_values[index] = scale * float(quantized);
}
)glsl";

constexpr char kQ5_0DequantizeGlsl[] = R"glsl(#version 310 es
precision highp float;
precision highp int;
layout(local_size_x = 64) in;
layout(std430, binding = 0) readonly buffer Encoded { uint encoded[]; };
layout(std430, binding = 1) writeonly buffer Output { float output_values[]; };
const uint ELEMENT_COUNT = 32u;
uint read_u8(uint byte_offset) {
    uint word = encoded[byte_offset >> 2u];
    return (word >> ((byte_offset & 3u) * 8u)) & 255u;
}
uint read_u16(uint byte_offset) {
    uint word = encoded[byte_offset >> 2u];
    return (word >> ((byte_offset & 3u) * 8u)) & 65535u;
}
void main() {
    uint index = gl_GlobalInvocationID.x;
    if (index >= ELEMENT_COUNT) return;
    uint block_index = index / 32u;
    uint within = index % 32u;
    uint offset = block_index * 22u;
    float scale = unpackHalf2x16(read_u16(offset)).x;
    uint high_bits = read_u8(offset + 2u) | (read_u8(offset + 3u) << 8u) |
                     (read_u8(offset + 4u) << 16u) | (read_u8(offset + 5u) << 24u);
    uint packed = read_u8(offset + 6u + (within % 16u));
    uint nibble = (within >= 16u) ? (packed >> 4u) : (packed & 15u);
    uint high = (high_bits >> within) & 1u;
    output_values[index] = scale * float(int(nibble | (high << 4u)) - 16);
}
)glsl";

constexpr char kQ5_1DequantizeGlsl[] = R"glsl(#version 310 es
precision highp float;
precision highp int;
layout(local_size_x = 64) in;
layout(std430, binding = 0) readonly buffer Encoded { uint encoded[]; };
layout(std430, binding = 1) writeonly buffer Output { float output_values[]; };
const uint ELEMENT_COUNT = 32u;
uint read_u8(uint byte_offset) {
    uint word = encoded[byte_offset >> 2u];
    return (word >> ((byte_offset & 3u) * 8u)) & 255u;
}
uint read_u16(uint byte_offset) {
    uint word = encoded[byte_offset >> 2u];
    return (word >> ((byte_offset & 3u) * 8u)) & 65535u;
}
void main() {
    uint index = gl_GlobalInvocationID.x;
    if (index >= ELEMENT_COUNT) return;
    uint block_index = index / 32u;
    uint within = index % 32u;
    uint offset = block_index * 24u;
    float scale = unpackHalf2x16(read_u16(offset)).x;
    float minimum = unpackHalf2x16(read_u16(offset + 2u)).x;
    uint high_bits = read_u8(offset + 4u) | (read_u8(offset + 5u) << 8u) |
                     (read_u8(offset + 6u) << 16u) | (read_u8(offset + 7u) << 24u);
    uint packed = read_u8(offset + 8u + (within % 16u));
    uint nibble = (within >= 16u) ? (packed >> 4u) : (packed & 15u);
    uint high = (high_bits >> within) & 1u;
    output_values[index] = scale * float(nibble | (high << 4u)) + minimum;
}
)glsl";

constexpr char kQ4_1DequantizeWgsl[] = R"wgsl(
@group(0) @binding(0) var<storage, read> encoded: array<u32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
const ELEMENT_COUNT: u32 = 32u;
fn read_u8(byte_offset: u32) -> u32 {
    let word = encoded[byte_offset >> 2u];
    let shift = (byte_offset & 3u) * 8u;
    return (word >> shift) & 255u;
}
fn read_u16(byte_offset: u32) -> u32 {
    let word = encoded[byte_offset >> 2u];
    let shift = (byte_offset & 3u) * 8u;
    return (word >> shift) & 65535u;
}
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= ELEMENT_COUNT) { return; }
    let block_index = gid.x / 32u;
    let within_block = gid.x % 32u;
    let block_offset = block_index * 20u;
    let scale = unpack2x16float(vec2<u32>(read_u16(block_offset), 0u)).x;
    let minimum = unpack2x16float(vec2<u32>(read_u16(block_offset + 2u), 0u)).x;
    let packed = read_u8(block_offset + 4u + (within_block % 16u));
    let nibble = select(packed & 15u, packed >> 4u, within_block >= 16u);
    output[gid.x] = scale * f32(nibble) + minimum;
}
)wgsl";

constexpr char kQ4_0DequantizeWgsl[] = R"wgsl(
@group(0) @binding(0) var<storage, read> encoded: array<u32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
const ELEMENT_COUNT: u32 = 32u;
fn read_u8(byte_offset: u32) -> u32 {
    let word = encoded[byte_offset >> 2u];
    let shift = (byte_offset & 3u) * 8u;
    return (word >> shift) & 255u;
}
fn read_u16(byte_offset: u32) -> u32 {
    let word = encoded[byte_offset >> 2u];
    let shift = (byte_offset & 3u) * 8u;
    return (word >> shift) & 65535u;
}
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= ELEMENT_COUNT) { return; }
    let block_index = gid.x / 32u;
    let within_block = gid.x % 32u;
    let block_offset = block_index * 18u;
    let scale_bits = read_u16(block_offset);
    let scale = unpack2x16float(vec2<u32>(scale_bits, 0u)).x;
    let packed = read_u8(block_offset + 2u + (within_block % 16u));
    let nibble = select(packed & 15u, packed >> 4u, within_block >= 16u);
    output[gid.x] = scale * f32(i32(nibble) - 8);
}
)wgsl";



constexpr char kQ5_0MatMulWgsl[] = R"wgsl(
@group(0) @binding(0) var<storage, read> activations: array<f32>;
@group(0) @binding(1) var<storage, read> weights: array<u32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
const ROWS: u32 = 1u;
const OUTPUT_CHANNELS: u32 = 1u;
const INNER: u32 = 32u;
const WEIGHT_ROW_BYTES: u32 = 22u;
fn read_u8(byte_offset: u32) -> u32 {
    let word = weights[byte_offset >> 2u];
    return (word >> ((byte_offset & 3u) * 8u)) & 255u;
}
fn read_u16(byte_offset: u32) -> u32 {
    let word = weights[byte_offset >> 2u];
    return (word >> ((byte_offset & 3u) * 8u)) & 65535u;
}
@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row = gid.y;
    let channel = gid.x;
    if (row >= ROWS || channel >= OUTPUT_CHANNELS) { return; }
    var sum = 0.0;
    for (var k = 0u; k < INNER; k = k + 1u) {
        let block = k / 32u;
        let within = k % 32u;
        let offset = channel * WEIGHT_ROW_BYTES + block * 22u;
        let scale = unpack2x16float(vec2<u32>(read_u16(offset), 0u)).x;
        let high_bits = read_u8(offset + 2u) | (read_u8(offset + 3u) << 8u) |
                        (read_u8(offset + 4u) << 16u) | (read_u8(offset + 5u) << 24u);
        let packed = read_u8(offset + 6u + (within % 16u));
        let nibble = select(packed & 15u, packed >> 4u, within >= 16u);
        let high = (high_bits >> within) & 1u;
        let quantized = i32(nibble | (high << 4u)) - 16;
        sum = sum + activations[row * INNER + k] * scale * f32(quantized);
    }
    output[row * OUTPUT_CHANNELS + channel] = sum;
}
)wgsl";

constexpr char kQ5_0MatMulGlsl[] = R"glsl(#version 310 es
precision highp float;
precision highp int;
layout(local_size_x = 8, local_size_y = 8) in;
layout(std430, binding = 0) readonly buffer Activations { float activations[]; };
layout(std430, binding = 1) readonly buffer Weights { uint weights[]; };
layout(std430, binding = 2) writeonly buffer Output { float output_values[]; };
const uint ROWS = 1u;
const uint OUTPUT_CHANNELS = 1u;
const uint INNER = 32u;
const uint WEIGHT_ROW_BYTES = 22u;
uint read_u8(uint byte_offset) {
    uint word = weights[byte_offset >> 2u];
    return (word >> ((byte_offset & 3u) * 8u)) & 255u;
}
uint read_u16(uint byte_offset) {
    uint word = weights[byte_offset >> 2u];
    return (word >> ((byte_offset & 3u) * 8u)) & 65535u;
}
void main() {
    uint row = gl_GlobalInvocationID.y;
    uint channel = gl_GlobalInvocationID.x;
    if (row >= ROWS || channel >= OUTPUT_CHANNELS) return;
    float sum = 0.0;
    for (uint k = 0u; k < INNER; ++k) {
        uint block = k / 32u;
        uint within = k % 32u;
        uint offset = channel * WEIGHT_ROW_BYTES + block * 22u;
        float scale = unpackHalf2x16(read_u16(offset)).x;
        uint high_bits = read_u8(offset + 2u) | (read_u8(offset + 3u) << 8u) |
                         (read_u8(offset + 4u) << 16u) | (read_u8(offset + 5u) << 24u);
        uint packed = read_u8(offset + 6u + (within % 16u));
        uint nibble = within < 16u ? (packed & 15u) : (packed >> 4u);
        uint high = (high_bits >> within) & 1u;
        int quantized = int(nibble | (high << 4u)) - 16;
        sum += activations[row * INNER + k] * scale * float(quantized);
    }
    output_values[row * OUTPUT_CHANNELS + channel] = sum;
}
)glsl";


constexpr char kQ5_1MatMulWgsl[] = R"wgsl(
@group(0) @binding(0) var<storage, read> activations: array<f32>;
@group(0) @binding(1) var<storage, read> weights: array<u32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
const ROWS: u32 = 1u;
const OUTPUT_CHANNELS: u32 = 1u;
const INNER: u32 = 32u;
const WEIGHT_ROW_BYTES: u32 = 24u;
fn read_u8(byte_offset: u32) -> u32 {
    let word = weights[byte_offset >> 2u];
    return (word >> ((byte_offset & 3u) * 8u)) & 255u;
}
fn read_u16(byte_offset: u32) -> u32 {
    let word = weights[byte_offset >> 2u];
    return (word >> ((byte_offset & 3u) * 8u)) & 65535u;
}
@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row = gid.y;
    let channel = gid.x;
    if (row >= ROWS || channel >= OUTPUT_CHANNELS) { return; }
    var sum = 0.0;
    for (var k = 0u; k < INNER; k = k + 1u) {
        let block = k / 32u;
        let within = k % 32u;
        let offset = channel * WEIGHT_ROW_BYTES + block * 24u;
        let scale = unpack2x16float(vec2<u32>(read_u16(offset), 0u)).x;
        let minimum = unpack2x16float(vec2<u32>(read_u16(offset + 2u), 0u)).x;
        let high_bits = read_u8(offset + 4u) | (read_u8(offset + 5u) << 8u) |
                        (read_u8(offset + 6u) << 16u) | (read_u8(offset + 7u) << 24u);
        let packed = read_u8(offset + 8u + (within % 16u));
        let nibble = select(packed & 15u, packed >> 4u, within >= 16u);
        let high = (high_bits >> within) & 1u;
        let quantized = nibble | (high << 4u);
        sum = sum + activations[row * INNER + k] * (scale * f32(quantized) + minimum);
    }
    output[row * OUTPUT_CHANNELS + channel] = sum;
}
)wgsl";

constexpr char kQ5_1MatMulGlsl[] = R"glsl(#version 310 es
precision highp float;
precision highp int;
layout(local_size_x = 8, local_size_y = 8) in;
layout(std430, binding = 0) readonly buffer Activations { float activations[]; };
layout(std430, binding = 1) readonly buffer Weights { uint weights[]; };
layout(std430, binding = 2) writeonly buffer Output { float output_values[]; };
const uint ROWS = 1u;
const uint OUTPUT_CHANNELS = 1u;
const uint INNER = 32u;
const uint WEIGHT_ROW_BYTES = 24u;
uint read_u8(uint byte_offset) {
    uint word = weights[byte_offset >> 2u];
    return (word >> ((byte_offset & 3u) * 8u)) & 255u;
}
uint read_u16(uint byte_offset) {
    uint word = weights[byte_offset >> 2u];
    return (word >> ((byte_offset & 3u) * 8u)) & 65535u;
}
void main() {
    uint row = gl_GlobalInvocationID.y;
    uint channel = gl_GlobalInvocationID.x;
    if (row >= ROWS || channel >= OUTPUT_CHANNELS) return;
    float sum = 0.0;
    for (uint k = 0u; k < INNER; ++k) {
        uint block = k / 32u;
        uint within = k % 32u;
        uint offset = channel * WEIGHT_ROW_BYTES + block * 24u;
        float scale = unpackHalf2x16(read_u16(offset)).x;
        float minimum = unpackHalf2x16(read_u16(offset + 2u)).x;
        uint high_bits = read_u8(offset + 4u) | (read_u8(offset + 5u) << 8u) |
                         (read_u8(offset + 6u) << 16u) | (read_u8(offset + 7u) << 24u);
        uint packed = read_u8(offset + 8u + (within % 16u));
        uint nibble = within < 16u ? (packed & 15u) : (packed >> 4u);
        uint high = (high_bits >> within) & 1u;
        uint quantized = nibble | (high << 4u);
        sum += activations[row * INNER + k] * (scale * float(quantized) + minimum);
    }
    output_values[row * OUTPUT_CHANNELS + channel] = sum;
}
)glsl";


constexpr char kF16MatMulWgsl[] = R"wgsl(
@group(0) @binding(0) var<storage, read> activations: array<f32>;
@group(0) @binding(1) var<storage, read> weights: array<u32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
const ROWS: u32 = 1u;
const OUTPUT_CHANNELS: u32 = 1u;
const INNER: u32 = 1u;
const WEIGHT_ROW_BYTES: u32 = 2u;
fn read_u16(byte_offset: u32) -> u32 {
    let word = weights[byte_offset >> 2u];
    return (word >> ((byte_offset & 3u) * 8u)) & 65535u;
}
@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row = gid.y;
    let channel = gid.x;
    if (row >= ROWS || channel >= OUTPUT_CHANNELS) { return; }
    var sum = 0.0;
    for (var k = 0u; k < INNER; k = k + 1u) {
        let offset = channel * WEIGHT_ROW_BYTES + k * 2u;
        let weight = unpack2x16float(vec2<u32>(read_u16(offset), 0u)).x;
        sum = sum + activations[row * INNER + k] * weight;
    }
    output[row * OUTPUT_CHANNELS + channel] = sum;
}
)wgsl";

constexpr char kF16MatMulGlsl[] = R"glsl(#version 310 es
precision highp float;
precision highp int;
layout(local_size_x = 8, local_size_y = 8) in;
layout(std430, binding = 0) readonly buffer Activations { float activations[]; };
layout(std430, binding = 1) readonly buffer Weights { uint weights[]; };
layout(std430, binding = 2) writeonly buffer Output { float output_values[]; };
const uint ROWS = 1u;
const uint OUTPUT_CHANNELS = 1u;
const uint INNER = 1u;
const uint WEIGHT_ROW_BYTES = 2u;
uint read_u16(uint byte_offset) {
    uint word = weights[byte_offset >> 2u];
    return (word >> ((byte_offset & 3u) * 8u)) & 65535u;
}
void main() {
    uint row = gl_GlobalInvocationID.y;
    uint channel = gl_GlobalInvocationID.x;
    if (row >= ROWS || channel >= OUTPUT_CHANNELS) return;
    float sum = 0.0;
    for (uint k = 0u; k < INNER; ++k) {
        uint offset = channel * WEIGHT_ROW_BYTES + k * 2u;
        float weight = unpackHalf2x16(read_u16(offset)).x;
        sum += activations[row * INNER + k] * weight;
    }
    output_values[row * OUTPUT_CHANNELS + channel] = sum;
}
)glsl";

constexpr char kBF16MatMulWgsl[] = R"wgsl(
@group(0) @binding(0) var<storage, read> activations: array<f32>;
@group(0) @binding(1) var<storage, read> weights: array<u32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
const ROWS: u32 = 1u;
const OUTPUT_CHANNELS: u32 = 1u;
const INNER: u32 = 1u;
const WEIGHT_ROW_BYTES: u32 = 2u;
fn read_u16(byte_offset: u32) -> u32 {
    let word = weights[byte_offset >> 2u];
    return (word >> ((byte_offset & 3u) * 8u)) & 65535u;
}
@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row = gid.y;
    let channel = gid.x;
    if (row >= ROWS || channel >= OUTPUT_CHANNELS) { return; }
    var sum = 0.0;
    for (var k = 0u; k < INNER; k = k + 1u) {
        let offset = channel * WEIGHT_ROW_BYTES + k * 2u;
        let weight = bitcast<f32>(read_u16(offset) << 16u);
        sum = sum + activations[row * INNER + k] * weight;
    }
    output[row * OUTPUT_CHANNELS + channel] = sum;
}
)wgsl";

constexpr char kBF16MatMulGlsl[] = R"glsl(#version 310 es
precision highp float;
precision highp int;
layout(local_size_x = 8, local_size_y = 8) in;
layout(std430, binding = 0) readonly buffer Activations { float activations[]; };
layout(std430, binding = 1) readonly buffer Weights { uint weights[]; };
layout(std430, binding = 2) writeonly buffer Output { float output_values[]; };
const uint ROWS = 1u;
const uint OUTPUT_CHANNELS = 1u;
const uint INNER = 1u;
const uint WEIGHT_ROW_BYTES = 2u;
uint read_u16(uint byte_offset) {
    uint word = weights[byte_offset >> 2u];
    return (word >> ((byte_offset & 3u) * 8u)) & 65535u;
}
void main() {
    uint row = gl_GlobalInvocationID.y;
    uint channel = gl_GlobalInvocationID.x;
    if (row >= ROWS || channel >= OUTPUT_CHANNELS) return;
    float sum = 0.0;
    for (uint k = 0u; k < INNER; ++k) {
        uint offset = channel * WEIGHT_ROW_BYTES + k * 2u;
        float weight = uintBitsToFloat(read_u16(offset) << 16u);
        sum += activations[row * INNER + k] * weight;
    }
    output_values[row * OUTPUT_CHANNELS + channel] = sum;
}
)glsl";

constexpr char kF16EmbeddingWgsl[] = R"wgsl(
@group(0) @binding(0) var<storage, read> weights: array<u32>;
@group(0) @binding(1) var<storage, read> token_ids: array<u32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
const VOCABULARY_SIZE: u32 = 1u;
const EMBEDDING_DIMENSION: u32 = 1u;
const TOKEN_COUNT: u32 = 1u;
const TOTAL_ELEMENTS: u32 = 1u;
const WEIGHT_ROW_BYTES: u32 = 2u;
fn read_u16(byte_offset: u32) -> u32 {
    let word = weights[byte_offset >> 2u];
    return (word >> ((byte_offset & 3u) * 8u)) & 65535u;
}
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= TOTAL_ELEMENTS) { return; }
    let token_row = gid.x / EMBEDDING_DIMENSION;
    let dimension = gid.x % EMBEDDING_DIMENSION;
    let token_id = token_ids[token_row];
    if (token_id >= VOCABULARY_SIZE) { return; }
    let offset = token_id * WEIGHT_ROW_BYTES + dimension * 2u;
    output[gid.x] = unpack2x16float(vec2<u32>(read_u16(offset), 0u)).x;
}
)wgsl";

constexpr char kF16EmbeddingGlsl[] = R"glsl(#version 310 es
precision highp float;
precision highp int;
layout(local_size_x = 64) in;
layout(std430, binding = 0) readonly buffer Weights { uint weights[]; };
layout(std430, binding = 1) readonly buffer TokenIds { uint token_ids[]; };
layout(std430, binding = 2) writeonly buffer Output { float output_values[]; };
const uint VOCABULARY_SIZE = 1u;
const uint EMBEDDING_DIMENSION = 1u;
const uint TOKEN_COUNT = 1u;
const uint TOTAL_ELEMENTS = 1u;
const uint WEIGHT_ROW_BYTES = 2u;
uint read_u16(uint byte_offset) {
    uint word = weights[byte_offset >> 2u];
    return (word >> ((byte_offset & 3u) * 8u)) & 65535u;
}
void main() {
    uint index = gl_GlobalInvocationID.x;
    if (index >= TOTAL_ELEMENTS) return;
    uint token_row = index / EMBEDDING_DIMENSION;
    uint dimension = index % EMBEDDING_DIMENSION;
    uint token_id = token_ids[token_row];
    if (token_id >= VOCABULARY_SIZE) return;
    uint offset = token_id * WEIGHT_ROW_BYTES + dimension * 2u;
    output_values[index] = unpackHalf2x16(read_u16(offset)).x;
}
)glsl";

constexpr char kBF16EmbeddingWgsl[] = R"wgsl(
@group(0) @binding(0) var<storage, read> weights: array<u32>;
@group(0) @binding(1) var<storage, read> token_ids: array<u32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
const VOCABULARY_SIZE: u32 = 1u;
const EMBEDDING_DIMENSION: u32 = 1u;
const TOKEN_COUNT: u32 = 1u;
const TOTAL_ELEMENTS: u32 = 1u;
const WEIGHT_ROW_BYTES: u32 = 2u;
fn read_u16(byte_offset: u32) -> u32 {
    let word = weights[byte_offset >> 2u];
    return (word >> ((byte_offset & 3u) * 8u)) & 65535u;
}
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= TOTAL_ELEMENTS) { return; }
    let token_row = gid.x / EMBEDDING_DIMENSION;
    let dimension = gid.x % EMBEDDING_DIMENSION;
    let token_id = token_ids[token_row];
    if (token_id >= VOCABULARY_SIZE) { return; }
    let offset = token_id * WEIGHT_ROW_BYTES + dimension * 2u;
    output[gid.x] = bitcast<f32>(read_u16(offset) << 16u);
}
)wgsl";

constexpr char kBF16EmbeddingGlsl[] = R"glsl(#version 310 es
precision highp float;
precision highp int;
layout(local_size_x = 64) in;
layout(std430, binding = 0) readonly buffer Weights { uint weights[]; };
layout(std430, binding = 1) readonly buffer TokenIds { uint token_ids[]; };
layout(std430, binding = 2) writeonly buffer Output { float output_values[]; };
const uint VOCABULARY_SIZE = 1u;
const uint EMBEDDING_DIMENSION = 1u;
const uint TOKEN_COUNT = 1u;
const uint TOTAL_ELEMENTS = 1u;
const uint WEIGHT_ROW_BYTES = 2u;
uint read_u16(uint byte_offset) {
    uint word = weights[byte_offset >> 2u];
    return (word >> ((byte_offset & 3u) * 8u)) & 65535u;
}
void main() {
    uint index = gl_GlobalInvocationID.x;
    if (index >= TOTAL_ELEMENTS) return;
    uint token_row = index / EMBEDDING_DIMENSION;
    uint dimension = index % EMBEDDING_DIMENSION;
    uint token_id = token_ids[token_row];
    if (token_id >= VOCABULARY_SIZE) return;
    uint offset = token_id * WEIGHT_ROW_BYTES + dimension * 2u;
    output_values[index] = uintBitsToFloat(read_u16(offset) << 16u);
}
)glsl";


constexpr char kQ6_KMatMulWgsl[] = R"wgsl(
@group(0) @binding(0) var<storage, read> activations: array<f32>;
@group(0) @binding(1) var<storage, read> weights: array<u32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
const ROWS: u32 = 1u;
const OUTPUT_CHANNELS: u32 = 1u;
const INNER: u32 = 256u;
const WEIGHT_ROW_BYTES: u32 = 210u;
fn read_u8(byte_offset:u32)->u32 {
    let word=weights[byte_offset>>2u];
    return (word>>((byte_offset&3u)*8u))&255u;
}
fn read_u16(byte_offset:u32)->u32 {
    let word=weights[byte_offset>>2u];
    return (word>>((byte_offset&3u)*8u))&65535u;
}
fn decode_weight(channel:u32,k:u32)->f32 {
    let block=k/256u;
    let within=k%256u;
    let group=within/128u;
    let local=within%128u;
    let segment=local/32u;
    let lane=local%32u;
    let offset=channel*WEIGHT_ROW_BYTES+block*210u;
    let ql_index=group*64u+(segment%2u)*32u+lane;
    let qh_index=128u+group*32u+lane;
    let scale_index=192u+group*8u+segment*2u+lane/16u;
    let ql=read_u8(offset+ql_index);
    let qh=read_u8(offset+qh_index);
    let nibble=select(ql&15u,ql>>4u,segment>=2u);
    let high=(qh>>(segment*2u))&3u;
    let quantized=i32(nibble|(high<<4u))-32;
    let scale=bitcast<i32>(read_u8(offset+scale_index)<<24u)>>24u;
    let d=unpack2x16float(vec2<u32>(read_u16(offset+208u),0u)).x;
    return d*f32(scale)*f32(quantized);
}
@compute @workgroup_size(8,8)
fn main(@builtin(global_invocation_id) gid:vec3<u32>){
    let row=gid.y;let channel=gid.x;
    if(row>=ROWS||channel>=OUTPUT_CHANNELS){return;}
    var sum=0.0;
    for(var k=0u;k<INNER;k=k+1u){sum=sum+activations[row*INNER+k]*decode_weight(channel,k);}
    output[row*OUTPUT_CHANNELS+channel]=sum;
}
)wgsl";

constexpr char kQ6_KMatMulGlsl[] = R"glsl(#version 310 es
precision highp float;
precision highp int;
layout(local_size_x=8,local_size_y=8) in;
layout(std430,binding=0) readonly buffer Activations{float activations[];};
layout(std430,binding=1) readonly buffer Weights{uint weights[];};
layout(std430,binding=2) writeonly buffer Output{float output_values[];};
const uint ROWS=1u;
const uint OUTPUT_CHANNELS=1u;
const uint INNER=256u;
const uint WEIGHT_ROW_BYTES=210u;
uint read_u8(uint byte_offset){uint word=weights[byte_offset>>2u];return(word>>((byte_offset&3u)*8u))&255u;}
uint read_u16(uint byte_offset){uint word=weights[byte_offset>>2u];return(word>>((byte_offset&3u)*8u))&65535u;}
float decode_weight(uint channel,uint k){
    uint block=k/256u,within=k%256u,group=within/128u,local=within%128u;
    uint segment=local/32u,lane=local%32u;
    uint offset=channel*WEIGHT_ROW_BYTES+block*210u;
    uint ql_index=group*64u+(segment%2u)*32u+lane;
    uint qh_index=128u+group*32u+lane;
    uint scale_index=192u+group*8u+segment*2u+lane/16u;
    uint ql=read_u8(offset+ql_index),qh=read_u8(offset+qh_index);
    uint nibble=segment<2u?(ql&15u):(ql>>4u);
    uint high=(qh>>(segment*2u))&3u;
    int quantized=int(nibble|(high<<4u))-32;
    int scale=int(read_u8(offset+scale_index));
    if(scale>=128)scale-=256;
    float d=unpackHalf2x16(read_u16(offset+208u)).x;
    return d*float(scale)*float(quantized);
}
void main(){
    uint row=gl_GlobalInvocationID.y,channel=gl_GlobalInvocationID.x;
    if(row>=ROWS||channel>=OUTPUT_CHANNELS)return;
    float sum=0.0;
    for(uint k=0u;k<INNER;++k)sum+=activations[row*INNER+k]*decode_weight(channel,k);
    output_values[row*OUTPUT_CHANNELS+channel]=sum;
}
)glsl";


constexpr char kQ6_KEmbeddingWgsl[] = R"wgsl(
@group(0) @binding(0) var<storage, read> weights: array<u32>;
@group(0) @binding(1) var<storage, read> token_ids: array<u32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
const VOCABULARY_SIZE:u32=1u;
const EMBEDDING_DIMENSION:u32=256u;
const TOKEN_COUNT:u32=1u;
const TOTAL_ELEMENTS:u32=256u;
const WEIGHT_ROW_BYTES:u32=210u;
fn read_u8(byte_offset:u32)->u32{let word=weights[byte_offset>>2u];return(word>>((byte_offset&3u)*8u))&255u;}
fn read_u16(byte_offset:u32)->u32{let word=weights[byte_offset>>2u];return(word>>((byte_offset&3u)*8u))&65535u;}
fn decode_weight(token:u32,dimension:u32)->f32{
 let block=dimension/256u;let within=dimension%256u;let group=within/128u;let local=within%128u;
 let segment=local/32u;let lane=local%32u;let offset=token*WEIGHT_ROW_BYTES+block*210u;
 let ql_index=group*64u+(segment%2u)*32u+lane;let qh_index=128u+group*32u+lane;
 let scale_index=192u+group*8u+segment*2u+lane/16u;
 let ql=read_u8(offset+ql_index);let qh=read_u8(offset+qh_index);
 let nibble=select(ql&15u,ql>>4u,segment>=2u);let high=(qh>>(segment*2u))&3u;
 let quantized=i32(nibble|(high<<4u))-32;
 let scale=bitcast<i32>(read_u8(offset+scale_index)<<24u)>>24u;
 let d=unpack2x16float(vec2<u32>(read_u16(offset+208u),0u)).x;
 return d*f32(scale)*f32(quantized);
}
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid:vec3<u32>){
 if(gid.x>=TOTAL_ELEMENTS)return;
 let token_row=gid.x/EMBEDDING_DIMENSION;let dimension=gid.x%EMBEDDING_DIMENSION;
 if(token_row>=TOKEN_COUNT)return;let token=token_ids[token_row];
 if(token>=VOCABULARY_SIZE){output[gid.x]=bitcast<f32>(0x7fc00000u);return;}
 output[gid.x]=decode_weight(token,dimension);
}
)wgsl";

constexpr char kQ6_KEmbeddingGlsl[] = R"glsl(#version 310 es
precision highp float;precision highp int;
layout(local_size_x=64) in;
layout(std430,binding=0) readonly buffer Weights{uint weights[];};
layout(std430,binding=1) readonly buffer TokenIds{uint token_ids[];};
layout(std430,binding=2) writeonly buffer Output{float output_values[];};
const uint VOCABULARY_SIZE=1u,EMBEDDING_DIMENSION=256u,TOKEN_COUNT=1u,TOTAL_ELEMENTS=256u,WEIGHT_ROW_BYTES=210u;
uint read_u8(uint byte_offset){uint word=weights[byte_offset>>2u];return(word>>((byte_offset&3u)*8u))&255u;}
uint read_u16(uint byte_offset){uint word=weights[byte_offset>>2u];return(word>>((byte_offset&3u)*8u))&65535u;}
float decode_weight(uint token,uint dimension){
 uint block=dimension/256u,within=dimension%256u,group=within/128u,local=within%128u;
 uint segment=local/32u,lane=local%32u,offset=token*WEIGHT_ROW_BYTES+block*210u;
 uint ql_index=group*64u+(segment%2u)*32u+lane,qh_index=128u+group*32u+lane;
 uint scale_index=192u+group*8u+segment*2u+lane/16u;
 uint ql=read_u8(offset+ql_index),qh=read_u8(offset+qh_index);
 uint nibble=segment<2u?(ql&15u):(ql>>4u),high=(qh>>(segment*2u))&3u;
 int quantized=int(nibble|(high<<4u))-32,scale=int(read_u8(offset+scale_index));if(scale>=128)scale-=256;
 float d=unpackHalf2x16(read_u16(offset+208u)).x;
 return d*float(scale)*float(quantized);
}
void main(){uint index=gl_GlobalInvocationID.x;if(index>=TOTAL_ELEMENTS)return;
 uint token_row=index/EMBEDDING_DIMENSION,dimension=index%EMBEDDING_DIMENSION;
 if(token_row>=TOKEN_COUNT)return;uint token=token_ids[token_row];
 if(token>=VOCABULARY_SIZE){output_values[index]=uintBitsToFloat(0x7fc00000u);return;}
 output_values[index]=decode_weight(token,dimension);
}
)glsl";

constexpr char kQ4_KMatMulWgsl[] = R"wgsl(
@group(0) @binding(0) var<storage,read> activations:array<f32>;
@group(0) @binding(1) var<storage,read> weights:array<u32>;
@group(0) @binding(2) var<storage,read_write> output:array<f32>;
const ROWS:u32=1u;
const OUTPUT_CHANNELS:u32=1u;
const INNER:u32=256u;
const WEIGHT_ROW_BYTES:u32=144u;
fn read_u8(byte_offset:u32)->u32{let word=weights[byte_offset>>2u];return(word>>((byte_offset&3u)*8u))&255u;}
fn read_u16(byte_offset:u32)->u32{let word=weights[byte_offset>>2u];return(word>>((byte_offset&3u)*8u))&65535u;}
fn decode_weight(channel:u32,k:u32)->f32{
 let block=k/256u;let within=k%256u;let group=within/64u;let local=within%64u;
 let segment=local/32u;let lane=local%32u;let index=group*2u+segment;
 let offset=channel*WEIGHT_ROW_BYTES+block*144u;
 let scales_offset=offset+4u;
 var scale:u32=0u;var minimum:u32=0u;
 if(index<4u){scale=read_u8(scales_offset+index)&63u;minimum=read_u8(scales_offset+index+4u)&63u;}
 else{scale=(read_u8(scales_offset+index+4u)&15u)|((read_u8(scales_offset+index-4u)>>6u)<<4u);
      minimum=(read_u8(scales_offset+index+4u)>>4u)|((read_u8(scales_offset+index)>>6u)<<4u);}
 let ql=read_u8(offset+16u+group*32u+lane);
 let nibble=select(ql&15u,ql>>4u,segment==1u);
 let high=0u;
 let quantized=i32(nibble);
 let d=unpack2x16float(vec2<u32>(read_u16(offset),0u)).x;
 let dmin=unpack2x16float(vec2<u32>(read_u16(offset+2u),0u)).x;
 return d*f32(scale)*f32(quantized)-dmin*f32(minimum);
}
@compute @workgroup_size(8,8)
fn main(@builtin(global_invocation_id)gid:vec3<u32>){
 let row=gid.y;let channel=gid.x;if(row>=ROWS||channel>=OUTPUT_CHANNELS){return;}
 var sum=0.0;for(var k=0u;k<INNER;k=k+1u){sum+=activations[row*INNER+k]*decode_weight(channel,k);}
 output[row*OUTPUT_CHANNELS+channel]=sum;
}
)wgsl";

constexpr char kQ4_KMatMulGlsl[] = R"glsl(#version 310 es
precision highp float;precision highp int;
layout(local_size_x=8,local_size_y=8)in;
layout(std430,binding=0)readonly buffer Activations{float activations[];};
layout(std430,binding=1)readonly buffer Weights{uint weights[];};
layout(std430,binding=2)writeonly buffer Output{float output_values[];};
const uint ROWS=1u;const uint OUTPUT_CHANNELS=1u;const uint INNER=256u;const uint WEIGHT_ROW_BYTES=144u;
uint read_u8(uint o){uint w=weights[o>>2u];return(w>>((o&3u)*8u))&255u;}
uint read_u16(uint o){uint w=weights[o>>2u];return(w>>((o&3u)*8u))&65535u;}
float decode_weight(uint channel,uint k){
 uint block=k/256u,within=k%256u,group=within/64u,local=within%64u,segment=local/32u,lane=local%32u,index=group*2u+segment;
 uint offset=channel*WEIGHT_ROW_BYTES+block*144u,scales_offset=offset+4u;
 uint scale=0u,minimum=0u;
 if(index<4u){scale=read_u8(scales_offset+index)&63u;minimum=read_u8(scales_offset+index+4u)&63u;}
 else{scale=(read_u8(scales_offset+index+4u)&15u)|((read_u8(scales_offset+index-4u)>>6u)<<4u);
      minimum=(read_u8(scales_offset+index+4u)>>4u)|((read_u8(scales_offset+index)>>6u)<<4u);}
 uint ql=read_u8(offset+16u+group*32u+lane);
 uint nibble=segment==0u?(ql&15u):(ql>>4u);
 uint high=0u;
 int quantized=int(nibble);
 float d=unpackHalf2x16(read_u16(offset)).x,dmin=unpackHalf2x16(read_u16(offset+2u)).x;
 return d*float(scale)*float(quantized)-dmin*float(minimum);
}
void main(){uint row=gl_GlobalInvocationID.y,channel=gl_GlobalInvocationID.x;if(row>=ROWS||channel>=OUTPUT_CHANNELS)return;
 float sum=0.0;for(uint k=0u;k<INNER;++k)sum+=activations[row*INNER+k]*decode_weight(channel,k);
 output_values[row*OUTPUT_CHANNELS+channel]=sum;}
)glsl";


constexpr char kQ5_KMatMulWgsl[] = R"wgsl(
@group(0) @binding(0) var<storage,read> activations:array<f32>;
@group(0) @binding(1) var<storage,read> weights:array<u32>;
@group(0) @binding(2) var<storage,read_write> output:array<f32>;
const ROWS:u32=1u;
const OUTPUT_CHANNELS:u32=1u;
const INNER:u32=256u;
const WEIGHT_ROW_BYTES:u32=176u;
fn read_u8(byte_offset:u32)->u32{let word=weights[byte_offset>>2u];return(word>>((byte_offset&3u)*8u))&255u;}
fn read_u16(byte_offset:u32)->u32{let word=weights[byte_offset>>2u];return(word>>((byte_offset&3u)*8u))&65535u;}
fn decode_weight(channel:u32,k:u32)->f32{
 let block=k/256u;let within=k%256u;let group=within/64u;let local=within%64u;
 let segment=local/32u;let lane=local%32u;let index=group*2u+segment;
 let offset=channel*WEIGHT_ROW_BYTES+block*176u;
 let scales_offset=offset+4u;
 var scale:u32=0u;var minimum:u32=0u;
 if(index<4u){scale=read_u8(scales_offset+index)&63u;minimum=read_u8(scales_offset+index+4u)&63u;}
 else{scale=(read_u8(scales_offset+index+4u)&15u)|((read_u8(scales_offset+index-4u)>>6u)<<4u);
      minimum=(read_u8(scales_offset+index+4u)>>4u)|((read_u8(scales_offset+index)>>6u)<<4u);}
 let ql=read_u8(offset+48u+group*32u+lane);
 let nibble=select(ql&15u,ql>>4u,segment==1u);
 let high=(read_u8(offset+16u+lane)>>(group*2u+segment))&1u;
 let quantized=i32(nibble|(high<<4u));
 let d=unpack2x16float(vec2<u32>(read_u16(offset),0u)).x;
 let dmin=unpack2x16float(vec2<u32>(read_u16(offset+2u),0u)).x;
 return d*f32(scale)*f32(quantized)-dmin*f32(minimum);
}
@compute @workgroup_size(8,8)
fn main(@builtin(global_invocation_id)gid:vec3<u32>){
 let row=gid.y;let channel=gid.x;if(row>=ROWS||channel>=OUTPUT_CHANNELS){return;}
 var sum=0.0;for(var k=0u;k<INNER;k=k+1u){sum+=activations[row*INNER+k]*decode_weight(channel,k);}
 output[row*OUTPUT_CHANNELS+channel]=sum;
}
)wgsl";

constexpr char kQ5_KMatMulGlsl[] = R"glsl(#version 310 es
precision highp float;precision highp int;
layout(local_size_x=8,local_size_y=8)in;
layout(std430,binding=0)readonly buffer Activations{float activations[];};
layout(std430,binding=1)readonly buffer Weights{uint weights[];};
layout(std430,binding=2)writeonly buffer Output{float output_values[];};
const uint ROWS=1u;const uint OUTPUT_CHANNELS=1u;const uint INNER=256u;const uint WEIGHT_ROW_BYTES=176u;
uint read_u8(uint o){uint w=weights[o>>2u];return(w>>((o&3u)*8u))&255u;}
uint read_u16(uint o){uint w=weights[o>>2u];return(w>>((o&3u)*8u))&65535u;}
float decode_weight(uint channel,uint k){
 uint block=k/256u,within=k%256u,group=within/64u,local=within%64u,segment=local/32u,lane=local%32u,index=group*2u+segment;
 uint offset=channel*WEIGHT_ROW_BYTES+block*176u,scales_offset=offset+4u;
 uint scale=0u,minimum=0u;
 if(index<4u){scale=read_u8(scales_offset+index)&63u;minimum=read_u8(scales_offset+index+4u)&63u;}
 else{scale=(read_u8(scales_offset+index+4u)&15u)|((read_u8(scales_offset+index-4u)>>6u)<<4u);
      minimum=(read_u8(scales_offset+index+4u)>>4u)|((read_u8(scales_offset+index)>>6u)<<4u);}
 uint ql=read_u8(offset+48u+group*32u+lane);
 uint nibble=segment==0u?(ql&15u):(ql>>4u);
 uint high=(read_u8(offset+16u+lane)>>(group*2u+segment))&1u;
 int quantized=int(nibble|(high<<4u));
 float d=unpackHalf2x16(read_u16(offset)).x,dmin=unpackHalf2x16(read_u16(offset+2u)).x;
 return d*float(scale)*float(quantized)-dmin*float(minimum);
}
void main(){uint row=gl_GlobalInvocationID.y,channel=gl_GlobalInvocationID.x;if(row>=ROWS||channel>=OUTPUT_CHANNELS)return;
 float sum=0.0;for(uint k=0u;k<INNER;++k)sum+=activations[row*INNER+k]*decode_weight(channel,k);
 output_values[row*OUTPUT_CHANNELS+channel]=sum;}
)glsl";


constexpr char kQ2_KMatMulWgsl[] = R"wgsl(
@group(0) @binding(0) var<storage,read> activations:array<f32>;
@group(0) @binding(1) var<storage,read> weights:array<u32>;
@group(0) @binding(2) var<storage,read_write> output:array<f32>;
const ROWS:u32=1u;const OUTPUT_CHANNELS:u32=1u;const INNER:u32=256u;const WEIGHT_ROW_BYTES:u32=84u;
fn read_u8(o:u32)->u32{let w=weights[o>>2u];return(w>>((o&3u)*8u))&255u;}
fn read_u16(o:u32)->u32{let w=weights[o>>2u];return(w>>((o&3u)*8u))&65535u;}
fn read_u32(o:u32)->u32{return read_u8(o)|(read_u8(o+1u)<<8u)|(read_u8(o+2u)<<16u)|(read_u8(o+3u)<<24u);}
fn decode_weight(channel:u32,k:u32)->f32{
 let block=k/256u;let within=k%256u;let n=within/128u;let local=within%128u;
 let group=local/32u;let side=(local%32u)/16u;let lane=local%16u;let shift=group*2u;
 let scale_index=n*8u+group*2u+side;let offset=channel*WEIGHT_ROW_BYTES+block*84u;
  let packed_scale=read_u8(offset+scale_index);
 let scale=packed_scale&15u;let minimum=packed_scale>>4u;
 let q=(read_u8(offset+16u+n*32u+side*16u+lane)>>shift)&3u;
 let d=unpack2x16float(vec2<u32>(read_u16(offset+80u),0u)).x;
 let dmin=unpack2x16float(vec2<u32>(read_u16(offset+82u),0u)).x;
 return d*f32(scale)*f32(q)-dmin*f32(minimum);
}
@compute @workgroup_size(8,8)
fn main(@builtin(global_invocation_id)gid:vec3<u32>){
 let row=gid.y;let channel=gid.x;if(row>=ROWS||channel>=OUTPUT_CHANNELS){return;}
 var sum=0.0;for(var k=0u;k<INNER;k=k+1u){sum+=activations[row*INNER+k]*decode_weight(channel,k);}
 output[row*OUTPUT_CHANNELS+channel]=sum;
}
)wgsl";
constexpr char kQ2_KMatMulGlsl[] = R"glsl(#version 310 es
precision highp float;precision highp int;
layout(local_size_x=8,local_size_y=8)in;
layout(std430,binding=0)readonly buffer Activations{float activations[];};
layout(std430,binding=1)readonly buffer Weights{uint weights[];};
layout(std430,binding=2)writeonly buffer Output{float output_values[];};
const uint ROWS=1u;const uint OUTPUT_CHANNELS=1u;const uint INNER=256u;const uint WEIGHT_ROW_BYTES=84u;
uint read_u8(uint o){uint w=weights[o>>2u];return(w>>((o&3u)*8u))&255u;}
uint read_u16(uint o){uint w=weights[o>>2u];return(w>>((o&3u)*8u))&65535u;}
uint read_u32(uint o){return read_u8(o)|(read_u8(o+1u)<<8u)|(read_u8(o+2u)<<16u)|(read_u8(o+3u)<<24u);}
float decode_weight(uint channel,uint k){
 uint block=k/256u,within=k%256u,n=within/128u,local=within%128u;
 uint group=local/32u,side=(local%32u)/16u,lane=local%16u,shift=group*2u;
 uint scale_index=n*8u+group*2u+side,offset=channel*WEIGHT_ROW_BYTES+block*84u;
  uint packed_scale=read_u8(offset+scale_index);uint scale=packed_scale&15u,minimum=packed_scale>>4u;
 uint q=(read_u8(offset+16u+n*32u+side*16u+lane)>>shift)&3u;
 float d=unpackHalf2x16(read_u16(offset+80u)).x,dmin=unpackHalf2x16(read_u16(offset+82u)).x;
 return d*float(scale)*float(q)-dmin*float(minimum);
}
void main(){uint row=gl_GlobalInvocationID.y,channel=gl_GlobalInvocationID.x;if(row>=ROWS||channel>=OUTPUT_CHANNELS)return;
 float sum=0.0;for(uint k=0u;k<INNER;++k)sum+=activations[row*INNER+k]*decode_weight(channel,k);
 output_values[row*OUTPUT_CHANNELS+channel]=sum;}
)glsl";


constexpr char kQ3_KMatMulWgsl[] = R"wgsl(
@group(0) @binding(0) var<storage,read> activations:array<f32>;
@group(0) @binding(1) var<storage,read> weights:array<u32>;
@group(0) @binding(2) var<storage,read_write> output:array<f32>;
const ROWS:u32=1u;const OUTPUT_CHANNELS:u32=1u;const INNER:u32=256u;const WEIGHT_ROW_BYTES:u32=110u;
fn read_u8(o:u32)->u32{let w=weights[o>>2u];return(w>>((o&3u)*8u))&255u;}
fn read_u16(o:u32)->u32{let w=weights[o>>2u];return(w>>((o&3u)*8u))&65535u;}
fn read_u32(o:u32)->u32{return read_u8(o)|(read_u8(o+1u)<<8u)|(read_u8(o+2u)<<16u)|(read_u8(o+3u)<<24u);}
fn decode_weight(channel:u32,k:u32)->f32{
 let block=k/256u;let within=k%256u;let n=within/128u;let local=within%128u;
 let group=local/32u;let side=(local%32u)/16u;let lane=local%16u;let shift=group*2u;
 let scale_index=n*8u+group*2u+side;let offset=channel*WEIGHT_ROW_BYTES+block*110u;
 
 let aux0=read_u32(offset+96u);let aux1=read_u32(offset+100u);let aux2=read_u32(offset+104u);
 var scale_word=0u;
 if(scale_index<4u){scale_word=(aux0&0x0f0f0f0fu)|((aux2&0x03030303u)<<4u);}
 else if(scale_index<8u){scale_word=(aux1&0x0f0f0f0fu)|(((aux2>>2u)&0x03030303u)<<4u);}
 else if(scale_index<12u){scale_word=((aux0>>4u)&0x0f0f0f0fu)|(((aux2>>4u)&0x03030303u)<<4u);}
 else{scale_word=((aux1>>4u)&0x0f0f0f0fu)|(((aux2>>6u)&0x03030303u)<<4u);}
 let raw_scale=(scale_word>>((scale_index&3u)*8u))&255u;
 let scale=i32(raw_scale)-32;
 let high=(read_u8(offset+side*16u+lane)&(1u<<(n*4u+group)))!=0u;
 let q=(read_u8(offset+32u+n*32u+side*16u+lane)>>shift)&3u;
 let quant=i32(q)-select(4,0,high);
 let d=unpack2x16float(vec2<u32>(read_u16(offset+108u),0u)).x;
 return d*f32(scale)*f32(quant);
}
@compute @workgroup_size(8,8)
fn main(@builtin(global_invocation_id)gid:vec3<u32>){
 let row=gid.y;let channel=gid.x;if(row>=ROWS||channel>=OUTPUT_CHANNELS){return;}
 var sum=0.0;for(var k=0u;k<INNER;k=k+1u){sum+=activations[row*INNER+k]*decode_weight(channel,k);}
 output[row*OUTPUT_CHANNELS+channel]=sum;
}
)wgsl";
constexpr char kQ3_KMatMulGlsl[] = R"glsl(#version 310 es
precision highp float;precision highp int;
layout(local_size_x=8,local_size_y=8)in;
layout(std430,binding=0)readonly buffer Activations{float activations[];};
layout(std430,binding=1)readonly buffer Weights{uint weights[];};
layout(std430,binding=2)writeonly buffer Output{float output_values[];};
const uint ROWS=1u;const uint OUTPUT_CHANNELS=1u;const uint INNER=256u;const uint WEIGHT_ROW_BYTES=110u;
uint read_u8(uint o){uint w=weights[o>>2u];return(w>>((o&3u)*8u))&255u;}
uint read_u16(uint o){uint w=weights[o>>2u];return(w>>((o&3u)*8u))&65535u;}
uint read_u32(uint o){return read_u8(o)|(read_u8(o+1u)<<8u)|(read_u8(o+2u)<<16u)|(read_u8(o+3u)<<24u);}
float decode_weight(uint channel,uint k){
 uint block=k/256u,within=k%256u,n=within/128u,local=within%128u;
 uint group=local/32u,side=(local%32u)/16u,lane=local%16u,shift=group*2u;
 uint scale_index=n*8u+group*2u+side,offset=channel*WEIGHT_ROW_BYTES+block*110u;
 
 uint aux0=read_u32(offset+96u),aux1=read_u32(offset+100u),aux2=read_u32(offset+104u);
 uint scale_word=0u;
 if(scale_index<4u)scale_word=(aux0&0x0f0f0f0fu)|((aux2&0x03030303u)<<4u);
 else if(scale_index<8u)scale_word=(aux1&0x0f0f0f0fu)|(((aux2>>2u)&0x03030303u)<<4u);
 else if(scale_index<12u)scale_word=((aux0>>4u)&0x0f0f0f0fu)|(((aux2>>4u)&0x03030303u)<<4u);
 else scale_word=((aux1>>4u)&0x0f0f0f0fu)|(((aux2>>6u)&0x03030303u)<<4u);
 uint raw_scale=(scale_word>>((scale_index&3u)*8u))&255u;int scale=int(raw_scale)-32;
 bool high=(read_u8(offset+side*16u+lane)&(1u<<(n*4u+group)))!=0u;
 uint q=(read_u8(offset+32u+n*32u+side*16u+lane)>>shift)&3u;
 int quant=int(q)-(high?0:4);float d=unpackHalf2x16(read_u16(offset+108u)).x;
 return d*float(scale)*float(quant);
}
void main(){uint row=gl_GlobalInvocationID.y,channel=gl_GlobalInvocationID.x;if(row>=ROWS||channel>=OUTPUT_CHANNELS)return;
 float sum=0.0;for(uint k=0u;k<INNER;++k)sum+=activations[row*INNER+k]*decode_weight(channel,k);
 output_values[row*OUTPUT_CHANNELS+channel]=sum;}
)glsl";

constexpr char kQ8_0MatMulWgsl[] = R"wgsl(
@group(0) @binding(0) var<storage, read> activations: array<f32>;
@group(0) @binding(1) var<storage, read> weights: array<u32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
const ROWS: u32 = 1u;
const OUTPUT_CHANNELS: u32 = 1u;
const INNER: u32 = 32u;
const WEIGHT_ROW_BYTES: u32 = 34u;
fn read_u8(byte_offset: u32) -> u32 {
    let word = weights[byte_offset >> 2u];
    let shift = (byte_offset & 3u) * 8u;
    return (word >> shift) & 255u;
}
fn read_u16(byte_offset: u32) -> u32 {
    let word = weights[byte_offset >> 2u];
    let shift = (byte_offset & 3u) * 8u;
    return (word >> shift) & 65535u;
}
@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row = gid.y;
    let channel = gid.x;
    if (row >= ROWS || channel >= OUTPUT_CHANNELS) { return; }
    var sum = 0.0;
    for (var k = 0u; k < INNER; k = k + 1u) {
        let block = k / 32u;
        let within = k % 32u;
        let block_offset = channel * WEIGHT_ROW_BYTES + block * 34u;
        let scale = unpack2x16float(vec2<u32>(read_u16(block_offset), 0u)).x;
        let raw = read_u8(block_offset + 2u + within);
        let quantized = select(i32(raw), i32(raw) - 256, raw >= 128u);
        sum = sum + activations[row * INNER + k] * scale * f32(quantized);
    }
    output[row * OUTPUT_CHANNELS + channel] = sum;
}
)wgsl";

constexpr char kQ8_0MatMulGlsl[] = R"glsl(#version 310 es
precision highp float;
precision highp int;
layout(local_size_x = 8, local_size_y = 8) in;
layout(std430, binding = 0) readonly buffer Activations { float activations[]; };
layout(std430, binding = 1) readonly buffer Weights { uint weights[]; };
layout(std430, binding = 2) writeonly buffer Output { float output_values[]; };
const uint ROWS = 1u;
const uint OUTPUT_CHANNELS = 1u;
const uint INNER = 32u;
const uint WEIGHT_ROW_BYTES = 34u;
uint read_u8(uint byte_offset) {
    uint word = weights[byte_offset >> 2u];
    return (word >> ((byte_offset & 3u) * 8u)) & 255u;
}
uint read_u16(uint byte_offset) {
    uint word = weights[byte_offset >> 2u];
    return (word >> ((byte_offset & 3u) * 8u)) & 65535u;
}
void main() {
    uint row = gl_GlobalInvocationID.y;
    uint channel = gl_GlobalInvocationID.x;
    if (row >= ROWS || channel >= OUTPUT_CHANNELS) return;
    float sum = 0.0;
    for (uint k = 0u; k < INNER; ++k) {
        uint block = k / 32u;
        uint within = k % 32u;
        uint block_offset = channel * WEIGHT_ROW_BYTES + block * 34u;
        float scale = unpackHalf2x16(read_u16(block_offset)).x;
        uint raw = read_u8(block_offset + 2u + within);
        int quantized = raw >= 128u ? int(raw) - 256 : int(raw);
        sum += activations[row * INNER + k] * scale * float(quantized);
    }
    output_values[row * OUTPUT_CHANNELS + channel] = sum;
}
)glsl";

constexpr char kQ4_0MatMulWgsl[] = R"wgsl(
@group(0) @binding(0) var<storage, read> activations: array<f32>;
@group(0) @binding(1) var<storage, read> weights: array<u32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
const ROWS: u32 = 1u;
const OUTPUT_CHANNELS: u32 = 1u;
const INNER: u32 = 32u;
const WEIGHT_ROW_BYTES: u32 = 18u;
fn read_u8(byte_offset: u32) -> u32 {
    let word = weights[byte_offset >> 2u];
    return (word >> ((byte_offset & 3u) * 8u)) & 255u;
}
fn read_u16(byte_offset: u32) -> u32 {
    let word = weights[byte_offset >> 2u];
    return (word >> ((byte_offset & 3u) * 8u)) & 65535u;
}
@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row = gid.y;
    let channel = gid.x;
    if (row >= ROWS || channel >= OUTPUT_CHANNELS) { return; }
    var sum = 0.0;
    for (var k = 0u; k < INNER; k = k + 1u) {
        let block = k / 32u;
        let within = k % 32u;
        let block_offset = channel * WEIGHT_ROW_BYTES + block * 18u;
        let scale = unpack2x16float(vec2<u32>(read_u16(block_offset), 0u)).x;
        let packed = read_u8(block_offset + 2u + (within % 16u));
        let nibble = select(packed & 15u, packed >> 4u, within >= 16u);
        let quantized = i32(nibble) - 8;
        sum = sum + activations[row * INNER + k] * scale * f32(quantized);
    }
    output[row * OUTPUT_CHANNELS + channel] = sum;
}
)wgsl";

constexpr char kQ4_0MatMulGlsl[] = R"glsl(#version 310 es
precision highp float;
precision highp int;
layout(local_size_x = 8, local_size_y = 8) in;
layout(std430, binding = 0) readonly buffer Activations { float activations[]; };
layout(std430, binding = 1) readonly buffer Weights { uint weights[]; };
layout(std430, binding = 2) writeonly buffer Output { float output_values[]; };
const uint ROWS = 1u;
const uint OUTPUT_CHANNELS = 1u;
const uint INNER = 32u;
const uint WEIGHT_ROW_BYTES = 18u;
uint read_u8(uint byte_offset) {
    uint word = weights[byte_offset >> 2u];
    return (word >> ((byte_offset & 3u) * 8u)) & 255u;
}
uint read_u16(uint byte_offset) {
    uint word = weights[byte_offset >> 2u];
    return (word >> ((byte_offset & 3u) * 8u)) & 65535u;
}
void main() {
    uint row = gl_GlobalInvocationID.y;
    uint channel = gl_GlobalInvocationID.x;
    if (row >= ROWS || channel >= OUTPUT_CHANNELS) return;
    float sum = 0.0;
    for (uint k = 0u; k < INNER; ++k) {
        uint block = k / 32u;
        uint within = k % 32u;
        uint block_offset = channel * WEIGHT_ROW_BYTES + block * 18u;
        float scale = unpackHalf2x16(read_u16(block_offset)).x;
        uint packed = read_u8(block_offset + 2u + (within % 16u));
        uint nibble = within < 16u ? (packed & 15u) : (packed >> 4u);
        int quantized = int(nibble) - 8;
        sum += activations[row * INNER + k] * scale * float(quantized);
    }
    output_values[row * OUTPUT_CHANNELS + channel] = sum;
}
)glsl";

constexpr char kQ4_1MatMulWgsl[] = R"wgsl(
@group(0) @binding(0) var<storage, read> activations: array<f32>;
@group(0) @binding(1) var<storage, read> weights: array<u32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
const ROWS: u32 = 1u;
const OUTPUT_CHANNELS: u32 = 1u;
const INNER: u32 = 32u;
const WEIGHT_ROW_BYTES: u32 = 20u;
fn read_u8(byte_offset: u32) -> u32 {
    let word = weights[byte_offset >> 2u];
    return (word >> ((byte_offset & 3u) * 8u)) & 255u;
}
fn read_u16(byte_offset: u32) -> u32 {
    let word = weights[byte_offset >> 2u];
    return (word >> ((byte_offset & 3u) * 8u)) & 65535u;
}
@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row = gid.y;
    let channel = gid.x;
    if (row >= ROWS || channel >= OUTPUT_CHANNELS) { return; }
    var sum = 0.0;
    for (var k = 0u; k < INNER; k = k + 1u) {
        let block = k / 32u;
        let within = k % 32u;
        let block_offset = channel * WEIGHT_ROW_BYTES + block * 20u;
        let scale = unpack2x16float(vec2<u32>(read_u16(block_offset), 0u)).x;
        let minimum = unpack2x16float(vec2<u32>(read_u16(block_offset + 2u), 0u)).x;
        let packed = read_u8(block_offset + 4u + (within % 16u));
        let nibble = select(packed & 15u, packed >> 4u, within >= 16u);
        let value = scale * f32(nibble) + minimum;
        sum = sum + activations[row * INNER + k] * value;
    }
    output[row * OUTPUT_CHANNELS + channel] = sum;
}
)wgsl";

constexpr char kQ4_1MatMulGlsl[] = R"glsl(#version 310 es
precision highp float;
precision highp int;
layout(local_size_x = 8, local_size_y = 8) in;
layout(std430, binding = 0) readonly buffer Activations { float activations[]; };
layout(std430, binding = 1) readonly buffer Weights { uint weights[]; };
layout(std430, binding = 2) writeonly buffer Output { float output_values[]; };
const uint ROWS = 1u;
const uint OUTPUT_CHANNELS = 1u;
const uint INNER = 32u;
const uint WEIGHT_ROW_BYTES = 20u;
uint read_u8(uint byte_offset) {
    uint word = weights[byte_offset >> 2u];
    return (word >> ((byte_offset & 3u) * 8u)) & 255u;
}
uint read_u16(uint byte_offset) {
    uint word = weights[byte_offset >> 2u];
    return (word >> ((byte_offset & 3u) * 8u)) & 65535u;
}
void main() {
    uint row = gl_GlobalInvocationID.y;
    uint channel = gl_GlobalInvocationID.x;
    if (row >= ROWS || channel >= OUTPUT_CHANNELS) return;
    float sum = 0.0;
    for (uint k = 0u; k < INNER; ++k) {
        uint block = k / 32u;
        uint within = k % 32u;
        uint block_offset = channel * WEIGHT_ROW_BYTES + block * 20u;
        float scale = unpackHalf2x16(read_u16(block_offset)).x;
        float minimum = unpackHalf2x16(read_u16(block_offset + 2u)).x;
        uint packed = read_u8(block_offset + 4u + (within % 16u));
        uint nibble = within < 16u ? (packed & 15u) : (packed >> 4u);
        float value = scale * float(nibble) + minimum;
        sum += activations[row * INNER + k] * value;
    }
    output_values[row * OUTPUT_CHANNELS + channel] = sum;
}
)glsl";


constexpr char kQ5_0EmbeddingWgsl[] = R"wgsl(
@group(0) @binding(0) var<storage, read> weights: array<u32>;
@group(0) @binding(1) var<storage, read> token_ids: array<u32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
const VOCABULARY_SIZE: u32 = 1u;
const EMBEDDING_DIMENSION: u32 = 32u;
const TOKEN_COUNT: u32 = 1u;
const TOTAL_ELEMENTS: u32 = 32u;
const WEIGHT_ROW_BYTES: u32 = 22u;
fn read_u8(byte_offset: u32) -> u32 {
    let word = weights[byte_offset >> 2u];
    return (word >> ((byte_offset & 3u) * 8u)) & 255u;
}
fn read_u16(byte_offset: u32) -> u32 {
    let word = weights[byte_offset >> 2u];
    return (word >> ((byte_offset & 3u) * 8u)) & 65535u;
}
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= TOTAL_ELEMENTS) { return; }
    let token_row = gid.x / EMBEDDING_DIMENSION;
    let dimension = gid.x % EMBEDDING_DIMENSION;
    let token_id = token_ids[token_row];
    if (token_id >= VOCABULARY_SIZE) { return; }
    let block = dimension / 32u;
    let within = dimension % 32u;
    let offset = token_id * WEIGHT_ROW_BYTES + block * 22u;
    let scale = unpack2x16float(vec2<u32>(read_u16(offset), 0u)).x;
    let high_bits = read_u8(offset + 2u) | (read_u8(offset + 3u) << 8u) |
                    (read_u8(offset + 4u) << 16u) | (read_u8(offset + 5u) << 24u);
    let packed = read_u8(offset + 6u + (within % 16u));
    let nibble = select(packed & 15u, packed >> 4u, within >= 16u);
    let high = (high_bits >> within) & 1u;
    output[gid.x] = scale * f32(i32(nibble | (high << 4u)) - 16);
}
)wgsl";

constexpr char kQ5_0EmbeddingGlsl[] = R"glsl(#version 310 es
precision highp float;
precision highp int;
layout(local_size_x = 64) in;
layout(std430, binding = 0) readonly buffer Weights { uint weights[]; };
layout(std430, binding = 1) readonly buffer TokenIds { uint token_ids[]; };
layout(std430, binding = 2) writeonly buffer Output { float output_values[]; };
const uint VOCABULARY_SIZE = 1u;
const uint EMBEDDING_DIMENSION = 32u;
const uint TOKEN_COUNT = 1u;
const uint TOTAL_ELEMENTS = 32u;
const uint WEIGHT_ROW_BYTES = 22u;
uint read_u8(uint byte_offset) {
    uint word = weights[byte_offset >> 2u];
    return (word >> ((byte_offset & 3u) * 8u)) & 255u;
}
uint read_u16(uint byte_offset) {
    uint word = weights[byte_offset >> 2u];
    return (word >> ((byte_offset & 3u) * 8u)) & 65535u;
}
void main() {
    uint index = gl_GlobalInvocationID.x;
    if (index >= TOTAL_ELEMENTS) return;
    uint token_row = index / EMBEDDING_DIMENSION;
    uint dimension = index % EMBEDDING_DIMENSION;
    uint token_id = token_ids[token_row];
    if (token_id >= VOCABULARY_SIZE) return;
    uint block = dimension / 32u;
    uint within = dimension % 32u;
    uint offset = token_id * WEIGHT_ROW_BYTES + block * 22u;
    float scale = unpackHalf2x16(read_u16(offset)).x;
    uint high_bits = read_u8(offset + 2u) | (read_u8(offset + 3u) << 8u) |
                     (read_u8(offset + 4u) << 16u) | (read_u8(offset + 5u) << 24u);
    uint packed = read_u8(offset + 6u + (within % 16u));
    uint nibble = within < 16u ? (packed & 15u) : (packed >> 4u);
    uint high = (high_bits >> within) & 1u;
    output_values[index] = scale * float(int(nibble | (high << 4u)) - 16);
}
)glsl";


constexpr char kQ5_1EmbeddingWgsl[] = R"wgsl(
@group(0) @binding(0) var<storage, read> weights: array<u32>;
@group(0) @binding(1) var<storage, read> token_ids: array<u32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
const VOCABULARY_SIZE: u32 = 1u;
const EMBEDDING_DIMENSION: u32 = 32u;
const TOKEN_COUNT: u32 = 1u;
const TOTAL_ELEMENTS: u32 = 32u;
const WEIGHT_ROW_BYTES: u32 = 24u;
fn read_u8(byte_offset: u32) -> u32 {
    let word = weights[byte_offset >> 2u];
    return (word >> ((byte_offset & 3u) * 8u)) & 255u;
}
fn read_u16(byte_offset: u32) -> u32 {
    let word = weights[byte_offset >> 2u];
    return (word >> ((byte_offset & 3u) * 8u)) & 65535u;
}
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= TOTAL_ELEMENTS) { return; }
    let token_row = gid.x / EMBEDDING_DIMENSION;
    let dimension = gid.x % EMBEDDING_DIMENSION;
    let token_id = token_ids[token_row];
    if (token_id >= VOCABULARY_SIZE) { return; }
    let block = dimension / 32u;
    let within = dimension % 32u;
    let offset = token_id * WEIGHT_ROW_BYTES + block * 24u;
    let scale = unpack2x16float(vec2<u32>(read_u16(offset), 0u)).x;
    let minimum = unpack2x16float(vec2<u32>(read_u16(offset + 2u), 0u)).x;
    let high_bits = read_u8(offset + 4u) | (read_u8(offset + 5u) << 8u) |
                    (read_u8(offset + 6u) << 16u) | (read_u8(offset + 7u) << 24u);
    let packed = read_u8(offset + 8u + (within % 16u));
    let nibble = select(packed & 15u, packed >> 4u, within >= 16u);
    let high = (high_bits >> within) & 1u;
    output[gid.x] = scale * f32(nibble | (high << 4u)) + minimum;
}
)wgsl";

constexpr char kQ5_1EmbeddingGlsl[] = R"glsl(#version 310 es
precision highp float;
precision highp int;
layout(local_size_x = 64) in;
layout(std430, binding = 0) readonly buffer Weights { uint weights[]; };
layout(std430, binding = 1) readonly buffer TokenIds { uint token_ids[]; };
layout(std430, binding = 2) writeonly buffer Output { float output_values[]; };
const uint VOCABULARY_SIZE = 1u;
const uint EMBEDDING_DIMENSION = 32u;
const uint TOKEN_COUNT = 1u;
const uint TOTAL_ELEMENTS = 32u;
const uint WEIGHT_ROW_BYTES = 24u;
uint read_u8(uint byte_offset) {
    uint word = weights[byte_offset >> 2u];
    return (word >> ((byte_offset & 3u) * 8u)) & 255u;
}
uint read_u16(uint byte_offset) {
    uint word = weights[byte_offset >> 2u];
    return (word >> ((byte_offset & 3u) * 8u)) & 65535u;
}
void main() {
    uint index = gl_GlobalInvocationID.x;
    if (index >= TOTAL_ELEMENTS) return;
    uint token_row = index / EMBEDDING_DIMENSION;
    uint dimension = index % EMBEDDING_DIMENSION;
    uint token_id = token_ids[token_row];
    if (token_id >= VOCABULARY_SIZE) return;
    uint block = dimension / 32u;
    uint within = dimension % 32u;
    uint offset = token_id * WEIGHT_ROW_BYTES + block * 24u;
    float scale = unpackHalf2x16(read_u16(offset)).x;
    float minimum = unpackHalf2x16(read_u16(offset + 2u)).x;
    uint high_bits = read_u8(offset + 4u) | (read_u8(offset + 5u) << 8u) |
                     (read_u8(offset + 6u) << 16u) | (read_u8(offset + 7u) << 24u);
    uint packed = read_u8(offset + 8u + (within % 16u));
    uint nibble = within < 16u ? (packed & 15u) : (packed >> 4u);
    uint high = (high_bits >> within) & 1u;
    output_values[index] = scale * float(nibble | (high << 4u)) + minimum;
}
)glsl";

constexpr char kQ4_0EmbeddingWgsl[] = R"wgsl(
@group(0) @binding(0) var<storage, read> weights: array<u32>;
@group(0) @binding(1) var<storage, read> token_ids: array<u32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
const VOCABULARY_SIZE: u32 = 1u;
const EMBEDDING_DIMENSION: u32 = 32u;
const TOKEN_COUNT: u32 = 1u;
const TOTAL_ELEMENTS: u32 = 32u;
const WEIGHT_ROW_BYTES: u32 = 18u;
fn read_u8(byte_offset: u32) -> u32 {
    let word = weights[byte_offset >> 2u];
    return (word >> ((byte_offset & 3u) * 8u)) & 255u;
}
fn read_u16(byte_offset: u32) -> u32 {
    let word = weights[byte_offset >> 2u];
    return (word >> ((byte_offset & 3u) * 8u)) & 65535u;
}
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= TOTAL_ELEMENTS) { return; }
    let token_row = gid.x / EMBEDDING_DIMENSION;
    let dimension = gid.x % EMBEDDING_DIMENSION;
    if (token_row >= TOKEN_COUNT) { return; }
    let token = token_ids[token_row];
    if (token >= VOCABULARY_SIZE) { return; }
    let block = dimension / 32u;
    let within = dimension % 32u;
    let block_offset = token * WEIGHT_ROW_BYTES + block * 18u;
    let scale = unpack2x16float(vec2<u32>(read_u16(block_offset), 0u)).x;
    let packed = read_u8(block_offset + 2u + (within % 16u));
    let nibble = select(packed & 15u, packed >> 4u, within >= 16u);
    output[gid.x] = scale * f32(i32(nibble) - 8);
}
)wgsl";

constexpr char kQ4_0EmbeddingGlsl[] = R"glsl(#version 310 es
precision highp float;
precision highp int;
layout(local_size_x = 64) in;
layout(std430, binding = 0) readonly buffer Weights { uint weights[]; };
layout(std430, binding = 1) readonly buffer TokenIds { uint token_ids[]; };
layout(std430, binding = 2) writeonly buffer Output { float output_values[]; };
const uint VOCABULARY_SIZE = 1u;
const uint EMBEDDING_DIMENSION = 32u;
const uint TOKEN_COUNT = 1u;
const uint TOTAL_ELEMENTS = 32u;
const uint WEIGHT_ROW_BYTES = 18u;
uint read_u8(uint byte_offset) {
    uint word = weights[byte_offset >> 2u];
    return (word >> ((byte_offset & 3u) * 8u)) & 255u;
}
uint read_u16(uint byte_offset) {
    uint word = weights[byte_offset >> 2u];
    return (word >> ((byte_offset & 3u) * 8u)) & 65535u;
}
void main() {
    uint index = gl_GlobalInvocationID.x;
    if (index >= TOTAL_ELEMENTS) return;
    uint token_row = index / EMBEDDING_DIMENSION;
    uint dimension = index % EMBEDDING_DIMENSION;
    if (token_row >= TOKEN_COUNT) return;
    uint token = token_ids[token_row];
    if (token >= VOCABULARY_SIZE) return;
    uint block = dimension / 32u;
    uint within = dimension % 32u;
    uint block_offset = token * WEIGHT_ROW_BYTES + block * 18u;
    float scale = unpackHalf2x16(read_u16(block_offset)).x;
    uint packed = read_u8(block_offset + 2u + (within % 16u));
    uint nibble = within < 16u ? (packed & 15u) : (packed >> 4u);
    output_values[index] = scale * float(int(nibble) - 8);
}
)glsl";

constexpr char kQ8_0EmbeddingWgsl[] = R"wgsl(
@group(0) @binding(0) var<storage, read> weights: array<u32>;
@group(0) @binding(1) var<storage, read> token_ids: array<u32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
const VOCABULARY_SIZE: u32 = 1u;
const EMBEDDING_DIMENSION: u32 = 32u;
const TOKEN_COUNT: u32 = 1u;
const TOTAL_ELEMENTS: u32 = 32u;
const WEIGHT_ROW_BYTES: u32 = 34u;
fn read_u8(byte_offset: u32) -> u32 {
    let word = weights[byte_offset >> 2u];
    return (word >> ((byte_offset & 3u) * 8u)) & 255u;
}
fn read_u16(byte_offset: u32) -> u32 {
    let word = weights[byte_offset >> 2u];
    return (word >> ((byte_offset & 3u) * 8u)) & 65535u;
}
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= TOTAL_ELEMENTS) { return; }
    let token_row = gid.x / EMBEDDING_DIMENSION;
    let dimension = gid.x % EMBEDDING_DIMENSION;
    if (token_row >= TOKEN_COUNT) { return; }
    let token = token_ids[token_row];
    if (token >= VOCABULARY_SIZE) { return; }
    let block = dimension / 32u;
    let within = dimension % 32u;
    let block_offset = token * WEIGHT_ROW_BYTES + block * 34u;
    let scale = unpack2x16float(vec2<u32>(read_u16(block_offset), 0u)).x;
    let raw = read_u8(block_offset + 2u + within);
    let quantized = select(i32(raw), i32(raw) - 256, raw >= 128u);
    output[gid.x] = scale * f32(quantized);
}
)wgsl";

constexpr char kQ8_0EmbeddingGlsl[] = R"glsl(#version 310 es
precision highp float;
precision highp int;
layout(local_size_x = 64) in;
layout(std430, binding = 0) readonly buffer Weights { uint weights[]; };
layout(std430, binding = 1) readonly buffer TokenIds { uint token_ids[]; };
layout(std430, binding = 2) writeonly buffer Output { float output_values[]; };
const uint VOCABULARY_SIZE = 1u;
const uint EMBEDDING_DIMENSION = 32u;
const uint TOKEN_COUNT = 1u;
const uint TOTAL_ELEMENTS = 32u;
const uint WEIGHT_ROW_BYTES = 34u;
uint read_u8(uint byte_offset) {
    uint word = weights[byte_offset >> 2u];
    return (word >> ((byte_offset & 3u) * 8u)) & 255u;
}
uint read_u16(uint byte_offset) {
    uint word = weights[byte_offset >> 2u];
    return (word >> ((byte_offset & 3u) * 8u)) & 65535u;
}
void main() {
    uint index = gl_GlobalInvocationID.x;
    if (index >= TOTAL_ELEMENTS) return;
    uint token_row = index / EMBEDDING_DIMENSION;
    uint dimension = index % EMBEDDING_DIMENSION;
    if (token_row >= TOKEN_COUNT) return;
    uint token = token_ids[token_row];
    if (token >= VOCABULARY_SIZE) return;
    uint block = dimension / 32u;
    uint within = dimension % 32u;
    uint block_offset = token * WEIGHT_ROW_BYTES + block * 34u;
    float scale = unpackHalf2x16(read_u16(block_offset)).x;
    uint raw = read_u8(block_offset + 2u + within);
    int quantized = raw >= 128u ? int(raw) - 256 : int(raw);
    output_values[index] = scale * float(quantized);
}
)glsl";

constexpr char kQ4_1EmbeddingWgsl[] = R"wgsl(
@group(0) @binding(0) var<storage, read> weights: array<u32>;
@group(0) @binding(1) var<storage, read> token_ids: array<u32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
const VOCABULARY_SIZE: u32 = 1u;
const EMBEDDING_DIMENSION: u32 = 32u;
const TOKEN_COUNT: u32 = 1u;
const TOTAL_ELEMENTS: u32 = 32u;
const WEIGHT_ROW_BYTES: u32 = 20u;
fn read_u8(byte_offset: u32) -> u32 {
    let word = weights[byte_offset >> 2u];
    return (word >> ((byte_offset & 3u) * 8u)) & 255u;
}
fn read_u16(byte_offset: u32) -> u32 {
    let word = weights[byte_offset >> 2u];
    return (word >> ((byte_offset & 3u) * 8u)) & 65535u;
}
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= TOTAL_ELEMENTS) { return; }
    let token_row = gid.x / EMBEDDING_DIMENSION;
    let dimension = gid.x % EMBEDDING_DIMENSION;
    if (token_row >= TOKEN_COUNT) { return; }
    let token = token_ids[token_row];
    if (token >= VOCABULARY_SIZE) { return; }
    let block = dimension / 32u;
    let within = dimension % 32u;
    let block_offset = token * WEIGHT_ROW_BYTES + block * 20u;
    let scale = unpack2x16float(vec2<u32>(read_u16(block_offset), 0u)).x;
    let minimum = unpack2x16float(vec2<u32>(read_u16(block_offset + 2u), 0u)).x;
    let packed = read_u8(block_offset + 4u + (within % 16u));
    let nibble = select(packed & 15u, packed >> 4u, within >= 16u);
    output[gid.x] = scale * f32(nibble) + minimum;
}
)wgsl";

constexpr char kQ4_1EmbeddingGlsl[] = R"glsl(#version 310 es
precision highp float;
precision highp int;
layout(local_size_x = 64) in;
layout(std430, binding = 0) readonly buffer Weights { uint weights[]; };
layout(std430, binding = 1) readonly buffer TokenIds { uint token_ids[]; };
layout(std430, binding = 2) writeonly buffer Output { float output_values[]; };
const uint VOCABULARY_SIZE = 1u;
const uint EMBEDDING_DIMENSION = 32u;
const uint TOKEN_COUNT = 1u;
const uint TOTAL_ELEMENTS = 32u;
const uint WEIGHT_ROW_BYTES = 20u;
uint read_u8(uint byte_offset) {
    uint word = weights[byte_offset >> 2u];
    return (word >> ((byte_offset & 3u) * 8u)) & 255u;
}
uint read_u16(uint byte_offset) {
    uint word = weights[byte_offset >> 2u];
    return (word >> ((byte_offset & 3u) * 8u)) & 65535u;
}
void main() {
    uint index = gl_GlobalInvocationID.x;
    if (index >= TOTAL_ELEMENTS) return;
    uint token_row = index / EMBEDDING_DIMENSION;
    uint dimension = index % EMBEDDING_DIMENSION;
    if (token_row >= TOKEN_COUNT) return;
    uint token = token_ids[token_row];
    if (token >= VOCABULARY_SIZE) return;
    uint block = dimension / 32u;
    uint within = dimension % 32u;
    uint block_offset = token * WEIGHT_ROW_BYTES + block * 20u;
    float scale = unpackHalf2x16(read_u16(block_offset)).x;
    float minimum = unpackHalf2x16(read_u16(block_offset + 2u)).x;
    uint packed = read_u8(block_offset + 4u + (within % 16u));
    uint nibble = within < 16u ? (packed & 15u) : (packed >> 4u);
    output_values[index] = scale * float(nibble) + minimum;
}
)glsl";

constexpr char kQ8_0DequantizeWgsl[] = R"wgsl(
@group(0) @binding(0) var<storage, read> encoded: array<u32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
const ELEMENT_COUNT: u32 = 32u;
fn read_u8(byte_offset: u32) -> u32 {
    let word = encoded[byte_offset >> 2u];
    let shift = (byte_offset & 3u) * 8u;
    return (word >> shift) & 255u;
}
fn read_u16(byte_offset: u32) -> u32 {
    let word = encoded[byte_offset >> 2u];
    let shift = (byte_offset & 3u) * 8u;
    return (word >> shift) & 65535u;
}
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= ELEMENT_COUNT) { return; }
    let block_index = gid.x / 32u;
    let within_block = gid.x % 32u;
    let block_offset = block_index * 34u;
    let scale_bits = read_u16(block_offset);
    let scale = unpack2x16float(vec2<u32>(scale_bits, 0u)).x;
    let raw = read_u8(block_offset + 2u + within_block);
    let signed_value = select(i32(raw), i32(raw) - 256, raw >= 128u);
    output[gid.x] = scale * f32(signed_value);
}
)wgsl";



bool is_device_contiguous_f32(const Tensor& tensor) {
    const TensorDesc& desc = tensor.descriptor();
    return desc.data_type == DataType::F32 &&
           desc.layout == TensorLayout::Contiguous &&
           desc.strides_bytes.empty() &&
           desc.memory != MemoryDomain::Host;
}

float half_to_float(uint16_t half) {
    const uint32_t sign = static_cast<uint32_t>(half & 0x8000u) << 16;
    uint32_t exponent = (half >> 10) & 0x1fu;
    uint32_t mantissa = half & 0x03ffu;
    uint32_t bits = 0;
    if (exponent == 0) {
        if (mantissa == 0) {
            bits = sign;
        } else {
            int shift = 0;
            while ((mantissa & 0x0400u) == 0) { mantissa <<= 1; ++shift; }
            mantissa &= 0x03ffu;
            const uint32_t float_exponent = static_cast<uint32_t>(127 - 14 - shift);
            bits = sign | (float_exponent << 23) | (mantissa << 13);
        }
    } else if (exponent == 0x1fu) {
        bits = sign | 0x7f800000u | (mantissa << 13);
    } else {
        exponent = exponent + (127 - 15);
        bits = sign | (exponent << 23) | (mantissa << 13);
    }
    float value = 0.0f;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

uint16_t read_u16_le(const uint8_t* source) {
    return static_cast<uint16_t>(source[0]) |
           static_cast<uint16_t>(static_cast<uint16_t>(source[1]) << 8);
}

} // namespace



Status decode_q5_0_blocks(const uint8_t* source, size_t source_bytes,
                          float* destination, size_t destination_elements) {
    constexpr size_t kElementsPerBlock = 32;
    constexpr size_t kBytesPerBlock = 22;
    if ((!source || !destination) && (source_bytes != 0 || destination_elements != 0))
        return {StatusCode::InvalidArgument, "Q5_0 decoder received a null buffer"};
    if (destination_elements == 0 || destination_elements % kElementsPerBlock != 0)
        return {StatusCode::InvalidArgument, "Q5_0 destination must contain complete blocks"};
    const size_t blocks = destination_elements / kElementsPerBlock;
    if (blocks > std::numeric_limits<size_t>::max() / kBytesPerBlock ||
        source_bytes != blocks * kBytesPerBlock)
        return {StatusCode::InvalidArgument, "Q5_0 source and destination sizes do not match complete blocks"};
    for (size_t block = 0; block < blocks; ++block) {
        const uint8_t* encoded = source + block * kBytesPerBlock;
        const float scale = half_to_float(read_u16_le(encoded));
        if (!std::isfinite(scale))
            return {StatusCode::NumericalFailure, "Q5_0 block has a non-finite FP16 scale"};
        const uint32_t high_bits = static_cast<uint32_t>(encoded[2]) |
            (static_cast<uint32_t>(encoded[3]) << 8) |
            (static_cast<uint32_t>(encoded[4]) << 16) |
            (static_cast<uint32_t>(encoded[5]) << 24);
        for (size_t i = 0; i < 16; ++i) {
            const uint8_t packed = encoded[6 + i];
            const int32_t low = static_cast<int32_t>(packed & 0x0fu) |
                                static_cast<int32_t>(((high_bits >> i) & 1u) << 4);
            const int32_t high = static_cast<int32_t>(packed >> 4) |
                                 static_cast<int32_t>(((high_bits >> (i + 16)) & 1u) << 4);
            destination[block * 32 + i] = scale * static_cast<float>(low - 16);
            destination[block * 32 + i + 16] = scale * static_cast<float>(high - 16);
        }
    }
    return Status::success();
}

Status decode_q5_1_blocks(const uint8_t* source, size_t source_bytes,
                          float* destination, size_t destination_elements) {
    constexpr size_t kElementsPerBlock = 32;
    constexpr size_t kBytesPerBlock = 24;
    if ((!source || !destination) && (source_bytes != 0 || destination_elements != 0))
        return {StatusCode::InvalidArgument, "Q5_1 decoder received a null buffer"};
    if (destination_elements == 0 || destination_elements % kElementsPerBlock != 0)
        return {StatusCode::InvalidArgument, "Q5_1 destination must contain complete blocks"};
    const size_t blocks = destination_elements / kElementsPerBlock;
    if (blocks > std::numeric_limits<size_t>::max() / kBytesPerBlock ||
        source_bytes != blocks * kBytesPerBlock)
        return {StatusCode::InvalidArgument, "Q5_1 source and destination sizes do not match complete blocks"};
    for (size_t block = 0; block < blocks; ++block) {
        const uint8_t* encoded = source + block * kBytesPerBlock;
        const float scale = half_to_float(read_u16_le(encoded));
        const float minimum = half_to_float(read_u16_le(encoded + 2));
        if (!std::isfinite(scale) || !std::isfinite(minimum))
            return {StatusCode::NumericalFailure, "Q5_1 block has a non-finite FP16 scale or minimum"};
        const uint32_t high_bits = static_cast<uint32_t>(encoded[4]) |
            (static_cast<uint32_t>(encoded[5]) << 8) |
            (static_cast<uint32_t>(encoded[6]) << 16) |
            (static_cast<uint32_t>(encoded[7]) << 24);
        for (size_t i = 0; i < 16; ++i) {
            const uint8_t packed = encoded[8 + i];
            const uint32_t low = static_cast<uint32_t>(packed & 0x0fu) |
                                 (((high_bits >> i) & 1u) << 4);
            const uint32_t high = static_cast<uint32_t>(packed >> 4) |
                                  (((high_bits >> (i + 16)) & 1u) << 4);
            destination[block * 32 + i] = scale * static_cast<float>(low) + minimum;
            destination[block * 32 + i + 16] = scale * static_cast<float>(high) + minimum;
        }
    }
    return Status::success();
}

Status decode_q4_0_blocks(const uint8_t* source, size_t source_bytes,
                          float* destination, size_t destination_elements) {
    constexpr size_t kElementsPerBlock = 32;
    constexpr size_t kBytesPerBlock = 18;
    if ((!source || !destination) && (source_bytes != 0 || destination_elements != 0))
        return {StatusCode::InvalidArgument, "Q4_0 decoder received a null buffer"};
    if (destination_elements == 0 || destination_elements % kElementsPerBlock != 0)
        return {StatusCode::InvalidArgument, "Q4_0 destination must contain complete blocks"};
    const size_t blocks = destination_elements / kElementsPerBlock;
    if (blocks > std::numeric_limits<size_t>::max() / kBytesPerBlock ||
        source_bytes != blocks * kBytesPerBlock)
        return {StatusCode::InvalidArgument, "Q4_0 source and destination sizes do not match complete blocks"};
    for (size_t block = 0; block < blocks; ++block) {
        const uint8_t* encoded = source + block * kBytesPerBlock;
        const float scale = half_to_float(read_u16_le(encoded));
        if (!std::isfinite(scale))
            return {StatusCode::NumericalFailure, "Q4_0 block has a non-finite FP16 scale"};
        for (size_t i = 0; i < 16; ++i) {
            const uint8_t packed = encoded[2 + i];
            destination[block * 32 + i] =
                scale * static_cast<float>(static_cast<int32_t>(packed & 0x0fu) - 8);
            destination[block * 32 + i + 16] =
                scale * static_cast<float>(static_cast<int32_t>(packed >> 4) - 8);
        }
    }
    return Status::success();
}

Status decode_q4_1_blocks(const uint8_t* source, size_t source_bytes,
                          float* destination, size_t destination_elements) {
    constexpr size_t kElementsPerBlock = 32;
    constexpr size_t kBytesPerBlock = 20;
    if ((!source || !destination) && (source_bytes != 0 || destination_elements != 0))
        return {StatusCode::InvalidArgument, "Q4_1 decoder received a null buffer"};
    if (destination_elements == 0 || destination_elements % kElementsPerBlock != 0)
        return {StatusCode::InvalidArgument, "Q4_1 destination must contain complete blocks"};
    const size_t blocks = destination_elements / kElementsPerBlock;
    if (blocks > std::numeric_limits<size_t>::max() / kBytesPerBlock ||
        source_bytes != blocks * kBytesPerBlock)
        return {StatusCode::InvalidArgument, "Q4_1 source and destination sizes do not match complete blocks"};
    for (size_t block = 0; block < blocks; ++block) {
        const uint8_t* encoded = source + block * kBytesPerBlock;
        const float scale = half_to_float(read_u16_le(encoded));
        const float minimum = half_to_float(read_u16_le(encoded + 2));
        if (!std::isfinite(scale) || !std::isfinite(minimum))
            return {StatusCode::NumericalFailure, "Q4_1 block has a non-finite FP16 scale or minimum"};
        for (size_t i = 0; i < 16; ++i) {
            const uint8_t packed = encoded[4 + i];
            destination[block * 32 + i] =
                scale * static_cast<float>(packed & 0x0fu) + minimum;
            destination[block * 32 + i + 16] =
                scale * static_cast<float>(packed >> 4) + minimum;
        }
    }
    return Status::success();
}

Status decode_q8_0_blocks(const uint8_t* source, size_t source_bytes,
                          float* destination, size_t destination_elements) {
    constexpr size_t kElementsPerBlock = 32;
    constexpr size_t kBytesPerBlock = 34;
    if ((!source || !destination) && (source_bytes != 0 || destination_elements != 0))
        return {StatusCode::InvalidArgument, "Q8_0 decoder received a null buffer"};
    if (destination_elements == 0 || destination_elements % kElementsPerBlock != 0)
        return {StatusCode::InvalidArgument, "Q8_0 destination size must contain complete blocks"};
    const size_t block_count = destination_elements / kElementsPerBlock;
    if (block_count > std::numeric_limits<size_t>::max() / kBytesPerBlock ||
        source_bytes != block_count * kBytesPerBlock)
        return {StatusCode::InvalidArgument, "Q8_0 source and destination sizes do not match complete blocks"};

    for (size_t block = 0; block < block_count; ++block) {
        const uint8_t* encoded = source + block * kBytesPerBlock;
        const float scale = half_to_float(read_u16_le(encoded));
        if (!std::isfinite(scale))
            return {StatusCode::NumericalFailure, "Q8_0 block has a non-finite FP16 scale"};
        for (size_t i = 0; i < kElementsPerBlock; ++i) {
            const uint8_t raw = encoded[2 + i];
            const int32_t quantized = raw < 128u ? static_cast<int32_t>(raw)
                                                  : static_cast<int32_t>(raw) - 256;
            destination[block * kElementsPerBlock + i] = scale * static_cast<float>(quantized);
        }
    }
    return Status::success();
}






const char* q8_0_matmul_wgsl_source() noexcept { return kQ8_0MatMulWgsl; }
const char* q4_0_matmul_wgsl_source() noexcept { return kQ4_0MatMulWgsl; }
const char* q4_0_matmul_glsl_source() noexcept { return kQ4_0MatMulGlsl; }
const char* q4_0_embedding_wgsl_source() noexcept { return kQ4_0EmbeddingWgsl; }
const char* q4_0_embedding_glsl_source() noexcept { return kQ4_0EmbeddingGlsl; }
const char* q8_0_embedding_wgsl_source() noexcept { return kQ8_0EmbeddingWgsl; }
const char* q8_0_embedding_glsl_source() noexcept { return kQ8_0EmbeddingGlsl; }
const char* q4_1_embedding_wgsl_source() noexcept { return kQ4_1EmbeddingWgsl; }
const char* q4_1_embedding_glsl_source() noexcept { return kQ4_1EmbeddingGlsl; }
const char* q4_1_matmul_wgsl_source() noexcept { return kQ4_1MatMulWgsl; }
const char* q4_1_matmul_glsl_source() noexcept { return kQ4_1MatMulGlsl; }

Status build_q8_0_matmul_candidate(const Tensor& activations,
                                  const std::shared_ptr<Buffer>& quantized_weights,
                                  uint64_t weight_bytes,
                                  const Tensor& output,
                                  ComputeDispatch& dispatch) {
    dispatch = ComputeDispatch{};
    if (!quantized_weights || !is_device_contiguous_f32(activations) ||
        !is_device_contiguous_f32(output) || !activations.buffer() || !output.buffer() ||
        activations.descriptor().shape.size() != 2 ||
        (output.descriptor().shape.size() != 2 && output.descriptor().shape.size() != 3))
        return {StatusCode::Unsupported, "Q8_0 MatMul requires F32 [M,K] activations and F32 [M,N] or [M,H,D] output"};
    const auto& a = activations.descriptor().shape;
    const auto& o = output.descriptor().shape;
    uint64_t output_channels = 0;
    if (o.size() == 2) output_channels = o[1];
    else if (o[1] <= UINT64_MAX / o[2]) output_channels = o[1] * o[2];
    if (a[0] == 0 || a[1] == 0 || a[1] % 32 != 0 ||
        o[0] != a[0] || output_channels == 0 || a[0] > UINT32_MAX ||
        a[1] > UINT32_MAX || output_channels > UINT32_MAX ||
        activations.element_count() > UINT32_MAX || output.element_count() > UINT32_MAX)
        return {StatusCode::InvalidArgument, "Q8_0 MatMul dimensions are invalid or inner dimension is not block-aligned"};
    const uint64_t blocks_per_row = a[1] / 32;
    if (blocks_per_row > UINT64_MAX / 34 ||
        output_channels > UINT64_MAX / (blocks_per_row * 34))
        return {StatusCode::InvalidArgument, "Q8_0 weight byte size overflows"};
    const uint64_t expected_bytes = output_channels * blocks_per_row * 34;
    if (weight_bytes != expected_bytes || weight_bytes > static_cast<uint64_t>(UINT32_MAX) - 3)
        return {StatusCode::InvalidArgument, "Q8_0 weight byte count does not match [N,K] shape"};
    const uint64_t padded_bytes = (weight_bytes + 3) & ~uint64_t{3};
    if (quantized_weights->size_bytes() < padded_bytes ||
        output.byte_size() != output.element_count() * sizeof(float) ||
        output.buffer().get() == activations.buffer().get() ||
        output.buffer().get() == quantized_weights.get())
        return {StatusCode::InvalidArgument, "Q8_0 MatMul buffer ranges are inconsistent"};
    dispatch.label = "q8-0-matmul-f32-activation-generic-unverified";
    dispatch.kernel_id = "dawn-q8-0-matmul-f32-activation-v1";
    dispatch.backend = BackendKind::DawnWebGpu;
    dispatch.wgsl_source = kQ8_0MatMulWgsl;
    dispatch.workgroup_size_x = 8;
    dispatch.workgroup_size_y = 8;
    const std::pair<std::string, std::string> constants[] = {
        {"const ROWS: u32 = 1u;", "const ROWS: u32 = " + std::to_string(a[0]) + "u;"},
        {"const OUTPUT_CHANNELS: u32 = 1u;", "const OUTPUT_CHANNELS: u32 = " + std::to_string(output_channels) + "u;"},
        {"const INNER: u32 = 32u;", "const INNER: u32 = " + std::to_string(a[1]) + "u;"},
        {"const WEIGHT_ROW_BYTES: u32 = 34u;", "const WEIGHT_ROW_BYTES: u32 = " +
            std::to_string(blocks_per_row * 34) + "u;"},
    };
    for (const auto& constant : constants) {
        const size_t position = dispatch.wgsl_source.find(constant.first);
        if (position == std::string::npos)
            return {StatusCode::BackendFailure, "Q8_0 MatMul WGSL placeholder is missing"};
        dispatch.wgsl_source.replace(position, constant.first.size(), constant.second);
    }
    dispatch.workgroups_x = static_cast<uint32_t>((output_channels + 7) / 8);
    dispatch.workgroups_y = static_cast<uint32_t>((a[0] + 7) / 8);
    dispatch.bindings = {
        {0, activations.buffer(), activations.byte_offset(), activations.byte_size()},
        {1, quantized_weights, 0, padded_bytes},
        {2, output.buffer(), output.byte_offset(), output.byte_size()},
    };
    return Status::success();
}

Status build_q4_0_matmul_candidate(const Tensor& activations,
                                  const std::shared_ptr<Buffer>& quantized_weights,
                                  uint64_t weight_bytes,
                                  const Tensor& output,
                                  ComputeDispatch& dispatch) {
    dispatch = ComputeDispatch{};
    if (!quantized_weights || !is_device_contiguous_f32(activations) ||
        !is_device_contiguous_f32(output) || !activations.buffer() || !output.buffer() ||
        activations.descriptor().shape.size() != 2 ||
        (output.descriptor().shape.size() != 2 && output.descriptor().shape.size() != 3))
        return {StatusCode::Unsupported, "Q4_0 MatMul requires F32 [M,K] activations and F32 [M,N] or [M,H,D] output"};
    const auto& a = activations.descriptor().shape;
    const auto& o = output.descriptor().shape;
    uint64_t output_channels = 0;
    if (o.size() == 2) output_channels = o[1];
    else if (o[1] <= UINT64_MAX / o[2]) output_channels = o[1] * o[2];
    if (a[0] == 0 || a[1] == 0 || a[1] % 32 != 0 ||
        o[0] != a[0] || output_channels == 0 || a[0] > UINT32_MAX ||
        a[1] > UINT32_MAX || output_channels > UINT32_MAX ||
        activations.element_count() > UINT32_MAX || output.element_count() > UINT32_MAX)
        return {StatusCode::InvalidArgument, "Q4_0 MatMul dimensions are invalid or inner dimension is not block-aligned"};
    const uint64_t blocks_per_row = a[1] / 32;
    if (blocks_per_row > UINT64_MAX / 18 ||
        output_channels > UINT64_MAX / (blocks_per_row * 18))
        return {StatusCode::InvalidArgument, "Q4_0 weight byte size overflows"};
    const uint64_t expected_bytes = output_channels * blocks_per_row * 18;
    if (weight_bytes != expected_bytes || weight_bytes > static_cast<uint64_t>(UINT32_MAX) - 3)
        return {StatusCode::InvalidArgument, "Q4_0 weight byte count does not match [N,K] shape"};
    const uint64_t padded_bytes = (weight_bytes + 3) & ~uint64_t{3};
    if (quantized_weights->size_bytes() < padded_bytes ||
        output.byte_size() != output.element_count() * sizeof(float) ||
        output.buffer().get() == activations.buffer().get() ||
        output.buffer().get() == quantized_weights.get())
        return {StatusCode::InvalidArgument, "Q4_0 MatMul buffer ranges are inconsistent"};
    dispatch.label = "q4-0-matmul-f32-activation-generic-unverified";
    dispatch.kernel_id = "dawn-q4-0-matmul-f32-activation-v1";
    dispatch.backend = BackendKind::DawnWebGpu;
    dispatch.wgsl_source = kQ4_0MatMulWgsl;
    dispatch.workgroup_size_x = 8;
    dispatch.workgroup_size_y = 8;
    const std::pair<std::string, std::string> constants[] = {
        {"const ROWS: u32 = 1u;", "const ROWS: u32 = " + std::to_string(a[0]) + "u;"},
        {"const OUTPUT_CHANNELS: u32 = 1u;", "const OUTPUT_CHANNELS: u32 = " + std::to_string(output_channels) + "u;"},
        {"const INNER: u32 = 32u;", "const INNER: u32 = " + std::to_string(a[1]) + "u;"},
        {"const WEIGHT_ROW_BYTES: u32 = 18u;", "const WEIGHT_ROW_BYTES: u32 = " +
            std::to_string(blocks_per_row * 18) + "u;"},
    };
    for (const auto& constant : constants) {
        const size_t position = dispatch.wgsl_source.find(constant.first);
        if (position == std::string::npos)
            return {StatusCode::BackendFailure, "Q4_0 MatMul WGSL placeholder is missing"};
        dispatch.wgsl_source.replace(position, constant.first.size(), constant.second);
    }
    dispatch.workgroups_x = static_cast<uint32_t>((output_channels + 7) / 8);
    dispatch.workgroups_y = static_cast<uint32_t>((a[0] + 7) / 8);
    dispatch.bindings = {
        {0, activations.buffer(), activations.byte_offset(), activations.byte_size()},
        {1, quantized_weights, 0, padded_bytes},
        {2, output.buffer(), output.byte_offset(), output.byte_size()},
    };
    return Status::success();
}

const char* q8_0_matmul_glsl_source() noexcept { return kQ8_0MatMulGlsl; }



Status build_q4_1_matmul_candidate(const Tensor& activations,
                                  const std::shared_ptr<Buffer>& quantized_weights,
                                  uint64_t weight_bytes,
                                  const Tensor& output,
                                  ComputeDispatch& dispatch) {
    dispatch = ComputeDispatch{};
    if (!quantized_weights || !is_device_contiguous_f32(activations) ||
        !is_device_contiguous_f32(output) || !activations.buffer() || !output.buffer() ||
        activations.descriptor().shape.size() != 2 ||
        (output.descriptor().shape.size() != 2 && output.descriptor().shape.size() != 3))
        return {StatusCode::Unsupported, "Q4_1 MatMul requires F32 [M,K] activations and F32 [M,N] or [M,H,D] output"};
    const auto& a = activations.descriptor().shape;
    const auto& o = output.descriptor().shape;
    uint64_t output_channels = 0;
    if (o.size() == 2) output_channels = o[1];
    else if (o[1] <= UINT64_MAX / o[2]) output_channels = o[1] * o[2];
    if (a[0] == 0 || a[1] == 0 || a[1] % 32 != 0 ||
        o[0] != a[0] || output_channels == 0 || a[0] > UINT32_MAX ||
        a[1] > UINT32_MAX || output_channels > UINT32_MAX ||
        activations.element_count() > UINT32_MAX || output.element_count() > UINT32_MAX)
        return {StatusCode::InvalidArgument, "Q4_1 MatMul dimensions are invalid or inner dimension is not block-aligned"};
    const uint64_t blocks_per_row = a[1] / 32;
    if (blocks_per_row > UINT64_MAX / 20 ||
        output_channels > UINT64_MAX / (blocks_per_row * 20))
        return {StatusCode::InvalidArgument, "Q4_1 weight byte size overflows"};
    const uint64_t expected_bytes = output_channels * blocks_per_row * 20;
    if (weight_bytes != expected_bytes || weight_bytes > static_cast<uint64_t>(UINT32_MAX) - 3)
        return {StatusCode::InvalidArgument, "Q4_1 weight byte count does not match [N,K] shape"};
    const uint64_t padded_bytes = (weight_bytes + 3) & ~uint64_t{3};
    if (quantized_weights->size_bytes() < padded_bytes ||
        output.byte_size() != output.element_count() * sizeof(float) ||
        output.buffer().get() == activations.buffer().get() ||
        output.buffer().get() == quantized_weights.get())
        return {StatusCode::InvalidArgument, "Q4_1 MatMul buffer ranges are inconsistent"};
    dispatch.label = "q4-1-matmul-f32-activation-generic-unverified";
    dispatch.kernel_id = "dawn-q4-1-matmul-f32-activation-v1";
    dispatch.backend = BackendKind::DawnWebGpu;
    dispatch.wgsl_source = kQ4_1MatMulWgsl;
    dispatch.workgroup_size_x = 8;
    dispatch.workgroup_size_y = 8;
    const std::pair<std::string, std::string> constants[] = {
        {"const ROWS: u32 = 1u;", "const ROWS: u32 = " + std::to_string(a[0]) + "u;"},
        {"const OUTPUT_CHANNELS: u32 = 1u;", "const OUTPUT_CHANNELS: u32 = " + std::to_string(output_channels) + "u;"},
        {"const INNER: u32 = 32u;", "const INNER: u32 = " + std::to_string(a[1]) + "u;"},
        {"const WEIGHT_ROW_BYTES: u32 = 20u;", "const WEIGHT_ROW_BYTES: u32 = " +
            std::to_string(blocks_per_row * 20) + "u;"},
    };
    for (const auto& constant : constants) {
        const size_t position = dispatch.wgsl_source.find(constant.first);
        if (position == std::string::npos)
            return {StatusCode::BackendFailure, "Q4_1 MatMul WGSL placeholder is missing"};
        dispatch.wgsl_source.replace(position, constant.first.size(), constant.second);
    }
    dispatch.workgroups_x = static_cast<uint32_t>((output_channels + 7) / 8);
    dispatch.workgroups_y = static_cast<uint32_t>((a[0] + 7) / 8);
    dispatch.bindings = {
        {0, activations.buffer(), activations.byte_offset(), activations.byte_size()},
        {1, quantized_weights, 0, padded_bytes},
        {2, output.buffer(), output.byte_offset(), output.byte_size()},
    };
    return Status::success();
}






Status validate_q8_0_matmul_kernel(Backend& backend, OperatorRegistry& operators,
                                   QuantizationRegistry& quantizations) {
    const BackendKind backend_kind = backend.kind();
    if ((backend_kind != BackendKind::DawnWebGpu && backend_kind != BackendKind::OpenGles) ||
        !backend.healthy())
        return {StatusCode::NotReady, "Q8_0 MatMul validation requires a healthy Dawn or GLES backend"};
    constexpr uint32_t kQ8_0TypeId = 8;
    auto format = quantizations.formats_.find(kQ8_0TypeId);
    if (format == quantizations.formats_.end() || format->second.decoder_state != SupportState::Validated)
        return {StatusCode::NotReady, "Q8_0 host decoder must be validated before GPU kernel promotion"};
    const auto backend_validation = format->second.backend_gpu_kernel_states.find(backend_kind);
    if (backend_validation != format->second.backend_gpu_kernel_states.end() &&
        backend_validation->second == SupportState::Validated)
        return Status::success();

    const bool use_gles = backend_kind == BackendKind::OpenGles;
    const std::string kernel_id = use_gles ? "gles-q8-0-matmul-f32-activation-v1"
                                           : "dawn-q8-0-matmul-f32-activation-v1";
    auto kernel = std::find_if(operators.kernels_.begin(), operators.kernels_.end(),
        [&](const KernelDefinition& definition) {
            return definition.kernel_id == kernel_id && definition.backend == backend_kind;
        });
    if (kernel == operators.kernels_.end())
        return {StatusCode::NotReady, "Q8_0 MatMul candidate is not registered for the selected backend"};
    if (kernel->validation == KernelValidation::Validated) {
        format->second.backend_gpu_kernel_states[backend_kind] = SupportState::Validated;
        format->second.gpu_kernel_state = SupportState::Validated;
        return Status::success();
    }
    if (kernel->validation != KernelValidation::Unverified)
        return {StatusCode::Unsupported, "Q8_0 MatMul candidate is rejected"};
    const std::string& registered_source = use_gles ? kernel->glsl_source : kernel->wgsl_source;
    if (registered_source.empty())
        return {StatusCode::Unsupported, "Q8_0 MatMul candidate has no source for its selected backend"};

    constexpr uint32_t rows = 2, output_channels = 2, inner = 64;
    std::vector<float> activations(rows * inner);
    for (size_t i = 0; i < activations.size(); ++i)
        activations[i] = (static_cast<int>(i % 9) - 4) * 0.25f;
    const uint64_t weight_bytes = static_cast<uint64_t>(output_channels) * (inner / 32) * 34;
    std::vector<uint8_t> encoded(static_cast<size_t>(weight_bytes), 0);
    for (uint32_t block = 0; block < output_channels * (inner / 32); ++block) {
        const size_t offset = static_cast<size_t>(block) * 34;
        encoded[offset] = 0x00;
        encoded[offset + 1] = 0x38; // FP16 scale 0.5
        for (uint32_t i = 0; i < 32; ++i)
            encoded[offset + 2 + i] = static_cast<uint8_t>(static_cast<int8_t>(static_cast<int>(i % 11) - 5));
    }
    float expected[rows * output_channels] = {};
    float actual[rows * output_channels] = {};
    Status status = reference_q8_0_matmul(activations.data(), encoded.data(), weight_bytes,
        expected, rows, output_channels, inner);
    if (!status.ok()) return status;

    BufferDesc activation_desc{activations.size() * sizeof(float), alignof(float), MemoryDomain::Device,
        false, "q8-validation-activations"};
    BufferDesc weights_desc{(weight_bytes + 3) & ~uint64_t{3}, 4, MemoryDomain::Device,
        true, "q8-validation-weights"};
    BufferDesc output_desc{sizeof(actual), alignof(float), MemoryDomain::Device,
        false, "q8-validation-output"};
    std::shared_ptr<Buffer> activation_buffer, weights_buffer, output_buffer;
    status = backend.allocate(activation_desc, activation_buffer); if (!status.ok()) return status;
    status = backend.allocate(weights_desc, weights_buffer); if (!status.ok()) return status;
    status = backend.allocate(output_desc, output_buffer); if (!status.ok()) return status;
    status = backend.upload(activation_buffer, 0, activations.data(), activations.size() * sizeof(float));
    if (!status.ok()) return status;
    status = backend.upload(weights_buffer, 0, encoded.data(), weight_bytes);
    if (!status.ok()) return status;

    TensorDesc activation_tensor_desc; activation_tensor_desc.shape = {rows, inner};
    activation_tensor_desc.data_type = DataType::F32;
    TensorDesc output_tensor_desc; output_tensor_desc.shape = {rows, output_channels};
    output_tensor_desc.data_type = DataType::F32;
    Tensor activation_tensor, output_tensor;
    status = Tensor::create(activation_tensor_desc, activation_buffer, 0, activation_tensor);
    if (!status.ok()) return status;
    status = Tensor::create(output_tensor_desc, output_buffer, 0, output_tensor);
    if (!status.ok()) return status;

    ComputeDispatch dispatch;
    status = use_gles
        ? build_q8_0_matmul_gles_candidate(activation_tensor, weights_buffer, weight_bytes, output_tensor, dispatch)
        : build_q8_0_matmul_candidate(activation_tensor, weights_buffer, weight_bytes, output_tensor, dispatch);
    if (!status.ok()) return status;
    const std::string& candidate_source = use_gles ? dispatch.glsl_source : dispatch.wgsl_source;
    const std::string& canonical_source = use_gles ? q8_0_matmul_glsl_source()
                                                    : q8_0_matmul_wgsl_source();
    if (dispatch.kernel_id != kernel_id || registered_source != canonical_source ||
        candidate_source.empty())
        return {StatusCode::InvalidArgument, "Q8_0 MatMul validation source differs from the registered canonical candidate"};
    dispatch.validation = KernelValidation::Validated;
    std::unique_ptr<CommandBatch> commands;
    status = backend.begin_commands(commands); if (!status.ok()) return status;
    status = commands->dispatch_compute(dispatch); if (!status.ok()) return status;
    status = backend.submit(std::move(commands)); if (!status.ok()) return status;
    status = backend.synchronize(); if (!status.ok()) return status;
    status = backend.readback(output_buffer, 0, actual, sizeof(actual)); if (!status.ok()) return status;
    for (size_t i = 0; i < rows * output_channels; ++i) {
        const float tolerance = 1.0e-4f * std::max(1.0f, std::fabs(expected[i]));
        if (!std::isfinite(actual[i]) || std::fabs(actual[i] - expected[i]) > tolerance)
            return {StatusCode::NumericalFailure, "Q8_0 MatMul GPU output differs from the independent block-decoder reference"};
    }
    kernel->validation = KernelValidation::Validated;
    format->second.backend_gpu_kernel_states[backend_kind] = SupportState::Validated;
    format->second.gpu_kernel_state = SupportState::Validated;
    return Status::success();
}


Status build_q8_0_embedding_candidate(const std::shared_ptr<Buffer>& quantized_weights,
                                      uint64_t weight_bytes, uint32_t vocabulary_size,
                                      uint32_t embedding_dimension,
                                      const std::shared_ptr<Buffer>& token_ids_buffer,
                                      const std::vector<uint32_t>& token_ids,
                                      const Tensor& output, ComputeDispatch& dispatch) {
    dispatch = ComputeDispatch{};
    if (!quantized_weights || !token_ids_buffer || !output.buffer() ||
        quantized_weights->memory_domain() == MemoryDomain::Host ||
        token_ids_buffer->memory_domain() == MemoryDomain::Host ||
        output.descriptor().data_type != DataType::F32 ||
        output.descriptor().layout != TensorLayout::Contiguous ||
        output.descriptor().memory == MemoryDomain::Host ||
        output.descriptor().shape.size() != 2 || token_ids.empty())
        return {StatusCode::Unsupported, "Q8_0 Embedding requires packed weights, device token IDs and contiguous F32 output"};
    if (vocabulary_size == 0 || embedding_dimension == 0 || embedding_dimension % 32 != 0 ||
        token_ids.size() != output.descriptor().shape[0] ||
        output.descriptor().shape[1] != embedding_dimension ||
        token_ids.size() > UINT32_MAX || output.element_count() > UINT32_MAX)
        return {StatusCode::InvalidArgument, "Q8_0 Embedding dimensions or token count are invalid"};
    for (uint32_t token : token_ids)
        if (token >= vocabulary_size) return {StatusCode::InvalidArgument, "Q8_0 Embedding token ID is outside vocabulary"};
    const uint64_t row_bytes = static_cast<uint64_t>(embedding_dimension / 32) * 34;
    if (row_bytes > UINT64_MAX / vocabulary_size || weight_bytes != row_bytes * vocabulary_size ||
        weight_bytes > static_cast<uint64_t>(UINT32_MAX) - 3)
        return {StatusCode::InvalidArgument, "Q8_0 Embedding weight byte size does not match [vocabulary,dimension]"};
    const uint64_t padded_weights = (weight_bytes + 3) & ~uint64_t{3};
    const uint64_t token_bytes = token_ids.size() * sizeof(uint32_t);
    if (quantized_weights->size_bytes() < padded_weights ||
        token_ids_buffer->size_bytes() < token_bytes ||
        output.byte_size() != output.element_count() * sizeof(float) ||
        output.buffer().get() == quantized_weights.get() ||
        output.buffer().get() == token_ids_buffer.get() ||
        quantized_weights.get() == token_ids_buffer.get())
        return {StatusCode::InvalidArgument, "Q8_0 Embedding buffer ranges are inconsistent"};
    dispatch.label = "q8-0-embedding-f32-activation-generic-unverified";
    dispatch.kernel_id = "dawn-q8-0-embedding-f32-v1";
    dispatch.backend = BackendKind::DawnWebGpu;
    dispatch.wgsl_source = kQ8_0EmbeddingWgsl;
    dispatch.workgroup_size_x = 64;
    const std::pair<std::string, std::string> constants[] = {
        {"const VOCABULARY_SIZE: u32 = 1u;", "const VOCABULARY_SIZE: u32 = " + std::to_string(vocabulary_size) + "u;"},
        {"const EMBEDDING_DIMENSION: u32 = 32u;", "const EMBEDDING_DIMENSION: u32 = " + std::to_string(embedding_dimension) + "u;"},
        {"const TOKEN_COUNT: u32 = 1u;", "const TOKEN_COUNT: u32 = " + std::to_string(token_ids.size()) + "u;"},
        {"const TOTAL_ELEMENTS: u32 = 32u;", "const TOTAL_ELEMENTS: u32 = " + std::to_string(output.element_count()) + "u;"},
        {"const WEIGHT_ROW_BYTES: u32 = 34u;", "const WEIGHT_ROW_BYTES: u32 = " + std::to_string(row_bytes) + "u;"}
    };
    for (const auto& constant : constants) {
        const size_t pos = dispatch.wgsl_source.find(constant.first);
        if (pos == std::string::npos) return {StatusCode::BackendFailure, "Q8_0 Embedding WGSL placeholder missing"};
        dispatch.wgsl_source.replace(pos, constant.first.size(), constant.second);
    }
    dispatch.workgroups_x = static_cast<uint32_t>((output.element_count() + 63) / 64);
    dispatch.bindings = {
        {0, quantized_weights, 0, padded_weights},
        {1, token_ids_buffer, 0, token_bytes},
        {2, output.buffer(), output.byte_offset(), output.byte_size()}
    };
    return Status::success();
}

Status build_q8_0_embedding_gles_candidate(const std::shared_ptr<Buffer>& quantized_weights,
                                           uint64_t weight_bytes, uint32_t vocabulary_size,
                                           uint32_t embedding_dimension,
                                           const std::shared_ptr<Buffer>& token_ids_buffer,
                                           const std::vector<uint32_t>& token_ids,
                                           const Tensor& output, ComputeDispatch& dispatch) {
    Status status = build_q8_0_embedding_candidate(quantized_weights, weight_bytes,
        vocabulary_size, embedding_dimension, token_ids_buffer, token_ids, output, dispatch);
    if (!status.ok()) return status;
    dispatch.backend = BackendKind::OpenGles;
    dispatch.kernel_id = "gles-q8-0-embedding-f32-v1";
    dispatch.label = "gles-q8-0-embedding-f32-unverified";
    dispatch.wgsl_source.clear();
    dispatch.glsl_source = kQ8_0EmbeddingGlsl;
    const uint64_t row_bytes = static_cast<uint64_t>(embedding_dimension / 32) * 34;
    const std::pair<std::string, std::string> constants[] = {
        {"const uint VOCABULARY_SIZE = 1u;", "const uint VOCABULARY_SIZE = " + std::to_string(vocabulary_size) + "u;"},
        {"const uint EMBEDDING_DIMENSION = 32u;", "const uint EMBEDDING_DIMENSION = " + std::to_string(embedding_dimension) + "u;"},
        {"const uint TOKEN_COUNT = 1u;", "const uint TOKEN_COUNT = " + std::to_string(token_ids.size()) + "u;"},
        {"const uint TOTAL_ELEMENTS = 32u;", "const uint TOTAL_ELEMENTS = " + std::to_string(output.element_count()) + "u;"},
        {"const uint WEIGHT_ROW_BYTES = 34u;", "const uint WEIGHT_ROW_BYTES = " + std::to_string(row_bytes) + "u;"}
    };
    for (const auto& constant : constants) {
        const size_t pos = dispatch.glsl_source.find(constant.first);
        if (pos == std::string::npos) return {StatusCode::BackendFailure, "Q8_0 Embedding GLSL placeholder missing"};
        dispatch.glsl_source.replace(pos, constant.first.size(), constant.second);
    }
    return Status::success();
}

Status reference_q8_0_embedding(const uint8_t* quantized_weights, uint64_t weight_bytes,
                                uint32_t vocabulary_size, uint32_t embedding_dimension,
                                const std::vector<uint32_t>& token_ids, float* output) {
    if ((!quantized_weights || !output) && !token_ids.empty())
        return {StatusCode::InvalidArgument, "Q8_0 Embedding reference received a null pointer"};
    if (vocabulary_size == 0 || embedding_dimension == 0 || embedding_dimension % 32 != 0 ||
        token_ids.empty())
        return {StatusCode::InvalidArgument, "Q8_0 Embedding dimensions and token list must be non-empty and block-aligned"};
    const uint64_t row_bytes = static_cast<uint64_t>(embedding_dimension / 32) * 34;
    if (row_bytes > UINT64_MAX / vocabulary_size || weight_bytes != row_bytes * vocabulary_size)
        return {StatusCode::InvalidArgument, "Q8_0 Embedding weight byte size is inconsistent"};
    std::vector<float> decoded(embedding_dimension);
    for (size_t row = 0; row < token_ids.size(); ++row) {
        if (token_ids[row] >= vocabulary_size)
            return {StatusCode::InvalidArgument, "Q8_0 Embedding token ID is outside vocabulary"};
        Status status = decode_q8_0_blocks(quantized_weights + token_ids[row] * row_bytes,
            row_bytes, decoded.data(), decoded.size());
        if (!status.ok()) return status;
        std::copy(decoded.begin(), decoded.end(), output + row * embedding_dimension);
    }
    return Status::success();
}

Status build_q4_0_embedding_candidate(const std::shared_ptr<Buffer>& quantized_weights,
                                      uint64_t weight_bytes, uint32_t vocabulary_size,
                                      uint32_t embedding_dimension,
                                      const std::shared_ptr<Buffer>& token_ids_buffer,
                                      const std::vector<uint32_t>& token_ids,
                                      const Tensor& output, ComputeDispatch& dispatch) {
    dispatch = ComputeDispatch{};
    if (!quantized_weights || !token_ids_buffer || !output.buffer() ||
        quantized_weights->memory_domain() == MemoryDomain::Host ||
        token_ids_buffer->memory_domain() == MemoryDomain::Host ||
        output.descriptor().data_type != DataType::F32 ||
        output.descriptor().layout != TensorLayout::Contiguous ||
        output.descriptor().memory == MemoryDomain::Host ||
        output.descriptor().shape.size() != 2 || token_ids.empty())
        return {StatusCode::Unsupported, "Q4_0 Embedding requires packed weights, device token IDs and contiguous F32 output"};
    if (vocabulary_size == 0 || embedding_dimension == 0 || embedding_dimension % 32 != 0 ||
        token_ids.size() != output.descriptor().shape[0] ||
        output.descriptor().shape[1] != embedding_dimension ||
        token_ids.size() > UINT32_MAX || output.element_count() > UINT32_MAX)
        return {StatusCode::InvalidArgument, "Q4_0 Embedding dimensions or token count are invalid"};
    for (uint32_t token : token_ids)
        if (token >= vocabulary_size) return {StatusCode::InvalidArgument, "Q4_0 Embedding token ID is outside vocabulary"};
    const uint64_t row_bytes = static_cast<uint64_t>(embedding_dimension / 32) * 18;
    if (row_bytes > UINT64_MAX / vocabulary_size || weight_bytes != row_bytes * vocabulary_size ||
        weight_bytes > static_cast<uint64_t>(UINT32_MAX) - 3)
        return {StatusCode::InvalidArgument, "Q4_0 Embedding weight byte size does not match [vocabulary,dimension]"};
    const uint64_t padded_weights = (weight_bytes + 3) & ~uint64_t{3};
    const uint64_t token_bytes = token_ids.size() * sizeof(uint32_t);
    if (quantized_weights->size_bytes() < padded_weights ||
        token_ids_buffer->size_bytes() < token_bytes ||
        output.byte_size() != output.element_count() * sizeof(float) ||
        output.buffer().get() == quantized_weights.get() ||
        output.buffer().get() == token_ids_buffer.get() ||
        quantized_weights.get() == token_ids_buffer.get())
        return {StatusCode::InvalidArgument, "Q4_0 Embedding buffer ranges are inconsistent"};
    dispatch.label = "q4-0-embedding-f32-activation-generic-unverified";
    dispatch.kernel_id = "dawn-q4-0-embedding-f32-v1";
    dispatch.backend = BackendKind::DawnWebGpu;
    dispatch.wgsl_source = kQ4_0EmbeddingWgsl;
    dispatch.workgroup_size_x = 64;
    const std::pair<std::string, std::string> constants[] = {
        {"const VOCABULARY_SIZE: u32 = 1u;", "const VOCABULARY_SIZE: u32 = " + std::to_string(vocabulary_size) + "u;"},
        {"const EMBEDDING_DIMENSION: u32 = 32u;", "const EMBEDDING_DIMENSION: u32 = " + std::to_string(embedding_dimension) + "u;"},
        {"const TOKEN_COUNT: u32 = 1u;", "const TOKEN_COUNT: u32 = " + std::to_string(token_ids.size()) + "u;"},
        {"const TOTAL_ELEMENTS: u32 = 32u;", "const TOTAL_ELEMENTS: u32 = " + std::to_string(output.element_count()) + "u;"},
        {"const WEIGHT_ROW_BYTES: u32 = 18u;", "const WEIGHT_ROW_BYTES: u32 = " + std::to_string(row_bytes) + "u;"}
    };
    for (const auto& constant : constants) {
        const size_t pos = dispatch.wgsl_source.find(constant.first);
        if (pos == std::string::npos) return {StatusCode::BackendFailure, "Q4_0 Embedding WGSL placeholder missing"};
        dispatch.wgsl_source.replace(pos, constant.first.size(), constant.second);
    }
    dispatch.workgroups_x = static_cast<uint32_t>((output.element_count() + 63) / 64);
    dispatch.bindings = {
        {0, quantized_weights, 0, padded_weights},
        {1, token_ids_buffer, 0, token_bytes},
        {2, output.buffer(), output.byte_offset(), output.byte_size()}
    };
    return Status::success();
}

Status build_q4_0_embedding_gles_candidate(const std::shared_ptr<Buffer>& quantized_weights,
                                           uint64_t weight_bytes, uint32_t vocabulary_size,
                                           uint32_t embedding_dimension,
                                           const std::shared_ptr<Buffer>& token_ids_buffer,
                                           const std::vector<uint32_t>& token_ids,
                                           const Tensor& output, ComputeDispatch& dispatch) {
    Status status = build_q4_0_embedding_candidate(quantized_weights, weight_bytes,
        vocabulary_size, embedding_dimension, token_ids_buffer, token_ids, output, dispatch);
    if (!status.ok()) return status;
    dispatch.backend = BackendKind::OpenGles;
    dispatch.kernel_id = "gles-q4-0-embedding-f32-v1";
    dispatch.label = "gles-q4-0-embedding-f32-unverified";
    dispatch.wgsl_source.clear();
    dispatch.glsl_source = kQ4_0EmbeddingGlsl;
    const uint64_t row_bytes = static_cast<uint64_t>(embedding_dimension / 32) * 18;
    const std::pair<std::string, std::string> constants[] = {
        {"const uint VOCABULARY_SIZE = 1u;", "const uint VOCABULARY_SIZE = " + std::to_string(vocabulary_size) + "u;"},
        {"const uint EMBEDDING_DIMENSION = 32u;", "const uint EMBEDDING_DIMENSION = " + std::to_string(embedding_dimension) + "u;"},
        {"const uint TOKEN_COUNT = 1u;", "const uint TOKEN_COUNT = " + std::to_string(token_ids.size()) + "u;"},
        {"const uint TOTAL_ELEMENTS = 32u;", "const uint TOTAL_ELEMENTS = " + std::to_string(output.element_count()) + "u;"},
        {"const uint WEIGHT_ROW_BYTES = 18u;", "const uint WEIGHT_ROW_BYTES = " + std::to_string(row_bytes) + "u;"}
    };
    for (const auto& constant : constants) {
        const size_t pos = dispatch.glsl_source.find(constant.first);
        if (pos == std::string::npos) return {StatusCode::BackendFailure, "Q4_0 Embedding GLSL placeholder missing"};
        dispatch.glsl_source.replace(pos, constant.first.size(), constant.second);
    }
    return Status::success();
}

Status reference_q4_0_embedding(const uint8_t* quantized_weights, uint64_t weight_bytes,
                                uint32_t vocabulary_size, uint32_t embedding_dimension,
                                const std::vector<uint32_t>& token_ids, float* output) {
    if ((!quantized_weights || !output) && !token_ids.empty())
        return {StatusCode::InvalidArgument, "Q4_0 Embedding reference received a null pointer"};
    if (vocabulary_size == 0 || embedding_dimension == 0 || embedding_dimension % 32 != 0 ||
        token_ids.empty())
        return {StatusCode::InvalidArgument, "Q4_0 Embedding dimensions and token list must be non-empty and block-aligned"};
    const uint64_t row_bytes = static_cast<uint64_t>(embedding_dimension / 32) * 18;
    if (row_bytes > UINT64_MAX / vocabulary_size || weight_bytes != row_bytes * vocabulary_size)
        return {StatusCode::InvalidArgument, "Q4_0 Embedding weight byte size is inconsistent"};
    std::vector<float> decoded(embedding_dimension);
    for (size_t row = 0; row < token_ids.size(); ++row) {
        if (token_ids[row] >= vocabulary_size)
            return {StatusCode::InvalidArgument, "Q4_0 Embedding token ID is outside vocabulary"};
        Status status = decode_q4_0_blocks(quantized_weights + token_ids[row] * row_bytes,
            row_bytes, decoded.data(), decoded.size());
        if (!status.ok()) return status;
        std::copy(decoded.begin(), decoded.end(), output + row * embedding_dimension);
    }
    return Status::success();
}

Status build_q4_1_embedding_candidate(const std::shared_ptr<Buffer>& quantized_weights,
                                      uint64_t weight_bytes, uint32_t vocabulary_size,
                                      uint32_t embedding_dimension,
                                      const std::shared_ptr<Buffer>& token_ids_buffer,
                                      const std::vector<uint32_t>& token_ids,
                                      const Tensor& output, ComputeDispatch& dispatch) {
    dispatch = ComputeDispatch{};
    if (!quantized_weights || !token_ids_buffer || !output.buffer() ||
        quantized_weights->memory_domain() == MemoryDomain::Host ||
        token_ids_buffer->memory_domain() == MemoryDomain::Host ||
        output.descriptor().data_type != DataType::F32 ||
        output.descriptor().layout != TensorLayout::Contiguous ||
        output.descriptor().memory == MemoryDomain::Host ||
        output.descriptor().shape.size() != 2 || token_ids.empty())
        return {StatusCode::Unsupported, "Q4_1 Embedding requires packed weights, device token IDs and contiguous F32 output"};
    if (vocabulary_size == 0 || embedding_dimension == 0 || embedding_dimension % 32 != 0 ||
        token_ids.size() != output.descriptor().shape[0] ||
        output.descriptor().shape[1] != embedding_dimension ||
        token_ids.size() > UINT32_MAX || output.element_count() > UINT32_MAX)
        return {StatusCode::InvalidArgument, "Q4_1 Embedding dimensions or token count are invalid"};
    for (uint32_t token : token_ids)
        if (token >= vocabulary_size) return {StatusCode::InvalidArgument, "Q4_1 Embedding token ID is outside vocabulary"};
    const uint64_t row_bytes = static_cast<uint64_t>(embedding_dimension / 32) * 20;
    if (row_bytes > UINT64_MAX / vocabulary_size || weight_bytes != row_bytes * vocabulary_size ||
        weight_bytes > static_cast<uint64_t>(UINT32_MAX) - 3)
        return {StatusCode::InvalidArgument, "Q4_1 Embedding weight byte size does not match [vocabulary,dimension]"};
    const uint64_t padded_weights = (weight_bytes + 3) & ~uint64_t{3};
    const uint64_t token_bytes = token_ids.size() * sizeof(uint32_t);
    if (quantized_weights->size_bytes() < padded_weights ||
        token_ids_buffer->size_bytes() < token_bytes ||
        output.byte_size() != output.element_count() * sizeof(float) ||
        output.buffer().get() == quantized_weights.get() ||
        output.buffer().get() == token_ids_buffer.get() ||
        quantized_weights.get() == token_ids_buffer.get())
        return {StatusCode::InvalidArgument, "Q4_1 Embedding buffer ranges are inconsistent"};
    dispatch.label = "q4-1-embedding-f32-activation-generic-unverified";
    dispatch.kernel_id = "dawn-q4-1-embedding-f32-v1";
    dispatch.backend = BackendKind::DawnWebGpu;
    dispatch.wgsl_source = kQ4_1EmbeddingWgsl;
    dispatch.workgroup_size_x = 64;
    const std::pair<std::string, std::string> constants[] = {
        {"const VOCABULARY_SIZE: u32 = 1u;", "const VOCABULARY_SIZE: u32 = " + std::to_string(vocabulary_size) + "u;"},
        {"const EMBEDDING_DIMENSION: u32 = 32u;", "const EMBEDDING_DIMENSION: u32 = " + std::to_string(embedding_dimension) + "u;"},
        {"const TOKEN_COUNT: u32 = 1u;", "const TOKEN_COUNT: u32 = " + std::to_string(token_ids.size()) + "u;"},
        {"const TOTAL_ELEMENTS: u32 = 32u;", "const TOTAL_ELEMENTS: u32 = " + std::to_string(output.element_count()) + "u;"},
        {"const WEIGHT_ROW_BYTES: u32 = 20u;", "const WEIGHT_ROW_BYTES: u32 = " + std::to_string(row_bytes) + "u;"}
    };
    for (const auto& constant : constants) {
        const size_t pos = dispatch.wgsl_source.find(constant.first);
        if (pos == std::string::npos) return {StatusCode::BackendFailure, "Q4_1 Embedding WGSL placeholder missing"};
        dispatch.wgsl_source.replace(pos, constant.first.size(), constant.second);
    }
    dispatch.workgroups_x = static_cast<uint32_t>((output.element_count() + 63) / 64);
    dispatch.bindings = {
        {0, quantized_weights, 0, padded_weights},
        {1, token_ids_buffer, 0, token_bytes},
        {2, output.buffer(), output.byte_offset(), output.byte_size()}
    };
    return Status::success();
}

Status build_q4_1_embedding_gles_candidate(const std::shared_ptr<Buffer>& quantized_weights,
                                           uint64_t weight_bytes, uint32_t vocabulary_size,
                                           uint32_t embedding_dimension,
                                           const std::shared_ptr<Buffer>& token_ids_buffer,
                                           const std::vector<uint32_t>& token_ids,
                                           const Tensor& output, ComputeDispatch& dispatch) {
    Status status = build_q4_1_embedding_candidate(quantized_weights, weight_bytes,
        vocabulary_size, embedding_dimension, token_ids_buffer, token_ids, output, dispatch);
    if (!status.ok()) return status;
    dispatch.backend = BackendKind::OpenGles;
    dispatch.kernel_id = "gles-q4-1-embedding-f32-v1";
    dispatch.label = "gles-q4-1-embedding-f32-unverified";
    dispatch.wgsl_source.clear();
    dispatch.glsl_source = kQ4_1EmbeddingGlsl;
    const uint64_t row_bytes = static_cast<uint64_t>(embedding_dimension / 32) * 20;
    const std::pair<std::string, std::string> constants[] = {
        {"const uint VOCABULARY_SIZE = 1u;", "const uint VOCABULARY_SIZE = " + std::to_string(vocabulary_size) + "u;"},
        {"const uint EMBEDDING_DIMENSION = 32u;", "const uint EMBEDDING_DIMENSION = " + std::to_string(embedding_dimension) + "u;"},
        {"const uint TOKEN_COUNT = 1u;", "const uint TOKEN_COUNT = " + std::to_string(token_ids.size()) + "u;"},
        {"const uint TOTAL_ELEMENTS = 32u;", "const uint TOTAL_ELEMENTS = " + std::to_string(output.element_count()) + "u;"},
        {"const uint WEIGHT_ROW_BYTES = 20u;", "const uint WEIGHT_ROW_BYTES = " + std::to_string(row_bytes) + "u;"}
    };
    for (const auto& constant : constants) {
        const size_t pos = dispatch.glsl_source.find(constant.first);
        if (pos == std::string::npos) return {StatusCode::BackendFailure, "Q4_1 Embedding GLSL placeholder missing"};
        dispatch.glsl_source.replace(pos, constant.first.size(), constant.second);
    }
    return Status::success();
}

Status reference_q4_1_embedding(const uint8_t* quantized_weights, uint64_t weight_bytes,
                                uint32_t vocabulary_size, uint32_t embedding_dimension,
                                const std::vector<uint32_t>& token_ids, float* output) {
    if ((!quantized_weights || !output) && !token_ids.empty())
        return {StatusCode::InvalidArgument, "Q4_1 Embedding reference received a null pointer"};
    if (vocabulary_size == 0 || embedding_dimension == 0 || embedding_dimension % 32 != 0 ||
        token_ids.empty())
        return {StatusCode::InvalidArgument, "Q4_1 Embedding dimensions and token list must be non-empty and block-aligned"};
    const uint64_t row_bytes = static_cast<uint64_t>(embedding_dimension / 32) * 20;
    if (row_bytes > UINT64_MAX / vocabulary_size || weight_bytes != row_bytes * vocabulary_size)
        return {StatusCode::InvalidArgument, "Q4_1 Embedding weight byte size is inconsistent"};
    std::vector<float> decoded(embedding_dimension);
    for (size_t row = 0; row < token_ids.size(); ++row) {
        if (token_ids[row] >= vocabulary_size)
            return {StatusCode::InvalidArgument, "Q4_1 Embedding token ID is outside vocabulary"};
        Status status = decode_q4_1_blocks(quantized_weights + token_ids[row] * row_bytes,
            row_bytes, decoded.data(), decoded.size());
        if (!status.ok()) return status;
        std::copy(decoded.begin(), decoded.end(), output + row * embedding_dimension);
    }
    return Status::success();
}

Status validate_q4_0_embedding_kernel(Backend& backend, OperatorRegistry& operators) {
    const BackendKind backend_kind = backend.kind();
    if ((backend_kind != BackendKind::DawnWebGpu && backend_kind != BackendKind::OpenGles) ||
        !backend.healthy())
        return {StatusCode::NotReady, "Q4_0 Embedding validation requires a healthy Dawn or GLES backend"};
    const bool use_gles = backend_kind == BackendKind::OpenGles;
    const std::string kernel_id = use_gles ? "gles-q4-0-embedding-f32-v1"
                                           : "dawn-q4-0-embedding-f32-v1";
    auto kernel = std::find_if(operators.kernels_.begin(), operators.kernels_.end(),
        [&](const KernelDefinition& definition) {
            return definition.kernel_id == kernel_id && definition.backend == backend_kind;
        });
    if (kernel == operators.kernels_.end())
        return {StatusCode::NotReady, "Q4_0 Embedding candidate is not registered for the selected backend"};
    if (kernel->validation == KernelValidation::Validated) return Status::success();
    if (kernel->validation != KernelValidation::Unverified)
        return {StatusCode::Unsupported, "Q4_0 Embedding candidate is rejected"};
    const std::string& registered_source = use_gles ? kernel->glsl_source : kernel->wgsl_source;
    const std::string& canonical_source = use_gles ? q4_0_embedding_glsl_source()
                                                    : q4_0_embedding_wgsl_source();
    if (registered_source.empty() || registered_source != canonical_source)
        return {StatusCode::InvalidArgument, "Q4_0 Embedding registered source differs from canonical source"};

    constexpr uint32_t vocabulary = 2, dimension = 32;
    const std::vector<uint32_t> token_ids = {1, 0};
    const uint64_t weight_bytes = static_cast<uint64_t>(vocabulary) * 18;
    std::vector<uint8_t> encoded(weight_bytes, 0);
    for (uint32_t row = 0; row < vocabulary; ++row) {
        const size_t offset = row * 18;
        encoded[offset] = 0x00;
        encoded[offset + 1] = 0x38; // FP16 scale 0.5
        for (uint32_t i = 0; i < 16; ++i) {
            const uint8_t low = static_cast<uint8_t>((static_cast<int>((i + row) % 11) - 5) + 8);
            const uint8_t high = static_cast<uint8_t>((static_cast<int>((i + row + 3) % 11) - 5) + 8);
            encoded[offset + 2 + i] = static_cast<uint8_t>(low | (high << 4));
        }
    }
    std::vector<float> expected(token_ids.size() * dimension, 0.0f);
    std::vector<float> actual(token_ids.size() * dimension, 0.0f);
    Status status = reference_q4_0_embedding(encoded.data(), weight_bytes, vocabulary,
        dimension, token_ids, expected.data());
    if (!status.ok()) return status;

    BufferDesc weight_desc{(weight_bytes + 3) & ~uint64_t{3}, 4, MemoryDomain::Device, true, "q4-0-embedding-weights"};
    BufferDesc token_desc{token_ids.size() * sizeof(uint32_t), alignof(uint32_t), MemoryDomain::Device, false, "q4-0-embedding-token-ids"};
    BufferDesc output_desc{actual.size() * sizeof(float), alignof(float), MemoryDomain::Device, false, "q4-0-embedding-output"};
    std::shared_ptr<Buffer> weight_buffer, token_buffer, output_buffer;
    status = backend.allocate(weight_desc, weight_buffer); if (!status.ok()) return status;
    status = backend.allocate(token_desc, token_buffer); if (!status.ok()) return status;
    status = backend.allocate(output_desc, output_buffer); if (!status.ok()) return status;
    status = backend.upload(weight_buffer, 0, encoded.data(), weight_bytes); if (!status.ok()) return status;
    status = backend.upload(token_buffer, 0, token_ids.data(), token_desc.size_bytes); if (!status.ok()) return status;

    TensorDesc output_td; output_td.shape = {token_ids.size(), dimension}; output_td.data_type = DataType::F32;
    Tensor output_tensor;
    status = Tensor::create(output_td, output_buffer, 0, output_tensor); if (!status.ok()) return status;
    ComputeDispatch dispatch;
    status = use_gles
        ? build_q4_0_embedding_gles_candidate(weight_buffer, weight_bytes, vocabulary, dimension,
            token_buffer, token_ids, output_tensor, dispatch)
        : build_q4_0_embedding_candidate(weight_buffer, weight_bytes, vocabulary, dimension,
            token_buffer, token_ids, output_tensor, dispatch);
    if (!status.ok()) return status;
    const std::string& candidate_source = use_gles ? dispatch.glsl_source : dispatch.wgsl_source;
    if (dispatch.kernel_id != kernel_id || candidate_source.empty())
        return {StatusCode::InvalidArgument, "Q4_0 Embedding candidate identity/source is invalid"};
    dispatch.validation = KernelValidation::Validated;
    std::unique_ptr<CommandBatch> commands;
    status = backend.begin_commands(commands); if (!status.ok()) return status;
    status = commands->dispatch_compute(dispatch); if (!status.ok()) return status;
    status = backend.submit(std::move(commands)); if (!status.ok()) return status;
    status = backend.synchronize(); if (!status.ok()) return status;
    status = backend.readback(output_buffer, 0, actual.data(), actual.size() * sizeof(float));
    if (!status.ok()) return status;
    for (size_t i = 0; i < actual.size(); ++i) {
        const float tolerance = 1.0e-5f * std::max(1.0f, std::fabs(expected[i]));
        if (!std::isfinite(actual[i]) || std::fabs(actual[i] - expected[i]) > tolerance)
            return {StatusCode::NumericalFailure, "Q4_0 Embedding GPU output differs from the independent decoder reference"};
    }
    kernel->validation = KernelValidation::Validated;
    return Status::success();
}

Status validate_q8_0_embedding_kernel(Backend& backend, OperatorRegistry& operators) {
    const BackendKind backend_kind = backend.kind();
    if ((backend_kind != BackendKind::DawnWebGpu && backend_kind != BackendKind::OpenGles) ||
        !backend.healthy())
        return {StatusCode::NotReady, "Q8_0 Embedding validation requires a healthy Dawn or GLES backend"};
    const bool use_gles = backend_kind == BackendKind::OpenGles;
    const std::string kernel_id = use_gles ? "gles-q8-0-embedding-f32-v1"
                                           : "dawn-q8-0-embedding-f32-v1";
    auto kernel = std::find_if(operators.kernels_.begin(), operators.kernels_.end(),
        [&](const KernelDefinition& definition) {
            return definition.kernel_id == kernel_id && definition.backend == backend_kind;
        });
    if (kernel == operators.kernels_.end())
        return {StatusCode::NotReady, "Q8_0 Embedding candidate is not registered for the selected backend"};
    if (kernel->validation == KernelValidation::Validated) return Status::success();
    if (kernel->validation != KernelValidation::Unverified)
        return {StatusCode::Unsupported, "Q8_0 Embedding candidate is rejected"};
    const std::string& registered_source = use_gles ? kernel->glsl_source : kernel->wgsl_source;
    const std::string& canonical_source = use_gles ? q8_0_embedding_glsl_source()
                                                    : q8_0_embedding_wgsl_source();
    if (registered_source.empty() || registered_source != canonical_source)
        return {StatusCode::InvalidArgument, "Q8_0 Embedding registered source differs from canonical source"};

    constexpr uint32_t vocabulary = 2, dimension = 32;
    const std::vector<uint32_t> token_ids = {1, 0};
    const uint64_t weight_bytes = static_cast<uint64_t>(vocabulary) * 34;
    std::vector<uint8_t> encoded(weight_bytes, 0);
    for (uint32_t row = 0; row < vocabulary; ++row) {
        const size_t offset = row * 34;
        encoded[offset] = 0x00;
        encoded[offset + 1] = 0x38; // FP16 scale 0.5
        for (uint32_t i = 0; i < 32; ++i) {
            const int8_t value = static_cast<int8_t>(static_cast<int>((i + row) % 11) - 5);
            encoded[offset + 2 + i] = static_cast<uint8_t>(value);
        }
    }
    std::vector<float> expected(token_ids.size() * dimension, 0.0f);
    std::vector<float> actual(token_ids.size() * dimension, 0.0f);
    Status status = reference_q8_0_embedding(encoded.data(), weight_bytes, vocabulary,
        dimension, token_ids, expected.data());
    if (!status.ok()) return status;

    BufferDesc weight_desc{(weight_bytes + 3) & ~uint64_t{3}, 4, MemoryDomain::Device, true, "q8-0-embedding-weights"};
    BufferDesc token_desc{token_ids.size() * sizeof(uint32_t), alignof(uint32_t), MemoryDomain::Device, false, "q8-0-embedding-token-ids"};
    BufferDesc output_desc{actual.size() * sizeof(float), alignof(float), MemoryDomain::Device, false, "q8-0-embedding-output"};
    std::shared_ptr<Buffer> weight_buffer, token_buffer, output_buffer;
    status = backend.allocate(weight_desc, weight_buffer); if (!status.ok()) return status;
    status = backend.allocate(token_desc, token_buffer); if (!status.ok()) return status;
    status = backend.allocate(output_desc, output_buffer); if (!status.ok()) return status;
    status = backend.upload(weight_buffer, 0, encoded.data(), weight_bytes); if (!status.ok()) return status;
    status = backend.upload(token_buffer, 0, token_ids.data(), token_desc.size_bytes); if (!status.ok()) return status;

    TensorDesc output_td; output_td.shape = {token_ids.size(), dimension}; output_td.data_type = DataType::F32;
    Tensor output_tensor;
    status = Tensor::create(output_td, output_buffer, 0, output_tensor); if (!status.ok()) return status;
    ComputeDispatch dispatch;
    status = use_gles
        ? build_q8_0_embedding_gles_candidate(weight_buffer, weight_bytes, vocabulary, dimension,
            token_buffer, token_ids, output_tensor, dispatch)
        : build_q8_0_embedding_candidate(weight_buffer, weight_bytes, vocabulary, dimension,
            token_buffer, token_ids, output_tensor, dispatch);
    if (!status.ok()) return status;
    const std::string& candidate_source = use_gles ? dispatch.glsl_source : dispatch.wgsl_source;
    if (dispatch.kernel_id != kernel_id || candidate_source.empty())
        return {StatusCode::InvalidArgument, "Q8_0 Embedding candidate identity/source is invalid"};
    dispatch.validation = KernelValidation::Validated;
    std::unique_ptr<CommandBatch> commands;
    status = backend.begin_commands(commands); if (!status.ok()) return status;
    status = commands->dispatch_compute(dispatch); if (!status.ok()) return status;
    status = backend.submit(std::move(commands)); if (!status.ok()) return status;
    status = backend.synchronize(); if (!status.ok()) return status;
    status = backend.readback(output_buffer, 0, actual.data(), actual.size() * sizeof(float));
    if (!status.ok()) return status;
    for (size_t i = 0; i < actual.size(); ++i) {
        const float tolerance = 1.0e-5f * std::max(1.0f, std::fabs(expected[i]));
        if (!std::isfinite(actual[i]) || std::fabs(actual[i] - expected[i]) > tolerance)
            return {StatusCode::NumericalFailure, "Q8_0 Embedding GPU output differs from the independent decoder reference"};
    }
    kernel->validation = KernelValidation::Validated;
    return Status::success();
}

Status validate_q4_1_embedding_kernel(Backend& backend, OperatorRegistry& operators) {
    const BackendKind backend_kind = backend.kind();
    if ((backend_kind != BackendKind::DawnWebGpu && backend_kind != BackendKind::OpenGles) ||
        !backend.healthy())
        return {StatusCode::NotReady, "Q4_1 Embedding validation requires a healthy Dawn or GLES backend"};
    const bool use_gles = backend_kind == BackendKind::OpenGles;
    const std::string kernel_id = use_gles ? "gles-q4-1-embedding-f32-v1"
                                           : "dawn-q4-1-embedding-f32-v1";
    auto kernel = std::find_if(operators.kernels_.begin(), operators.kernels_.end(),
        [&](const KernelDefinition& definition) {
            return definition.kernel_id == kernel_id && definition.backend == backend_kind;
        });
    if (kernel == operators.kernels_.end())
        return {StatusCode::NotReady, "Q4_1 Embedding candidate is not registered for the selected backend"};
    if (kernel->validation == KernelValidation::Validated) return Status::success();
    if (kernel->validation != KernelValidation::Unverified)
        return {StatusCode::Unsupported, "Q4_1 Embedding candidate is rejected"};
    const std::string& registered_source = use_gles ? kernel->glsl_source : kernel->wgsl_source;
    const std::string& canonical_source = use_gles ? q4_1_embedding_glsl_source()
                                                    : q4_1_embedding_wgsl_source();
    if (registered_source.empty() || registered_source != canonical_source)
        return {StatusCode::InvalidArgument, "Q4_1 Embedding registered source differs from canonical source"};

    constexpr uint32_t vocabulary = 2, dimension = 32;
    const std::vector<uint32_t> token_ids = {1, 0};
    const uint64_t weight_bytes = static_cast<uint64_t>(vocabulary) * 20;
    std::vector<uint8_t> encoded(weight_bytes, 0);
    for (uint32_t row = 0; row < vocabulary; ++row) {
        const size_t offset = row * 20;
        encoded[offset] = 0x00;
        encoded[offset + 1] = 0x38; // FP16 scale 0.5
        encoded[offset + 2] = 0x00;
        encoded[offset + 3] = 0xbc; // FP16 minimum -1.0
        for (uint32_t i = 0; i < 16; ++i) {
            const uint8_t low = static_cast<uint8_t>((i + row) % 16);
            const uint8_t high = static_cast<uint8_t>((i + row + 3) % 16);
            encoded[offset + 4 + i] = static_cast<uint8_t>(low | (high << 4));
        }
    }
    std::vector<float> expected(token_ids.size() * dimension, 0.0f);
    std::vector<float> actual(token_ids.size() * dimension, 0.0f);
    Status status = reference_q4_1_embedding(encoded.data(), weight_bytes, vocabulary,
        dimension, token_ids, expected.data());
    if (!status.ok()) return status;

    BufferDesc weight_desc{(weight_bytes + 3) & ~uint64_t{3}, 4, MemoryDomain::Device, true, "q4-1-embedding-weights"};
    BufferDesc token_desc{token_ids.size() * sizeof(uint32_t), alignof(uint32_t), MemoryDomain::Device, false, "q4-1-embedding-token-ids"};
    BufferDesc output_desc{actual.size() * sizeof(float), alignof(float), MemoryDomain::Device, false, "q4-1-embedding-output"};
    std::shared_ptr<Buffer> weight_buffer, token_buffer, output_buffer;
    status = backend.allocate(weight_desc, weight_buffer); if (!status.ok()) return status;
    status = backend.allocate(token_desc, token_buffer); if (!status.ok()) return status;
    status = backend.allocate(output_desc, output_buffer); if (!status.ok()) return status;
    status = backend.upload(weight_buffer, 0, encoded.data(), weight_bytes); if (!status.ok()) return status;
    status = backend.upload(token_buffer, 0, token_ids.data(), token_desc.size_bytes); if (!status.ok()) return status;

    TensorDesc output_td; output_td.shape = {token_ids.size(), dimension}; output_td.data_type = DataType::F32;
    Tensor output_tensor;
    status = Tensor::create(output_td, output_buffer, 0, output_tensor); if (!status.ok()) return status;
    ComputeDispatch dispatch;
    status = use_gles
        ? build_q4_1_embedding_gles_candidate(weight_buffer, weight_bytes, vocabulary, dimension,
            token_buffer, token_ids, output_tensor, dispatch)
        : build_q4_1_embedding_candidate(weight_buffer, weight_bytes, vocabulary, dimension,
            token_buffer, token_ids, output_tensor, dispatch);
    if (!status.ok()) return status;
    const std::string& candidate_source = use_gles ? dispatch.glsl_source : dispatch.wgsl_source;
    if (dispatch.kernel_id != kernel_id || candidate_source.empty())
        return {StatusCode::InvalidArgument, "Q4_1 Embedding candidate identity/source is invalid"};
    dispatch.validation = KernelValidation::Validated;
    std::unique_ptr<CommandBatch> commands;
    status = backend.begin_commands(commands); if (!status.ok()) return status;
    status = commands->dispatch_compute(dispatch); if (!status.ok()) return status;
    status = backend.submit(std::move(commands)); if (!status.ok()) return status;
    status = backend.synchronize(); if (!status.ok()) return status;
    status = backend.readback(output_buffer, 0, actual.data(), actual.size() * sizeof(float));
    if (!status.ok()) return status;
    for (size_t i = 0; i < actual.size(); ++i) {
        const float tolerance = 1.0e-5f * std::max(1.0f, std::fabs(expected[i]));
        if (!std::isfinite(actual[i]) || std::fabs(actual[i] - expected[i]) > tolerance)
            return {StatusCode::NumericalFailure, "Q4_1 Embedding GPU output differs from the independent decoder reference"};
    }
    kernel->validation = KernelValidation::Validated;
    return Status::success();
}

Status validate_q4_0_matmul_kernel(Backend& backend, OperatorRegistry& operators,
                                   QuantizationRegistry& quantizations) {
    const BackendKind backend_kind = backend.kind();
    if ((backend_kind != BackendKind::DawnWebGpu && backend_kind != BackendKind::OpenGles) ||
        !backend.healthy())
        return {StatusCode::NotReady, "Q4_0 MatMul validation requires a healthy Dawn or GLES backend"};
    constexpr uint32_t kQ4_0TypeId = 2;
    auto format = quantizations.formats_.find(kQ4_0TypeId);
    if (format == quantizations.formats_.end() || format->second.decoder_state != SupportState::Validated)
        return {StatusCode::NotReady, "Q4_0 host decoder must be validated before GPU kernel promotion"};
    const auto backend_validation = format->second.backend_gpu_kernel_states.find(backend_kind);
    if (backend_validation != format->second.backend_gpu_kernel_states.end() &&
        backend_validation->second == SupportState::Validated)
        return Status::success();

    const bool use_gles = backend_kind == BackendKind::OpenGles;
    const std::string kernel_id = use_gles ? "gles-q4-0-matmul-f32-activation-v1"
                                           : "dawn-q4-0-matmul-f32-activation-v1";
    auto kernel = std::find_if(operators.kernels_.begin(), operators.kernels_.end(),
        [&](const KernelDefinition& definition) {
            return definition.kernel_id == kernel_id && definition.backend == backend_kind;
        });
    if (kernel == operators.kernels_.end())
        return {StatusCode::NotReady, "Q4_0 MatMul candidate is not registered for the selected backend"};
    if (kernel->validation == KernelValidation::Validated) {
        format->second.backend_gpu_kernel_states[backend_kind] = SupportState::Validated;
        format->second.gpu_kernel_state = SupportState::Validated;
        return Status::success();
    }
    if (kernel->validation != KernelValidation::Unverified)
        return {StatusCode::Unsupported, "Q4_0 MatMul candidate is rejected"};
    const std::string& registered_source = use_gles ? kernel->glsl_source : kernel->wgsl_source;
    if (registered_source.empty())
        return {StatusCode::Unsupported, "Q4_0 MatMul candidate has no source for its selected backend"};

    constexpr uint32_t rows = 2, output_channels = 2, inner = 64;
    std::vector<float> activations(rows * inner);
    for (size_t i = 0; i < activations.size(); ++i)
        activations[i] = (static_cast<int>(i % 9) - 4) * 0.25f;
    const uint64_t weight_bytes = static_cast<uint64_t>(output_channels) * (inner / 32) * 18;
    std::vector<uint8_t> encoded(static_cast<size_t>(weight_bytes), 0);
    for (uint32_t block = 0; block < output_channels * (inner / 32); ++block) {
        const size_t offset = static_cast<size_t>(block) * 18;
        encoded[offset] = 0x00;
        encoded[offset + 1] = 0x38; // FP16 scale 0.5
        for (uint32_t i = 0; i < 16; ++i) {
            const uint8_t low = static_cast<uint8_t>((static_cast<int>(i % 11) - 5) + 8);
            const uint8_t high = static_cast<uint8_t>((static_cast<int>((i + 3) % 11) - 5) + 8);
            encoded[offset + 2 + i] = static_cast<uint8_t>(low | (high << 4));
        }
    }
    float expected[rows * output_channels] = {};
    float actual[rows * output_channels] = {};
    Status status = reference_q4_0_matmul(activations.data(), encoded.data(), weight_bytes,
        expected, rows, output_channels, inner);
    if (!status.ok()) return status;

    BufferDesc activation_desc{activations.size() * sizeof(float), alignof(float), MemoryDomain::Device,
        false, "q4-validation-activations"};
    BufferDesc weights_desc{(weight_bytes + 3) & ~uint64_t{3}, 4, MemoryDomain::Device,
        true, "q4-validation-weights"};
    BufferDesc output_desc{sizeof(actual), alignof(float), MemoryDomain::Device,
        false, "q4-validation-output"};
    std::shared_ptr<Buffer> activation_buffer, weights_buffer, output_buffer;
    status = backend.allocate(activation_desc, activation_buffer); if (!status.ok()) return status;
    status = backend.allocate(weights_desc, weights_buffer); if (!status.ok()) return status;
    status = backend.allocate(output_desc, output_buffer); if (!status.ok()) return status;
    status = backend.upload(activation_buffer, 0, activations.data(), activations.size() * sizeof(float));
    if (!status.ok()) return status;
    status = backend.upload(weights_buffer, 0, encoded.data(), weight_bytes);
    if (!status.ok()) return status;

    TensorDesc activation_tensor_desc; activation_tensor_desc.shape = {rows, inner};
    activation_tensor_desc.data_type = DataType::F32;
    TensorDesc output_tensor_desc; output_tensor_desc.shape = {rows, output_channels};
    output_tensor_desc.data_type = DataType::F32;
    Tensor activation_tensor, output_tensor;
    status = Tensor::create(activation_tensor_desc, activation_buffer, 0, activation_tensor);
    if (!status.ok()) return status;
    status = Tensor::create(output_tensor_desc, output_buffer, 0, output_tensor);
    if (!status.ok()) return status;

    ComputeDispatch dispatch;
    status = use_gles
        ? build_q4_0_matmul_gles_candidate(activation_tensor, weights_buffer, weight_bytes, output_tensor, dispatch)
        : build_q4_0_matmul_candidate(activation_tensor, weights_buffer, weight_bytes, output_tensor, dispatch);
    if (!status.ok()) return status;
    const std::string& candidate_source = use_gles ? dispatch.glsl_source : dispatch.wgsl_source;
    const std::string& canonical_source = use_gles ? q4_0_matmul_glsl_source()
                                                    : q4_0_matmul_wgsl_source();
    if (dispatch.kernel_id != kernel_id || registered_source != canonical_source ||
        candidate_source.empty())
        return {StatusCode::InvalidArgument, "Q4_0 MatMul validation source differs from the registered canonical candidate"};
    dispatch.validation = KernelValidation::Validated;
    std::unique_ptr<CommandBatch> commands;
    status = backend.begin_commands(commands); if (!status.ok()) return status;
    status = commands->dispatch_compute(dispatch); if (!status.ok()) return status;
    status = backend.submit(std::move(commands)); if (!status.ok()) return status;
    status = backend.synchronize(); if (!status.ok()) return status;
    status = backend.readback(output_buffer, 0, actual, sizeof(actual)); if (!status.ok()) return status;
    for (size_t i = 0; i < rows * output_channels; ++i) {
        const float tolerance = 1.0e-4f * std::max(1.0f, std::fabs(expected[i]));
        if (!std::isfinite(actual[i]) || std::fabs(actual[i] - expected[i]) > tolerance)
            return {StatusCode::NumericalFailure, "Q4_0 MatMul GPU output differs from the independent block-decoder reference"};
    }
    kernel->validation = KernelValidation::Validated;
    format->second.backend_gpu_kernel_states[backend_kind] = SupportState::Validated;
    format->second.gpu_kernel_state = SupportState::Validated;
    return Status::success();
}



Status validate_q4_1_matmul_kernel(Backend& backend, OperatorRegistry& operators,
                                   QuantizationRegistry& quantizations) {
    const BackendKind backend_kind = backend.kind();
    if ((backend_kind != BackendKind::DawnWebGpu && backend_kind != BackendKind::OpenGles) ||
        !backend.healthy())
        return {StatusCode::NotReady, "Q4_1 MatMul validation requires a healthy Dawn or GLES backend"};
    constexpr uint32_t kQ4_1TypeId = 3;
    auto format = quantizations.formats_.find(kQ4_1TypeId);
    if (format == quantizations.formats_.end() || format->second.decoder_state != SupportState::Validated)
        return {StatusCode::NotReady, "Q4_1 host decoder must be validated before GPU kernel promotion"};
    const auto backend_validation = format->second.backend_gpu_kernel_states.find(backend_kind);
    if (backend_validation != format->second.backend_gpu_kernel_states.end() &&
        backend_validation->second == SupportState::Validated)
        return Status::success();

    const bool use_gles = backend_kind == BackendKind::OpenGles;
    const std::string kernel_id = use_gles ? "gles-q4-1-matmul-f32-activation-v1"
                                           : "dawn-q4-1-matmul-f32-activation-v1";
    auto kernel = std::find_if(operators.kernels_.begin(), operators.kernels_.end(),
        [&](const KernelDefinition& definition) {
            return definition.kernel_id == kernel_id && definition.backend == backend_kind;
        });
    if (kernel == operators.kernels_.end())
        return {StatusCode::NotReady, "Q4_1 MatMul candidate is not registered for the selected backend"};
    if (kernel->validation == KernelValidation::Validated) {
        format->second.backend_gpu_kernel_states[backend_kind] = SupportState::Validated;
        format->second.gpu_kernel_state = SupportState::Validated;
        return Status::success();
    }
    if (kernel->validation != KernelValidation::Unverified)
        return {StatusCode::Unsupported, "Q4_1 MatMul candidate is rejected"};
    const std::string& registered_source = use_gles ? kernel->glsl_source : kernel->wgsl_source;
    if (registered_source.empty())
        return {StatusCode::Unsupported, "Q4_1 MatMul candidate has no source for its selected backend"};

    constexpr uint32_t rows = 2, output_channels = 2, inner = 64;
    std::vector<float> activations(rows * inner);
    for (size_t i = 0; i < activations.size(); ++i)
        activations[i] = (static_cast<int>(i % 9) - 4) * 0.25f;
    const uint64_t weight_bytes = static_cast<uint64_t>(output_channels) * (inner / 32) * 20;
    std::vector<uint8_t> encoded(static_cast<size_t>(weight_bytes), 0);
    for (uint32_t block = 0; block < output_channels * (inner / 32); ++block) {
        const size_t offset = static_cast<size_t>(block) * 20;
        encoded[offset] = 0x00;
        encoded[offset + 1] = 0x38; // FP16 scale 0.5
        encoded[offset + 2] = 0x00;
        encoded[offset + 3] = 0xbc; // FP16 minimum -1.0
        for (uint32_t i = 0; i < 16; ++i) {
            const uint8_t low = static_cast<uint8_t>(i % 16);
            const uint8_t high = static_cast<uint8_t>((i + 3) % 16);
            encoded[offset + 4 + i] = static_cast<uint8_t>(low | (high << 4));
        }
    }
    float expected[rows * output_channels] = {};
    float actual[rows * output_channels] = {};
    Status status = reference_q4_1_matmul(activations.data(), encoded.data(), weight_bytes,
        expected, rows, output_channels, inner);
    if (!status.ok()) return status;

    BufferDesc activation_desc{activations.size() * sizeof(float), alignof(float), MemoryDomain::Device,
        false, "q4_1-validation-activations"};
    BufferDesc weights_desc{(weight_bytes + 3) & ~uint64_t{3}, 4, MemoryDomain::Device,
        true, "q4_1-validation-weights"};
    BufferDesc output_desc{sizeof(actual), alignof(float), MemoryDomain::Device,
        false, "q4_1-validation-output"};
    std::shared_ptr<Buffer> activation_buffer, weights_buffer, output_buffer;
    status = backend.allocate(activation_desc, activation_buffer); if (!status.ok()) return status;
    status = backend.allocate(weights_desc, weights_buffer); if (!status.ok()) return status;
    status = backend.allocate(output_desc, output_buffer); if (!status.ok()) return status;
    status = backend.upload(activation_buffer, 0, activations.data(), activations.size() * sizeof(float));
    if (!status.ok()) return status;
    status = backend.upload(weights_buffer, 0, encoded.data(), weight_bytes);
    if (!status.ok()) return status;

    TensorDesc activation_tensor_desc; activation_tensor_desc.shape = {rows, inner};
    activation_tensor_desc.data_type = DataType::F32;
    TensorDesc output_tensor_desc; output_tensor_desc.shape = {rows, output_channels};
    output_tensor_desc.data_type = DataType::F32;
    Tensor activation_tensor, output_tensor;
    status = Tensor::create(activation_tensor_desc, activation_buffer, 0, activation_tensor);
    if (!status.ok()) return status;
    status = Tensor::create(output_tensor_desc, output_buffer, 0, output_tensor);
    if (!status.ok()) return status;

    ComputeDispatch dispatch;
    status = use_gles
        ? build_q4_1_matmul_gles_candidate(activation_tensor, weights_buffer, weight_bytes, output_tensor, dispatch)
        : build_q4_1_matmul_candidate(activation_tensor, weights_buffer, weight_bytes, output_tensor, dispatch);
    if (!status.ok()) return status;
    const std::string& candidate_source = use_gles ? dispatch.glsl_source : dispatch.wgsl_source;
    const std::string& canonical_source = use_gles ? q4_1_matmul_glsl_source()
                                                    : q4_1_matmul_wgsl_source();
    if (dispatch.kernel_id != kernel_id || registered_source != canonical_source ||
        candidate_source.empty())
        return {StatusCode::InvalidArgument, "Q4_1 MatMul validation source differs from the registered canonical candidate"};
    dispatch.validation = KernelValidation::Validated;
    std::unique_ptr<CommandBatch> commands;
    status = backend.begin_commands(commands); if (!status.ok()) return status;
    status = commands->dispatch_compute(dispatch); if (!status.ok()) return status;
    status = backend.submit(std::move(commands)); if (!status.ok()) return status;
    status = backend.synchronize(); if (!status.ok()) return status;
    status = backend.readback(output_buffer, 0, actual, sizeof(actual)); if (!status.ok()) return status;
    for (size_t i = 0; i < rows * output_channels; ++i) {
        const float tolerance = 1.0e-4f * std::max(1.0f, std::fabs(expected[i]));
        if (!std::isfinite(actual[i]) || std::fabs(actual[i] - expected[i]) > tolerance)
            return {StatusCode::NumericalFailure, "Q4_1 MatMul GPU output differs from the independent block-decoder reference"};
    }
    kernel->validation = KernelValidation::Validated;
    format->second.backend_gpu_kernel_states[backend_kind] = SupportState::Validated;
    format->second.gpu_kernel_state = SupportState::Validated;
    return Status::success();
}




Status build_q8_0_matmul_gles_candidate(const Tensor& activations,
                                       const std::shared_ptr<Buffer>& quantized_weights,
                                       uint64_t weight_bytes, const Tensor& output,
                                       ComputeDispatch& dispatch) {
    Status status = build_q8_0_matmul_candidate(activations, quantized_weights,
                                                weight_bytes, output, dispatch);
    if (!status.ok()) return status;
    const uint64_t blocks_per_row = activations.descriptor().shape[1] / 32;
    dispatch.backend = BackendKind::OpenGles;
    dispatch.kernel_id = "gles-q8-0-matmul-f32-activation-v1";
    dispatch.label = "gles-q8-0-matmul-f32-activation-unverified";
    dispatch.wgsl_source.clear();
    dispatch.glsl_source = kQ8_0MatMulGlsl;
    const std::pair<std::string, std::string> constants[] = {
        {"const uint ROWS = 1u;", "const uint ROWS = " + std::to_string(activations.descriptor().shape[0]) + "u;"},
        {"const uint OUTPUT_CHANNELS = 1u;", "const uint OUTPUT_CHANNELS = " + std::to_string(output.element_count() / activations.descriptor().shape[0]) + "u;"},
        {"const uint INNER = 32u;", "const uint INNER = " + std::to_string(activations.descriptor().shape[1]) + "u;"},
        {"const uint WEIGHT_ROW_BYTES = 34u;", "const uint WEIGHT_ROW_BYTES = " + std::to_string(blocks_per_row * 34) + "u;"}
    };
    for (const auto& constant : constants) {
        const size_t pos = dispatch.glsl_source.find(constant.first);
        if (pos == std::string::npos) return {StatusCode::BackendFailure, "Q8_0 GLES GLSL constant placeholder is missing"};
        dispatch.glsl_source.replace(pos, constant.first.size(), constant.second);
    }
    return Status::success();
}

Status build_q4_0_matmul_gles_candidate(const Tensor& activations,
                                       const std::shared_ptr<Buffer>& quantized_weights,
                                       uint64_t weight_bytes, const Tensor& output,
                                       ComputeDispatch& dispatch) {
    Status status = build_q4_0_matmul_candidate(activations, quantized_weights,
                                                weight_bytes, output, dispatch);
    if (!status.ok()) return status;
    const uint64_t blocks_per_row = activations.descriptor().shape[1] / 32;
    dispatch.backend = BackendKind::OpenGles;
    dispatch.kernel_id = "gles-q4-0-matmul-f32-activation-v1";
    dispatch.label = "gles-q4-0-matmul-f32-activation-unverified";
    dispatch.wgsl_source.clear();
    dispatch.glsl_source = kQ4_0MatMulGlsl;
    const std::pair<std::string, std::string> constants[] = {
        {"const uint ROWS = 1u;", "const uint ROWS = " + std::to_string(activations.descriptor().shape[0]) + "u;"},
        {"const uint OUTPUT_CHANNELS = 1u;", "const uint OUTPUT_CHANNELS = " + std::to_string(output.element_count() / activations.descriptor().shape[0]) + "u;"},
        {"const uint INNER = 32u;", "const uint INNER = " + std::to_string(activations.descriptor().shape[1]) + "u;"},
        {"const uint WEIGHT_ROW_BYTES = 18u;", "const uint WEIGHT_ROW_BYTES = " + std::to_string(blocks_per_row * 18) + "u;"}
    };
    for (const auto& constant : constants) {
        const size_t pos = dispatch.glsl_source.find(constant.first);
        if (pos == std::string::npos) return {StatusCode::BackendFailure, "Q4_0 GLES GLSL constant placeholder is missing"};
        dispatch.glsl_source.replace(pos, constant.first.size(), constant.second);
    }
    return Status::success();
}

Status reference_q8_0_matmul(const float* activations, const uint8_t* quantized_weights,
                             uint64_t weight_bytes, float* output,
                             uint32_t rows, uint32_t output_channels, uint32_t inner) {
    if ((!activations || !quantized_weights || !output) && rows && output_channels && inner)
        return {StatusCode::InvalidArgument, "Q8_0 MatMul reference received a null pointer"};
    if (rows == 0 || output_channels == 0 || inner == 0 || inner % 32 != 0)
        return {StatusCode::InvalidArgument, "Q8_0 MatMul dimensions must be non-zero and inner block-aligned"};
    const uint64_t blocks_per_row = inner / 32;
    if (blocks_per_row > UINT64_MAX / 34 ||
        output_channels > UINT64_MAX / (blocks_per_row * 34) ||
        weight_bytes != static_cast<uint64_t>(output_channels) * blocks_per_row * 34)
        return {StatusCode::InvalidArgument, "Q8_0 MatMul weight byte size is inconsistent"};
    std::vector<float> decoded(inner);
    for (uint32_t channel = 0; channel < output_channels; ++channel) {
        const uint8_t* row_weights = quantized_weights + static_cast<uint64_t>(channel) * blocks_per_row * 34;
        Status status = decode_q8_0_blocks(row_weights, blocks_per_row * 34, decoded.data(), inner);
        if (!status.ok()) return status;
        for (uint32_t row = 0; row < rows; ++row) {
            float sum = 0.0f;
            for (uint32_t k = 0; k < inner; ++k)
                sum += activations[static_cast<uint64_t>(row) * inner + k] * decoded[k];
            output[static_cast<uint64_t>(row) * output_channels + channel] = sum;
        }
    }
    return Status::success();
}

Status reference_q4_0_matmul(const float* activations, const uint8_t* quantized_weights,
                             uint64_t weight_bytes, float* output,
                             uint32_t rows, uint32_t output_channels, uint32_t inner) {
    if ((!activations || !quantized_weights || !output) && rows && output_channels && inner)
        return {StatusCode::InvalidArgument, "Q4_0 MatMul reference received a null pointer"};
    if (rows == 0 || output_channels == 0 || inner == 0 || inner % 32 != 0)
        return {StatusCode::InvalidArgument, "Q4_0 MatMul dimensions must be non-zero and inner block-aligned"};
    const uint64_t blocks_per_row = inner / 32;
    if (blocks_per_row > UINT64_MAX / 18 ||
        output_channels > UINT64_MAX / (blocks_per_row * 18) ||
        weight_bytes != static_cast<uint64_t>(output_channels) * blocks_per_row * 18)
        return {StatusCode::InvalidArgument, "Q4_0 MatMul weight byte size is inconsistent"};
    std::vector<float> decoded(inner);
    for (uint32_t channel = 0; channel < output_channels; ++channel) {
        const uint8_t* row_weights = quantized_weights + static_cast<uint64_t>(channel) * blocks_per_row * 18;
        Status status = decode_q4_0_blocks(row_weights, blocks_per_row * 18, decoded.data(), inner);
        if (!status.ok()) return status;
        for (uint32_t row = 0; row < rows; ++row) {
            float sum = 0.0f;
            for (uint32_t k = 0; k < inner; ++k)
                sum += activations[static_cast<uint64_t>(row) * inner + k] * decoded[k];
            output[static_cast<uint64_t>(row) * output_channels + channel] = sum;
        }
    }
    return Status::success();
}

Status reference_q4_1_matmul(const float* activations, const uint8_t* quantized_weights,
                             uint64_t weight_bytes, float* output,
                             uint32_t rows, uint32_t output_channels, uint32_t inner) {
    if ((!activations || !quantized_weights || !output) && rows && output_channels && inner)
        return {StatusCode::InvalidArgument, "Q4_1 MatMul reference received a null pointer"};
    if (rows == 0 || output_channels == 0 || inner == 0 || inner % 32 != 0)
        return {StatusCode::InvalidArgument, "Q4_1 MatMul dimensions must be non-zero and inner block-aligned"};
    const uint64_t blocks_per_row = inner / 32;
    if (blocks_per_row > UINT64_MAX / 20 ||
        output_channels > UINT64_MAX / (blocks_per_row * 20) ||
        weight_bytes != static_cast<uint64_t>(output_channels) * blocks_per_row * 20)
        return {StatusCode::InvalidArgument, "Q4_1 MatMul weight byte size is inconsistent"};
    std::vector<float> decoded(inner);
    for (uint32_t channel = 0; channel < output_channels; ++channel) {
        const uint8_t* row_weights = quantized_weights + static_cast<uint64_t>(channel) * blocks_per_row * 20;
        Status status = decode_q4_1_blocks(row_weights, blocks_per_row * 20, decoded.data(), inner);
        if (!status.ok()) return status;
        for (uint32_t row = 0; row < rows; ++row) {
            float sum = 0.0f;
            for (uint32_t k = 0; k < inner; ++k)
                sum += activations[static_cast<uint64_t>(row) * inner + k] * decoded[k];
            output[static_cast<uint64_t>(row) * output_channels + channel] = sum;
        }
    }
    return Status::success();
}


Status build_q4_1_matmul_gles_candidate(const Tensor& activations,
                                       const std::shared_ptr<Buffer>& quantized_weights,
                                       uint64_t weight_bytes, const Tensor& output,
                                       ComputeDispatch& dispatch) {
    Status status = build_q4_1_matmul_candidate(activations, quantized_weights,
                                                weight_bytes, output, dispatch);
    if (!status.ok()) return status;
    const uint64_t blocks_per_row = activations.descriptor().shape[1] / 32;
    dispatch.backend = BackendKind::OpenGles;
    dispatch.kernel_id = "gles-q4-1-matmul-f32-activation-v1";
    dispatch.label = "gles-q4-1-matmul-f32-activation-unverified";
    dispatch.wgsl_source.clear();
    dispatch.glsl_source = kQ4_1MatMulGlsl;
    const std::pair<std::string, std::string> constants[] = {
        {"const uint ROWS = 1u;", "const uint ROWS = " + std::to_string(activations.descriptor().shape[0]) + "u;"},
        {"const uint OUTPUT_CHANNELS = 1u;", "const uint OUTPUT_CHANNELS = " + std::to_string(output.element_count() / activations.descriptor().shape[0]) + "u;"},
        {"const uint INNER = 32u;", "const uint INNER = " + std::to_string(activations.descriptor().shape[1]) + "u;"},
        {"const uint WEIGHT_ROW_BYTES = 20u;", "const uint WEIGHT_ROW_BYTES = " + std::to_string(blocks_per_row * 20) + "u;"}
    };
    for (const auto& constant : constants) {
        const size_t pos = dispatch.glsl_source.find(constant.first);
        if (pos == std::string::npos) return {StatusCode::BackendFailure, "Q4_1 GLES GLSL constant placeholder is missing"};
        dispatch.glsl_source.replace(pos, constant.first.size(), constant.second);
    }
    return Status::success();
}

const char* q5_0_dequantize_wgsl_source() noexcept { return kQ5_0DequantizeWgsl; }

Status build_q5_0_dequantize_candidate(const std::shared_ptr<Buffer>& encoded,
                                       uint64_t encoded_bytes,
                                       const Tensor& output,
                                       ComputeDispatch& dispatch) {
    dispatch = ComputeDispatch{};
    if (!encoded || !output.buffer() || output.descriptor().data_type != DataType::F32 ||
        output.descriptor().layout != TensorLayout::Contiguous ||
        output.descriptor().memory == MemoryDomain::Host || output.descriptor().shape.empty())
        return {StatusCode::Unsupported, "Q5_0 GPU decoder requires a device/shared encoded buffer and contiguous F32 output"};
    constexpr uint64_t block_elements = 32, block_bytes = 22;
    if (output.element_count() == 0 || output.element_count() % block_elements != 0 ||
        output.element_count() > UINT32_MAX)
        return {StatusCode::InvalidArgument, "Q5_0 output must contain complete blocks and fit WGSL indexing"};
    const uint64_t blocks = output.element_count() / block_elements;
    if (blocks > UINT64_MAX / block_bytes || encoded_bytes != blocks * block_bytes ||
        encoded_bytes > static_cast<uint64_t>(UINT32_MAX) - 3)
        return {StatusCode::InvalidArgument, "Q5_0 encoded size does not match output elements"};
    const uint64_t padded = (encoded_bytes + 3) & ~uint64_t{3};
    if (encoded->size_bytes() < padded || output.byte_size() != output.element_count() * sizeof(float) ||
        encoded.get() == output.buffer().get())
        return {StatusCode::InvalidArgument, "Q5_0 input/output buffer ranges are inconsistent"};
    dispatch.label = "q5-0-dequantize-generic-unverified";
    dispatch.kernel_id = "dawn-q5-0-dequantize-generic-v1";
    dispatch.backend = BackendKind::DawnWebGpu;
    dispatch.wgsl_source = kQ5_0DequantizeWgsl;
    dispatch.workgroup_size_x = 64;
    const std::string placeholder = "const ELEMENT_COUNT: u32 = 32u;";
    const std::string replacement = "const ELEMENT_COUNT: u32 = " + std::to_string(output.element_count()) + "u;";
    const size_t position = dispatch.wgsl_source.find(placeholder);
    if (position == std::string::npos) return {StatusCode::BackendFailure, "Q5_0 WGSL placeholder missing"};
    dispatch.wgsl_source.replace(position, placeholder.size(), replacement);
    dispatch.workgroups_x = static_cast<uint32_t>((output.element_count() + 63) / 64);
    dispatch.bindings = {{0, encoded, 0, padded}, {1, output.buffer(), output.byte_offset(), output.byte_size()}};
    return Status::success();
}

const char* q5_1_dequantize_wgsl_source() noexcept { return kQ5_1DequantizeWgsl; }

Status build_q5_1_dequantize_candidate(const std::shared_ptr<Buffer>& encoded,
                                       uint64_t encoded_bytes,
                                       const Tensor& output,
                                       ComputeDispatch& dispatch) {
    dispatch = ComputeDispatch{};
    if (!encoded || !output.buffer() || output.descriptor().data_type != DataType::F32 ||
        output.descriptor().layout != TensorLayout::Contiguous ||
        output.descriptor().memory == MemoryDomain::Host || output.descriptor().shape.empty())
        return {StatusCode::Unsupported, "Q5_1 GPU decoder requires a device/shared encoded buffer and contiguous F32 output"};
    constexpr uint64_t block_elements = 32, block_bytes = 24;
    if (output.element_count() == 0 || output.element_count() % block_elements != 0 ||
        output.element_count() > UINT32_MAX)
        return {StatusCode::InvalidArgument, "Q5_1 output must contain complete blocks and fit WGSL indexing"};
    const uint64_t blocks = output.element_count() / block_elements;
    if (blocks > UINT64_MAX / block_bytes || encoded_bytes != blocks * block_bytes ||
        encoded_bytes > static_cast<uint64_t>(UINT32_MAX) - 3)
        return {StatusCode::InvalidArgument, "Q5_1 encoded size does not match output elements"};
    const uint64_t padded = (encoded_bytes + 3) & ~uint64_t{3};
    if (encoded->size_bytes() < padded || output.byte_size() != output.element_count() * sizeof(float) ||
        encoded.get() == output.buffer().get())
        return {StatusCode::InvalidArgument, "Q5_1 input/output buffer ranges are inconsistent"};
    dispatch.label = "q5-1-dequantize-generic-unverified";
    dispatch.kernel_id = "dawn-q5-1-dequantize-generic-v1";
    dispatch.backend = BackendKind::DawnWebGpu;
    dispatch.wgsl_source = kQ5_1DequantizeWgsl;
    dispatch.workgroup_size_x = 64;
    const std::string placeholder = "const ELEMENT_COUNT: u32 = 32u;";
    const std::string replacement = "const ELEMENT_COUNT: u32 = " + std::to_string(output.element_count()) + "u;";
    const size_t position = dispatch.wgsl_source.find(placeholder);
    if (position == std::string::npos) return {StatusCode::BackendFailure, "Q5_1 WGSL placeholder missing"};
    dispatch.wgsl_source.replace(position, placeholder.size(), replacement);
    dispatch.workgroups_x = static_cast<uint32_t>((output.element_count() + 63) / 64);
    dispatch.bindings = {{0, encoded, 0, padded}, {1, output.buffer(), output.byte_offset(), output.byte_size()}};
    return Status::success();
}

const char* q4_1_dequantize_wgsl_source() noexcept { return kQ4_1DequantizeWgsl; }

Status build_q4_1_dequantize_candidate(const std::shared_ptr<Buffer>& encoded,
                                       uint64_t encoded_bytes,
                                       const Tensor& output,
                                       ComputeDispatch& dispatch) {
    dispatch = ComputeDispatch{};
    if (!encoded || !output.buffer() || output.descriptor().data_type != DataType::F32 ||
        output.descriptor().layout != TensorLayout::Contiguous ||
        output.descriptor().memory == MemoryDomain::Host || output.descriptor().shape.empty())
        return {StatusCode::Unsupported, "Q4_1 GPU decoder requires a device/shared encoded buffer and contiguous F32 output"};
    constexpr uint64_t kBlockElements = 32;
    constexpr uint64_t kBlockBytes = 20;
    if (output.element_count() == 0 || output.element_count() % kBlockElements != 0 ||
        output.element_count() > UINT32_MAX)
        return {StatusCode::InvalidArgument, "Q4_1 output must contain complete blocks and fit the WGSL index range"};
    const uint64_t blocks = output.element_count() / kBlockElements;
    if (blocks > UINT64_MAX / kBlockBytes || encoded_bytes != blocks * kBlockBytes ||
        encoded_bytes > static_cast<uint64_t>(UINT32_MAX) - 3)
        return {StatusCode::InvalidArgument, "Q4_1 encoded byte count does not match the output tensor"};
    const uint64_t padded_bytes = (encoded_bytes + 3) & ~uint64_t{3};
    if (encoded->size_bytes() < padded_bytes ||
        output.byte_size() != output.element_count() * sizeof(float) ||
        encoded.get() == output.buffer().get())
        return {StatusCode::InvalidArgument, "Q4_1 input/output buffer ranges are inconsistent"};
    dispatch.label = "q4-1-dequantize-generic-unverified";
    dispatch.kernel_id = "dawn-q4-1-dequantize-generic-v1";
    dispatch.backend = BackendKind::DawnWebGpu;
    dispatch.wgsl_source = kQ4_1DequantizeWgsl;
    dispatch.workgroup_size_x = 64;
    const std::string placeholder = "const ELEMENT_COUNT: u32 = 32u;";
    const std::string replacement = "const ELEMENT_COUNT: u32 = " +
                                    std::to_string(output.element_count()) + "u;";
    const size_t position = dispatch.wgsl_source.find(placeholder);
    if (position == std::string::npos)
        return {StatusCode::BackendFailure, "Q4_1 WGSL element-count placeholder is missing"};
    dispatch.wgsl_source.replace(position, placeholder.size(), replacement);
    dispatch.workgroups_x = static_cast<uint32_t>((output.element_count() + 63) / 64);
    dispatch.bindings = {
        {0, encoded, 0, padded_bytes},
        {1, output.buffer(), output.byte_offset(), output.byte_size()},
    };
    return Status::success();
}

const char* q4_0_dequantize_wgsl_source() noexcept { return kQ4_0DequantizeWgsl; }

Status build_q4_0_dequantize_candidate(const std::shared_ptr<Buffer>& encoded,
                                       uint64_t encoded_bytes,
                                       const Tensor& output,
                                       ComputeDispatch& dispatch) {
    dispatch = ComputeDispatch{};
    if (!encoded || !output.buffer() || output.descriptor().data_type != DataType::F32 ||
        output.descriptor().layout != TensorLayout::Contiguous ||
        output.descriptor().memory == MemoryDomain::Host || output.descriptor().shape.empty())
        return {StatusCode::Unsupported, "Q4_0 GPU decoder requires a device/shared encoded buffer and contiguous F32 output"};
    constexpr uint64_t kBlockElements = 32;
    constexpr uint64_t kBlockBytes = 18;
    if (output.element_count() == 0 || output.element_count() % kBlockElements != 0 ||
        output.element_count() > UINT32_MAX)
        return {StatusCode::InvalidArgument, "Q4_0 output must contain complete blocks and fit the WGSL index range"};
    const uint64_t blocks = output.element_count() / kBlockElements;
    if (blocks > UINT64_MAX / kBlockBytes || encoded_bytes != blocks * kBlockBytes ||
        encoded_bytes > static_cast<uint64_t>(UINT32_MAX) - 3)
        return {StatusCode::InvalidArgument, "Q4_0 encoded byte count does not match the output tensor"};
    const uint64_t padded_bytes = (encoded_bytes + 3) & ~uint64_t{3};
    if (encoded->size_bytes() < padded_bytes ||
        output.byte_size() != output.element_count() * sizeof(float) ||
        encoded.get() == output.buffer().get())
        return {StatusCode::InvalidArgument, "Q4_0 input/output buffer ranges are inconsistent"};
    dispatch.label = "q4-0-dequantize-generic-unverified";
    dispatch.kernel_id = "dawn-q4-0-dequantize-generic-v1";
    dispatch.backend = BackendKind::DawnWebGpu;
    dispatch.wgsl_source = kQ4_0DequantizeWgsl;
    dispatch.workgroup_size_x = 64;
    const std::string placeholder = "const ELEMENT_COUNT: u32 = 32u;";
    const std::string replacement = "const ELEMENT_COUNT: u32 = " +
                                    std::to_string(output.element_count()) + "u;";
    const size_t position = dispatch.wgsl_source.find(placeholder);
    if (position == std::string::npos)
        return {StatusCode::BackendFailure, "Q4_0 WGSL element-count placeholder is missing"};
    dispatch.wgsl_source.replace(position, placeholder.size(), replacement);
    dispatch.workgroups_x = static_cast<uint32_t>((output.element_count() + 63) / 64);
    dispatch.bindings = {
        {0, encoded, 0, padded_bytes},
        {1, output.buffer(), output.byte_offset(), output.byte_size()},
    };
    return Status::success();
}

const char* q8_0_dequantize_wgsl_source() noexcept { return kQ8_0DequantizeWgsl; }

Status build_q8_0_dequantize_candidate(const std::shared_ptr<Buffer>& encoded,
                                       uint64_t encoded_bytes,
                                       const Tensor& output,
                                       ComputeDispatch& dispatch) {
    dispatch = ComputeDispatch{};
    if (!encoded || !output.buffer() || output.descriptor().data_type != DataType::F32 ||
        output.descriptor().layout != TensorLayout::Contiguous ||
        output.descriptor().memory == MemoryDomain::Host || output.descriptor().shape.empty()) {
        return {StatusCode::Unsupported, "Q8_0 GPU decoder requires a device/shared encoded buffer and contiguous F32 output"};
    }
    constexpr uint64_t kBlockElements = 32;
    constexpr uint64_t kBlockBytes = 34;
    if (output.element_count() == 0 || output.element_count() % kBlockElements != 0 ||
        output.element_count() > UINT32_MAX) {
        return {StatusCode::InvalidArgument, "Q8_0 output must contain complete blocks and fit the WGSL index range"};
    }
    const uint64_t blocks = output.element_count() / kBlockElements;
    if (blocks > UINT64_MAX / kBlockBytes || encoded_bytes != blocks * kBlockBytes ||
        encoded_bytes > static_cast<uint64_t>(UINT32_MAX) - 3) {
        return {StatusCode::InvalidArgument, "Q8_0 encoded byte count does not match the output tensor"};
    }
    const uint64_t padded_bytes = (encoded_bytes + 3) & ~uint64_t{3};
    if (encoded->size_bytes() < padded_bytes ||
        output.byte_size() != output.element_count() * sizeof(float) ||
        encoded.get() == output.buffer().get()) {
        return {StatusCode::InvalidArgument, "Q8_0 input/output buffer ranges are inconsistent"};
    }
    dispatch.label = "q8-0-dequantize-generic-unverified";
    dispatch.kernel_id = "dawn-q8-0-dequantize-generic-v1";
    dispatch.backend = BackendKind::DawnWebGpu;
    dispatch.wgsl_source = kQ8_0DequantizeWgsl;
    dispatch.workgroup_size_x = 64;
    const std::string placeholder = "const ELEMENT_COUNT: u32 = 32u;";
    const std::string replacement = "const ELEMENT_COUNT: u32 = " +
                                    std::to_string(output.element_count()) + "u;";
    const size_t position = dispatch.wgsl_source.find(placeholder);
    if (position == std::string::npos)
        return {StatusCode::BackendFailure, "Q8_0 WGSL element-count placeholder is missing"};
    dispatch.wgsl_source.replace(position, placeholder.size(), replacement);
    dispatch.workgroups_x = static_cast<uint32_t>((output.element_count() + 63) / 64);
    dispatch.bindings = {
        {0, encoded, 0, padded_bytes},
        {1, output.buffer(), output.byte_offset(), output.byte_size()},
    };
    return Status::success();
}


Status register_builtin_quantized_operator_candidates(OperatorRegistry& registry) {
    KernelDefinition q8;
    q8.operator_name = "MatMul_Q8_0";
    q8.backend = BackendKind::DawnWebGpu;
    q8.input_type = DataType::F32;
    q8.input_types = {DataType::F32, DataType::U32};
    q8.output_type = DataType::F32;
    q8.min_input_count = 2;
    q8.max_input_count = 2;
    q8.min_rank = 2;
    q8.max_rank = 3;
    q8.validation = KernelValidation::Unverified;
    q8.kernel_id = "dawn-q8-0-matmul-f32-activation-v1";
    q8.wgsl_source = q8_0_matmul_wgsl_source();
    Status status = registry.register_kernel(std::move(q8));
    if (!status.ok()) return status;

    KernelDefinition q8_gles;
    q8_gles.operator_name = "MatMul_Q8_0";
    q8_gles.backend = BackendKind::OpenGles;
    q8_gles.input_type = DataType::F32;
    q8_gles.input_types = {DataType::F32, DataType::U32};
    q8_gles.output_type = DataType::F32;
    q8_gles.min_input_count = 2;
    q8_gles.max_input_count = 2;
    q8_gles.min_rank = 2;
    q8_gles.max_rank = 3;
    q8_gles.validation = KernelValidation::Unverified;
    q8_gles.kernel_id = "gles-q8-0-matmul-f32-activation-v1";
    q8_gles.glsl_source = q8_0_matmul_glsl_source();
    status = registry.register_kernel(std::move(q8_gles));
    if (!status.ok()) return status;

    KernelDefinition q4;
    q4.operator_name = "MatMul_Q4_0";
    q4.backend = BackendKind::DawnWebGpu;
    q4.input_type = DataType::F32;
    q4.input_types = {DataType::F32, DataType::U32};
    q4.output_type = DataType::F32;
    q4.min_input_count = 2;
    q4.max_input_count = 2;
    q4.min_rank = 2;
    q4.max_rank = 3;
    q4.validation = KernelValidation::Unverified;
    q4.kernel_id = "dawn-q4-0-matmul-f32-activation-v1";
    q4.wgsl_source = q4_0_matmul_wgsl_source();
    status = registry.register_kernel(std::move(q4));
    if (!status.ok()) return status;

    KernelDefinition q4_gles;
    q4_gles.operator_name = "MatMul_Q4_0";
    q4_gles.backend = BackendKind::OpenGles;
    q4_gles.input_type = DataType::F32;
    q4_gles.input_types = {DataType::F32, DataType::U32};
    q4_gles.output_type = DataType::F32;
    q4_gles.min_input_count = 2;
    q4_gles.max_input_count = 2;
    q4_gles.min_rank = 2;
    q4_gles.max_rank = 3;
    q4_gles.validation = KernelValidation::Unverified;
    q4_gles.kernel_id = "gles-q4-0-matmul-f32-activation-v1";
    q4_gles.glsl_source = q4_0_matmul_glsl_source();
    status = registry.register_kernel(std::move(q4_gles));
    if (!status.ok()) return status;

    KernelDefinition q4_embedding;
    q4_embedding.operator_name = "Embedding_Q4_0";
    q4_embedding.backend = BackendKind::DawnWebGpu;
    q4_embedding.input_type = DataType::U32;
    q4_embedding.input_types = {DataType::U32, DataType::U32};
    q4_embedding.output_type = DataType::F32;
    q4_embedding.min_input_count = 2;
    q4_embedding.max_input_count = 2;
    q4_embedding.min_rank = 2;
    q4_embedding.max_rank = 2;
    q4_embedding.validation = KernelValidation::Unverified;
    q4_embedding.kernel_id = "dawn-q4-0-embedding-f32-v1";
    q4_embedding.wgsl_source = q4_0_embedding_wgsl_source();
    status = registry.register_kernel(std::move(q4_embedding));
    if (!status.ok()) return status;

    KernelDefinition q4_embedding_gles;
    q4_embedding_gles.operator_name = "Embedding_Q4_0";
    q4_embedding_gles.backend = BackendKind::OpenGles;
    q4_embedding_gles.input_type = DataType::U32;
    q4_embedding_gles.input_types = {DataType::U32, DataType::U32};
    q4_embedding_gles.output_type = DataType::F32;
    q4_embedding_gles.min_input_count = 2;
    q4_embedding_gles.max_input_count = 2;
    q4_embedding_gles.min_rank = 2;
    q4_embedding_gles.max_rank = 2;
    q4_embedding_gles.validation = KernelValidation::Unverified;
    q4_embedding_gles.kernel_id = "gles-q4-0-embedding-f32-v1";
    q4_embedding_gles.glsl_source = q4_0_embedding_glsl_source();
    status = registry.register_kernel(std::move(q4_embedding_gles));
    if (!status.ok()) return status;

    KernelDefinition q4_1_embedding;
    q4_1_embedding.operator_name = "Embedding_Q4_1";
    q4_1_embedding.backend = BackendKind::DawnWebGpu;
    q4_1_embedding.input_type = DataType::U32;
    q4_1_embedding.input_types = {DataType::U32, DataType::U32};
    q4_1_embedding.output_type = DataType::F32;
    q4_1_embedding.min_input_count = 2;
    q4_1_embedding.max_input_count = 2;
    q4_1_embedding.min_rank = 2;
    q4_1_embedding.max_rank = 2;
    q4_1_embedding.validation = KernelValidation::Unverified;
    q4_1_embedding.kernel_id = "dawn-q4-1-embedding-f32-v1";
    q4_1_embedding.wgsl_source = q4_1_embedding_wgsl_source();
    status = registry.register_kernel(std::move(q4_1_embedding));
    if (!status.ok()) return status;

    KernelDefinition q4_1_embedding_gles;
    q4_1_embedding_gles.operator_name = "Embedding_Q4_1";
    q4_1_embedding_gles.backend = BackendKind::OpenGles;
    q4_1_embedding_gles.input_type = DataType::U32;
    q4_1_embedding_gles.input_types = {DataType::U32, DataType::U32};
    q4_1_embedding_gles.output_type = DataType::F32;
    q4_1_embedding_gles.min_input_count = 2;
    q4_1_embedding_gles.max_input_count = 2;
    q4_1_embedding_gles.min_rank = 2;
    q4_1_embedding_gles.max_rank = 2;
    q4_1_embedding_gles.validation = KernelValidation::Unverified;
    q4_1_embedding_gles.kernel_id = "gles-q4-1-embedding-f32-v1";
    q4_1_embedding_gles.glsl_source = q4_1_embedding_glsl_source();
    status = registry.register_kernel(std::move(q4_1_embedding_gles));
    if (!status.ok()) return status;

    KernelDefinition q8_embedding;
    q8_embedding.operator_name = "Embedding_Q8_0";
    q8_embedding.backend = BackendKind::DawnWebGpu;
    q8_embedding.input_type = DataType::U32;
    q8_embedding.input_types = {DataType::U32, DataType::U32};
    q8_embedding.output_type = DataType::F32;
    q8_embedding.min_input_count = 2;
    q8_embedding.max_input_count = 2;
    q8_embedding.min_rank = 2;
    q8_embedding.max_rank = 2;
    q8_embedding.validation = KernelValidation::Unverified;
    q8_embedding.kernel_id = "dawn-q8-0-embedding-f32-v1";
    q8_embedding.wgsl_source = q8_0_embedding_wgsl_source();
    status = registry.register_kernel(std::move(q8_embedding));
    if (!status.ok()) return status;

    KernelDefinition q8_embedding_gles;
    q8_embedding_gles.operator_name = "Embedding_Q8_0";
    q8_embedding_gles.backend = BackendKind::OpenGles;
    q8_embedding_gles.input_type = DataType::U32;
    q8_embedding_gles.input_types = {DataType::U32, DataType::U32};
    q8_embedding_gles.output_type = DataType::F32;
    q8_embedding_gles.min_input_count = 2;
    q8_embedding_gles.max_input_count = 2;
    q8_embedding_gles.min_rank = 2;
    q8_embedding_gles.max_rank = 2;
    q8_embedding_gles.validation = KernelValidation::Unverified;
    q8_embedding_gles.kernel_id = "gles-q8-0-embedding-f32-v1";
    q8_embedding_gles.glsl_source = q8_0_embedding_glsl_source();
    status = registry.register_kernel(std::move(q8_embedding_gles));
    if (!status.ok()) return status;

    KernelDefinition q4_1;
    q4_1.operator_name = "MatMul_Q4_1";
    q4_1.backend = BackendKind::DawnWebGpu;
    q4_1.input_type = DataType::F32;
    q4_1.input_types = {DataType::F32, DataType::U32};
    q4_1.output_type = DataType::F32;
    q4_1.min_input_count = 2;
    q4_1.max_input_count = 2;
    q4_1.min_rank = 2;
    q4_1.max_rank = 3;
    q4_1.validation = KernelValidation::Unverified;
    q4_1.kernel_id = "dawn-q4-1-matmul-f32-activation-v1";
    q4_1.wgsl_source = q4_1_matmul_wgsl_source();
    status = registry.register_kernel(std::move(q4_1));
    if (!status.ok()) return status;

    KernelDefinition q4_1_gles;
    q4_1_gles.operator_name = "MatMul_Q4_1";
    q4_1_gles.backend = BackendKind::OpenGles;
    q4_1_gles.input_type = DataType::F32;
    q4_1_gles.input_types = {DataType::F32, DataType::U32};
    q4_1_gles.output_type = DataType::F32;
    q4_1_gles.min_input_count = 2;
    q4_1_gles.max_input_count = 2;
    q4_1_gles.min_rank = 2;
    q4_1_gles.max_rank = 3;
    q4_1_gles.validation = KernelValidation::Unverified;
    q4_1_gles.kernel_id = "gles-q4-1-matmul-f32-activation-v1";
    q4_1_gles.glsl_source = q4_1_matmul_glsl_source();
    status = registry.register_kernel(std::move(q4_1_gles));
    if (!status.ok()) return status;




    for(const bool q3:{false,true}){
        const std::string suffix=q3?"q3-k":"q2-k",op=q3?"MatMul_Q3_K":"MatMul_Q2_K";
        const char*wgsl=q3?kQ3_KMatMulWgsl:kQ2_KMatMulWgsl;const char*glsl=q3?kQ3_KMatMulGlsl:kQ2_KMatMulGlsl;
        KernelDefinition dawn;dawn.operator_name=op;dawn.backend=BackendKind::DawnWebGpu;
        dawn.input_type=DataType::F32;dawn.input_types={DataType::F32,DataType::U32};dawn.output_type=DataType::F32;
        dawn.min_input_count=2;dawn.max_input_count=2;dawn.min_rank=2;dawn.max_rank=3;
        dawn.validation=KernelValidation::Unverified;dawn.kernel_id="dawn-"+suffix+"-matmul-f32-activation-v1";dawn.wgsl_source=wgsl;
        status=registry.register_kernel(std::move(dawn));if(!status.ok())return status;
        KernelDefinition gles;gles.operator_name=op;gles.backend=BackendKind::OpenGles;
        gles.input_type=DataType::F32;gles.input_types={DataType::F32,DataType::U32};gles.output_type=DataType::F32;
        gles.min_input_count=2;gles.max_input_count=2;gles.min_rank=2;gles.max_rank=3;
        gles.validation=KernelValidation::Unverified;gles.kernel_id="gles-"+suffix+"-matmul-f32-activation-v1";gles.glsl_source=glsl;
        status=registry.register_kernel(std::move(gles));if(!status.ok())return status;
    }

    for(const bool q5:{false,true}){
        const std::string suffix=q5?"q5-k":"q4-k",op=q5?"MatMul_Q5_K":"MatMul_Q4_K";
        const char*wgsl=q5?kQ5_KMatMulWgsl:kQ4_KMatMulWgsl;
        const char*glsl=q5?kQ5_KMatMulGlsl:kQ4_KMatMulGlsl;
        KernelDefinition dawn;dawn.operator_name=op;dawn.backend=BackendKind::DawnWebGpu;
        dawn.input_type=DataType::F32;dawn.input_types={DataType::F32,DataType::U32};dawn.output_type=DataType::F32;
        dawn.min_input_count=2;dawn.max_input_count=2;dawn.min_rank=2;dawn.max_rank=3;
        dawn.validation=KernelValidation::Unverified;dawn.kernel_id="dawn-"+suffix+"-matmul-f32-activation-v1";dawn.wgsl_source=wgsl;
        status=registry.register_kernel(std::move(dawn));if(!status.ok())return status;
        KernelDefinition gles;gles.operator_name=op;gles.backend=BackendKind::OpenGles;
        gles.input_type=DataType::F32;gles.input_types={DataType::F32,DataType::U32};gles.output_type=DataType::F32;
        gles.min_input_count=2;gles.max_input_count=2;gles.min_rank=2;gles.max_rank=3;
        gles.validation=KernelValidation::Unverified;gles.kernel_id="gles-"+suffix+"-matmul-f32-activation-v1";gles.glsl_source=glsl;
        status=registry.register_kernel(std::move(gles));if(!status.ok())return status;
    }

    KernelDefinition q6_k;
    q6_k.operator_name="MatMul_Q6_K";q6_k.backend=BackendKind::DawnWebGpu;
    q6_k.input_type=DataType::F32;q6_k.input_types={DataType::F32,DataType::U32};q6_k.output_type=DataType::F32;
    q6_k.min_input_count=2;q6_k.max_input_count=2;q6_k.min_rank=2;q6_k.max_rank=3;
    q6_k.validation=KernelValidation::Unverified;q6_k.kernel_id="dawn-q6-k-matmul-f32-activation-v1";q6_k.wgsl_source=kQ6_KMatMulWgsl;
    status=registry.register_kernel(std::move(q6_k));if(!status.ok())return status;
    KernelDefinition q6_k_gles;
    q6_k_gles.operator_name="MatMul_Q6_K";q6_k_gles.backend=BackendKind::OpenGles;
    q6_k_gles.input_type=DataType::F32;q6_k_gles.input_types={DataType::F32,DataType::U32};q6_k_gles.output_type=DataType::F32;
    q6_k_gles.min_input_count=2;q6_k_gles.max_input_count=2;q6_k_gles.min_rank=2;q6_k_gles.max_rank=3;
    q6_k_gles.validation=KernelValidation::Unverified;q6_k_gles.kernel_id="gles-q6-k-matmul-f32-activation-v1";q6_k_gles.glsl_source=kQ6_KMatMulGlsl;
    status=registry.register_kernel(std::move(q6_k_gles));if(!status.ok())return status;

    for (const bool bf16 : {false,true}) {
        const std::string suffix=bf16?"bf16":"f16";
        const std::string matmul=bf16?"MatMul_BF16":"MatMul_F16";
        const std::string embedding=bf16?"Embedding_BF16":"Embedding_F16";
        const char* mw=bf16?kBF16MatMulWgsl:kF16MatMulWgsl;
        const char* mg=bf16?kBF16MatMulGlsl:kF16MatMulGlsl;
        const char* ew=bf16?kBF16EmbeddingWgsl:kF16EmbeddingWgsl;
        const char* eg=bf16?kBF16EmbeddingGlsl:kF16EmbeddingGlsl;
        for(const bool gles:{false,true}){
            KernelDefinition m;
            m.operator_name=matmul;m.backend=gles?BackendKind::OpenGles:BackendKind::DawnWebGpu;
            m.input_type=DataType::F32;m.input_types={DataType::F32,DataType::U32};m.output_type=DataType::F32;
            m.min_input_count=2;m.max_input_count=2;m.min_rank=2;m.max_rank=3;
            m.validation=KernelValidation::Unverified;m.kernel_id=(gles?"gles-":"dawn-")+suffix+"-matmul-f32-activation-v1";
            if(gles)m.glsl_source=mg;else m.wgsl_source=mw;
            status=registry.register_kernel(std::move(m));if(!status.ok())return status;
            KernelDefinition e;
            e.operator_name=embedding;e.backend=gles?BackendKind::OpenGles:BackendKind::DawnWebGpu;
            e.input_type=DataType::U32;e.input_types={DataType::U32,DataType::U32};e.output_type=DataType::F32;
            e.min_input_count=2;e.max_input_count=2;e.min_rank=2;e.max_rank=2;
            e.validation=KernelValidation::Unverified;e.kernel_id=(gles?"gles-":"dawn-")+suffix+"-embedding-f32-v1";
            if(gles)e.glsl_source=eg;else e.wgsl_source=ew;
            status=registry.register_kernel(std::move(e));if(!status.ok())return status;
        }
    }

    for (const bool q5_1 : {false, true}) {
        const std::string suffix = q5_1 ? "q5-1" : "q5-0";
        const std::string op = q5_1 ? "MatMul_Q5_1" : "MatMul_Q5_0";
        const char* wgsl = q5_1 ? q5_1_matmul_wgsl_source() : q5_0_matmul_wgsl_source();
        const char* glsl = q5_1 ? q5_1_matmul_glsl_source() : q5_0_matmul_glsl_source();
        KernelDefinition dawn; dawn.operator_name=op; dawn.backend=BackendKind::DawnWebGpu;
        dawn.input_type=DataType::F32; dawn.input_types={DataType::F32,DataType::U32}; dawn.output_type=DataType::F32;
        dawn.min_input_count=2; dawn.max_input_count=2; dawn.min_rank=2; dawn.max_rank=3;
        dawn.validation=KernelValidation::Unverified; dawn.kernel_id="dawn-"+suffix+"-matmul-f32-activation-v1"; dawn.wgsl_source=wgsl;
        status=registry.register_kernel(std::move(dawn)); if(!status.ok())return status;
        KernelDefinition gles; gles.operator_name=op; gles.backend=BackendKind::OpenGles;
        gles.input_type=DataType::F32; gles.input_types={DataType::F32,DataType::U32}; gles.output_type=DataType::F32;
        gles.min_input_count=2; gles.max_input_count=2; gles.min_rank=2; gles.max_rank=3;
        gles.validation=KernelValidation::Unverified; gles.kernel_id="gles-"+suffix+"-matmul-f32-activation-v1"; gles.glsl_source=glsl;
        status=registry.register_kernel(std::move(gles)); if(!status.ok())return status;

        const std::string embedding_op=q5_1?"Embedding_Q5_1":"Embedding_Q5_0";
        const char* embedding_wgsl=q5_1?q5_1_embedding_wgsl_source():q5_0_embedding_wgsl_source();
        const char* embedding_glsl=q5_1?q5_1_embedding_glsl_source():q5_0_embedding_glsl_source();
        KernelDefinition embedding_dawn;
        embedding_dawn.operator_name=embedding_op; embedding_dawn.backend=BackendKind::DawnWebGpu;
        embedding_dawn.input_type=DataType::U32; embedding_dawn.input_types={DataType::U32,DataType::U32};
        embedding_dawn.output_type=DataType::F32; embedding_dawn.min_input_count=2; embedding_dawn.max_input_count=2;
        embedding_dawn.min_rank=2; embedding_dawn.max_rank=2; embedding_dawn.validation=KernelValidation::Unverified;
        embedding_dawn.kernel_id="dawn-"+suffix+"-embedding-f32-v1"; embedding_dawn.wgsl_source=embedding_wgsl;
        status=registry.register_kernel(std::move(embedding_dawn)); if(!status.ok())return status;
        KernelDefinition embedding_gles;
        embedding_gles.operator_name=embedding_op; embedding_gles.backend=BackendKind::OpenGles;
        embedding_gles.input_type=DataType::U32; embedding_gles.input_types={DataType::U32,DataType::U32};
        embedding_gles.output_type=DataType::F32; embedding_gles.min_input_count=2; embedding_gles.max_input_count=2;
        embedding_gles.min_rank=2; embedding_gles.max_rank=2; embedding_gles.validation=KernelValidation::Unverified;
        embedding_gles.kernel_id="gles-"+suffix+"-embedding-f32-v1"; embedding_gles.glsl_source=embedding_glsl;
        status=registry.register_kernel(std::move(embedding_gles)); if(!status.ok())return status;
    }
    return Status::success();
}

ModelQuantizationBackendReport evaluate_model_quantization_backend(
        const GgufFileInfo& model, const QuantizationRegistry& registry, BackendKind backend) {
    ModelQuantizationBackendReport report;
    report.backend = backend;
    if (backend != BackendKind::DawnWebGpu && backend != BackendKind::OpenGles) {
        report.blockers.emplace_back("UNSUPPORTED_OR_UNKNOWN_GPU_BACKEND");
        return report;
    }
    for (const auto& tensor : model.tensors) {
        const GgufTensorTypeInfo type = GgufReader::tensor_type_info(tensor.type_id);
        if (!type.storage_size_known) {
            report.blockers.push_back("UNKNOWN_TENSOR_STORAGE_TYPE:" + tensor.name);
            report.unsupported_file_type_ids.push_back(tensor.type_id);
            continue;
        }
        const bool requires_packed_gpu_kernel = type.quantized || type.name == "F16" || type.name == "BF16";
        if (!requires_packed_gpu_kernel) continue;
        const QuantizationFormat* format = registry.find(tensor.type_id);
        if (!format || format->decoder_state != SupportState::Validated) {
            report.blockers.push_back("QUANTIZATION_DECODER_NOT_VALIDATED:" + tensor.name);
            report.unsupported_file_type_ids.push_back(tensor.type_id);
            continue;
        }
        if (format->block_elements != type.block_elements ||
            format->bytes_per_block != type.bytes_per_block) {
            report.blockers.push_back("QUANTIZATION_FORMAT_METADATA_MISMATCH:" + tensor.name);
            report.unsupported_file_type_ids.push_back(tensor.type_id);
            continue;
        }
        if (!registry.can_execute_on_gpu(tensor.type_id, backend)) {
            report.blockers.push_back("GPU_QUANTIZATION_KERNEL_NOT_VALIDATED:" + tensor.name +
                ":backend=" + std::to_string(static_cast<unsigned>(backend)));
            report.unsupported_file_type_ids.push_back(tensor.type_id);
        }
    }
    std::sort(report.unsupported_file_type_ids.begin(), report.unsupported_file_type_ids.end());
    report.unsupported_file_type_ids.erase(
        std::unique(report.unsupported_file_type_ids.begin(), report.unsupported_file_type_ids.end()),
        report.unsupported_file_type_ids.end());
    report.compatible = report.blockers.empty();
    return report;
}

Status register_builtin_quantization_formats(QuantizationRegistry& registry) {
    const auto initialize_backend_states = [](QuantizationFormat& format) {
        format.backend_gpu_kernel_states[BackendKind::DawnWebGpu] = SupportState::Unverified;
        format.backend_gpu_kernel_states[BackendKind::OpenGles] = SupportState::Unverified;
    };

    QuantizationFormat f16;
    f16.file_type_id=1;f16.name="F16";f16.block_elements=1;f16.bytes_per_block=2;
    f16.quantized=false;f16.decoder_state=SupportState::Validated;f16.gpu_kernel_state=SupportState::Unverified;
    initialize_backend_states(f16);
    Status status=registry.register_format(std::move(f16));if(!status.ok())return status;

    QuantizationFormat bf16;
    bf16.file_type_id=30;bf16.name="BF16";bf16.block_elements=1;bf16.bytes_per_block=2;
    bf16.quantized=false;bf16.decoder_state=SupportState::Validated;bf16.gpu_kernel_state=SupportState::Unverified;
    initialize_backend_states(bf16);
    status=registry.register_format(std::move(bf16));if(!status.ok())return status;

    QuantizationFormat q4_0;
    q4_0.file_type_id = 2;
    q4_0.name = "Q4_0";
    q4_0.block_elements = 32;
    q4_0.bytes_per_block = 18;
    q4_0.quantized = true;
    q4_0.decoder_state = SupportState::Validated;
    q4_0.gpu_kernel_state = SupportState::Unverified;
    initialize_backend_states(q4_0);
    status = registry.register_format(std::move(q4_0));
    if (!status.ok()) return status;

    QuantizationFormat q4_1;
    q4_1.file_type_id = 3;
    q4_1.name = "Q4_1";
    q4_1.block_elements = 32;
    q4_1.bytes_per_block = 20;
    q4_1.quantized = true;
    q4_1.decoder_state = SupportState::Validated;
    q4_1.gpu_kernel_state = SupportState::Unverified;
    initialize_backend_states(q4_1);
    status = registry.register_format(std::move(q4_1));
    if (!status.ok()) return status;

    QuantizationFormat q5_0;
    q5_0.file_type_id = 6;
    q5_0.name = "Q5_0";
    q5_0.block_elements = 32;
    q5_0.bytes_per_block = 22;
    q5_0.quantized = true;
    q5_0.decoder_state = SupportState::Validated;
    q5_0.gpu_kernel_state = SupportState::Unverified;
    initialize_backend_states(q5_0);
    status = registry.register_format(std::move(q5_0));
    if (!status.ok()) return status;

    QuantizationFormat q5_1;
    q5_1.file_type_id = 7;
    q5_1.name = "Q5_1";
    q5_1.block_elements = 32;
    q5_1.bytes_per_block = 24;
    q5_1.quantized = true;
    q5_1.decoder_state = SupportState::Validated;
    q5_1.gpu_kernel_state = SupportState::Unverified;
    initialize_backend_states(q5_1);
    status = registry.register_format(std::move(q5_1));
    if (!status.ok()) return status;




    QuantizationFormat q2_k;
    q2_k.file_type_id=10;q2_k.name="Q2_K";q2_k.block_elements=256;q2_k.bytes_per_block=84;
    q2_k.quantized=true;q2_k.decoder_state=SupportState::Validated;q2_k.gpu_kernel_state=SupportState::Unverified;
    initialize_backend_states(q2_k);
    status=registry.register_format(std::move(q2_k));if(!status.ok())return status;

    QuantizationFormat q3_k;
    q3_k.file_type_id=11;q3_k.name="Q3_K";q3_k.block_elements=256;q3_k.bytes_per_block=110;
    q3_k.quantized=true;q3_k.decoder_state=SupportState::Validated;q3_k.gpu_kernel_state=SupportState::Unverified;
    initialize_backend_states(q3_k);
    status=registry.register_format(std::move(q3_k));if(!status.ok())return status;

    QuantizationFormat q4_k;
    q4_k.file_type_id=12;q4_k.name="Q4_K";q4_k.block_elements=256;q4_k.bytes_per_block=144;
    q4_k.quantized=true;q4_k.decoder_state=SupportState::Validated;q4_k.gpu_kernel_state=SupportState::Unverified;
    initialize_backend_states(q4_k);
    status=registry.register_format(std::move(q4_k));if(!status.ok())return status;

    QuantizationFormat q5_k;
    q5_k.file_type_id=13;q5_k.name="Q5_K";q5_k.block_elements=256;q5_k.bytes_per_block=176;
    q5_k.quantized=true;q5_k.decoder_state=SupportState::Validated;q5_k.gpu_kernel_state=SupportState::Unverified;
    initialize_backend_states(q5_k);
    status=registry.register_format(std::move(q5_k));if(!status.ok())return status;

    QuantizationFormat q6_k;
    q6_k.file_type_id=14;q6_k.name="Q6_K";q6_k.block_elements=256;q6_k.bytes_per_block=210;
    q6_k.quantized=true;q6_k.decoder_state=SupportState::Validated;q6_k.gpu_kernel_state=SupportState::Unverified;
    initialize_backend_states(q6_k);
    status=registry.register_format(std::move(q6_k));if(!status.ok())return status;

    QuantizationFormat q8;
    q8.file_type_id = 8;
    q8.name = "Q8_0";
    q8.block_elements = 32;
    q8.bytes_per_block = 34;
    q8.quantized = true;
    q8.decoder_state = SupportState::Validated;
    q8.gpu_kernel_state = SupportState::Unverified;
    initialize_backend_states(q8);
    return registry.register_format(std::move(q8));
}



const char* q5_0_embedding_wgsl_source() noexcept { return kQ5_0EmbeddingWgsl; }
const char* q5_0_embedding_glsl_source() noexcept { return kQ5_0EmbeddingGlsl; }
const char* q5_1_embedding_wgsl_source() noexcept { return kQ5_1EmbeddingWgsl; }
const char* q5_1_embedding_glsl_source() noexcept { return kQ5_1EmbeddingGlsl; }

const char* q5_0_matmul_wgsl_source() noexcept { return kQ5_0MatMulWgsl; }
const char* q5_0_matmul_glsl_source() noexcept { return kQ5_0MatMulGlsl; }
const char* q5_1_matmul_wgsl_source() noexcept { return kQ5_1MatMulWgsl; }
const char* q5_1_matmul_glsl_source() noexcept { return kQ5_1MatMulGlsl; }

const char* q5_0_dequantize_glsl_source() noexcept { return kQ5_0DequantizeGlsl; }
const char* q5_1_dequantize_glsl_source() noexcept { return kQ5_1DequantizeGlsl; }

Status build_q5_0_dequantize_gles_candidate(const std::shared_ptr<Buffer>& encoded,
                                            uint64_t encoded_bytes,
                                            const Tensor& output,
                                            ComputeDispatch& dispatch) {
    Status status = build_q5_0_dequantize_candidate(encoded, encoded_bytes, output, dispatch);
    if (!status.ok()) return status;
    dispatch.label = "q5-0-dequantize-gles-generic-unverified";
    dispatch.kernel_id = "gles-q5-0-dequantize-generic-v1";
    dispatch.backend = BackendKind::OpenGles;
    dispatch.wgsl_source.clear();
    dispatch.glsl_source = kQ5_0DequantizeGlsl;
    const std::string placeholder = "const uint ELEMENT_COUNT = 32u;";
    const std::string replacement = "const uint ELEMENT_COUNT = " +
        std::to_string(output.element_count()) + "u;";
    const size_t position = dispatch.glsl_source.find(placeholder);
    if (position == std::string::npos)
        return {StatusCode::BackendFailure, "Q5_0 GLSL element-count placeholder is missing"};
    dispatch.glsl_source.replace(position, placeholder.size(), replacement);
    return Status::success();
}

Status build_q5_1_dequantize_gles_candidate(const std::shared_ptr<Buffer>& encoded,
                                            uint64_t encoded_bytes,
                                            const Tensor& output,
                                            ComputeDispatch& dispatch) {
    Status status = build_q5_1_dequantize_candidate(encoded, encoded_bytes, output, dispatch);
    if (!status.ok()) return status;
    dispatch.label = "q5-1-dequantize-gles-generic-unverified";
    dispatch.kernel_id = "gles-q5-1-dequantize-generic-v1";
    dispatch.backend = BackendKind::OpenGles;
    dispatch.wgsl_source.clear();
    dispatch.glsl_source = kQ5_1DequantizeGlsl;
    const std::string placeholder = "const uint ELEMENT_COUNT = 32u;";
    const std::string replacement = "const uint ELEMENT_COUNT = " +
        std::to_string(output.element_count()) + "u;";
    const size_t position = dispatch.glsl_source.find(placeholder);
    if (position == std::string::npos)
        return {StatusCode::BackendFailure, "Q5_1 GLSL element-count placeholder is missing"};
    dispatch.glsl_source.replace(position, placeholder.size(), replacement);
    return Status::success();
}


const char* q4_0_dequantize_glsl_source() noexcept { return kQ4_0DequantizeGlsl; }
const char* q4_1_dequantize_glsl_source() noexcept { return kQ4_1DequantizeGlsl; }
const char* q8_0_dequantize_glsl_source() noexcept { return kQ8_0DequantizeGlsl; }

Status build_q4_0_dequantize_gles_candidate(const std::shared_ptr<Buffer>& encoded, uint64_t bytes,
                                            const Tensor& output, ComputeDispatch& dispatch) {
    Status status = build_q4_0_dequantize_candidate(encoded, bytes, output, dispatch);
    if (!status.ok()) return status;
    dispatch.label = "q4-0-dequantize-gles-generic-unverified";
    dispatch.kernel_id = "gles-q4-0-dequantize-generic-v1";
    dispatch.backend = BackendKind::OpenGles;
    dispatch.wgsl_source.clear();
    dispatch.glsl_source = kQ4_0DequantizeGlsl;
    const std::string placeholder = "const uint ELEMENT_COUNT = 32u;";
    const std::string replacement = "const uint ELEMENT_COUNT = " + std::to_string(output.element_count()) + "u;";
    const size_t pos = dispatch.glsl_source.find(placeholder);
    if (pos == std::string::npos) return {StatusCode::BackendFailure, "Q4_0 GLSL element-count placeholder missing"};
    dispatch.glsl_source.replace(pos, placeholder.size(), replacement);
    return Status::success();
}

Status build_q4_1_dequantize_gles_candidate(const std::shared_ptr<Buffer>& encoded, uint64_t bytes,
                                            const Tensor& output, ComputeDispatch& dispatch) {
    Status status = build_q4_1_dequantize_candidate(encoded, bytes, output, dispatch);
    if (!status.ok()) return status;
    dispatch.label = "q4-1-dequantize-gles-generic-unverified";
    dispatch.kernel_id = "gles-q4-1-dequantize-generic-v1";
    dispatch.backend = BackendKind::OpenGles;
    dispatch.wgsl_source.clear();
    dispatch.glsl_source = kQ4_1DequantizeGlsl;
    const std::string placeholder = "const uint ELEMENT_COUNT = 32u;";
    const std::string replacement = "const uint ELEMENT_COUNT = " + std::to_string(output.element_count()) + "u;";
    const size_t pos = dispatch.glsl_source.find(placeholder);
    if (pos == std::string::npos) return {StatusCode::BackendFailure, "Q4_1 GLSL element-count placeholder missing"};
    dispatch.glsl_source.replace(pos, placeholder.size(), replacement);
    return Status::success();
}

Status build_q8_0_dequantize_gles_candidate(const std::shared_ptr<Buffer>& encoded, uint64_t bytes,
                                            const Tensor& output, ComputeDispatch& dispatch) {
    Status status = build_q8_0_dequantize_candidate(encoded, bytes, output, dispatch);
    if (!status.ok()) return status;
    dispatch.label = "q8-0-dequantize-gles-generic-unverified";
    dispatch.kernel_id = "gles-q8-0-dequantize-generic-v1";
    dispatch.backend = BackendKind::OpenGles;
    dispatch.wgsl_source.clear();
    dispatch.glsl_source = kQ8_0DequantizeGlsl;
    const std::string placeholder = "const uint ELEMENT_COUNT = 32u;";
    const std::string replacement = "const uint ELEMENT_COUNT = " + std::to_string(output.element_count()) + "u;";
    const size_t pos = dispatch.glsl_source.find(placeholder);
    if (pos == std::string::npos) return {StatusCode::BackendFailure, "Q8_0 GLSL element-count placeholder missing"};
    dispatch.glsl_source.replace(pos, placeholder.size(), replacement);
    return Status::success();
}


namespace {
using DequantizeDecoder = Status (*)(const uint8_t*, size_t, float*, size_t);

template <typename Builder>
Status validate_dequantize_candidate(Backend& backend, const std::vector<uint8_t>& source,
                                     uint32_t block_bytes, DequantizeDecoder decoder,
                                     Builder&& builder, ComputeDispatch& validated_dispatch) {
    validated_dispatch = ComputeDispatch{};
    if (!backend.healthy() ||
        (backend.kind() != BackendKind::DawnWebGpu && backend.kind() != BackendKind::OpenGles))
        return {StatusCode::NotReady, "Quantized dequantization validation requires a healthy supported GPU backend"};
    if (source.empty() || block_bytes == 0 || source.size() % block_bytes != 0 || source.size() > UINT64_MAX - 3ull)
        return {StatusCode::InvalidArgument, "Quantized dequantization validation requires complete encoded blocks"};
    const uint64_t elements = (source.size() / block_bytes) * 32ull;
    if (elements == 0 || elements > UINT32_MAX || elements > UINT64_MAX / sizeof(float))
        return {StatusCode::Unsupported, "Quantized dequantization validation tensor exceeds supported indexing"};
    const uint64_t padded_bytes = (source.size() + 3ull) & ~3ull;
    std::vector<uint8_t> padded(static_cast<size_t>(padded_bytes), 0);
    std::copy(source.begin(), source.end(), padded.begin());

    BufferDesc input_desc;
    input_desc.size_bytes = padded_bytes;
    input_desc.alignment = 4;
    input_desc.memory = MemoryDomain::Device;
    input_desc.persistent = false;
    input_desc.label = "quant-dequant-validation-input";
    std::shared_ptr<Buffer> encoded;
    Status status = backend.allocate(input_desc, encoded);
    if (!status.ok()) return status;
    if (!encoded || encoded->size_bytes() < padded_bytes)
        return {StatusCode::BackendFailure, "Backend returned an undersized quantized validation buffer"};
    status = backend.upload(encoded, 0, padded.data(), padded_bytes);
    if (!status.ok()) return status;

    BufferDesc output_desc;
    output_desc.size_bytes = elements * sizeof(float);
    output_desc.alignment = alignof(float);
    output_desc.memory = MemoryDomain::Device;
    output_desc.persistent = false;
    output_desc.label = "quant-dequant-validation-output";
    std::shared_ptr<Buffer> output_buffer;
    status = backend.allocate(output_desc, output_buffer);
    if (!status.ok()) return status;
    TensorDesc tensor_desc;
    tensor_desc.shape = {elements};
    tensor_desc.data_type = DataType::F32;
    tensor_desc.layout = TensorLayout::Contiguous;
    tensor_desc.memory = MemoryDomain::Device;
    tensor_desc.alignment = alignof(float);
    Tensor output;
    status = Tensor::create(std::move(tensor_desc), output_buffer, 0, output);
    if (!status.ok()) return status;

    std::vector<float> expected(static_cast<size_t>(elements));
    status = decoder(source.data(), source.size(), expected.data(), expected.size());
    if (!status.ok()) return status;
    ComputeDispatch candidate;
    status = builder(encoded, source.size(), output, candidate);
    if (!status.ok()) return status;
    if (candidate.backend != backend.kind() || candidate.kernel_id.empty())
        return {StatusCode::InvalidArgument, "Quantized dequantization candidate backend identity mismatch"};

    ComputeDispatch validation_dispatch = candidate;
    validation_dispatch.validation = KernelValidation::Validated;
    std::unique_ptr<CommandBatch> commands;
    status = backend.begin_commands(commands);
    if (!status.ok()) return status;
    if (!commands) return {StatusCode::BackendFailure, "Backend returned an empty dequantization validation command batch"};
    status = commands->dispatch_compute(validation_dispatch);
    if (!status.ok()) return status;
    status = backend.submit(std::move(commands));
    if (!status.ok()) return status;
    if (!backend.healthy()) return {StatusCode::DeviceLost, "GPU backend was lost during quantized dequantization validation"};
    status = backend.synchronize();
    if (!status.ok()) return status;
    std::vector<float> actual(static_cast<size_t>(elements));
    status = backend.readback(output_buffer, 0, actual.data(), output_desc.size_bytes);
    if (!status.ok()) return status;
    constexpr float absolute_tolerance = 2.0e-4f;
    constexpr float relative_tolerance = 2.0e-4f;
    for (size_t i = 0; i < actual.size(); ++i) {
        if (!std::isfinite(actual[i]) || !std::isfinite(expected[i]))
            return {StatusCode::NumericalFailure, "Quantized dequantization produced a non-finite value"};
        const float tolerance = absolute_tolerance +
            relative_tolerance * std::max(std::fabs(expected[i]), 1.0f);
        if (std::fabs(actual[i] - expected[i]) > tolerance)
            return {StatusCode::NumericalFailure, "GPU dequantization differs from the independent CPU decoder"};
    }
    candidate.validation = KernelValidation::Validated;
    validated_dispatch = std::move(candidate);
    return Status::success();
}

template <typename GlesBuilder, typename DawnBuilder>
Status validate_dequantizer_for_backend(Backend& backend, uint32_t block_bytes,
                                        DequantizeDecoder decoder,
                                        GlesBuilder&& gles_builder, DawnBuilder&& dawn_builder,
                                        ComputeDispatch& out) {
    auto builder = [&](const std::shared_ptr<Buffer>& encoded, uint64_t bytes,
                       const Tensor& output, ComputeDispatch& dispatch) {
        return backend.kind() == BackendKind::OpenGles
            ? gles_builder(encoded, bytes, output, dispatch)
            : dawn_builder(encoded, bytes, output, dispatch);
    };
    std::vector<uint8_t> sample(block_bytes, 0);
    sample[0] = 0x00;
    sample[1] = 0x3c; // little-endian FP16 1.0 scale
    return validate_dequantize_candidate(backend, sample, block_bytes, decoder, builder, out);
}
} // namespace

Status validate_q4_0_dequantize_kernel(Backend& backend, ComputeDispatch& out) {
    return validate_dequantizer_for_backend(backend, 18, decode_q4_0_blocks,
        build_q4_0_dequantize_gles_candidate, build_q4_0_dequantize_candidate, out);
}
Status validate_q4_1_dequantize_kernel(Backend& backend, ComputeDispatch& out) {
    return validate_dequantizer_for_backend(backend, 20, decode_q4_1_blocks,
        build_q4_1_dequantize_gles_candidate, build_q4_1_dequantize_candidate, out);
}
Status validate_q5_0_dequantize_kernel(Backend& backend, ComputeDispatch& out) {
    return validate_dequantizer_for_backend(backend, 22, decode_q5_0_blocks,
        build_q5_0_dequantize_gles_candidate, build_q5_0_dequantize_candidate, out);
}
Status validate_q5_1_dequantize_kernel(Backend& backend, ComputeDispatch& out) {
    return validate_dequantizer_for_backend(backend, 24, decode_q5_1_blocks,
        build_q5_1_dequantize_gles_candidate, build_q5_1_dequantize_candidate, out);
}
Status validate_q8_0_dequantize_kernel(Backend& backend, ComputeDispatch& out) {
    return validate_dequantizer_for_backend(backend, 34, decode_q8_0_blocks,
        build_q8_0_dequantize_gles_candidate, build_q8_0_dequantize_candidate, out);
}


namespace {
Status build_q5_matmul_candidate_impl(const Tensor& activations, const std::shared_ptr<Buffer>& weights,
                                      uint64_t weight_bytes, const Tensor& output, ComputeDispatch& dispatch,
                                      bool q5_1) {
    dispatch = ComputeDispatch{};
    const uint64_t block_bytes = q5_1 ? 24 : 22;
    const std::string type = q5_1 ? "Q5_1" : "Q5_0";
    if (!weights || weights->memory_domain() == MemoryDomain::Host || !is_device_contiguous_f32(activations) || !is_device_contiguous_f32(output) ||
        !activations.buffer() || !output.buffer() || activations.descriptor().shape.size() != 2 ||
        (output.descriptor().shape.size() != 2 && output.descriptor().shape.size() != 3))
        return {StatusCode::Unsupported, type + " MatMul requires F32 [M,K] activations and F32 [M,N] or [M,H,D] output"};
    const auto& a = activations.descriptor().shape;
    const auto& o = output.descriptor().shape;
    const uint64_t channels = o.size() == 2 ? o[1] : (o[2] != 0 && o[1] <= UINT64_MAX / o[2] ? o[1] * o[2] : 0);
    if (!a[0] || !a[1] || a[1] % 32 || o[0] != a[0] || !channels ||
        a[0] > UINT32_MAX || a[1] > UINT32_MAX || channels > UINT32_MAX ||
        activations.element_count() > UINT32_MAX || output.element_count() > UINT32_MAX)
        return {StatusCode::InvalidArgument, type + " MatMul dimensions are invalid or inner dimension is not block-aligned"};
    const uint64_t blocks = a[1] / 32;
    if (blocks > UINT64_MAX / block_bytes || channels > UINT64_MAX / (blocks * block_bytes))
        return {StatusCode::InvalidArgument, type + " weight byte size overflows"};
    const uint64_t expected = channels * blocks * block_bytes;
    if (weight_bytes != expected || weight_bytes > UINT32_MAX - 3ull)
        return {StatusCode::InvalidArgument, type + " weight byte count does not match [N,K] shape"};
    const uint64_t padded = (weight_bytes + 3) & ~uint64_t{3};
    if (weights->size_bytes() < padded || output.byte_size() != output.element_count() * sizeof(float) ||
        output.buffer().get() == activations.buffer().get() || output.buffer().get() == weights.get())
        return {StatusCode::InvalidArgument, type + " MatMul buffer ranges are inconsistent"};
    const std::string suffix = q5_1 ? "q5-1" : "q5-0";
    dispatch.label = suffix + "-matmul-f32-activation-generic-unverified";
    dispatch.kernel_id = "dawn-" + suffix + "-matmul-f32-activation-v1";
    dispatch.backend = BackendKind::DawnWebGpu;
    dispatch.wgsl_source = q5_1 ? kQ5_1MatMulWgsl : kQ5_0MatMulWgsl;
    dispatch.workgroup_size_x = dispatch.workgroup_size_y = 8;
    const std::pair<std::string,std::string> constants[] = {
        {"const ROWS: u32 = 1u;", "const ROWS: u32 = "+std::to_string(a[0])+"u;"},
        {"const OUTPUT_CHANNELS: u32 = 1u;", "const OUTPUT_CHANNELS: u32 = "+std::to_string(channels)+"u;"},
        {"const INNER: u32 = 32u;", "const INNER: u32 = "+std::to_string(a[1])+"u;"},
        {q5_1 ? "const WEIGHT_ROW_BYTES: u32 = 24u;" : "const WEIGHT_ROW_BYTES: u32 = 22u;",
         "const WEIGHT_ROW_BYTES: u32 = "+std::to_string(blocks*block_bytes)+"u;"}
    };
    for(const auto& item:constants){const size_t pos=dispatch.wgsl_source.find(item.first);if(pos==std::string::npos)return {StatusCode::BackendFailure,type+" WGSL placeholder missing"};dispatch.wgsl_source.replace(pos,item.first.size(),item.second);}
    dispatch.workgroups_x=static_cast<uint32_t>((channels+7)/8);
    dispatch.workgroups_y=static_cast<uint32_t>((a[0]+7)/8);
    dispatch.bindings={{0,activations.buffer(),activations.byte_offset(),activations.byte_size()},
                       {1,weights,0,padded},{2,output.buffer(),output.byte_offset(),output.byte_size()}};
    return Status::success();
}
}
Status build_q5_0_matmul_candidate(const Tensor& a,const std::shared_ptr<Buffer>& w,uint64_t b,const Tensor& o,ComputeDispatch& d){return build_q5_matmul_candidate_impl(a,w,b,o,d,false);}
Status build_q5_1_matmul_candidate(const Tensor& a,const std::shared_ptr<Buffer>& w,uint64_t b,const Tensor& o,ComputeDispatch& d){return build_q5_matmul_candidate_impl(a,w,b,o,d,true);}

static Status build_q5_matmul_gles_candidate_impl(const Tensor& activations,const std::shared_ptr<Buffer>& weights,
                                           uint64_t bytes,const Tensor& output,ComputeDispatch& dispatch,bool q5_1){
    Status status=build_q5_matmul_candidate_impl(activations,weights,bytes,output,dispatch,q5_1);
    if(!status.ok())return status;
    const uint64_t block_bytes=q5_1?24:22, blocks=activations.descriptor().shape[1]/32;
    const uint64_t channels=output.element_count()/activations.descriptor().shape[0];
    const std::string suffix=q5_1?"q5-1":"q5-0";
    dispatch.label=suffix+"-matmul-f32-activation-gles-generic-unverified";
    dispatch.kernel_id="gles-"+suffix+"-matmul-f32-activation-v1";dispatch.backend=BackendKind::OpenGles;
    dispatch.wgsl_source.clear();dispatch.glsl_source=q5_1?kQ5_1MatMulGlsl:kQ5_0MatMulGlsl;
    const std::pair<std::string,std::string> constants[]={
        {"const uint ROWS = 1u;","const uint ROWS = "+std::to_string(activations.descriptor().shape[0])+"u;"},
        {"const uint OUTPUT_CHANNELS = 1u;","const uint OUTPUT_CHANNELS = "+std::to_string(channels)+"u;"},
        {"const uint INNER = 32u;","const uint INNER = "+std::to_string(activations.descriptor().shape[1])+"u;"},
        {q5_1?"const uint WEIGHT_ROW_BYTES = 24u;":"const uint WEIGHT_ROW_BYTES = 22u;",
         "const uint WEIGHT_ROW_BYTES = "+std::to_string(blocks*block_bytes)+"u;"}
    };
    for(const auto& item:constants){const size_t pos=dispatch.glsl_source.find(item.first);if(pos==std::string::npos)return {StatusCode::BackendFailure,"Q5 GLES GLSL placeholder missing"};dispatch.glsl_source.replace(pos,item.first.size(),item.second);}
    return Status::success();
}
Status build_q5_0_matmul_gles_candidate(const Tensor&a,const std::shared_ptr<Buffer>&w,uint64_t b,const Tensor&o,ComputeDispatch&d){return build_q5_matmul_gles_candidate_impl(a,w,b,o,d,false);}
Status build_q5_1_matmul_gles_candidate(const Tensor&a,const std::shared_ptr<Buffer>&w,uint64_t b,const Tensor&o,ComputeDispatch&d){return build_q5_matmul_gles_candidate_impl(a,w,b,o,d,true);}

Status reference_q5_0_matmul(const float* a,const uint8_t*w,uint64_t bytes,float*out,uint32_t rows,uint32_t channels,uint32_t inner){
    if((!a||!w||!out)&&rows&&channels&&inner)return {StatusCode::InvalidArgument,"Q5_0 MatMul reference received a null pointer"};
    if(!rows||!channels||!inner||inner%32)return {StatusCode::InvalidArgument,"Q5_0 MatMul dimensions must be block-aligned"};
    const uint64_t blocks=inner/32;if(bytes!=static_cast<uint64_t>(channels)*blocks*22)return {StatusCode::InvalidArgument,"Q5_0 MatMul weight bytes mismatch"};
    std::vector<float> decoded(inner);
    for(uint32_t channel=0;channel<channels;++channel){Status st=decode_q5_0_blocks(w+static_cast<uint64_t>(channel)*blocks*22,blocks*22,decoded.data(),inner);if(!st.ok())return st;
        for(uint32_t row=0;row<rows;++row){float sum=0;for(uint32_t k=0;k<inner;++k)sum+=a[static_cast<uint64_t>(row)*inner+k]*decoded[k];out[static_cast<uint64_t>(row)*channels+channel]=sum;}}
    return Status::success();
}
Status reference_q5_1_matmul(const float* a,const uint8_t*w,uint64_t bytes,float*out,uint32_t rows,uint32_t channels,uint32_t inner){
    if((!a||!w||!out)&&rows&&channels&&inner)return {StatusCode::InvalidArgument,"Q5_1 MatMul reference received a null pointer"};
    if(!rows||!channels||!inner||inner%32)return {StatusCode::InvalidArgument,"Q5_1 MatMul dimensions must be block-aligned"};
    const uint64_t blocks=inner/32;if(bytes!=static_cast<uint64_t>(channels)*blocks*24)return {StatusCode::InvalidArgument,"Q5_1 MatMul weight bytes mismatch"};
    std::vector<float> decoded(inner);
    for(uint32_t channel=0;channel<channels;++channel){Status st=decode_q5_1_blocks(w+static_cast<uint64_t>(channel)*blocks*24,blocks*24,decoded.data(),inner);if(!st.ok())return st;
        for(uint32_t row=0;row<rows;++row){float sum=0;for(uint32_t k=0;k<inner;++k)sum+=a[static_cast<uint64_t>(row)*inner+k]*decoded[k];out[static_cast<uint64_t>(row)*channels+channel]=sum;}}
    return Status::success();
}


Status validate_q5_matmul_kernel_impl(Backend& backend, OperatorRegistry& operators,
                                      QuantizationRegistry& quantizations, bool q5_1) {
    const BackendKind kind=backend.kind();
    if((kind!=BackendKind::DawnWebGpu&&kind!=BackendKind::OpenGles)||!backend.healthy())
        return {StatusCode::NotReady,"Q5 MatMul validation requires a healthy Dawn or GLES backend"};
    const uint32_t type_id=q5_1?7u:6u;
    auto format=quantizations.formats_.find(type_id);
    if(format==quantizations.formats_.end()||format->second.decoder_state!=SupportState::Validated)
        return {StatusCode::NotReady,"Q5 host decoder must be validated before GPU kernel promotion"};
    const std::string suffix=q5_1?"q5-1":"q5-0";
    const std::string kernel_id=(kind==BackendKind::OpenGles?"gles-":"dawn-")+suffix+"-matmul-f32-activation-v1";
    auto kernel=std::find_if(operators.kernels_.begin(),operators.kernels_.end(),[&](const KernelDefinition& k){
        return k.kernel_id==kernel_id&&k.backend==kind;
    });
    if(kernel==operators.kernels_.end())return {StatusCode::NotReady,"Q5 MatMul candidate is not registered for selected backend"};
    if(kernel->validation==KernelValidation::Validated){format->second.backend_gpu_kernel_states[kind]=SupportState::Validated;format->second.gpu_kernel_state=SupportState::Validated;return Status::success();}
    if(kernel->validation!=KernelValidation::Unverified)return {StatusCode::Unsupported,"Q5 MatMul candidate is rejected"};
    const bool gles=kind==BackendKind::OpenGles;
    const std::string& registered=gles?kernel->glsl_source:kernel->wgsl_source;
    const std::string& canonical=q5_1?(gles?q5_1_matmul_glsl_source():q5_1_matmul_wgsl_source())
                                    :(gles?q5_0_matmul_glsl_source():q5_0_matmul_wgsl_source());
    if(registered.empty()||registered!=canonical)return {StatusCode::InvalidArgument,"Q5 MatMul registered shader differs from canonical source"};
    constexpr uint32_t rows=2,channels=2,inner=64;
    const uint32_t block_bytes=q5_1?24u:22u;
    const uint64_t bytes=static_cast<uint64_t>(channels)*(inner/32)*block_bytes;
    std::vector<float> activations(rows*inner);
    for(size_t i=0;i<activations.size();++i)activations[i]=(static_cast<int>(i%9)-4)*0.25f;
    std::vector<uint8_t> encoded(static_cast<size_t>(bytes),0);
    for(uint32_t block=0;block<channels*(inner/32);++block){
        const size_t off=static_cast<size_t>(block)*block_bytes;
        encoded[off]=0x00;encoded[off+1]=0x38;
        const uint32_t qh_offset=q5_1?4u:2u, qs_offset=q5_1?8u:6u;
        if(q5_1){encoded[off+2]=0x00;encoded[off+3]=0xbc;}
        for(uint32_t i=0;i<4;++i)encoded[off+qh_offset+i]=static_cast<uint8_t>(0x96u+(i*17u));
        for(uint32_t i=0;i<16;++i){
            const uint8_t low=static_cast<uint8_t>((i+block)%16);
            const uint8_t high=static_cast<uint8_t>((i+block+5)%16);
            encoded[off+qs_offset+i]=static_cast<uint8_t>(low|(high<<4));
        }
    }
    float expected[rows*channels]={},actual[rows*channels]={};
    Status status=q5_1?reference_q5_1_matmul(activations.data(),encoded.data(),bytes,expected,rows,channels,inner)
                      :reference_q5_0_matmul(activations.data(),encoded.data(),bytes,expected,rows,channels,inner);
    if(!status.ok())return status;
    BufferDesc ad{activations.size()*sizeof(float),alignof(float),MemoryDomain::Device,false,"q5-validation-activations"};
    BufferDesc wd{(bytes+3)&~uint64_t{3},4,MemoryDomain::Device,true,"q5-validation-weights"};
    BufferDesc od{sizeof(actual),alignof(float),MemoryDomain::Device,false,"q5-validation-output"};
    std::shared_ptr<Buffer> ab,wb,ob;
    status=backend.allocate(ad,ab);if(!status.ok())return status;
    status=backend.allocate(wd,wb);if(!status.ok())return status;
    status=backend.allocate(od,ob);if(!status.ok())return status;
    status=backend.upload(ab,0,activations.data(),ad.size_bytes);if(!status.ok())return status;
    status=backend.upload(wb,0,encoded.data(),bytes);if(!status.ok())return status;
    TensorDesc atd;atd.shape={rows,inner};atd.data_type=DataType::F32;
    TensorDesc otd;otd.shape={rows,channels};otd.data_type=DataType::F32;
    Tensor at,ot;status=Tensor::create(atd,ab,0,at);if(!status.ok())return status;
    status=Tensor::create(otd,ob,0,ot);if(!status.ok())return status;
    ComputeDispatch dispatch;
    if(q5_1)status=gles?build_q5_1_matmul_gles_candidate(at,wb,bytes,ot,dispatch):build_q5_1_matmul_candidate(at,wb,bytes,ot,dispatch);
    else status=gles?build_q5_0_matmul_gles_candidate(at,wb,bytes,ot,dispatch):build_q5_0_matmul_candidate(at,wb,bytes,ot,dispatch);
    if(!status.ok())return status;
    if(dispatch.kernel_id!=kernel_id||dispatch.backend!=kind)return {StatusCode::InvalidArgument,"Q5 MatMul candidate identity mismatch"};
    dispatch.validation=KernelValidation::Validated;
    std::unique_ptr<CommandBatch> commands;status=backend.begin_commands(commands);if(!status.ok())return status;
    if(!commands)return {StatusCode::BackendFailure,"Q5 MatMul backend returned empty command batch"};
    status=commands->dispatch_compute(dispatch);if(!status.ok())return status;
    status=backend.submit(std::move(commands));if(!status.ok())return status;
    status=backend.synchronize();if(!status.ok())return status;
    status=backend.readback(ob,0,actual,sizeof(actual));if(!status.ok())return status;
    for(size_t i=0;i<rows*channels;++i){
        const float tolerance=1.0e-4f*std::max(1.0f,std::fabs(expected[i]));
        if(!std::isfinite(actual[i])||std::fabs(actual[i]-expected[i])>tolerance)
            return {StatusCode::NumericalFailure,"Q5 MatMul GPU output differs from independent block-decoder reference"};
    }
    kernel->validation=KernelValidation::Validated;
    format->second.backend_gpu_kernel_states[kind]=SupportState::Validated;
    format->second.gpu_kernel_state=SupportState::Validated;
    return Status::success();
}
Status validate_q5_0_matmul_kernel(Backend& b,OperatorRegistry& o,QuantizationRegistry& q){return validate_q5_matmul_kernel_impl(b,o,q,false);}
Status validate_q5_1_matmul_kernel(Backend& b,OperatorRegistry& o,QuantizationRegistry& q){return validate_q5_matmul_kernel_impl(b,o,q,true);}

namespace {
Status build_q5_embedding_candidate_impl(const std::shared_ptr<Buffer>& weights,uint64_t weight_bytes,
        uint32_t vocabulary,uint32_t dimension,const std::shared_ptr<Buffer>& token_buffer,
        const std::vector<uint32_t>& token_ids,const Tensor& output,ComputeDispatch& dispatch,bool q5_1) {
    dispatch=ComputeDispatch{};
    const uint64_t block_bytes=q5_1?24:22;
    const std::string type=q5_1?"Q5_1":"Q5_0",suffix=q5_1?"q5-1":"q5-0";
    if(!weights||!token_buffer||!output.buffer()||weights->memory_domain()==MemoryDomain::Host||
       token_buffer->memory_domain()==MemoryDomain::Host||output.descriptor().data_type!=DataType::F32||
       output.descriptor().layout!=TensorLayout::Contiguous||output.descriptor().memory==MemoryDomain::Host||
       output.descriptor().shape.size()!=2||token_ids.empty())
        return {StatusCode::Unsupported,type+" Embedding requires packed weights, device token IDs and contiguous F32 output"};
    if(!vocabulary||!dimension||dimension%32||token_ids.size()!=output.descriptor().shape[0]||
       output.descriptor().shape[1]!=dimension||token_ids.size()>UINT32_MAX||output.element_count()>UINT32_MAX)
        return {StatusCode::InvalidArgument,type+" Embedding dimensions or token count are invalid"};
    for(uint32_t id:token_ids)if(id>=vocabulary)return {StatusCode::InvalidArgument,type+" Embedding token ID is outside vocabulary"};
    const uint64_t row_bytes=static_cast<uint64_t>(dimension/32)*block_bytes;
    if(row_bytes>UINT64_MAX/vocabulary||weight_bytes!=row_bytes*vocabulary||weight_bytes>UINT32_MAX-3ull)
        return {StatusCode::InvalidArgument,type+" Embedding weight byte size does not match [vocabulary,dimension]"};
    const uint64_t padded=(weight_bytes+3)&~uint64_t{3},token_bytes=token_ids.size()*sizeof(uint32_t);
    if(weights->size_bytes()<padded||token_buffer->size_bytes()<token_bytes||
       output.byte_size()!=output.element_count()*sizeof(float)||output.buffer().get()==weights.get()||
       output.buffer().get()==token_buffer.get()||weights.get()==token_buffer.get())
        return {StatusCode::InvalidArgument,type+" Embedding buffer ranges are inconsistent"};
    dispatch.label=suffix+"-embedding-f32-generic-unverified";
    dispatch.kernel_id="dawn-"+suffix+"-embedding-f32-v1";
    dispatch.backend=BackendKind::DawnWebGpu;
    dispatch.wgsl_source=q5_1?kQ5_1EmbeddingWgsl:kQ5_0EmbeddingWgsl;
    dispatch.workgroup_size_x=64;
    const std::pair<std::string,std::string> constants[]={
        {"const VOCABULARY_SIZE: u32 = 1u;","const VOCABULARY_SIZE: u32 = "+std::to_string(vocabulary)+"u;"},
        {"const EMBEDDING_DIMENSION: u32 = 32u;","const EMBEDDING_DIMENSION: u32 = "+std::to_string(dimension)+"u;"},
        {"const TOKEN_COUNT: u32 = 1u;","const TOKEN_COUNT: u32 = "+std::to_string(token_ids.size())+"u;"},
        {"const TOTAL_ELEMENTS: u32 = 32u;","const TOTAL_ELEMENTS: u32 = "+std::to_string(output.element_count())+"u;"},
        {q5_1?"const WEIGHT_ROW_BYTES: u32 = 24u;":"const WEIGHT_ROW_BYTES: u32 = 22u;","const WEIGHT_ROW_BYTES: u32 = "+std::to_string(row_bytes)+"u;"}
    };
    for(const auto& item:constants){const size_t pos=dispatch.wgsl_source.find(item.first);if(pos==std::string::npos)return {StatusCode::BackendFailure,type+" Embedding WGSL placeholder missing"};dispatch.wgsl_source.replace(pos,item.first.size(),item.second);}
    dispatch.workgroups_x=static_cast<uint32_t>((output.element_count()+63)/64);
    dispatch.bindings={{0,weights,0,padded},{1,token_buffer,0,token_bytes},{2,output.buffer(),output.byte_offset(),output.byte_size()}};
    return Status::success();
}
}
Status build_q5_0_embedding_candidate(const std::shared_ptr<Buffer>&w,uint64_t b,uint32_t v,uint32_t d,const std::shared_ptr<Buffer>&tb,const std::vector<uint32_t>&ids,const Tensor&o,ComputeDispatch&x){return build_q5_embedding_candidate_impl(w,b,v,d,tb,ids,o,x,false);}
Status build_q5_1_embedding_candidate(const std::shared_ptr<Buffer>&w,uint64_t b,uint32_t v,uint32_t d,const std::shared_ptr<Buffer>&tb,const std::vector<uint32_t>&ids,const Tensor&o,ComputeDispatch&x){return build_q5_embedding_candidate_impl(w,b,v,d,tb,ids,o,x,true);}
static Status build_q5_embedding_gles_candidate_impl(const std::shared_ptr<Buffer>&weights,uint64_t bytes,uint32_t vocabulary,uint32_t dimension,
        const std::shared_ptr<Buffer>&token_buffer,const std::vector<uint32_t>&token_ids,const Tensor&output,ComputeDispatch&dispatch,bool q5_1){
    Status status=build_q5_embedding_candidate_impl(weights,bytes,vocabulary,dimension,token_buffer,token_ids,output,dispatch,q5_1);
    if(!status.ok())return status;
    const uint64_t block_bytes=q5_1?24:22,row_bytes=static_cast<uint64_t>(dimension/32)*block_bytes;
    const std::string suffix=q5_1?"q5-1":"q5-0";
    dispatch.label=suffix+"-embedding-gles-f32-generic-unverified";
    dispatch.kernel_id="gles-"+suffix+"-embedding-f32-v1";dispatch.backend=BackendKind::OpenGles;
    dispatch.wgsl_source.clear();dispatch.glsl_source=q5_1?kQ5_1EmbeddingGlsl:kQ5_0EmbeddingGlsl;
    const std::pair<std::string,std::string> constants[]={
        {"const uint VOCABULARY_SIZE = 1u;","const uint VOCABULARY_SIZE = "+std::to_string(vocabulary)+"u;"},
        {"const uint EMBEDDING_DIMENSION = 32u;","const uint EMBEDDING_DIMENSION = "+std::to_string(dimension)+"u;"},
        {"const uint TOKEN_COUNT = 1u;","const uint TOKEN_COUNT = "+std::to_string(token_ids.size())+"u;"},
        {"const uint TOTAL_ELEMENTS = 32u;","const uint TOTAL_ELEMENTS = "+std::to_string(output.element_count())+"u;"},
        {q5_1?"const uint WEIGHT_ROW_BYTES = 24u;":"const uint WEIGHT_ROW_BYTES = 22u;","const uint WEIGHT_ROW_BYTES = "+std::to_string(row_bytes)+"u;"}
    };
    for(const auto& item:constants){const size_t pos=dispatch.glsl_source.find(item.first);if(pos==std::string::npos)return {StatusCode::BackendFailure,"Q5 GLES Embedding GLSL placeholder missing"};dispatch.glsl_source.replace(pos,item.first.size(),item.second);}
    return Status::success();
}
Status build_q5_0_embedding_gles_candidate(const std::shared_ptr<Buffer>&w,uint64_t b,uint32_t v,uint32_t d,const std::shared_ptr<Buffer>&tb,const std::vector<uint32_t>&ids,const Tensor&o,ComputeDispatch&x){return build_q5_embedding_gles_candidate_impl(w,b,v,d,tb,ids,o,x,false);}
Status build_q5_1_embedding_gles_candidate(const std::shared_ptr<Buffer>&w,uint64_t b,uint32_t v,uint32_t d,const std::shared_ptr<Buffer>&tb,const std::vector<uint32_t>&ids,const Tensor&o,ComputeDispatch&x){return build_q5_embedding_gles_candidate_impl(w,b,v,d,tb,ids,o,x,true);}

Status reference_q5_0_embedding(const uint8_t*w,uint64_t bytes,uint32_t vocabulary,uint32_t dimension,const std::vector<uint32_t>&ids,float*out){
    if((!w||!out)&&!ids.empty())return {StatusCode::InvalidArgument,"Q5_0 Embedding reference received a null pointer"};
    if(!vocabulary||!dimension||dimension%32||ids.empty())return {StatusCode::InvalidArgument,"Q5_0 Embedding dimensions are invalid"};
    const uint64_t row_bytes=static_cast<uint64_t>(dimension/32)*22;
    if(row_bytes>UINT64_MAX/vocabulary||bytes!=row_bytes*vocabulary)return {StatusCode::InvalidArgument,"Q5_0 Embedding weight bytes mismatch"};
    std::vector<float> decoded(dimension);
    for(size_t row=0;row<ids.size();++row){if(ids[row]>=vocabulary)return {StatusCode::InvalidArgument,"Q5_0 Embedding token ID out of range"};
        Status st=decode_q5_0_blocks(w+ids[row]*row_bytes,row_bytes,decoded.data(),decoded.size());if(!st.ok())return st;
        std::copy(decoded.begin(),decoded.end(),out+row*dimension);}
    return Status::success();
}
Status reference_q5_1_embedding(const uint8_t*w,uint64_t bytes,uint32_t vocabulary,uint32_t dimension,const std::vector<uint32_t>&ids,float*out){
    if((!w||!out)&&!ids.empty())return {StatusCode::InvalidArgument,"Q5_1 Embedding reference received a null pointer"};
    if(!vocabulary||!dimension||dimension%32||ids.empty())return {StatusCode::InvalidArgument,"Q5_1 Embedding dimensions are invalid"};
    const uint64_t row_bytes=static_cast<uint64_t>(dimension/32)*24;
    if(row_bytes>UINT64_MAX/vocabulary||bytes!=row_bytes*vocabulary)return {StatusCode::InvalidArgument,"Q5_1 Embedding weight bytes mismatch"};
    std::vector<float> decoded(dimension);
    for(size_t row=0;row<ids.size();++row){if(ids[row]>=vocabulary)return {StatusCode::InvalidArgument,"Q5_1 Embedding token ID out of range"};
        Status st=decode_q5_1_blocks(w+ids[row]*row_bytes,row_bytes,decoded.data(),decoded.size());if(!st.ok())return st;
        std::copy(decoded.begin(),decoded.end(),out+row*dimension);}
    return Status::success();
}


Status validate_q5_embedding_kernel_impl(Backend& backend,OperatorRegistry& operators,QuantizationRegistry& quantizations,bool q5_1){
    const BackendKind kind=backend.kind();
    if((kind!=BackendKind::DawnWebGpu&&kind!=BackendKind::OpenGles)||!backend.healthy())
        return {StatusCode::NotReady,"Q5 Embedding validation requires a healthy Dawn or GLES backend"};
    const uint32_t type_id=q5_1?7u:6u;
    auto format=quantizations.formats_.find(type_id);
    if(format==quantizations.formats_.end()||format->second.decoder_state!=SupportState::Validated)
        return {StatusCode::NotReady,"Q5 host decoder must be validated before embedding kernel promotion"};
    const std::string suffix=q5_1?"q5-1":"q5-0";
    const std::string kernel_id=(kind==BackendKind::OpenGles?"gles-":"dawn-")+suffix+"-embedding-f32-v1";
    auto kernel=std::find_if(operators.kernels_.begin(),operators.kernels_.end(),[&](const KernelDefinition& k){
        return k.kernel_id==kernel_id&&k.backend==kind;
    });
    if(kernel==operators.kernels_.end())return {StatusCode::NotReady,"Q5 Embedding candidate is not registered for selected backend"};
    if(kernel->validation==KernelValidation::Validated){
        format->second.backend_gpu_kernel_states[kind]=SupportState::Validated;
        format->second.gpu_kernel_state=SupportState::Validated;
        return Status::success();
    }
    if(kernel->validation!=KernelValidation::Unverified)return {StatusCode::Unsupported,"Q5 Embedding candidate is rejected"};
    const bool gles=kind==BackendKind::OpenGles;
    const std::string& registered=gles?kernel->glsl_source:kernel->wgsl_source;
    const std::string& canonical=q5_1?(gles?q5_1_embedding_glsl_source():q5_1_embedding_wgsl_source())
                                    :(gles?q5_0_embedding_glsl_source():q5_0_embedding_wgsl_source());
    if(registered.empty()||registered!=canonical)return {StatusCode::InvalidArgument,"Q5 Embedding registered shader differs from canonical source"};
    constexpr uint32_t vocabulary=2,dimension=32;
    const uint32_t block_bytes=q5_1?24u:22u;
    const uint64_t bytes=static_cast<uint64_t>(vocabulary)*block_bytes;
    const std::vector<uint32_t> token_ids={1,0};
    std::vector<uint8_t> encoded(static_cast<size_t>(bytes),0);
    for(uint32_t row=0;row<vocabulary;++row){
        const size_t off=static_cast<size_t>(row)*block_bytes;
        encoded[off]=0x00;encoded[off+1]=0x38;
        const uint32_t qh=q5_1?4u:2u,qs=q5_1?8u:6u;
        if(q5_1){encoded[off+2]=0x00;encoded[off+3]=0xbc;}
        for(uint32_t i=0;i<4;++i)encoded[off+qh+i]=static_cast<uint8_t>(0x69u+row+i*29u);
        for(uint32_t i=0;i<16;++i){
            const uint8_t low=static_cast<uint8_t>((i+row)%16);
            const uint8_t high=static_cast<uint8_t>((i+row+7)%16);
            encoded[off+qs+i]=static_cast<uint8_t>(low|(high<<4));
        }
    }
    std::vector<float> expected(token_ids.size()*dimension),actual(token_ids.size()*dimension);
    Status status=q5_1?reference_q5_1_embedding(encoded.data(),bytes,vocabulary,dimension,token_ids,expected.data())
                      :reference_q5_0_embedding(encoded.data(),bytes,vocabulary,dimension,token_ids,expected.data());
    if(!status.ok())return status;
    const uint64_t padded=(bytes+3)&~uint64_t{3};
    BufferDesc wd{padded,4,MemoryDomain::Device,true,"q5-embedding-validation-weights"};
    BufferDesc td{token_ids.size()*sizeof(uint32_t),alignof(uint32_t),MemoryDomain::Device,false,"q5-embedding-validation-ids"};
    BufferDesc od{actual.size()*sizeof(float),alignof(float),MemoryDomain::Device,false,"q5-embedding-validation-output"};
    std::shared_ptr<Buffer> wb,tb,ob;
    status=backend.allocate(wd,wb);if(!status.ok())return status;
    status=backend.allocate(td,tb);if(!status.ok())return status;
    status=backend.allocate(od,ob);if(!status.ok())return status;
    status=backend.upload(wb,0,encoded.data(),bytes);if(!status.ok())return status;
    status=backend.upload(tb,0,token_ids.data(),td.size_bytes);if(!status.ok())return status;
    TensorDesc otd;otd.shape={token_ids.size(),dimension};otd.data_type=DataType::F32;
    Tensor output;status=Tensor::create(otd,ob,0,output);if(!status.ok())return status;
    ComputeDispatch dispatch;
    if(q5_1)status=gles?build_q5_1_embedding_gles_candidate(wb,bytes,vocabulary,dimension,tb,token_ids,output,dispatch)
                       :build_q5_1_embedding_candidate(wb,bytes,vocabulary,dimension,tb,token_ids,output,dispatch);
    else status=gles?build_q5_0_embedding_gles_candidate(wb,bytes,vocabulary,dimension,tb,token_ids,output,dispatch)
                    :build_q5_0_embedding_candidate(wb,bytes,vocabulary,dimension,tb,token_ids,output,dispatch);
    if(!status.ok())return status;
    if(dispatch.kernel_id!=kernel_id||dispatch.backend!=kind)return {StatusCode::InvalidArgument,"Q5 Embedding candidate identity mismatch"};
    dispatch.validation=KernelValidation::Validated;
    std::unique_ptr<CommandBatch> commands;status=backend.begin_commands(commands);if(!status.ok())return status;
    if(!commands)return {StatusCode::BackendFailure,"Q5 Embedding backend returned empty command batch"};
    status=commands->dispatch_compute(dispatch);if(!status.ok())return status;
    status=backend.submit(std::move(commands));if(!status.ok())return status;
    status=backend.synchronize();if(!status.ok())return status;
    status=backend.readback(ob,0,actual.data(),actual.size()*sizeof(float));if(!status.ok())return status;
    for(size_t i=0;i<actual.size();++i){
        const float tolerance=1.0e-4f*std::max(1.0f,std::fabs(expected[i]));
        if(!std::isfinite(actual[i])||std::fabs(actual[i]-expected[i])>tolerance)
            return {StatusCode::NumericalFailure,"Q5 Embedding GPU output differs from independent decoder reference"};
    }
    kernel->validation=KernelValidation::Validated;
    format->second.backend_gpu_kernel_states[kind]=SupportState::Validated;
    format->second.gpu_kernel_state=SupportState::Validated;
    return Status::success();
}
Status validate_q5_0_embedding_kernel(Backend& b,OperatorRegistry& o,QuantizationRegistry& q){return validate_q5_embedding_kernel_impl(b,o,q,false);}
Status validate_q5_1_embedding_kernel(Backend& b,OperatorRegistry& o,QuantizationRegistry& q){return validate_q5_embedding_kernel_impl(b,o,q,true);}



namespace {
float decode_f16_value(uint16_t value) {
    const uint32_t sign = static_cast<uint32_t>(value & 0x8000u) << 16u;
    uint32_t exponent = (value >> 10u) & 0x1fu;
    uint32_t mantissa = value & 0x03ffu;
    uint32_t bits = 0;
    if (exponent == 0) {
        if (mantissa == 0) bits = sign;
        else {
            int shift = 0;
            while ((mantissa & 0x0400u) == 0) { mantissa <<= 1u; ++shift; }
            mantissa &= 0x03ffu;
            const uint32_t exp32 = static_cast<uint32_t>(127 - 14 - shift);
            bits = sign | (exp32 << 23u) | (mantissa << 13u);
        }
    } else if (exponent == 0x1fu) {
        bits = sign | 0x7f800000u | (mantissa << 13u);
    } else {
        bits = sign | ((exponent + (127u - 15u)) << 23u) | (mantissa << 13u);
    }
    float result = 0.0f;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}
float decode_bf16_value(uint16_t value) {
    const uint32_t bits = static_cast<uint32_t>(value) << 16u;
    float result = 0.0f;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}
Status decode_fp16_storage(const uint8_t* source,uint64_t bytes,float* output,uint64_t elements,bool bf16){
    if((!source||!output)&&elements)return {StatusCode::InvalidArgument,"16-bit weight decoder received a null pointer"};
    if(elements>UINT64_MAX/2||bytes!=elements*2)return {StatusCode::InvalidArgument,"16-bit weight storage size does not match element count"};
    for(uint64_t i=0;i<elements;++i){
        const uint16_t value=static_cast<uint16_t>(source[i*2])|static_cast<uint16_t>(source[i*2+1]<<8u);
        output[i]=bf16?decode_bf16_value(value):decode_f16_value(value);
        if(!std::isfinite(output[i]))return {StatusCode::NumericalFailure,"16-bit weight decoder produced a non-finite value"};
    }
    return Status::success();
}
Status build_fp16_matmul_candidate_impl(const Tensor& activations,const std::shared_ptr<Buffer>& weights,
    uint64_t bytes,const Tensor& output,ComputeDispatch& dispatch,bool bf16){
    dispatch=ComputeDispatch{};
    const std::string type=bf16?"BF16":"F16",suffix=bf16?"bf16":"f16";
    if(!weights||weights->memory_domain()==MemoryDomain::Host||!is_device_contiguous_f32(activations)||
       !is_device_contiguous_f32(output)||!activations.buffer()||!output.buffer()||
       activations.descriptor().shape.size()!=2||
       (output.descriptor().shape.size()!=2&&output.descriptor().shape.size()!=3))
        return {StatusCode::Unsupported,type+" MatMul requires F32 activations and packed 16-bit weights"};
    const auto&a=activations.descriptor().shape;const auto&o=output.descriptor().shape;
    const uint64_t channels=o.size()==2?o[1]:(o[2]&&o[1]<=UINT64_MAX/o[2]?o[1]*o[2]:0);
    if(!a[0]||!a[1]||o[0]!=a[0]||!channels||a[0]>UINT32_MAX||a[1]>UINT32_MAX||
       channels>UINT32_MAX||activations.element_count()>UINT32_MAX||output.element_count()>UINT32_MAX)
        return {StatusCode::InvalidArgument,type+" MatMul dimensions are invalid"};
    if(a[1]>UINT64_MAX/2||channels>UINT64_MAX/(a[1]*2))
        return {StatusCode::InvalidArgument,type+" weight byte size overflows"};
    const uint64_t expected=channels*a[1]*2;
    if(bytes!=expected||bytes>UINT32_MAX-3ull)return {StatusCode::InvalidArgument,type+" weight bytes do not match [N,K]"};
    const uint64_t padded=(bytes+3)&~uint64_t{3};
    if(weights->size_bytes()<padded||output.byte_size()!=output.element_count()*sizeof(float)||
       output.buffer().get()==activations.buffer().get()||output.buffer().get()==weights.get())
        return {StatusCode::InvalidArgument,type+" MatMul buffer ranges are inconsistent"};
    dispatch.label=suffix+"-matmul-f32-activation-generic-unverified";
    dispatch.kernel_id="dawn-"+suffix+"-matmul-f32-activation-v1";
    dispatch.backend=BackendKind::DawnWebGpu;
    dispatch.wgsl_source=bf16?kBF16MatMulWgsl:kF16MatMulWgsl;
    dispatch.workgroup_size_x=dispatch.workgroup_size_y=8;
    const std::pair<std::string,std::string> constants[]={
        {"const ROWS: u32 = 1u;","const ROWS: u32 = "+std::to_string(a[0])+"u;"},
        {"const OUTPUT_CHANNELS: u32 = 1u;","const OUTPUT_CHANNELS: u32 = "+std::to_string(channels)+"u;"},
        {"const INNER: u32 = 1u;","const INNER: u32 = "+std::to_string(a[1])+"u;"},
        {"const WEIGHT_ROW_BYTES: u32 = 2u;","const WEIGHT_ROW_BYTES: u32 = "+std::to_string(a[1]*2)+"u;"}
    };
    for(const auto& item:constants){const size_t pos=dispatch.wgsl_source.find(item.first);if(pos==std::string::npos)return {StatusCode::BackendFailure,type+" WGSL placeholder missing"};dispatch.wgsl_source.replace(pos,item.first.size(),item.second);}
    dispatch.workgroups_x=static_cast<uint32_t>((channels+7)/8);dispatch.workgroups_y=static_cast<uint32_t>((a[0]+7)/8);
    dispatch.bindings={{0,activations.buffer(),activations.byte_offset(),activations.byte_size()},
                       {1,weights,0,padded},{2,output.buffer(),output.byte_offset(),output.byte_size()}};
    return Status::success();
}
}
Status build_f16_matmul_candidate(const Tensor&a,const std::shared_ptr<Buffer>&w,uint64_t b,const Tensor&o,ComputeDispatch&d){return build_fp16_matmul_candidate_impl(a,w,b,o,d,false);}
Status build_bf16_matmul_candidate(const Tensor&a,const std::shared_ptr<Buffer>&w,uint64_t b,const Tensor&o,ComputeDispatch&d){return build_fp16_matmul_candidate_impl(a,w,b,o,d,true);}
static Status build_fp16_matmul_gles_candidate_impl(const Tensor&a,const std::shared_ptr<Buffer>&w,uint64_t b,const Tensor&o,ComputeDispatch&d,bool bf16){
    Status status=build_fp16_matmul_candidate_impl(a,w,b,o,d,bf16);if(!status.ok())return status;
    const std::string suffix=bf16?"bf16":"f16";
    d.label=suffix+"-matmul-f32-activation-gles-generic-unverified";
    d.kernel_id="gles-"+suffix+"-matmul-f32-activation-v1";d.backend=BackendKind::OpenGles;
    d.wgsl_source.clear();d.glsl_source=bf16?kBF16MatMulGlsl:kF16MatMulGlsl;
    const uint64_t channels=o.element_count()/a.descriptor().shape[0];
    const std::pair<std::string,std::string> constants[]={
        {"const uint ROWS = 1u;","const uint ROWS = "+std::to_string(a.descriptor().shape[0])+"u;"},
        {"const uint OUTPUT_CHANNELS = 1u;","const uint OUTPUT_CHANNELS = "+std::to_string(channels)+"u;"},
        {"const uint INNER = 1u;","const uint INNER = "+std::to_string(a.descriptor().shape[1])+"u;"},
        {"const uint WEIGHT_ROW_BYTES = 2u;","const uint WEIGHT_ROW_BYTES = "+std::to_string(a.descriptor().shape[1]*2)+"u;"}
    };
    for(const auto& item:constants){const size_t pos=d.glsl_source.find(item.first);if(pos==std::string::npos)return {StatusCode::BackendFailure,"16-bit GLES MatMul placeholder missing"};d.glsl_source.replace(pos,item.first.size(),item.second);}
    return Status::success();
}
Status build_f16_matmul_gles_candidate(const Tensor&a,const std::shared_ptr<Buffer>&w,uint64_t b,const Tensor&o,ComputeDispatch&d){return build_fp16_matmul_gles_candidate_impl(a,w,b,o,d,false);}
Status build_bf16_matmul_gles_candidate(const Tensor&a,const std::shared_ptr<Buffer>&w,uint64_t b,const Tensor&o,ComputeDispatch&d){return build_fp16_matmul_gles_candidate_impl(a,w,b,o,d,true);}

Status reference_f16_matmul(const float*a,const uint8_t*w,uint64_t bytes,float*out,uint32_t rows,uint32_t channels,uint32_t inner,bool bf16){
    if((!a||!w||!out)&&rows&&channels&&inner)return {StatusCode::InvalidArgument,"16-bit MatMul reference received a null pointer"};
    if(!rows||!channels||!inner||inner>UINT64_MAX/2||channels>UINT64_MAX/(static_cast<uint64_t>(inner)*2)||
       bytes!=static_cast<uint64_t>(channels)*inner*2)
        return {StatusCode::InvalidArgument,"16-bit MatMul dimensions or weight bytes are invalid"};
    std::vector<float> decoded(inner);
    for(uint32_t channel=0;channel<channels;++channel){
        Status status=decode_fp16_storage(w+static_cast<uint64_t>(channel)*inner*2,static_cast<uint64_t>(inner)*2,decoded.data(),inner,bf16);
        if(!status.ok())return status;
        for(uint32_t row=0;row<rows;++row){float sum=0;for(uint32_t k=0;k<inner;++k)sum+=a[static_cast<uint64_t>(row)*inner+k]*decoded[k];
            out[static_cast<uint64_t>(row)*channels+channel]=sum;}
    }
    return Status::success();
}

namespace {
Status build_fp16_embedding_candidate_impl(const std::shared_ptr<Buffer>&weights,uint64_t bytes,
    uint32_t vocabulary,uint32_t dimension,const std::shared_ptr<Buffer>&token_buffer,
    const std::vector<uint32_t>&token_ids,const Tensor&output,ComputeDispatch&dispatch,bool bf16){
    dispatch=ComputeDispatch{};
    const std::string type=bf16?"BF16":"F16",suffix=bf16?"bf16":"f16";
    if(!weights||!token_buffer||!output.buffer()||weights->memory_domain()==MemoryDomain::Host||
       token_buffer->memory_domain()==MemoryDomain::Host||output.descriptor().data_type!=DataType::F32||
       output.descriptor().layout!=TensorLayout::Contiguous||output.descriptor().memory==MemoryDomain::Host||
       output.descriptor().shape.size()!=2||token_ids.empty())
        return {StatusCode::Unsupported,type+" Embedding requires packed weights, device token IDs and contiguous F32 output"};
    if(!vocabulary||!dimension||token_ids.size()!=output.descriptor().shape[0]||
       output.descriptor().shape[1]!=dimension||token_ids.size()>UINT32_MAX||output.element_count()>UINT32_MAX)
        return {StatusCode::InvalidArgument,type+" Embedding dimensions or token count are invalid"};
    for(uint32_t id:token_ids)if(id>=vocabulary)return {StatusCode::InvalidArgument,type+" Embedding token ID is outside vocabulary"};
    if(static_cast<uint64_t>(dimension)>UINT64_MAX/2||static_cast<uint64_t>(vocabulary)>UINT64_MAX/(static_cast<uint64_t>(dimension)*2))
        return {StatusCode::InvalidArgument,type+" Embedding weight byte size overflows"};
    const uint64_t expected=static_cast<uint64_t>(vocabulary)*dimension*2;
    if(bytes!=expected||bytes>UINT32_MAX-3ull)return {StatusCode::InvalidArgument,type+" Embedding weight bytes do not match [vocabulary,dimension]"};
    const uint64_t padded=(bytes+3)&~uint64_t{3},token_bytes=token_ids.size()*sizeof(uint32_t);
    if(weights->size_bytes()<padded||token_buffer->size_bytes()<token_bytes||
       output.byte_size()!=output.element_count()*sizeof(float)||output.buffer().get()==weights.get()||
       output.buffer().get()==token_buffer.get()||weights.get()==token_buffer.get())
        return {StatusCode::InvalidArgument,type+" Embedding buffer ranges are inconsistent"};
    dispatch.label=suffix+"-embedding-f32-generic-unverified";
    dispatch.kernel_id="dawn-"+suffix+"-embedding-f32-v1";
    dispatch.backend=BackendKind::DawnWebGpu;
    dispatch.wgsl_source=bf16?kBF16EmbeddingWgsl:kF16EmbeddingWgsl;
    dispatch.workgroup_size_x=64;
    const std::pair<std::string,std::string> constants[]={
        {"const VOCABULARY_SIZE: u32 = 1u;","const VOCABULARY_SIZE: u32 = "+std::to_string(vocabulary)+"u;"},
        {"const EMBEDDING_DIMENSION: u32 = 1u;","const EMBEDDING_DIMENSION: u32 = "+std::to_string(dimension)+"u;"},
        {"const TOKEN_COUNT: u32 = 1u;","const TOKEN_COUNT: u32 = "+std::to_string(token_ids.size())+"u;"},
        {"const TOTAL_ELEMENTS: u32 = 1u;","const TOTAL_ELEMENTS: u32 = "+std::to_string(output.element_count())+"u;"},
        {"const WEIGHT_ROW_BYTES: u32 = 2u;","const WEIGHT_ROW_BYTES: u32 = "+std::to_string(static_cast<uint64_t>(dimension)*2)+"u;"}
    };
    for(const auto& item:constants){const size_t pos=dispatch.wgsl_source.find(item.first);if(pos==std::string::npos)return {StatusCode::BackendFailure,type+" Embedding WGSL placeholder missing"};dispatch.wgsl_source.replace(pos,item.first.size(),item.second);}
    dispatch.workgroups_x=static_cast<uint32_t>((output.element_count()+63)/64);
    dispatch.bindings={{0,weights,0,padded},{1,token_buffer,0,token_bytes},{2,output.buffer(),output.byte_offset(),output.byte_size()}};
    return Status::success();
}
}
Status build_f16_embedding_candidate(const std::shared_ptr<Buffer>&w,uint64_t b,uint32_t v,uint32_t d,const std::shared_ptr<Buffer>&tb,const std::vector<uint32_t>&ids,const Tensor&o,ComputeDispatch&x){return build_fp16_embedding_candidate_impl(w,b,v,d,tb,ids,o,x,false);}
Status build_bf16_embedding_candidate(const std::shared_ptr<Buffer>&w,uint64_t b,uint32_t v,uint32_t d,const std::shared_ptr<Buffer>&tb,const std::vector<uint32_t>&ids,const Tensor&o,ComputeDispatch&x){return build_fp16_embedding_candidate_impl(w,b,v,d,tb,ids,o,x,true);}
static Status build_fp16_embedding_gles_candidate_impl(const std::shared_ptr<Buffer>&w,uint64_t b,uint32_t v,uint32_t d,const std::shared_ptr<Buffer>&tb,const std::vector<uint32_t>&ids,const Tensor&o,ComputeDispatch&x,bool bf16){
    Status status=build_fp16_embedding_candidate_impl(w,b,v,d,tb,ids,o,x,bf16);if(!status.ok())return status;
    const std::string suffix=bf16?"bf16":"f16";
    x.label=suffix+"-embedding-gles-f32-generic-unverified";
    x.kernel_id="gles-"+suffix+"-embedding-f32-v1";x.backend=BackendKind::OpenGles;
    x.wgsl_source.clear();x.glsl_source=bf16?kBF16EmbeddingGlsl:kF16EmbeddingGlsl;
    const std::pair<std::string,std::string> constants[]={
        {"const uint VOCABULARY_SIZE = 1u;","const uint VOCABULARY_SIZE = "+std::to_string(v)+"u;"},
        {"const uint EMBEDDING_DIMENSION = 1u;","const uint EMBEDDING_DIMENSION = "+std::to_string(d)+"u;"},
        {"const uint TOKEN_COUNT = 1u;","const uint TOKEN_COUNT = "+std::to_string(ids.size())+"u;"},
        {"const uint TOTAL_ELEMENTS = 1u;","const uint TOTAL_ELEMENTS = "+std::to_string(o.element_count())+"u;"},
        {"const uint WEIGHT_ROW_BYTES = 2u;","const uint WEIGHT_ROW_BYTES = "+std::to_string(static_cast<uint64_t>(d)*2)+"u;"}
    };
    for(const auto& item:constants){const size_t pos=x.glsl_source.find(item.first);if(pos==std::string::npos)return {StatusCode::BackendFailure,"16-bit GLES Embedding placeholder missing"};x.glsl_source.replace(pos,item.first.size(),item.second);}
    return Status::success();
}
Status build_f16_embedding_gles_candidate(const std::shared_ptr<Buffer>&w,uint64_t b,uint32_t v,uint32_t d,const std::shared_ptr<Buffer>&tb,const std::vector<uint32_t>&ids,const Tensor&o,ComputeDispatch&x){return build_fp16_embedding_gles_candidate_impl(w,b,v,d,tb,ids,o,x,false);}
Status build_bf16_embedding_gles_candidate(const std::shared_ptr<Buffer>&w,uint64_t b,uint32_t v,uint32_t d,const std::shared_ptr<Buffer>&tb,const std::vector<uint32_t>&ids,const Tensor&o,ComputeDispatch&x){return build_fp16_embedding_gles_candidate_impl(w,b,v,d,tb,ids,o,x,true);}

Status reference_f16_embedding(const uint8_t*w,uint64_t bytes,uint32_t vocabulary,uint32_t dimension,const std::vector<uint32_t>&ids,float*out,bool bf16){
    if((!w||!out)&&!ids.empty())return {StatusCode::InvalidArgument,"16-bit Embedding reference received a null pointer"};
    if(!vocabulary||!dimension||ids.empty()||dimension>UINT64_MAX/2||
       vocabulary>UINT64_MAX/(static_cast<uint64_t>(dimension)*2)||
       bytes!=static_cast<uint64_t>(vocabulary)*dimension*2)
        return {StatusCode::InvalidArgument,"16-bit Embedding dimensions or weight bytes are invalid"};
    for(size_t row=0;row<ids.size();++row){
        if(ids[row]>=vocabulary)return {StatusCode::InvalidArgument,"16-bit Embedding token ID is outside vocabulary"};
        Status status=decode_fp16_storage(w+static_cast<uint64_t>(ids[row])*dimension*2,static_cast<uint64_t>(dimension)*2,
            out+row*dimension,dimension,bf16);
        if(!status.ok())return status;
    }
    return Status::success();
}

Status validate_fp16_matmul_kernel_impl(Backend&backend,OperatorRegistry&operators,QuantizationRegistry&quantizations,bool bf16){
    const BackendKind kind=backend.kind();if((kind!=BackendKind::DawnWebGpu&&kind!=BackendKind::OpenGles)||!backend.healthy())
        return {StatusCode::NotReady,"16-bit MatMul validation requires a healthy Dawn or GLES backend"};
    const uint32_t type_id=bf16?30u:1u;auto format=quantizations.formats_.find(type_id);
    if(format==quantizations.formats_.end()||format->second.decoder_state!=SupportState::Validated)
        return {StatusCode::NotReady,"16-bit host decoder is not registered"};
    const std::string suffix=bf16?"bf16":"f16";
    const std::string kernel_id=(kind==BackendKind::OpenGles?"gles-":"dawn-")+suffix+"-matmul-f32-activation-v1";
    auto kernel=std::find_if(operators.kernels_.begin(),operators.kernels_.end(),[&](const KernelDefinition&k){return k.kernel_id==kernel_id&&k.backend==kind;});
    if(kernel==operators.kernels_.end())return {StatusCode::NotReady,"16-bit MatMul candidate is not registered"};
    if(kernel->validation==KernelValidation::Validated){format->second.backend_gpu_kernel_states[kind]=SupportState::Validated;format->second.gpu_kernel_state=SupportState::Validated;return Status::success();}
    if(kernel->validation!=KernelValidation::Unverified)return {StatusCode::Unsupported,"16-bit MatMul candidate is rejected"};
    const bool gles=kind==BackendKind::OpenGles;
    const std::string& source=gles?kernel->glsl_source:kernel->wgsl_source;
    const std::string& canonical=bf16?(gles?std::string(kBF16MatMulGlsl):std::string(kBF16MatMulWgsl))
                                      :(gles?std::string(kF16MatMulGlsl):std::string(kF16MatMulWgsl));
    if(source.empty()||source!=canonical)return {StatusCode::InvalidArgument,"16-bit MatMul shader differs from canonical source"};
    constexpr uint32_t rows=2,channels=2,inner=4;
    std::vector<float> activations={1.0f,0.5f,-1.0f,2.0f,0.25f,1.5f,0.5f,-0.5f};
    const float values[8]={1.0f,2.0f,0.5f,-1.0f,0.25f,1.0f,-0.5f,2.0f};
    std::vector<uint8_t> encoded(sizeof(values)/sizeof(float)*2);
    auto write16=[&](size_t index,uint16_t value){encoded[index*2]=static_cast<uint8_t>(value&255u);encoded[index*2+1]=static_cast<uint8_t>(value>>8u);};
    for(size_t i=0;i<8;++i){
        uint16_t bits=0;
        if(bf16){uint32_t raw=0;std::memcpy(&raw,&values[i],sizeof(raw));bits=static_cast<uint16_t>(raw>>16u);}
        else {
            // Values are exact half representables; use their canonical IEEE-754 binary16 encodings.
            const uint16_t table[8]={0x3c00,0x4000,0x3800,0xbc00,0x3400,0x3c00,0xb800,0x4000};
            bits=table[i];
        }
        write16(i,bits);
    }
    float expected[rows*channels]={},actual[rows*channels]={};
    Status status=reference_f16_matmul(activations.data(),encoded.data(),encoded.size(),expected,rows,channels,inner,bf16);
    if(!status.ok())return status;
    BufferDesc ad{activations.size()*sizeof(float),alignof(float),MemoryDomain::Device,false,"fp16-validation-activations"};
    BufferDesc wd{encoded.size(),4,MemoryDomain::Device,true,"fp16-validation-weights"};
    BufferDesc od{sizeof(actual),alignof(float),MemoryDomain::Device,false,"fp16-validation-output"};
    std::shared_ptr<Buffer> ab,wb,ob;
    status=backend.allocate(ad,ab);if(!status.ok())return status;
    status=backend.allocate(wd,wb);if(!status.ok())return status;
    status=backend.allocate(od,ob);if(!status.ok())return status;
    status=backend.upload(ab,0,activations.data(),ad.size_bytes);if(!status.ok())return status;
    status=backend.upload(wb,0,encoded.data(),encoded.size());if(!status.ok())return status;
    TensorDesc atd;atd.shape={rows,inner};atd.data_type=DataType::F32;
    TensorDesc otd;otd.shape={rows,channels};otd.data_type=DataType::F32;
    Tensor at,ot;status=Tensor::create(atd,ab,0,at);if(!status.ok())return status;
    status=Tensor::create(otd,ob,0,ot);if(!status.ok())return status;
    ComputeDispatch dispatch;
    if(bf16)status=gles?build_bf16_matmul_gles_candidate(at,wb,encoded.size(),ot,dispatch):build_bf16_matmul_candidate(at,wb,encoded.size(),ot,dispatch);
    else status=gles?build_f16_matmul_gles_candidate(at,wb,encoded.size(),ot,dispatch):build_f16_matmul_candidate(at,wb,encoded.size(),ot,dispatch);
    if(!status.ok())return status;
    if(dispatch.kernel_id!=kernel_id||dispatch.backend!=kind)return {StatusCode::InvalidArgument,"16-bit MatMul candidate identity mismatch"};
    dispatch.validation=KernelValidation::Validated;
    std::unique_ptr<CommandBatch> commands;status=backend.begin_commands(commands);if(!status.ok())return status;
    if(!commands)return {StatusCode::BackendFailure,"16-bit MatMul backend returned empty command batch"};
    status=commands->dispatch_compute(dispatch);if(!status.ok())return status;
    status=backend.submit(std::move(commands));if(!status.ok())return status;
    status=backend.synchronize();if(!status.ok())return status;
    status=backend.readback(ob,0,actual,sizeof(actual));if(!status.ok())return status;
    for(size_t i=0;i<rows*channels;++i){const float tolerance=1.0e-4f*std::max(1.0f,std::fabs(expected[i]));
        if(!std::isfinite(actual[i])||std::fabs(actual[i]-expected[i])>tolerance)return {StatusCode::NumericalFailure,"16-bit MatMul GPU output differs from host decoder reference"};}
    kernel->validation=KernelValidation::Validated;
    format->second.backend_gpu_kernel_states[kind]=SupportState::Validated;format->second.gpu_kernel_state=SupportState::Validated;
    return Status::success();
}
Status validate_fp16_embedding_kernel_impl(Backend&backend,OperatorRegistry&operators,QuantizationRegistry&quantizations,bool bf16){
    const BackendKind kind=backend.kind();if((kind!=BackendKind::DawnWebGpu&&kind!=BackendKind::OpenGles)||!backend.healthy())
        return {StatusCode::NotReady,"16-bit Embedding validation requires a healthy Dawn or GLES backend"};
    const uint32_t type_id=bf16?30u:1u;auto format=quantizations.formats_.find(type_id);
    if(format==quantizations.formats_.end()||format->second.decoder_state!=SupportState::Validated)
        return {StatusCode::NotReady,"16-bit host decoder is not registered"};
    const std::string suffix=bf16?"bf16":"f16";
    const std::string kernel_id=(kind==BackendKind::OpenGles?"gles-":"dawn-")+suffix+"-embedding-f32-v1";
    auto kernel=std::find_if(operators.kernels_.begin(),operators.kernels_.end(),[&](const KernelDefinition&k){return k.kernel_id==kernel_id&&k.backend==kind;});
    if(kernel==operators.kernels_.end())return {StatusCode::NotReady,"16-bit Embedding candidate is not registered"};
    if(kernel->validation==KernelValidation::Validated){format->second.backend_gpu_kernel_states[kind]=SupportState::Validated;format->second.gpu_kernel_state=SupportState::Validated;return Status::success();}
    if(kernel->validation!=KernelValidation::Unverified)return {StatusCode::Unsupported,"16-bit Embedding candidate is rejected"};
    const bool gles=kind==BackendKind::OpenGles;
    const std::string& source=gles?kernel->glsl_source:kernel->wgsl_source;
    const std::string& canonical=bf16?(gles?std::string(kBF16EmbeddingGlsl):std::string(kBF16EmbeddingWgsl))
                                      :(gles?std::string(kF16EmbeddingGlsl):std::string(kF16EmbeddingWgsl));
    if(source.empty()||source!=canonical)return {StatusCode::InvalidArgument,"16-bit Embedding shader differs from canonical source"};
    constexpr uint32_t vocabulary=2,dimension=4;
    const std::vector<uint32_t> ids={1,0};
    const float values[8]={1.0f,2.0f,3.0f,4.0f,-1.0f,0.5f,2.0f,0.0f};
    std::vector<uint8_t> encoded(16);
    const uint16_t half_bits[8]={0x3c00,0x4000,0x4200,0x4400,0xbc00,0x3800,0x4000,0x0000};
    for(size_t i=0;i<8;++i){uint16_t bits=half_bits[i];if(bf16){uint32_t raw=0;std::memcpy(&raw,&values[i],sizeof(raw));bits=static_cast<uint16_t>(raw>>16u);}encoded[i*2]=static_cast<uint8_t>(bits&255u);encoded[i*2+1]=static_cast<uint8_t>(bits>>8u);}
    float expected[8]={},actual[8]={};
    Status status=reference_f16_embedding(encoded.data(),encoded.size(),vocabulary,dimension,ids,expected,bf16);
    if(!status.ok())return status;
    BufferDesc wd{encoded.size(),4,MemoryDomain::Device,true,"fp16-embedding-validation-weights"};
    BufferDesc td{ids.size()*sizeof(uint32_t),4,MemoryDomain::Device,false,"fp16-embedding-validation-ids"};
    BufferDesc od{sizeof(actual),alignof(float),MemoryDomain::Device,false,"fp16-embedding-validation-output"};
    std::shared_ptr<Buffer> wb,tb,ob;
    status=backend.allocate(wd,wb);if(!status.ok())return status;
    status=backend.allocate(td,tb);if(!status.ok())return status;
    status=backend.allocate(od,ob);if(!status.ok())return status;
    status=backend.upload(wb,0,encoded.data(),encoded.size());if(!status.ok())return status;
    status=backend.upload(tb,0,ids.data(),td.size_bytes);if(!status.ok())return status;
    TensorDesc otd;otd.shape={ids.size(),dimension};otd.data_type=DataType::F32;
    Tensor output;status=Tensor::create(otd,ob,0,output);if(!status.ok())return status;
    ComputeDispatch dispatch;
    if(bf16)status=gles?build_bf16_embedding_gles_candidate(wb,encoded.size(),vocabulary,dimension,tb,ids,output,dispatch)
                       :build_bf16_embedding_candidate(wb,encoded.size(),vocabulary,dimension,tb,ids,output,dispatch);
    else status=gles?build_f16_embedding_gles_candidate(wb,encoded.size(),vocabulary,dimension,tb,ids,output,dispatch)
                    :build_f16_embedding_candidate(wb,encoded.size(),vocabulary,dimension,tb,ids,output,dispatch);
    if(!status.ok())return status;
    if(dispatch.kernel_id!=kernel_id||dispatch.backend!=kind)return {StatusCode::InvalidArgument,"16-bit Embedding candidate identity mismatch"};
    dispatch.validation=KernelValidation::Validated;
    std::unique_ptr<CommandBatch> commands;status=backend.begin_commands(commands);if(!status.ok())return status;
    if(!commands)return {StatusCode::BackendFailure,"16-bit Embedding backend returned empty command batch"};
    status=commands->dispatch_compute(dispatch);if(!status.ok())return status;
    status=backend.submit(std::move(commands));if(!status.ok())return status;
    status=backend.synchronize();if(!status.ok())return status;
    status=backend.readback(ob,0,actual,sizeof(actual));if(!status.ok())return status;
    for(size_t i=0;i<8;++i){const float tolerance=1.0e-4f*std::max(1.0f,std::fabs(expected[i]));
        if(!std::isfinite(actual[i])||std::fabs(actual[i]-expected[i])>tolerance)return {StatusCode::NumericalFailure,"16-bit Embedding GPU output differs from host decoder reference"};}
    kernel->validation=KernelValidation::Validated;
    format->second.backend_gpu_kernel_states[kind]=SupportState::Validated;format->second.gpu_kernel_state=SupportState::Validated;
    return Status::success();
}
Status validate_f16_matmul_kernel(Backend&b,OperatorRegistry&o,QuantizationRegistry&q){return validate_fp16_matmul_kernel_impl(b,o,q,false);}
Status validate_bf16_matmul_kernel(Backend&b,OperatorRegistry&o,QuantizationRegistry&q){return validate_fp16_matmul_kernel_impl(b,o,q,true);}
Status validate_f16_embedding_kernel(Backend&b,OperatorRegistry&o,QuantizationRegistry&q){return validate_fp16_embedding_kernel_impl(b,o,q,false);}
Status validate_bf16_embedding_kernel(Backend&b,OperatorRegistry&o,QuantizationRegistry&q){return validate_fp16_embedding_kernel_impl(b,o,q,true);}

Status decode_q6_k_blocks(const uint8_t*source,size_t source_bytes,float*destination,size_t destination_elements){
    constexpr size_t block_bytes=210,block_elements=256;
    if((!source||!destination)&&source_bytes)return {StatusCode::InvalidArgument,"Q6_K decoder received a null pointer"};
    if(source_bytes%block_bytes!=0)
        return {StatusCode::InvalidArgument,"Q6_K source byte count does not contain complete blocks"};
    const size_t blocks=source_bytes/block_bytes;
    if(blocks>std::numeric_limits<size_t>::max()/block_elements)
        return {StatusCode::Unsupported,"Q6_K destination element count overflows"};
    if(destination_elements!=blocks*block_elements)
        return {StatusCode::InvalidArgument,"Q6_K destination element count does not match complete blocks"};
    for(size_t block=0;block<blocks;++block){
        const uint8_t*encoded=source+block*block_bytes;
        const uint8_t*ql=encoded;
        const uint8_t*qh=encoded+128;
        const int8_t*scales=reinterpret_cast<const int8_t*>(encoded+192);
        const uint16_t half=static_cast<uint16_t>(encoded[208])|static_cast<uint16_t>(encoded[209]<<8u);
        const float d=decode_f16_value(half);
        if(!std::isfinite(d))return {StatusCode::NumericalFailure,"Q6_K block scale is non-finite"};
        float*output=destination+block*block_elements;
        for(size_t n=0;n<block_elements;n+=128){
            for(size_t l=0;l<32;++l){
                const size_t scale_group=l/16;
                const int8_t q1=static_cast<int8_t>((ql[l]&0x0fu)|(((qh[l]>>0u)&3u)<<4u))-32;
                const int8_t q2=static_cast<int8_t>((ql[l+32]&0x0fu)|(((qh[l]>>2u)&3u)<<4u))-32;
                const int8_t q3=static_cast<int8_t>((ql[l]>>4u)|(((qh[l]>>4u)&3u)<<4u))-32;
                const int8_t q4=static_cast<int8_t>((ql[l+32]>>4u)|(((qh[l]>>6u)&3u)<<4u))-32;
                output[n+l]=d*scales[scale_group+0]*q1;
                output[n+l+32]=d*scales[scale_group+2]*q2;
                output[n+l+64]=d*scales[scale_group+4]*q3;
                output[n+l+96]=d*scales[scale_group+6]*q4;
            }
            ql+=64;qh+=32;scales+=8;
        }
    }
    return Status::success();
}


namespace {
Status build_q6_k_matmul_candidate_impl(const Tensor&activations,const std::shared_ptr<Buffer>&weights,
    uint64_t bytes,const Tensor&output,ComputeDispatch&dispatch){
    dispatch=ComputeDispatch{};
    if(!weights||weights->memory_domain()==MemoryDomain::Host||!is_device_contiguous_f32(activations)||
       !is_device_contiguous_f32(output)||!activations.buffer()||!output.buffer()||
       activations.descriptor().shape.size()!=2||
       (output.descriptor().shape.size()!=2&&output.descriptor().shape.size()!=3))
        return {StatusCode::Unsupported,"Q6_K MatMul requires F32 activations and packed Q6_K weights"};
    const auto&a=activations.descriptor().shape;const auto&o=output.descriptor().shape;
    const uint64_t channels=o.size()==2?o[1]:(o[2]&&o[1]<=UINT64_MAX/o[2]?o[1]*o[2]:0);
    if(!a[0]||!a[1]||a[1]%256||o[0]!=a[0]||!channels||a[0]>UINT32_MAX||a[1]>UINT32_MAX||
       channels>UINT32_MAX||activations.element_count()>UINT32_MAX||output.element_count()>UINT32_MAX)
        return {StatusCode::InvalidArgument,"Q6_K MatMul dimensions are invalid or K is not block-aligned"};
    const uint64_t blocks=a[1]/256;
    if(blocks>UINT64_MAX/210||channels>UINT64_MAX/(blocks*210))
        return {StatusCode::InvalidArgument,"Q6_K weight byte size overflows"};
    const uint64_t expected=channels*blocks*210;
    if(bytes!=expected||bytes>UINT32_MAX-3ull)return {StatusCode::InvalidArgument,"Q6_K weight bytes do not match [N,K]"};
    const uint64_t padded=(bytes+3)&~uint64_t{3};
    if(weights->size_bytes()<padded||output.byte_size()!=output.element_count()*sizeof(float)||
       output.buffer().get()==activations.buffer().get()||output.buffer().get()==weights.get())
        return {StatusCode::InvalidArgument,"Q6_K MatMul buffer ranges are inconsistent"};
    dispatch.label="q6-k-matmul-f32-activation-generic-unverified";
    dispatch.kernel_id="dawn-q6-k-matmul-f32-activation-v1";
    dispatch.backend=BackendKind::DawnWebGpu;dispatch.wgsl_source=kQ6_KMatMulWgsl;
    dispatch.workgroup_size_x=dispatch.workgroup_size_y=8;
    const std::pair<std::string,std::string> constants[]={
        {"const ROWS: u32 = 1u;","const ROWS: u32 = "+std::to_string(a[0])+"u;"},
        {"const OUTPUT_CHANNELS: u32 = 1u;","const OUTPUT_CHANNELS: u32 = "+std::to_string(channels)+"u;"},
        {"const INNER: u32 = 256u;","const INNER: u32 = "+std::to_string(a[1])+"u;"},
        {"const WEIGHT_ROW_BYTES: u32 = 210u;","const WEIGHT_ROW_BYTES: u32 = "+std::to_string(blocks*210)+"u;"}
    };
    for(const auto&item:constants){const size_t pos=dispatch.wgsl_source.find(item.first);if(pos==std::string::npos)return {StatusCode::BackendFailure,"Q6_K WGSL placeholder missing"};dispatch.wgsl_source.replace(pos,item.first.size(),item.second);}
    dispatch.workgroups_x=static_cast<uint32_t>((channels+7)/8);dispatch.workgroups_y=static_cast<uint32_t>((a[0]+7)/8);
    dispatch.bindings={{0,activations.buffer(),activations.byte_offset(),activations.byte_size()},
                       {1,weights,0,padded},{2,output.buffer(),output.byte_offset(),output.byte_size()}};
    return Status::success();
}
}
Status build_q6_k_matmul_candidate(const Tensor&a,const std::shared_ptr<Buffer>&w,uint64_t b,const Tensor&o,ComputeDispatch&d){return build_q6_k_matmul_candidate_impl(a,w,b,o,d);}
static Status build_q6_k_matmul_gles_candidate_impl(const Tensor&a,const std::shared_ptr<Buffer>&w,uint64_t b,const Tensor&o,ComputeDispatch&d){
    Status status=build_q6_k_matmul_candidate_impl(a,w,b,o,d);if(!status.ok())return status;
    const uint64_t channels=o.element_count()/a.descriptor().shape[0],blocks=a.descriptor().shape[1]/256;
    d.label="q6-k-matmul-f32-activation-gles-generic-unverified";
    d.kernel_id="gles-q6-k-matmul-f32-activation-v1";d.backend=BackendKind::OpenGles;
    d.wgsl_source.clear();d.glsl_source=kQ6_KMatMulGlsl;
    const std::pair<std::string,std::string> constants[]={
        {"const uint ROWS=1u;","const uint ROWS="+std::to_string(a.descriptor().shape[0])+"u;"},
        {"const uint OUTPUT_CHANNELS=1u;","const uint OUTPUT_CHANNELS="+std::to_string(channels)+"u;"},
        {"const uint INNER=256u;","const uint INNER="+std::to_string(a.descriptor().shape[1])+"u;"},
        {"const uint WEIGHT_ROW_BYTES=210u;","const uint WEIGHT_ROW_BYTES="+std::to_string(blocks*210)+"u;"}
    };
    for(const auto&item:constants){const size_t pos=d.glsl_source.find(item.first);if(pos==std::string::npos)return {StatusCode::BackendFailure,"Q6_K GLES GLSL placeholder missing"};d.glsl_source.replace(pos,item.first.size(),item.second);}
    return Status::success();
}
Status build_q6_k_matmul_gles_candidate(const Tensor&a,const std::shared_ptr<Buffer>&w,uint64_t b,const Tensor&o,ComputeDispatch&d){return build_q6_k_matmul_gles_candidate_impl(a,w,b,o,d);}
Status reference_q6_k_matmul(const float*a,const uint8_t*w,uint64_t bytes,float*out,uint32_t rows,uint32_t channels,uint32_t inner){
    if((!a||!w||!out)&&rows&&channels&&inner)return {StatusCode::InvalidArgument,"Q6_K MatMul reference received a null pointer"};
    if(!rows||!channels||!inner||inner%256)return {StatusCode::InvalidArgument,"Q6_K MatMul dimensions must be block-aligned"};
    const uint64_t blocks=inner/256;
    if(blocks==0||blocks>UINT64_MAX/210||channels>UINT64_MAX/(blocks*210)||
       bytes!=static_cast<uint64_t>(channels)*blocks*210)
        return {StatusCode::InvalidArgument,"Q6_K MatMul weight bytes mismatch"};
    std::vector<float> decoded(inner);
    for(uint32_t channel=0;channel<channels;++channel){
        Status status=decode_q6_k_blocks(w+static_cast<uint64_t>(channel)*blocks*210,blocks*210,decoded.data(),inner);
        if(!status.ok())return status;
        for(uint32_t row=0;row<rows;++row){float sum=0;for(uint32_t k=0;k<inner;++k)sum+=a[static_cast<uint64_t>(row)*inner+k]*decoded[k];
            out[static_cast<uint64_t>(row)*channels+channel]=sum;}
    }
    return Status::success();
}


Status validate_q6_k_matmul_kernel_impl(Backend&backend,OperatorRegistry&operators,QuantizationRegistry&quantizations){
    const BackendKind kind=backend.kind();if((kind!=BackendKind::DawnWebGpu&&kind!=BackendKind::OpenGles)||!backend.healthy())
        return {StatusCode::NotReady,"Q6_K MatMul validation requires a healthy Dawn or GLES backend"};
    auto format=quantizations.formats_.find(14);
    if(format==quantizations.formats_.end()||format->second.decoder_state!=SupportState::Validated)
        return {StatusCode::NotReady,"Q6_K host decoder is not validated"};
    const std::string kernel_id=(kind==BackendKind::OpenGles?"gles-":"dawn-")+std::string("q6-k-matmul-f32-activation-v1");
    auto kernel=std::find_if(operators.kernels_.begin(),operators.kernels_.end(),[&](const KernelDefinition&k){return k.kernel_id==kernel_id&&k.backend==kind;});
    if(kernel==operators.kernels_.end())return {StatusCode::NotReady,"Q6_K MatMul candidate is not registered"};
    if(kernel->validation==KernelValidation::Validated){format->second.backend_gpu_kernel_states[kind]=SupportState::Validated;format->second.gpu_kernel_state=SupportState::Validated;return Status::success();}
    if(kernel->validation!=KernelValidation::Unverified)return {StatusCode::Unsupported,"Q6_K MatMul candidate is rejected"};
    const bool gles=kind==BackendKind::OpenGles;
    const std::string& registered=gles?kernel->glsl_source:kernel->wgsl_source;
    const std::string canonical=gles?std::string(kQ6_KMatMulGlsl):std::string(kQ6_KMatMulWgsl);
    if(registered.empty()||registered!=canonical)return {StatusCode::InvalidArgument,"Q6_K MatMul shader differs from canonical source"};
    constexpr uint32_t rows=2,channels=2,inner=256;
    const uint64_t bytes=static_cast<uint64_t>(channels)*210;
    std::vector<uint8_t> encoded(static_cast<size_t>(bytes),0);
    for(uint32_t channel=0;channel<channels;++channel){
        uint8_t*block=encoded.data()+channel*210;
        std::fill(block,block+128,0x11);
        for(uint32_t i=0;i<64;++i)block[128+i]=static_cast<uint8_t>(0x55u+i);
        for(uint32_t i=0;i<16;++i)block[192+i]=static_cast<uint8_t>(1+(i%3));
        block[208]=0x00;block[209]=0x3c;
    }
    std::vector<float> activations(rows*inner);
    for(size_t i=0;i<activations.size();++i)activations[i]=(static_cast<int>(i%11)-5)*0.125f;
    float expected[rows*channels]={},actual[rows*channels]={};
    Status status=reference_q6_k_matmul(activations.data(),encoded.data(),bytes,expected,rows,channels,inner);
    if(!status.ok())return status;
    BufferDesc ad{activations.size()*sizeof(float),alignof(float),MemoryDomain::Device,false,"q6-k-validation-activations"};
    BufferDesc wd{bytes,4,MemoryDomain::Device,true,"q6-k-validation-weights"};
    BufferDesc od{sizeof(actual),alignof(float),MemoryDomain::Device,false,"q6-k-validation-output"};
    std::shared_ptr<Buffer> ab,wb,ob;
    status=backend.allocate(ad,ab);if(!status.ok())return status;
    status=backend.allocate(wd,wb);if(!status.ok())return status;
    status=backend.allocate(od,ob);if(!status.ok())return status;
    status=backend.upload(ab,0,activations.data(),ad.size_bytes);if(!status.ok())return status;
    status=backend.upload(wb,0,encoded.data(),bytes);if(!status.ok())return status;
    TensorDesc atd;atd.shape={rows,inner};atd.data_type=DataType::F32;
    TensorDesc otd;otd.shape={rows,channels};otd.data_type=DataType::F32;
    Tensor at,ot;status=Tensor::create(atd,ab,0,at);if(!status.ok())return status;
    status=Tensor::create(otd,ob,0,ot);if(!status.ok())return status;
    ComputeDispatch dispatch;
    status=gles?build_q6_k_matmul_gles_candidate(at,wb,bytes,ot,dispatch):build_q6_k_matmul_candidate(at,wb,bytes,ot,dispatch);
    if(!status.ok())return status;
    if(dispatch.kernel_id!=kernel_id||dispatch.backend!=kind)return {StatusCode::InvalidArgument,"Q6_K MatMul candidate identity mismatch"};
    dispatch.validation=KernelValidation::Validated;
    std::unique_ptr<CommandBatch>commands;status=backend.begin_commands(commands);if(!status.ok())return status;
    if(!commands)return {StatusCode::BackendFailure,"Q6_K backend returned empty command batch"};
    status=commands->dispatch_compute(dispatch);if(!status.ok())return status;
    status=backend.submit(std::move(commands));if(!status.ok())return status;
    status=backend.synchronize();if(!status.ok())return status;
    status=backend.readback(ob,0,actual,sizeof(actual));if(!status.ok())return status;
    for(size_t i=0;i<rows*channels;++i){const float tolerance=1.0e-4f*std::max(1.0f,std::fabs(expected[i]));
        if(!std::isfinite(actual[i])||std::fabs(actual[i]-expected[i])>tolerance)
            return {StatusCode::NumericalFailure,"Q6_K MatMul GPU output differs from independent block decoder"};}
    kernel->validation=KernelValidation::Validated;
    format->second.backend_gpu_kernel_states[kind]=SupportState::Validated;format->second.gpu_kernel_state=SupportState::Validated;
    return Status::success();
}
Status validate_q6_k_matmul_kernel(Backend&b,OperatorRegistry&o,QuantizationRegistry&q){return validate_q6_k_matmul_kernel_impl(b,o,q);}


namespace {
void decode_k4_scale_min(uint32_t index,const uint8_t*packed,uint8_t&scale,uint8_t&minimum){
    if(index<4){scale=packed[index]&63u;minimum=packed[index+4]&63u;}
    else{
        scale=static_cast<uint8_t>((packed[index+4]&15u)|((packed[index-4]>>6u)<<4u));
        minimum=static_cast<uint8_t>((packed[index+4]>>4u)|((packed[index]>>6u)<<4u));
    }
}
}
Status decode_q4_k_blocks(const uint8_t*source,size_t source_bytes,float*destination,size_t destination_elements){
    constexpr size_t block_bytes=144,block_elements=256;
    if((!source||!destination)&&source_bytes)return {StatusCode::InvalidArgument,"Q4_K decoder received a null pointer"};
    if(source_bytes%block_bytes)return {StatusCode::InvalidArgument,"Q4_K source does not contain complete blocks"};
    const size_t blocks=source_bytes/block_bytes;
    if(blocks>std::numeric_limits<size_t>::max()/block_elements)return {StatusCode::Unsupported,"Q4_K output size overflows"};
    if(destination_elements!=blocks*block_elements)return {StatusCode::InvalidArgument,"Q4_K destination element count mismatch"};
    for(size_t block=0;block<blocks;++block){
        const uint8_t*encoded=source+block*block_bytes;
        const uint16_t d_bits=static_cast<uint16_t>(encoded[0])|static_cast<uint16_t>(encoded[1]<<8u);
        const uint16_t min_bits=static_cast<uint16_t>(encoded[2])|static_cast<uint16_t>(encoded[3]<<8u);
        const float d=decode_f16_value(d_bits),minimum=decode_f16_value(min_bits);
        if(!std::isfinite(d)||!std::isfinite(minimum))return {StatusCode::NumericalFailure,"Q4_K block scales are non-finite"};
        const uint8_t*scales=encoded+4;const uint8_t*qs=encoded+16;float*out=destination+block*block_elements;
        uint32_t is=0;
        for(uint32_t j=0;j<256;j+=64){
            uint8_t sc1=0,m1=0,sc2=0,m2=0;
            decode_k4_scale_min(is,scales,sc1,m1);decode_k4_scale_min(is+1,scales,sc2,m2);
            for(uint32_t l=0;l<32;++l)out[j+l]=d*sc1*(qs[l]&15u)-minimum*m1;
            for(uint32_t l=0;l<32;++l)out[j+l+32]=d*sc2*(qs[l]>>4u)-minimum*m2;
            qs+=32;is+=2;
        }
    }
    return Status::success();
}
Status decode_q5_k_blocks(const uint8_t*source,size_t source_bytes,float*destination,size_t destination_elements){
    constexpr size_t block_bytes=176,block_elements=256;
    if((!source||!destination)&&source_bytes)return {StatusCode::InvalidArgument,"Q5_K decoder received a null pointer"};
    if(source_bytes%block_bytes)return {StatusCode::InvalidArgument,"Q5_K source does not contain complete blocks"};
    const size_t blocks=source_bytes/block_bytes;
    if(blocks>std::numeric_limits<size_t>::max()/block_elements)return {StatusCode::Unsupported,"Q5_K output size overflows"};
    if(destination_elements!=blocks*block_elements)return {StatusCode::InvalidArgument,"Q5_K destination element count mismatch"};
    for(size_t block=0;block<blocks;++block){
        const uint8_t*encoded=source+block*block_bytes;
        const uint16_t d_bits=static_cast<uint16_t>(encoded[0])|static_cast<uint16_t>(encoded[1]<<8u);
        const uint16_t min_bits=static_cast<uint16_t>(encoded[2])|static_cast<uint16_t>(encoded[3]<<8u);
        const float d=decode_f16_value(d_bits),minimum=decode_f16_value(min_bits);
        if(!std::isfinite(d)||!std::isfinite(minimum))return {StatusCode::NumericalFailure,"Q5_K block scales are non-finite"};
        const uint8_t*scales=encoded+4;const uint8_t*qh=encoded+16;const uint8_t*ql=encoded+48;
        float*out=destination+block*block_elements;uint32_t is=0;uint8_t u1=1,u2=2;
        for(uint32_t j=0;j<256;j+=64){
            uint8_t sc1=0,m1=0,sc2=0,m2=0;
            decode_k4_scale_min(is,scales,sc1,m1);decode_k4_scale_min(is+1,scales,sc2,m2);
            for(uint32_t l=0;l<32;++l)out[j+l]=d*sc1*((ql[l]&15u)+((qh[l]&u1)?16u:0u))-minimum*m1;
            for(uint32_t l=0;l<32;++l)out[j+l+32]=d*sc2*((ql[l]>>4u)+((qh[l]&u2)?16u:0u))-minimum*m2;
            ql+=32;is+=2;u1=static_cast<uint8_t>(u1<<2u);u2=static_cast<uint8_t>(u2<<2u);
        }
    }
    return Status::success();
}


namespace {
Status build_k4_matmul_candidate_impl(const Tensor&a,const std::shared_ptr<Buffer>&w,uint64_t bytes,const Tensor&o,ComputeDispatch&d,bool q5){
 d=ComputeDispatch{};const uint64_t block_bytes=q5?176:144;const std::string type=q5?"Q5_K":"Q4_K",suffix=q5?"q5-k":"q4-k";
 if(!w||w->memory_domain()==MemoryDomain::Host||!is_device_contiguous_f32(a)||!is_device_contiguous_f32(o)||
    !a.buffer()||!o.buffer()||a.descriptor().shape.size()!=2||(o.descriptor().shape.size()!=2&&o.descriptor().shape.size()!=3))
    return {StatusCode::Unsupported,type+" MatMul requires F32 activations and packed K-quantized weights"};
 const auto&as=a.descriptor().shape;const auto&os=o.descriptor().shape;
 const uint64_t channels=os.size()==2?os[1]:(os[2]&&os[1]<=UINT64_MAX/os[2]?os[1]*os[2]:0);
 if(!as[0]||!as[1]||as[1]%256||os[0]!=as[0]||!channels||as[0]>UINT32_MAX||as[1]>UINT32_MAX||
    channels>UINT32_MAX||a.element_count()>UINT32_MAX||o.element_count()>UINT32_MAX)
    return {StatusCode::InvalidArgument,type+" MatMul dimensions are invalid or K is not block-aligned"};
 const uint64_t blocks=as[1]/256;if(blocks>UINT64_MAX/block_bytes||channels>UINT64_MAX/(blocks*block_bytes))
    return {StatusCode::InvalidArgument,type+" weight byte size overflows"};
 const uint64_t expected=channels*blocks*block_bytes;if(bytes!=expected||bytes>UINT32_MAX-3ull)
    return {StatusCode::InvalidArgument,type+" weight bytes do not match [N,K]"};
 const uint64_t padded=(bytes+3)&~uint64_t{3};
 if(w->size_bytes()<padded||o.byte_size()!=o.element_count()*sizeof(float)||o.buffer().get()==a.buffer().get()||o.buffer().get()==w.get())
    return {StatusCode::InvalidArgument,type+" MatMul buffer ranges are inconsistent"};
 d.label=suffix+"-matmul-f32-activation-generic-unverified";d.kernel_id="dawn-"+suffix+"-matmul-f32-activation-v1";
 d.backend=BackendKind::DawnWebGpu;d.wgsl_source=q5?kQ5_KMatMulWgsl:kQ4_KMatMulWgsl;d.workgroup_size_x=d.workgroup_size_y=8;
 const std::pair<std::string,std::string> constants[]={
  {"const ROWS:u32=1u;","const ROWS:u32="+std::to_string(as[0])+"u;"},
  {"const OUTPUT_CHANNELS:u32=1u;","const OUTPUT_CHANNELS:u32="+std::to_string(channels)+"u;"},
  {"const INNER:u32=256u;","const INNER:u32="+std::to_string(as[1])+"u;"},
  {q5?"const WEIGHT_ROW_BYTES:u32=176u;":"const WEIGHT_ROW_BYTES:u32=144u;","const WEIGHT_ROW_BYTES:u32="+std::to_string(blocks*block_bytes)+"u;"}
 };
 for(const auto&x:constants){const size_t pos=d.wgsl_source.find(x.first);if(pos==std::string::npos)return {StatusCode::BackendFailure,type+" WGSL placeholder missing"};d.wgsl_source.replace(pos,x.first.size(),x.second);}
 d.workgroups_x=static_cast<uint32_t>((channels+7)/8);d.workgroups_y=static_cast<uint32_t>((as[0]+7)/8);
 d.bindings={{0,a.buffer(),a.byte_offset(),a.byte_size()},{1,w,0,padded},{2,o.buffer(),o.byte_offset(),o.byte_size()}};
 return Status::success();
}
}
Status build_q4_k_matmul_candidate(const Tensor&a,const std::shared_ptr<Buffer>&w,uint64_t b,const Tensor&o,ComputeDispatch&d){return build_k4_matmul_candidate_impl(a,w,b,o,d,false);}
Status build_q5_k_matmul_candidate(const Tensor&a,const std::shared_ptr<Buffer>&w,uint64_t b,const Tensor&o,ComputeDispatch&d){return build_k4_matmul_candidate_impl(a,w,b,o,d,true);}
static Status build_k4_matmul_gles_candidate_impl(const Tensor&a,const std::shared_ptr<Buffer>&w,uint64_t b,const Tensor&o,ComputeDispatch&d,bool q5){
 Status status=build_k4_matmul_candidate_impl(a,w,b,o,d,q5);if(!status.ok())return status;
 const uint64_t channels=o.element_count()/a.descriptor().shape[0],blocks=a.descriptor().shape[1]/256,bb=q5?176:144;
 const std::string suffix=q5?"q5-k":"q4-k";d.label=suffix+"-matmul-f32-activation-gles-generic-unverified";
 d.kernel_id="gles-"+suffix+"-matmul-f32-activation-v1";d.backend=BackendKind::OpenGles;d.wgsl_source.clear();d.glsl_source=q5?kQ5_KMatMulGlsl:kQ4_KMatMulGlsl;
 const std::pair<std::string,std::string> constants[]={
  {"const uint ROWS=1u;","const uint ROWS="+std::to_string(a.descriptor().shape[0])+"u;"},
  {"const uint OUTPUT_CHANNELS=1u;","const uint OUTPUT_CHANNELS="+std::to_string(channels)+"u;"},
  {"const uint INNER=256u;","const uint INNER="+std::to_string(a.descriptor().shape[1])+"u;"},
  {q5?"const uint WEIGHT_ROW_BYTES=176u;":"const uint WEIGHT_ROW_BYTES=144u;","const uint WEIGHT_ROW_BYTES="+std::to_string(blocks*bb)+"u;"}
 };
 for(const auto&x:constants){const size_t pos=d.glsl_source.find(x.first);if(pos==std::string::npos)return {StatusCode::BackendFailure,"K4 GLES GLSL placeholder missing"};d.glsl_source.replace(pos,x.first.size(),x.second);}
 return Status::success();
}
Status build_q4_k_matmul_gles_candidate(const Tensor&a,const std::shared_ptr<Buffer>&w,uint64_t b,const Tensor&o,ComputeDispatch&d){return build_k4_matmul_gles_candidate_impl(a,w,b,o,d,false);}
Status build_q5_k_matmul_gles_candidate(const Tensor&a,const std::shared_ptr<Buffer>&w,uint64_t b,const Tensor&o,ComputeDispatch&d){return build_k4_matmul_gles_candidate_impl(a,w,b,o,d,true);}
Status reference_q4_k_matmul(const float*a,const uint8_t*w,uint64_t bytes,float*out,uint32_t rows,uint32_t channels,uint32_t inner){
 if((!a||!w||!out)&&rows&&channels&&inner)return {StatusCode::InvalidArgument,"Q4_K MatMul reference received null pointer"};
 if(!rows||!channels||!inner||inner%256)return {StatusCode::InvalidArgument,"Q4_K MatMul dimensions must be block-aligned"};
 const uint64_t blocks=inner/256;if(blocks>UINT64_MAX/144||channels>UINT64_MAX/(blocks*144)||bytes!=static_cast<uint64_t>(channels)*blocks*144)return {StatusCode::InvalidArgument,"Q4_K MatMul weight bytes mismatch"};
 std::vector<float>decoded(inner);
 for(uint32_t ch=0;ch<channels;++ch){Status st=decode_q4_k_blocks(w+static_cast<uint64_t>(ch)*blocks*144,blocks*144,decoded.data(),inner);if(!st.ok())return st;
  for(uint32_t row=0;row<rows;++row){float sum=0;for(uint32_t k=0;k<inner;++k)sum+=a[static_cast<uint64_t>(row)*inner+k]*decoded[k];out[static_cast<uint64_t>(row)*channels+ch]=sum;}}
 return Status::success();
}
Status reference_q5_k_matmul(const float*a,const uint8_t*w,uint64_t bytes,float*out,uint32_t rows,uint32_t channels,uint32_t inner){
 if((!a||!w||!out)&&rows&&channels&&inner)return {StatusCode::InvalidArgument,"Q5_K MatMul reference received null pointer"};
 if(!rows||!channels||!inner||inner%256)return {StatusCode::InvalidArgument,"Q5_K MatMul dimensions must be block-aligned"};
 const uint64_t blocks=inner/256;if(blocks>UINT64_MAX/176||channels>UINT64_MAX/(blocks*176)||bytes!=static_cast<uint64_t>(channels)*blocks*176)return {StatusCode::InvalidArgument,"Q5_K MatMul weight bytes mismatch"};
 std::vector<float>decoded(inner);
 for(uint32_t ch=0;ch<channels;++ch){Status st=decode_q5_k_blocks(w+static_cast<uint64_t>(ch)*blocks*176,blocks*176,decoded.data(),inner);if(!st.ok())return st;
  for(uint32_t row=0;row<rows;++row){float sum=0;for(uint32_t k=0;k<inner;++k)sum+=a[static_cast<uint64_t>(row)*inner+k]*decoded[k];out[static_cast<uint64_t>(row)*channels+ch]=sum;}}
 return Status::success();
}


Status validate_k4_matmul_kernel_impl(Backend&backend,OperatorRegistry&operators,QuantizationRegistry&quantizations,bool q5){
    const BackendKind kind=backend.kind();if((kind!=BackendKind::DawnWebGpu&&kind!=BackendKind::OpenGles)||!backend.healthy())
        return {StatusCode::NotReady,"K-quantized MatMul validation requires a healthy Dawn or GLES backend"};
    const uint32_t type_id=q5?13u:12u;auto format=quantizations.formats_.find(type_id);
    if(format==quantizations.formats_.end()||format->second.decoder_state!=SupportState::Validated)
        return {StatusCode::NotReady,"K-quantized host decoder is not validated"};
    const std::string suffix=q5?"q5-k":"q4-k";
    const std::string kernel_id=(kind==BackendKind::OpenGles?"gles-":"dawn-")+suffix+"-matmul-f32-activation-v1";
    auto kernel=std::find_if(operators.kernels_.begin(),operators.kernels_.end(),[&](const KernelDefinition&k){return k.kernel_id==kernel_id&&k.backend==kind;});
    if(kernel==operators.kernels_.end())return {StatusCode::NotReady,"K-quantized MatMul candidate is not registered"};
    if(kernel->validation==KernelValidation::Validated){format->second.backend_gpu_kernel_states[kind]=SupportState::Validated;format->second.gpu_kernel_state=SupportState::Validated;return Status::success();}
    if(kernel->validation!=KernelValidation::Unverified)return {StatusCode::Unsupported,"K-quantized MatMul candidate is rejected"};
    const bool gles=kind==BackendKind::OpenGles;
    const std::string&registered=gles?kernel->glsl_source:kernel->wgsl_source;
    const std::string canonical=q5?(gles?std::string(kQ5_KMatMulGlsl):std::string(kQ5_KMatMulWgsl))
                                  :(gles?std::string(kQ4_KMatMulGlsl):std::string(kQ4_KMatMulWgsl));
    if(registered.empty()||registered!=canonical)return {StatusCode::InvalidArgument,"K-quantized MatMul shader differs from canonical source"};
    constexpr uint32_t rows=2,channels=2,inner=256;
    const uint32_t block_bytes=q5?176u:144u;const uint64_t bytes=static_cast<uint64_t>(channels)*block_bytes;
    std::vector<uint8_t>encoded(static_cast<size_t>(bytes),0);
    for(uint32_t ch=0;ch<channels;++ch){
        uint8_t*block=encoded.data()+ch*block_bytes;block[0]=0;block[1]=0x3c;
        std::fill(block+4,block+8,1);std::fill(block+8,block+12,1);
        if(q5)std::fill(block+16,block+48,0xff);
        std::fill(block+(q5?48:16),block+block_bytes,0x11);
    }
    std::vector<float>activations(rows*inner);
    for(size_t i=0;i<activations.size();++i)activations[i]=(static_cast<int>(i%13)-6)*0.125f;
    float expected[rows*channels]={},actual[rows*channels]={};
    Status status=q5?reference_q5_k_matmul(activations.data(),encoded.data(),bytes,expected,rows,channels,inner)
                    :reference_q4_k_matmul(activations.data(),encoded.data(),bytes,expected,rows,channels,inner);
    if(!status.ok())return status;
    BufferDesc ad{activations.size()*sizeof(float),alignof(float),MemoryDomain::Device,false,"k4-validation-activations"};
    BufferDesc wd{bytes,4,MemoryDomain::Device,true,"k4-validation-weights"};
    BufferDesc od{sizeof(actual),alignof(float),MemoryDomain::Device,false,"k4-validation-output"};
    std::shared_ptr<Buffer>ab,wb,ob;
    status=backend.allocate(ad,ab);if(!status.ok())return status;
    status=backend.allocate(wd,wb);if(!status.ok())return status;
    status=backend.allocate(od,ob);if(!status.ok())return status;
    status=backend.upload(ab,0,activations.data(),ad.size_bytes);if(!status.ok())return status;
    status=backend.upload(wb,0,encoded.data(),bytes);if(!status.ok())return status;
    TensorDesc atd;atd.shape={rows,inner};atd.data_type=DataType::F32;
    TensorDesc otd;otd.shape={rows,channels};otd.data_type=DataType::F32;
    Tensor at,ot;status=Tensor::create(atd,ab,0,at);if(!status.ok())return status;
    status=Tensor::create(otd,ob,0,ot);if(!status.ok())return status;
    ComputeDispatch dispatch;
    if(q5)status=gles?build_q5_k_matmul_gles_candidate(at,wb,bytes,ot,dispatch):build_q5_k_matmul_candidate(at,wb,bytes,ot,dispatch);
    else status=gles?build_q4_k_matmul_gles_candidate(at,wb,bytes,ot,dispatch):build_q4_k_matmul_candidate(at,wb,bytes,ot,dispatch);
    if(!status.ok())return status;
    if(dispatch.kernel_id!=kernel_id||dispatch.backend!=kind)return {StatusCode::InvalidArgument,"K-quantized MatMul candidate identity mismatch"};
    dispatch.validation=KernelValidation::Validated;
    std::unique_ptr<CommandBatch>commands;status=backend.begin_commands(commands);if(!status.ok())return status;
    if(!commands)return {StatusCode::BackendFailure,"K-quantized backend returned empty command batch"};
    status=commands->dispatch_compute(dispatch);if(!status.ok())return status;
    status=backend.submit(std::move(commands));if(!status.ok())return status;
    status=backend.synchronize();if(!status.ok())return status;
    status=backend.readback(ob,0,actual,sizeof(actual));if(!status.ok())return status;
    for(size_t i=0;i<rows*channels;++i){const float tolerance=1.0e-4f*std::max(1.0f,std::fabs(expected[i]));
        if(!std::isfinite(actual[i])||std::fabs(actual[i]-expected[i])>tolerance)
            return {StatusCode::NumericalFailure,"K-quantized MatMul GPU output differs from independent decoder"};}
    kernel->validation=KernelValidation::Validated;format->second.backend_gpu_kernel_states[kind]=SupportState::Validated;
    format->second.gpu_kernel_state=SupportState::Validated;return Status::success();
}
Status validate_q4_k_matmul_kernel(Backend&b,OperatorRegistry&o,QuantizationRegistry&q){return validate_k4_matmul_kernel_impl(b,o,q,false);}
Status validate_q5_k_matmul_kernel(Backend&b,OperatorRegistry&o,QuantizationRegistry&q){return validate_k4_matmul_kernel_impl(b,o,q,true);}


Status decode_q2_k_blocks(const uint8_t*source,size_t source_bytes,float*destination,size_t destination_elements){
    constexpr size_t block_bytes=84,block_elements=256;
    if((!source||!destination)&&source_bytes)return {StatusCode::InvalidArgument,"Q2_K decoder received a null pointer"};
    if(source_bytes%block_bytes)return {StatusCode::InvalidArgument,"Q2_K source does not contain complete blocks"};
    const size_t blocks=source_bytes/block_bytes;
    if(blocks>std::numeric_limits<size_t>::max()/block_elements)return {StatusCode::Unsupported,"Q2_K output size overflows"};
    if(destination_elements!=blocks*block_elements)return {StatusCode::InvalidArgument,"Q2_K destination element count mismatch"};
    for(size_t block=0;block<blocks;++block){
        const uint8_t*encoded=source+block*block_bytes;
        // GGML block_q2_K stores 16 scale/min bytes, 64 quant bytes,
        // then the two FP16 super-block multipliers (d and dmin).
        const uint16_t d_bits=static_cast<uint16_t>(encoded[80])|static_cast<uint16_t>(encoded[81]<<8u);
        const uint16_t min_bits=static_cast<uint16_t>(encoded[82])|static_cast<uint16_t>(encoded[83]<<8u);
        const float d=decode_f16_value(d_bits),minimum=decode_f16_value(min_bits);
        if(!std::isfinite(d)||!std::isfinite(minimum))return {StatusCode::NumericalFailure,"Q2_K block scales are non-finite"};
        const uint8_t*scales=encoded;const uint8_t*qs=encoded+16;float*out=destination+block*block_elements;
        size_t scale_index=0,output_index=0;
        for(size_t n=0;n<256;n+=128){
            uint32_t shift=0;
            for(size_t group=0;group<4;++group){
                for(size_t side=0;side<2;++side){
                    const uint8_t packed_scale=scales[scale_index++];
                    const float dl=d*static_cast<float>(packed_scale&15u);
                    const float ml=minimum*static_cast<float>(packed_scale>>4u);
                    const uint8_t*q=qs+side*16;
                    for(size_t lane=0;lane<16;++lane)out[output_index++]=dl*static_cast<float>((q[lane]>>shift)&3u)-ml;
                }
                shift+=2;
            }
            qs+=32;
        }
    }
    return Status::success();
}
Status decode_q3_k_blocks(const uint8_t*source,size_t source_bytes,float*destination,size_t destination_elements){
    constexpr size_t block_bytes=110,block_elements=256;
    if((!source||!destination)&&source_bytes)return {StatusCode::InvalidArgument,"Q3_K decoder received a null pointer"};
    if(source_bytes%block_bytes)return {StatusCode::InvalidArgument,"Q3_K source does not contain complete blocks"};
    const size_t blocks=source_bytes/block_bytes;
    if(blocks>std::numeric_limits<size_t>::max()/block_elements)return {StatusCode::Unsupported,"Q3_K output size overflows"};
    if(destination_elements!=blocks*block_elements)return {StatusCode::InvalidArgument,"Q3_K destination element count mismatch"};
    for(size_t block=0;block<blocks;++block){
        const uint8_t*encoded=source+block*block_bytes;
        const uint8_t*hm=encoded;const uint8_t*q=encoded+32;const uint8_t*packed_scales=encoded+96;
        uint32_t aux0=0,aux1=0,aux2=0;
        std::memcpy(&aux0,packed_scales,4);std::memcpy(&aux1,packed_scales+4,4);std::memcpy(&aux2,packed_scales+8,4);
        constexpr uint32_t kmask1=0x03030303u,kmask2=0x0f0f0f0fu;
        const uint32_t tmp=aux2;
        const uint32_t unpacked[4]={
            (aux0&kmask2)|(((tmp>>0u)&kmask1)<<4u),
            (aux1&kmask2)|(((tmp>>2u)&kmask1)<<4u),
            ((aux0>>4u)&kmask2)|(((tmp>>4u)&kmask1)<<4u),
            ((aux1>>4u)&kmask2)|(((tmp>>6u)&kmask1)<<4u)
        };
        int8_t scales[16];
        for(size_t index=0;index<16;++index){
            const uint32_t word=unpacked[index/4];
            const uint8_t value=static_cast<uint8_t>((word>>((index%4)*8u))&255u);
            scales[index]=static_cast<int8_t>(value);
        }
        const uint16_t d_bits=static_cast<uint16_t>(encoded[108])|static_cast<uint16_t>(encoded[109]<<8u);
        const float d=decode_f16_value(d_bits);
        if(!std::isfinite(d))return {StatusCode::NumericalFailure,"Q3_K block scale is non-finite"};
        float*out=destination+block*block_elements;size_t output_index=0,scale_index=0;
        uint8_t mask=1;
        for(size_t n=0;n<256;n+=128){
            uint32_t shift=0;
            for(size_t group=0;group<4;++group){
                for(size_t side=0;side<2;++side){
                    const float dl=d*static_cast<float>(static_cast<int>(scales[scale_index++])-32);
                    const uint8_t*qq=q+side*16;const uint8_t*hh=hm+side*16;
                    for(size_t lane=0;lane<16;++lane){
                        const int low=static_cast<int>((qq[lane]>>shift)&3u);
                        const int quant=low-((hh[lane]&mask)?0:4);
                        out[output_index++]=dl*static_cast<float>(quant);
                    }
                }
                shift+=2;mask=static_cast<uint8_t>(mask<<1u);
            }
            q+=32;
        }
    }
    return Status::success();
}


namespace {
Status build_k23_matmul_candidate_impl(const Tensor&a,const std::shared_ptr<Buffer>&w,uint64_t bytes,const Tensor&o,ComputeDispatch&d,bool q3){
 d=ComputeDispatch{};const uint64_t block_bytes=q3?110:84;const std::string type=q3?"Q3_K":"Q2_K",suffix=q3?"q3-k":"q2-k";
 if(!w||w->memory_domain()==MemoryDomain::Host||!is_device_contiguous_f32(a)||!is_device_contiguous_f32(o)||
    !a.buffer()||!o.buffer()||a.descriptor().shape.size()!=2||(o.descriptor().shape.size()!=2&&o.descriptor().shape.size()!=3))
    return {StatusCode::Unsupported,type+" MatMul requires F32 activations and packed K-quantized weights"};
 const auto&as=a.descriptor().shape;const auto&os=o.descriptor().shape;
 const uint64_t channels=os.size()==2?os[1]:(os[2]&&os[1]<=UINT64_MAX/os[2]?os[1]*os[2]:0);
 if(!as[0]||!as[1]||as[1]%256||os[0]!=as[0]||!channels||as[0]>UINT32_MAX||as[1]>UINT32_MAX||
    channels>UINT32_MAX||a.element_count()>UINT32_MAX||o.element_count()>UINT32_MAX)
    return {StatusCode::InvalidArgument,type+" MatMul dimensions are invalid or K is not block-aligned"};
 const uint64_t blocks=as[1]/256;if(blocks>UINT64_MAX/block_bytes||channels>UINT64_MAX/(blocks*block_bytes))
    return {StatusCode::InvalidArgument,type+" weight byte size overflows"};
 const uint64_t expected=channels*blocks*block_bytes;if(bytes!=expected||bytes>UINT32_MAX-3ull)
    return {StatusCode::InvalidArgument,type+" weight bytes do not match [N,K]"};
 const uint64_t padded=(bytes+3)&~uint64_t{3};
 if(w->size_bytes()<padded||o.byte_size()!=o.element_count()*sizeof(float)||o.buffer().get()==a.buffer().get()||o.buffer().get()==w.get())
    return {StatusCode::InvalidArgument,type+" MatMul buffer ranges are inconsistent"};
 d.label=suffix+"-matmul-f32-activation-generic-unverified";d.kernel_id="dawn-"+suffix+"-matmul-f32-activation-v1";
 d.backend=BackendKind::DawnWebGpu;d.wgsl_source=q3?kQ3_KMatMulWgsl:kQ2_KMatMulWgsl;d.workgroup_size_x=d.workgroup_size_y=8;
 const std::pair<std::string,std::string> constants[]={
  {"const ROWS:u32=1u;","const ROWS:u32="+std::to_string(as[0])+"u;"},
  {"const OUTPUT_CHANNELS:u32=1u;","const OUTPUT_CHANNELS:u32="+std::to_string(channels)+"u;"},
  {"const INNER:u32=256u;","const INNER:u32="+std::to_string(as[1])+"u;"},
  {q3?"const WEIGHT_ROW_BYTES:u32=110u;":"const WEIGHT_ROW_BYTES:u32=84u;","const WEIGHT_ROW_BYTES:u32="+std::to_string(blocks*block_bytes)+"u;"}
 };
 for(const auto&x:constants){const size_t pos=d.wgsl_source.find(x.first);if(pos==std::string::npos)return {StatusCode::BackendFailure,type+" WGSL placeholder missing"};d.wgsl_source.replace(pos,x.first.size(),x.second);}
 d.workgroups_x=static_cast<uint32_t>((channels+7)/8);d.workgroups_y=static_cast<uint32_t>((as[0]+7)/8);
 d.bindings={{0,a.buffer(),a.byte_offset(),a.byte_size()},{1,w,0,padded},{2,o.buffer(),o.byte_offset(),o.byte_size()}};
 return Status::success();
}
}
Status build_q2_k_matmul_candidate(const Tensor&a,const std::shared_ptr<Buffer>&w,uint64_t b,const Tensor&o,ComputeDispatch&d){return build_k23_matmul_candidate_impl(a,w,b,o,d,false);}
Status build_q3_k_matmul_candidate(const Tensor&a,const std::shared_ptr<Buffer>&w,uint64_t b,const Tensor&o,ComputeDispatch&d){return build_k23_matmul_candidate_impl(a,w,b,o,d,true);}
static Status build_k23_matmul_gles_candidate_impl(const Tensor&a,const std::shared_ptr<Buffer>&w,uint64_t b,const Tensor&o,ComputeDispatch&d,bool q3){
 Status status=build_k23_matmul_candidate_impl(a,w,b,o,d,q3);if(!status.ok())return status;
 const uint64_t channels=o.element_count()/a.descriptor().shape[0],blocks=a.descriptor().shape[1]/256,bb=q3?110:84;
 const std::string suffix=q3?"q3-k":"q2-k";d.label=suffix+"-matmul-f32-activation-gles-generic-unverified";
 d.kernel_id="gles-"+suffix+"-matmul-f32-activation-v1";d.backend=BackendKind::OpenGles;d.wgsl_source.clear();d.glsl_source=q3?kQ3_KMatMulGlsl:kQ2_KMatMulGlsl;
 const std::pair<std::string,std::string> constants[]={
  {"const uint ROWS=1u;","const uint ROWS="+std::to_string(a.descriptor().shape[0])+"u;"},
  {"const uint OUTPUT_CHANNELS=1u;","const uint OUTPUT_CHANNELS="+std::to_string(channels)+"u;"},
  {"const uint INNER=256u;","const uint INNER="+std::to_string(a.descriptor().shape[1])+"u;"},
  {q3?"const uint WEIGHT_ROW_BYTES=110u;":"const uint WEIGHT_ROW_BYTES=84u;","const uint WEIGHT_ROW_BYTES="+std::to_string(blocks*bb)+"u;"}
 };
 for(const auto&x:constants){const size_t pos=d.glsl_source.find(x.first);if(pos==std::string::npos)return {StatusCode::BackendFailure,"K2/K3 GLES GLSL placeholder missing"};d.glsl_source.replace(pos,x.first.size(),x.second);}
 return Status::success();
}
Status build_q2_k_matmul_gles_candidate(const Tensor&a,const std::shared_ptr<Buffer>&w,uint64_t b,const Tensor&o,ComputeDispatch&d){return build_k23_matmul_gles_candidate_impl(a,w,b,o,d,false);}
Status build_q3_k_matmul_gles_candidate(const Tensor&a,const std::shared_ptr<Buffer>&w,uint64_t b,const Tensor&o,ComputeDispatch&d){return build_k23_matmul_gles_candidate_impl(a,w,b,o,d,true);}
Status reference_q2_k_matmul(const float*a,const uint8_t*w,uint64_t bytes,float*out,uint32_t rows,uint32_t channels,uint32_t inner){
 if((!a||!w||!out)&&rows&&channels&&inner)return {StatusCode::InvalidArgument,"Q2_K MatMul reference received null pointer"};
 if(!rows||!channels||!inner||inner%256)return {StatusCode::InvalidArgument,"Q2_K MatMul dimensions must be block-aligned"};
 const uint64_t blocks=inner/256;if(blocks>UINT64_MAX/84||channels>UINT64_MAX/(blocks*84)||bytes!=static_cast<uint64_t>(channels)*blocks*84)return {StatusCode::InvalidArgument,"Q2_K MatMul weight bytes mismatch"};
 std::vector<float>decoded(inner);
 for(uint32_t ch=0;ch<channels;++ch){Status st=decode_q2_k_blocks(w+static_cast<uint64_t>(ch)*blocks*84,blocks*84,decoded.data(),inner);if(!st.ok())return st;
  for(uint32_t row=0;row<rows;++row){float sum=0;for(uint32_t k=0;k<inner;++k)sum+=a[static_cast<uint64_t>(row)*inner+k]*decoded[k];out[static_cast<uint64_t>(row)*channels+ch]=sum;}}
 return Status::success();
}
Status reference_q3_k_matmul(const float*a,const uint8_t*w,uint64_t bytes,float*out,uint32_t rows,uint32_t channels,uint32_t inner){
 if((!a||!w||!out)&&rows&&channels&&inner)return {StatusCode::InvalidArgument,"Q3_K MatMul reference received null pointer"};
 if(!rows||!channels||!inner||inner%256)return {StatusCode::InvalidArgument,"Q3_K MatMul dimensions must be block-aligned"};
 const uint64_t blocks=inner/256;if(blocks>UINT64_MAX/110||channels>UINT64_MAX/(blocks*110)||bytes!=static_cast<uint64_t>(channels)*blocks*110)return {StatusCode::InvalidArgument,"Q3_K MatMul weight bytes mismatch"};
 std::vector<float>decoded(inner);
 for(uint32_t ch=0;ch<channels;++ch){Status st=decode_q3_k_blocks(w+static_cast<uint64_t>(ch)*blocks*110,blocks*110,decoded.data(),inner);if(!st.ok())return st;
  for(uint32_t row=0;row<rows;++row){float sum=0;for(uint32_t k=0;k<inner;++k)sum+=a[static_cast<uint64_t>(row)*inner+k]*decoded[k];out[static_cast<uint64_t>(row)*channels+ch]=sum;}}
 return Status::success();
}


Status validate_k23_matmul_kernel_impl(Backend&backend,OperatorRegistry&operators,QuantizationRegistry&quantizations,bool q3){
    const BackendKind kind=backend.kind();if((kind!=BackendKind::DawnWebGpu&&kind!=BackendKind::OpenGles)||!backend.healthy())
        return {StatusCode::NotReady,"K2/K3 MatMul validation requires a healthy Dawn or GLES backend"};
    const uint32_t type_id=q3?11u:10u;auto format=quantizations.formats_.find(type_id);
    if(format==quantizations.formats_.end()||format->second.decoder_state!=SupportState::Validated)
        return {StatusCode::NotReady,"K2/K3 host decoder is not validated"};
    const std::string suffix=q3?"q3-k":"q2-k";
    const std::string kernel_id=(kind==BackendKind::OpenGles?"gles-":"dawn-")+suffix+"-matmul-f32-activation-v1";
    auto kernel=std::find_if(operators.kernels_.begin(),operators.kernels_.end(),[&](const KernelDefinition&k){return k.kernel_id==kernel_id&&k.backend==kind;});
    if(kernel==operators.kernels_.end())return {StatusCode::NotReady,"K2/K3 MatMul candidate is not registered"};
    if(kernel->validation==KernelValidation::Validated){format->second.backend_gpu_kernel_states[kind]=SupportState::Validated;format->second.gpu_kernel_state=SupportState::Validated;return Status::success();}
    if(kernel->validation!=KernelValidation::Unverified)return {StatusCode::Unsupported,"K2/K3 MatMul candidate is rejected"};
    const bool gles=kind==BackendKind::OpenGles;
    const std::string&registered=gles?kernel->glsl_source:kernel->wgsl_source;
    const std::string canonical=q3?(gles?std::string(kQ3_KMatMulGlsl):std::string(kQ3_KMatMulWgsl))
                                  :(gles?std::string(kQ2_KMatMulGlsl):std::string(kQ2_KMatMulWgsl));
    if(registered.empty()||registered!=canonical)return {StatusCode::InvalidArgument,"K2/K3 MatMul shader differs from canonical source"};
    constexpr uint32_t rows=2,channels=2,inner=256;
    const uint32_t block_bytes=q3?110u:84u;const uint64_t bytes=static_cast<uint64_t>(channels)*block_bytes;
    std::vector<uint8_t>encoded(static_cast<size_t>(bytes),0);
    for(uint32_t ch=0;ch<channels;++ch){
        uint8_t*block=encoded.data()+ch*block_bytes;
        if(q3){
            std::fill(block,block+32,0xff);std::fill(block+32,block+96,0x55);
            std::fill(block+96,block+104,0x11);std::fill(block+104,block+108,0xaa);
            block[108]=0;block[109]=0x3c;
        }else{
            std::fill(block,block+16,0x01);std::fill(block+16,block+80,0x55);
            block[80]=0;block[81]=0x3c; // d = 1.0; dmin remains zero
        }
    }
    std::vector<float>activations(rows*inner);
    for(size_t i=0;i<activations.size();++i)activations[i]=(static_cast<int>(i%9)-4)*0.125f;
    float expected[rows*channels]={},actual[rows*channels]={};
    Status status=q3?reference_q3_k_matmul(activations.data(),encoded.data(),bytes,expected,rows,channels,inner)
                    :reference_q2_k_matmul(activations.data(),encoded.data(),bytes,expected,rows,channels,inner);
    if(!status.ok())return status;
    BufferDesc ad{activations.size()*sizeof(float),alignof(float),MemoryDomain::Device,false,"k23-validation-activations"};
    BufferDesc wd{bytes,4,MemoryDomain::Device,true,"k23-validation-weights"};
    BufferDesc od{sizeof(actual),alignof(float),MemoryDomain::Device,false,"k23-validation-output"};
    std::shared_ptr<Buffer>ab,wb,ob;
    status=backend.allocate(ad,ab);if(!status.ok())return status;status=backend.allocate(wd,wb);if(!status.ok())return status;
    status=backend.allocate(od,ob);if(!status.ok())return status;
    status=backend.upload(ab,0,activations.data(),ad.size_bytes);if(!status.ok())return status;
    status=backend.upload(wb,0,encoded.data(),bytes);if(!status.ok())return status;
    TensorDesc atd;atd.shape={rows,inner};atd.data_type=DataType::F32;
    TensorDesc otd;otd.shape={rows,channels};otd.data_type=DataType::F32;
    Tensor at,ot;status=Tensor::create(atd,ab,0,at);if(!status.ok())return status;status=Tensor::create(otd,ob,0,ot);if(!status.ok())return status;
    ComputeDispatch dispatch;
    if(q3)status=gles?build_q3_k_matmul_gles_candidate(at,wb,bytes,ot,dispatch):build_q3_k_matmul_candidate(at,wb,bytes,ot,dispatch);
    else status=gles?build_q2_k_matmul_gles_candidate(at,wb,bytes,ot,dispatch):build_q2_k_matmul_candidate(at,wb,bytes,ot,dispatch);
    if(!status.ok())return status;
    if(dispatch.kernel_id!=kernel_id||dispatch.backend!=kind)return {StatusCode::InvalidArgument,"K2/K3 MatMul candidate identity mismatch"};
    dispatch.validation=KernelValidation::Validated;
    std::unique_ptr<CommandBatch>commands;status=backend.begin_commands(commands);if(!status.ok())return status;
    if(!commands)return {StatusCode::BackendFailure,"K2/K3 backend returned empty command batch"};
    status=commands->dispatch_compute(dispatch);if(!status.ok())return status;status=backend.submit(std::move(commands));if(!status.ok())return status;
    status=backend.synchronize();if(!status.ok())return status;status=backend.readback(ob,0,actual,sizeof(actual));if(!status.ok())return status;
    for(size_t i=0;i<rows*channels;++i){const float tolerance=1.0e-4f*std::max(1.0f,std::fabs(expected[i]));
        if(!std::isfinite(actual[i])||std::fabs(actual[i]-expected[i])>tolerance)return {StatusCode::NumericalFailure,"K2/K3 MatMul GPU output differs from independent decoder"};}
    kernel->validation=KernelValidation::Validated;format->second.backend_gpu_kernel_states[kind]=SupportState::Validated;
    format->second.gpu_kernel_state=SupportState::Validated;return Status::success();
}
Status validate_q2_k_matmul_kernel(Backend&b,OperatorRegistry&o,QuantizationRegistry&q){return validate_k23_matmul_kernel_impl(b,o,q,false);}
Status validate_q3_k_matmul_kernel(Backend&b,OperatorRegistry&o,QuantizationRegistry&q){return validate_k23_matmul_kernel_impl(b,o,q,true);}

} // namespace ai_chat::gpu_engine
