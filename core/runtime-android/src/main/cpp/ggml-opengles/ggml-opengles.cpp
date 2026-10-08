#include "ggml-opengles.h"
#include "ggml-backend-impl.h"
#include "ggml.h"
#include "../opengles_runtime.h"

#include <EGL/egl.h>
#include <GLES3/gl31.h>

#include <sys/mman.h>
#include <unistd.h>
#include <cstring>
#include <cmath>
#include <cstdio>
#include <string>

namespace {

static constexpr size_t kMaxSsboBytes = 128u * 1024u * 1024u - 256u;
static const char * kName = "OpenGL ES";
static const char * kDescription = "Android OpenGL ES 3.1 compute backend";

struct BufferContext {
    GLuint buffer = 0;
    void * virtual_base = MAP_FAILED;
    size_t size = 0;
};

struct DeviceContext {};

static ggml_backend_buffer_type ggml_buft;
static ggml_backend_device ggml_device;
static bool g_ready = false;

static GLuint compile_compute(const char * source) {
    GLuint shader = glCreateShader(GL_COMPUTE_SHADER);
    glShaderSource(shader, 1, &source, nullptr);
    glCompileShader(shader);
    GLint ok = GL_FALSE;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        char log[4096] = {};
        glGetShaderInfoLog(shader, sizeof(log), nullptr, log);
        std::fprintf(stderr, "ggml-opengles shader compile failed: %s\n", log);
        glDeleteShader(shader);
        return 0;
    }
    GLuint program = glCreateProgram();
    glAttachShader(program, shader);
    glLinkProgram(program);
    glDeleteShader(shader);
    glGetProgramiv(program, GL_LINK_STATUS, &ok);
    if (!ok) {
        char log[4096] = {};
        glGetProgramInfoLog(program, sizeof(log), nullptr, log);
        std::fprintf(stderr, "ggml-opengles program link failed: %s\n", log);
        glDeleteProgram(program);
        return 0;
    }
    return program;
}

static const char * elementwise_shader() {
    return R"(#version 310 es
layout(local_size_x = 64) in;
layout(std430, binding = 0) readonly buffer A { float a[]; };
layout(std430, binding = 1) readonly buffer B { float b[]; };
layout(std430, binding = 2) writeonly buffer C { float c[]; };
uniform uint a_off;
uniform uint b_off;
uniform uint c_off;
uniform uint n;
uniform uint op;
void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i >= n) return;
    uint ai = (a_off >> 2u) + i;
    uint bi = (b_off >> 2u) + i;
    uint ci = (c_off >> 2u) + i;
    float x = a[ai];
    float y = b[bi];
    c[ci] = op == 0u ? x + y : x * y;
})";
}

static const char * q6k_matmul_shader() {
    return R"(#version 310 es
layout(local_size_x = 64) in;
layout(std430, binding = 0) readonly buffer W { uint w[]; };
layout(std430, binding = 1) readonly buffer X { float x[]; };
layout(std430, binding = 2) writeonly buffer Y { float y[]; };

uniform uint w_off;
uniform uint x_off;
uniform uint y_off;
uniform uint k;
uniform uint rows;
uniform uint cols;

uint byte_u8(uint byte_offset) {
    uint word = w[(w_off + byte_offset) >> 2u];
    uint shift = (byte_offset & 3u) * 8u;
    return (word >> shift) & 255u;
}

int byte_i8(uint byte_offset) {
    uint v = byte_u8(byte_offset);
    return v >= 128u ? int(v) - 256 : int(v);
}

float half_at(uint byte_offset) {
    uint word = w[(w_off + byte_offset) >> 2u];
    uint bits = word & 65535u;
    uint lo = bits & 1023u;
    uint exp = (bits >> 10u) & 31u;
    uint sign = bits >> 15u;
    if (exp == 0u) {
        if (lo == 0u) return sign != 0u ? -0.0 : 0.0;
        return (sign != 0u ? -1.0 : 1.0) * exp2(-14.0) * (float(lo) / 1024.0);
    }
    if (exp == 31u) return sign != 0u ? -1.0/0.0 : 1.0/0.0;
    return (sign != 0u ? -1.0 : 1.0) * exp2(float(exp) - 15.0) * (1.0 + float(lo) / 1024.0);
}

float q6(uint block, uint idx) {
    uint base = block * 210u;
    float d = half_at(base);
    uint group = idx / 16u;
    int sc = byte_i8(base + 194u + group);
    uint n128 = idx / 128u;
    uint l = idx & 31u;
    uint qbase = base + n128 * 64u;
    uint hbase = base + 128u + n128 * 32u;
    uint q1 = byte_u8(qbase + l) & 15u;
    uint q2 = byte_u8(qbase + 32u + l) & 15u;
    uint q3 = (byte_u8(qbase + l) >> 4u) & 15u;
    uint q4 = (byte_u8(qbase + 32u + l) >> 4u) & 15u;
    uint h = byte_u8(hbase + l);
    uint q;
    if ((idx & 127u) < 32u) q = q1 | ((h & 3u) << 4u);
    else if ((idx & 127u) < 64u) q = q2 | (((h >> 2u) & 3u) << 4u);
    else if ((idx & 127u) < 96u) q = q3 | (((h >> 4u) & 3u) << 4u);
    else q = q4 | (((h >> 6u) & 3u) << 4u);
    return d * float(sc) * (float(q) - 32.0);
}

void main() {
    uint out = gl_GlobalInvocationID.x;
    uint total = rows * cols;
    if (out >= total) return;
    uint row = out % rows;
    uint col = out / rows;
    float sum = 0.0;
    uint blocks = k / 256u;
    for (uint b = 0u; b < blocks; ++b) {
        for (uint j = 0u; j < 256u; ++j) {
            uint kk = b * 256u + j;
            float xv = x[(x_off >> 2u) + col * k + kk];
            sum += q6(b + row * blocks, j) * xv;
        }
    }
    y[(y_off >> 2u) + col * rows + row] = sum;
})";
}

static const char * silu_shader() {
    return R"(#version 310 es
layout(local_size_x = 64) in;
layout(std430, binding = 0) readonly buffer A { float a[]; };
layout(std430, binding = 1) writeonly buffer C { float c[]; };
uniform uint a_off, c_off, n;
void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i >= n) return;
    float x = a[(a_off >> 2u) + i];
    c[(c_off >> 2u) + i] = x / (1.0 + exp(-x));
})";
}

static const char * ssm_conv_shader() {
    return R"(#version 310 es
layout(local_size_x = 64) in;
layout(std430, binding = 0) readonly buffer X { float x[]; };
layout(std430, binding = 1) readonly buffer C { float c[]; };
layout(std430, binding = 2) writeonly buffer Y { float y[]; };
uniform uint x_off, c_off, y_off;
uniform uint d_conv, d_inner, n_tokens, n_seqs;
void main() {
    uint out_i = gl_GlobalInvocationID.x;
    uint total = d_inner * n_tokens * n_seqs;
    if (out_i >= total) return;
    uint ch = out_i % d_inner;
    uint tok = (out_i / d_inner) % n_tokens;
    uint seq = out_i / (d_inner * n_tokens);
    float sum = 0.0;
    for (uint j = 0u; j < d_conv; ++j) {
        uint xi = (x_off >> 2u) + ((seq * d_inner + ch) * (n_tokens + d_conv - 1u) + tok + j);
        uint ci = (c_off >> 2u) + ch * d_conv + j;
        sum += x[xi] * c[ci];
    }
    y[(y_off >> 2u) + (seq * n_tokens + tok) * d_inner + ch] = sum;
})";
}

static const char * gated_delta_net_shader() {
    return R"(#version 310 es
layout(local_size_x = 128) in;
layout(std430, binding = 0) readonly buffer Q { float q[]; };
layout(std430, binding = 1) readonly buffer K { float k[]; };
layout(std430, binding = 2) readonly buffer V { float v[]; };
layout(std430, binding = 3) readonly buffer G { float g[]; };
layout(std430, binding = 4) readonly buffer B { float beta[]; };
layout(std430, binding = 5) readonly buffer S { float state[]; };
layout(std430, binding = 6) writeonly buffer O { float outv[]; };

uniform uint q_off, k_off, v_off, g_off, beta_off, state_off, out_off;
uniform uint q_s0, q_s1, q_s2, q_s3;
uniform uint k_s0, k_s1, k_s2, k_s3;
uniform uint v_s0, v_s1, v_s2, v_s3;
uniform uint g_s0, g_s1, g_s2, g_s3;
uniform uint b_s0, b_s1, b_s2, b_s3;
uniform uint S_v, H_v, H_k, n_tokens, n_seqs, K_snap;
uniform float scale;

void main() {
    uint col = gl_LocalInvocationID.x;
    uint head = gl_WorkGroupID.x;
    uint seq = gl_WorkGroupID.y;
    if (col >= S_v || head >= H_v || seq >= n_seqs) return;

    uint qhead = head % H_k;
    uint qbase = q_off >> 2u;
    uint kbase = k_off >> 2u;
    uint vbase = v_off >> 2u;
    uint gbase = g_off >> 2u;
    uint bbase = beta_off >> 2u;
    uint sbase = state_off >> 2u;
    uint obase = out_off >> 2u;

    float s[128];
    for (uint r = 0u; r < S_v; ++r) {
        uint si = sbase + seq * (S_v * S_v * H_v) +
                  head * (S_v * S_v) + col * S_v + r;
        s[r] = state[si];
    }

    for (uint t = 0u; t < n_tokens; ++t) {
        float decay_log = g[gbase + seq*g_s3 + head*g_s1 + t*g_s2];
        float decay = exp(decay_log);
        float bv = beta[bbase + seq*b_s3 + head*b_s1 + t*b_s2];

        float pred = 0.0;
        for (uint r = 0u; r < S_v; ++r) {
            uint qi = qbase + seq*q_s3 + qhead*q_s1 + t*q_s2 + r*q_s0;
            uint ki = kbase + seq*k_s3 + qhead*k_s1 + t*k_s2 + r*k_s0;
            s[r] *= decay;
            pred += s[r] * k[ki];
        }

        uint vi = vbase + seq*v_s3 + head*v_s1 + t*v_s2 + col*v_s0;
        float delta = (v[vi] - pred) * bv;

        float result = 0.0;
        for (uint r = 0u; r < S_v; ++r) {
            uint ki = kbase + seq*k_s3 + qhead*k_s1 + t*k_s2 + r*k_s0;
            s[r] += k[ki] * delta;
            uint qi = qbase + seq*q_s3 + qhead*q_s1 + t*q_s2 + r*q_s0;
            result += s[r] * q[qi];
        }

        uint oi = obase + (seq*n_tokens*H_v + t*H_v + head)*S_v + col;
        outv[oi] = result * scale;

        if (K_snap > 1u) {
            int slot = int(n_tokens) - 1 - int(t);
            if (slot >= 0 && slot < int(K_snap)) {
                uint snap_base = obase + n_tokens*n_seqs*H_v*S_v +
                    uint(slot) * S_v*S_v*H_v*n_seqs +
                    seq*S_v*S_v*H_v + head*S_v*S_v;
                for (uint r = 0u; r < S_v; ++r) {
                    outv[snap_base + col*S_v + r] = s[r];
                }
            }
        }
    }

    if (K_snap == 1u) {
        uint final_base = obase + n_tokens*n_seqs*H_v*S_v +
            seq*S_v*S_v*H_v + head*S_v*S_v;
        for (uint r = 0u; r < S_v; ++r) {
            outv[final_base + col*S_v + r] = s[r];
        }
    }
})";
}

static GLuint g_elementwise = 0;
static GLuint g_q6k = 0;
static GLuint g_gdn = 0;
static GLuint g_silu = 0;
static GLuint g_ssm_conv = 0;

static bool ensure_programs() {
    if (g_elementwise && g_q6k && g_gdn && g_silu && g_ssm_conv) return true;
    if (!g_elementwise) g_elementwise = compile_compute(elementwise_shader());
    if (!g_q6k) g_q6k = compile_compute(q6k_matmul_shader());
    if (!g_gdn) g_gdn = compile_compute(gated_delta_net_shader());
    if (!g_silu) g_silu = compile_compute(silu_shader());
    if (!g_ssm_conv) g_ssm_conv = compile_compute(ssm_conv_shader());
    return g_elementwise && g_q6k && g_gdn && g_silu && g_ssm_conv;
}

static size_t tensor_offset(const ggml_backend_buffer_t buffer, const ggml_tensor * tensor) {
    auto * ctx = static_cast<BufferContext *>(buffer->context);
    return static_cast<size_t>(reinterpret_cast<uintptr_t>(tensor->data) -
                               reinterpret_cast<uintptr_t>(ctx->virtual_base));
}

static BufferContext * buffer_ctx(const ggml_tensor * tensor) {
    return tensor && tensor->buffer ? static_cast<BufferContext *>(tensor->buffer->context) : nullptr;
}

static ggml_backend_buffer_t alloc_buffer(ggml_backend_buffer_type_t, size_t size) {
    if (size == 0 || size > kMaxSsboBytes) return nullptr;
    auto * ctx = new BufferContext;
    ctx->size = size;
    ctx->virtual_base = mmap(nullptr, size, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (ctx->virtual_base == MAP_FAILED) {
        delete ctx;
        return nullptr;
    }
    glGenBuffers(1, &ctx->buffer);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, ctx->buffer);
    glBufferData(GL_SHADER_STORAGE_BUFFER, static_cast<GLsizeiptr>(size), nullptr, GL_DYNAMIC_DRAW);
    if (glGetError() != GL_NO_ERROR) {
        glDeleteBuffers(1, &ctx->buffer);
        munmap(ctx->virtual_base, size);
        delete ctx;
        return nullptr;
    }
    return ggml_backend_buffer_init(&ggml_buft, {
        [](ggml_backend_buffer_t b) {
            auto * c = static_cast<BufferContext *>(b->context);
            glDeleteBuffers(1, &c->buffer);
            munmap(c->virtual_base, c->size);
            delete c;
        },
        [](ggml_backend_buffer_t b) -> void * {
            return static_cast<BufferContext *>(b->context)->virtual_base;
        },
        nullptr,
        [](ggml_backend_buffer_t b, ggml_tensor * t, uint8_t v, size_t o, size_t s) {
            auto * c = static_cast<BufferContext *>(b->context);
            std::string zeros(s, static_cast<char>(v));
            glBindBuffer(GL_SHADER_STORAGE_BUFFER, c->buffer);
            glBufferSubData(GL_SHADER_STORAGE_BUFFER, static_cast<GLintptr>(reinterpret_cast<uintptr_t>(t->data) -
                reinterpret_cast<uintptr_t>(c->virtual_base) + o), static_cast<GLsizeiptr>(s), zeros.data());
        },
        [](ggml_backend_buffer_t b, ggml_tensor * t, const void * data, size_t o, size_t s) {
            auto * c = static_cast<BufferContext *>(b->context);
            size_t off = static_cast<size_t>(reinterpret_cast<uintptr_t>(t->data) -
                reinterpret_cast<uintptr_t>(c->virtual_base)) + o;
            glBindBuffer(GL_SHADER_STORAGE_BUFFER, c->buffer);
            glBufferSubData(GL_SHADER_STORAGE_BUFFER, static_cast<GLintptr>(off), static_cast<GLsizeiptr>(s), data);
        },
        [](ggml_backend_buffer_t b, const ggml_tensor * t, void * data, size_t o, size_t s) {
            auto * c = static_cast<BufferContext *>(b->context);
            size_t off = static_cast<size_t>(reinterpret_cast<uintptr_t>(t->data) -
                reinterpret_cast<uintptr_t>(c->virtual_base)) + o;
            glBindBuffer(GL_SHADER_STORAGE_BUFFER, c->buffer);
            void * mapped = glMapBufferRange(GL_SHADER_STORAGE_BUFFER, static_cast<GLintptr>(off),
                                             static_cast<GLsizeiptr>(s), GL_MAP_READ_BIT);
            if (mapped) {
                std::memcpy(data, mapped, s);
                glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
            }
        },
        nullptr,
        nullptr,
        nullptr,
        [](ggml_backend_buffer_t b, uint8_t value) {
            auto * c = static_cast<BufferContext *>(b->context);
            glBindBuffer(GL_SHADER_STORAGE_BUFFER, c->buffer);
            glClearBufferData(GL_SHADER_STORAGE_BUFFER, GL_R8UI, GL_RED_INTEGER, GL_UNSIGNED_BYTE, &value);
        },
        nullptr
    }, ctx, size);
}

static const char * buft_name(ggml_backend_buffer_type_t) { return "OPENGL_ES_BUFFER"; }
static size_t buft_alignment(ggml_backend_buffer_type_t) { return 256; }
static size_t buft_max(ggml_backend_buffer_type_t) { return kMaxSsboBytes; }
static size_t buft_alloc_size(ggml_backend_buffer_type_t, const ggml_tensor * t) { return ggml_nbytes(t); }
static bool buft_host(ggml_backend_buffer_type_t) { return false; }

static ggml_backend_buffer_type_i buft_i = {
    buft_name, alloc_buffer, nullptr, buft_alignment, buft_max, buft_alloc_size, nullptr, buft_host
};

static bool supports_op(ggml_backend_dev_t, const ggml_tensor * op) {
    if (!op) return false;
    if (op->op == GGML_OP_NONE || op->op == GGML_OP_VIEW || op->op == GGML_OP_RESHAPE ||
        op->op == GGML_OP_PERMUTE || op->op == GGML_OP_TRANSPOSE) return true;
    if ((op->op == GGML_OP_ADD || op->op == GGML_OP_MUL) &&
        op->type == GGML_TYPE_F32 && op->src[0] && op->src[1] &&
        op->src[0]->type == GGML_TYPE_F32 && op->src[1]->type == GGML_TYPE_F32) return true;
    if (op->op == GGML_OP_MUL_MAT && op->type == GGML_TYPE_F32 &&
        op->src[0] && op->src[1] && op->src[0]->type == GGML_TYPE_Q6_K &&
        op->src[1]->type == GGML_TYPE_F32 && op->src[0]->ne[0] % 256 == 0) return true;
    if (op->op == GGML_OP_SSM_CONV && op->type == GGML_TYPE_F32 &&
        op->src[0] && op->src[1] && op->src[0]->type == GGML_TYPE_F32 &&
        (op->src[1]->type == GGML_TYPE_F32 || op->src[1]->type == GGML_TYPE_F16) &&
        op->src[0]->ne[1] == op->src[1]->ne[1]) return true;
    if (op->op == GGML_OP_SILU && op->type == GGML_TYPE_F32 &&
        op->src[0] && op->src[0]->type == GGML_TYPE_F32) return true;
    if (op->op == GGML_OP_GATED_DELTA_NET && op->type == GGML_TYPE_F32 &&
        op->src[0] && op->src[1] && op->src[2] && op->src[3] && op->src[4] && op->src[5] &&
        op->src[0]->type == GGML_TYPE_F32 && op->src[1]->type == GGML_TYPE_F32 &&
        op->src[2]->type == GGML_TYPE_F32 && op->src[3]->type == GGML_TYPE_F32 &&
        op->src[4]->type == GGML_TYPE_F32 && op->src[5]->type == GGML_TYPE_F32 &&
        op->src[0]->ne[0] <= 128 && op->src[2]->ne[0] <= 128 &&
        op->src[2]->ne[0] == op->src[2]->ne[1] &&
        op->src[5]->ne[0] == op->src[5]->ne[1]) return true;
    return false;
}

static enum ggml_status graph_compute(ggml_backend_t, ggml_cgraph * graph) {
    if (!ensure_programs()) return GGML_STATUS_FAILED;
    for (int i = 0; i < graph->n_nodes; ++i) {
        ggml_tensor * op = graph->nodes[i];
        if (!supports_op(nullptr, op)) return GGML_STATUS_FAILED;
        if (op->op == GGML_OP_NONE || op->op == GGML_OP_VIEW || op->op == GGML_OP_RESHAPE ||
            op->op == GGML_OP_PERMUTE || op->op == GGML_OP_TRANSPOSE) continue;

        if (op->op == GGML_OP_ADD || op->op == GGML_OP_MUL) {
            auto * a = buffer_ctx(op->src[0]);
            auto * b = buffer_ctx(op->src[1]);
            auto * c = buffer_ctx(op);
            if (!a || !b || !c) return GGML_STATUS_FAILED;
            glUseProgram(g_elementwise);
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, a->buffer);
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, b->buffer);
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, c->buffer);
            glUniform1ui(glGetUniformLocation(g_elementwise, "a_off"), static_cast<GLuint>(tensor_offset(op->src[0]->buffer, op->src[0])));
            glUniform1ui(glGetUniformLocation(g_elementwise, "b_off"), static_cast<GLuint>(tensor_offset(op->src[1]->buffer, op->src[1])));
            glUniform1ui(glGetUniformLocation(g_elementwise, "c_off"), static_cast<GLuint>(tensor_offset(op->buffer, op)));
            glUniform1ui(glGetUniformLocation(g_elementwise, "n"), static_cast<GLuint>(ggml_nelements(op)));
            glUniform1ui(glGetUniformLocation(g_elementwise, "op"), op->op == GGML_OP_ADD ? 0u : 1u);
            glDispatchCompute(static_cast<GLuint>((ggml_nelements(op) + 63) / 64), 1, 1);
            glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
            continue;
        }

        if (op->op == GGML_OP_SILU) {
            auto * a = buffer_ctx(op->src[0]);
            auto * out = buffer_ctx(op);
            if (!a || !out) return GGML_STATUS_FAILED;
            glUseProgram(g_silu);
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, a->buffer);
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, out->buffer);
            glUniform1ui(glGetUniformLocation(g_silu, "a_off"), static_cast<GLuint>(tensor_offset(op->src[0]->buffer, op->src[0])));
            glUniform1ui(glGetUniformLocation(g_silu, "c_off"), static_cast<GLuint>(tensor_offset(op->buffer, op)));
            glUniform1ui(glGetUniformLocation(g_silu, "n"), static_cast<GLuint>(ggml_nelements(op)));
            glDispatchCompute(static_cast<GLuint>((ggml_nelements(op) + 63) / 64), 1, 1);
            glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
            continue;
        }

        if (op->op == GGML_OP_SSM_CONV) {
            auto * x = buffer_ctx(op->src[0]);
            auto * cbuf = buffer_ctx(op->src[1]);
            auto * out = buffer_ctx(op);
            if (!x || !cbuf || !out) return GGML_STATUS_FAILED;
            glUseProgram(g_ssm_conv);
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, x->buffer);
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, cbuf->buffer);
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, out->buffer);
            glUniform1ui(glGetUniformLocation(g_ssm_conv, "x_off"), static_cast<GLuint>(tensor_offset(op->src[0]->buffer, op->src[0])));
            glUniform1ui(glGetUniformLocation(g_ssm_conv, "c_off"), static_cast<GLuint>(tensor_offset(op->src[1]->buffer, op->src[1])));
            glUniform1ui(glGetUniformLocation(g_ssm_conv, "y_off"), static_cast<GLuint>(tensor_offset(op->buffer, op)));
            glUniform1ui(glGetUniformLocation(g_ssm_conv, "d_conv"), static_cast<GLuint>(op->src[1]->ne[0]));
            glUniform1ui(glGetUniformLocation(g_ssm_conv, "d_inner"), static_cast<GLuint>(op->src[1]->ne[1]));
            glUniform1ui(glGetUniformLocation(g_ssm_conv, "n_tokens"), static_cast<GLuint>(op->ne[1]));
            glUniform1ui(glGetUniformLocation(g_ssm_conv, "n_seqs"), static_cast<GLuint>(op->ne[2]));
            glDispatchCompute(static_cast<GLuint>((ggml_nelements(op) + 63) / 64), 1, 1);
            glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
            continue;
        }

        if (op->op == GGML_OP_GATED_DELTA_NET) {
            auto * q = buffer_ctx(op->src[0]);
            auto * k = buffer_ctx(op->src[1]);
            auto * v = buffer_ctx(op->src[2]);
            auto * g = buffer_ctx(op->src[3]);
            auto * b = buffer_ctx(op->src[4]);
            auto * s = buffer_ctx(op->src[5]);
            auto * out = buffer_ctx(op);
            if (!q || !k || !v || !g || !b || !s || !out) return GGML_STATUS_FAILED;

            glUseProgram(g_gdn);
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, q->buffer);
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, k->buffer);
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, v->buffer);
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, g->buffer);
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, b->buffer);
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 5, s->buffer);
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 6, out->buffer);

            auto setoff = [&](const char * name, const ggml_tensor * t) {
                glUniform1ui(glGetUniformLocation(g_gdn, name),
                    static_cast<GLuint>(tensor_offset(t->buffer, t)));
            };
            setoff("q_off", op->src[0]); setoff("k_off", op->src[1]);
            setoff("v_off", op->src[2]); setoff("g_off", op->src[3]);
            setoff("beta_off", op->src[4]); setoff("state_off", op->src[5]);
            setoff("out_off", op);

            auto stride = [&](const ggml_tensor * t, int d) -> GLuint {
                return static_cast<GLuint>(t->nb[d] / sizeof(float));
            };
            const char * qn[] = {"q_s0","q_s1","q_s2","q_s3"};
            const char * kn[] = {"k_s0","k_s1","k_s2","k_s3"};
            const char * vn[] = {"v_s0","v_s1","v_s2","v_s3"};
            const char * gn[] = {"g_s0","g_s1","g_s2","g_s3"};
            const char * bn[] = {"b_s0","b_s1","b_s2","b_s3"};
            const ggml_tensor * ts[] = {op->src[0],op->src[1],op->src[2],op->src[3],op->src[4]};
            const char ** ns[] = {qn,kn,vn,gn,bn};
            for (int ti=0; ti<5; ++ti) for (int d=0; d<4; ++d)
                glUniform1ui(glGetUniformLocation(g_gdn, ns[ti][d]), stride(ts[ti], d));

            const GLuint sv = static_cast<GLuint>(op->src[2]->ne[0]);
            const GLuint hv = static_cast<GLuint>(op->src[2]->ne[1]);
            const GLuint hk = static_cast<GLuint>(op->src[0]->ne[1]);
            const GLuint nt = static_cast<GLuint>(op->src[2]->ne[2]);
            const GLuint nsq = static_cast<GLuint>(op->src[2]->ne[3]);
            const GLuint ksnap = static_cast<GLuint>(op->op_params[0]);
            glUniform1ui(glGetUniformLocation(g_gdn, "S_v"), sv);
            glUniform1ui(glGetUniformLocation(g_gdn, "H_v"), hv);
            glUniform1ui(glGetUniformLocation(g_gdn, "H_k"), hk);
            glUniform1ui(glGetUniformLocation(g_gdn, "n_tokens"), nt);
            glUniform1ui(glGetUniformLocation(g_gdn, "n_seqs"), nsq);
            glUniform1ui(glGetUniformLocation(g_gdn, "K_snap"), ksnap);
            glUniform1f(glGetUniformLocation(g_gdn, "scale"), 1.0f / std::sqrt(float(op->src[0]->ne[0])));

            glDispatchCompute(hv, nsq, 1);
            glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
            continue;
        }

        if (op->op == GGML_OP_MUL_MAT) {
            auto * w = buffer_ctx(op->src[0]);
            auto * x = buffer_ctx(op->src[1]);
            auto * y = buffer_ctx(op);
            if (!w || !x || !y) return GGML_STATUS_FAILED;
            glUseProgram(g_q6k);
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, w->buffer);
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, x->buffer);
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, y->buffer);
            glUniform1ui(glGetUniformLocation(g_q6k, "w_off"), static_cast<GLuint>(tensor_offset(op->src[0]->buffer, op->src[0])));
            glUniform1ui(glGetUniformLocation(g_q6k, "x_off"), static_cast<GLuint>(tensor_offset(op->src[1]->buffer, op->src[1])));
            glUniform1ui(glGetUniformLocation(g_q6k, "y_off"), static_cast<GLuint>(tensor_offset(op->buffer, op)));
            glUniform1ui(glGetUniformLocation(g_q6k, "k"), static_cast<GLuint>(op->src[0]->ne[0]));
            glUniform1ui(glGetUniformLocation(g_q6k, "rows"), static_cast<GLuint>(op->src[0]->ne[1]));
            glUniform1ui(glGetUniformLocation(g_q6k, "cols"), static_cast<GLuint>(op->src[1]->ne[1]));
            uint64_t total = static_cast<uint64_t>(op->src[0]->ne[1]) * static_cast<uint64_t>(op->src[1]->ne[1]);
            glDispatchCompute(static_cast<GLuint>((total + 63) / 64), 1, 1);
            glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
        }
    }
    return glGetError() == GL_NO_ERROR ? GGML_STATUS_SUCCESS : GGML_STATUS_FAILED;
}

static ggml_guid_t backend_guid() {
    static ggml_guid guid = { 0x61,0x2f,0x1d,0x8a,0x93,0x71,0x4c,0x55,0x8e,0x2b,0x0d,0x70,0x55,0x9c,0x33,0x42 };
    return &guid;
}

static ggml_backend_t init_backend(ggml_backend_dev_t dev, const char *) {
    if (!opengles_runtime_available() || !ensure_programs()) return nullptr;
    static const struct ggml_backend_i backend_i = {
        [](ggml_backend_t){ return "OpenGL ES"; },
        [](ggml_backend_t b){ delete static_cast<DeviceContext *>(b->context); delete b; },
        nullptr, nullptr, nullptr, nullptr, nullptr,
        nullptr,
        nullptr, nullptr, nullptr, nullptr,
        graph_compute,
        nullptr, nullptr, nullptr
    };
    return new ggml_backend{ backend_guid(), backend_i, dev, new DeviceContext{} };
}

static const char * device_name(ggml_backend_dev_t) { return kName; }
static const char * device_description(ggml_backend_dev_t) { return kDescription; }
static void device_memory(ggml_backend_dev_t, size_t * free, size_t * total) {
    *free = 0; *total = 0;
}
static enum ggml_backend_dev_type device_type(ggml_backend_dev_t) { return GGML_BACKEND_DEVICE_TYPE_GPU; }
static void device_props(ggml_backend_dev_t dev, ggml_backend_dev_props * p) {
    p->name = device_name(dev);
    p->description = device_description(dev);
    p->memory_free = 0;
    p->memory_total = 0;
    p->type = GGML_BACKEND_DEVICE_TYPE_GPU;
    p->device_id = nullptr;
    p->caps = { false, false, false, false, false };
}
static ggml_backend_buffer_type_t device_buft(ggml_backend_dev_t) { return &ggml_buft; }
static bool device_supports_buft(ggml_backend_dev_t, ggml_backend_buffer_type_t b) { return b == &ggml_buft; }
static bool device_offload(ggml_backend_dev_t, const ggml_tensor * op) { return supports_op(nullptr, op); }

static const ggml_backend_device_i device_i = {
    device_name, device_description, device_memory, device_type, device_props,
    init_backend, device_buft, nullptr, nullptr, supports_op, device_supports_buft,
    device_offload, nullptr, nullptr, nullptr
};

static size_t reg_count(ggml_backend_reg_t) { return 1; }
static ggml_backend_dev_t reg_device(ggml_backend_reg_t reg, size_t) {
    static DeviceContext ctx;
    static ggml_backend_device dev = { device_i, reg, &ctx };
    ggml_buft.device = &dev;
    return &dev;
}
static const char * reg_name(ggml_backend_reg_t) { return kName; }
static void * reg_proc(ggml_backend_reg_t, const char *) { return nullptr; }

}

ggml_backend_reg_t ggml_backend_opengles_reg(void) {
    if (!opengles_runtime_available()) return nullptr;
    if (!g_ready) {
        ggml_buft = { buft_i, nullptr, nullptr };
        g_ready = true;
    }
    static ggml_backend_reg reg = { GGML_BACKEND_API_VERSION,
        { reg_name, reg_count, reg_device, reg_proc }, nullptr };
    return &reg;
}
