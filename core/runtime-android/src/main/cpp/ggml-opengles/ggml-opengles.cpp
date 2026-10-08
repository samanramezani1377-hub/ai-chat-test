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
#include <algorithm>
#include <cstdio>
#include <string>

namespace {

static constexpr size_t kMaxSsboBytes = 128u * 1024u * 1024u - 256u;
static const char * kName = "OpenGL ES";
static const char * kDescription = "Generic Android OpenGL ES 3.1 GGML GPU backend";

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
uniform float scale;
uniform float bias;
void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i >= n) return;
    uint ai = (a_off >> 2u) + i;
    uint bi = (b_off >> 2u) + i;
    uint ci = (c_off >> 2u) + i;
    float x = a[ai];
    float y = b[bi];
    c[ci] = op == 0u ? x + y : (op == 1u ? x * y : x * scale + bias);
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
uniform uint x_s0, x_s1, x_s2, x_s3, y_s0, y_s1, y_s2, y_s3, x_n2, x_n3, w_type;

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

float q4_0(uint block, uint idx) {
    uint base = block * 18u;
    float d = half_at(base);
    uint j = idx & 31u;
    uint q = byte_u8(base + 2u + (j & 15u));
    return d * (float(int(j < 16u ? (q & 15u) : (q >> 4u)) - 8));
}
float q8_0_mm(uint block, uint idx) {
    uint base = block * 34u;
    float d = half_at(base);
    return d * float(byte_i8(base + 2u + (idx & 31u)));
}

float q6(uint block, uint idx) {
    uint base = block * 210u;
    float d = half_at(base);
    uint group = idx / 16u;
    int sc = byte_i8(base + 192u + group);
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
    uint total = rows * cols * x_n2 * x_n3;
    if (out >= total) return;
    uint row = out % rows;
    uint col = (out / rows) % cols;
    float sum = 0.0;
    uint blocks = k / 256u;
    for (uint b = 0u; b < blocks; ++b) {
        for (uint j = 0u; j < 256u; ++j) {
            uint kk = b * 256u + j;
            float xv = x[(x_off >> 2u) + col*x_s1 + (out/(rows*cols))*x_s2 + kk*x_s0];
            uint wb = b + row * blocks;
            float wv = w_type == 0u ? q6(wb, j) : (w_type == 1u ? q4_0(wb, j) : q8_0_mm(wb, j));
            sum += wv * xv;
        }
    }
    y[(y_off >> 2u) + col*y_s1 + (out/(rows*cols))*y_s2 + row*y_s0] = sum;
})";
}

static const char * get_rows_shader() {
    return R"(#version 310 es
layout(local_size_x = 64) in;
layout(std430, binding = 0) readonly buffer W { uint w[]; };
layout(std430, binding = 1) readonly buffer I { int ids[]; };
layout(std430, binding = 2) writeonly buffer Y { float y[]; };

uniform uint w_off, i_off, y_off, row_bytes, cols, n_rows, w_type;

uint byte_u8(uint off) {
    uint word = w[(w_off + off) >> 2u];
    return (word >> ((off & 3u) * 8u)) & 255u;
}
int byte_i8(uint off) {
    uint v = byte_u8(off);
    return v >= 128u ? int(v) - 256 : int(v);
}
float half_at(uint off) {
    uint word = w[(w_off + off) >> 2u];
    uint bits = word & 65535u;
    uint lo = bits & 1023u;
    uint ex = (bits >> 10u) & 31u;
    uint sign = bits >> 15u;
    if (ex == 0u) return sign != 0u ? -exp2(-14.0) * (float(lo)/1024.0) : exp2(-14.0) * (float(lo)/1024.0);
    if (ex == 31u) return sign != 0u ? -1.0/0.0 : 1.0/0.0;
    return (sign != 0u ? -1.0 : 1.0) * exp2(float(ex)-15.0) * (1.0 + float(lo)/1024.0);
}
float q6(uint row_off, uint idx) {
    uint block = idx / 256u;
    uint j = idx & 255u;
    uint base = row_off + block * 210u;
    float d = half_at(base);
    uint n128 = j / 128u;
    uint l = j & 31u;
    uint qbase = base + n128 * 64u;
    uint hbase = base + 128u + n128 * 32u;
    uint scbase = base + 192u + n128 * 8u;
    uint q1 = byte_u8(qbase + l) & 15u;
    uint q2 = byte_u8(qbase + 32u + l) & 15u;
    uint q3 = (byte_u8(qbase + l) >> 4u) & 15u;
    uint q4 = (byte_u8(qbase + 32u + l) >> 4u) & 15u;
    uint h = byte_u8(hbase + l);
    uint q;
    uint si;
    if (j < 32u) { q = q1 | ((h & 3u) << 4u); si = 0u; }
    else if (j < 64u) { q = q2 | (((h >> 2u) & 3u) << 4u); si = 2u; }
    else if (j < 96u) { q = q3 | (((h >> 4u) & 3u) << 4u); si = 4u; }
    else { q = q4 | (((h >> 6u) & 3u) << 4u); si = 6u; }
    int sc = byte_i8(scbase + si);
    return d * float(sc) * (float(q) - 32.0);
}
float q8_0(uint row_off, uint idx) {
    uint block = idx / 32u;
    uint j = idx & 31u;
    uint base = row_off + block * 34u;
    float d = half_at(base);
    return d * float(byte_i8(base + 2u + j));
}
void main() {
    uint idx = gl_GlobalInvocationID.x;
    uint total = cols * n_rows;
    if (idx >= total) return;
    uint col = idx % cols;
    uint row = idx / cols;
    int id = ids[(i_off >> 2u) + row];
    if (id < 0) { y[(y_off >> 2u) + idx] = 0.0; return; }
    uint row_off = uint(id) * row_bytes;
    float v;
    if (w_type == 0u) {
        uint word = w[(w_off + row_off + col * 4u) >> 2u];
        v = uintBitsToFloat(word);
    } else if (w_type == 1u) {
        v = half_at(row_off + col * 2u);
    } else if (w_type == 2u) {
        v = q6(row_off, col);
    } else {
        v = q8_0(row_off, col);
    }
    y[(y_off >> 2u) + idx] = v;
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
uniform uint S_v, H_v, H_k, n_tokens, n_seqs, K_snap, g_width;
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
        float decay_log;
        if (g_width == 1u) {
            decay_log = g[gbase + seq*g_s3 + head*g_s1 + t*g_s2];
        } else {
            decay_log = g[gbase + seq*g_s3 + head*g_s1 + t*g_s2 + col*g_s0];
        }
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


static const char * rms_norm_shader() {
    return R"(#version 310 es
layout(local_size_x = 64) in;
layout(std430, binding = 0) readonly buffer X { float x[]; };
layout(std430, binding = 1) writeonly buffer Y { float y[]; };
uniform uint x_off, y_off;
uniform uint n0, n1, n2, n3;
uniform uint xs0, xs1, xs2, xs3;
uniform uint ys0, ys1, ys2, ys3;
uniform float eps;
void main() {
    uint row = gl_GlobalInvocationID.x;
    uint rows = n1*n2*n3;
    if (row >= rows) return;
    uint d3 = row / (n1*n2);
    uint rem = row - d3*n1*n2;
    uint d2 = rem / n1;
    uint d1 = rem - d2*n1;
    uint xb = (x_off>>2u) + d1*xs1 + d2*xs2 + d3*xs3;
    uint yb = (y_off>>2u) + d1*ys1 + d2*ys2 + d3*ys3;
    float ss = 0.0;
    for (uint i=0u;i<n0;++i) {
        float v=x[xb+i*xs0]; ss += v*v;
    }
    float inv=inversesqrt(ss/float(n0)+eps);
    for (uint i=0u;i<n0;++i) y[yb+i*ys0] = x[xb+i*xs0]*inv;
})";
}

static const char * unary_shader() {
    return R"(#version 310 es
layout(local_size_x = 64) in;
layout(std430, binding = 0) readonly buffer A { float a[]; };
layout(std430, binding = 1) writeonly buffer C { float c[]; };
uniform uint a_off, c_off, n, kind;
void main() {
    uint i=gl_GlobalInvocationID.x; if(i>=n)return;
    float x=a[(a_off>>2u)+i];
    float y;
    if (kind == 0u) y = 1.0/(1.0+exp(-x));                 // sigmoid
    else if (kind == 1u) y = log(1.0+exp(-abs(x))) + max(x,0.0); // softplus
    else if (kind == 2u) y = 0.5*x*(1.0 + tanh(0.7978845608*(x + 0.044715*x*x*x))); // gelu erf approximation
    else if (kind == 3u) y = 0.5*x*(1.0 + tanh(0.7978845608*(x + 0.044715*x*x*x))); // gelu tanh
    else if (kind == 4u) y = tanh(x);
    else if (kind == 5u) y = exp(x);
    else y = log(max(x, 1e-20));
    c[(c_off>>2u)+i]=y;
})";
}

static const char * softmax_shader() {
    return R"(#version 310 es
layout(local_size_x = 1) in;
layout(std430, binding = 0) readonly buffer X { float x[]; };
layout(std430, binding = 1) readonly buffer M { float m[]; };
layout(std430, binding = 2) writeonly buffer Y { float y[]; };
uniform uint x_off,m_off,y_off,n0,n1,n2,n3;
uniform uint xs0,xs1,xs2,xs3, ms0,ms1,ms2,ms3, ys0,ys1,ys2,ys3;
uniform uint has_mask; uniform float scale;
void main() {
    uint row=gl_GlobalInvocationID.x, rows=n1*n2*n3; if(row>=rows)return;
    uint d3=row/(n1*n2), rem=row-d3*n1*n2, d2=rem/n1, d1=rem-d2*n1;
    uint xb=(x_off>>2u)+d1*xs1+d2*xs2+d3*xs3;
    uint yb=(y_off>>2u)+d1*ys1+d2*ys2+d3*ys3;
    uint mb=(m_off>>2u)+d1*ms1+d2*ms2+d3*ms3;
    float mx=-3.402823e38;
    for(uint i=0u;i<n0;++i){float v=x[xb+i*xs0]*scale; if(has_mask!=0u)v+=m[mb+i*ms0]; mx=max(mx,v);}
    float sum=0.0;
    for(uint i=0u;i<n0;++i){float v=x[xb+i*xs0]*scale; if(has_mask!=0u)v+=m[mb+i*ms0]; float e=exp(v-mx); y[yb+i*ys0]=e; sum+=e;}
    float inv=1.0/max(sum,1e-20);
    for(uint i=0u;i<n0;++i)y[yb+i*ys0]*=inv;
})";
}

static const char * matmul_f32_shader() {
    return R"(#version 310 es
layout(local_size_x = 64) in;
layout(std430,binding=0) readonly buffer A{float a[];};
layout(std430,binding=1) readonly buffer B{float b[];};
layout(std430,binding=2) writeonly buffer C{float c[];};
uniform uint K,M,N,A2,A3,B2,B3;
uniform uint a_off,b_off,c_off,as0,as1,as2,as3,bs0,bs1,bs2,bs3,cs0,cs1,cs2,cs3;
void main(){
 uint idx=gl_GlobalInvocationID.x, total=M*N*B2*B3; if(idx>=total)return;
 uint n=idx%N, t=idx/N, m=t%M, bt=t/M, d2=bt%B2, d3=bt/B2;
 uint ad2=d2%A2, ad3=d3%A3;
 uint ab=(a_off>>2u)+ad2*as2+ad3*as3;
 uint bb=(b_off>>2u)+d2*bs2+d3*bs3;
 uint cb=(c_off>>2u)+d2*cs2+d3*cs3;
 float sum=0.0;
 for(uint k=0u;k<K;++k) sum+=a[ab+m*as1+k*as0]*b[bb+n*bs1+k*bs0];
 c[cb+n*cs1+m*cs0]=sum;
})";
}

static const char * rope_shader() {
    return R"(#version 310 es
layout(local_size_x=64) in;
layout(std430,binding=0) readonly buffer X{float x[];};
layout(std430,binding=1) readonly buffer P{int p[];};
layout(std430,binding=2) writeonly buffer Y{float y[];};
uniform uint x_off,p_off,y_off,n0,n1,n2,n3,ps0,ps1,xs0,xs1,xs2,xs3,ys0,ys1,ys2,ys3;
uniform int n_dims,mode,n_ctx_orig;
uniform float freq_base,freq_scale,ext_factor,attn_factor,beta_fast,beta_slow;
uniform int s0,s1,s2,s3;
void main(){
 uint idx=gl_GlobalInvocationID.x,total=n0*n1*n2*n3;if(idx>=total)return;
 uint d0=idx%n0, r=idx/n0, d1=r%n1; r/=n1; uint d2=r%n2,d3=r/n2;
 uint xb=(x_off>>2u)+d0*xs0+d1*xs1+d2*xs2+d3*xs3;
 uint yb=(y_off>>2u)+d0*ys0+d1*ys1+d2*ys2+d3*ys3;
 float v=x[xb];
 if(d0>=uint(n_dims) || (d0&1u)!=0u){y[yb]=v;return;}
 int axis=0; int cumulative=0;
 if((mode & 4)!=0){
   if(d0/2u >= uint(s0)) { cumulative=s0; axis=1; }
   if(d0/2u >= uint(s0+s1)) { cumulative=s0+s1; axis=2; }
   if(d0/2u >= uint(s0+s1+s2)) { cumulative=s0+s1+s2; axis=3; }
 }
 int pos;
 if(axis==0) pos=p[(p_off>>2u)+d1*ps1];
 else if(axis==1) pos=p[(p_off>>2u)+ps0+d1*ps1];
 else if(axis==2) pos=p[(p_off>>2u)+2*ps0+d1*ps1];
 else pos=p[(p_off>>2u)+3*ps0+d1*ps1];
 float dim=float(d0/2u-cumulative);
 float theta=float(pos)*pow(freq_base,-2.0*dim/float(n_dims))*freq_scale;
 float cs=cos(theta), sn=sin(theta);
 uint pair=d0+1u;
 uint xp=(x_off>>2u)+pair*xs0+d1*xs1+d2*xs2+d3*xs3;
 uint yp=(y_off>>2u)+pair*ys0+d1*ys1+d2*ys2+d3*ys3;
 y[yb]=v*cs-x[xp]*sn;
})";
}

static GLuint g_get_rows = 0;
static GLuint g_elementwise = 0;
static GLuint g_q6k = 0;
static GLuint g_gdn = 0;
static GLuint g_silu = 0;
static GLuint g_ssm_conv = 0;
static GLuint g_rms_norm = 0;
static GLuint g_unary = 0;
static GLuint g_softmax = 0;
static GLuint g_matmul_f32 = 0;
static GLuint g_rope = 0;

static bool ensure_programs() {
    const auto & caps = opengles_runtime_info();
    if (g_get_rows && g_elementwise && g_q6k && g_silu && g_ssm_conv && g_rms_norm && g_unary && g_softmax && g_matmul_f32 && g_rope &&
        (caps.max_compute_ssbo_blocks < 7 || g_gdn != 0)) return true;
    if (!g_get_rows) g_get_rows = compile_compute(get_rows_shader());
    if (!g_gdn && caps.max_compute_ssbo_blocks >= 7) g_gdn = compile_compute(gated_delta_net_shader());
    if (!g_elementwise) g_elementwise = compile_compute(elementwise_shader());
    if (!g_q6k) g_q6k = compile_compute(q6k_matmul_shader());
    if (!g_silu) g_silu = compile_compute(silu_shader());
    if (!g_ssm_conv) g_ssm_conv = compile_compute(ssm_conv_shader());
    if (!g_rms_norm) g_rms_norm = compile_compute(rms_norm_shader());
    if (!g_unary) g_unary = compile_compute(unary_shader());
    if (!g_softmax) g_softmax = compile_compute(softmax_shader());
    if (!g_matmul_f32) g_matmul_f32 = compile_compute(matmul_f32_shader());
    if (!g_rope) g_rope = compile_compute(rope_shader());
    return g_get_rows && g_elementwise && g_q6k && g_silu && g_ssm_conv && g_rms_norm && g_unary && g_softmax && g_matmul_f32 && g_rope;
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
    const auto & caps = opengles_runtime_info();
    const size_t max_ssbo = caps.max_ssbo_block_size > 256 ? std::min<size_t>(
        kMaxSsboBytes, static_cast<size_t>(caps.max_ssbo_block_size) - 256u) : 0u;
    if (size == 0 || max_ssbo == 0 || size > max_ssbo) return nullptr;
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
static size_t buft_max(ggml_backend_buffer_type_t) {
    const auto & caps = opengles_runtime_info();
    return caps.max_ssbo_block_size > 256 ? std::min<size_t>(
        kMaxSsboBytes, static_cast<size_t>(caps.max_ssbo_block_size) - 256u) : 0u;
}
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
        op->src[0] && op->src[1] && op->src[1]->type == GGML_TYPE_F32 &&
        ((op->src[0]->type == GGML_TYPE_Q6_K && op->src[0]->ne[0] % 256 == 0) ||
         (op->src[0]->type == GGML_TYPE_Q4_0 && op->src[0]->ne[0] % 32 == 0) ||
         (op->src[0]->type == GGML_TYPE_Q8_0 && op->src[0]->ne[0] % 32 == 0))) return true;
    if (op->op == GGML_OP_GET_ROWS && op->type == GGML_TYPE_F32 && op->src[0] && op->src[1] &&
        op->src[1]->type == GGML_TYPE_I32 &&
        (op->src[0]->type == GGML_TYPE_F32 || op->src[0]->type == GGML_TYPE_F16 ||
         op->src[0]->type == GGML_TYPE_Q6_K || op->src[0]->type == GGML_TYPE_Q8_0) &&
        op->src[0]->ne[0] > 0 && op->src[0]->ne[0] % (op->src[0]->type == GGML_TYPE_Q8_0 ? 32 : 1) == 0) return true;
    if (op->op == GGML_OP_SCALE && op->type == GGML_TYPE_F32 && op->src[0] &&
        op->src[0]->type == GGML_TYPE_F32 && ggml_is_contiguous(op->src[0])) return true;
    if (op->op == GGML_OP_SSM_CONV && op->type == GGML_TYPE_F32 &&
        op->src[0] && op->src[1] && op->src[0]->type == GGML_TYPE_F32 &&
        op->src[1]->type == GGML_TYPE_F32 &&
        op->src[0]->ne[1] == op->src[1]->ne[1]) return true;
    if (op->op == GGML_OP_SILU && op->type == GGML_TYPE_F32 &&
        op->src[0] && op->src[0]->type == GGML_TYPE_F32) return true;
    if (op->op == GGML_OP_GATED_DELTA_NET && opengles_runtime_info().max_compute_ssbo_blocks >= 7 &&
        g_gdn != 0 && op->type == GGML_TYPE_F32 &&
        op->src[0] && op->src[1] && op->src[2] && op->src[3] && op->src[4] && op->src[5] &&
        op->src[0]->type == GGML_TYPE_F32 && op->src[1]->type == GGML_TYPE_F32 &&
        op->src[2]->type == GGML_TYPE_F32 && op->src[3]->type == GGML_TYPE_F32 &&
        op->src[4]->type == GGML_TYPE_F32 && op->src[5]->type == GGML_TYPE_F32 &&
        op->src[0]->ne[0] <= 128 && op->src[2]->ne[0] <= 128 &&
        op->src[2]->ne[0] == op->src[2]->ne[1] &&
        op->src[5]->ne[0] == op->src[5]->ne[1]) return true;
    if (op->op == GGML_OP_SCALE && op->type == GGML_TYPE_F32 && op->src[0] &&
        op->src[0]->type == GGML_TYPE_F32 && ggml_is_contiguous(op->src[0])) return true;
    if (op->type == GGML_TYPE_F32 && op->src[0]) {
        if (op->op == GGML_OP_RMS_NORM && op->src[0]->type == GGML_TYPE_F32 &&
            op->src[0]->ne[0] % 4 == 0) return true;
        if ((op->op == GGML_OP_SIGMOID || op->op == GGML_OP_SOFTPLUS ||
             op->op == GGML_OP_GELU || op->op == GGML_OP_GELU_ERF || op->op == GGML_OP_GELU_QUICK ||
             op->op == GGML_OP_TANH || op->op == GGML_OP_EXP || op->op == GGML_OP_LOG) &&
            op->src[0]->type == GGML_TYPE_F32) return true;
        if (op->op == GGML_OP_SOFT_MAX && op->src[0]->type == GGML_TYPE_F32 &&
            (!op->src[1] || op->src[1]->type == GGML_TYPE_F32)) return true;
        if (op->op == GGML_OP_ROPE && op->src[0]->type == GGML_TYPE_F32 &&
            op->src[1] && op->src[1]->type == GGML_TYPE_I32) return true;
        if (op->op == GGML_OP_MUL_MAT && op->src[1]->type == GGML_TYPE_F32 &&
            op->src[0]->type == GGML_TYPE_F32) return true;
    }
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

        if (op->op == GGML_OP_GET_ROWS) {
            auto * w = buffer_ctx(op->src[0]); auto * ids = buffer_ctx(op->src[1]); auto * y = buffer_ctx(op);
            if (!w || !ids || !y) return GGML_STATUS_FAILED;
            glUseProgram(g_get_rows);
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, w->buffer);
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, ids->buffer);
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, y->buffer);
            auto U=[&](const char * n, GLuint v){ glUniform1ui(glGetUniformLocation(g_get_rows,n),v); };
            U("w_off",(GLuint)tensor_offset(op->src[0]->buffer,op->src[0]));
            U("i_off",(GLuint)tensor_offset(op->src[1]->buffer,op->src[1]));
            U("y_off",(GLuint)tensor_offset(op->buffer,op));
            U("row_bytes",(GLuint)op->src[0]->nb[1]);
            U("cols",(GLuint)op->src[0]->ne[0]);
            U("n_rows",(GLuint)ggml_nelements(op->src[1]));
            uint type = 0u;
            if (op->src[0]->type == GGML_TYPE_F16) type = 1u;
            else if (op->src[0]->type == GGML_TYPE_Q6_K) type = 2u;
            else if (op->src[0]->type == GGML_TYPE_Q8_0) type = 3u;
            U("w_type",type);
            glDispatchCompute((GLuint)((ggml_nelements(op)+63)/64),1,1);
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


        if (op->op == GGML_OP_RMS_NORM) {
            auto * x=buffer_ctx(op->src[0]); auto * y=buffer_ctx(op);
            if(!x||!y)return GGML_STATUS_FAILED;
            glUseProgram(g_rms_norm);
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER,0,x->buffer); glBindBufferBase(GL_SHADER_STORAGE_BUFFER,1,y->buffer);
            auto U=[&](const char*n,GLuint v){glUniform1ui(glGetUniformLocation(g_rms_norm,n),v);};
            U("x_off",tensor_offset(op->src[0]->buffer,op->src[0])); U("y_off",tensor_offset(op->buffer,op));
            U("n0",(GLuint)op->ne[0]); U("n1",(GLuint)op->ne[1]); U("n2",(GLuint)op->ne[2]); U("n3",(GLuint)op->ne[3]);
            for(int d=0;d<4;++d){U((std::string("xs")+std::to_string(d)).c_str(),(GLuint)(op->src[0]->nb[d]/4));U((std::string("ys")+std::to_string(d)).c_str(),(GLuint)(op->nb[d]/4));}
            glUniform1f(glGetUniformLocation(g_rms_norm,"eps"),*reinterpret_cast<const float *>(op->op_params));
            glDispatchCompute((GLuint)((op->ne[1]*op->ne[2]*op->ne[3]+63)/64),1,1); glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT); continue;
        }

        if (op->op == GGML_OP_SIGMOID || op->op == GGML_OP_SOFTPLUS ||
            op->op == GGML_OP_GELU || op->op == GGML_OP_GELU_ERF || op->op == GGML_OP_GELU_QUICK ||
            op->op == GGML_OP_TANH || op->op == GGML_OP_EXP || op->op == GGML_OP_LOG) {
            auto * x=buffer_ctx(op->src[0]); auto * y=buffer_ctx(op); if(!x||!y)return GGML_STATUS_FAILED;
            glUseProgram(g_unary); glBindBufferBase(GL_SHADER_STORAGE_BUFFER,0,x->buffer); glBindBufferBase(GL_SHADER_STORAGE_BUFFER,1,y->buffer);
            glUniform1ui(glGetUniformLocation(g_unary,"a_off"),(GLuint)tensor_offset(op->src[0]->buffer,op->src[0]));
            glUniform1ui(glGetUniformLocation(g_unary,"c_off"),(GLuint)tensor_offset(op->buffer,op));
            glUniform1ui(glGetUniformLocation(g_unary,"n"),(GLuint)ggml_nelements(op));
            GLuint kind = op->op == GGML_OP_SIGMOID ? 0u :
                op->op == GGML_OP_SOFTPLUS ? 1u :
                op->op == GGML_OP_GELU_ERF ? 2u :
                op->op == GGML_OP_GELU ? 3u :
                op->op == GGML_OP_GELU_QUICK ? 3u :
                op->op == GGML_OP_TANH ? 4u :
                op->op == GGML_OP_EXP ? 5u : 6u;
            glUniform1ui(glGetUniformLocation(g_unary,"kind"),kind);
            glDispatchCompute((GLuint)((ggml_nelements(op)+63)/64),1,1); glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT); continue;
        }

        if (op->op == GGML_OP_SOFT_MAX) {
            auto * x=buffer_ctx(op->src[0]); auto * y=buffer_ctx(op); auto * m=op->src[1]?buffer_ctx(op->src[1]):nullptr;
            if(!x||!y||(op->src[1]&&!m))return GGML_STATUS_FAILED;
            glUseProgram(g_softmax); glBindBufferBase(GL_SHADER_STORAGE_BUFFER,0,x->buffer); glBindBufferBase(GL_SHADER_STORAGE_BUFFER,1,m?m->buffer:0); glBindBufferBase(GL_SHADER_STORAGE_BUFFER,2,y->buffer);
            auto U=[&](const char*n,GLuint v){glUniform1ui(glGetUniformLocation(g_softmax,n),v);};
            U("x_off",tensor_offset(op->src[0]->buffer,op->src[0])); U("y_off",tensor_offset(op->buffer,op)); U("m_off",m?(GLuint)tensor_offset(op->src[1]->buffer,op->src[1]):0);
            U("n0",(GLuint)op->ne[0]);U("n1",(GLuint)op->ne[1]);U("n2",(GLuint)op->ne[2]);U("n3",(GLuint)op->ne[3]);
            for(int d=0;d<4;++d){U((std::string("xs")+std::to_string(d)).c_str(),(GLuint)(op->src[0]->nb[d]/4));U((std::string("ys")+std::to_string(d)).c_str(),(GLuint)(op->nb[d]/4)); if(m)U((std::string("ms")+std::to_string(d)).c_str(),(GLuint)(op->src[1]->nb[d]/4));}
            U("has_mask",m?1u:0u); glUniform1f(glGetUniformLocation(g_softmax,"scale"),*reinterpret_cast<const float *>(op->op_params));
            glDispatchCompute((GLuint)(op->ne[1]*op->ne[2]*op->ne[3]),1,1); glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT); continue;
        }

        if (op->op == GGML_OP_ROPE) {
            auto * x=buffer_ctx(op->src[0]); auto * p=buffer_ctx(op->src[1]); auto * y=buffer_ctx(op);
            if(!x||!p||!y)return GGML_STATUS_FAILED;
            glUseProgram(g_rope); glBindBufferBase(GL_SHADER_STORAGE_BUFFER,0,x->buffer);glBindBufferBase(GL_SHADER_STORAGE_BUFFER,1,p->buffer);glBindBufferBase(GL_SHADER_STORAGE_BUFFER,2,y->buffer);
            auto U=[&](const char*n,GLuint v){glUniform1ui(glGetUniformLocation(g_rope,n),v);};
            U("x_off",tensor_offset(op->src[0]->buffer,op->src[0]));U("p_off",tensor_offset(op->src[1]->buffer,op->src[1]));U("y_off",tensor_offset(op->buffer,op));
            U("n0",(GLuint)op->ne[0]);U("n1",(GLuint)op->ne[1]);U("n2",(GLuint)op->ne[2]);U("n3",(GLuint)op->ne[3]); U("ps0",(GLuint)(op->src[1]->nb[0]/4));U("ps1",(GLuint)(op->src[1]->nb[1]/4));
            for(int d=0;d<4;++d){U((std::string("xs")+std::to_string(d)).c_str(),(GLuint)(op->src[0]->nb[d]/4));U((std::string("ys")+std::to_string(d)).c_str(),(GLuint)(op->nb[d]/4));}
            glUniform1i(glGetUniformLocation(g_rope,"n_dims"),op->op_params[1]);
            glUniform1i(glGetUniformLocation(g_rope,"mode"),op->op_params[2]);
            glUniform1i(glGetUniformLocation(g_rope,"n_ctx_orig"),op->op_params[4]);
            const float *fp=reinterpret_cast<const float*>(op->op_params);
            glUniform1f(glGetUniformLocation(g_rope,"freq_base"),fp[5]);glUniform1f(glGetUniformLocation(g_rope,"freq_scale"),fp[6]);glUniform1f(glGetUniformLocation(g_rope,"ext_factor"),fp[7]);glUniform1f(glGetUniformLocation(g_rope,"attn_factor"),fp[8]);glUniform1f(glGetUniformLocation(g_rope,"beta_fast"),fp[9]);glUniform1f(glGetUniformLocation(g_rope,"beta_slow"),fp[10]);
            const int32_t *ip=reinterpret_cast<const int32_t*>(op->op_params); for(int i=0;i<4;++i)glUniform1i(glGetUniformLocation(g_rope,(std::string("s")+std::to_string(i)).c_str()),ip[11+i]);
            glDispatchCompute((GLuint)((ggml_nelements(op)+63)/64),1,1); glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT); continue;
        }

        if (op->op == GGML_OP_MUL_MAT && op->src[0]->type == GGML_TYPE_F32) {
            auto * a=buffer_ctx(op->src[0]);auto * b=buffer_ctx(op->src[1]);auto * c=buffer_ctx(op);if(!a||!b||!c)return GGML_STATUS_FAILED;
            glUseProgram(g_matmul_f32);glBindBufferBase(GL_SHADER_STORAGE_BUFFER,0,a->buffer);glBindBufferBase(GL_SHADER_STORAGE_BUFFER,1,b->buffer);glBindBufferBase(GL_SHADER_STORAGE_BUFFER,2,c->buffer);
            auto U=[&](const char*n,GLuint v){glUniform1ui(glGetUniformLocation(g_matmul_f32,n),v);};
            U("K",(GLuint)op->src[0]->ne[0]);U("M",(GLuint)op->src[0]->ne[1]);U("N",(GLuint)op->src[1]->ne[1]);U("A2",(GLuint)op->src[0]->ne[2]);U("A3",(GLuint)op->src[0]->ne[3]);U("B2",(GLuint)op->src[1]->ne[2]);U("B3",(GLuint)op->src[1]->ne[3]);
            U("a_off",tensor_offset(op->src[0]->buffer,op->src[0]));U("b_off",tensor_offset(op->src[1]->buffer,op->src[1]));U("c_off",tensor_offset(op->buffer,op));
            const ggml_tensor * ts[3]={op->src[0],op->src[1],op}; const char *pre[3]={"a","b","c"};
            for(int z=0;z<3;++z)for(int d=0;d<4;++d)U((std::string(pre[z])+"s"+std::to_string(d)).c_str(),(GLuint)(ts[z]->nb[d]/4));
            uint64_t total=(uint64_t)op->ne[0]*op->ne[1]*op->ne[2]*op->ne[3]; glDispatchCompute((GLuint)((total+63)/64),1,1);glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);continue;
        }

        if (op->op == GGML_OP_SCALE) {
            auto * a=buffer_ctx(op->src[0]); auto * c=buffer_ctx(op);
            if(!a||!c)return GGML_STATUS_FAILED;
            glUseProgram(g_elementwise);
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER,0,a->buffer);
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER,1,a->buffer);
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER,2,c->buffer);
            glUniform1ui(glGetUniformLocation(g_elementwise,"a_off"),(GLuint)tensor_offset(op->src[0]->buffer,op->src[0]));
            glUniform1ui(glGetUniformLocation(g_elementwise,"b_off"),(GLuint)tensor_offset(op->src[0]->buffer,op->src[0]));
            glUniform1ui(glGetUniformLocation(g_elementwise,"c_off"),(GLuint)tensor_offset(op->buffer,op));
            glUniform1ui(glGetUniformLocation(g_elementwise,"n"),(GLuint)ggml_nelements(op));
            glUniform1ui(glGetUniformLocation(g_elementwise,"op"),2u);
            const float * params = reinterpret_cast<const float *>(op->op_params);
            glUniform1f(glGetUniformLocation(g_elementwise,"scale"), params[0]);
            glUniform1f(glGetUniformLocation(g_elementwise,"bias"), params[1]);
            glDispatchCompute((GLuint)((ggml_nelements(op)+63)/64),1,1);glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);continue;
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
            const GLuint gwidth = static_cast<GLuint>(op->src[3]->ne[0]);
            const GLuint ksnap = static_cast<GLuint>(op->op_params[0]);
            glUniform1ui(glGetUniformLocation(g_gdn, "S_v"), sv);
            glUniform1ui(glGetUniformLocation(g_gdn, "H_v"), hv);
            glUniform1ui(glGetUniformLocation(g_gdn, "H_k"), hk);
            glUniform1ui(glGetUniformLocation(g_gdn, "n_tokens"), nt);
            glUniform1ui(glGetUniformLocation(g_gdn, "n_seqs"), nsq);
            glUniform1ui(glGetUniformLocation(g_gdn, "K_snap"), ksnap);
            glUniform1ui(glGetUniformLocation(g_gdn, "g_width"), gwidth);
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
            const GLuint wtype = op->src[0]->type == GGML_TYPE_Q6_K ? 0u :
                (op->src[0]->type == GGML_TYPE_Q4_0 ? 1u : 2u);
            glUniform1ui(glGetUniformLocation(g_q6k, "w_type"), wtype);
            glUniform1ui(glGetUniformLocation(g_q6k, "cols"), static_cast<GLuint>(op->src[1]->ne[1]));
            const ggml_tensor * xt=op->src[1]; const ggml_tensor * yt=op;
            glUniform1ui(glGetUniformLocation(g_q6k,"x_s0"),(GLuint)(xt->nb[0]/4)); glUniform1ui(glGetUniformLocation(g_q6k,"x_s1"),(GLuint)(xt->nb[1]/4));
            glUniform1ui(glGetUniformLocation(g_q6k,"x_s2"),(GLuint)(xt->nb[2]/4)); glUniform1ui(glGetUniformLocation(g_q6k,"x_s3"),(GLuint)(xt->nb[3]/4));
            glUniform1ui(glGetUniformLocation(g_q6k,"y_s0"),(GLuint)(yt->nb[0]/4)); glUniform1ui(glGetUniformLocation(g_q6k,"y_s1"),(GLuint)(yt->nb[1]/4));
            glUniform1ui(glGetUniformLocation(g_q6k,"y_s2"),(GLuint)(yt->nb[2]/4)); glUniform1ui(glGetUniformLocation(g_q6k,"y_s3"),(GLuint)(yt->nb[3]/4));
            uint64_t total = static_cast<uint64_t>(op->src[0]->ne[1]) * static_cast<uint64_t>(op->src[1]->ne[1]) *
                static_cast<uint64_t>(op->src[1]->ne[2]) * static_cast<uint64_t>(op->src[1]->ne[3]);
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
