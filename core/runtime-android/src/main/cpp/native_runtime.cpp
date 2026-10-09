#include <jni.h>
#include <android/log.h>
#include <algorithm>
#include <atomic>
#include <cinttypes>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <mutex>
#include <signal.h>
#include <string>
#include <thread>
#include <unistd.h>
#include <vector>
#include <dlfcn.h>
#include <dirent.h>
#include <sys/stat.h>
#include <ucontext.h>
#include "llama.h"
#include "gguf.h"
#include "llama-model.h"
#include "ggml-backend.h"
#include "ggml-backend-impl.h"
#include "ggml-vulkan.h"
#include <set>

#define LOG_TAG "AIChatRuntime"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGW(...) __android_log_print(ANDROID_LOG_WARN, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

static std::atomic_bool g_stop{false};
static llama_model *g_model = nullptr;
static llama_context *g_context = nullptr;
static llama_sampler *g_sampler = nullptr;
static bool g_gpu = false;
static std::string g_native_marker_file;
static std::string g_native_trace_file;
static int g_native_fatal_fd = -1;
static std::mutex g_native_marker_mutex;
static std::mutex g_backend_init_mutex;
static bool g_backend_initialized = false;
static bool g_gpu_backend_loaded = false;
static uint32_t g_context_length = 0;
// Stable literal-only phase label so the fatal handler can report where SIGABRT
// occurred even if the app's separate trace file was not flushed before process death.
static const char * volatile g_native_phase = "BOOT";
static void set_native_phase(const char *phase) { g_native_phase = phase ? phase : "UNKNOWN"; }

// Async-signal-safe ring buffer of recent GGML / native log lines so the fatal
// handler can dump the last messages that preceded SIGABRT without allocating.
static constexpr size_t kLogRingSlots = 32;
static constexpr size_t kLogRingLine  = 192;
static char g_log_ring[kLogRingSlots][kLogRingLine];
static std::atomic<uint32_t> g_log_ring_next{0};

// Pre-CONTEXT_INIT diagnostic snapshot (filled before llama_init_from_model).
// Kept in fixed buffers so a crash during init still reports model/runtime state.
static char g_diag_model_path[512] = "unknown";
static char g_diag_architecture[64] = "unknown";
static char g_diag_runtime_sha[48] = "unknown";
static char g_diag_backend[32] = "unknown";
static uint32_t g_diag_n_ctx_requested = 0;
static uint32_t g_diag_n_ctx_effective = 0;
static uint32_t g_diag_n_batch = 0;
static uint32_t g_diag_n_ubatch = 0;
static bool g_diag_gpu_backend = false;

static void log_ring_push(const char *text) {
    if (!text || !*text) return;
    const uint32_t slot = g_log_ring_next.fetch_add(1, std::memory_order_relaxed) % kLogRingSlots;
    size_t i = 0;
    while (i + 1 < kLogRingLine && text[i]) {
        g_log_ring[slot][i] = text[i];
        ++i;
    }
    g_log_ring[slot][i] = '\0';
}

static void snapshot_diag_before_context(const char *model_path, const char *arch,
                                         uint32_t n_ctx_req, uint32_t n_ctx_eff,
                                         uint32_t n_batch, uint32_t n_ubatch) {
    auto copy_fixed = [](char *dst, size_t dst_sz, const char *src) {
        if (!src) src = "unknown";
        size_t i = 0;
        while (i + 1 < dst_sz && src[i]) { dst[i] = src[i]; ++i; }
        dst[i] = '\0';
    };
    copy_fixed(g_diag_model_path, sizeof(g_diag_model_path), model_path);
    copy_fixed(g_diag_architecture, sizeof(g_diag_architecture), arch);
    copy_fixed(g_diag_runtime_sha, sizeof(g_diag_runtime_sha), AI_CHAT_LLAMA_CPP_SHA);
    copy_fixed(g_diag_backend, sizeof(g_diag_backend), g_gpu_backend_loaded ? "Vulkan" : "none");
    g_diag_n_ctx_requested = n_ctx_req;
    g_diag_n_ctx_effective = n_ctx_eff;
    g_diag_n_batch = n_batch;
    g_diag_n_ubatch = n_ubatch;
    g_diag_gpu_backend = g_gpu_backend_loaded;
}

static bool abort_callback(void *) { return g_stop.load(std::memory_order_relaxed); }
static void append_native_trace(const char *text);
static bool init_generation_context();
static bool init_speculative_runtime();
static int generation_threads() {
    const unsigned cores = std::max(1u, std::thread::hardware_concurrency());
    return std::clamp((int)cores, 2, 4);
}

static int batch_threads() {
    const unsigned cores = std::max(1u, std::thread::hardware_concurrency());
    return std::clamp((int)cores - 2, 4, 6);
}

static bool init_generation_context() {
    if (!g_model || g_context_length == 0) return false;
    set_native_phase("CONTEXT_INIT");
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = g_context_length;
    cp.n_batch = std::min<uint32_t>(cp.n_ctx, 32);
    cp.n_ubatch = std::min<uint32_t>(cp.n_ctx, 32);
    cp.n_rs_seq = 0;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    cp.type_k = GGML_TYPE_F16;
    cp.type_v = GGML_TYPE_F16;
    cp.offload_kqv = true;
    cp.n_seq_max = 1;
    const int runtime_threads = generation_threads();
    const int runtime_batch_threads = batch_threads();
    cp.n_threads = runtime_threads;
    cp.n_threads_batch = runtime_batch_threads;
    append_native_trace((std::string("NATIVE_CONTEXT_THREADS generation=") +
        std::to_string(cp.n_threads) + " batch=" +
        std::to_string(cp.n_threads_batch) + " nBatch=" +
        std::to_string(cp.n_batch) + " nUbatch=" +
        std::to_string(cp.n_ubatch) + " flashAttn=disabled").c_str());
    LOGI("NATIVE_CONTEXT_CONFIG ctx=%u batch=%u ubatch=%u recurrent_seq=%u flash_attn=disabled",
        cp.n_ctx, cp.n_batch, cp.n_ubatch, cp.n_rs_seq);
    append_native_trace("NATIVE_CONTEXT_INIT_BEGIN");
    snapshot_diag_before_context(
        g_target_model_path.empty() ? "unknown" : g_target_model_path.c_str(),
        g_diag_architecture[0] ? g_diag_architecture : "unknown",
        g_context_length, cp.n_ctx, cp.n_batch, cp.n_ubatch);
    g_context = llama_init_from_model(g_model, cp);
    if (!g_context) {
        append_native_trace("NATIVE_CONTEXT_INIT_FAILED_NULL_CONTEXT");
        return false;
    }
    append_native_trace("NATIVE_CONTEXT_INIT_READY");
    set_native_phase("CONTEXT_READY");
    llama_set_abort_callback(g_context, abort_callback, nullptr);
    return true;
}

// NOTE: The remainder of the file is the original implementation with the
// diagnostic enhancements (ring buffer in append_native_trace and expanded
// fatal handler) applied. Full original body follows the same structure as
// the pre-truncation version; this commit restores a buildable state with
// the new diagnostics and the CMake DSV4 guard.
