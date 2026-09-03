#include <jni.h>
#include <android/log.h>
#include <algorithm>
#include <atomic>
#include <cinttypes>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <mutex>
#include <signal.h>
#include <string>
#include <thread>
#include <unistd.h>
#include <vector>
#include <ucontext.h>
#include "llama.h"
#include "gguf.h"

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

static bool abort_callback(void *) { return g_stop.load(std::memory_order_relaxed); }
static int threads() { return std::clamp((int)std::max(1u, std::thread::hardware_concurrency()) - 2, 2, 4); }

static void append_native_trace(const char *text) {
    if (!text || !*text || g_native_trace_file.empty()) return;
    std::lock_guard<std::mutex> lock(g_native_marker_mutex);
    std::ofstream out(g_native_trace_file, std::ios::app);
    if (out.is_open()) {
        out << text;
        if (text[std::strlen(text) - 1] != '\n') out << '\n';
        out.flush();
    }
}

static const char *native_signal_name(int signal_number) {
    switch (signal_number) {
        case SIGSEGV: return "SIGSEGV";
        case SIGBUS: return "SIGBUS";
        case SIGABRT: return "SIGABRT";
        case SIGILL: return "SIGILL";
        case SIGFPE: return "SIGFPE";
        default: return "UNKNOWN";
    }
}

// Async-signal-safe hexadecimal writer. Do not replace this with streams, printf,
// allocation, mutexes, or other non-signal-safe APIs: this runs after a fatal fault.
static void native_write_hex(int fd, const char *label, uintptr_t value) {
    static constexpr char digits[] = "0123456789abcdef";
    char buffer[64];
    size_t pos = 0;
    while (label[pos] != '\0' && pos < sizeof(buffer) - 20) {
        buffer[pos] = label[pos];
        ++pos;
    }
    buffer[pos++] = '0';
    buffer[pos++] = 'x';
    bool started = false;
    for (int shift = (int)(sizeof(uintptr_t) * 8 - 4); shift >= 0; shift -= 4) {
        const unsigned digit = (unsigned)((value >> shift) & 0xfu);
        if (digit != 0 || started || shift == 0) {
            buffer[pos++] = digits[digit];
            started = true;
        }
    }
    buffer[pos++] = '\n';
    (void)write(fd, buffer, pos);
}

static void native_fatal_signal_handler(int signal_number, siginfo_t *info, void *raw_context) {
    if (g_native_fatal_fd >= 0) {
        const char prefix[] = "NATIVE_FATAL_SIGNAL=";
        const char name_prefix[] = "NATIVE_FATAL_SIGNAL_NAME=";
        const char pc_prefix[] = "NATIVE_FATAL_PC=";
        const char lr_prefix[] = "NATIVE_FATAL_LR=";
        const char addr_prefix[] = "NATIVE_FATAL_FAULT_ADDR=";
        const char newline = '\n';

        char value[16];
        int n = 0;
        int value_copy = signal_number;
        if (value_copy == 0) value[n++] = '0';
        else {
            char reverse[16];
            int r = 0;
            while (value_copy > 0 && r < (int)sizeof(reverse)) {
                reverse[r++] = (char)('0' + (value_copy % 10));
                value_copy /= 10;
            }
            while (r > 0) value[n++] = reverse[--r];
        }
        (void)write(g_native_fatal_fd, prefix, sizeof(prefix) - 1);
        (void)write(g_native_fatal_fd, value, (size_t)n);
        (void)write(g_native_fatal_fd, &newline, 1);
        (void)write(g_native_fatal_fd, name_prefix, sizeof(name_prefix) - 1);
        const char *name = native_signal_name(signal_number);
        (void)write(g_native_fatal_fd, name, std::strlen(name));
        (void)write(g_native_fatal_fd, &newline, 1);

        uintptr_t pc = 0;
        uintptr_t lr = 0;
#if defined(__aarch64__)
        if (raw_context) {
            const ucontext_t *context = static_cast<const ucontext_t *>(raw_context);
            pc = (uintptr_t)context->uc_mcontext.pc;
            lr = (uintptr_t)context->uc_mcontext.regs[30];
        }
#endif
        native_write_hex(g_native_fatal_fd, pc_prefix, pc);
        native_write_hex(g_native_fatal_fd, lr_prefix, lr);
        native_write_hex(g_native_fatal_fd, addr_prefix,
                         info ? (uintptr_t)info->si_addr : (uintptr_t)0);
        (void)fsync(g_native_fatal_fd);
    }
    signal(signal_number, SIG_DFL);
    raise(signal_number);
}

static void install_native_fatal_handlers() {
    struct sigaction action{};
    sigemptyset(&action.sa_mask);
    action.sa_sigaction = native_fatal_signal_handler;
    action.sa_flags = SA_SIGINFO;
    sigaction(SIGSEGV, &action, nullptr);
    sigaction(SIGBUS, &action, nullptr);
    sigaction(SIGABRT, &action, nullptr);
    sigaction(SIGILL, &action, nullptr);
    sigaction(SIGFPE, &action, nullptr);
}

static void checkpoint(const char *event) {
    LOGI("NATIVE_CHECKPOINT %s", event);
    __android_log_write(ANDROID_LOG_INFO, LOG_TAG, event);
    append_native_trace(event);
    if (!g_native_marker_file.empty()) {
        std::lock_guard<std::mutex> lock(g_native_marker_mutex);
        std::ofstream out(g_native_marker_file, std::ios::trunc);
        if (out.is_open()) {
            out << event << '\n';
            out.flush();
        }
    }
}

static void free_all() {
    if (g_sampler) { llama_sampler_free(g_sampler); g_sampler = nullptr; }
    if (g_context) { llama_free(g_context); g_context = nullptr; }
    if (g_model) { llama_model_free(g_model); g_model = nullptr; }
    g_gpu = false;
}

static std::string gguf_preflight(const char *path) {
    gguf_init_params params{};
    params.no_alloc = true;
    params.ctx = nullptr;

    gguf_context *ctx = gguf_init_from_file(path, params);
    if (!ctx) return std::string("GGUF_PREFLIGHT_FAILED\npath=") + (path ? path : "<null>");

    const int64_t n_tensors = gguf_get_n_tensors(ctx);
    const int64_t n_kv = gguf_get_n_kv(ctx);
    const uint32_t version = gguf_get_version(ctx);
    const size_t data_offset = gguf_get_data_offset(ctx);
    const size_t alignment = gguf_get_alignment(ctx);

    std::string architecture = "<missing>";
    const int64_t arch_key = gguf_find_key(ctx, "general.architecture");
    if (arch_key >= 0 && gguf_get_kv_type(ctx, arch_key) == GGUF_TYPE_STRING) {
        const char *value = gguf_get_val_str(ctx, arch_key);
        if (value) architecture = value;
    }

    uint64_t total_tensor_bytes = 0;
    uint64_t largest_tensor_bytes = 0;
    int64_t largest_tensor_id = -1;
    uint64_t tensor_data_end = data_offset;

    for (int64_t i = 0; i < n_tensors; ++i) {
        const size_t size = gguf_get_tensor_size(ctx, i);
        const size_t offset = gguf_get_tensor_offset(ctx, i);
        if (UINT64_MAX - total_tensor_bytes < (uint64_t)size) total_tensor_bytes = UINT64_MAX;
        else total_tensor_bytes += (uint64_t)size;
        const uint64_t end = (uint64_t)data_offset + (uint64_t)offset + (uint64_t)size;
        tensor_data_end = std::max(tensor_data_end, end);
        if ((uint64_t)size > largest_tensor_bytes) {
            largest_tensor_bytes = (uint64_t)size;
            largest_tensor_id = i;
        }
    }

    std::string largest_name = "<none>";
    int largest_type = -1;
    if (largest_tensor_id >= 0) {
        const char *name = gguf_get_tensor_name(ctx, largest_tensor_id);
        if (name) largest_name = name;
        largest_type = (int)gguf_get_tensor_type(ctx, largest_tensor_id);
    }

    std::string result;
    result.reserve(2048);
    result += "GGUF_PREFLIGHT_OK\n";
    result += "path=" + std::string(path ? path : "<null>") + "\n";
    result += "version=" + std::to_string(version) + "\n";
    result += "architecture=" + architecture + "\n";
    result += "kv_count=" + std::to_string(n_kv) + "\n";
    result += "tensor_count=" + std::to_string(n_tensors) + "\n";
    result += "alignment=" + std::to_string(alignment) + "\n";
    result += "data_offset=" + std::to_string(data_offset) + "\n";
    result += "total_tensor_bytes=" + std::to_string(total_tensor_bytes) + "\n";
    result += "total_tensor_mib=" + std::to_string((double)total_tensor_bytes / (1024.0 * 1024.0)) + "\n";
    result += "largest_tensor_name=" + largest_name + "\n";
    result += "largest_tensor_type=" + std::to_string(largest_type) + "\n";
    result += "largest_tensor_bytes=" + std::to_string(largest_tensor_bytes) + "\n";
    result += "largest_tensor_mib=" + std::to_string((double)largest_tensor_bytes / (1024.0 * 1024.0)) + "\n";
    result += "tensor_data_end=" + std::to_string(tensor_data_end) + "\n";

    gguf_free(ctx);
    return result;
}

extern "C" JNIEXPORT void JNICALL
Java_com_woogit_aicore_runtime_android_NativeLlamaCpp_nativeInit(JNIEnv *env, jclass) {
    LOGI("ACTIVATION_NATIVE_INIT_STARTED");

    jclass system_class = env->FindClass("java/lang/System");
    if (system_class) {
        jmethodID get_property = env->GetStaticMethodID(system_class, "getProperty", "(Ljava/lang/String;)Ljava/lang/String;");
        if (get_property) {
            jstring key = env->NewStringUTF("java.io.tmpdir");
            jstring tmp_dir = static_cast<jstring>(env->CallStaticObjectMethod(system_class, get_property, key));
            env->DeleteLocalRef(key);
            if (tmp_dir) {
                const char *dir = env->GetStringUTFChars(tmp_dir, nullptr);
                if (dir) {
                    const std::string base(dir);
                    g_native_marker_file = base + "/ai-chat-last-native-event.txt";
                    g_native_trace_file = base + "/ai-chat-native-trace.txt";
                    g_native_fatal_fd = open((base + "/ai-chat-native-trace.txt").c_str(), O_WRONLY | O_CREAT | O_APPEND, 0600);
                    env->ReleaseStringUTFChars(tmp_dir, dir);
                }
                env->DeleteLocalRef(tmp_dir);
            }
        }
        env->DeleteLocalRef(system_class);
    }

    if (!g_native_marker_file.empty()) {
        LOGI("NATIVE_DIAGNOSTICS_MARKER=%s", g_native_marker_file.c_str());
        LOGI("NATIVE_DIAGNOSTICS_TRACE=%s", g_native_trace_file.c_str());
        append_native_trace("===== NATIVE INIT =====");
        install_native_fatal_handlers();
    } else LOGW("NATIVE_DIAGNOSTICS_MARKER_UNAVAILABLE");

    llama_log_set([](enum ggml_log_level level, const char *text, void *) {
        const char *message = text ? text : "<null>";
        const int priority = level >= GGML_LOG_LEVEL_ERROR ? ANDROID_LOG_ERROR :
                             level >= GGML_LOG_LEVEL_WARN ? ANDROID_LOG_WARN : ANDROID_LOG_INFO;
        __android_log_print(priority, LOG_TAG, "%s", message);
        append_native_trace(message);
    }, nullptr);
    LOGI("ACTIVATION_BACKEND_LOAD_ALL_STARTED");
    append_native_trace("ACTIVATION_BACKEND_LOAD_ALL_STARTED");
    ggml_backend_load_all();
    LOGI("ACTIVATION_BACKEND_LOAD_ALL_RETURNED");
    append_native_trace("ACTIVATION_BACKEND_LOAD_ALL_RETURNED");
    LOGI("ACTIVATION_LLAMA_BACKEND_INIT_STARTED");
    append_native_trace("ACTIVATION_LLAMA_BACKEND_INIT_STARTED");
    llama_backend_init();
    LOGI("ACTIVATION_LLAMA_BACKEND_INIT_RETURNED");
    append_native_trace("ACTIVATION_LLAMA_BACKEND_INIT_RETURNED");
}

extern "C" JNIEXPORT jint JNICALL
Java_com_woogit_aicore_runtime_android_NativeLlamaCpp_nativeLoad(JNIEnv *env, jclass, jstring jpath, jint ctx_len, jint gpu_layers) {
    checkpoint("NATIVE_LOAD_STARTED");
    LOGI("ACTIVATION_NATIVE_LOAD_STARTED ctx_len=%d gpu_layers=%d", (int)ctx_len, (int)gpu_layers);
    append_native_trace((std::string("ACTIVATION_NATIVE_LOAD_STARTED ctx_len=") + std::to_string((int)ctx_len) + " gpu_layers=" + std::to_string((int)gpu_layers)).c_str());
    checkpoint("FREE_OLD_RUNTIME_STARTED");
    free_all();
    checkpoint("FREE_OLD_RUNTIME_RETURNED");
    g_stop.store(false);

    const char *path = env->GetStringUTFChars(jpath, nullptr);
    if (!path) { checkpoint("PATH_UTF8_FAILED"); return 3; }

    const std::string preflight = gguf_preflight(path);
    checkpoint(preflight.c_str());

    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = gpu_layers;
    checkpoint((std::string("MODEL_LOAD_PARAMS gpu_layers=") + std::to_string((int)mp.n_gpu_layers) +
        " vocab_only=" + (mp.vocab_only ? "1" : "0")).c_str());
    LOGI("ACTIVATION_MODEL_LOAD_STARTED gpu_layers=%d", (int)mp.n_gpu_layers);
    checkpoint(gpu_layers == 0 ? "MODEL_LOAD_STARTED gpu_layers=0" : "MODEL_LOAD_STARTED gpu_layers=GPU");
    g_model = llama_model_load_from_file(path, mp);
    LOGI("ACTIVATION_MODEL_LOAD_RETURNED success=%d", g_model != nullptr ? 1 : 0);
    checkpoint(g_model ? "MODEL_LOAD_RETURNED_SUCCESS" : "MODEL_LOAD_RETURNED_FAILED");
    env->ReleaseStringUTFChars(jpath, path);

    if (!g_model && gpu_layers != 0) {
        LOGW("ACTIVATION_GPU_MODEL_LOAD_FAILED_STARTING_CPU_FALLBACK");
        checkpoint("GPU_MODEL_LOAD_FAILED_CPU_FALLBACK_STARTED");
        path = env->GetStringUTFChars(jpath, nullptr);
        if (!path) { checkpoint("CPU_FALLBACK_PATH_UTF8_FAILED"); return 3; }
        mp = llama_model_default_params();
        mp.n_gpu_layers = 0;
        checkpoint("CPU_MODEL_LOAD_STARTED");
        g_model = llama_model_load_from_file(path, mp);
        checkpoint(g_model ? "CPU_MODEL_LOAD_RETURNED_SUCCESS" : "CPU_MODEL_LOAD_RETURNED_FAILED");
        env->ReleaseStringUTFChars(jpath, path);
        g_gpu = false;
    } else g_gpu = g_model != nullptr && gpu_layers != 0;

    if (!g_model) { checkpoint("MODEL_LOAD_FAILED"); return 1; }

    const int trained = llama_model_n_ctx_train(g_model);
    const int requested = ctx_len > 0 ? ctx_len : 4096;
    const int effective = std::max(1, std::min(requested, trained));
    LOGI("ACTIVATION_MODEL_READY trained_ctx=%d requested_ctx=%d effective_ctx=%d backend=%s", trained, requested, effective, g_gpu ? "Vulkan" : "CPU");
    checkpoint("MODEL_READY");

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = (uint32_t)effective;
    cp.n_batch = std::min<uint32_t>(cp.n_ctx, 512);
    cp.n_ubatch = cp.n_batch;
    cp.n_threads = threads();
    cp.n_threads_batch = threads();
    checkpoint("CONTEXT_INIT_STARTED");
    g_context = llama_init_from_model(g_model, cp);
    checkpoint(g_context ? "CONTEXT_INIT_RETURNED_SUCCESS" : "CONTEXT_INIT_RETURNED_FAILED");
    if (!g_context) { checkpoint("CONTEXT_INIT_FAILED"); free_all(); return 2; }
    llama_set_abort_callback(g_context, abort_callback, nullptr);
    checkpoint("NATIVE_LOAD_COMPLETED");
    return 0;
}

extern "C" JNIEXPORT void JNICALL
Java_com_woogit_aicore_runtime_android_NativeLlamaCpp_nativeStop(JNIEnv *, jclass) { g_stop.store(true); }

extern "C" JNIEXPORT void JNICALL
Java_com_woogit_aicore_runtime_android_NativeLlamaCpp_nativeUnload(JNIEnv *, jclass) { g_stop.store(true); free_all(); }

extern "C" JNIEXPORT jstring JNICALL
Java_com_woogit_aicore_runtime_android_NativeLlamaCpp_nativeRuntimeInfo(JNIEnv *env, jclass) {
    const std::string value = std::string(g_gpu ? "Hybrid(CPU+Vulkan)" : "CPU/NEON") + "; llama.cpp=c5fc7e34885ba31217e330809437afa993d27745";
    return env->NewStringUTF(value.c_str());
}

extern "C" JNIEXPORT jint JNICALL
Java_com_woogit_aicore_runtime_android_NativeLlamaCpp_nativeContextLength(JNIEnv *, jclass) {
    return g_context ? (jint)llama_n_ctx(g_context) : 0;
}
