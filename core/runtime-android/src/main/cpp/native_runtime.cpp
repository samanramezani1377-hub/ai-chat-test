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
static std::mutex g_backend_init_mutex;
static bool g_backend_initialized = false;
static bool g_gpu_backend_loaded = false;

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

static void native_write_text(int fd, const char *text, size_t len) {
    while (len) {
        const ssize_t written = write(fd, text, len);
        if (written <= 0) return;
        text += written;
        len -= (size_t)written;
    }
}

static void native_write_hex(int fd, const char *label, uintptr_t value) {
    static constexpr char digits[] = "0123456789abcdef";
    char buffer[128];
    size_t pos = 0;
    while (label[pos] && pos < sizeof(buffer) - 20) buffer[pos] = label[pos], ++pos;
    buffer[pos++] = '0'; buffer[pos++] = 'x';
    bool started = false;
    for (int shift = (int)(sizeof(uintptr_t) * 8 - 4); shift >= 0; shift -= 4) {
        const unsigned digit = (unsigned)((value >> shift) & 0xfu);
        if (digit || started || shift == 0) { buffer[pos++] = digits[digit]; started = true; }
    }
    buffer[pos++] = '\n';
    native_write_text(fd, buffer, pos);
}

static void native_fatal_signal_handler(int signal_number, siginfo_t *info, void *raw_context) {
    if (g_native_fatal_fd >= 0) {
        const char prefix[] = "NATIVE_FATAL_SIGNAL=";
        const char name_prefix[] = "NATIVE_FATAL_SIGNAL_NAME=";
        const char code_prefix[] = "NATIVE_FATAL_SI_CODE=";
        const char pc_prefix[] = "NATIVE_FATAL_PC=";
        const char lr_prefix[] = "NATIVE_FATAL_LR=";
        const char addr_prefix[] = "NATIVE_FATAL_FAULT_ADDR=";
        char value[24];
        int n = 0;
        int v = signal_number;
        if (v == 0) value[n++] = '0';
        else {
            char reverse[24]; int r = 0;
            while (v > 0 && r < (int)sizeof(reverse)) { reverse[r++] = (char)('0' + v % 10); v /= 10; }
            while (r) value[n++] = reverse[--r];
        }
        native_write_text(g_native_fatal_fd, prefix, sizeof(prefix)-1);
        native_write_text(g_native_fatal_fd, value, (size_t)n); native_write_text(g_native_fatal_fd, "\n", 1);
        const char *name = native_signal_name(signal_number);
        native_write_text(g_native_fatal_fd, name_prefix, sizeof(name_prefix)-1);
        native_write_text(g_native_fatal_fd, name, std::strlen(name)); native_write_text(g_native_fatal_fd, "\n", 1);
        n = snprintf(value, sizeof(value), "%d", info ? info->si_code : 0);
        native_write_text(g_native_fatal_fd, code_prefix, sizeof(code_prefix)-1);
        if (n > 0) native_write_text(g_native_fatal_fd, value, (size_t)n);
        native_write_text(g_native_fatal_fd, "\n", 1);
        uintptr_t pc = 0, lr = 0;
#if defined(__aarch64__)
        if (raw_context) {
            const ucontext_t *context = static_cast<const ucontext_t *>(raw_context);
            pc = (uintptr_t)context->uc_mcontext.pc;
            lr = (uintptr_t)context->uc_mcontext.regs[30];
        }
#endif
        native_write_hex(g_native_fatal_fd, pc_prefix, pc);
        native_write_hex(g_native_fatal_fd, lr_prefix, lr);
        native_write_hex(g_native_fatal_fd, addr_prefix, info ? (uintptr_t)info->si_addr : 0);
        fsync(g_native_fatal_fd);
    }
    _exit(128 + signal_number);
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
    append_native_trace(event);
    if (!g_native_marker_file.empty()) {
        std::lock_guard<std::mutex> lock(g_native_marker_mutex);
        std::ofstream out(g_native_marker_file, std::ios::trunc);
        if (out.is_open()) { out << event << '\n'; out.flush(); }
    }
}

static void free_all() {
    clear_android_generation_cache();
    if (g_sampler) { llama_sampler_free(g_sampler); g_sampler = nullptr; }
    if (g_context) { llama_free(g_context); g_context = nullptr; }
    if (g_model) { llama_model_free(g_model); g_model = nullptr; }
    g_gpu = false;
}

static std::string gguf_preflight(const char *path) {
    gguf_init_params params{}; params.no_alloc = true; params.ctx = nullptr;
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
        const char *value = gguf_get_val_str(ctx, arch_key); if (value) architecture = value;
    }
    uint64_t total_tensor_bytes = 0, largest_tensor_bytes = 0, tensor_data_end = data_offset;
    int64_t largest_tensor_id = -1;
    for (int64_t i = 0; i < n_tensors; ++i) {
        const uint64_t size = gguf_get_tensor_size(ctx, i);
        const uint64_t offset = gguf_get_tensor_offset(ctx, i);
        total_tensor_bytes = (UINT64_MAX - total_tensor_bytes < size) ? UINT64_MAX : total_tensor_bytes + size;
        tensor_data_end = std::max(tensor_data_end, (uint64_t)data_offset + offset + size);
        if (size > largest_tensor_bytes) { largest_tensor_bytes = size; largest_tensor_id = i; }
    }
    std::string largest_name = "<none>"; int largest_type = -1;
    if (largest_tensor_id >= 0) {
        const char *name = gguf_get_tensor_name(ctx, largest_tensor_id); if (name) largest_name = name;
        largest_type = (int)gguf_get_tensor_type(ctx, largest_tensor_id);
    }
    std::string result = "GGUF_PREFLIGHT_OK\n";
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
    gguf_free(ctx); return result;
}

extern "C" JNIEXPORT void JNICALL
Java_com_woogit_aicore_runtime_android_NativeLlamaCpp_nativeInit(JNIEnv *env, jclass, jboolean enable_gpu) {
    LOGI("ACTIVATION_NATIVE_INIT_STARTED gpu=%d", enable_gpu ? 1 : 0);
    if (!g_native_marker_file.empty()) {
        append_native_trace(enable_gpu ? "===== NATIVE INIT GPU REQUESTED =====" : "===== NATIVE INIT CPU ONLY =====");
        install_native_fatal_handlers();
    }
    if (g_native_marker_file.empty()) {
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
            append_native_trace(enable_gpu ? "===== NATIVE INIT GPU REQUESTED =====" : "===== NATIVE INIT CPU ONLY =====");
            install_native_fatal_handlers();
        }
    }
    llama_log_set([](enum ggml_log_level level, const char *text, void *) {
        const char *message = text ? text : "<null>";
        const int priority = level >= GGML_LOG_LEVEL_ERROR ? ANDROID_LOG_ERROR : level >= GGML_LOG_LEVEL_WARN ? ANDROID_LOG_WARN : ANDROID_LOG_INFO;
        __android_log_print(priority, LOG_TAG, "%s", message);
        append_native_trace(message);
    }, nullptr);

    std::lock_guard<std::mutex> lock(g_backend_init_mutex);
    if (!g_backend_initialized) {
        unsetenv("GGML_DISABLE_OPENCL");
        append_native_trace("OPENCL_BACKEND_LOAD_ALL_STARTED");
        ggml_backend_load_all();
        append_native_trace("OPENCL_BACKEND_LOAD_ALL_RETURNED");
        g_gpu_backend_loaded = true;
        append_native_trace("ACTIVATION_LLAMA_BACKEND_INIT_STARTED");
        llama_backend_init();
        append_native_trace("ACTIVATION_LLAMA_BACKEND_INIT_RETURNED");
        g_backend_initialized = true;
    } else {
        append_native_trace("OPENCL_BACKEND_ALREADY_INITIALIZED");
    }
}

static llama_model *load_model_android(const char *path, llama_model_params mp, bool gpu) {
    // Avoid the mmap -> OpenCL host-pointer import path on Android. Mobile UMA OpenCL
    // drivers can fail inside buffer_from_host_ptr while the model is being initialized.
    // LLAMA_LOAD_MODE_NONE keeps bounded file reads and normal backend allocations, so
    // GPU layers remain enabled without relying on the fragile mmap buffer import.
    mp.load_mode = LLAMA_LOAD_MODE_NONE;
    mp.check_tensors = false;
    checkpoint(gpu ? "ANDROID_MODEL_LOAD_POLICY_GPU_STAGED" : "ANDROID_MODEL_LOAD_POLICY_CPU_STAGED");
    checkpoint((std::string("ANDROID_MODEL_LOAD_PARAMS load_mode=") + llama_load_mode_name(mp.load_mode) +
        " check_tensors=" + (mp.check_tensors ? "1" : "0")).c_str());
    return llama_model_load_from_file(path, mp);
}

extern "C" JNIEXPORT jint JNICALL
Java_com_woogit_aicore_runtime_android_NativeLlamaCpp_nativeLoad(JNIEnv *env, jclass, jstring jpath, jint ctx_len, jint gpu_layers) {
    checkpoint("NATIVE_LOAD_STARTED");
    append_native_trace((std::string("ACTIVATION_NATIVE_LOAD_STARTED ctx_len=") + std::to_string((int)ctx_len) + " gpu_layers=" + std::to_string((int)gpu_layers)).c_str());
    checkpoint("FREE_OLD_RUNTIME_STARTED"); free_all(); checkpoint("FREE_OLD_RUNTIME_RETURNED"); g_stop.store(false);
    const char *path = env->GetStringUTFChars(jpath, nullptr);
    if (!path) { checkpoint("PATH_UTF8_FAILED"); return 3; }
    const std::string preflight = gguf_preflight(path); checkpoint(preflight.c_str());
    if (gpu_layers <= 0) {
        checkpoint("OPENCL_GPU_ONLY_REJECTED_INVALID_GPU_LAYERS");
        env->ReleaseStringUTFChars(jpath, path);
        return 4;
    }
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = gpu_layers;
    mp.load_mode = LLAMA_LOAD_MODE_NONE;
    mp.check_tensors = false;
    checkpoint("OPENCL_GPU_DEVICE_SELECTION");
    checkpoint("OPENCL_GPU_ONLY_LOAD_MODE_STAGED");
    checkpoint("OPENCL_GPU_ONLY_CHECK_TENSORS_DISABLED");
    checkpoint((std::string("MODEL_LOAD_PARAMS gpu_layers=") + std::to_string((int)mp.n_gpu_layers) +
        " load_mode=" + llama_load_mode_name(mp.load_mode) +
        " check_tensors=" + (mp.check_tensors ? "1" : "0")).c_str());
    checkpoint("MODEL_LOAD_STARTED gpu_layers=OPENCL");
    g_model = load_model_android(path, mp, true);
    checkpoint(g_model ? "MODEL_LOAD_RETURNED_SUCCESS" : "MODEL_LOAD_RETURNED_FAILED");
    env->ReleaseStringUTFChars(jpath, path);
    if (!g_model) {
        checkpoint("OPENCL_GPU_ONLY_MODEL_LOAD_FAILED_NO_CPU_FALLBACK");
        g_gpu = false;
        return 1;
    }
    g_gpu = true;
    if (!g_model) { checkpoint("MODEL_LOAD_FAILED"); return 1; }
    const int trained = llama_model_n_ctx_train(g_model);
    const int requested = ctx_len > 0 ? ctx_len : 4096;
    const int effective = std::max(1, std::min(requested, trained));
    checkpoint((std::string("MODEL_READY trained_ctx=") + std::to_string(trained) + " effective_ctx=" + std::to_string(effective)).c_str());
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = (uint32_t)effective;
    // Keep graph/batch working sets small on mobile GPUs. The previous 512-token
    // batch is unnecessarily large for interactive single-message generation.
    cp.n_batch = std::min<uint32_t>(cp.n_ctx, 128);
    cp.n_ubatch = cp.n_batch;
    // This runtime serves exactly one interactive generation sequence. Keep the
    // recurrent/hybrid state single-sequence as well; LFM2/LFM2.5 uses that state.
    cp.n_seq_max = 1;
    cp.n_threads = threads(); cp.n_threads_batch = threads();
    checkpoint("CONTEXT_INIT_STARTED"); g_context = llama_init_from_model(g_model, cp);
    checkpoint(g_context ? "CONTEXT_INIT_RETURNED_SUCCESS" : "CONTEXT_INIT_RETURNED_FAILED");
    if (!g_context) { checkpoint("CONTEXT_INIT_FAILED"); free_all(); return 2; }
    llama_set_abort_callback(g_context, abort_callback, nullptr); checkpoint("NATIVE_LOAD_COMPLETED"); return 0;
}

extern "C" JNIEXPORT void JNICALL
Java_com_woogit_aicore_runtime_android_NativeLlamaCpp_nativeStop(JNIEnv *, jclass) { g_stop.store(true); }
extern "C" JNIEXPORT void JNICALL
Java_com_woogit_aicore_runtime_android_NativeLlamaCpp_nativeUnload(JNIEnv *, jclass) { g_stop.store(true); free_all(); }
extern "C" JNIEXPORT jstring JNICALL
Java_com_woogit_aicore_runtime_android_NativeLlamaCpp_nativeRuntimeInfo(JNIEnv *env, jclass) {
    const std::string value = std::string(g_gpu ? "OpenCL-GPU-ONLY" : "OpenCL-UNAVAILABLE") + "; llama.cpp=" + AI_CHAT_LLAMA_CPP_SHA;
    return env->NewStringUTF(value.c_str());
}

extern "C" JNIEXPORT jint JNICALL
Java_com_woogit_aicore_runtime_android_NativeLlamaCpp_nativeContextLength(JNIEnv *, jclass) {
    if (g_context) {
        return (jint) llama_n_ctx(g_context);
    }
    if (g_model) {
        return (jint) llama_model_n_ctx_train(g_model);
    }
    return 0;
}
