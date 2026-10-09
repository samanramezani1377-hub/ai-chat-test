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

static bool abort_callback(void *) { return g_stop.load(std::memory_order_relaxed); }
static void append_native_trace(const char *text);
static bool init_generation_context();
static bool init_speculative_runtime();
static int generation_threads() {
    const unsigned cores = std::max(1u, std::thread::hardware_concurrency());
    // Decode is memory-bandwidth bound; keep a bounded number of workers and
    // derive it from the host topology rather than hard-coding a device.
    return std::clamp((int)cores, 2, 4);
}

static int batch_threads() {
    const unsigned cores = std::max(1u, std::thread::hardware_concurrency());
    // Prompt evaluation benefits from more host workers for scheduling/token
    // preparation while model inference remains GPU-only.
    return std::clamp((int)cores - 2, 4, 6);
}

static bool init_generation_context() {
    if (!g_model || g_context_length == 0) return false;
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = g_context_length;
    // Bound the scheduler's worst-case graph reservation on mobile GPUs. A 128-token
    // ubatch can reserve a much larger Vulkan graph than is useful for interactive
    // chat, especially at 8K context. This affects prompt prefill chunk size, not
    // the decode token loop.
    cp.n_batch = std::min<uint32_t>(cp.n_ctx, 32);
    cp.n_ubatch = std::min<uint32_t>(cp.n_ctx, 32);
    cp.n_rs_seq = 0;
    // The device trace showed FLASH_ATTN_EXT nodes immediately before SIGABRT
    // during context graph reservation on the affected Vulkan/Mali path. Keep the
    // portable attention graph for Vulkan until this optimized path is verified
    // against the device; GPU-only model placement remains enforced separately.
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
    g_context = llama_init_from_model(g_model, cp);
    if (!g_context) return false;
    llama_set_abort_callback(g_context, abort_callback, nullptr);
    return true;
}

static bool validate_gpu_model_residency() {
    if (!g_model) return false;
    size_t gpu_bytes = 0, cpu_bytes = 0, other_bytes = 0;
    size_t gpu_tensors = 0, cpu_tensors = 0, other_tensors = 0;
    for (const auto & entry : g_model->tensors_by_name) {
        const ggml_tensor * tensor = entry.second;
        if (!tensor || !tensor->buffer) continue;
        const ggml_backend_buffer_type_t buft = ggml_backend_buffer_get_type(tensor->buffer);
        const ggml_backend_dev_t dev = ggml_backend_buft_get_device(buft);
        const auto type = dev ? ggml_backend_dev_type(dev) : GGML_BACKEND_DEVICE_TYPE_CPU;
        const size_t bytes = ggml_nbytes(tensor);
        if (type == GGML_BACKEND_DEVICE_TYPE_GPU || type == GGML_BACKEND_DEVICE_TYPE_IGPU) {
            gpu_bytes += bytes; ++gpu_tensors;
        } else if (type == GGML_BACKEND_DEVICE_TYPE_CPU) {
            cpu_bytes += bytes; ++cpu_tensors;
        } else {
            other_bytes += bytes; ++other_tensors;
        }
    }
    append_native_trace((std::string("VULKAN_MODEL_RESIDENCY gpuBytes=") +
        std::to_string(gpu_bytes) + " gpuTensors=" + std::to_string(gpu_tensors) +
        " cpuBytes=" + std::to_string(cpu_bytes) + " cpuTensors=" +
        std::to_string(cpu_tensors) + " otherBytes=" + std::to_string(other_bytes) +
        " otherTensors=" + std::to_string(other_tensors)).c_str());

    // A tiny CPU-side tensor is not useful for inference, but a substantial CPU
    // weight allocation means the GPU-only contract has been violated.
    constexpr size_t kCpuWeightToleranceBytes = 64 * 1024;
    const bool ok = gpu_bytes > 0 && cpu_bytes <= kCpuWeightToleranceBytes && other_bytes == 0;
    append_native_trace(ok ? "VULKAN_MODEL_RESIDENCY_GPU_ONLY_OK"
                           : "VULKAN_MODEL_RESIDENCY_GPU_ONLY_REJECTED");
    return ok;
}


static bool register_static_vulkan_backend() {
    // Vulkan is linked statically into the Android runtime. Register it explicitly so
    // backend discovery does not depend on loading MODULE libraries from an APK.
    ggml_backend_reg_t reg = ggml_backend_reg_by_name(GGML_VK_NAME);
    if (!reg) {
        append_native_trace("VULKAN_STATIC_REGISTRATION_STARTED");
        ggml_backend_reg_t vulkan_reg = ggml_backend_vk_reg();
        if (!vulkan_reg) {
            append_native_trace("VULKAN_STATIC_REGISTRATION_FAILED_NULL_REGISTRY");
            return false;
        }
        ggml_backend_register(vulkan_reg);
        reg = ggml_backend_reg_by_name(GGML_VK_NAME);
    }
    const bool registered = reg != nullptr;
    append_native_trace((std::string("VULKAN_STATIC_REGISTRATION registered=") +
        (registered ? "1" : "0")).c_str());
    return registered;
}

static bool has_vulkan_gpu_device() {
    ggml_backend_reg_t reg = ggml_backend_reg_by_name(GGML_VK_NAME);
    if (!reg) {
        append_native_trace("VULKAN_BACKEND_REGISTRY_NOT_FOUND");
        return false;
    }
    const size_t count = ggml_backend_reg_dev_count(reg);
    append_native_trace((std::string("VULKAN_BACKEND_DEVICE_COUNT count=") +
        std::to_string(count)).c_str());
    for (size_t i = 0; i < count; ++i) {
        ggml_backend_dev_t dev = ggml_backend_reg_dev_get(reg, i);
        if (!dev) continue;
        const auto type = ggml_backend_dev_type(dev);
        const char *name = ggml_backend_dev_name(dev);
        const char *description = ggml_backend_dev_description(dev);
        append_native_trace((std::string("VULKAN_GPU_DEVICE index=") +
            std::to_string(i) + " name=" + (name ? name : "unknown") +
            " description=" + (description ? description : "unknown") +
            " type=" + std::to_string((int)type)).c_str());
        if (type == GGML_BACKEND_DEVICE_TYPE_GPU || type == GGML_BACKEND_DEVICE_TYPE_IGPU) {
            return true;
        }
    }
    return false;
}

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
        const char phase_prefix[] = "NATIVE_FATAL_SPEC_PHASE=";
        native_write_text(g_native_fatal_fd, phase_prefix, sizeof(phase_prefix) - 1);
        char phase[96];
        size_t phase_len = 0;
        const volatile char *phase_src = g_spec_phase ? g_spec_phase : "UNKNOWN";
        while (phase_len + 1 < sizeof(phase) && phase_src[phase_len]) {
            phase[phase_len] = phase_src[phase_len];
            ++phase_len;
        }
        phase[phase_len] = '\0';
        native_write_text(g_native_fatal_fd, phase, phase_len);
        native_write_text(g_native_fatal_fd, "\n", 1);
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
    if (g_spec) { common_speculative_free(g_spec); g_spec = nullptr; }
    g_spec_init.reset();
    g_spec_requested = false;
    g_spec_draft_path.clear();
    clear_android_generation_cache();
    if (g_sampler) { llama_sampler_free(g_sampler); g_sampler = nullptr; }
    if (g_context) { llama_synchronize(g_context); llama_free(g_context); g_context = nullptr; }
    g_context_length = 0;
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
        append_native_trace("VULKAN_BACKEND_INITIALIZATION_STARTED");
        const bool registered = register_static_vulkan_backend();
        if (registered) {
            // The backend registry must be populated before llama's backend init.
            ggml_backend_load_all();
            append_native_trace("ACTIVATION_LLAMA_BACKEND_INIT_STARTED");
            llama_backend_init();
            append_native_trace("ACTIVATION_LLAMA_BACKEND_INIT_RETURNED");
            g_gpu_backend_loaded = has_vulkan_gpu_device();
        } else {
            g_gpu_backend_loaded = false;
        }
        append_native_trace((std::string("VULKAN_BACKEND_DEVICE_STATE loaded=") +
            (g_gpu_backend_loaded ? "1" : "0") +
            " registeredDevices=" + std::to_string(ggml_backend_dev_count())).c_str());
        g_backend_initialized = true;
    } else {
        append_native_trace("VULKAN_BACKEND_ALREADY_INITIALIZED");
        if (!g_gpu_backend_loaded) {
            append_native_trace("VULKAN_BACKEND_RETRY_STARTED");
            const bool registered = register_static_vulkan_backend();
            if (registered) {
                ggml_backend_load_all();
                g_gpu_backend_loaded = has_vulkan_gpu_device();
            }
            append_native_trace((std::string("VULKAN_BACKEND_RETRY_RESULT loaded=") +
                (g_gpu_backend_loaded ? "1" : "0")).c_str());
        }
    }
}

static void append_weight_residency_trace() {
    if (!g_model) return;

    size_t gpu_bytes = 0, host_bytes = 0, cpu_bytes = 0, other_bytes = 0;
    size_t gpu_tensors = 0, host_tensors = 0, cpu_tensors = 0, other_tensors = 0;
    std::set<ggml_backend_buffer_t> gpu_buffers, host_buffers, cpu_buffers, other_buffers;

    for (const auto & entry : g_model->tensors_by_name) {
        const ggml_tensor * tensor = entry.second;
        if (!tensor || !tensor->buffer) continue;

        const size_t bytes = ggml_nbytes(tensor);
        const ggml_backend_buffer_t buffer = tensor->buffer;
        const ggml_backend_buffer_type_t buft = ggml_backend_buffer_get_type(buffer);
        const bool is_host = ggml_backend_buft_is_host(buft);
        const ggml_backend_dev_t dev = ggml_backend_buft_get_device(buft);
        const auto dev_type = dev ? ggml_backend_dev_type(dev) : GGML_BACKEND_DEVICE_TYPE_CPU;

        // Vulkan buffers on Android may be host-accessible because the device is
        // unified memory. Classify by backend device first; otherwise a real GPU
        // buffer is incorrectly reported as "Host Tensor Memory".
        if (dev_type == GGML_BACKEND_DEVICE_TYPE_GPU || dev_type == GGML_BACKEND_DEVICE_TYPE_IGPU) {
            gpu_bytes += bytes; gpu_tensors++; gpu_buffers.insert(buffer);
        } else if (is_host) {
            host_bytes += bytes; host_tensors++; host_buffers.insert(buffer);
        } else if (dev_type == GGML_BACKEND_DEVICE_TYPE_CPU) {
            cpu_bytes += bytes; cpu_tensors++; cpu_buffers.insert(buffer);
        } else {
            other_bytes += bytes; other_tensors++; other_buffers.insert(buffer);
        }
    }

    // Prefer the actual non-CPU device registered by the Vulkan backend.
    ggml_backend_dev_t gpu_dev = nullptr;
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t candidate = ggml_backend_dev_get(i);
        if (!candidate) continue;
        const auto type = ggml_backend_dev_type(candidate);
        if (type == GGML_BACKEND_DEVICE_TYPE_GPU || type == GGML_BACKEND_DEVICE_TYPE_IGPU) {
            gpu_dev = candidate;
            break;
        }
    }

    // If the registry classification is unusual, use the device actually
    // attached to a model tensor as the authoritative runtime device.
    if (!gpu_dev) {
        for (const auto & entry : g_model->tensors_by_name) {
            const ggml_tensor * tensor = entry.second;
            if (!tensor || !tensor->buffer) continue;
            ggml_backend_dev_t candidate = ggml_backend_buft_get_device(
                    ggml_backend_buffer_get_type(tensor->buffer));
            if (!candidate) continue;
            const auto type = ggml_backend_dev_type(candidate);
            if (type != GGML_BACKEND_DEVICE_TYPE_CPU && type != GGML_BACKEND_DEVICE_TYPE_ACCEL) {
                gpu_dev = candidate;
                break;
            }
        }
    }

    // Expose the complete registry state so a real backend-registration problem
    // cannot be hidden behind an "unavailable" UI value.
    append_native_trace((std::string("NATIVE_BACKEND_DEVICES count=") +
        std::to_string(ggml_backend_dev_count())).c_str());
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t candidate = ggml_backend_dev_get(i);
        if (!candidate) continue;
        ggml_backend_dev_props props{};
        ggml_backend_dev_get_props(candidate, &props);
        append_native_trace((std::string("NATIVE_BACKEND_DEVICE index=") +
            std::to_string(i) + " name=" + (props.name ? props.name : "unknown") +
            " description=" + (props.description ? props.description : "unknown") +
            " type=" + std::to_string((int)props.type) +
            " memoryFreeMiB=" + std::to_string((double)props.memory_free / (1024.0 * 1024.0)) +
            " memoryTotalMiB=" + std::to_string((double)props.memory_total / (1024.0 * 1024.0))).c_str());
    }
    const char * gpu_name = gpu_dev ? ggml_backend_dev_name(gpu_dev) : "unavailable";
    const char * gpu_description = gpu_dev ? ggml_backend_dev_description(gpu_dev) : "unavailable";
    size_t free_bytes = 0, total_bytes = 0;
    if (gpu_dev) {
        ggml_backend_dev_memory(gpu_dev, &free_bytes, &total_bytes);
    }

    std::string gpu_name_text = gpu_name ? gpu_name : "unknown";
    std::string gpu_description_text = gpu_description ? gpu_description : "unknown";
    std::replace(gpu_name_text.begin(), gpu_name_text.end(), ' ', '_');
    std::replace(gpu_description_text.begin(), gpu_description_text.end(), ' ', '_');

    append_native_trace((std::string("NATIVE_VULKAN_DEVICE") +
        " name=" + gpu_name_text +
        " description=" + gpu_description_text +
        " memoryFreeMiB=" + std::to_string((double) free_bytes / (1024.0 * 1024.0)) +
        " memoryTotalMiB=" + std::to_string((double) total_bytes / (1024.0 * 1024.0)) +
        " memoryKnown=" + ((free_bytes > 0 && total_bytes > 0) ? "1" : "0")).c_str());

    append_native_trace((std::string("NATIVE_WEIGHT_RESIDENCY") +
        " gpuTensorBytes=" + std::to_string(gpu_bytes) +
        " gpuTensorMiB=" + std::to_string((double) gpu_bytes / (1024.0 * 1024.0)) +
        " gpuTensors=" + std::to_string(gpu_tensors) +
        " gpuBuffers=" + std::to_string(gpu_buffers.size()) +
        " hostTensorBytes=" + std::to_string(host_bytes) +
        " hostTensorMiB=" + std::to_string((double) host_bytes / (1024.0 * 1024.0)) +
        " hostTensors=" + std::to_string(host_tensors) +
        " hostBuffers=" + std::to_string(host_buffers.size()) +
        " cpuTensorBytes=" + std::to_string(cpu_bytes) +
        " cpuTensorMiB=" + std::to_string((double) cpu_bytes / (1024.0 * 1024.0)) +
        " cpuTensors=" + std::to_string(cpu_tensors) +
        " cpuBuffers=" + std::to_string(cpu_buffers.size()) +
        " otherTensorBytes=" + std::to_string(other_bytes) +
        " otherTensors=" + std::to_string(other_tensors) +
        " gpuFreeMiB=" + std::to_string((double) free_bytes / (1024.0 * 1024.0)) +
        " gpuTotalMiB=" + std::to_string((double) total_bytes / (1024.0 * 1024.0))).c_str());
}

static llama_model *load_model_android(const char *path, llama_model_params mp, bool gpu) {
    // Avoid the mmap -> Vulkan host-pointer import path on Android. Mobile UMA Vulkan
    // drivers can fail inside buffer_from_host_ptr while the model is being initialized.
    // LLAMA_LOAD_MODE_NONE keeps bounded file reads and normal backend allocations.
    mp.load_mode = LLAMA_LOAD_MODE_NONE;
    mp.check_tensors = false;
    mp.no_host = true;
    mp.use_extra_bufts = false;

    // n_gpu_layers controls transformer blocks, but llama.cpp may still choose a
    // host buffer for large non-block tensors (notably token embeddings / output
    // weights). That leaves hundreds of MiB off Vulkan even when all layers are
    // requested. Explicitly route every model tensor through the Vulkan device's
    // buffer type. The post-load residency check remains strict; there is no CPU
    // fallback. If Vulkan cannot allocate a tensor, loading fails instead.
    llama_model_tensor_buft_override gpu_tensor_overrides[] = {
        { ".*", nullptr },
        { nullptr, nullptr },
    };
    if (gpu) {
        ggml_backend_reg_t reg = ggml_backend_reg_by_name(GGML_VK_NAME);
        ggml_backend_dev_t gpu_device = nullptr;
        if (reg) {
            for (size_t i = 0; i < ggml_backend_reg_dev_count(reg); ++i) {
                ggml_backend_dev_t candidate = ggml_backend_reg_dev_get(reg, i);
                if (!candidate) continue;
                const auto type = ggml_backend_dev_type(candidate);
                if (type == GGML_BACKEND_DEVICE_TYPE_GPU || type == GGML_BACKEND_DEVICE_TYPE_IGPU) {
                    gpu_device = candidate;
                    break;
                }
            }
        }
        if (!gpu_device) {
            checkpoint("VULKAN_GPU_ONLY_TENSOR_OVERRIDE_FAILED_NO_DEVICE");
            return nullptr;
        }
        gpu_tensor_overrides[0].buft = ggml_backend_dev_buffer_type(gpu_device);
        if (!gpu_tensor_overrides[0].buft) {
            checkpoint("VULKAN_GPU_ONLY_TENSOR_OVERRIDE_FAILED_NO_BUFFER_TYPE");
            return nullptr;
        }
        mp.tensor_buft_overrides = gpu_tensor_overrides;
        checkpoint("VULKAN_GPU_ONLY_ALL_TENSORS_FORCED_TO_DEVICE_BUFFER");
    }

    checkpoint("VULKAN_LOAD_POLICY no_host=1 use_extra_bufts=0 async_uploads=disabled");
    checkpoint(gpu ? "ANDROID_MODEL_LOAD_POLICY_GPU_RESIDENT" : "ANDROID_MODEL_LOAD_POLICY_CPU_STAGED");
    checkpoint((std::string("ANDROID_MODEL_LOAD_PARAMS load_mode=") + llama_load_mode_name(mp.load_mode) +
        " check_tensors=" + (mp.check_tensors ? "1" : "0") +
        " tensor_buft_overrides=" + (gpu ? "vulkan-all" : "default")).c_str());
    return llama_model_load_from_file(path, mp);
}

extern "C" JNIEXPORT jint JNICALL
Java_com_woogit_aicore_runtime_android_NativeLlamaCpp_nativeLoad(JNIEnv *env, jclass, jstring jpath, jint ctx_len, jint gpu_layers, jstring jdraftpath) {
    checkpoint("NATIVE_LOAD_STARTED");
    append_native_trace((std::string("ACTIVATION_NATIVE_LOAD_STARTED ctx_len=") + std::to_string((int)ctx_len) + " gpu_layers=" + std::to_string((int)gpu_layers)).c_str());
    checkpoint("FREE_OLD_RUNTIME_STARTED"); free_all(); checkpoint("FREE_OLD_RUNTIME_RETURNED"); g_stop.store(false);
    const char *path = env->GetStringUTFChars(jpath, nullptr);
    if (!path) { checkpoint("PATH_UTF8_FAILED"); return 3; }
    g_spec_draft_path.clear(); g_spec_requested = false; g_spec_mtp = false; g_target_model_path.clear(); g_spec_accept_ema = 1.0;
    // Draft/speculative decoding is removed from the product; retain the legacy JNI argument but never use it.
    (void) jdraftpath;
    g_spec_draft_path.clear();
    g_spec_requested = false;
    g_spec_mtp = false;
    g_spec_accept_ema = 1.0;
    append_native_trace("DRAFT_FEATURE_REMOVED");
    g_target_model_path = path;
    const std::string preflight = gguf_preflight(path); checkpoint(preflight.c_str());
    // Target activation must never implicitly enable speculative/MTP.
    // A target GGUF may contain MTP metadata, but that is not a request to
    // allocate a second context during activation. Draft/speculative setup
    // happens only after the target runtime is fully ready.
    append_native_trace("SPECULATIVE_AUTO_DETECTION_DISABLED_DURING_TARGET_LOAD");
    if (!g_gpu_backend_loaded || !has_vulkan_gpu_device()) {
        checkpoint("VULKAN_GPU_ONLY_REJECTED_NO_VULKAN_GPU_DEVICE");
        append_native_trace("VULKAN_GPU_ONLY_NO_CPU_FALLBACK");
        env->ReleaseStringUTFChars(jpath, path);
        return 5;
    }
    if (gpu_layers <= 0) {
        checkpoint("VULKAN_GPU_ONLY_REJECTED_INVALID_GPU_LAYERS");
        env->ReleaseStringUTFChars(jpath, path);
        return 4;
    }
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 999;
    append_native_trace((std::string("VULKAN_GPU_ONLY_ALL_LAYERS requested=") + std::to_string((int)gpu_layers) + " effective=999").c_str());
    mp.load_mode = LLAMA_LOAD_MODE_NONE;
    mp.check_tensors = false;
    checkpoint("VULKAN_GPU_DEVICE_SELECTION");
    checkpoint("VULKAN_GPU_ONLY_LOAD_MODE_STAGED");
    checkpoint("VULKAN_GPU_ONLY_CHECK_TENSORS_DISABLED");
    checkpoint((std::string("MODEL_LOAD_PARAMS gpu_layers=") + std::to_string((int)mp.n_gpu_layers) +
        " load_mode=" + llama_load_mode_name(mp.load_mode) +
        " check_tensors=" + (mp.check_tensors ? "1" : "0")).c_str());
    checkpoint("MODEL_LOAD_STARTED gpu_layers=VULKAN");
    g_model = load_model_android(path, mp, true);
    checkpoint(g_model ? "MODEL_LOAD_RETURNED_SUCCESS" : "MODEL_LOAD_RETURNED_FAILED");
    if (g_model) {
        append_weight_residency_trace();
        if (!validate_gpu_model_residency()) {
            checkpoint("VULKAN_GPU_ONLY_MODEL_RESIDENCY_REJECTED");
            append_native_trace("VULKAN_GPU_ONLY_NO_CPU_WEIGHT_FALLBACK");
            env->ReleaseStringUTFChars(jpath, path);
            free_all();
            return 6;
        }
    }
    env->ReleaseStringUTFChars(jpath, path);
    if (!g_model) {
        checkpoint("VULKAN_GPU_ONLY_MODEL_LOAD_FAILED_NO_CPU_FALLBACK");
        g_gpu = false;
        return 1;
    }
    g_gpu = true;
    if (!g_model) { checkpoint("MODEL_LOAD_FAILED"); return 1; }
    const int trained = llama_model_n_ctx_train(g_model);
    const int requested = ctx_len > 0 ? ctx_len : 4096;
    const int effective = std::max(1, std::min(requested, trained));
    checkpoint((std::string("MODEL_READY trained_ctx=") + std::to_string(trained) + " effective_ctx=" + std::to_string(effective)).c_str());
    g_context_length = (uint32_t)effective;
    checkpoint("CONTEXT_INIT_STARTED");
    const bool context_ready = init_generation_context();
    checkpoint(context_ready ? "CONTEXT_INIT_RETURNED_SUCCESS" : "CONTEXT_INIT_RETURNED_FAILED");
    if (!context_ready) { checkpoint("CONTEXT_INIT_FAILED"); free_all(); return 2; }
    append_native_trace("TARGET_RUNTIME_READY_BEFORE_SPECULATIVE");
    if (g_spec_requested) {
        append_native_trace("SPECULATIVE_ATTACH_AFTER_TARGET_READY");
        const bool spec_ready = init_speculative_runtime();
        checkpoint(spec_ready ? "SPECULATIVE_INIT_RETURNED_SUCCESS" : "SPECULATIVE_INIT_RETURNED_FAILED");
    }
    checkpoint("NATIVE_LOAD_COMPLETED"); return 0;
}

extern "C" JNIEXPORT void JNICALL
Java_com_woogit_aicore_runtime_android_NativeLlamaCpp_nativeStop(JNIEnv *, jclass) { g_stop.store(true); }
extern "C" JNIEXPORT void JNICALL
Java_com_woogit_aicore_runtime_android_NativeLlamaCpp_nativeUnload(JNIEnv *, jclass) { g_stop.store(true); free_all(); }
extern "C" JNIEXPORT jstring JNICALL
Java_com_woogit_aicore_runtime_android_NativeLlamaCpp_nativeRuntimeInfo(JNIEnv *env, jclass) {
    const std::string value = std::string(g_gpu && g_gpu_backend_loaded ? "Vulkan-GPU-ONLY" : "Vulkan-UNAVAILABLE") + "; llama.cpp=" + AI_CHAT_LLAMA_CPP_SHA;
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
