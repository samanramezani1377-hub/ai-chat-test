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
#include <sys/stat.h>
#include <ucontext.h>
#include "llama.h"
#include "gguf.h"
#include "llama-model.h"
#include "ggml-backend.h"
#include "ggml-backend-impl.h"

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
    cp.n_batch = std::min<uint32_t>(cp.n_ctx, 128);
    cp.n_ubatch = std::min<uint32_t>(cp.n_ctx, 128);
    cp.n_rs_seq = 0;
    // Let llama.cpp select the backend-safe attention implementation. The previous
    // forced-disabled path expanded attention work and was a major mobile decode
    // cost at 8K context. AUTO preserves the normal attention math while allowing
    // OpenCL to use its optimized path when supported.
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_AUTO;
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
        std::to_string(cp.n_ubatch) + " flashAttn=auto").c_str());
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
    append_native_trace((std::string("OPENGL_ES_MODEL_RESIDENCY gpuBytes=") +
        std::to_string(gpu_bytes) + " gpuTensors=" + std::to_string(gpu_tensors) +
        " cpuBytes=" + std::to_string(cpu_bytes) + " cpuTensors=" +
        std::to_string(cpu_tensors) + " otherBytes=" + std::to_string(other_bytes) +
        " otherTensors=" + std::to_string(other_tensors)).c_str());

    // A tiny CPU-side tensor is not useful for inference, but a substantial CPU
    // weight allocation means the GPU-only contract has been violated.
    constexpr size_t kCpuWeightToleranceBytes = 64 * 1024;
    const bool ok = gpu_bytes > 0 && cpu_bytes <= kCpuWeightToleranceBytes && other_bytes == 0;
    append_native_trace(ok ? "OPENGL_ES_MODEL_RESIDENCY_GPU_ONLY_OK"
                           : "OPENGL_ES_MODEL_RESIDENCY_GPU_ONLY_REJECTED");
    return ok;
}


static bool has_opengles_gpu_device() {
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (!dev) continue;
        const char * name = ggml_backend_dev_name(dev);
        if (name && std::string(name) == "OpenGL ES") return true;
    }
    return false;
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

        // OpenCL buffers on Android may be host-accessible because the device is
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

    // Prefer the actual non-CPU device registered by the OpenCL backend.
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

    append_native_trace((std::string("NATIVE_OPENCL_DEVICE") +
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
    // Avoid the mmap -> OpenCL host-pointer import path on Android. Mobile UMA OpenCL
    // drivers can fail inside buffer_from_host_ptr while the model is being initialized.
    // LLAMA_LOAD_MODE_NONE keeps bounded file reads and normal backend allocations, so
    // GPU layers remain enabled without relying on the fragile mmap buffer import.
    mp.load_mode = LLAMA_LOAD_MODE_NONE;
    mp.check_tensors = false;
    mp.no_host = true;

    // Do not force token_embd onto OpenCL. llama.cpp's backend placement is
    // architecture/device aware; overriding the embedding buffer here adds a
    // large device allocation and has caused avoidable load pressure on Android.
    // GPU-only residency is still enforced after the model is loaded.
    checkpoint(gpu ? "ANDROID_MODEL_LOAD_POLICY_GPU_RESIDENT" : "ANDROID_MODEL_LOAD_POLICY_CPU_STAGED");
    checkpoint((std::string("ANDROID_MODEL_LOAD_PARAMS load_mode=") + llama_load_mode_name(mp.load_mode) +
        " check_tensors=" + (mp.check_tensors ? "1" : "0")).c_str());
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
    // Draft/speculative decoding is removed from the product; retain the legacy JNI argument but never use it.\n    (void) jdraftpath;\n    g_spec_draft_path.clear(); g_spec_requested = false; g_spec_mtp = false; g_spec_accept_ema = 1.0;\n    append_native_trace("DRAFT_FEATURE_REMOVED");
    g_target_model_path = path;
    const std::string preflight = gguf_preflight(path); checkpoint(preflight.c_str());
    // Target activation must never implicitly enable speculative/MTP.
    // A target GGUF may contain MTP metadata, but that is not a request to
    // allocate a second context during activation. Draft/speculative setup
    // happens only after the target runtime is fully ready.
    append_native_trace("SPECULATIVE_AUTO_DETECTION_DISABLED_DURING_TARGET_LOAD");
    if (!g_gpu_backend_loaded || !has_opengles_gpu_device()) {
        checkpoint("OPENGL_ES_GPU_ONLY_REJECTED_NO_OPENGL_ES_GPU_DEVICE");
        append_native_trace("OPENGL_ES_GPU_ONLY_NO_CPU_FALLBACK");
        env->ReleaseStringUTFChars(jpath, path);
        return 5;
    }
    if (gpu_layers <= 0) {
        checkpoint("OPENGL_ES_GPU_ONLY_REJECTED_INVALID_GPU_LAYERS");
        env->ReleaseStringUTFChars(jpath, path);
        return 4;
    }
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 999;
    append_native_trace((std::string("OPENGL_ES_GPU_ONLY_ALL_LAYERS requested=") + std::to_string((int)gpu_layers) + " effective=999").c_str());
    mp.load_mode = LLAMA_LOAD_MODE_NONE;
    mp.check_tensors = false;
    checkpoint("OPENGL_ES_GPU_DEVICE_SELECTION");
    checkpoint("OPENGL_ES_GPU_ONLY_LOAD_MODE_STAGED");
    checkpoint("OPENGL_ES_GPU_ONLY_CHECK_TENSORS_DISABLED");
    checkpoint((std::string("MODEL_LOAD_PARAMS gpu_layers=") + std::to_string((int)mp.n_gpu_layers) +
        " load_mode=" + llama_load_mode_name(mp.load_mode) +
        " check_tensors=" + (mp.check_tensors ? "1" : "0")).c_str());
    checkpoint("MODEL_LOAD_STARTED gpu_layers=OPENCL");
    g_model = load_model_android(path, mp, true);
    checkpoint(g_model ? "MODEL_LOAD_RETURNED_SUCCESS" : "MODEL_LOAD_RETURNED_FAILED");
    if (g_model) {
        append_weight_residency_trace();
        if (!validate_gpu_model_residency()) {
            checkpoint("OPENGL_ES_GPU_ONLY_MODEL_RESIDENCY_REJECTED");
            append_native_trace("OPENGL_ES_GPU_ONLY_NO_CPU_WEIGHT_FALLBACK");
            env->ReleaseStringUTFChars(jpath, path);
            free_all();
            return 6;
        }
    }
    env->ReleaseStringUTFChars(jpath, path);
    if (!g_model) {
        checkpoint("OPENGL_ES_GPU_ONLY_MODEL_LOAD_FAILED_NO_CPU_FALLBACK");
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
    const std::string value = std::string(g_gpu && g_gpu_backend_loaded ? "OpenGL-ES-GPU-ONLY" : "OpenGL-ES-UNAVAILABLE") + "; llama.cpp=" + AI_CHAT_LLAMA_CPP_SHA;
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
