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
#include <sys/syscall.h>
#include <ucontext.h>
#include "llama.h"
#include "gguf.h"
#include "llama-model.h"
#include "ggml-backend.h"
#include "ggml-backend-impl.h"
#include "ggml-opencl.h"
#include "ggml-webgpu.h"
#include <CL/cl.h>
#include <set>
#include "android_opencl_dispatch.h"
#include "universal_gpu_adapter.h"

// CL_PLATFORM_ICD_SUFFIX_KHR belongs to cl_khr_icd and is not declared by every
// vendor-neutral OpenCL header package. Keep the standard extension value available
// for diagnostics without requiring a particular vendor SDK.
#ifndef CL_PLATFORM_ICD_SUFFIX_KHR
#define CL_PLATFORM_ICD_SUFFIX_KHR 0x0920
#endif

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
// Fixed-size last-event snapshot, readable from the fatal signal handler.
static std::atomic<char> g_native_runtime_phase[192]{};
static_assert(std::atomic<char>::is_always_lock_free, "Fatal signal phase snapshot requires lock-free atomics");
static std::mutex g_native_marker_mutex;
static std::mutex g_backend_init_mutex;
static bool g_backend_initialized = false;
static bool g_gpu_backend_loaded = false;
static bool g_webgpu_backend_loaded = false;
static uint32_t g_context_length = 0;
static bool g_universal_gles_version_known = false;
static bool g_universal_gles_compute_supported = false;

static bool abort_callback(void *) { return g_stop.load(std::memory_order_relaxed); }
static void append_native_trace(const char *text);
static void checkpoint(const char *event);
static bool init_generation_context(bool use_webgpu_backend);
static ggml_backend_dev_t find_webgpu_gpu_device();
static int available_cpu_cores() {
    // Use every CPU core reported by Android's native runtime. This is the
    // device's available hardware concurrency, not a fixed 2-4 worker cap.
    return (int)std::max(1u, std::thread::hardware_concurrency());
}

static int generation_threads() {
    return available_cpu_cores();
}

static int batch_threads() {
    return available_cpu_cores();
}

static bool init_generation_context(bool use_webgpu_backend) {
    if (!g_model || g_context_length == 0) return false;
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = g_context_length;
    cp.n_batch = std::min<uint32_t>(cp.n_ctx, 128);
    cp.n_ubatch = std::min<uint32_t>(cp.n_ctx, 128);
    cp.n_rs_seq = 0;
    // Keep fused FLASH_ATTN_EXT disabled on Android for both execution modes.
    // The last device run aborted while reserving a graph containing FLASH_ATTN_EXT
    // on the Dawn/OpenGLES path. The pinned llama.cpp revision has also shown
    // scheduler failures with this fused path on Android CPU for hybrid/recurrent
    // prompt batches. Let GGML schedule the regular attention operations instead;
    // this does not change model weights, quantization, sampling, or answer quality.
    // It only avoids selecting the fused kernel whose Android backend path is not
    // reliable across the supported devices and model architectures.
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    append_native_trace((std::string("NATIVE_ATTENTION_POLICY backend=") +
        (use_webgpu_backend ? "WEBGPU" : "CPU") +
        " flashAttention=DISABLED reason=android_fused_flash_attention_stability").c_str());
    cp.type_k = GGML_TYPE_F16;
    cp.type_v = GGML_TYPE_F16;
    // Keep KV-cache and attention K/Q/V work on the CPU for WebGPU contexts.
    // This avoids allocating a second large, device-owned cache during graph
    // reservation while retaining GPU-resident model layers and GPU matmul.
    // CPU contexts keep llama.cpp's normal KQV offload policy.
    cp.offload_kqv = !use_webgpu_backend;
    append_native_trace((std::string("NATIVE_KQV_OFFLOAD_POLICY backend=") +
        (use_webgpu_backend ? "WEBGPU enabled=0 reason=portable_device_memory_reservation" :
                              "CPU enabled=1 reason=upstream_default")).c_str());
    cp.n_seq_max = 1;
    const int runtime_threads = generation_threads();
    const int runtime_batch_threads = batch_threads();
    cp.n_threads = runtime_threads;
    cp.n_threads_batch = runtime_batch_threads;
    append_native_trace((std::string("NATIVE_CONTEXT_THREADS generation=") +
        std::to_string(cp.n_threads) + " batch=" +
        std::to_string(cp.n_threads_batch) + " nBatch=" +
        std::to_string(cp.n_batch) + " nUbatch=" +
        std::to_string(cp.n_ubatch) + " flashAttn=DISABLED").c_str());
#if defined(AI_CHAT_GGML_OPENMP_ENABLED)
    append_native_trace("NATIVE_OPENMP_POLICY enabled=1 linkage=static");
#else
    append_native_trace("NATIVE_OPENMP_POLICY enabled=0");
#endif
#if defined(AI_CHAT_GGML_CPU_KLEIDIAI_ENABLED)
    append_native_trace("NATIVE_KLEIDIAI_POLICY enabled=1 cpuRuntimeDispatch=1");
#else
    append_native_trace("NATIVE_KLEIDIAI_POLICY enabled=0");
#endif
#if defined(AI_CHAT_CPU_ARM_BASELINE)
    append_native_trace((std::string("NATIVE_CPU_ARCH_POLICY baseline=") +
        AI_CHAT_CPU_ARM_BASELINE +
        " genericBackendPortable=1 specializedKernelDispatch=runtime").c_str());
#endif
    // Hybrid/recurrent architectures (including Qwen3.5-family models) carry
    // recurrent state across prompt batches. On Android, graph reuse in the pinned
    // llama.cpp build can fail on the second otherwise-identical prefill batch
    // (ggml_backend_sched_graph_compute_async -> GGML_STATUS_FAILED), leaving the
    // request with zero generated tokens. Disable graph reuse only while creating
    // hybrid/recurrent contexts; llama.cpp snapshots this environment setting into
    // the context, so restore the process environment immediately afterwards.
    const bool hybrid_or_recurrent =
        llama_model_is_hybrid(g_model) || llama_model_is_recurrent(g_model);
    const char * previous_graph_reuse = std::getenv("LLAMA_GRAPH_REUSE_DISABLE");
    const bool had_previous_graph_reuse = previous_graph_reuse != nullptr;
    const std::string previous_graph_reuse_value =
        previous_graph_reuse ? previous_graph_reuse : "";
    if (hybrid_or_recurrent) {
        setenv("LLAMA_GRAPH_REUSE_DISABLE", "1", 1);
        append_native_trace("NATIVE_GRAPH_REUSE_POLICY hybridOrRecurrent=1 disabled=1 reason=android_recurrent_prefill_stability");
    } else {
        append_native_trace("NATIVE_GRAPH_REUSE_POLICY hybridOrRecurrent=0 disabled=0");
    }
    if (use_webgpu_backend) {
        size_t free_bytes = 0, total_bytes = 0;
        ggml_backend_dev_t device = find_webgpu_gpu_device();
        if (device) ggml_backend_dev_memory(device, &free_bytes, &total_bytes);
        append_native_trace((std::string("WEBGPU_CONTEXT_RESERVATION_PREFLIGHT ctx=") +
            std::to_string(cp.n_ctx) + " batch=" + std::to_string(cp.n_batch) +
            " ubatch=" + std::to_string(cp.n_ubatch) +
            " seqMax=" + std::to_string(cp.n_seq_max) +
            " kvTypeK=" + std::to_string((int)cp.type_k) +
            " kvTypeV=" + std::to_string((int)cp.type_v) +
            " offloadKQV=" + (cp.offload_kqv ? "1" : "0") +
            " gpuFreeBytes=" + std::to_string(free_bytes) +
            " gpuTotalBytes=" + std::to_string(total_bytes) +
            " gpuMemoryKnown=" + ((free_bytes > 0 && total_bytes > 0) ? "1" : "0")).c_str());
    }
    checkpoint(use_webgpu_backend
        ? "WEBGPU_CONTEXT_GRAPH_RESERVATION_BEGIN"
        : "CPU_CONTEXT_GRAPH_RESERVATION_BEGIN");
    g_context = llama_init_from_model(g_model, cp);
    checkpoint(g_context
        ? (use_webgpu_backend ? "WEBGPU_CONTEXT_GRAPH_RESERVATION_RETURNED" : "CPU_CONTEXT_GRAPH_RESERVATION_RETURNED")
        : (use_webgpu_backend ? "WEBGPU_CONTEXT_GRAPH_RESERVATION_FAILED" : "CPU_CONTEXT_GRAPH_RESERVATION_FAILED"));
    if (hybrid_or_recurrent) {
        if (had_previous_graph_reuse) {
            setenv("LLAMA_GRAPH_REUSE_DISABLE", previous_graph_reuse_value.c_str(), 1);
        } else {
            unsetenv("LLAMA_GRAPH_REUSE_DISABLE");
        }
    }
    if (!g_context) return false;
    llama_set_abort_callback(g_context, abort_callback, nullptr);
    return true;
}

static bool is_webgpu_backend_device(ggml_backend_dev_t dev);

static bool validate_gpu_model_residency() {
    if (!g_model) return false;
    size_t gpu_bytes = 0, cpu_bytes = 0, other_bytes = 0;
    size_t gpu_tensors = 0, cpu_tensors = 0, other_tensors = 0;
    size_t total_weight_bytes = 0;
    for (const auto & entry : g_model->tensors_by_name) {
        const ggml_tensor * tensor = entry.second;
        if (!tensor) continue;
        const size_t bytes = ggml_nbytes(tensor);
        total_weight_bytes += bytes;
        if (!tensor->buffer) {
            ++other_tensors;
            other_bytes += bytes;
            continue;
        }
        const ggml_backend_buffer_type_t buft = ggml_backend_buffer_get_type(tensor->buffer);
        const ggml_backend_dev_t dev = ggml_backend_buft_get_device(buft);
        const auto type = dev ? ggml_backend_dev_type(dev) : GGML_BACKEND_DEVICE_TYPE_CPU;
        if (is_webgpu_backend_device(dev)) {
            gpu_bytes += bytes; ++gpu_tensors;
        } else if (type == GGML_BACKEND_DEVICE_TYPE_CPU) {
            cpu_bytes += bytes; ++cpu_tensors;
        } else {
            other_bytes += bytes; ++other_tensors;
        }
    }
    const double gpu_percent = total_weight_bytes > 0
        ? (100.0 * static_cast<double>(gpu_bytes) / static_cast<double>(total_weight_bytes))
        : 0.0;
    // Reject negligible GPU weight residency before graph/context construction.
    // The byte-ratio policy is independent of model family and quantization.
    constexpr double kMinimumUsefulGpuWeightPercent = 1.0;
    const bool useful_gpu_residency =
        gpu_bytes > 0 && gpu_percent >= kMinimumUsefulGpuWeightPercent;
    append_native_trace((std::string("WEBGPU_MODEL_RESIDENCY gpuBytes=") +
        std::to_string(gpu_bytes) + " gpuTensors=" + std::to_string(gpu_tensors) +
        " cpuBytes=" + std::to_string(cpu_bytes) + " cpuTensors=" +
        std::to_string(cpu_tensors) + " otherBytes=" + std::to_string(other_bytes) +
        " otherTensors=" + std::to_string(other_tensors) +
        " totalWeightBytes=" + std::to_string(total_weight_bytes) +
        " gpuWeightPercent=" + std::to_string(gpu_percent) +
        " minimumUsefulGpuWeightPercent=" + std::to_string(kMinimumUsefulGpuWeightPercent)).c_str());

    const char * residency_mode = gpu_bytes == 0 ? "WEBGPU_REJECTED_NO_GPU_WEIGHTS"
        : (!useful_gpu_residency ? "WEBGPU_REJECTED_NEGLIGIBLE_WEIGHT_RESIDENCY"
           : (gpu_percent >= 99.0 ? "WEBGPU_FULL_WEIGHT_RESIDENCY"
                                  : "WEBGPU_MIXED_WEIGHT_RESIDENCY"));
    append_native_trace((std::string("WEBGPU_MODEL_RESIDENCY_POLICY mode=") +
        residency_mode + " gpuWeightPercent=" + std::to_string(gpu_percent) +
        " wholeModelCpuFallback=enabled").c_str());
    if (!useful_gpu_residency) {
        append_native_trace("WEBGPU_RESIDENCY_REJECTED action=release_gpu_model_and_retry_cpu");
    }
    return useful_gpu_residency;
}


static const char * opencl_error_name(cl_int code) {
    switch (code) {
        case CL_SUCCESS: return "CL_SUCCESS";
        case CL_DEVICE_NOT_FOUND: return "CL_DEVICE_NOT_FOUND";
        case CL_DEVICE_NOT_AVAILABLE: return "CL_DEVICE_NOT_AVAILABLE";
        case CL_COMPILER_NOT_AVAILABLE: return "CL_COMPILER_NOT_AVAILABLE";
        case CL_MEM_OBJECT_ALLOCATION_FAILURE: return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
        case CL_OUT_OF_RESOURCES: return "CL_OUT_OF_RESOURCES";
        case CL_OUT_OF_HOST_MEMORY: return "CL_OUT_OF_HOST_MEMORY";
        case CL_INVALID_VALUE: return "CL_INVALID_VALUE";
        case CL_INVALID_PLATFORM: return "CL_INVALID_PLATFORM";
        case -1001: return "CL_PLATFORM_NOT_FOUND_KHR";
        default: return "CL_UNKNOWN_ERROR";
    }
}

static std::string opencl_platform_info(cl_platform_id platform, cl_platform_info info) {
    size_t size = 0;
    if (clGetPlatformInfo(platform, info, 0, nullptr, &size) != CL_SUCCESS || size == 0) {
        return "<unavailable>";
    }
    std::string value(size, '\0');
    if (clGetPlatformInfo(platform, info, size, value.data(), nullptr) != CL_SUCCESS) {
        return "<unavailable>";
    }
    if (!value.empty() && value.back() == '\0') value.pop_back();
    return value;
}

static void probe_android_opencl_library_visibility() {
    append_native_trace("OPENCL_LIBRARY_VISIBILITY_PROBE_STARTED");
    const char * candidates[] = {
        "/vendor/lib64/libOpenCL.so",
        "/vendor/lib64/libOpenCL_adreno.so",
        "/system/vendor/lib64/libOpenCL.so",
        "/system/vendor/lib/libOpenCL.so",
        "/system_ext/lib64/libOpenCL_system.so",
        "/system/lib64/libOpenCL.so",
        "/vendor/lib/libOpenCL.so",
    };
    for (const char * path : candidates) {
        struct stat st{};
        const bool exists = stat(path, &st) == 0;
        if (!exists) {
            append_native_trace((std::string("OPENCL_LIBRARY_CANDIDATE path=") + path + " exists=0").c_str());
            continue;
        }
        // Clear any previous dynamic-loader error before this independent attempt.
        (void) dlerror();
        void * handle = dlopen(path, RTLD_NOW | RTLD_LOCAL);
        const char * err = handle ? nullptr : dlerror();
        append_native_trace((std::string("OPENCL_LIBRARY_CANDIDATE path=") + path +
            " exists=1 dlopen=" + (handle ? "1" : "0") +
            " error=" + (err ? err : "none")).c_str());
        if (handle) dlclose(handle);
    }
    append_native_trace("OPENCL_LIBRARY_VISIBILITY_PROBE_COMPLETED");
}

static bool has_icd_vendor_file(const char *directory) {
    if (!directory || !*directory) return false;
    DIR *dir = opendir(directory);
    if (!dir) {
        append_native_trace((std::string("OPENCL_ICD_DIRECTORY path=") + directory +
            " readable=0 reason=opendir_failed").c_str());
        return false;
    }

    bool found = false;
    struct dirent *entry = nullptr;
    while ((entry = readdir(dir)) != nullptr) {
        const std::string name(entry->d_name);
        if (name == "." || name == "..") continue;
        const bool icd_file = name.size() >= 4 &&
            (name.compare(name.size() - 4, 4, ".icd") == 0 ||
             (name.size() >= 6 && name.compare(name.size() - 6, 6, ".icd64") == 0));
        if (!icd_file) continue;

        const std::string full_path = std::string(directory) + "/" + name;
        std::ifstream input(full_path);
        std::string library;
        std::getline(input, library);
        while (!library.empty() && (library.back() == '\r' || library.back() == ' ' || library.back() == '\t')) {
            library.pop_back();
        }
        size_t first = library.find_first_not_of(" \t");
        if (first != std::string::npos) library.erase(0, first);
        append_native_trace((std::string("OPENCL_ICD_FILE path=") + full_path +
            " readable=" + (input.good() || !library.empty() ? "1" : "0") +
            " library=" + (library.empty() ? "<empty>" : library)).c_str());
        if (!library.empty()) found = true;
    }
    closedir(dir);
    append_native_trace((std::string("OPENCL_ICD_DIRECTORY path=") + directory +
        " readable=1 hasUsableIcd=" + (found ? "1" : "0")).c_str());
    return found;
}

static void configure_android_icd_search_path() {
    append_native_trace("OPENCL_ICD_DIRECTORY_DISCOVERY_STARTED");
    // The Khronos loader's Android default is /system/vendor/Khronos/OpenCL/vendors,
    // but some vendor images put ICD manifests under /vendor/etc/OpenCL/vendors.
    // Only override the loader path when a real manifest is present; never point it
    // at a guessed directory or at libOpenCL.so (which can be the loader itself).
    const char * candidates[] = {
        "/system/vendor/Khronos/OpenCL/vendors",
        "/vendor/etc/OpenCL/vendors",
        "/odm/etc/OpenCL/vendors",
        "/vendor/Khronos/OpenCL/vendors",
        "/system_ext/vendor/Khronos/OpenCL/vendors",
    };
    for (const char *directory : candidates) {
        if (!has_icd_vendor_file(directory)) continue;
        if (setenv("OCL_ICD_VENDORS", directory, 1) == 0) {
            append_native_trace((std::string("OPENCL_ICD_SEARCH_PATH_SELECTED path=") +
                directory + " source=discovered_manifest").c_str());
        } else {
            append_native_trace((std::string("OPENCL_ICD_SEARCH_PATH_SET_FAILED path=") +
                directory).c_str());
        }
        append_native_trace("OPENCL_ICD_DIRECTORY_DISCOVERY_COMPLETED");
        return;
    }
    append_native_trace("OPENCL_ICD_SEARCH_PATH_SELECTED path=<loader_default> source=no_manifest_found");
    append_native_trace("OPENCL_ICD_DIRECTORY_DISCOVERY_COMPLETED");
}

// Probe vendor libraries independently of the statically linked Khronos ICD loader.
// A candidate is eligible for OCL_ICD_FILENAMES only if it exports the ICD entry point
// and that entry point returns at least one platform. This avoids feeding the loader
// its own libOpenCL.so and creating recursive dispatch.
using cl_get_platform_ids_fn = cl_int (CL_API_CALL *)(cl_uint, cl_platform_id *, cl_uint *);
using cl_icd_get_platform_ids_khr_fn = cl_int (CL_API_CALL *)(cl_uint, cl_platform_id *, cl_uint *);

static std::string take_dlerror() {
    const char *error = dlerror();
    return error ? std::string(error) : std::string("none");
}

static std::string direct_platform_string(
    cl_int (CL_API_CALL *get_platform_info)(cl_platform_id, cl_platform_info, size_t, void *, size_t *),
    cl_platform_id platform,
    cl_platform_info field) {
    size_t size = 0;
    if (!get_platform_info || get_platform_info(platform, field, 0, nullptr, &size) != CL_SUCCESS || size == 0) {
        return "<unavailable>";
    }
    std::string value(size, '\0');
    if (get_platform_info(platform, field, size, value.data(), nullptr) != CL_SUCCESS) return "<unavailable>";
    if (!value.empty() && value.back() == '\0') value.pop_back();
    return value;
}

static bool probe_android_vendor_icd_candidates() {
    append_native_trace("OPENCL_VENDOR_ICD_DIRECT_PROBE_STARTED");
    const char * candidates[] = {
        "libOpenCL.so",
        "/vendor/lib64/libOpenCL.so",
        "/system/vendor/lib64/libOpenCL.so",
        "/vendor/lib64/libOpenCL_adreno.so",
        "/system/vendor/lib64/libOpenCL_adreno.so",
        "/vendor/lib64/libGLES_mali.so",
        "/system/vendor/lib64/egl/libGLES_mali.so",
        "/vendor/lib64/libmali.so",
        "/system/vendor/lib64/libmali.so",
    };

    for (const char *path : candidates) {
        (void) dlerror();
        void *handle = dlopen(path, RTLD_NOW | RTLD_LOCAL);
        const std::string load_error = handle ? "none" : take_dlerror();
        if (!handle) {
            append_native_trace((std::string("OPENCL_VENDOR_LIBRARY_CANDIDATE path=") + path +
                " dlopen=0 error=" + load_error).c_str());
            continue;
        }

        (void) dlerror();
        auto get_platforms = reinterpret_cast<cl_get_platform_ids_fn>(dlsym(handle, "clGetPlatformIDs"));
        const std::string get_platforms_error = take_dlerror();
        (void) dlerror();
        auto get_icd_platforms = reinterpret_cast<cl_icd_get_platform_ids_khr_fn>(
            dlsym(handle, "clIcdGetPlatformIDsKHR"));
        const std::string icd_symbol_error = take_dlerror();
        (void) dlerror();
        auto get_platform_info = reinterpret_cast<
            cl_int (CL_API_CALL *)(cl_platform_id, cl_platform_info, size_t, void *, size_t *)>(
                dlsym(handle, "clGetPlatformInfo"));
        const std::string platform_info_error = take_dlerror();
        (void) dlerror();
        auto get_devices = reinterpret_cast<
            cl_int (CL_API_CALL *)(cl_platform_id, cl_device_type, cl_uint, cl_device_id *, cl_uint *)>(
                dlsym(handle, "clGetDeviceIDs"));
        const std::string get_devices_error = take_dlerror();
        (void) dlerror();
        auto get_device_info = reinterpret_cast<
            cl_int (CL_API_CALL *)(cl_device_id, cl_device_info, size_t, void *, size_t *)>(
                dlsym(handle, "clGetDeviceInfo"));
        const std::string device_info_error = take_dlerror();
        (void) dlerror();
        void *get_extension_function_address = dlsym(handle, "clGetExtensionFunctionAddress");
        const std::string extension_error = take_dlerror();

        cl_uint direct_count = 0;
        cl_int direct_result = (cl_int)-9999;
        if (get_platforms) direct_result = get_platforms(0, nullptr, &direct_count);

        append_native_trace((std::string("OPENCL_VENDOR_LIBRARY_CANDIDATE path=") + path +
            " dlopen=1 clGetPlatformIDs=" + (get_platforms ? "1" : "0") +
            " getPlatformIDsError=" + get_platforms_error +
            " directResult=" + std::to_string((int)direct_result) +
            " directName=" + opencl_error_name(direct_result) +
            " directPlatformCount=" + std::to_string(direct_count) +
            " clIcdGetPlatformIDsKHR=" + (get_icd_platforms ? "1" : "0") +
            " icdSymbolError=" + icd_symbol_error +
            " clGetPlatformInfo=" + (get_platform_info ? "1" : "0") +
            " platformInfoError=" + platform_info_error +
            " clGetDeviceIDs=" + (get_devices ? "1" : "0") +
            " getDeviceIDsError=" + get_devices_error +
            " clGetDeviceInfo=" + (get_device_info ? "1" : "0") +
            " deviceInfoError=" + device_info_error +
            " clGetExtensionFunctionAddress=" + (get_extension_function_address ? "1" : "0") +
            " extensionAddressError=" + extension_error).c_str());

        // Collect identity information through the exact library that returned the
        // direct platform count. Do not assume that a successful dlopen means this
        // is an ICD, or inject a non-ICD loader into OCL_ICD_FILENAMES.
        if (get_platforms && direct_result == CL_SUCCESS && direct_count > 0) {
            std::vector<cl_platform_id> platforms(direct_count);
            const cl_int platforms_result = get_platforms(direct_count, platforms.data(), nullptr);
            append_native_trace((std::string("OPENCL_DIRECT_PLATFORM_LIST path=") + path +
                " result=" + std::to_string((int)platforms_result) +
                " count=" + std::to_string(platforms_result == CL_SUCCESS ? direct_count : 0)).c_str());
            if (platforms_result == CL_SUCCESS) {
                for (cl_uint i = 0; i < direct_count; ++i) {
                    const std::string name = direct_platform_string(get_platform_info, platforms[i], CL_PLATFORM_NAME);
                    const std::string vendor = direct_platform_string(get_platform_info, platforms[i], CL_PLATFORM_VENDOR);
                    const std::string version = direct_platform_string(get_platform_info, platforms[i], CL_PLATFORM_VERSION);
                    const std::string icd_suffix = direct_platform_string(
                        get_platform_info, platforms[i], CL_PLATFORM_ICD_SUFFIX_KHR);
                    cl_uint gpu_count = 0;
                    const cl_int gpu_result = get_devices
                        ? get_devices(platforms[i], CL_DEVICE_TYPE_GPU, 0, nullptr, &gpu_count)
                        : (cl_int)-9999;
                    append_native_trace((std::string("OPENCL_DIRECT_PLATFORM path=") + path +
                        " index=" + std::to_string(i) + " name=" + name +
                        " vendor=" + vendor + " version=" + version +
                        " icdSuffix=" + icd_suffix +
                        " gpuQueryResult=" + std::to_string((int)gpu_result) +
                        " gpuQueryName=" + opencl_error_name(gpu_result) +
                        " gpuCount=" + std::to_string(gpu_count)).c_str());
                    if (gpu_result == CL_SUCCESS && gpu_count > 0 && get_device_info) {
                        std::vector<cl_device_id> devices(gpu_count);
                        const cl_int device_list_result = get_devices(
                            platforms[i], CL_DEVICE_TYPE_GPU, gpu_count, devices.data(), nullptr);
                        if (device_list_result == CL_SUCCESS) {
                            for (cl_uint d = 0; d < gpu_count; ++d) {
                                auto read_device_string = [&](cl_device_info field) -> std::string {
                                    size_t size = 0;
                                    if (get_device_info(devices[d], field, 0, nullptr, &size) != CL_SUCCESS || size == 0)
                                        return "<unavailable>";
                                    std::string value(size, '\0');
                                    if (get_device_info(devices[d], field, size, value.data(), nullptr) != CL_SUCCESS)
                                        return "<unavailable>";
                                    if (!value.empty() && value.back() == '\0') value.pop_back();
                                    return value;
                                };
                                append_native_trace((std::string("OPENCL_DIRECT_GPU path=") + path +
                                    " platform=" + std::to_string(i) + "." + std::to_string(d) +
                                    " name=" + read_device_string(CL_DEVICE_NAME) +
                                    " vendor=" + read_device_string(CL_DEVICE_VENDOR) +
                                    " driver=" + read_device_string(CL_DRIVER_VERSION) +
                                    " version=" + read_device_string(CL_DEVICE_VERSION)).c_str());
                            }
                        }
                    }
                }
            }
        }

        // Only a genuine ICD implementation with its ICD entry point and a
        // non-empty platform list can be passed to the bundled Khronos loader.
        if (get_icd_platforms && get_platform_info && get_extension_function_address) {
            cl_uint icd_count = 0;
            const cl_int icd_result = get_icd_platforms(0, nullptr, &icd_count);
            append_native_trace((std::string("OPENCL_VENDOR_ICD_VALIDATION path=") + path +
                " result=" + std::to_string((int)icd_result) +
                " name=" + opencl_error_name(icd_result) +
                " platformCount=" + std::to_string(icd_count)).c_str());
            if (icd_result == CL_SUCCESS && icd_count > 0) {
                if (setenv("OCL_ICD_FILENAMES", path, 1) == 0) {
                    append_native_trace((std::string("OPENCL_VENDOR_ICD_SELECTED path=") + path +
                        " reason=icd_entrypoint_and_platform_verified").c_str());
                    dlclose(handle);
                    append_native_trace("OPENCL_VENDOR_ICD_DIRECT_PROBE_COMPLETED selected=1");
                    return true;
                }
                append_native_trace((std::string("OPENCL_VENDOR_ICD_SELECT_FAILED path=") + path).c_str());
            }
        } else if (get_platforms && direct_result == CL_SUCCESS && direct_count > 0) {
            append_native_trace((std::string("OPENCL_DIRECT_API_AVAILABLE_BUT_NOT_ICD path=") + path +
                " reason=core_platforms_found_without_required_icd_entrypoints").c_str());
        }
        dlclose(handle);
    }

    append_native_trace("OPENCL_VENDOR_ICD_DIRECT_PROBE_COMPLETED selected=0");
    return false;
}

static void probe_android_opencl_driver() {
    append_native_trace("OPENCL_DRIVER_PROBE_STARTED");
    cl_uint platform_count = 0;
    const cl_int platform_result = clGetPlatformIDs(0, nullptr, &platform_count);
    append_native_trace((std::string("OPENCL_DRIVER_PROBE_PLATFORM_RESULT code=") +
        std::to_string((int)platform_result) + " name=" + opencl_error_name(platform_result) +
        " platformCount=" + std::to_string(platform_count)).c_str());

    if (platform_result != CL_SUCCESS || platform_count == 0) {
        append_native_trace("OPENCL_DRIVER_PROBE_NO_PLATFORM");
        return;
    }

    std::vector<cl_platform_id> platforms(platform_count);
    const cl_int list_result = clGetPlatformIDs(platform_count, platforms.data(), nullptr);
    append_native_trace((std::string("OPENCL_DRIVER_PROBE_PLATFORM_LIST code=") +
        std::to_string((int)list_result) + " name=" + opencl_error_name(list_result)).c_str());
    if (list_result != CL_SUCCESS) return;

    for (cl_uint i = 0; i < platform_count; ++i) {
        const std::string name = opencl_platform_info(platforms[i], CL_PLATFORM_NAME);
        const std::string vendor = opencl_platform_info(platforms[i], CL_PLATFORM_VENDOR);
        const std::string version = opencl_platform_info(platforms[i], CL_PLATFORM_VERSION);
        const std::string extensions = opencl_platform_info(platforms[i], CL_PLATFORM_EXTENSIONS);
        append_native_trace((std::string("OPENCL_PLATFORM index=") + std::to_string(i) +
            " name=" + name + " vendor=" + vendor + " version=" + version).c_str());
        append_native_trace((std::string("OPENCL_PLATFORM_EXTENSIONS index=") +
            std::to_string(i) + " value=" + extensions).c_str());

        cl_uint device_count = 0;
        const cl_int device_result = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_GPU, 0, nullptr, &device_count);
        append_native_trace((std::string("OPENCL_GPU_PROBE index=") + std::to_string(i) +
            " code=" + std::to_string((int)device_result) + " name=" +
            opencl_error_name(device_result) + " deviceCount=" +
            std::to_string(device_count)).c_str());
        if (device_result != CL_SUCCESS || device_count == 0) continue;

        std::vector<cl_device_id> devices(device_count);
        const cl_int device_list_result = clGetDeviceIDs(
            platforms[i], CL_DEVICE_TYPE_GPU, device_count, devices.data(), nullptr);
        append_native_trace((std::string("OPENCL_GPU_PROBE_LIST index=") +
            std::to_string(i) + " code=" + std::to_string((int)device_list_result) +
            " name=" + opencl_error_name(device_list_result)).c_str());
        if (device_list_result != CL_SUCCESS) continue;

        for (cl_uint d = 0; d < device_count; ++d) {
            auto device_info = [&](cl_device_info info) -> std::string {
                size_t size = 0;
                if (clGetDeviceInfo(devices[d], info, 0, nullptr, &size) != CL_SUCCESS || size == 0) {
                    return "<unavailable>";
                }
                std::string value(size, '\0');
                if (clGetDeviceInfo(devices[d], info, size, value.data(), nullptr) != CL_SUCCESS) {
                    return "<unavailable>";
                }
                if (!value.empty() && value.back() == '\0') value.pop_back();
                return value;
            };
            const std::string device_name = device_info(CL_DEVICE_NAME);
            const std::string device_vendor = device_info(CL_DEVICE_VENDOR);
            const std::string driver_version = device_info(CL_DRIVER_VERSION);
            const std::string device_version = device_info(CL_DEVICE_VERSION);
            const std::string opencl_c_version = device_info(CL_DEVICE_OPENCL_C_VERSION);
            const std::string device_extensions = device_info(CL_DEVICE_EXTENSIONS);
            cl_bool device_available = CL_FALSE;
            cl_bool compiler_available = CL_FALSE;
            cl_uint compute_units = 0;
            cl_ulong global_memory_bytes = 0;
            cl_ulong local_memory_bytes = 0;
            size_t max_work_group_size = 0;
            const cl_int available_status = clGetDeviceInfo(
                devices[d], CL_DEVICE_AVAILABLE, sizeof(device_available), &device_available, nullptr);
            const cl_int compiler_status = clGetDeviceInfo(
                devices[d], CL_DEVICE_COMPILER_AVAILABLE, sizeof(compiler_available), &compiler_available, nullptr);
            const cl_int compute_status = clGetDeviceInfo(
                devices[d], CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(compute_units), &compute_units, nullptr);
            const cl_int global_mem_status = clGetDeviceInfo(
                devices[d], CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(global_memory_bytes), &global_memory_bytes, nullptr);
            const cl_int local_mem_status = clGetDeviceInfo(
                devices[d], CL_DEVICE_LOCAL_MEM_SIZE, sizeof(local_memory_bytes), &local_memory_bytes, nullptr);
            const cl_int work_group_status = clGetDeviceInfo(
                devices[d], CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(max_work_group_size), &max_work_group_size, nullptr);
            // FP16 capability is reported through the standard extension string;
            // CL_DEVICE_HALF_FP_CONFIG is not exposed by all portable OpenCL headers.
            const bool has_fp16_extension =
                device_extensions.find("cl_khr_fp16") != std::string::npos;
            const bool has_subgroup_extension =
                device_extensions.find("cl_khr_subgroups") != std::string::npos ||
                device_extensions.find("cl_intel_subgroups") != std::string::npos;
            append_native_trace((std::string("OPENCL_GPU_DEVICE index=") + std::to_string(i) +
                "." + std::to_string(d) + " name=" + device_name +
                " vendor=" + device_vendor + " driver=" + driver_version +
                " version=" + device_version + " openclC=" + opencl_c_version).c_str());
            append_native_trace((std::string("OPENCL_DEVICE_CAPABILITIES index=") +
                std::to_string(i) + "." + std::to_string(d) +
                " available=" + (available_status == CL_SUCCESS && device_available == CL_TRUE ? "1" : "0") +
                " compilerAvailable=" + (compiler_status == CL_SUCCESS && compiler_available == CL_TRUE ? "1" : "0") +
                " computeUnits=" + (compute_status == CL_SUCCESS ? std::to_string(compute_units) : "unknown") +
                " globalMemoryBytes=" + (global_mem_status == CL_SUCCESS ? std::to_string(global_memory_bytes) : "unknown") +
                " localMemoryBytes=" + (local_mem_status == CL_SUCCESS ? std::to_string(local_memory_bytes) : "unknown") +
                " maxWorkGroupSize=" + (work_group_status == CL_SUCCESS ? std::to_string(max_work_group_size) : "unknown") +
                " fp16Extension=" + (has_fp16_extension ? "1" : "0") +
                " subgroupExtension=" + (has_subgroup_extension ? "1" : "0") +
                " selectionPolicy=UPSTREAM_KERNEL_CAPABILITY_GATE").c_str());
        }
    }
    append_native_trace("OPENCL_DRIVER_PROBE_COMPLETED");
}

static ggml_backend_dev_t find_webgpu_gpu_device() {
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (!dev) continue;
        const auto type = ggml_backend_dev_type(dev);
        if (type != GGML_BACKEND_DEVICE_TYPE_GPU && type != GGML_BACKEND_DEVICE_TYPE_IGPU) continue;
        ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
        const char * reg_name = reg ? ggml_backend_reg_name(reg) : nullptr;
        if (reg_name && (std::string(reg_name) == GGML_WEBGPU_NAME || std::string(reg_name) == "WEBGPU")) return dev;
    }
    return nullptr;
}

static bool is_webgpu_backend_device(ggml_backend_dev_t dev) {
    if (!dev) return false;
    ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
    const char * name = reg ? ggml_backend_reg_name(reg) : nullptr;
    return name && (std::string(name) == GGML_WEBGPU_NAME || std::string(name) == "WEBGPU");
}

static bool register_static_webgpu_backend() {
    if (ggml_backend_reg_by_name(GGML_WEBGPU_NAME) != nullptr ||
        ggml_backend_reg_by_name("WEBGPU") != nullptr) {
        append_native_trace("WEBGPU_STATIC_REGISTRATION_ALREADY_PRESENT");
        return true;
    }
    const size_t before = ggml_backend_reg_count();
    ggml_backend_register(ggml_backend_webgpu_reg());
    const size_t after = ggml_backend_reg_count();
    const bool registered = ggml_backend_reg_by_name(GGML_WEBGPU_NAME) != nullptr ||
        ggml_backend_reg_by_name("WEBGPU") != nullptr;
    append_native_trace((std::string("WEBGPU_STATIC_REGISTRATION before=") +
        std::to_string(before) + " after=" + std::to_string(after) +
        " registered=" + (registered ? "1" : "0")).c_str());
    return registered;
}

static bool has_webgpu_gpu_device() {
    ggml_backend_dev_t dev = find_webgpu_gpu_device();
    if (!dev) return false;
    const char * name = ggml_backend_dev_name(dev);
    const char * desc = ggml_backend_dev_description(dev);
    append_native_trace((std::string("WEBGPU_GPU_DEVICE_FOUND name=") +
        (name ? name : "unknown") + " description=" + (desc ? desc : "unknown")).c_str());
    return true;
}

static bool register_static_opencl_backend() {
    // GGML_BACKEND_DL is disabled on Android. Explicitly register the statically
    // linked OpenCL backend so runtime discovery never depends on APK filesystem
    // scanning or a MODULE shared library that cannot be loaded from the APK.
    // Re-registration is avoided so retrying a failed driver probe cannot create
    // duplicate backend registrations.
    if (ggml_backend_reg_by_name("OPENCL") != nullptr) {
        append_native_trace("OPENCL_STATIC_REGISTRATION_ALREADY_PRESENT");
        return true;
    }
    const size_t before = ggml_backend_reg_count();
    ggml_backend_register(ggml_backend_opencl_reg());
    const size_t after = ggml_backend_reg_count();
    const bool registered = ggml_backend_reg_by_name("OPENCL") != nullptr;
    append_native_trace((std::string("OPENCL_STATIC_REGISTRATION before=") +
        std::to_string(before) + " after=" + std::to_string(after) +
        " registered=" + (registered ? "1" : "0")).c_str());
    return registered;
}

static bool has_opencl_gpu_device() {
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (!dev) continue;
        const auto type = ggml_backend_dev_type(dev);
        if (type == GGML_BACKEND_DEVICE_TYPE_GPU || type == GGML_BACKEND_DEVICE_TYPE_IGPU) {
            const char * name = ggml_backend_dev_name(dev);
            const char * desc = ggml_backend_dev_description(dev);
            append_native_trace((std::string("OPENCL_GPU_DEVICE_FOUND name=") +
                (name ? name : "unknown") + " description=" +
                (desc ? desc : "unknown") + " type=" +
                std::to_string((int)type)).c_str());
            return true;
        }
    }
    return false;
}

static void append_native_trace(const char *text) {
    if (!text || !*text) return;
    if (g_native_trace_file.empty()) return;
    std::lock_guard<std::mutex> lock(g_native_marker_mutex);
    // Keep the persistent diagnostic trace bounded. The newest 1 MiB is enough
    // for crash triage and avoids unbounded app-private log growth over time.
    struct stat trace_stat{};
    if (stat(g_native_trace_file.c_str(), &trace_stat) == 0 &&
        trace_stat.st_size > 1024 * 1024) {
        std::ofstream reset(g_native_trace_file, std::ios::trunc);
        reset.close();
    }
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

static void native_write_decimal(int fd, const char *label, long value) {
    size_t label_len = 0;
    while (label[label_len]) ++label_len;
    native_write_text(fd, label, label_len);
    unsigned long magnitude = value < 0
        ? (unsigned long)(-(value + 1)) + 1ul
        : (unsigned long)value;
    char digits[24];
    size_t count = 0;
    do {
        digits[count++] = (char)('0' + (magnitude % 10ul));
        magnitude /= 10ul;
    } while (magnitude && count < sizeof(digits));
    if (value < 0) native_write_text(fd, "-", 1);
    while (count) native_write_text(fd, &digits[--count], 1);
    native_write_text(fd, "\n", 1);
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
    const long thread_id = syscall(SYS_gettid);
    if (g_native_fatal_fd >= 0) { 
        const char prefix[] = "NATIVE_FATAL_SIGNAL=";
        const char name_prefix[] = "NATIVE_FATAL_SIGNAL_NAME=";
        const char code_prefix[] = "NATIVE_FATAL_SI_CODE=";
        const char pc_prefix[] = "NATIVE_FATAL_PC=";
        const char lr_prefix[] = "NATIVE_FATAL_LR=";
        const char addr_prefix[] = "NATIVE_FATAL_FAULT_ADDR=";
        const char thread_prefix[] = "NATIVE_FATAL_THREAD_ID=";
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
        size_t name_len = 0;
        while (name[name_len]) ++name_len;
        native_write_text(g_native_fatal_fd, name, name_len);
        native_write_text(g_native_fatal_fd, "\n", 1);
        native_write_decimal(g_native_fatal_fd, code_prefix, info ? info->si_code : 0);
        native_write_decimal(g_native_fatal_fd, thread_prefix, thread_id);
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
        while (phase_len + 1 < sizeof(phase)) {
            const char current = g_native_spec_phase[phase_len].load(std::memory_order_relaxed);
            if (!current) break;
            phase[phase_len++] = current;
        }
        phase[phase_len] = '\0';
        native_write_text(g_native_fatal_fd, phase, phase_len);
        native_write_text(g_native_fatal_fd, "\n", 1);
        const char runtime_phase_prefix[] = "NATIVE_FATAL_RUNTIME_PHASE=";
        native_write_text(g_native_fatal_fd, runtime_phase_prefix, sizeof(runtime_phase_prefix) - 1);
        char runtime_phase[192];
        size_t runtime_phase_len = 0;
        while (runtime_phase_len + 1 < sizeof(runtime_phase)) {
            const char current = g_native_runtime_phase[runtime_phase_len].load(std::memory_order_relaxed);
            if (!current) break;
            runtime_phase[runtime_phase_len] = current;
            ++runtime_phase_len;
        }
        runtime_phase[runtime_phase_len] = '\0';
        native_write_text(g_native_fatal_fd, runtime_phase, runtime_phase_len);
        native_write_text(g_native_fatal_fd, "\n", 1);
        native_write_hex(g_native_fatal_fd, pc_prefix, pc);
        native_write_hex(g_native_fatal_fd, lr_prefix, lr);
        if (info && signal_number != SIGABRT) {
            native_write_hex(g_native_fatal_fd, addr_prefix, (uintptr_t)info->si_addr);
        }
        fsync(g_native_fatal_fd);
    }
    // Preserve the platform's default fatal-signal handling after writing our
    // compact phase snapshot. Android debuggerd can then emit a normal tombstone
    // with the complete native backtrace and Build IDs instead of losing it here.
    struct sigaction default_action{};
    sigemptyset(&default_action.sa_mask);
    default_action.sa_handler = SIG_DFL;
    if (sigaction(signal_number, &default_action, nullptr) == 0 &&
        syscall(SYS_tgkill, getpid(), thread_id, signal_number) == 0) {
        return; // pending signal is delivered with SIG_DFL when this handler returns
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
    // Only explicit checkpoints update the fatal phase snapshot. Raw ggml graph
    // diagnostics are not lifecycle phases and can contain very long tensor dumps.
    if (event) {
        size_t phase_len = 0;
        while (phase_len + 1 < sizeof(g_native_runtime_phase) &&
               event[phase_len] && event[phase_len] != '\n' && event[phase_len] != '\r') {
            g_native_runtime_phase[phase_len].store(event[phase_len], std::memory_order_relaxed);
            ++phase_len;
        }
        g_native_runtime_phase[phase_len].store('\0', std::memory_order_relaxed);
    }
    LOGI("NATIVE_CHECKPOINT %s", event ? event : "UNKNOWN");
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
    set_spec_phase("NATIVE_INIT");
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
                        // Keep the OpenCL program cache in an app-private subdirectory.
                        // llama.cpp already invalidates entries by kernel source, build
                        // options, device, driver and platform, so this is a persistence
                        // location only; it does not alter kernel selection.
                        const std::string cl_cache_dir = base + "/ai-chat-opencl-cache";
                        setenv("GGML_OPENCL_KERNEL_CACHE_DIR", cl_cache_dir.c_str(), 0);
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
        // Start the adapter's process-lifetime GLES context on its dedicated
        // owner thread before backend registration. This remains capability
        // discovery/self-test only; Dawn/GGML still executes model operations.
        const auto universal_gpu_caps = ai_chat::gpu::probe_universal_gpu_adapter();
        append_native_trace(universal_gpu_caps.diagnostic_line().c_str());
        g_universal_gles_version_known =
            universal_gpu_caps.egl_context_created && universal_gpu_caps.gles_major > 0;
        g_universal_gles_compute_supported = universal_gpu_caps.gles_compute_supported &&
            universal_gpu_caps.gles_compute_self_test_passed;
        append_native_trace((std::string("UNIVERSAL_GPU_ADAPTER_STATUS compute=") +
            (universal_gpu_caps.gles_compute_supported && universal_gpu_caps.gles_compute_self_test_passed
                ? "verified" : (universal_gpu_caps.gles_compute_supported ? "self_test_failed" : "unavailable")) +
            " eglContext=" + (universal_gpu_caps.egl_context_created ? "created" : "not_created") +
            " computeSelfTest=" + (universal_gpu_caps.gles_compute_self_test_passed ? "passed" : "failed") +
            " executionBackend=GGML_WEBGPU_DAWN").c_str());
        // Use the same portable OpenCL policy on every Android GPU vendor.
        // Do not force Q6_K kernel variants globally: let upstream choose the
        // supported implementation for the active OpenCL device.
        unsetenv("GGML_OPENCL_Q6K_GEMV_TILED");
        unsetenv("GGML_OPENCL_Q6K_GEMV_O4");
        unsetenv("GGML_OPENCL_Q6K_GEMV_O4_GLOBAL");
        append_native_trace("OPENCL_KERNEL_POLICY=UPSTREAM_DEVICE_CAPABILITY_SELECTION");
        // Allow upstream to select the Adreno xmem F16xF32 GEMM path where its
        // shape/device gates say it is beneficial. This does not affect the
        // one-token Q6_K GEMV path used for decode, but can reduce prompt-time
        // F16 GEMM cost without changing model math or quantization.
        unsetenv("GGML_OPENCL_ADRENO_XMEM_GEMM");
        unsetenv("GGML_DISABLE_OPENCL");

        // Do not point OCL_ICD_FILENAMES at /vendor/lib*/libOpenCL.so here.
        // On Android that path is commonly the system Khronos ICD loader itself,
        // not the vendor GPU implementation. Feeding the loader its own library
        // can recurse and make clGetPlatformIDs appear to hang with no platform.
        // Let the Android/Khronos loader discover the vendor ICD through its
        // registered vendor paths instead, and record the actual search paths.
        unsetenv("OCL_ICD_FILENAMES");
        unsetenv("AI_CHAT_OPENCL_ALLOW_DIRECT_PROVIDER");
        setenv("AI_CHAT_BACKEND_TRACE", "1", 0);
        append_native_trace("NATIVE_BACKEND_ASSIGNMENT_TRACE=one_shot_per_process");
        // Do not override the Android ICD vendor directory. Khronos documents
        // OCL_ICD_VENDORS as a replacement for the loader's default search path;
        // hard-coding guessed directories can hide the device vendor's real ICD.
        unsetenv("OCL_ICD_VENDORS");
        append_native_trace("OPENCL_ICD_FILENAMES_CLEARED_ANDROID_LOADER");
        append_native_trace("OPENCL_ICD_VENDORS_CLEARED_BEFORE_DISCOVERY");
        configure_android_icd_search_path();
        append_native_trace("OPENCL_Q6K_KERNEL_OVERRIDES=NONE");
        append_native_trace("OPENCL_STATIC_REGISTRATION_STARTED");
        probe_android_opencl_library_visibility();
        // Try a direct, verified ICD probe before the linked loader performs its
        // first enumeration. OCL_ICD_FILENAMES is set only for an ICD-compatible
        // library that reports a non-zero platform count; no guessed path is used.
        const bool direct_icd_selected = probe_android_vendor_icd_candidates();
        append_native_trace((std::string("OPENCL_VENDOR_ICD_SELECTION_RESULT selected=") +
            (direct_icd_selected ? "1" : "0")).c_str());
        probe_android_opencl_driver();
        const bool opencl_registered = register_static_opencl_backend();
        if (!opencl_registered) {
            append_native_trace("OPENCL_STATIC_REGISTRATION_FAILED");
            g_gpu_backend_loaded = false;
        } else {
            append_native_trace("OPENCL_BACKEND_LOAD_ALL_STARTED");
            // Keep this call for any auxiliary dynamically discoverable backends,
            // but OpenCL itself is already registered statically above.
            ggml_backend_load_all();
            append_native_trace("OPENCL_BACKEND_LOAD_ALL_RETURNED");
            g_gpu_backend_loaded = has_opencl_gpu_device();
        }
        if (opencl_registered) g_gpu_backend_loaded = has_opencl_gpu_device();
        append_native_trace((std::string("OPENCL_BACKEND_DEVICE_STATE loaded=") +
            (g_gpu_backend_loaded ? "1" : "0") +
            " deviceCount=" + std::to_string(ggml_backend_dev_count())).c_str());
        const bool webgpu_registered = register_static_webgpu_backend();
        if (webgpu_registered) {
            ggml_backend_load_all();
            g_webgpu_backend_loaded = has_webgpu_gpu_device();
        }
        append_native_trace((std::string("WEBGPU_BACKEND_DEVICE_STATE registered=") +
            (webgpu_registered ? "1" : "0") + " loaded=" +
            (g_webgpu_backend_loaded ? "1" : "0") +
            " adapterProfile=compatibility backend=opengles vulkan=disabled").c_str());
        append_native_trace("ACTIVATION_LLAMA_BACKEND_INIT_STARTED");
        llama_backend_init();
        append_native_trace("ACTIVATION_LLAMA_BACKEND_INIT_RETURNED");
        g_backend_initialized = true;
    } else {
        append_native_trace("OPENCL_BACKEND_ALREADY_INITIALIZED");
        if (!g_gpu_backend_loaded) {
            // A first probe can fail because Android's vendor OpenCL namespace is
            // not visible to the loader yet. Retry the visibility/driver probe on
            // later initialization attempts instead of permanently latching the
            // process into OPENCL-UNAVAILABLE.
            append_native_trace("OPENCL_BACKEND_RETRY_STARTED");
            probe_android_opencl_library_visibility();
            probe_android_opencl_driver();
            const bool opencl_registered = register_static_opencl_backend();
            if (opencl_registered) {
                ggml_backend_load_all();
                g_gpu_backend_loaded = has_opencl_gpu_device();
            }
            append_native_trace((std::string("OPENCL_BACKEND_RETRY_RESULT loaded=") +
                (g_gpu_backend_loaded ? "1" : "0") +
                " deviceCount=" + std::to_string(ggml_backend_dev_count())).c_str());
        }
        if (!g_webgpu_backend_loaded) {
            const bool webgpu_registered = register_static_webgpu_backend();
            if (webgpu_registered) {
                ggml_backend_load_all();
                g_webgpu_backend_loaded = has_webgpu_gpu_device();
            }
            append_native_trace((std::string("WEBGPU_BACKEND_RETRY_RESULT loaded=") +
                (g_webgpu_backend_loaded ? "1" : "0")).c_str());
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

        // WebGPU buffers may be host-visible on unified-memory devices. Classify
        // by the owning backend device before the host-buffer flag so real GPU
        // residency is not incorrectly reported as host memory.
        if (is_webgpu_backend_device(dev)) {
            gpu_bytes += bytes; gpu_tensors++; gpu_buffers.insert(buffer);
        } else if (is_host) {
            host_bytes += bytes; host_tensors++; host_buffers.insert(buffer);
        } else if (dev_type == GGML_BACKEND_DEVICE_TYPE_CPU) {
            cpu_bytes += bytes; cpu_tensors++; cpu_buffers.insert(buffer);
        } else {
            other_bytes += bytes; other_tensors++; other_buffers.insert(buffer);
        }
    }

    // Report memory from the selected WebGPU device only. OpenCL may also be
    // registered for legacy diagnostics, but it is not an inference fallback.
    ggml_backend_dev_t gpu_dev = nullptr;
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t candidate = ggml_backend_dev_get(i);
        if (is_webgpu_backend_device(candidate)) {
            gpu_dev = candidate;
            break;
        }
    }

    // If registry ordering is unusual, use the WebGPU device attached to a
    // model tensor as the authoritative runtime device.
    if (!gpu_dev) {
        for (const auto & entry : g_model->tensors_by_name) {
            const ggml_tensor * tensor = entry.second;
            if (!tensor || !tensor->buffer) continue;
            ggml_backend_dev_t candidate = ggml_backend_buft_get_device(
                    ggml_backend_buffer_get_type(tensor->buffer));
            if (!candidate) continue;
            if (is_webgpu_backend_device(candidate)) {
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

    append_native_trace((std::string("NATIVE_GPU_DEVICE") +
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
    // Keep the WebGPU model-load path on bounded reads. GPU-owned WebGPU buffers
    // must be allocated by the backend rather than imported from mmap host pointers.
    // CPU fallback uses llama.cpp's automatic load-mode selection independently.
    mp.load_mode = gpu ? LLAMA_LOAD_MODE_NONE : LLAMA_LOAD_MODE_AUTO;
    mp.check_tensors = false;
    mp.no_host = gpu;

    // Prefer WebGPU when requested; on any load/residency/context failure the caller
    // releases partial state and retries the same model on CPU.
    checkpoint(gpu ? "ANDROID_MODEL_LOAD_POLICY_WEBGPU_PREFERRED" : "ANDROID_MODEL_LOAD_POLICY_CPU_FALLBACK");
    checkpoint((std::string("ANDROID_MODEL_LOAD_PARAMS load_mode=") + llama_load_mode_name(mp.load_mode) +
        " check_tensors=" + (mp.check_tensors ? "1" : "0") +
        " no_host=" + (mp.no_host ? "1" : "0")).c_str());
    return llama_model_load_from_file(path, mp);
}

extern "C" JNIEXPORT jint JNICALL
Java_com_woogit_aicore_runtime_android_NativeLlamaCpp_nativeLoad(JNIEnv *env, jclass, jstring jpath, jint ctx_len, jint gpu_layers) {
    // Model/context mutation must never overlap tokenization or generation.
    // nativeStop intentionally remains lock-free so it can interrupt a decode.
    std::lock_guard<std::mutex> runtime_lock(g_native_runtime_mutex);
    checkpoint("NATIVE_LOAD_LOCK_ACQUIRED");
    checkpoint("NATIVE_LOAD_STARTED");
    append_native_trace((std::string("ACTIVATION_NATIVE_LOAD_STARTED ctx_len=") + std::to_string((int)ctx_len) + " gpu_layers=" + std::to_string((int)gpu_layers)).c_str());
    preserve_android_system_prefix_cache_for_active_model();
    checkpoint("FREE_OLD_RUNTIME_STARTED"); free_all(); checkpoint("FREE_OLD_RUNTIME_RETURNED"); g_stop.store(false);
    const char *path = env->GetStringUTFChars(jpath, nullptr);
    if (!path) { checkpoint("PATH_UTF8_FAILED"); return 3; }
    const std::string stable_model_path(path);
    append_native_trace("DRAFT_FEATURE_REMOVED");
    const std::string preflight = gguf_preflight(path); checkpoint(preflight.c_str());
    ggml_backend_dev_t webgpu_device = find_webgpu_gpu_device();
    const bool universal_gles_rejects_compute = g_universal_gles_version_known && !g_universal_gles_compute_supported;
    const bool gpu_available = g_webgpu_backend_loaded && webgpu_device != nullptr && gpu_layers != 0 && !universal_gles_rejects_compute;
    std::string gpu_failure_reason = "gpu_not_requested_or_unavailable";
    if (gpu_layers == 0) {
        gpu_failure_reason = "gpu_offload_disabled";
    } else if (universal_gles_rejects_compute) {
        gpu_failure_reason = "universal_gles_compute_not_supported";
    } else if (!ggml_backend_reg_by_name(GGML_WEBGPU_NAME) &&
               !ggml_backend_reg_by_name("WEBGPU")) {
        gpu_failure_reason = "webgpu_backend_not_registered";
    } else if (!g_webgpu_backend_loaded) {
        gpu_failure_reason = "webgpu_compatibility_adapter_unavailable";
    } else if (!webgpu_device) {
        gpu_failure_reason = "webgpu_no_gpu_device";
    }
    append_native_trace((std::string("NATIVE_EXECUTION_POLICY mode=WEBGPU_OPENGL_ES_COMPATIBILITY gpuRequested=") +
        (gpu_layers != 0 ? "1" : "0") +
        " webgpuAvailable=" + (gpu_available ? "1" : "0") +
        " glesComputeAvailable=" + (g_universal_gles_compute_supported ? "1" : "0") +
        " glesCapabilityKnown=" + (g_universal_gles_version_known ? "1" : "0") +
        " openclFallback=disabled wholeModelCpuFallback=enabled unsupportedOpPolicy=ggml_scheduler").c_str());

    const int requested = ctx_len > 0 ? ctx_len : 4096;
    bool runtime_ready = false;
    bool gpu_attempted = false;

    // GPU is an optimization, not a requirement for activating a valid model.
    // Any recoverable WebGPU model-load, residency-validation, or context-init
    // failure releases partial state and retries the same model with the CPU backend.
    if (gpu_available) {
        gpu_attempted = true;
        llama_model_params gpu_params = llama_model_default_params();
        const int effective_gpu_layers = gpu_layers == -1 ? 999 : gpu_layers;
        gpu_params.n_gpu_layers = effective_gpu_layers;
        gpu_params.load_mode = LLAMA_LOAD_MODE_NONE;
        gpu_params.check_tensors = false;
        ggml_backend_dev_t preferred_webgpu_devices[] = { webgpu_device, nullptr };
        gpu_params.devices = preferred_webgpu_devices;
        append_native_trace("MODEL_LOAD_STARTED_WEBGPU_PREFERRED");
        append_native_trace((std::string("MODEL_LOAD_PARAMS requestedGpuLayers=") +
            std::to_string(effective_gpu_layers) + " load_mode=" +
            llama_load_mode_name(gpu_params.load_mode) + " no_host=1").c_str());
        g_model = load_model_android(stable_model_path.c_str(), gpu_params, true);
        checkpoint(g_model ? "WEBGPU_MODEL_LOAD_RETURNED_SUCCESS" : "WEBGPU_MODEL_LOAD_RETURNED_FAILED");

        if (g_model) {
            append_weight_residency_trace();
            if (validate_gpu_model_residency()) {
                append_native_trace("WEBGPU_MODEL_RESIDENCY_VALIDATED");
                const int trained = llama_model_n_ctx_train(g_model);
                const int effective = std::max(1, std::min(requested, trained));
                g_context_length = (uint32_t)effective;
                checkpoint((std::string("WEBGPU_CONTEXT_INIT_STARTED effective_ctx=") +
                    std::to_string(effective)).c_str());
                runtime_ready = init_generation_context(true);
                checkpoint(runtime_ready ? "WEBGPU_CONTEXT_INIT_RETURNED_SUCCESS" : "WEBGPU_CONTEXT_INIT_RETURNED_FAILED");
                if (runtime_ready) {
                    g_gpu = true;
                    gpu_failure_reason = "none";
                    checkpoint((std::string("MODEL_READY execution=WEBGPU_OPENGL_ES_COMPATIBILITY ctx=") +
                        std::to_string(effective)).c_str());
                } else {
                    gpu_failure_reason = "webgpu_context_init_failed";
                }
            } else {
                gpu_failure_reason = "webgpu_weight_residency_below_useful_threshold";
            }
        } else {
            gpu_failure_reason = "webgpu_model_load_failed";
        }

        if (!runtime_ready) {
            append_native_trace((std::string("WEBGPU_ATTEMPT_FAILED reason=") + gpu_failure_reason +
                " action=release_partial_state_and_retry_cpu").c_str());
            free_all();
        }
    } else {
        append_native_trace("WEBGPU_ATTEMPT_SKIPPED reason=gpu_not_requested_or_compatibility_adapter_unavailable");
    }

    if (!runtime_ready) {
        append_native_trace((std::string("CPU_FALLBACK_STARTED gpuAttempted=") +
            (gpu_attempted ? "1" : "0") + " gpuFailureReason=" + gpu_failure_reason).c_str());
        llama_model_params cpu_params = llama_model_default_params();
        cpu_params.n_gpu_layers = 0;
        cpu_params.load_mode = LLAMA_LOAD_MODE_AUTO;
        cpu_params.check_tensors = false;
        g_model = load_model_android(stable_model_path.c_str(), cpu_params, false);
        checkpoint(g_model ? "CPU_MODEL_LOAD_RETURNED_SUCCESS" : "CPU_MODEL_LOAD_RETURNED_FAILED");
        // Refresh residency after a failed WebGPU attempt so diagnostics cannot
        // retain partial GPU buffers from the discarded model instance.
        if (g_model) append_weight_residency_trace();
        if (!g_model) {
            append_native_trace("CPU_FALLBACK_FAILED reason=model_load_failed");
            env->ReleaseStringUTFChars(jpath, path);
            g_gpu = false;
            return 1;
        }

        const int trained = llama_model_n_ctx_train(g_model);
        const int effective = std::max(1, std::min(requested, trained));
        g_context_length = (uint32_t)effective;
        checkpoint((std::string("CPU_CONTEXT_INIT_STARTED effective_ctx=") +
            std::to_string(effective)).c_str());
        runtime_ready = init_generation_context(false);
        checkpoint(runtime_ready ? "CPU_CONTEXT_INIT_RETURNED_SUCCESS" : "CPU_CONTEXT_INIT_RETURNED_FAILED");
        if (!runtime_ready) {
            append_native_trace("CPU_FALLBACK_FAILED reason=context_init_failed");
            free_all();
            env->ReleaseStringUTFChars(jpath, path);
            return 2;
        }
        g_gpu = false;
        checkpoint((std::string("MODEL_READY execution=CPU_FALLBACK ctx=") +
            std::to_string(effective)).c_str());
    } else {
        append_native_trace("CPU_FALLBACK_NOT_NEEDED");
    }
    append_native_trace((std::string("NATIVE_EXECUTION_PROFILE mode=WEBGPU_OPENGL_ES_COMPATIBILITY gpuBackendRegistered=") +
        (ggml_backend_reg_by_name(GGML_WEBGPU_NAME) || ggml_backend_reg_by_name("WEBGPU") ? "1" : "0") +
        " webgpuDeviceAvailable=" + (webgpu_device ? "1" : "0") +
        " selectedBackend=" + (g_gpu ? "WEBGPU" : "CPU") +
        " fallbackReason=" + gpu_failure_reason +
        " cpuFallbackAllowed=1").c_str());
    append_native_trace((std::string("TARGET_RUNTIME_READY execution=") + (g_gpu ? "WEBGPU_GPU" : "CPU_FALLBACK")).c_str());
    set_android_cache_model_key(stable_model_path, g_context_length);
    env->ReleaseStringUTFChars(jpath, path);
    checkpoint("NATIVE_LOAD_COMPLETED"); return 0;
}

extern "C" JNIEXPORT void JNICALL
Java_com_woogit_aicore_runtime_android_NativeLlamaCpp_nativeStop(JNIEnv *, jclass) { g_stop.store(true); }
extern "C" JNIEXPORT void JNICALL
Java_com_woogit_aicore_runtime_android_NativeLlamaCpp_nativeUnload(JNIEnv *, jclass) {
    g_stop.store(true, std::memory_order_relaxed);
    std::lock_guard<std::mutex> runtime_lock(g_native_runtime_mutex);
    checkpoint("NATIVE_UNLOAD_LOCK_ACQUIRED");
    preserve_android_system_prefix_cache_for_active_model();
    free_all();
    checkpoint("NATIVE_UNLOAD_COMPLETED");
}
extern "C" JNIEXPORT jstring JNICALL
Java_com_woogit_aicore_runtime_android_NativeLlamaCpp_nativeRuntimeInfo(JNIEnv *env, jclass) {
    std::lock_guard<std::mutex> runtime_lock(g_native_runtime_mutex);
    const std::string value = std::string(g_gpu && g_webgpu_backend_loaded
        ? "WebGPU-OpenGLES-GPU"
        : (g_model && g_context ? "CPU-FALLBACK" : "GPU_UNAVAILABLE")) +
        "; llama.cpp=" + AI_CHAT_LLAMA_CPP_SHA +
        "; generationThreads=" + std::to_string(generation_threads()) +
        "; batchThreads=" + std::to_string(batch_threads());
    return env->NewStringUTF(value.c_str());
}

extern "C" JNIEXPORT jint JNICALL
Java_com_woogit_aicore_runtime_android_NativeLlamaCpp_nativeContextLength(JNIEnv *, jclass) {
    std::lock_guard<std::mutex> runtime_lock(g_native_runtime_mutex);
    if (g_context) {
        return (jint) llama_n_ctx(g_context);
    }
    if (g_model) {
        return (jint) llama_model_n_ctx_train(g_model);
    }
    return 0;
}
