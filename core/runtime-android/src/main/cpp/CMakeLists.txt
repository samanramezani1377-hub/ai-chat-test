cmake_minimum_required(VERSION 3.22)
project(ai_chat_runtime LANGUAGES C CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(FetchContent)
set(LLAMA_CPP_TAG "9871df5911a03a813518bcd623ee2eba91bf32aa" CACHE STRING "Pinned llama.cpp revision with Qwen3.x/OpenCL/Q6_K support")
set(FETCHCONTENT_QUIET OFF)
set(FETCHCONTENT_BASE_DIR "${CMAKE_CURRENT_LIST_DIR}/../../.cxx/fetch-content" CACHE PATH "Stable FetchContent base directory")
set(FETCHCONTENT_UPDATES_DISCONNECTED ON CACHE BOOL "Do not update pinned FetchContent dependencies")

FetchContent_Declare(
    llama_cpp
    GIT_REPOSITORY https://github.com/ggml-org/llama.cpp.git
    GIT_TAG ${LLAMA_CPP_TAG}
    GIT_SHALLOW TRUE
)
FetchContent_Declare(
    opencl_headers
    GIT_REPOSITORY https://github.com/KhronosGroup/OpenCL-Headers.git
    GIT_TAG v2025.07.22
    GIT_SHALLOW TRUE
)
FetchContent_Declare(
    opencl_icd_loader
    GIT_REPOSITORY https://github.com/KhronosGroup/OpenCL-ICD-Loader.git
    GIT_TAG v2025.07.22
    GIT_SHALLOW TRUE
)

FetchContent_MakeAvailable(opencl_headers)
set(OPENCL_ICD_LOADER_HEADERS_DIR "${opencl_headers_SOURCE_DIR}" CACHE PATH "OpenCL headers for the ICD loader" FORCE)
set(OPENCL_ICD_LOADER_BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(ENABLE_OPENCL_LAYER OFF CACHE BOOL "" FORCE)
set(ENABLE_OPENCL_ICD_LOADER ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(opencl_icd_loader)

list(PREPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")

set(GGML_OPENCL ON CACHE BOOL "Keep legacy OpenCL backend available for diagnostics" FORCE)
# Android inference uses upstream GGML WebGPU linked to a native Dawn build.
# AndroidX supplies only the pinned Dawn source revision metadata; its bundled
# JNI library is not a public WebGPU C ABI library for native GGML consumers.
set(GGML_WEBGPU ON CACHE BOOL "Enable GGML WebGPU backend through native Dawn" FORCE)
if (NOT DEFINED AI_CHAT_DAWN_METADATA_FILE OR NOT EXISTS "${AI_CHAT_DAWN_METADATA_FILE}")
    message(FATAL_ERROR "AndroidX WebGPU Dawn build metadata was not extracted before CMake configuration")
endif()
# Android runtime links the OpenCL backend statically when available. Dynamic backend
# loading is unreliable on Android because ggml_backend_load_all() cannot discover
# MODULE libraries from the APK's native library namespace.
set(GGML_BACKEND_DL OFF CACHE BOOL "Disable dynamic ggml backend loading on Android" FORCE)
set(GGML_BACKEND_DIR "" CACHE STRING "No dynamic backend directory" FORCE)
# Disable Adreno-only optimized kernels. This does NOT make the pinned upstream
# backend vendor-neutral: its device gate and several kernel dispatch paths still
# target supported Adreno/Intel families. Do not bypass that gate for Mali or any
# unknown family until a real, tested generic kernel/dispatch path exists.
set(GGML_OPENCL_USE_ADRENO_KERNELS OFF CACHE BOOL "Disable Adreno-specific OpenCL kernels" FORCE)
set(GGML_OPENCL_USE_ADRENO_BIN_KERNELS OFF CACHE BOOL "Do not use vendor-specific Adreno binary kernels" FORCE)
set(GGML_VULKAN OFF CACHE BOOL "Vulkan is intentionally disabled" FORCE)
set(GGML_OPENCL_USE_SVM OFF CACHE BOOL "Avoid vendor SVM driver instability on Android" FORCE)

# Profiling is enabled by the Android release configuration so the diagnostics screen
# can expose real OpenCL kernel timing. Compare with profiling disabled only after
# correctness and GPU registration are verified.
option(AI_CHAT_OPENCL_PROFILING "Enable app-visible OpenCL GPU profiling" OFF)
if (AI_CHAT_OPENCL_PROFILING)
    set(GGML_OPENCL_PROFILING ON CACHE BOOL "Enable OpenCL event profiling" FORCE)
endif()
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_COMMON ON CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_MTMD ON CACHE BOOL "Build libmtmd for local image understanding" FORCE)
set(MTMD_VIDEO OFF CACHE BOOL "No video helper binaries are needed for screen images" FORCE)
set(LLAMA_BUILD_SERVER OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_APP OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_UI OFF CACHE BOOL "" FORCE)
set(LLAMA_OPENSSL OFF CACHE BOOL "" FORCE)
# Keep the generic ARM64 backend portable. Never compile the whole ggml CPU
# backend for the build host or force newer ISA features: phones in the same
# arm64-v8a ABI can differ in DotProd/i8mm support. KleidiAI builds its optimized
# microkernels separately and uses its compatible dispatch paths at runtime.
set(GGML_NATIVE OFF CACHE BOOL "Keep the generic CPU backend ISA-portable" FORCE)
if (ANDROID)
    if (NOT CMAKE_ANDROID_ARCH_ABI STREQUAL "arm64-v8a")
        message(FATAL_ERROR "AI Chat Test native runtime supports arm64-v8a only; got '${CMAKE_ANDROID_ARCH_ABI}'")
    endif()
    # Pin the generic backend to the ARMv8-A baseline. Do not raise this to
    # armv8.2-a/i8mm or a device-specific -mcpu: that would risk SIGILL on
    # otherwise supported arm64 phones. KleidiAI's own kernels remain enabled.
    set(GGML_CPU_ARM_ARCH "armv8-a" CACHE STRING "Portable generic ARM64 CPU baseline" FORCE)
    message(STATUS "AI_CHAT_CPU_ARCH_POLICY baseline=armv8-a native=OFF kleidiai=ON runtime_compatible_dispatch=ON")
endif()
set(GGML_CPU_KLEIDIAI ON CACHE BOOL "Enable portable ARM KleidiAI CPU microkernels" FORCE)

# OpenMP is enabled for the CPU backend. Android's NDK Clang requires static
# OpenMP runtime linkage so the APK does not depend on a system libomp.so.
set(GGML_OPENMP ON CACHE BOOL "Enable GGML OpenMP CPU parallelism" FORCE)
if (ANDROID)
    set(OpenMP_C_FLAGS "-fopenmp -static-openmp" CACHE STRING "Android static OpenMP C flags" FORCE)
    set(OpenMP_CXX_FLAGS "-fopenmp -static-openmp" CACHE STRING "Android static OpenMP C++ flags" FORCE)
endif()
set(GGML_LLAMAFILE OFF CACHE BOOL "" FORCE)

FetchContent_MakeAvailable(llama_cpp)

# Force the pinned upstream WebGPU adapter to the no-Vulkan OpenGL ES compatibility path.
set(AI_CHAT_WEBGPU_SOURCE "${llama_cpp_SOURCE_DIR}/ggml/src/ggml-webgpu/ggml-webgpu.cpp")
if (NOT EXISTS "${AI_CHAT_WEBGPU_SOURCE}")
    message(FATAL_ERROR "Pinned llama.cpp revision does not contain the GGML WebGPU backend")
endif()
file(READ "${AI_CHAT_WEBGPU_SOURCE}" AI_CHAT_WEBGPU_SOURCE_TEXT)
string(FIND "${AI_CHAT_WEBGPU_SOURCE_TEXT}" "wgpu::RequestAdapterOptions options = {};" AI_CHAT_WEBGPU_OPTIONS_POS)
if (AI_CHAT_WEBGPU_OPTIONS_POS EQUAL -1)
    message(FATAL_ERROR "Could not locate GGML WebGPU adapter options in pinned source")
endif()
string(FIND "${AI_CHAT_WEBGPU_SOURCE_TEXT}" "options.featureLevel = wgpu::FeatureLevel::Compatibility;" AI_CHAT_WEBGPU_COMPAT_ALREADY_SET)
if (AI_CHAT_WEBGPU_COMPAT_ALREADY_SET EQUAL -1)
    string(REPLACE
        "wgpu::RequestAdapterOptions options = {};"
        "wgpu::RequestAdapterOptions options = {};\n    options.featureLevel = wgpu::FeatureLevel::Compatibility;"
        AI_CHAT_WEBGPU_SOURCE_TEXT "${AI_CHAT_WEBGPU_SOURCE_TEXT}")
endif()
string(FIND "${AI_CHAT_WEBGPU_SOURCE_TEXT}" "options.backendType = wgpu::BackendType::OpenGLES;" AI_CHAT_WEBGPU_GLES_ALREADY_SET)
if (AI_CHAT_WEBGPU_GLES_ALREADY_SET EQUAL -1)
    string(REPLACE
        "options.featureLevel = wgpu::FeatureLevel::Compatibility;"
        "options.featureLevel = wgpu::FeatureLevel::Compatibility;\n    options.backendType = wgpu::BackendType::OpenGLES;"
        AI_CHAT_WEBGPU_SOURCE_TEXT "${AI_CHAT_WEBGPU_SOURCE_TEXT}")
endif()
string(FIND "${AI_CHAT_WEBGPU_SOURCE_TEXT}" "options.featureLevel = wgpu::FeatureLevel::Compatibility;" AI_CHAT_WEBGPU_COMPAT_VERIFY)
string(FIND "${AI_CHAT_WEBGPU_SOURCE_TEXT}" "options.backendType = wgpu::BackendType::OpenGLES;" AI_CHAT_WEBGPU_GLES_VERIFY)
if (AI_CHAT_WEBGPU_COMPAT_VERIFY EQUAL -1 OR AI_CHAT_WEBGPU_GLES_VERIFY EQUAL -1)
    message(FATAL_ERROR "Could not enforce both WebGPU Compatibility and OpenGL ES adapter selection")
endif()
# Remove Vulkan-only Dawn toggles from the Android OpenGL ES request.
string(FIND "${AI_CHAT_WEBGPU_SOURCE_TEXT}" "const char * const          adapterEnabledToggles[]" AI_CHAT_WEBGPU_VULKAN_TOGGLE_POS)
if (NOT AI_CHAT_WEBGPU_VULKAN_TOGGLE_POS EQUAL -1)
    string(REPLACE
        "#ifndef __EMSCRIPTEN__\n    // TODO: track need for these toggles: https://issues.chromium.org/issues/42251215\n    const char * const          adapterEnabledToggles[] = { \"vulkan_enable_f16_on_nvidia\", \"use_vulkan_memory_model\" };\n    wgpu::DawnTogglesDescriptor adapterTogglesDesc;\n    adapterTogglesDesc.enabledToggles     = adapterEnabledToggles;\n    adapterTogglesDesc.enabledToggleCount = 2;\n    options.nextInChain                   = &adapterTogglesDesc;\n#endif"
        ""
        AI_CHAT_WEBGPU_SOURCE_TEXT "${AI_CHAT_WEBGPU_SOURCE_TEXT}")
endif()
string(FIND "${AI_CHAT_WEBGPU_SOURCE_TEXT}" "vulkan_enable_f16_on_nvidia" AI_CHAT_WEBGPU_VULKAN_TOGGLE_REMAINS)
if (NOT AI_CHAT_WEBGPU_VULKAN_TOGGLE_REMAINS EQUAL -1)
    message(FATAL_ERROR "Vulkan-specific Dawn toggles remain in the forced OpenGL ES WebGPU path")
endif()
# AI_CHAT_GENERIC_FP32_FALLBACK_V1
# Make ShaderF16 optional. Devices without it may execute the strictly
# F32/I32-safe subset plus the separately verified quantized F32 mat-vec path.
# F16 operations and larger quantized prompt-prefill GEMMs remain on CPU until
# dedicated non-F16 kernels are implemented and validated. Never spoof a feature
# or compile an unsupported WGSL shader and hope that the driver accepts it.
# Emit Dawn's actual WebGPU feature/limit snapshot separately from the GLES probe.
# The two reports must not be conflated: GLES extensions do not imply WebGPU features.
string(FIND "${AI_CHAT_WEBGPU_SOURCE_TEXT}" "AI_CHAT_WEBGPU_CAPABILITIES" AI_CHAT_WEBGPU_CAPABILITIES_LOGGED)
if (AI_CHAT_WEBGPU_CAPABILITIES_LOGGED EQUAL -1)
    set(AI_CHAT_WEBGPU_CAPABILITIES_ANCHOR "ctx->webgpu_global_ctx->capabilities.max_subgroup_size = info.subgroupMaxSize;")
    string(FIND "${AI_CHAT_WEBGPU_SOURCE_TEXT}" "${AI_CHAT_WEBGPU_CAPABILITIES_ANCHOR}" AI_CHAT_WEBGPU_CAPABILITIES_POS)
    if (AI_CHAT_WEBGPU_CAPABILITIES_POS EQUAL -1)
        message(FATAL_ERROR "Could not locate Dawn WebGPU capability diagnostic anchor")
    endif()
    string(REPLACE
        "${AI_CHAT_WEBGPU_CAPABILITIES_ANCHOR}"
        "${AI_CHAT_WEBGPU_CAPABILITIES_ANCHOR}
    // AI_CHAT_WEBGPU_CAPABILITIES: report actual Dawn features, not GLES hints.
    GGML_LOG_INFO(\"AI_CHAT_WEBGPU_CAPABILITIES vendor=%s shaderF16=%d subgroups=%d dotProduct=%d subgroupMatrix=%d maxStorageBufferBindingSize=%llu maxComputeInvocationsPerWorkgroup=%u maxComputeWorkgroupStorageSize=%u\n\",
        ctx->webgpu_global_ctx->vendor.c_str(),
        static_cast<int>(ctx->webgpu_global_ctx->capabilities.supports_shader_f16),
        static_cast<int>(ctx->webgpu_global_ctx->capabilities.supports_subgroups),
        static_cast<int>(ctx->webgpu_global_ctx->capabilities.supports_dot_product),
        static_cast<int>(ctx->webgpu_global_ctx->capabilities.supports_subgroup_matrix),
        static_cast<unsigned long long>(ctx->webgpu_global_ctx->capabilities.limits.maxStorageBufferBindingSize),
        ctx->webgpu_global_ctx->capabilities.limits.maxComputeInvocationsPerWorkgroup,
        ctx->webgpu_global_ctx->capabilities.limits.maxComputeWorkgroupStorageSize);"
        AI_CHAT_WEBGPU_SOURCE_TEXT "${AI_CHAT_WEBGPU_SOURCE_TEXT}")
endif()
string(FIND "${AI_CHAT_WEBGPU_SOURCE_TEXT}" "AI_CHAT_WEBGPU_CAPABILITIES" AI_CHAT_WEBGPU_CAPABILITIES_VERIFY)
if (AI_CHAT_WEBGPU_CAPABILITIES_VERIFY EQUAL -1)
    message(FATAL_ERROR "Could not add the Dawn WebGPU capability diagnostic")
endif()

string(FIND "${AI_CHAT_WEBGPU_SOURCE_TEXT}" "AI_CHAT_GENERIC_FP32_FALLBACK_V1" AI_CHAT_WEBGPU_FP32_PATCHED)
if (AI_CHAT_WEBGPU_FP32_PATCHED EQUAL -1)
    string(FIND "${AI_CHAT_WEBGPU_SOURCE_TEXT}" "bool         supports_subgroups       = false;" AI_CHAT_WEBGPU_FP32_CAPS_ANCHOR)
    string(FIND "${AI_CHAT_WEBGPU_SOURCE_TEXT}" "ctx->webgpu_global_ctx->vendor                    = info.vendor;" AI_CHAT_WEBGPU_FP32_VENDOR_ANCHOR)
    string(FIND "${AI_CHAT_WEBGPU_SOURCE_TEXT}" "std::vector<wgpu::FeatureName> required_features       = { wgpu::FeatureName::ShaderF16 };" AI_CHAT_WEBGPU_FP32_FEATURE_ANCHOR)
    string(FIND "${AI_CHAT_WEBGPU_SOURCE_TEXT}" "ggml_tensor * src2 = op->src[2];" AI_CHAT_WEBGPU_FP32_SUPPORT_ANCHOR)
    string(FIND "${AI_CHAT_WEBGPU_SOURCE_TEXT}" "if (adapter != nullptr && adapter.HasFeature(wgpu::FeatureName::ShaderF16)" AI_CHAT_WEBGPU_FP32_REG_ANCHOR)
    if (AI_CHAT_WEBGPU_FP32_CAPS_ANCHOR EQUAL -1 OR
        AI_CHAT_WEBGPU_FP32_VENDOR_ANCHOR EQUAL -1 OR
        AI_CHAT_WEBGPU_FP32_FEATURE_ANCHOR EQUAL -1 OR
        AI_CHAT_WEBGPU_FP32_SUPPORT_ANCHOR EQUAL -1 OR
        AI_CHAT_WEBGPU_FP32_REG_ANCHOR EQUAL -1)
        message(FATAL_ERROR "Pinned GGML WebGPU source no longer matches the optional ShaderF16 patch anchors")
    endif()

    string(REPLACE
        "bool         supports_subgroups       = false;"
        "bool         supports_shader_f16      = false;\n    bool         supports_subgroups       = false;"
        AI_CHAT_WEBGPU_SOURCE_TEXT "${AI_CHAT_WEBGPU_SOURCE_TEXT}")
    string(REPLACE
        "ctx->webgpu_global_ctx->vendor                    = info.vendor;"
        "ctx->webgpu_global_ctx->vendor                    = info.vendor;\n    ctx->webgpu_global_ctx->capabilities.supports_shader_f16 =\n        ctx->webgpu_global_ctx->adapter.HasFeature(wgpu::FeatureName::ShaderF16);\n    if (!ctx->webgpu_global_ctx->capabilities.supports_shader_f16) {\n        GGML_LOG_WARN(\"AI_CHAT_WEBGPU_FP16_FAST_KERNELS_DISABLED reason=shader_f16_unavailable; quantized_f32_matvec_fallback=enabled; prompt_prefill_gemm_fallback=portable_f32_register_tile;quantized_prefill_gemm_fallback=portable_f32_register_tile\n\");\n    }"
        AI_CHAT_WEBGPU_SOURCE_TEXT "${AI_CHAT_WEBGPU_SOURCE_TEXT}")
    string(REPLACE
        "std::vector<wgpu::FeatureName> required_features       = { wgpu::FeatureName::ShaderF16 };"
        "std::vector<wgpu::FeatureName> required_features;\n    if (ctx->webgpu_global_ctx->capabilities.supports_shader_f16) {\n        required_features.push_back(wgpu::FeatureName::ShaderF16);\n    }"
        AI_CHAT_WEBGPU_SOURCE_TEXT "${AI_CHAT_WEBGPU_SOURCE_TEXT}")
    string(REPLACE
        "    ggml_tensor * src2 = op->src[2];"
        "    ggml_tensor * src2 = op->src[2];\n\n    // AI_CHAT_GENERIC_FP32_FALLBACK_V1: conservative dispatch when ShaderF16 is absent.\n    if (!ctx->webgpu_global_ctx->capabilities.supports_shader_f16) {\n        bool uses_f16 = op->type == GGML_TYPE_F16;\n        bool uses_quantized = ggml_is_quantized(op->type);\n        // Inspect every possible source, not only src0-src2. Recurrent/attention\n        // operations may carry additional tensors in src3-src6.\n        for (int src_index = 0; src_index < GGML_MAX_SRC; ++src_index) {\n            const ggml_tensor * source = op->src[src_index];\n            if (source == nullptr) continue;\n            uses_f16 = uses_f16 || source->type == GGML_TYPE_F16;\n            uses_quantized = uses_quantized || ggml_is_quantized(source->type);\n        }\n        if (uses_f16 || uses_quantized) return false;\n    }"
        AI_CHAT_WEBGPU_SOURCE_TEXT "${AI_CHAT_WEBGPU_SOURCE_TEXT}")
    string(REPLACE
        "if (adapter != nullptr && adapter.HasFeature(wgpu::FeatureName::ShaderF16)"
        "if (adapter != nullptr"
        AI_CHAT_WEBGPU_SOURCE_TEXT "${AI_CHAT_WEBGPU_SOURCE_TEXT}")
endif()
string(FIND "${AI_CHAT_WEBGPU_SOURCE_TEXT}" "AI_CHAT_GENERIC_FP32_FALLBACK_V1" AI_CHAT_WEBGPU_FP32_PATCH_VERIFY)
string(FIND "${AI_CHAT_WEBGPU_SOURCE_TEXT}" "supports_shader_f16" AI_CHAT_WEBGPU_FP32_CAPABILITY_VERIFY)
if (AI_CHAT_WEBGPU_FP32_PATCH_VERIFY EQUAL -1 OR AI_CHAT_WEBGPU_FP32_CAPABILITY_VERIFY EQUAL -1)
    message(FATAL_ERROR "Could not apply the conservative optional ShaderF16 / FP32 fallback patch")
endif()

# The existing shader library contains unconditional enable-f16 modules (for
# example softmax and matrix-vector paths). On no-ShaderF16 adapters, do not
# merely allow every F32 tensor: admit only operations whose selected upstream
# WGSL path is verified F32-only. This whitelist can expand only with shader tests.
string(FIND "${AI_CHAT_WEBGPU_SOURCE_TEXT}" "AI_CHAT_GENERIC_FP32_OP_WHITELIST_V2" AI_CHAT_WEBGPU_FP32_WHITELIST_PATCHED)
if (AI_CHAT_WEBGPU_FP32_WHITELIST_PATCHED EQUAL -1)
    string(FIND "${AI_CHAT_WEBGPU_SOURCE_TEXT}" "if (uses_f16 || uses_quantized) return false;\n    }" AI_CHAT_WEBGPU_FP32_WHITELIST_ANCHOR)
    if (AI_CHAT_WEBGPU_FP32_WHITELIST_ANCHOR EQUAL -1)
        message(FATAL_ERROR "Could not locate the optional ShaderF16 operation-dispatch guard")
    endif()
    string(REPLACE
        "if (uses_f16 || uses_quantized) return false;\n    }"
        "const bool f32_quantized_matmul = op->op == GGML_OP_MUL_MAT && op->type == GGML_TYPE_F32 && op->src[0] != nullptr && ggml_is_quantized(op->src[0]->type) && op->src[1] != nullptr && op->src[1]->type == GGML_TYPE_F32;\n        const bool f32_plain_matmul = op->op == GGML_OP_MUL_MAT && op->type == GGML_TYPE_F32 && op->src[0] != nullptr && op->src[0]->type == GGML_TYPE_F32 && op->src[1] != nullptr && op->src[1]->type == GGML_TYPE_F32;\n        if (uses_f16 || (uses_quantized && !f32_quantized_matmul)) return false;\n        const bool f32_i32_safe_op =\n            // llama_model::select_buft() probes a plain F32 ADD to choose each layer buffer.\n            // Without this entry, no-ShaderF16 adapters reject the GPU buffer even for\n            // otherwise supported F32 tensors, forcing layer placement back to CPU.\n            op->op == GGML_OP_ADD || op->op == GGML_OP_ADD_ID || op->op == GGML_OP_CONCAT || op->op == GGML_OP_SET ||\n            op->op == GGML_OP_RMS_NORM || op->op == GGML_OP_NORM || op->op == GGML_OP_L2_NORM ||\n            op->op == GGML_OP_SCALE || op->op == GGML_OP_UNARY || op->op == GGML_OP_CLAMP ||\n            op->op == GGML_OP_FILL || op->op == GGML_OP_LOG || op->op == GGML_OP_SQR ||\n            op->op == GGML_OP_SQRT || op->op == GGML_OP_SIN || op->op == GGML_OP_COS ||\n            op->op == GGML_OP_SUM || op->op == GGML_OP_SUM_ROWS || op->op == GGML_OP_CUMSUM ||\n            op->op == GGML_OP_ARGMAX || op->op == GGML_OP_ARGSORT || op->op == GGML_OP_TOP_K ||\n            op->op == GGML_OP_SOFT_MAX || op->op == GGML_OP_DIAG || op->op == GGML_OP_TRI ||\n            op->op == GGML_OP_SOLVE_TRI || op->op == GGML_OP_PAD || op->op == GGML_OP_UPSCALE ||\n            op->op == GGML_OP_SSM_CONV || op->op == GGML_OP_SSM_SCAN || op->op == GGML_OP_GATED_DELTA_NET;\n        if (!f32_i32_safe_op && !f32_quantized_matmul && !f32_plain_matmul) return false;\n    }\n    // AI_CHAT_GENERIC_FP32_OP_WHITELIST_V2\n    // AI_CHAT_NO_SHADER_F16_QUANT_MATMUL_OP_V4\n    // AI_CHAT_NO_SHADER_F16_QUANTIZED_GEMM_OP_V1
endif()
string(FIND "${AI_CHAT_WEBGPU_SOURCE_TEXT}" "AI_CHAT_GENERIC_FP32_OP_WHITELIST_V2" AI_CHAT_WEBGPU_FP32_WHITELIST_VERIFY)
if (AI_CHAT_WEBGPU_FP32_WHITELIST_VERIFY EQUAL -1)
    message(FATAL_ERROR "Could not apply the conservative F32-only operation whitelist")
endif()

# The upstream softmax shader only needs ShaderF16 for an F16 mask. Make that
# WGSL feature declaration conditional so F32/no-mask softmax can run on
# adapters without ShaderF16; F16 masks remain rejected by supports_op.
set(AI_CHAT_WEBGPU_SOFTMAX_SOURCE "${llama_cpp_SOURCE_DIR}/ggml/src/ggml-webgpu/wgsl-shaders/soft_max.wgsl")
if (NOT EXISTS "${AI_CHAT_WEBGPU_SOFTMAX_SOURCE}")
    message(FATAL_ERROR "Pinned GGML WebGPU softmax shader source is missing")
endif()
file(READ "${AI_CHAT_WEBGPU_SOFTMAX_SOURCE}" AI_CHAT_WEBGPU_SOFTMAX_TEXT)
string(FIND "${AI_CHAT_WEBGPU_SOFTMAX_TEXT}" "AI_CHAT_SOFTMAX_FP32_NO_F16_V1" AI_CHAT_WEBGPU_SOFTMAX_PATCHED)
if (AI_CHAT_WEBGPU_SOFTMAX_PATCHED EQUAL -1)
    string(FIND "${AI_CHAT_WEBGPU_SOFTMAX_TEXT}" "enable f16;" AI_CHAT_WEBGPU_SOFTMAX_ENABLE_ANCHOR)
    if (AI_CHAT_WEBGPU_SOFTMAX_ENABLE_ANCHOR EQUAL -1)
        message(FATAL_ERROR "Could not locate the upstream softmax ShaderF16 declaration")
    endif()
    string(REPLACE
        "enable f16;"
        "#ifdef MASK_F16\nenable f16;\n#endif\n// AI_CHAT_SOFTMAX_FP32_NO_F16_V1"
        AI_CHAT_WEBGPU_SOFTMAX_TEXT "${AI_CHAT_WEBGPU_SOFTMAX_TEXT}")
    file(WRITE "${AI_CHAT_WEBGPU_SOFTMAX_SOURCE}" "${AI_CHAT_WEBGPU_SOFTMAX_TEXT}")
endif()
string(FIND "${AI_CHAT_WEBGPU_SOFTMAX_TEXT}" "AI_CHAT_SOFTMAX_FP32_NO_F16_V1" AI_CHAT_WEBGPU_SOFTMAX_VERIFY)
if (AI_CHAT_WEBGPU_SOFTMAX_VERIFY EQUAL -1)
    message(FATAL_ERROR "Could not apply the F32-only softmax WGSL patch")
endif()


# AI_CHAT_WEBGPU_NO_SHADER_F16_QUANT_MATVEC_V1
# The upstream quantized mat-vec shaders store FP16 scale pairs, but FP16
# arithmetic is not required to consume them. Keep the exact 16-bit storage
# layout and decode the values with baseline WGSL unpack2x16float(), which
# returns vec2<f32>. This enables the existing integer-packed Q4/Q5/Q6/Q8
# dequantization kernels on adapters without the optional ShaderF16 feature.
function(ai_chat_replace_once var_name old_text new_text description)
    string(FIND "${${var_name}}" "${old_text}" _patch_pos)
    if (_patch_pos EQUAL -1)
        message(FATAL_ERROR "Could not apply no-ShaderF16 WebGPU patch: ${description}")
    endif()
    string(REPLACE "${old_text}" "${new_text}" _patch_result "${${var_name}}")
    set(${var_name} "${_patch_result}" PARENT_SCOPE)
endfunction()

set(AI_CHAT_WEBGPU_SHADER_DIR "${llama_cpp_SOURCE_DIR}/ggml/src/ggml-webgpu/wgsl-shaders")
set(AI_CHAT_WEBGPU_COMMON_DECLS "${AI_CHAT_WEBGPU_SHADER_DIR}/common_decls.tmpl")
set(AI_CHAT_WEBGPU_MMV "${AI_CHAT_WEBGPU_SHADER_DIR}/mul_mat_vec.wgsl")
set(AI_CHAT_WEBGPU_MMV_ACC "${AI_CHAT_WEBGPU_SHADER_DIR}/mul_mat_vec_acc.tmpl")
set(AI_CHAT_WEBGPU_MMV_Q_ACC "${AI_CHAT_WEBGPU_SHADER_DIR}/mul_mat_vec_q_acc.tmpl")
set(AI_CHAT_WEBGPU_Q8 "${AI_CHAT_WEBGPU_SHADER_DIR}/quantize_q8.wgsl")

foreach(AI_CHAT_SHADER_FILE IN ITEMS
    "${AI_CHAT_WEBGPU_COMMON_DECLS}"
    "${AI_CHAT_WEBGPU_MMV}"
    "${AI_CHAT_WEBGPU_MMV_ACC}"
    "${AI_CHAT_WEBGPU_MMV_Q_ACC}"
    "${AI_CHAT_WEBGPU_Q8}")
    if (NOT EXISTS "${AI_CHAT_SHADER_FILE}")
        message(FATAL_ERROR "Required pinned WebGPU quantized mat-vec shader is missing: ${AI_CHAT_SHADER_FILE}")
    endif()
endforeach()

file(READ "${AI_CHAT_WEBGPU_COMMON_DECLS}" AI_CHAT_WEBGPU_COMMON_TEXT)
string(FIND "${AI_CHAT_WEBGPU_COMMON_TEXT}" "AI_CHAT_NO_SHADER_F16_COMMON_V1" AI_CHAT_WEBGPU_COMMON_PATCHED)
if (AI_CHAT_WEBGPU_COMMON_PATCHED EQUAL -1)
    ai_chat_replace_once(AI_CHAT_WEBGPU_COMMON_TEXT
        "fn f16_from_u16(bits: u32) -> f16 {\n    let packed = unpack2x16float(bits);\n    return f16(packed[0]);\n}"
        "#ifdef NO_SHADER_F16\nfn f16_from_u16(bits: u32) -> f32 {\n    return unpack2x16float(bits)[0];\n}\n#else\nfn f16_from_u16(bits: u32) -> f16 {\n    let packed = unpack2x16float(bits);\n    return f16(packed[0]);\n}\n#endif"
        "FP16 scalar decode helper")
    ai_chat_replace_once(AI_CHAT_WEBGPU_COMMON_TEXT
        "fn load_f16_at_src(byte_offset: u32) -> f16 {\n    let packed = unpack2x16float(load_u16_at_src(byte_offset));\n    return f16(packed[0]);\n}"
        "#ifdef NO_SHADER_F16\nfn load_f16_at_src(byte_offset: u32) -> f32 {\n    return unpack2x16float(load_u16_at_src(byte_offset))[0];\n}\n#else\nfn load_f16_at_src(byte_offset: u32) -> f16 {\n    let packed = unpack2x16float(load_u16_at_src(byte_offset));\n    return f16(packed[0]);\n}\n#endif"
        "FP16 source decode helper")
    ai_chat_replace_once(AI_CHAT_WEBGPU_COMMON_TEXT
        "fn load_f16_at_src0(byte_offset: u32) -> f16 {\n    let packed = unpack2x16float(load_u16_at_src0(byte_offset));\n    return f16(packed[0]);\n}"
        "#ifdef NO_SHADER_F16\nfn load_f16_at_src0(byte_offset: u32) -> f32 {\n    return unpack2x16float(load_u16_at_src0(byte_offset))[0];\n}\n#else\nfn load_f16_at_src0(byte_offset: u32) -> f16 {\n    let packed = unpack2x16float(load_u16_at_src0(byte_offset));\n    return f16(packed[0]);\n}\n#endif"
        "FP16 source0 decode helper")
    ai_chat_replace_once(AI_CHAT_WEBGPU_COMMON_TEXT
        "struct q4_1 {\n    d: f16,\n    m: f16,\n    qs: array<u32, 4>\n};"
        "struct q4_1 {\n#ifdef NO_SHADER_F16\n    dm: u32,\n#else\n    d: f16,\n    m: f16,\n#endif\n    qs: array<u32, 4>\n};"
        "Q4_1 packed scale layout")
    ai_chat_replace_once(AI_CHAT_WEBGPU_COMMON_TEXT
        "struct q5_1 {\n    d: f16,\n    m: f16,\n    qh: u32,\n    qs: array<u32, 4>\n};"
        "struct q5_1 {\n#ifdef NO_SHADER_F16\n    dm: u32,\n#else\n    d: f16,\n    m: f16,\n#endif\n    qh: u32,\n    qs: array<u32, 4>\n};"
        "Q5_1 packed scale layout")
    ai_chat_replace_once(AI_CHAT_WEBGPU_COMMON_TEXT
        "struct q8_1 {\n    d: f16,\n    s: f16, // d * sum(qs[i])\n    qs: array<u32, 8>\n};"
        "struct q8_1 {\n#ifdef NO_SHADER_F16\n    ds: u32, // packed f16 scale and sum; decoded to f32\n#else\n    d: f16,\n    s: f16, // d * sum(qs[i])\n#endif\n    qs: array<u32, 8>\n};"
        "Q8_1 packed scale layout")
    ai_chat_replace_once(AI_CHAT_WEBGPU_COMMON_TEXT
        "    d: f16,\n    dmin: f16\n};"
        "#ifdef NO_SHADER_F16\n    dm: u32,\n#else\n    d: f16,\n    dmin: f16\n#endif\n};"
        "Q2_K packed scale layout")
    ai_chat_replace_once(AI_CHAT_WEBGPU_COMMON_TEXT
        "struct q4_K {\n    d: f16,\n    dmin: f16,"
        "struct q4_K {\n#ifdef NO_SHADER_F16\n    dm: u32,\n#else\n    d: f16,\n    dmin: f16,\n#endif"
        "Q4_K packed scale layout")
    ai_chat_replace_once(AI_CHAT_WEBGPU_COMMON_TEXT
        "struct q5_K {\n    d: f16,\n    dmin: f16,"
        "struct q5_K {\n#ifdef NO_SHADER_F16\n    dm: u32,\n#else\n    d: f16,\n    dmin: f16,\n#endif"
        "Q5_K packed scale layout")
    string(APPEND AI_CHAT_WEBGPU_COMMON_TEXT "\n// AI_CHAT_NO_SHADER_F16_COMMON_V1\n")
    file(WRITE "${AI_CHAT_WEBGPU_COMMON_DECLS}" "${AI_CHAT_WEBGPU_COMMON_TEXT}")
endif()

file(READ "${AI_CHAT_WEBGPU_MMV}" AI_CHAT_WEBGPU_MMV_TEXT)
string(FIND "${AI_CHAT_WEBGPU_MMV_TEXT}" "AI_CHAT_NO_SHADER_F16_MMV_V1" AI_CHAT_WEBGPU_MMV_PATCHED)
if (AI_CHAT_WEBGPU_MMV_PATCHED EQUAL -1)
    ai_chat_replace_once(AI_CHAT_WEBGPU_MMV_TEXT "enable f16;" "#define NO_SHADER_F16\n// FP16 scales are decoded to F32; no ShaderF16 feature is needed." "mat-vec f16 feature declaration")
    string(APPEND AI_CHAT_WEBGPU_MMV_TEXT "\n// AI_CHAT_NO_SHADER_F16_MMV_V1\n")
    file(WRITE "${AI_CHAT_WEBGPU_MMV}" "${AI_CHAT_WEBGPU_MMV_TEXT}")
endif()

file(READ "${AI_CHAT_WEBGPU_Q8}" AI_CHAT_WEBGPU_Q8_TEXT)
string(FIND "${AI_CHAT_WEBGPU_Q8_TEXT}" "AI_CHAT_NO_SHADER_F16_Q8_V1" AI_CHAT_WEBGPU_Q8_PATCHED)
if (AI_CHAT_WEBGPU_Q8_PATCHED EQUAL -1)
    ai_chat_replace_once(AI_CHAT_WEBGPU_Q8_TEXT "enable f16;" "#define NO_SHADER_F16\n// Quantized activations retain the same packed 16-bit scale layout." "Q8 quantizer f16 feature declaration")
    ai_chat_replace_once(AI_CHAT_WEBGPU_Q8_TEXT "src1q[src1q_idx].d = f16(d);" "src1q[src1q_idx].ds = pack2x16float(vec2<f32>(d, 0.0));" "Q8 quantizer scale write")
    ai_chat_replace_once(AI_CHAT_WEBGPU_Q8_TEXT "let s = f16(d * f32(cluster_add_i4x8(q4_quants_sum)));" "let s = d * f32(cluster_add_i4x8(q4_quants_sum));" "Q8 quantizer sum calculation")
    ai_chat_replace_once(AI_CHAT_WEBGPU_Q8_TEXT "src1q[src1q_idx].s = s;" "src1q[src1q_idx].ds = pack2x16float(vec2<f32>(d, s));" "Q8 quantizer packed scale/sum write")
    ai_chat_replace_once(AI_CHAT_WEBGPU_Q8_TEXT "src1q[src1q_idx].s = f16(s);" "src1q[src1q_idx].ds = pack2x16float(vec2<f32>(d, s));" "Q8 quantizer workgroup packed scale/sum write")
    string(APPEND AI_CHAT_WEBGPU_Q8_TEXT "\n// AI_CHAT_NO_SHADER_F16_Q8_V1\n")
    file(WRITE "${AI_CHAT_WEBGPU_Q8}" "${AI_CHAT_WEBGPU_Q8_TEXT}")
endif()

file(READ "${AI_CHAT_WEBGPU_MMV_Q_ACC}" AI_CHAT_WEBGPU_MMV_Q_ACC_TEXT)
string(FIND "${AI_CHAT_WEBGPU_MMV_Q_ACC_TEXT}" "AI_CHAT_NO_SHADER_F16_Q8_READ_V1" AI_CHAT_WEBGPU_MMV_Q_ACC_PATCHED)
if (AI_CHAT_WEBGPU_MMV_Q_ACC_PATCHED EQUAL -1)
    string(REPLACE "f32(src1q[block].d)" "unpack2x16float(src1q[block].ds)[0]" AI_CHAT_WEBGPU_MMV_Q_ACC_TEXT "${AI_CHAT_WEBGPU_MMV_Q_ACC_TEXT}")
    string(REPLACE "f32(src1q[block].s)" "unpack2x16float(src1q[block].ds)[1]" AI_CHAT_WEBGPU_MMV_Q_ACC_TEXT "${AI_CHAT_WEBGPU_MMV_Q_ACC_TEXT}")
    string(REPLACE "B_DS_TYPE(src1q[block].d)" "B_DS_TYPE(unpack2x16float(src1q[block].ds)[0])" AI_CHAT_WEBGPU_MMV_Q_ACC_TEXT "${AI_CHAT_WEBGPU_MMV_Q_ACC_TEXT}")
    string(REPLACE "B_DS_TYPE(src1q[q8_block_idx].d)" "B_DS_TYPE(unpack2x16float(src1q[q8_block_idx].ds)[0])" AI_CHAT_WEBGPU_MMV_Q_ACC_TEXT "${AI_CHAT_WEBGPU_MMV_Q_ACC_TEXT}")
    string(REPLACE "f32(src1q[q8_block_idx].d)" "unpack2x16float(src1q[q8_block_idx].ds)[0]" AI_CHAT_WEBGPU_MMV_Q_ACC_TEXT "${AI_CHAT_WEBGPU_MMV_Q_ACC_TEXT}")
    string(REPLACE "f32(src1q[q8_block_idx].s)" "unpack2x16float(src1q[q8_block_idx].ds)[1]" AI_CHAT_WEBGPU_MMV_Q_ACC_TEXT "${AI_CHAT_WEBGPU_MMV_Q_ACC_TEXT}")
    string(FIND "${AI_CHAT_WEBGPU_MMV_Q_ACC_TEXT}" "src1q[block].d)" AI_CHAT_WEBGPU_Q8_D_REMAINS)
    string(FIND "${AI_CHAT_WEBGPU_MMV_Q_ACC_TEXT}" "src1q[q8_block_idx].d)" AI_CHAT_WEBGPU_Q8_D_INDEX_REMAINS)
    if (NOT AI_CHAT_WEBGPU_Q8_D_REMAINS EQUAL -1 OR NOT AI_CHAT_WEBGPU_Q8_D_INDEX_REMAINS EQUAL -1)
        message(FATAL_ERROR "Could not convert all Q8_1 scale reads to packed F32 decoding")
    endif()
    string(APPEND AI_CHAT_WEBGPU_MMV_Q_ACC_TEXT "\n// AI_CHAT_NO_SHADER_F16_Q8_READ_V1\n")
    file(WRITE "${AI_CHAT_WEBGPU_MMV_Q_ACC}" "${AI_CHAT_WEBGPU_MMV_Q_ACC_TEXT}")
endif()

file(READ "${AI_CHAT_WEBGPU_MMV_ACC}" AI_CHAT_WEBGPU_MMV_ACC_TEXT)
string(FIND "${AI_CHAT_WEBGPU_MMV_ACC_TEXT}" "AI_CHAT_NO_SHADER_F16_MMV_ACC_V1" AI_CHAT_WEBGPU_MMV_ACC_PATCHED)
if (AI_CHAT_WEBGPU_MMV_ACC_PATCHED EQUAL -1)
    ai_chat_replace_once(AI_CHAT_WEBGPU_MMV_ACC_TEXT
        "f32(bitcast<vec2<f16>>(d_bits)[0])"
        "unpack2x16float(d_bits)[0]"
        "packed FP16 scalar decode in quantized mat-vec accumulator")
    string(APPEND AI_CHAT_WEBGPU_MMV_ACC_TEXT "\n// AI_CHAT_NO_SHADER_F16_MMV_ACC_V1\n")
    file(WRITE "${AI_CHAT_WEBGPU_MMV_ACC}" "${AI_CHAT_WEBGPU_MMV_ACC_TEXT}")
endif()

file(READ "${AI_CHAT_WEBGPU_MMV_Q_ACC}" AI_CHAT_WEBGPU_MMV_Q_ACC_TEXT)
file(READ "${AI_CHAT_WEBGPU_MMV_ACC}" AI_CHAT_WEBGPU_MMV_ACC_TEXT)

# Verify the patched runtime shader sources cannot accidentally request ShaderF16.
foreach(AI_CHAT_NO_F16_SHADER IN ITEMS "${AI_CHAT_WEBGPU_MMV}" "${AI_CHAT_WEBGPU_Q8}")
    file(READ "${AI_CHAT_NO_F16_SHADER}" AI_CHAT_NO_F16_SHADER_TEXT)
    string(FIND "${AI_CHAT_NO_F16_SHADER_TEXT}" "enable f16;" AI_CHAT_NO_F16_ENABLE_REMAINS)
    if (NOT AI_CHAT_NO_F16_ENABLE_REMAINS EQUAL -1)
        message(FATAL_ERROR "No-ShaderF16 quantized mat-vec shader still declares enable f16: ${AI_CHAT_NO_F16_SHADER}")
    endif()
endforeach()
string(FIND "${AI_CHAT_WEBGPU_MMV_Q_ACC_TEXT}" "src1q[block].d)" AI_CHAT_Q8_SCALE_D_REMAINS)
string(FIND "${AI_CHAT_WEBGPU_MMV_Q_ACC_TEXT}" "src1q[block].s)" AI_CHAT_Q8_SCALE_S_REMAINS)
string(FIND "${AI_CHAT_WEBGPU_MMV_Q_ACC_TEXT}" "src1q[q8_block_idx].d)" AI_CHAT_Q8_SCALE_D_INDEX_REMAINS)
string(FIND "${AI_CHAT_WEBGPU_MMV_Q_ACC_TEXT}" "src1q[q8_block_idx].s)" AI_CHAT_Q8_SCALE_S_INDEX_REMAINS)
string(FIND "${AI_CHAT_WEBGPU_MMV_ACC_TEXT}" "vec2<f16>" AI_CHAT_MMV_ACC_F16_REMAINS)
if (NOT AI_CHAT_Q8_SCALE_D_REMAINS EQUAL -1 OR NOT AI_CHAT_Q8_SCALE_S_REMAINS EQUAL -1 OR NOT AI_CHAT_Q8_SCALE_D_INDEX_REMAINS EQUAL -1 OR NOT AI_CHAT_Q8_SCALE_S_INDEX_REMAINS EQUAL -1 OR NOT AI_CHAT_MMV_ACC_F16_REMAINS EQUAL -1)
    message(FATAL_ERROR "No-ShaderF16 quantized mat-vec still contains an FP16 arithmetic/type dependency")
endif()

string(FIND "${AI_CHAT_WEBGPU_SOURCE_TEXT}" "AI_CHAT_NO_SHADER_F16_QUANT_MATMUL_OP_V4" AI_CHAT_WEBGPU_QUANT_MATMUL_OP_VERIFY)
if (AI_CHAT_WEBGPU_QUANT_MATVEC_OP_VERIFY EQUAL -1)
    message(FATAL_ERROR "No-ShaderF16 quantized MatMul was not integrated into the generic operation guard")
endif()
string(FIND "${AI_CHAT_WEBGPU_SOURCE_TEXT}" "AI_CHAT_NO_SHADER_F16_QUANTIZED_GEMM_OP_V1" AI_CHAT_WEBGPU_F32_GEMM_OP_VERIFY)
if (AI_CHAT_WEBGPU_F32_GEMM_OP_VERIFY EQUAL -1)
    message(FATAL_ERROR "No-ShaderF16 quantized/F32 GEMM admission is missing")
endif()

# Add a no-ShaderF16 variant of upstream's register-tile GEMM. Quantized weights
# remain packed; only temporary decode/shared-memory arithmetic becomes F32.
set(AI_CHAT_WEBGPU_SHADER_LIB_SOURCE "${llama_cpp_SOURCE_DIR}/ggml/src/ggml-webgpu/ggml-webgpu-shader-lib.hpp")
set(AI_CHAT_WEBGPU_CPP_SOURCE "${llama_cpp_SOURCE_DIR}/ggml/src/ggml-webgpu/ggml-webgpu.cpp")
set(AI_CHAT_WEBGPU_MATMUL_TILE "${llama_cpp_SOURCE_DIR}/ggml/src/ggml-webgpu/wgsl-shaders/mul_mat_reg_tile.wgsl")
set(AI_CHAT_WEBGPU_MATMUL_DECLS "${llama_cpp_SOURCE_DIR}/ggml/src/ggml-webgpu/wgsl-shaders/mul_mat_decls.tmpl")
foreach(AI_CHAT_MATMUL_SOURCE IN ITEMS "${AI_CHAT_WEBGPU_SHADER_LIB_SOURCE}" "${AI_CHAT_WEBGPU_CPP_SOURCE}" "${AI_CHAT_WEBGPU_MATMUL_TILE}" "${AI_CHAT_WEBGPU_MATMUL_DECLS}")
    if (NOT EXISTS "${AI_CHAT_MATMUL_SOURCE}")
        message(FATAL_ERROR "Pinned GGML no-ShaderF16 matmul source is missing: ${AI_CHAT_MATMUL_SOURCE}")
    endif()
endforeach()

file(READ "${AI_CHAT_WEBGPU_SHADER_LIB_SOURCE}" AI_CHAT_WEBGPU_SHADER_LIB_TEXT)
string(FIND "${AI_CHAT_WEBGPU_SHADER_LIB_TEXT}" "AI_CHAT_NO_SHADER_F16_QUANTIZED_GEMM_PIPELINE_V1" AI_CHAT_WEBGPU_QUANTIZED_GEMM_PIPELINE_PATCHED)
if (AI_CHAT_WEBGPU_QUANTIZED_GEMM_PIPELINE_PATCHED EQUAL -1)
    ai_chat_replace_once(AI_CHAT_WEBGPU_SHADER_LIB_TEXT
        "bool        supports_subgroups       = false;"
        "bool        supports_shader_f16      = true;\n    bool        supports_subgroups       = false;"
        "shader library ShaderF16 capability field")
    ai_chat_replace_once(AI_CHAT_WEBGPU_SHADER_LIB_TEXT
        "key.use_subgroup_matrix = context.supports_subgroup_matrix;"
        "key.use_subgroup_matrix = context.supports_subgroup_matrix && context.supports_shader_f16;"
        "disable F16 subgroup matrix path without ShaderF16")
    ai_chat_replace_once(AI_CHAT_WEBGPU_SHADER_LIB_TEXT
        "std::vector<std::string> defines;\n        std::string              variant = key.use_subgroup_matrix ? \"mul_mat_subgroup_matrix\" : \"mul_mat_reg_tile\";"
        "std::vector<std::string> defines;\n        if (!context.supports_shader_f16) defines.push_back(\"NO_SHADER_F16\");\n        std::string              variant = key.use_subgroup_matrix ? \"mul_mat_subgroup_matrix\" : \"mul_mat_reg_tile\";"
        "select portable F32 arithmetic in upstream register-tile GEMM")
    string(APPEND AI_CHAT_WEBGPU_SHADER_LIB_TEXT "\n// AI_CHAT_NO_SHADER_F16_QUANTIZED_GEMM_PIPELINE_V1\n")
    file(WRITE "${AI_CHAT_WEBGPU_SHADER_LIB_SOURCE}" "${AI_CHAT_WEBGPU_SHADER_LIB_TEXT}")
endif()

file(READ "${AI_CHAT_WEBGPU_CPP_SOURCE}" AI_CHAT_WEBGPU_CPP_TEXT)
string(FIND "${AI_CHAT_WEBGPU_CPP_TEXT}" "AI_CHAT_NO_SHADER_F16_QUANTIZED_GEMM_CONTEXT_V1" AI_CHAT_WEBGPU_QUANTIZED_GEMM_CONTEXT_PATCHED)
if (AI_CHAT_WEBGPU_QUANTIZED_GEMM_CONTEXT_PATCHED EQUAL -1)
    ai_chat_replace_once(AI_CHAT_WEBGPU_CPP_TEXT
        "shader_lib_ctx.src0                     = src0;"
        "shader_lib_ctx.src0                     = src0;\n    shader_lib_ctx.supports_shader_f16 = ctx->global_ctx->capabilities.supports_shader_f16;"
        "propagate ShaderF16 capability to upstream matmul shader library")
    string(APPEND AI_CHAT_WEBGPU_CPP_TEXT "\n// AI_CHAT_NO_SHADER_F16_QUANTIZED_GEMM_CONTEXT_V1\n")
    file(WRITE "${AI_CHAT_WEBGPU_CPP_SOURCE}" "${AI_CHAT_WEBGPU_CPP_TEXT}")
endif()

file(READ "${AI_CHAT_WEBGPU_MATMUL_TILE}" AI_CHAT_WEBGPU_MATMUL_TILE_TEXT)
string(FIND "${AI_CHAT_WEBGPU_MATMUL_TILE_TEXT}" "AI_CHAT_NO_SHADER_F16_QUANTIZED_GEMM_TILE_V1" AI_CHAT_WEBGPU_QUANTIZED_GEMM_TILE_PATCHED)
if (AI_CHAT_WEBGPU_QUANTIZED_GEMM_TILE_PATCHED EQUAL -1)
    ai_chat_replace_once(AI_CHAT_WEBGPU_MATMUL_TILE_TEXT
        "enable f16;"
        "#ifndef NO_SHADER_F16\nenable f16;\n#endif"
        "make register-tile ShaderF16 declaration conditional")
    ai_chat_replace_once(AI_CHAT_WEBGPU_MATMUL_TILE_TEXT
        "#include \"common_decls.tmpl\"\n#include \"mul_mat_decls.tmpl\""
        "#ifdef NO_SHADER_F16\n#define f16 f32\n#endif\n#include \"common_decls.tmpl\"\n#include \"mul_mat_decls.tmpl\""
        "apply F32 alias before all shared quantized declarations")
    string(APPEND AI_CHAT_WEBGPU_MATMUL_TILE_TEXT "\n// AI_CHAT_NO_SHADER_F16_QUANTIZED_GEMM_TILE_V1\n")
    file(WRITE "${AI_CHAT_WEBGPU_MATMUL_TILE}" "${AI_CHAT_WEBGPU_MATMUL_TILE_TEXT}")
endif()

file(READ "${AI_CHAT_WEBGPU_MATMUL_DECLS}" AI_CHAT_WEBGPU_MATMUL_DECLS_TEXT)
string(FIND "${AI_CHAT_WEBGPU_MATMUL_DECLS_TEXT}" "AI_CHAT_NO_SHADER_F16_QUANTIZED_GEMM_DECLS_V1" AI_CHAT_WEBGPU_QUANTIZED_GEMM_DECLS_PATCHED)
if (AI_CHAT_WEBGPU_QUANTIZED_GEMM_DECLS_PATCHED EQUAL -1)
    ai_chat_replace_once(AI_CHAT_WEBGPU_MATMUL_DECLS_TEXT
        "let d          = bitcast<vec2<f16>>(d_scales_h).x;"
        "#ifdef NO_SHADER_F16\n        let d = unpack2x16float(d_scales_h).x;\n#else\n        let d = bitcast<vec2<f16>>(d_scales_h).x;\n#endif"
        "decode IQ scale as F32 without ShaderF16")
    ai_chat_replace_once(AI_CHAT_WEBGPU_MATMUL_DECLS_TEXT
        "let d = f32(bitcast<vec2<f16>>(scale_packed).x);"
        "#ifdef NO_SHADER_F16\n        let d = unpack2x16float(scale_packed).x;\n#else\n        let d = f32(bitcast<vec2<f16>>(scale_packed).x);\n#endif"
        "decode packed IQ scale as F32 without ShaderF16")
    string(APPEND AI_CHAT_WEBGPU_MATMUL_DECLS_TEXT "\n// AI_CHAT_NO_SHADER_F16_QUANTIZED_GEMM_DECLS_V1\n")
    file(WRITE "${AI_CHAT_WEBGPU_MATMUL_DECLS}" "${AI_CHAT_WEBGPU_MATMUL_DECLS_TEXT}")
endif()


string(FIND "${AI_CHAT_WEBGPU_SHADER_LIB_TEXT}" "AI_CHAT_NO_SHADER_F16_QUANTIZED_GEMM_PIPELINE_V1" AI_CHAT_WEBGPU_QUANTIZED_GEMM_PIPELINE_VERIFY)
string(FIND "${AI_CHAT_WEBGPU_MATMUL_TILE_TEXT}" "AI_CHAT_NO_SHADER_F16_QUANTIZED_GEMM_TILE_V1" AI_CHAT_WEBGPU_QUANTIZED_GEMM_TILE_VERIFY)
string(FIND "${AI_CHAT_WEBGPU_MATMUL_TILE_TEXT}" "#define f16 f32" AI_CHAT_WEBGPU_F16_ALIAS_VERIFY)
string(FIND "${AI_CHAT_WEBGPU_MATMUL_TILE_TEXT}" "#include \"common_decls.tmpl\"" AI_CHAT_WEBGPU_COMMON_DECLS_INCLUDE)
if (AI_CHAT_WEBGPU_F16_ALIAS_VERIFY EQUAL -1 OR AI_CHAT_WEBGPU_COMMON_DECLS_INCLUDE EQUAL -1 OR AI_CHAT_WEBGPU_F16_ALIAS_VERIFY GREATER AI_CHAT_WEBGPU_COMMON_DECLS_INCLUDE)
    message(FATAL_ERROR "No-ShaderF16 F32 alias must precede common declarations and all quantized WGSL helpers")
endif()
string(FIND "${AI_CHAT_WEBGPU_MATMUL_DECLS_TEXT}" "AI_CHAT_NO_SHADER_F16_QUANTIZED_GEMM_DECLS_V1" AI_CHAT_WEBGPU_QUANTIZED_GEMM_DECLS_VERIFY)
if (AI_CHAT_WEBGPU_QUANTIZED_GEMM_PIPELINE_VERIFY EQUAL -1 OR AI_CHAT_WEBGPU_QUANTIZED_GEMM_TILE_VERIFY EQUAL -1 OR AI_CHAT_WEBGPU_F16_ALIAS_VERIFY EQUAL -1 OR AI_CHAT_WEBGPU_QUANTIZED_GEMM_DECLS_VERIFY EQUAL -1)
    message(FATAL_ERROR "Portable no-ShaderF16 quantized GEMM source integration is incomplete")
endif()

file(WRITE "${AI_CHAT_WEBGPU_SOURCE}" "${AI_CHAT_WEBGPU_SOURCE_TEXT}")

# Do not silently ship a build where OpenMP was requested but FindOpenMP could
# not enable it. This also catches NDK/runtime-linking regressions during CI.
if (NOT GGML_OPENMP_ENABLED)
    message(FATAL_ERROR "GGML_OPENMP=ON was requested, but the pinned GGML build did not enable OpenMP")
endif()
if (NOT GGML_CPU_KLEIDIAI)
    message(FATAL_ERROR "GGML_CPU_KLEIDIAI=ON was requested, but KleidiAI is disabled")
endif()

# Add one-shot per-node CPU/OpenCL backend assignment logs for performance diagnostics.
include("${CMAKE_CURRENT_LIST_DIR}/cmake/patch_backend_assignment_trace.cmake")

# Android OpenCL vendor libraries are often complete OpenCL implementations but
# not ICDs. Patch only the pinned ggml OpenCL call sites to use our direct API
# dispatcher; the Khronos loader remains linked for ABI compatibility but is not
# used for driver dispatch. This avoids fabricating the ICD extension entrypoint.
set(AI_CHAT_OPENCL_SOURCE_FILES
    "${llama_cpp_SOURCE_DIR}/ggml/src/ggml-opencl/ggml-opencl.cpp"
    "${llama_cpp_SOURCE_DIR}/ggml/src/ggml-opencl/cl-program-cache.cpp"
)
foreach(AI_CHAT_OPENCL_SOURCE IN LISTS AI_CHAT_OPENCL_SOURCE_FILES)
    file(READ "${AI_CHAT_OPENCL_SOURCE}" AI_CHAT_OPENCL_SOURCE_TEXT)
    string(FIND "${AI_CHAT_OPENCL_SOURCE_TEXT}" "android_opencl_dispatch.h" AI_CHAT_OPENCL_ALREADY_PATCHED)
    if (AI_CHAT_OPENCL_ALREADY_PATCHED EQUAL -1)
        if (AI_CHAT_OPENCL_SOURCE MATCHES "ggml-opencl\\.cpp$")
            string(FIND "${AI_CHAT_OPENCL_SOURCE_TEXT}" "#include <CL/cl.h>" AI_CHAT_OPENCL_INCLUDE_POS)
            if (AI_CHAT_OPENCL_INCLUDE_POS EQUAL -1)
                message(FATAL_ERROR "Could not patch direct Android OpenCL dispatch in ${AI_CHAT_OPENCL_SOURCE}")
            endif()
            string(REPLACE "#include <CL/cl.h>"
                "#include <CL/cl.h>\n#include \"android_opencl_dispatch.h\""
                AI_CHAT_OPENCL_SOURCE_TEXT "${AI_CHAT_OPENCL_SOURCE_TEXT}")
        else()
            string(FIND "${AI_CHAT_OPENCL_SOURCE_TEXT}" "#include \"cl-program-cache.h\"" AI_CHAT_OPENCL_INCLUDE_POS)
            if (AI_CHAT_OPENCL_INCLUDE_POS EQUAL -1)
                message(FATAL_ERROR "Could not patch direct Android OpenCL dispatch in ${AI_CHAT_OPENCL_SOURCE}")
            endif()
            string(REPLACE "#include \"cl-program-cache.h\""
                "#include \"cl-program-cache.h\"\n#include \"android_opencl_dispatch.h\""
                AI_CHAT_OPENCL_SOURCE_TEXT "${AI_CHAT_OPENCL_SOURCE_TEXT}")
        endif()
        file(WRITE "${AI_CHAT_OPENCL_SOURCE}" "${AI_CHAT_OPENCL_SOURCE_TEXT}")
    endif()
endforeach()

# Do not spoof an unsupported GPU family by matching device-name substrings. The pinned
# upstream OpenCL backend contains family-specific kernels and subgroup assumptions.
# Device-agnostic support must be added through runtime capability checks and genuinely
# generic kernels, not by relabeling Mali/unknown devices as Adreno or Intel.
target_include_directories(ggml-opencl PRIVATE "${CMAKE_CURRENT_LIST_DIR}")

if (NOT TARGET ggml-opencl)
    message(FATAL_ERROR "OpenCL-preferred runtime requires the statically linked ggml-opencl target")
endif()
target_include_directories(ggml-opencl PRIVATE "${CMAKE_CURRENT_LIST_DIR}")

# Static backend hardening: make the OpenCL registration visible to the final
# runtime target as a real link dependency. This prevents linker GC from
# discarding the backend registration object and makes a missing OpenCL target
# a build-time error instead of a silent CPU-only runtime.
if (NOT TARGET ggml-opencl)
    message(FATAL_ERROR "OpenCL-preferred runtime requires the statically linked ggml-opencl target")
endif()
if (TARGET ggml)
    target_compile_definitions(ggml PUBLIC GGML_USE_OPENCL=1)
endif()

if (AI_CHAT_OPENCL_PROFILING)
    include("${CMAKE_CURRENT_LIST_DIR}/cmake/patch_opencl_profiling.cmake")
endif()

add_library(ai_chat_runtime SHARED native_runtime_android_safe.cpp vision_runtime_android.cpp universal_gpu_adapter.cpp)

# native_runtime.cpp intentionally includes llama.cpp/ggml internal headers for
# strict backend and tensor-residency validation. Keep both internal source
# directories on the include path; the public ggml include directory alone is
# insufficient for ggml-backend-impl.h and ggml-opencl.h on this pinned revision.
target_include_directories(ai_chat_runtime PRIVATE
    ${llama_cpp_SOURCE_DIR}/include
    ${llama_cpp_SOURCE_DIR}/src
    ${llama_cpp_SOURCE_DIR}/ggml/include
    ${llama_cpp_SOURCE_DIR}/ggml/src
    ${llama_cpp_SOURCE_DIR}/ggml/src/ggml-opencl
    ${llama_cpp_SOURCE_DIR}/ggml/src/ggml-webgpu
    ${llama_cpp_SOURCE_DIR}/tools/mtmd
    ${opencl_headers_SOURCE_DIR}
)

# The runtime is a single arm64 production native library. Strip unused sections
# and symbols at link time to reduce the installed footprint without changing inference.
target_compile_options(ai_chat_runtime PRIVATE
    -Os
    -ffunction-sections
    -fdata-sections
)
target_link_options(ai_chat_runtime PRIVATE
    -Wl,--gc-sections
    -Wl,--strip-all
)

if (NOT TARGET ggml-webgpu)
    message(FATAL_ERROR "GGML_WEBGPU=ON was requested, but ggml-webgpu target was not created")
endif()
target_link_libraries(ai_chat_runtime PRIVATE llama llama-common mtmd ggml-opencl ggml-webgpu android log OpenCL EGL GLESv3)

target_compile_definitions(ai_chat_runtime PRIVATE
    AI_CHAT_LLAMA_CPP_SHA="${LLAMA_CPP_TAG}"
    AI_CHAT_GPU_BACKEND_OPENCL=1
    AI_CHAT_GPU_BACKEND_WEBGPU=1
    AI_CHAT_GPU_BACKEND_VULKAN=0
    AI_CHAT_GPU_ADAPTER_OPENGL_ES=1
    AI_CHAT_GGML_OPENMP_ENABLED=1
    AI_CHAT_GGML_CPU_KLEIDIAI_ENABLED=1
    AI_CHAT_CPU_ARM_BASELINE="armv8-a"
)

if (ANDROID)
    set_target_properties(ai_chat_runtime PROPERTIES
        CXX_VISIBILITY_PRESET hidden
        VISIBILITY_INLINES_HIDDEN ON
    )
endif()
