cmake_minimum_required(VERSION 3.22)
project(ai_chat_runtime LANGUAGES C CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(FetchContent)
set(LLAMA_CPP_TAG "9871df5911a03a813518bcd623ee2eba91bf32aa" CACHE STRING "Pinned llama.cpp revision with Qwen3.x and Vulkan backend support")
set(FETCHCONTENT_QUIET OFF)
set(FETCHCONTENT_BASE_DIR "${CMAKE_CURRENT_LIST_DIR}/../../.cxx/fetch-content" CACHE PATH "Stable FetchContent base directory")
set(FETCHCONTENT_UPDATES_DISCONNECTED ON CACHE BOOL "Do not update pinned FetchContent dependencies")

# Vulkan is supplied by the Android system loader at runtime. CI/local native builds
# need glslc and SPIR-V headers from the LunarG SDK to compile llama.cpp's shaders.
if (ANDROID)
    if (DEFINED ENV{VULKAN_SDK})
        set(Vulkan_INCLUDE_DIR "$ENV{VULKAN_SDK}/include" CACHE PATH "Vulkan headers" FORCE)
        set(Vulkan_GLSLC_EXECUTABLE "$ENV{VULKAN_SDK}/bin/glslc" CACHE FILEPATH "Vulkan shader compiler" FORCE)
        set(SPIRV-Headers_DIR "$ENV{VULKAN_SDK}/share/cmake/SPIRV-Headers" CACHE PATH "SPIRV-Headers CMake package" FORCE)
        set(CMAKE_PREFIX_PATH "$ENV{VULKAN_SDK};${CMAKE_PREFIX_PATH}" CACHE STRING "Vulkan package search path" FORCE)
    endif()
    set(Vulkan_LIBRARY "${CMAKE_ANDROID_NDK}/toolchains/llvm/prebuilt/linux-x86_64/sysroot/usr/lib/aarch64-linux-android/29/libvulkan.so" CACHE FILEPATH "Android Vulkan loader" FORCE)
endif()

if (NOT DEFINED Vulkan_GLSLC_EXECUTABLE AND NOT DEFINED ENV{VULKAN_SDK})
    message(FATAL_ERROR "Vulkan SDK is required to compile Vulkan shaders. Set VULKAN_SDK or run ci/prepare-vulkan.sh.")
endif()

FetchContent_Declare(
    llama_cpp
    GIT_REPOSITORY https://github.com/ggml-org/llama.cpp.git
    GIT_TAG ${LLAMA_CPP_TAG}
    GIT_SHALLOW TRUE
)

# GPU-first Android runtime: no OpenCL or full-model CPU fallback. The scheduler may run an individual op on CPU only when no earlier GPU backend supports it.
set(GGML_VULKAN ON CACHE BOOL "Use Vulkan GPU backend" FORCE)
set(GGML_OPENCL OFF CACHE BOOL "OpenCL backend removed on the Vulkan branch" FORCE)
set(GGML_BACKEND_DL OFF CACHE BOOL "Statically link the Vulkan backend on Android" FORCE)
set(GGML_BACKEND_DIR "" CACHE STRING "No dynamic backend directory" FORCE)
set(GGML_NATIVE OFF CACHE BOOL "Portable Android native build" FORCE)
set(GGML_OPENMP OFF CACHE BOOL "Avoid OpenMP runtime dependency on Android" FORCE)
set(GGML_LLAMAFILE OFF CACHE BOOL "Disable llamafile on Android" FORCE)
set(GGML_VULKAN_RUN_TESTS OFF CACHE BOOL "Do not build Vulkan backend tests" FORCE)
set(GGML_VULKAN_VALIDATE OFF CACHE BOOL "Do not enable validation layers in production" FORCE)
set(BUILD_SHARED_LIBS OFF CACHE BOOL "Static llama.cpp dependencies" FORCE)
set(LLAMA_BUILD_COMMON ON CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_APP OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_UI OFF CACHE BOOL "" FORCE)
set(LLAMA_OPENSSL OFF CACHE BOOL "" FORCE)

FetchContent_MakeAvailable(llama_cpp)

# Android Vulkan stability mitigation: the pinned runtime auto-enables fused
# DeepSeek V4 hyper-connection ops before backend support is known. Disable only
# these optional fused variants on Android so unsupported fused graph nodes do not
# reach the GPU-only scheduler with no eligible backend. This does not enable CPU
# compute fallback; ordinary unfused graph ops still must be supported by Vulkan.
set(_ai_chat_dsv4_context_cpp "${llama_cpp_SOURCE_DIR}/src/llama-context.cpp")
file(READ "${_ai_chat_dsv4_context_cpp}" _ai_chat_dsv4_context_source)
if(NOT _ai_chat_dsv4_context_source MATCHES "AI_CHAT_ANDROID_DISABLE_DSV4_HC_VULKAN_V1")
    foreach(_ai_chat_dsv4_setting
            "cparams.fused_dsv4_hc_pre  = true;"
            "cparams.fused_dsv4_hc_comb = true;"
            "cparams.fused_dsv4_hc_post = true;"
            "cparams.auto_fhc           = true;")
        string(FIND "${_ai_chat_dsv4_context_source}" "${_ai_chat_dsv4_setting}" _ai_chat_dsv4_location)
        if(_ai_chat_dsv4_location EQUAL -1)
            message(FATAL_ERROR "Could not locate pinned llama.cpp DSV4 HC setting: ${_ai_chat_dsv4_setting}")
        endif()
        string(REPLACE "${_ai_chat_dsv4_setting}"
            "${_ai_chat_dsv4_setting}\n    // AI_CHAT_ANDROID_DISABLE_DSV4_HC_VULKAN_V1: unsupported fused ops disabled for Android Vulkan."
            _ai_chat_dsv4_context_source "${_ai_chat_dsv4_context_source}")
    endforeach()
    # Replace the true setting itself, not merely annotate it.
    string(REPLACE "cparams.fused_dsv4_hc_pre  = true;" "cparams.fused_dsv4_hc_pre  = false;"
        _ai_chat_dsv4_context_source "${_ai_chat_dsv4_context_source}")
    string(REPLACE "cparams.fused_dsv4_hc_comb = true;" "cparams.fused_dsv4_hc_comb = false;"
        _ai_chat_dsv4_context_source "${_ai_chat_dsv4_context_source}")
    string(REPLACE "cparams.fused_dsv4_hc_post = true;" "cparams.fused_dsv4_hc_post = false;"
        _ai_chat_dsv4_context_source "${_ai_chat_dsv4_context_source}")
    string(REPLACE "cparams.auto_fhc           = true;" "cparams.auto_fhc           = false;"
        _ai_chat_dsv4_context_source "${_ai_chat_dsv4_context_source}")
    file(WRITE "${_ai_chat_dsv4_context_cpp}" "${_ai_chat_dsv4_context_source}")
endif()
file(READ "${_ai_chat_dsv4_context_cpp}" _ai_chat_dsv4_context_verify)
if(NOT _ai_chat_dsv4_context_verify MATCHES "AI_CHAT_ANDROID_DISABLE_DSV4_HC_VULKAN_V1" OR
   _ai_chat_dsv4_context_verify MATCHES "cparams.auto_fhc[ ]*= true;")
    message(FATAL_ERROR "Android Vulkan DSV4 HC stability mitigation was not applied")
endif()

# Android Vulkan: use the portable unfused recurrent graph until the fused
# GDN and Lightning-Indexer kernels have passed device validation on Mali.
# This changes kernel selection only, not model weights or sampling semantics.
if (ANDROID)
    string(REPLACE "cparams.fused_gdn_ar = true;" "cparams.fused_gdn_ar = false;"
        _ai_chat_dsv4_context_verify "${_ai_chat_dsv4_context_verify}")
    string(REPLACE "cparams.fused_gdn_ch = true;" "cparams.fused_gdn_ch = false;"
        _ai_chat_dsv4_context_verify "${_ai_chat_dsv4_context_verify}")
    string(REPLACE "cparams.fused_lid = true;" "cparams.fused_lid = false;"
        _ai_chat_dsv4_context_verify "${_ai_chat_dsv4_context_verify}")
    file(WRITE "${_ai_chat_dsv4_context_cpp}" "${_ai_chat_dsv4_context_verify}")
    file(READ "${_ai_chat_dsv4_context_cpp}" _ai_chat_fused_verify)
    if(_ai_chat_fused_verify MATCHES "cparams.fused_gdn_ar[ ]*= true;" OR
       _ai_chat_fused_verify MATCHES "cparams.fused_gdn_ch[ ]*= true;" OR
       _ai_chat_fused_verify MATCHES "cparams.fused_lid[ ]*= true;" OR
       NOT _ai_chat_fused_verify MATCHES "cparams.fused_gdn_ar[ ]*= false;" OR
       NOT _ai_chat_fused_verify MATCHES "cparams.fused_gdn_ch[ ]*= false;" OR
       NOT _ai_chat_fused_verify MATCHES "cparams.fused_lid[ ]*= false;")
        message(FATAL_ERROR "Android Vulkan fused recurrent-op mitigation was not applied")
    endif()
endif()

# Keep CPU available for host/control tensors and as a last-resort backend for
# individual operators that no GPU backend supports. The normal ggml scheduler
# orders Vulkan before CPU, so supported inference ops remain on Vulkan; CPU is
# not selected when an earlier GPU backend supports the op. This avoids aborting
# graph reservation for unsupported operators without forcing whole-model CPU
# inference. Do not patch ggml-backend.cpp to reject CPU compute assignments.
set(_ai_chat_backend_cpp "${llama_cpp_SOURCE_DIR}/ggml/src/ggml-backend.cpp")
file(READ "${_ai_chat_backend_cpp}" _ai_chat_backend_source)
if(NOT _ai_chat_backend_source MATCHES "AI_CHAT_SELECTIVE_CPU_OP_FALLBACK_V1")
    set(_ai_chat_backend_policy_needle [=[static void ggml_backend_sched_set_if_supported(ggml_backend_sched_t sched, struct ggml_tensor * node, int cur_backend_id, int * node_backend_id) {]=])
    set(_ai_chat_backend_policy_replacement [=[// AI_CHAT_SELECTIVE_CPU_OP_FALLBACK_V1: preserve upstream GPU-first scheduling; CPU may execute an op only if no earlier GPU backend supports it.
static void ggml_backend_sched_set_if_supported(ggml_backend_sched_t sched, struct ggml_tensor * node, int cur_backend_id, int * node_backend_id) {]=])
    string(FIND "${_ai_chat_backend_source}" "${_ai_chat_backend_policy_needle}" _ai_chat_backend_policy_location)
    if(_ai_chat_backend_policy_location EQUAL -1)
        message(FATAL_ERROR "Could not locate pinned ggml scheduler assignment helper")
    endif()
    string(REPLACE "${_ai_chat_backend_policy_needle}" "${_ai_chat_backend_policy_replacement}" _ai_chat_backend_source "${_ai_chat_backend_source}")
    file(WRITE "${_ai_chat_backend_cpp}" "${_ai_chat_backend_source}")
endif()
file(READ "${_ai_chat_backend_cpp}" _ai_chat_backend_verify)
string(CONCAT _ai_chat_old_guard_token "AI_CHAT_GPU_ONLY_NO_CPU_COMPUTE_" "FALLBACK")
if(NOT _ai_chat_backend_verify MATCHES "AI_CHAT_SELECTIVE_CPU_OP_FALLBACK_V1" OR
   _ai_chat_backend_verify MATCHES "${_ai_chat_old_guard_token}")
    message(FATAL_ERROR "Selective GPU-first CPU operator fallback policy is missing or an old CPU-compute guard remains")
endif()

# Avoid process-fatal GGML_ASSERT when the scheduler cannot assign a graph node.
# View nodes must inherit the placement of their underlying source. Truly unsupported
# compute ops fail context creation with a detailed error, which llama_init_from_model
# catches and returns as nullptr instead of terminating the Android process.
if(NOT _ai_chat_backend_source MATCHES "AI_CHAT_UNASSIGNED_NODE_DIAGNOSTICS_V1")
    set(_ai_chat_unassigned_needle [=[        GGML_ASSERT(*cur_backend_id != -1);]=])
    set(_ai_chat_unassigned_replacement [=[        // AI_CHAT_UNASSIGNED_NODE_DIAGNOSTICS_V1: resolve view chains, then fail context init safely.
        if (*cur_backend_id == -1 && ggml_is_view_op(node->op)) {
            struct ggml_tensor * ai_chat_view_src = node->view_src;
            for (int ai_chat_depth = 0; ai_chat_view_src != nullptr && ai_chat_depth < 32 && *cur_backend_id == -1; ++ai_chat_depth) {
                const int ai_chat_source_backend = tensor_backend_id(ai_chat_view_src);
                if (ai_chat_source_backend != -1) {
                    *cur_backend_id = ai_chat_source_backend;
                    SET_CAUSE(node, "4.view-chain");
                    break;
                }
                ai_chat_view_src = ai_chat_view_src->view_src;
            }
        }
        if (*cur_backend_id == -1) {
            std::string ai_chat_support = "AI_CHAT_UNSUPPORTED_GRAPH_OP node=";
            ai_chat_support += node->name;
            ai_chat_support += " op=";
            ai_chat_support += ggml_op_name(node->op);
            ai_chat_support += " type=";
            ai_chat_support += ggml_type_name(node->type);
            ai_chat_support += " ne=" + std::to_string(node->ne[0]) + "," +
                std::to_string(node->ne[1]) + "," + std::to_string(node->ne[2]) + "," +
                std::to_string(node->ne[3]) + " backend_support=";
            for (int ai_chat_b = 0; ai_chat_b < sched->n_backends; ++ai_chat_b) {
                if (ai_chat_b > 0) ai_chat_support += ",";
                ai_chat_support += std::to_string(ai_chat_b) + ":" +
                    (ggml_backend_supports_op(sched->backends[ai_chat_b], node) ? "1" : "0");
            }
            GGML_LOG_ERROR("%s\n", ai_chat_support.c_str());
            throw std::runtime_error(ai_chat_support);
        }]=])
    string(FIND "${_ai_chat_backend_source}" "${_ai_chat_unassigned_needle}" _ai_chat_unassigned_location)
    if(_ai_chat_unassigned_location EQUAL -1)
        message(FATAL_ERROR "Could not locate pinned ggml unassigned-node assertion; refusing an unsafe scheduler patch")
    endif()
    string(REPLACE "${_ai_chat_unassigned_needle}" "${_ai_chat_unassigned_replacement}" _ai_chat_backend_source "${_ai_chat_backend_source}")
    file(WRITE "${_ai_chat_backend_cpp}" "${_ai_chat_backend_source}")
endif()
file(READ "${_ai_chat_backend_cpp}" _ai_chat_unassigned_verify)
string(FIND "${_ai_chat_unassigned_verify}" "GGML_ASSERT(*cur_backend_id != -1);" _ai_chat_stale_assert_location)
if(NOT _ai_chat_unassigned_verify MATCHES "AI_CHAT_UNASSIGNED_NODE_DIAGNOSTICS_V1" OR
   NOT _ai_chat_stale_assert_location EQUAL -1)
    message(FATAL_ERROR "Safe scheduler handling for unassigned graph nodes was not applied")
endif()

# Compact scheduler placement telemetry. One summary line per graph avoids per-node disk I/O.
if(NOT _ai_chat_backend_source MATCHES "AI_CHAT_BACKEND_ASSIGNMENT_PROFILE_V1")
    set(_ai_chat_assignment_needle [=[    if (sched->debug) {
        ggml_backend_sched_print_assignments(sched, graph);
    }]=])
    set(_ai_chat_assignment_replacement [=[    // AI_CHAT_BACKEND_ASSIGNMENT_PROFILE_V1: scheduler-selected device placement.
    {
        int ai_chat_gpu_nodes = 0;
        int ai_chat_cpu_nodes = 0;
        int ai_chat_other_nodes = 0;
        int ai_chat_unassigned_nodes = 0;
        std::unordered_map<std::string, int> ai_chat_cpu_ops;
        for (int ai_chat_i = 0; ai_chat_i < graph->n_nodes; ++ai_chat_i) {
            struct ggml_tensor * ai_chat_node = graph->nodes[ai_chat_i];
            if (ggml_is_view_op(ai_chat_node->op)) continue;
            ggml_backend_t ai_chat_backend = ggml_backend_sched_get_tensor_backend(sched, ai_chat_node);
            if (!ai_chat_backend) {
                ++ai_chat_unassigned_nodes;
                continue;
            }
            ggml_backend_dev_t ai_chat_dev = ggml_backend_get_device(ai_chat_backend);
            const enum ggml_backend_dev_type ai_chat_type = ai_chat_dev
                ? ggml_backend_dev_type(ai_chat_dev) : GGML_BACKEND_DEVICE_TYPE_CPU;
            if (ai_chat_type == GGML_BACKEND_DEVICE_TYPE_GPU || ai_chat_type == GGML_BACKEND_DEVICE_TYPE_IGPU) {
                ++ai_chat_gpu_nodes;
            } else if (ai_chat_type == GGML_BACKEND_DEVICE_TYPE_CPU) {
                ++ai_chat_cpu_nodes;
                ++ai_chat_cpu_ops[ggml_op_name(ai_chat_node->op)];
            } else {
                ++ai_chat_other_nodes;
            }
        }
        std::string ai_chat_cpu_op_summary;
        for (const auto & ai_chat_entry : ai_chat_cpu_ops) {
            if (!ai_chat_cpu_op_summary.empty()) ai_chat_cpu_op_summary += ",";
            ai_chat_cpu_op_summary += ai_chat_entry.first + ":" + std::to_string(ai_chat_entry.second);
        }
        const std::string ai_chat_assignment_line =
            std::string("AI_CHAT_BACKEND_GRAPH_ASSIGNMENT nodes=") + std::to_string(graph->n_nodes) +
            " splits=" + std::to_string(sched->n_splits) +
            " gpuNodes=" + std::to_string(ai_chat_gpu_nodes) +
            " cpuNodes=" + std::to_string(ai_chat_cpu_nodes) +
            " otherNodes=" + std::to_string(ai_chat_other_nodes) +
            " unassignedNodes=" + std::to_string(ai_chat_unassigned_nodes) +
            " cpuOpTypes=" + (ai_chat_cpu_op_summary.empty() ? "none" : ai_chat_cpu_op_summary);
        GGML_LOG_DEBUG("%s\n", ai_chat_assignment_line.c_str());
    }
    if (sched->debug) {
        ggml_backend_sched_print_assignments(sched, graph);
    }]=])
    string(FIND "${_ai_chat_backend_source}" "${_ai_chat_assignment_needle}" _ai_chat_assignment_location)
    if(_ai_chat_assignment_location EQUAL -1)
        message(FATAL_ERROR "Could not locate pinned ggml scheduler graph assignment logging point")
    endif()
    if(NOT _ai_chat_backend_source MATCHES "ai_chat_append_native_trace")
        set(_ai_chat_trace_include [=[#include <unordered_map>
#include <string>
#include <stdexcept>
#if defined(__GNUC__)
extern "C" void ai_chat_append_native_trace(const char *) __attribute__((weak));
#endif]=])
        string(REPLACE "#include <unordered_map>" "${_ai_chat_trace_include}" _ai_chat_backend_source "${_ai_chat_backend_source}")
    endif()
    string(REPLACE "${_ai_chat_assignment_needle}" "${_ai_chat_assignment_replacement}" _ai_chat_backend_source "${_ai_chat_backend_source}")
    file(WRITE "${_ai_chat_backend_cpp}" "${_ai_chat_backend_source}")
endif()
# Ensure the scheduler's controlled error path has its required standard exception declaration,
# including when CMake reconfigures an already-patched FetchContent checkout.
file(READ "${_ai_chat_backend_cpp}" _ai_chat_backend_include_source)
if(NOT _ai_chat_backend_include_source MATCHES "#include <stdexcept>")
    set(_ai_chat_stdexcept_include [=[#include <string>
#include <stdexcept>]=])
    string(REPLACE "#include <string>" "${_ai_chat_stdexcept_include}"
        _ai_chat_backend_include_source "${_ai_chat_backend_include_source}")
    file(WRITE "${_ai_chat_backend_cpp}" "${_ai_chat_backend_include_source}")
endif()
file(READ "${_ai_chat_backend_cpp}" _ai_chat_backend_profile_verify)
if(NOT _ai_chat_backend_profile_verify MATCHES "AI_CHAT_BACKEND_ASSIGNMENT_PROFILE_V1")
    message(FATAL_ERROR "Scheduler backend assignment profiling patch was not applied")
endif()

# Android Vulkan drivers are not uniformly reliable with llama.cpp's pinned-memory
# asynchronous model-upload path. The app has already observed a SIGSEGV at a null
# address while loading Qwen3.5 immediately after "using async uploads". Disable only
# this *load-time transfer optimization* on Android; tensors still upload to Vulkan
# buffers synchronously, so this does not enable CPU inference or alter model quality.
# Keep the patch narrow, source-verified, and idempotent.
if (ANDROID)
    set(AI_CHAT_LOADER_SOURCE "${llama_cpp_SOURCE_DIR}/src/llama-model-loader.cpp")
    if (NOT EXISTS "${AI_CHAT_LOADER_SOURCE}")
        message(FATAL_ERROR "Pinned llama.cpp model loader source is missing")
    endif()
    file(READ "${AI_CHAT_LOADER_SOURCE}" AI_CHAT_LOADER_TEXT)
    string(FIND "${AI_CHAT_LOADER_TEXT}" "AI_CHAT_ANDROID_DISABLE_ASYNC_UPLOADS_V1" AI_CHAT_ASYNC_UPLOAD_PATCHED)
    if (AI_CHAT_ASYNC_UPLOAD_PATCHED EQUAL -1)
        set(AI_CHAT_ASYNC_UPLOAD_ORIGINAL "        if (use_mmap || check_tensors) {")
        string(FIND "${AI_CHAT_LOADER_TEXT}" "${AI_CHAT_ASYNC_UPLOAD_ORIGINAL}" AI_CHAT_ASYNC_UPLOAD_LOCATION)
        if (AI_CHAT_ASYNC_UPLOAD_LOCATION EQUAL -1)
            message(FATAL_ERROR "Could not locate pinned llama.cpp async upload gate; refusing an unsafe patch")
        endif()
        string(REPLACE
            "${AI_CHAT_ASYNC_UPLOAD_ORIGINAL}"
            "        // AI_CHAT_ANDROID_DISABLE_ASYNC_UPLOADS_V1: synchronous staging is safer on Android Vulkan drivers.\n#if defined(__ANDROID__)\n        return nullptr;\n#endif\n        if (use_mmap || check_tensors) {"
            AI_CHAT_LOADER_TEXT "${AI_CHAT_LOADER_TEXT}")
        file(WRITE "${AI_CHAT_LOADER_SOURCE}" "${AI_CHAT_LOADER_TEXT}")
    endif()
endif()

# Android devices may expose Vulkan 1.1 without core Vulkan 1.2 buffer-device-address
# support. Keep the optional BDA optimization disabled on those devices rather than
# letting the backend use an unavailable core entry point. This does not disable Vulkan.
set(AI_CHAT_VULKAN_SOURCE "${llama_cpp_SOURCE_DIR}/ggml/src/ggml-vulkan/ggml-vulkan.cpp")
if (NOT EXISTS "${AI_CHAT_VULKAN_SOURCE}")
    message(FATAL_ERROR "Pinned llama.cpp Vulkan source is missing: ${AI_CHAT_VULKAN_SOURCE}")
endif()
file(READ "${AI_CHAT_VULKAN_SOURCE}" AI_CHAT_VULKAN_SOURCE_TEXT)
string(FIND "${AI_CHAT_VULKAN_SOURCE_TEXT}" "AI_CHAT_VULKAN_ANDROID_BDA_GUARD_V1" AI_CHAT_VULKAN_BDA_PATCHED)
if (AI_CHAT_VULKAN_BDA_PATCHED EQUAL -1)
    string(FIND "${AI_CHAT_VULKAN_SOURCE_TEXT}" "device->buffer_device_address = vk12_features.bufferDeviceAddress;" AI_CHAT_VULKAN_BDA_ASSIGNMENT)
    if (AI_CHAT_VULKAN_BDA_ASSIGNMENT EQUAL -1)
        message(FATAL_ERROR "Could not locate Vulkan buffer-device-address feature assignment in pinned llama.cpp")
    endif()
    string(REPLACE
        "VkPhysicalDeviceVulkan12Features vk12_features;"
        "VkPhysicalDeviceVulkan12Features vk12_features{};"
        AI_CHAT_VULKAN_SOURCE_TEXT "${AI_CHAT_VULKAN_SOURCE_TEXT}")
    string(REPLACE
        "device->buffer_device_address = vk12_features.bufferDeviceAddress;"
        "device->buffer_device_address = vk12_features.bufferDeviceAddress;\n        // AI_CHAT_VULKAN_ANDROID_BDA_GUARD_V1: Vulkan 1.1 devices must not use the core 1.2 BDA path.\n        if (device->properties.apiVersion < VK_API_VERSION_1_2) {\n            device->buffer_device_address = false;\n        }"
        AI_CHAT_VULKAN_SOURCE_TEXT "${AI_CHAT_VULKAN_SOURCE_TEXT}")
    file(WRITE "${AI_CHAT_VULKAN_SOURCE}" "${AI_CHAT_VULKAN_SOURCE_TEXT}")
endif()

if (NOT TARGET ggml-vulkan)
    message(FATAL_ERROR "GPU-only runtime requires the statically linked ggml-vulkan target")
endif()

add_library(ai_chat_runtime SHARED native_runtime_android_safe.cpp)

target_include_directories(ai_chat_runtime PRIVATE
    ${llama_cpp_SOURCE_DIR}/include
    ${llama_cpp_SOURCE_DIR}/src
    ${llama_cpp_SOURCE_DIR}/ggml/include
    ${llama_cpp_SOURCE_DIR}/ggml/src
)

target_compile_options(ai_chat_runtime PRIVATE
    -Os
    -ffunction-sections
    -fdata-sections
)
target_link_options(ai_chat_runtime PRIVATE
    -Wl,--gc-sections
    # Preserve native function symbols so Android tombstones can be symbolicated.
    -Wl,--strip-debug
)

target_link_libraries(ai_chat_runtime PRIVATE
    llama
    llama-common
    ggml-vulkan
    android
    log
    "${Vulkan_LIBRARY}"
)

target_compile_definitions(ai_chat_runtime PRIVATE
    AI_CHAT_LLAMA_CPP_SHA="${LLAMA_CPP_TAG}"
    AI_CHAT_GPU_BACKEND_VULKAN=1
    AI_CHAT_REQUIRE_VULKAN_GPU=1
)

if (ANDROID)
    set_target_properties(ai_chat_runtime PROPERTIES
        CXX_VISIBILITY_PRESET hidden
        VISIBILITY_INLINES_HIDDEN ON
    )
endif()
