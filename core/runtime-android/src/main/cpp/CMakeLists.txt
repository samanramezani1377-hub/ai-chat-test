cmake_minimum_required(VERSION 3.22)
project(ai_chat_runtime LANGUAGES C CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(FetchContent)
set(LLAMA_CPP_TAG "9871df5911a03a813518bcd623ee2eba91bf32aa" CACHE STRING "Pinned llama.cpp revision for the Android OpenGL ES runtime")
set(FETCHCONTENT_QUIET OFF)
set(FETCHCONTENT_BASE_DIR "${CMAKE_CURRENT_LIST_DIR}/../../.cxx/fetch-content" CACHE PATH "Stable FetchContent base directory")
set(FETCHCONTENT_UPDATES_DISCONNECTED ON CACHE BOOL "Do not update pinned FetchContent dependencies")

FetchContent_Declare(
    llama_cpp
    GIT_REPOSITORY https://github.com/ggml-org/llama.cpp.git
    GIT_TAG ${LLAMA_CPP_TAG}
    GIT_SHALLOW TRUE
)
# OpenGL ES is provided by Android's system EGL/GLES implementation.
# No vendor-specific GPU SDK is bundled in the APK.
set(GGML_BACKEND_DL OFF CACHE BOOL "Disable dynamic ggml backend loading on Android" FORCE)
set(GGML_BACKEND_DIR "" CACHE STRING "No dynamic backend directory" FORCE)
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_COMMON ON CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_APP OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_UI OFF CACHE BOOL "" FORCE)
set(LLAMA_OPENSSL OFF CACHE BOOL "" FORCE)
set(GGML_NATIVE OFF CACHE BOOL "" FORCE)
# GPU-only Android runtime: disable CPU_REPACK so quantized model weights are never diverted to CPU_REPACK.
# OpenGL ES owns all model weights; CPU remains only as an internal host-side helper.
set(GGML_CPU_REPACK OFF CACHE BOOL "Disable CPU weight repacking for GPU-only runtime" FORCE)
set(GGML_OPENMP OFF CACHE BOOL "" FORCE)
set(GGML_LLAMAFILE OFF CACHE BOOL "" FORCE)

FetchContent_MakeAvailable(llama_cpp)

# Android GPU-only contract: keep the CPU backend registered for host-side metadata
# and preallocated GGML_OP_NONE tensors, but never let it execute inference ops.
# The previous patch removed CPU from the scheduler entirely; recurrent models keep
# rs_seq and related control tensors in CPU buffers, so llama.cpp aborts on those
# preallocated GGML_OP_NONE leaves during sched_reserve().
set(_llama_context_cpp "${llama_cpp_SOURCE_DIR}/src/llama-context.cpp")
file(READ "${_llama_context_cpp}" _llama_context_source)

# Undo the earlier over-broad patch when configuring an existing FetchContent tree.
set(_old_scheduler_replacement [=[            auto backend_type = ggml_backend_dev_type(ggml_backend_get_device(backend.get()));

            // AI_CHAT_GPU_ONLY_SCHEDULER_PATCH: CPU remains initialized as a host helper,
            // but it is never inserted into the graph scheduler backend list.
            if (backend_type == GGML_BACKEND_DEVICE_TYPE_CPU) {
                continue;
            }

            if (backend_type == GGML_BACKEND_DEVICE_TYPE_CPU && !model.devices.empty()) {]=])
set(_scheduler_needle [=[            auto backend_type = ggml_backend_dev_type(ggml_backend_get_device(backend.get()));

            if (backend_type == GGML_BACKEND_DEVICE_TYPE_CPU && !model.devices.empty()) {]=])
string(REPLACE "${_old_scheduler_replacement}" "${_scheduler_needle}" _llama_context_source "${_llama_context_source}")

if(NOT _llama_context_source MATCHES "AI_CHAT_GPU_ONLY_SCHEDULER_PATCH_V2")
    set(_scheduler_replacement [=[            auto backend_type = ggml_backend_dev_type(ggml_backend_get_device(backend.get()));

            // AI_CHAT_GPU_ONLY_SCHEDULER_PATCH_V2: retain CPU backend for host metadata,
            // graph inputs, and preallocated GGML_OP_NONE tensors only.
            if (backend_type == GGML_BACKEND_DEVICE_TYPE_CPU && !model.devices.empty()) {]=])
    string(REPLACE "${_scheduler_needle}" "${_scheduler_replacement}" _llama_context_source "${_llama_context_source}")
endif()

# Remove the old direct free, which was only added when CPU was excluded from the scheduler.
set(_old_cleanup_replacement [=[    ggml_opt_free(opt_ctx);
    if (backend_cpu) {
        ggml_backend_free(backend_cpu);
        backend_cpu = nullptr;
    }
}]=])
string(REPLACE "${_old_cleanup_replacement}" [=[    ggml_opt_free(opt_ctx);
}]=] _llama_context_source "${_llama_context_source}")
file(WRITE "${_llama_context_cpp}" "${_llama_context_source}")

if(NOT _llama_context_source MATCHES "AI_CHAT_GPU_ONLY_SCHEDULER_PATCH_V2")
    message(FATAL_ERROR "AI_CHAT_GPU_ONLY_SCHEDULER_PATCH_V2 was not applied; refusing an unsafe GPU-only scheduler build")
endif()

# CPU stays in the scheduler to resolve CPU-resident metadata leaves, but it must
# not be selected for any inference/computation op. Gate both scheduler assignment
# paths; views are handled structurally by the scheduler and GGML_OP_NONE does no compute.
set(_ggml_backend_cpp "${llama_cpp_SOURCE_DIR}/ggml/src/ggml-backend.cpp")
file(READ "${_ggml_backend_cpp}" _ggml_backend_source)
if(NOT _ggml_backend_source MATCHES "AI_CHAT_GPU_ONLY_NO_CPU_COMPUTE_FALLBACK")
    set(_cpu_guard_needle [=[static void ggml_backend_sched_set_if_supported(ggml_backend_sched_t sched, struct ggml_tensor * node, int cur_backend_id, int * node_backend_id) {
    if (ggml_backend_supports_op(sched->backends[cur_backend_id], node)) {]=])
    set(_cpu_guard_replacement [=[static void ggml_backend_sched_set_if_supported(ggml_backend_sched_t sched, struct ggml_tensor * node, int cur_backend_id, int * node_backend_id) {
    // AI_CHAT_GPU_ONLY_NO_CPU_COMPUTE_FALLBACK: CPU is host/metadata-only.
    if (cur_backend_id == sched->n_backends - 1 && node->op != GGML_OP_NONE) {
        return;
    }
    if (ggml_backend_supports_op(sched->backends[cur_backend_id], node)) {]=])
    string(REPLACE "${_cpu_guard_needle}" "${_cpu_guard_replacement}" _ggml_backend_source "${_ggml_backend_source}")

    set(_cpu_pass3_needle [=[if (ggml_backend_supports_op(sched->backends[b], node)) {]=])
    set(_cpu_pass3_replacement [=[if (ggml_backend_supports_op(sched->backends[b], node) &&
                    !(b == sched->n_backends - 1 && node->op != GGML_OP_NONE)) {]=])
    string(REPLACE "${_cpu_pass3_needle}" "${_cpu_pass3_replacement}" _ggml_backend_source "${_ggml_backend_source}")
    file(WRITE "${_ggml_backend_cpp}" "${_ggml_backend_source}")
endif()

file(READ "${_ggml_backend_cpp}" _ggml_backend_verify)
if(NOT _ggml_backend_verify MATCHES "AI_CHAT_GPU_ONLY_NO_CPU_COMPUTE_FALLBACK")
    message(FATAL_ERROR "AI_CHAT_GPU_ONLY_NO_CPU_COMPUTE_FALLBACK was not applied; refusing a CPU-fallback runtime")
endif()

add_library(ai_chat_runtime SHARED native_runtime_android_safe.cpp opengles_runtime.cpp ggml-opengles/ggml-opengles.cpp ggml-opengles/ggml-opengles-registration.cpp)

# native_runtime_android_safe.cpp intentionally includes llama.cpp/ggml internal
# headers for strict backend and tensor-residency validation. Keep both
# internal source directories on the include path; the public ggml include
# directory alone is insufficient for ggml-backend-impl.h on this pinned revision.
target_include_directories(ai_chat_runtime PRIVATE
    ${llama_cpp_SOURCE_DIR}/include
    ${llama_cpp_SOURCE_DIR}/src
    ${llama_cpp_SOURCE_DIR}/ggml/include
    ${llama_cpp_SOURCE_DIR}/ggml/src
    ${CMAKE_CURRENT_LIST_DIR}/ggml-opengles
)

# The runtime is a single arm64 production native library. Strip unused sections
# and symbols at link time to reduce the installed footprint without changing inference.
target_compile_options(ai_chat_runtime PRIVATE
    -Os
    -g
    -ffunction-sections
    -fdata-sections
)
target_link_options(ai_chat_runtime PRIVATE
    -Wl,--gc-sections
    # Keep native symbols/debug line tables so Android tombstones can identify the
    # crashing function and resolve source lines with the matching unstripped build.
)

target_link_libraries(ai_chat_runtime PRIVATE llama llama-common android log EGL GLESv3)

target_compile_definitions(ai_chat_runtime PRIVATE
    AI_CHAT_LLAMA_CPP_SHA="${LLAMA_CPP_TAG}"
    AI_CHAT_GPU_BACKEND_OPENGL=1
    AI_CHAT_REQUIRE_OPENGL_GPU=1
)

if (ANDROID)
    set_target_properties(ai_chat_runtime PROPERTIES
        CXX_VISIBILITY_PRESET hidden
        VISIBILITY_INLINES_HIDDEN ON
    )
endif()
