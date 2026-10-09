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

# Android GPU-only contract: keep llama.cpp's CPU backend available as an internal
# helper for host-side tensor utilities, but exclude it from the graph scheduler.
# This prevents unsupported GGML nodes from silently being reassigned to CPU.
set(_llama_context_cpp "${llama_cpp_SOURCE_DIR}/src/llama-context.cpp")
file(READ "${_llama_context_cpp}" _llama_context_source)
if(NOT _llama_context_source MATCHES "AI_CHAT_GPU_ONLY_SCHEDULER_PATCH")
    # Store multiline C++ fragments in variables first, then pass quoted expansions
    # to string(REPLACE). This prevents CMake list splitting on C++ semicolons.
    set(_scheduler_needle [=[            auto backend_type = ggml_backend_dev_type(ggml_backend_get_device(backend.get()));

            if (backend_type == GGML_BACKEND_DEVICE_TYPE_CPU && !model.devices.empty()) {]=])
    set(_scheduler_replacement [=[            auto backend_type = ggml_backend_dev_type(ggml_backend_get_device(backend.get()));

            // AI_CHAT_GPU_ONLY_SCHEDULER_PATCH: CPU remains initialized as a host helper,
            // but it is never inserted into the graph scheduler backend list.
            if (backend_type == GGML_BACKEND_DEVICE_TYPE_CPU) {
                continue;
            }

            if (backend_type == GGML_BACKEND_DEVICE_TYPE_CPU && !model.devices.empty()) {]=])
    set(_scheduler_cleanup_needle [=[    ggml_opt_free(opt_ctx);
}]=])
    set(_scheduler_cleanup_replacement [=[    ggml_opt_free(opt_ctx);
    if (backend_cpu) {
        ggml_backend_free(backend_cpu);
        backend_cpu = nullptr;
    }
}]=])
    string(REPLACE "${_scheduler_needle}" "${_scheduler_replacement}" _llama_context_source "${_llama_context_source}")
    string(REPLACE "${_scheduler_cleanup_needle}" "${_scheduler_cleanup_replacement}" _llama_context_source "${_llama_context_source}")
    file(WRITE "${_llama_context_cpp}" "${_llama_context_source}")
endif()

# Fail configuration if the pinned llama.cpp source could not be patched. In GPU-only
# mode a silent miss here would re-enable the CPU scheduler and violate the runtime contract.
file(READ "${_llama_context_cpp}" _llama_context_verify)
if(NOT _llama_context_verify MATCHES "AI_CHAT_GPU_ONLY_SCHEDULER_PATCH")
    message(FATAL_ERROR "AI_CHAT_GPU_ONLY_SCHEDULER_PATCH was not applied to pinned llama-context.cpp; refusing to build a CPU-fallback runtime")
endif()

# Add traceable scheduler reservation stages to the pinned llama.cpp source.
# llama_log_set() forwards these markers to the app-private native trace, so
# an abort inside scheduler reservation is distinguishable from later context work.
file(READ "${_llama_context_cpp}" _llama_context_source)
if(NOT _llama_context_source MATCHES "AI_CHAT_SCHEDULER_TRACE_PATCH")
    set(_trace_reserve_needle [=[    LLAMA_LOG_INFO("%s: reserving ...\n", __func__);]=])
    set(_trace_reserve_replacement [=[    // AI_CHAT_SCHEDULER_TRACE_PATCH
    LLAMA_LOG_INFO("AI_CHAT_SCHED_STAGE=reserve_enter\n");
    LLAMA_LOG_INFO("%s: reserving ...\n", __func__);]=])
    set(_trace_graph_needle [=[    gf_res_reserve.reset(new llm_graph_result(max_nodes));]=])
    set(_trace_graph_replacement [=[    LLAMA_LOG_INFO("AI_CHAT_SCHED_STAGE=before_graph_result_reserve\n");
    gf_res_reserve.reset(new llm_graph_result(max_nodes));
    LLAMA_LOG_INFO("AI_CHAT_SCHED_STAGE=after_graph_result_reserve\n");]=])
    set(_trace_sched_needle [=[    sched.reset(ggml_backend_sched_new(backend_ptrs.data(), backend_buft.data(), backend_ptrs.size(), max_nodes, cparams.pipeline_parallel, cparams.op_offload));]=])
    set(_trace_sched_replacement [=[    LLAMA_LOG_INFO("AI_CHAT_SCHED_STAGE=before_backend_scheduler_create\n");
    sched.reset(ggml_backend_sched_new(backend_ptrs.data(), backend_buft.data(), backend_ptrs.size(), max_nodes, cparams.pipeline_parallel, cparams.op_offload));
    LLAMA_LOG_INFO("AI_CHAT_SCHED_STAGE=after_backend_scheduler_create\n");]=])
    set(_trace_memory_needle [=[        mctx = memory->init_full();]=])
    set(_trace_memory_replacement [=[        LLAMA_LOG_INFO("AI_CHAT_SCHED_STAGE=before_memory_init_full\n");
        mctx = memory->init_full();
        LLAMA_LOG_INFO("AI_CHAT_SCHED_STAGE=after_memory_init_full\n");]=])
    set(_trace_fused_needle [=[    resolve_fused_ops(mctx.get(), n_seqs);]=])
    set(_trace_fused_replacement [=[    LLAMA_LOG_INFO("AI_CHAT_SCHED_STAGE=before_resolve_fused_ops\n");
    resolve_fused_ops(mctx.get(), n_seqs);
    LLAMA_LOG_INFO("AI_CHAT_SCHED_STAGE=after_resolve_fused_ops\n");]=])
    set(_trace_pp_needle [=[    // reserve worst-case graph
    int n_splits_pp        = -1;]=])
    set(_trace_pp_replacement [=[    // reserve worst-case graph
    LLAMA_LOG_INFO("AI_CHAT_SCHED_STAGE=before_prompt_graph_reserve\n");
    int n_splits_pp        = -1;]=])
    set(_trace_pp_done_needle [=[                model.hparams.no_alloc, model.hparams.no_alloc ? backend_buf_exp_size.data() : nullptr);
        if (!gf) {]=])
    set(_trace_pp_done_replacement [=[                model.hparams.no_alloc, model.hparams.no_alloc ? backend_buf_exp_size.data() : nullptr);
        LLAMA_LOG_INFO("AI_CHAT_SCHED_STAGE=after_prompt_graph_reserve\n");
        if (!gf) {]=])
    set(_trace_tg_needle [=[        auto * gf = graph_reserve(n_seqs, n_seqs, n_seqs, mctx.get(), model.hparams.no_alloc);]=])
    set(_trace_tg_replacement [=[        LLAMA_LOG_INFO("AI_CHAT_SCHED_STAGE=before_token_graph_reserve\n");
        auto * gf = graph_reserve(n_seqs, n_seqs, n_seqs, mctx.get(), model.hparams.no_alloc);
        LLAMA_LOG_INFO("AI_CHAT_SCHED_STAGE=after_token_graph_reserve\n");]=])
    set(_trace_done_needle [=[    const int64_t t_end_us = ggml_time_us();]=])
    set(_trace_done_replacement [=[    LLAMA_LOG_INFO("AI_CHAT_SCHED_STAGE=reserve_graphs_completed\n");
    const int64_t t_end_us = ggml_time_us();]=])
    string(REPLACE "${_trace_reserve_needle}" "${_trace_reserve_replacement}" _llama_context_source "${_llama_context_source}")
    string(REPLACE "${_trace_graph_needle}" "${_trace_graph_replacement}" _llama_context_source "${_llama_context_source}")
    string(REPLACE "${_trace_sched_needle}" "${_trace_sched_replacement}" _llama_context_source "${_llama_context_source}")
    string(REPLACE "${_trace_memory_needle}" "${_trace_memory_replacement}" _llama_context_source "${_llama_context_source}")
    string(REPLACE "${_trace_fused_needle}" "${_trace_fused_replacement}" _llama_context_source "${_llama_context_source}")
    string(REPLACE "${_trace_pp_needle}" "${_trace_pp_replacement}" _llama_context_source "${_llama_context_source}")
    string(REPLACE "${_trace_pp_done_needle}" "${_trace_pp_done_replacement}" _llama_context_source "${_llama_context_source}")
    string(REPLACE "${_trace_tg_needle}" "${_trace_tg_replacement}" _llama_context_source "${_llama_context_source}")
    string(REPLACE "${_trace_done_needle}" "${_trace_done_replacement}" _llama_context_source "${_llama_context_source}")
    file(WRITE "${_llama_context_cpp}" "${_llama_context_source}")
endif()
file(READ "${_llama_context_cpp}" _llama_context_trace_verify)
foreach(_trace_marker
        "AI_CHAT_SCHEDULER_TRACE_PATCH"
        "AI_CHAT_SCHED_STAGE=before_backend_scheduler_create"
        "AI_CHAT_SCHED_STAGE=before_memory_init_full"
        "AI_CHAT_SCHED_STAGE=before_resolve_fused_ops"
        "AI_CHAT_SCHED_STAGE=before_prompt_graph_reserve"
        "AI_CHAT_SCHED_STAGE=after_prompt_graph_reserve"
        "AI_CHAT_SCHED_STAGE=before_token_graph_reserve"
        "AI_CHAT_SCHED_STAGE=reserve_graphs_completed")
    if(NOT _llama_context_trace_verify MATCHES "${_trace_marker}")
        message(FATAL_ERROR "Required native scheduler diagnostic marker missing: ${_trace_marker}")
    endif()
endforeach()

# Trace allocations inside llm_graph_result::reset(). The latest device log
# stops while constructing this object, before the scheduler itself is created.
set(_llama_graph_cpp "${llama_cpp_SOURCE_DIR}/src/llama-graph.cpp")
file(READ "${_llama_graph_cpp}" _llama_graph_source)
if(NOT _llama_graph_source MATCHES "AI_CHAT_GRAPH_RESULT_TRACE_PATCH")
    set(_graph_resize_needle [=[    buf_compute_meta.resize(ggml_tensor_overhead()*max_nodes + ggml_graph_overhead_custom(max_nodes, false));]=])
    set(_graph_resize_replacement [=[    // AI_CHAT_GRAPH_RESULT_TRACE_PATCH
    LLAMA_LOG_INFO("AI_CHAT_GRAPH_RESULT_STAGE=before_compute_meta_resize max_nodes=%lld\\n", (long long) max_nodes);
    buf_compute_meta.resize(ggml_tensor_overhead()*max_nodes + ggml_graph_overhead_custom(max_nodes, false));
    LLAMA_LOG_INFO("AI_CHAT_GRAPH_RESULT_STAGE=after_compute_meta_resize bytes=%zu\\n", buf_compute_meta.size());]=])
    set(_graph_init_needle [=[    ctx_compute.reset(ggml_init(params));]=])
    set(_graph_init_replacement [=[    LLAMA_LOG_INFO("AI_CHAT_GRAPH_RESULT_STAGE=before_ggml_init bytes=%zu\\n", buf_compute_meta.size());
    ctx_compute.reset(ggml_init(params));
    LLAMA_LOG_INFO("AI_CHAT_GRAPH_RESULT_STAGE=after_ggml_init success=%d\\n", ctx_compute ? 1 : 0);]=])
    set(_graph_new_needle [=[    gf = ggml_new_graph_custom(ctx_compute.get(), max_nodes, false);]=])
    set(_graph_new_replacement [=[    LLAMA_LOG_INFO("AI_CHAT_GRAPH_RESULT_STAGE=before_new_graph_custom max_nodes=%lld\\n", (long long) max_nodes);
    gf = ggml_new_graph_custom(ctx_compute.get(), max_nodes, false);
    LLAMA_LOG_INFO("AI_CHAT_GRAPH_RESULT_STAGE=after_new_graph_custom success=%d\\n", gf ? 1 : 0);]=])
    string(REPLACE "${_graph_resize_needle}" "${_graph_resize_replacement}" _llama_graph_source "${_llama_graph_source}")
    string(REPLACE "${_graph_init_needle}" "${_graph_init_replacement}" _llama_graph_source "${_llama_graph_source}")
    string(REPLACE "${_graph_new_needle}" "${_graph_new_replacement}" _llama_graph_source "${_llama_graph_source}")
    file(WRITE "${_llama_graph_cpp}" "${_llama_graph_source}")
endif()
file(READ "${_llama_graph_cpp}" _llama_graph_trace_verify)
foreach(_graph_marker
        "AI_CHAT_GRAPH_RESULT_TRACE_PATCH"
        "AI_CHAT_GRAPH_RESULT_STAGE=before_compute_meta_resize"
        "AI_CHAT_GRAPH_RESULT_STAGE=before_ggml_init"
        "AI_CHAT_GRAPH_RESULT_STAGE=before_new_graph_custom"
        "AI_CHAT_GRAPH_RESULT_STAGE=after_new_graph_custom")
    if(NOT _llama_graph_trace_verify MATCHES "${_graph_marker}")
        message(FATAL_ERROR "Required graph-result diagnostic marker missing: ${_graph_marker}")
    endif()
endforeach()

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
    -ffunction-sections
    -fdata-sections
)
target_link_options(ai_chat_runtime PRIVATE
    -Wl,--gc-sections
    -Wl,--strip-all
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
