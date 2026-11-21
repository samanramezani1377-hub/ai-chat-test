# Instrument graph planning, backend splitting, and allocator reservation as separate stages.
# The trace is intentionally independent of model architecture and quantization.
if (NOT DEFINED llama_cpp_SOURCE_DIR)
    message(FATAL_ERROR "llama_cpp_SOURCE_DIR is required")
endif()

set(_backend_file "${llama_cpp_SOURCE_DIR}/ggml/src/ggml-backend.cpp")
set(_webgpu_file "${llama_cpp_SOURCE_DIR}/ggml/src/ggml-webgpu/ggml-webgpu.cpp")
if (NOT EXISTS "${_backend_file}" OR NOT EXISTS "${_webgpu_file}")
    message(FATAL_ERROR "AI Chat graph diagnostics: pinned GGML source files are missing")
endif()

file(READ "${_backend_file}" _backend)
if (NOT _backend MATCHES "AI_CHAT_GRAPH_RESERVE_DIAGNOSTICS_V2")
    if (NOT _backend MATCHES "#include <chrono>")
        string(REPLACE "#include <algorithm>" "#include <algorithm>\n#include <chrono>" _backend "${_backend}")
    endif()

    # Replace the complete function body by locating stable function boundaries.
    # This upgrades the previous V1 patch even when FetchContent reuses its source tree.
    set(_reserve_start_anchor "bool ggml_backend_sched_reserve(ggml_backend_sched_t sched, struct ggml_cgraph * measure_graph) {")
    set(_reserve_end_anchor "\nbool ggml_backend_sched_alloc_graph(ggml_backend_sched_t sched, struct ggml_cgraph * graph) {")
    string(FIND "${_backend}" "${_reserve_start_anchor}" _reserve_start)
    string(FIND "${_backend}" "${_reserve_end_anchor}" _reserve_end)
    if (_reserve_start LESS 0 OR _reserve_end LESS 0 OR _reserve_end LESS _reserve_start)
        message(FATAL_ERROR "AI Chat graph diagnostics: scheduler reserve function boundaries not found")
    endif()
    math(EXPR _reserve_length "${_reserve_end} - ${_reserve_start}")
    string(SUBSTRING "${_backend}" 0 ${_reserve_start} _backend_before_reserve)
    string(SUBSTRING "${_backend}" ${_reserve_end} -1 _backend_after_reserve)
    set(_reserve_replacement [=[
bool ggml_backend_sched_reserve(ggml_backend_sched_t sched, struct ggml_cgraph * measure_graph) {
    GGML_ASSERT(sched);
    GGML_ASSERT((int)sched->hash_set.size >= measure_graph->n_nodes + measure_graph->n_leafs);

    // AI_CHAT_GRAPH_RESERVE_DIAGNOSTICS_V2
    const auto ai_chat_reserve_started = std::chrono::steady_clock::now();
    GGML_LOG_INFO("AI_CHAT_GRAPH_RESERVE_BEGIN inputNodes=%d inputLeafs=%d backends=%d\n",
        measure_graph->n_nodes, measure_graph->n_leafs, sched->n_backends);

    GGML_LOG_INFO("AI_CHAT_GRAPH_SYNC_BEGIN backends=%d\n", sched->n_backends);
    ggml_backend_sched_synchronize(sched);
    GGML_LOG_INFO("AI_CHAT_GRAPH_SYNC_END success=1\n");

    GGML_LOG_INFO("AI_CHAT_GRAPH_SPLIT_BEGIN inputNodes=%d inputLeafs=%d\n",
        measure_graph->n_nodes, measure_graph->n_leafs);
    ggml_backend_sched_split_graph(sched, measure_graph);
    GGML_LOG_INFO("AI_CHAT_GRAPH_SPLIT_END nodes=%d leafs=%d backends=%d splits=%d\n",
        sched->graph.n_nodes, sched->graph.n_leafs, sched->n_backends, sched->n_splits);

    const auto ai_chat_alloc_started = std::chrono::steady_clock::now();
    GGML_LOG_INFO("AI_CHAT_GRAPH_ALLOCATOR_RESERVE_BEGIN nodes=%d leafs=%d backends=%d splits=%d\n",
        sched->graph.n_nodes, sched->graph.n_leafs, sched->n_backends, sched->n_splits);
    const bool ai_chat_reserve_ok = ggml_gallocr_reserve_n(
        sched->galloc, &sched->graph, sched->node_backend_ids, sched->leaf_backend_ids);
    const double ai_chat_alloc_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - ai_chat_alloc_started).count();
    GGML_LOG_INFO("AI_CHAT_GRAPH_ALLOCATOR_RESERVE_END success=%d elapsedMs=%.3f\n",
        ai_chat_reserve_ok ? 1 : 0, ai_chat_alloc_ms);

    if (!ai_chat_reserve_ok) {
        GGML_LOG_ERROR("AI_CHAT_GRAPH_RESERVE_FAILED stage=allocator nodes=%d leafs=%d backends=%d splits=%d elapsedMs=%.3f\n",
            sched->graph.n_nodes, sched->graph.n_leafs, sched->n_backends, sched->n_splits,
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - ai_chat_reserve_started).count());
        return false;
    }

    ggml_backend_sched_reset(sched);
    GGML_LOG_INFO("AI_CHAT_GRAPH_RESERVE_END success=1 elapsedMs=%.3f\n",
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - ai_chat_reserve_started).count());
    return true;
}
]=])
    set(_backend "${_backend_before_reserve}${_reserve_replacement}${_backend_after_reserve}")

    # Give graph-split context creation its own pre-abort checkpoint.
    ai_chat_replace_once(_backend
        "void ggml_backend_sched_split_graph(ggml_backend_sched_t sched, struct ggml_cgraph * graph) {"
        "void ggml_backend_sched_split_graph(ggml_backend_sched_t sched, struct ggml_cgraph * graph) {\n    GGML_LOG_INFO(\"AI_CHAT_GRAPH_SPLIT_INTERNAL_BEGIN nodes=%d leafs=%d\\n\", graph->n_nodes, graph->n_leafs);"
        "record entry to backend graph splitting")

    ai_chat_replace_once(_backend
        "    sched->ctx = ggml_init(params);\n    if (sched->ctx == NULL) {\n        GGML_ABORT(\"%s: failed to initialize context\\n\", __func__);\n    }"
        "    GGML_LOG_INFO(\"AI_CHAT_GRAPH_SPLIT_CONTEXT_CREATE_BEGIN bytes=%zu\\n\", sched->context_buffer_size);\n    sched->ctx = ggml_init(params);\n    if (sched->ctx == NULL) {\n        GGML_LOG_ERROR(\"AI_CHAT_GRAPH_SPLIT_CONTEXT_CREATE_FAILED bytes=%zu\\n\", sched->context_buffer_size);\n        GGML_ABORT(\"%s: failed to initialize context\\n\", __func__);\n    }\n    GGML_LOG_INFO(\"AI_CHAT_GRAPH_SPLIT_CONTEXT_CREATE_END success=1\\n\");"
        "record scheduler context allocation failures before upstream abort")

    file(WRITE "${_backend_file}" "${_backend}")
endif()

# Dawn buffer allocation checkpoint. BEGIN is emitted before entering Dawn, so it
# remains useful even if the driver aborts inside CreateBuffer.
file(READ "${_webgpu_file}" _webgpu)
if (NOT _webgpu MATCHES "AI_CHAT_WEBGPU_BUFFER_DIAGNOSTICS_V2")
    set(_buffer_anchor [=[
    ggml_webgpu_create_buffer(ctx->webgpu_global_ctx->device, buf, ROUNDUP_POW2(size, WEBGPU_STORAGE_BUF_BINDING_MULT),
                              wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc | wgpu::BufferUsage::CopyDst,
                              buf_name.c_str());
]=])
    set(_buffer_replacement [=[
    // AI_CHAT_WEBGPU_BUFFER_DIAGNOSTICS_V2
    const size_t ai_chat_aligned_size = ROUNDUP_POW2(size, WEBGPU_STORAGE_BUF_BINDING_MULT);
    GGML_LOG_INFO("AI_CHAT_WEBGPU_BUFFER_BEGIN id=%d name=%s device=%s requestedBytes=%zu alignedBytes=%zu maxBufferBytes=%llu maxStorageBindingBytes=%llu\n",
        buffer_id, buf_name.c_str(), ctx->device_name.c_str(), size, ai_chat_aligned_size,
        static_cast<unsigned long long>(ctx->webgpu_global_ctx->capabilities.limits.maxBufferSize),
        static_cast<unsigned long long>(ctx->webgpu_global_ctx->capabilities.limits.maxStorageBufferBindingSize));
    ggml_webgpu_create_buffer(ctx->webgpu_global_ctx->device, buf, ai_chat_aligned_size,
                              wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc | wgpu::BufferUsage::CopyDst,
                              buf_name.c_str());
    GGML_LOG_INFO("AI_CHAT_WEBGPU_BUFFER_RETURNED id=%d name=%s success=%d requestedBytes=%zu alignedBytes=%zu\n",
        buffer_id, buf_name.c_str(), buf != nullptr ? 1 : 0, size, ai_chat_aligned_size);
]=])
    if (_webgpu MATCHES "AI_CHAT_WEBGPU_BUFFER_DIAGNOSTICS_V1")
        set(_legacy_buffer_block [=[
    // AI_CHAT_WEBGPU_BUFFER_DIAGNOSTICS_V1
    const size_t ai_chat_aligned_size = ROUNDUP_POW2(size, WEBGPU_STORAGE_BUF_BINDING_MULT);
    GGML_LOG_INFO("AI_CHAT_WEBGPU_BUFFER_BEGIN id=%d name=%s device=%s requestedBytes=%zu alignedBytes=%zu maxStorageBindingBytes=%zu\n",
        buffer_id, buf_name.c_str(), ctx->device_name.c_str(), size, ai_chat_aligned_size,
        ctx->webgpu_global_ctx->capabilities.limits.maxStorageBufferBindingSize);
    ggml_webgpu_create_buffer(ctx->webgpu_global_ctx->device, buf, ai_chat_aligned_size,
                              wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc | wgpu::BufferUsage::CopyDst,
                              buf_name.c_str());
    GGML_LOG_INFO("AI_CHAT_WEBGPU_BUFFER_RETURNED id=%d name=%s requestedBytes=%zu alignedBytes=%zu\n",
        buffer_id, buf_name.c_str(), size, ai_chat_aligned_size);
]=])
        string(FIND "${_webgpu}" "${_legacy_buffer_block}" _legacy_buffer_pos)
        if (_legacy_buffer_pos LESS 0)
            message(FATAL_ERROR "AI Chat graph diagnostics: legacy WebGPU buffer block is incomplete")
        endif()
        string(REPLACE "${_legacy_buffer_block}" "${_buffer_replacement}" _webgpu "${_webgpu}")
    else()
        string(FIND "${_webgpu}" "${_buffer_anchor}" _buffer_pos)
        if (_buffer_pos LESS 0)
            message(FATAL_ERROR "AI Chat graph diagnostics: WebGPU buffer allocation anchor not found in pinned source")
        endif()
        string(REPLACE "${_buffer_anchor}" "${_buffer_replacement}" _webgpu "${_webgpu}")
    endif()
    file(WRITE "${_webgpu_file}" "${_webgpu}")
endif()

message(STATUS "AI Chat: staged graph split/reservation and WebGPU buffer diagnostics enabled")
