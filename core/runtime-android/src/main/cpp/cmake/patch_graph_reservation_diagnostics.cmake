# Instrument the pinned llama.cpp graph reservation and WebGPU buffer allocation.
# This is diagnostic-only: it does not change backend placement or allocation sizes.
if (NOT DEFINED llama_cpp_SOURCE_DIR)
    message(FATAL_ERROR "llama_cpp_SOURCE_DIR is required")
endif()

set(_backend_file "${llama_cpp_SOURCE_DIR}/ggml/src/ggml-backend.cpp")
set(_webgpu_file "${llama_cpp_SOURCE_DIR}/ggml/src/ggml-webgpu/ggml-webgpu.cpp")
if (NOT EXISTS "${_backend_file}" OR NOT EXISTS "${_webgpu_file}")
    message(FATAL_ERROR "AI Chat graph diagnostics: pinned GGML source files are missing")
endif()

# Scheduler reserve: record graph shape, backend count, elapsed time and outcome.
file(READ "${_backend_file}" _backend)
if (NOT _backend MATCHES "AI_CHAT_GRAPH_RESERVE_DIAGNOSTICS_V1")
    set(_reserve_anchor [=[
        if (!ggml_gallocr_reserve_n(sched->galloc, &sched->graph, sched->node_backend_ids, sched->leaf_backend_ids)) {
            GGML_LOG_ERROR("%s: failed to reserve graph buffers\n", __func__);
            return false;
        }
]=])
    set(_reserve_replacement [=[
        // AI_CHAT_GRAPH_RESERVE_DIAGNOSTICS_V1
        const auto ai_chat_reserve_started = std::chrono::steady_clock::now();
        GGML_LOG_INFO("AI_CHAT_GRAPH_RESERVE_BEGIN nodes=%d leafs=%d backends=%d splits=%d\n",
            sched->graph.n_nodes, sched->graph.n_leafs, sched->n_backends, sched->n_splits);
        const bool ai_chat_reserve_ok = ggml_gallocr_reserve_n(
            sched->galloc, &sched->graph, sched->node_backend_ids, sched->leaf_backend_ids);
        const double ai_chat_reserve_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - ai_chat_reserve_started).count();
        GGML_LOG_INFO("AI_CHAT_GRAPH_RESERVE_END success=%d nodes=%d leafs=%d backends=%d elapsedMs=%.3f\n",
            ai_chat_reserve_ok ? 1 : 0, sched->graph.n_nodes, sched->graph.n_leafs,
            sched->n_backends, ai_chat_reserve_ms);
        if (!ai_chat_reserve_ok) {
            GGML_LOG_ERROR("%s: failed to reserve graph buffers\n", __func__);
            return false;
        }
]=])
    string(FIND "${_backend}" "${_reserve_anchor}" _reserve_pos)
    if (_reserve_pos LESS 0)
        message(FATAL_ERROR "AI Chat graph diagnostics: ggml_gallocr_reserve_n anchor not found in pinned scheduler")
    endif()
    string(REPLACE "${_reserve_anchor}" "${_reserve_replacement}" _backend "${_backend}")
    file(WRITE "${_backend_file}" "${_backend}")
endif()

# WebGPU: record exact requested and aligned buffer sizes immediately before and
# after Dawn's synchronous buffer creation. If Dawn aborts, the BEGIN line remains.
file(READ "${_webgpu_file}" _webgpu)
if (NOT _webgpu MATCHES "AI_CHAT_WEBGPU_BUFFER_DIAGNOSTICS_V1")
    set(_buffer_anchor [=[
    ggml_webgpu_create_buffer(ctx->webgpu_global_ctx->device, buf, ROUNDUP_POW2(size, WEBGPU_STORAGE_BUF_BINDING_MULT),
                              wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc | wgpu::BufferUsage::CopyDst,
                              buf_name.c_str());
]=])
    set(_buffer_replacement [=[
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
    string(FIND "${_webgpu}" "${_buffer_anchor}" _buffer_pos)
    if (_buffer_pos LESS 0)
        message(FATAL_ERROR "AI Chat graph diagnostics: WebGPU buffer allocation anchor not found in pinned source")
    endif()
    string(REPLACE "${_buffer_anchor}" "${_buffer_replacement}" _webgpu "${_webgpu}")
    file(WRITE "${_webgpu_file}" "${_webgpu}")
endif()

message(STATUS "AI Chat: graph-reservation and WebGPU allocation diagnostics enabled")
