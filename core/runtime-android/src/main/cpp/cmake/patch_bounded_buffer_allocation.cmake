# Harden GGML virtual-buffer allocation and WebGPU buffer creation.
# Applied only to the pinned llama.cpp revision; all anchors are verified.
if (NOT DEFINED llama_cpp_SOURCE_DIR)
    message(FATAL_ERROR "llama_cpp_SOURCE_DIR is required")
endif()

set(_alloc_file "${llama_cpp_SOURCE_DIR}/ggml/src/ggml-alloc.c")
set(_webgpu_file "${llama_cpp_SOURCE_DIR}/ggml/src/ggml-webgpu/ggml-webgpu.cpp")
foreach(_f IN ITEMS "${_alloc_file}" "${_webgpu_file}")
    if (NOT EXISTS "${_f}")
        message(FATAL_ERROR "AI Chat allocation hardening: missing source ${_f}")
    endif()
endforeach()

# The old allocator made its final chunk virtually unbounded (SIZE_MAX/2).
# That bypassed the backend's max chunk limit and could request an impossible
# Dawn buffer. Use bounded chunks and propagate exhaustion as reserve failure.
file(READ "${_alloc_file}" _alloc)
if (NOT _alloc MATCHES "AI_CHAT_BOUNDED_VBUFFER_ALLOCATOR_V1")
    ai_chat_replace_once(_alloc
        "#define GGML_VBUFFER_MAX_CHUNKS 16"
        "#define GGML_VBUFFER_MAX_CHUNKS 64"
        "raise bounded virtual-buffer chunk count")
    ai_chat_replace_once(_alloc
        "    int n_chunks;\n\n#ifdef GGML_ALLOCATOR_DEBUG"
        "    int n_chunks;\n    bool failed; // allocation exhaustion is reported to reserve, never process-aborted\n\n#ifdef GGML_ALLOCATOR_DEBUG"
        "add allocator failure state")
    ai_chat_replace_once(_alloc
        [=[    struct tallocr_chunk * chunk = calloc(1, sizeof(struct tallocr_chunk));
    chunk->n_free_blocks = 1;
    chunk->free_blocks[0].offset = 0;
    // available space in a chunk is limited to max_chunk_size, but can be higher if:
    // 1. a single tensor exceeds the maximum, and cannot fit any other way
    // 2. we are running out of chunks
    // backends will either manage to allocate the larger size, or report an error.
    chunk->free_blocks[0].size = MAX(min_size, alloc->max_chunk_size);
    if (alloc->n_chunks == GGML_VBUFFER_MAX_CHUNKS - 1) {
        chunk->free_blocks[0].size = SIZE_MAX/2;
    }]=]
        [=[    struct tallocr_chunk * chunk = calloc(1, sizeof(struct tallocr_chunk));
    if (chunk == NULL) {
        return -1;
    }
    chunk->n_free_blocks = 1;
    chunk->free_blocks[0].offset = 0;
    // Keep every chunk bounded by the backend's maximum buffer/binding size.
    // An individually oversized tensor is rejected by the backend allocator.
    chunk->free_blocks[0].size = MAX(min_size, alloc->max_chunk_size);]=]
        "remove unbounded final virtual-buffer chunk")
    ai_chat_replace_once(_alloc
        [=[    if (best_fit_chunk == -1) {
        // since the last chunk always has virtually endless memory, this should never happen
        GGML_LOG_ERROR("%s: not enough space in the buffer to allocate %zu bytes, largest block available %zu bytes\n",
            __func__, size, max_avail);
        GGML_ABORT("graph allocation: failed to reserve memory");
    }]=]
        [=[    if (best_fit_chunk == -1) {
        GGML_LOG_ERROR("%s: bounded graph buffer allocation exhausted for %zu bytes (largest free block %zu bytes, chunks %d/%d)\n",
            __func__, size, max_avail, alloc->n_chunks, GGML_VBUFFER_MAX_CHUNKS);
        alloc->failed = true;
        return GGML_BUFFER_ADDRESS_INVALID;
    }]=]
        "propagate bounded allocator exhaustion")
    ai_chat_replace_once(_alloc
        [=[    alloc->n_chunks = 0;

#ifdef GGML_ALLOCATOR_DEBUG]=]
        [=[    alloc->n_chunks = 0;
    alloc->failed = false;

#ifdef GGML_ALLOCATOR_DEBUG]=]
        "reset allocator failure state")
    ai_chat_replace_once(_alloc
        [=[    ggml_gallocr_alloc_graph_impl(galloc, graph, node_buffer_ids, leaf_buffer_ids);

    // set the node_allocs from the hash table]=]
        [=[    ggml_gallocr_alloc_graph_impl(galloc, graph, node_buffer_ids, leaf_buffer_ids);

    // AI_CHAT_BOUNDED_VBUFFER_ALLOCATOR_V1: fail reservation cleanly before invalid
    // chunk addresses can be consumed by node/leaf allocation metadata.
    for (int i = 0; i < galloc->n_buffers; i++) {
        if (galloc->buf_tallocs[i]->failed) {
            GGML_LOG_ERROR("%s: graph reservation failed: backend buffer allocator exhausted (buffer type %s)\n",
                __func__, ggml_backend_buft_name(galloc->bufts[i]));
            return false;
        }
    }

    // set the node_allocs from the hash table]=]
        "propagate allocator failure from graph planning")
    ai_chat_replace_once(_alloc
        [=[    if (!ggml_gallocr_is_allocated(galloc, node) && !ggml_impl_is_view(node)) {
        hn->allocated = true;
        assert(hn->addr.offset == 0);]=]
        [=[    if (!ggml_gallocr_is_allocated(galloc, node) && !ggml_impl_is_view(node)) {
        struct ggml_dyn_tallocr * alloc = galloc->buf_tallocs[buffer_id];
        if (alloc->failed) {
            return;
        }
        hn->allocated = true;
        assert(hn->addr.offset == 0);]=]
        "skip allocations after a bounded allocator failure")
    ai_chat_replace_once(_alloc
        [=[        // allocate tensor from the buffer
        struct ggml_dyn_tallocr * alloc = galloc->buf_tallocs[buffer_id];
        ggml_backend_buffer_type_t buft = galloc->bufts[buffer_id];]=]
        [=[        // allocate tensor from the buffer
        ggml_backend_buffer_type_t buft = galloc->bufts[buffer_id];]=]
        "avoid redeclaring the allocator after the early failure guard")
    ai_chat_replace_once(_alloc
        [=[    struct hash_node * hn = ggml_gallocr_hash_get(galloc, node);
    int buffer_id = hn->buffer_id;
    struct ggml_dyn_tallocr * alloc = galloc->buf_tallocs[buffer_id];
    ggml_backend_buffer_type_t buft = galloc->bufts[buffer_id];]=]
        [=[    struct hash_node * hn = ggml_gallocr_hash_get(galloc, node);
    int buffer_id = hn->buffer_id;
    struct ggml_dyn_tallocr * alloc = galloc->buf_tallocs[buffer_id];
    if (alloc->failed) {
        return;
    }
    ggml_backend_buffer_type_t buft = galloc->bufts[buffer_id];]=]
        "skip freeing allocator addresses after exhaustion")
    string(APPEND _alloc "\n// AI_CHAT_BOUNDED_ALLOCATOR_FAILURE_GUARDS_V1\n")
    string(APPEND _alloc "\n// AI_CHAT_BOUNDED_VBUFFER_ALLOCATOR_V1\n")
    file(WRITE "${_alloc_file}" "${_alloc}")
endif()

# WebGPU: reject allocations exceeding either the whole-buffer or storage-binding
# limit before calling Dawn. A failed buffer allocation is returned to ggml-alloc
# as NULL so graph reservation can fail cleanly and the caller can select fallback.
file(READ "${_webgpu_file}" _webgpu)
# std::nothrow and std::min are used by the allocation checks.
if (NOT _webgpu MATCHES "AI_CHAT_WEBGPU_NOTHROW_INCLUDE_V1")
    ai_chat_replace_once(_webgpu
        "#include <memory>"
        "#include <memory>\n#include <new>\n#include <algorithm> // AI_CHAT_WEBGPU_NOTHROW_INCLUDE_V1"
        "include std::nothrow and algorithm declarations")
endif()
if (NOT _webgpu MATCHES "AI_CHAT_WEBGPU_BOUNDED_BUFFER_ALLOCATION_V1")
    ai_chat_replace_once(_webgpu
        [=[    buffer = device.CreateBuffer(&buffer_desc);]=]
        [=[    GGML_LOG_INFO("AI_CHAT_WEBGPU_CREATE_BUFFER_BEGIN label=%s bytes=%llu usage=%u\n",
        label, static_cast<unsigned long long>(size), static_cast<unsigned int>(usage));
    buffer = device.CreateBuffer(&buffer_desc);
    if (!buffer) {
        GGML_LOG_ERROR("AI_CHAT_WEBGPU_CREATE_BUFFER_FAILED label=%s bytes=%llu\n",
            label, static_cast<unsigned long long>(size));
    } else {
        GGML_LOG_INFO("AI_CHAT_WEBGPU_CREATE_BUFFER_OK label=%s bytes=%llu\n",
            label, static_cast<unsigned long long>(size));
    }]=]
        "log every Dawn buffer allocation and its outcome")
    ai_chat_replace_once(_webgpu
        [=[    ggml_backend_webgpu_device_context * ctx = static_cast<ggml_backend_webgpu_device_context *>(buft->device->context);
    wgpu::Buffer                         buf;
    ggml_webgpu_create_buffer(ctx->webgpu_global_ctx->device, buf, ROUNDUP_POW2(size, WEBGPU_STORAGE_BUF_BINDING_MULT),
                              wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc | wgpu::BufferUsage::CopyDst,
                              buf_name.c_str());

    ggml_backend_webgpu_buffer_context * buf_ctx =
        new ggml_backend_webgpu_buffer_context(buf, buf_name, ctx->webgpu_global_ctx);

    return ggml_backend_buffer_init(buft, ggml_backend_webgpu_buffer_interface, buf_ctx, size);]=]
        [=[    ggml_backend_webgpu_device_context * ctx = static_cast<ggml_backend_webgpu_device_context *>(buft->device->context);
    const size_t aligned_size = ROUNDUP_POW2(size, WEBGPU_STORAGE_BUF_BINDING_MULT);
    const size_t max_buffer_size = ctx->webgpu_global_ctx->capabilities.limits.maxBufferSize;
    const size_t max_storage_binding_size = ctx->webgpu_global_ctx->capabilities.limits.maxStorageBufferBindingSize;
    if (aligned_size > max_buffer_size || aligned_size > max_storage_binding_size) {
        ctx->webgpu_global_ctx->device_error.store(true);
        GGML_LOG_ERROR("ggml_webgpu: refusing oversized tensor buffer name=%s requested=%zu aligned=%zu maxBufferSize=%zu maxStorageBufferBindingSize=%zu\n",
            buf_name.c_str(), size, aligned_size, max_buffer_size, max_storage_binding_size);
        return nullptr;
    }

    wgpu::Buffer buf;
    ggml_webgpu_create_buffer(ctx->webgpu_global_ctx->device, buf, aligned_size,
                              wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc | wgpu::BufferUsage::CopyDst,
                              buf_name.c_str());
    if (!buf) {
        ctx->webgpu_global_ctx->device_error.store(true);
        GGML_LOG_ERROR("ggml_webgpu: Dawn did not create tensor buffer name=%s requested=%zu aligned=%zu\n",
            buf_name.c_str(), size, aligned_size);
        return nullptr;
    }

    ggml_backend_webgpu_buffer_context * buf_ctx =
        new (std::nothrow) ggml_backend_webgpu_buffer_context(buf, buf_name, ctx->webgpu_global_ctx);
    if (buf_ctx == nullptr) {
        ctx->webgpu_global_ctx->device_error.store(true);
        GGML_LOG_ERROR("ggml_webgpu: host allocation failed for tensor buffer context name=%s bytes=%zu\n",
            buf_name.c_str(), size);
        return nullptr;
    }

    return ggml_backend_buffer_init(buft, ggml_backend_webgpu_buffer_interface, buf_ctx, size);]=]
        "validate and safely construct WebGPU tensor buffers")
    ai_chat_replace_once(_webgpu
        [=[    return dev_ctx->webgpu_global_ctx->capabilities.limits.maxStorageBufferBindingSize;]=]
        [=[    return std::min(
        static_cast<size_t>(dev_ctx->webgpu_global_ctx->capabilities.limits.maxStorageBufferBindingSize),
        static_cast<size_t>(dev_ctx->webgpu_global_ctx->capabilities.limits.maxBufferSize));]=]
        "bound allocator chunk size by both WebGPU limits")
    ai_chat_replace_once(_webgpu
        [=[            GGML_ABORT("ggml_webgpu: Device error! Reason: %d, Message: %s\n", static_cast<int>(reason),
                       std::string(message).c_str());]=]
        [=[            GGML_LOG_ERROR("ggml_webgpu: uncaptured device error (non-fatal diagnostic): reason=%d message=%s\n",
                           static_cast<int>(reason), std::string(message).c_str());]=]
        "do not turn recoverable WebGPU validation errors into SIGABRT")
    ai_chat_replace_once(_webgpu
        "    wgpu::Queue    queue;"
        "    wgpu::Queue    queue;\n    std::atomic<bool> device_error{false};"
        "track unrecoverable WebGPU device errors")
    ai_chat_replace_once(_webgpu
        [=[        [](const wgpu::Device & device, wgpu::DeviceLostReason reason, wgpu::StringView message) {
            if (reason == wgpu::DeviceLostReason::Destroyed) {
                return;
            }
            GGML_UNUSED(device);
            GGML_LOG_ERROR("ggml_webgpu: Device lost! Reason: %d, Message: %s\n", static_cast<int>(reason),
                           std::string(message).c_str());
        });]=]
        [=[        [ctx](const wgpu::Device & device, wgpu::DeviceLostReason reason, wgpu::StringView message) {
            if (reason == wgpu::DeviceLostReason::Destroyed) {
                return;
            }
            GGML_UNUSED(device);
            ctx->webgpu_global_ctx->device_error.store(true);
            GGML_LOG_ERROR("ggml_webgpu: Device lost! Reason: %d, Message: %s\n", static_cast<int>(reason),
                           std::string(message).c_str());
        });]=]
        "mark device lost and force future operation placement away from WebGPU")
    ai_chat_replace_once(_webgpu
        [=[        [](const wgpu::Device & device, wgpu::ErrorType reason, wgpu::StringView message) {
            GGML_UNUSED(device);
            GGML_LOG_ERROR("ggml_webgpu: uncaptured device error (non-fatal diagnostic): reason=%d message=%s\n",
                           static_cast<int>(reason), std::string(message).c_str());
        });]=]
        [=[        [ctx](const wgpu::Device & device, wgpu::ErrorType reason, wgpu::StringView message) {
            GGML_UNUSED(device);
            ctx->webgpu_global_ctx->device_error.store(true);
            GGML_LOG_ERROR("ggml_webgpu: uncaptured device error; disabling WebGPU backend for subsequent graph placement: reason=%d message=%s\n",
                           static_cast<int>(reason), std::string(message).c_str());
        });]=]
        "disable WebGPU after uncaptured device errors instead of aborting")
    ai_chat_replace_once(_webgpu
        [=[    ggml_backend_webgpu_device_context * ctx = static_cast<ggml_backend_webgpu_device_context *>(buft->device->context);
    const size_t aligned_size]=]
        [=[    ggml_backend_webgpu_device_context * ctx = static_cast<ggml_backend_webgpu_device_context *>(buft->device->context);
    if (ctx->webgpu_global_ctx->device_error.load()) {
        GGML_LOG_ERROR("ggml_webgpu: refusing allocation because the device is in an error state\n");
        return nullptr;
    }
    const size_t aligned_size]=]
        "refuse allocations after a device error")
    ai_chat_replace_once(_webgpu
        [=[    ggml_backend_webgpu_device_context * ctx = static_cast<ggml_backend_webgpu_device_context *>(dev->context);

    ggml_tensor * src0 = op->src[0];]=]
        [=[    ggml_backend_webgpu_device_context * ctx = static_cast<ggml_backend_webgpu_device_context *>(dev->context);
    if (ctx->webgpu_global_ctx->device_error.load()) {
        return false;
    }

    ggml_tensor * src0 = op->src[0];]=]
        "route future graph operations away from a failed WebGPU device")
    string(APPEND _webgpu "\n// AI_CHAT_WEBGPU_BOUNDED_BUFFER_ALLOCATION_V1\n")
    file(WRITE "${_webgpu_file}" "${_webgpu}")
endif()

message(STATUS "AI Chat: bounded GGML allocator and WebGPU buffer validation enabled")
