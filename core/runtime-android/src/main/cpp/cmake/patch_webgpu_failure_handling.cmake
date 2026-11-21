# Convert Dawn callback and wait failures into explicit GGML backend failures.
# AllowSpontaneous callbacks own heap-backed result state, never stack references.
if (NOT DEFINED llama_cpp_SOURCE_DIR)
    message(FATAL_ERROR "llama_cpp_SOURCE_DIR is required")
endif()
set(_webgpu_file "${llama_cpp_SOURCE_DIR}/ggml/src/ggml-webgpu/ggml-webgpu.cpp")
if (NOT EXISTS "${_webgpu_file}")
    message(FATAL_ERROR "AI Chat WebGPU failure handling: pinned source is missing")
endif()

file(READ "${_webgpu_file}" _webgpu)

# Keep terminal device loss separate from recoverable graph/wait failures.
if (NOT _webgpu MATCHES "AI_CHAT_WEBGPU_GRAPH_ERROR_STATE_V1")
    ai_chat_replace_once(_webgpu
        "static std::atomic<bool> g_webgpu_device_error{false};"
        "static std::atomic<bool> g_webgpu_device_error{false};\nstatic std::atomic<bool> g_webgpu_graph_error{false};"
        "separate device loss from recoverable graph errors")
    string(APPEND _webgpu "\n// AI_CHAT_WEBGPU_GRAPH_ERROR_STATE_V1\n")
endif()

if (NOT _webgpu MATCHES "AI_CHAT_WEBGPU_WAIT_RESULT_V2")
    if (NOT _webgpu MATCHES "#include <mutex>")
        string(REPLACE "#include <memory>" "#include <memory>\n#include <mutex>" _webgpu "${_webgpu}")
    endif()

    set(_wait_start "template <typename T>\nstatic void ggml_backend_webgpu_check_wait_status(")
    set(_wait_end "\nstatic void ggml_backend_webgpu_submit_commands(")
    string(FIND "${_webgpu}" "${_wait_start}" _wait_pos)
    string(FIND "${_webgpu}" "${_wait_end}" _wait_end_pos)
    if (_wait_pos LESS 0 OR _wait_end_pos LESS 0 OR _wait_end_pos LESS _wait_pos)
        message(FATAL_ERROR "AI Chat WebGPU failure handling: wait helper boundaries not found")
    endif()
    string(SUBSTRING "${_webgpu}" 0 ${_wait_pos} _webgpu_before_wait)
    string(SUBSTRING "${_webgpu}" ${_wait_end_pos} -1 _webgpu_after_wait)
    set(_wait_replacement [=[
struct ai_chat_webgpu_wait_result {
    std::atomic<int> status{-1};
    std::mutex message_mutex;
    std::string message;
};

// AI_CHAT_WEBGPU_WAIT_RESULT_V2
template <typename T>
static bool ggml_backend_webgpu_check_wait_status(wgpu::WaitStatus wait_status,
                                                  T callback_status,
                                                  T success_status,
                                                  const char * wait_name,
                                                  const char * failure_name,
                                                  const char * callback_message) {
    const bool wait_ok = wait_status == wgpu::WaitStatus::Success;
    const bool callback_ok = callback_status == success_status;
    if (!wait_ok || !callback_ok) {
        g_webgpu_graph_error.store(true, std::memory_order_release);
        if (wait_status == wgpu::WaitStatus::TimedOut) {
            GGML_LOG_ERROR("AI_CHAT_WEBGPU_WAIT_FAILED kind=%s reason=timeout timeoutMs=%u callbackStatus=%d message=%s\n",
                wait_name, WEBGPU_RUNTIME_WAIT_TIMEOUT_MS, static_cast<int>(callback_status), callback_message);
        } else if (wait_status == wgpu::WaitStatus::Error) {
            GGML_LOG_ERROR("AI_CHAT_WEBGPU_WAIT_FAILED kind=%s reason=wait_error callbackStatus=%d message=%s\n",
                wait_name, static_cast<int>(callback_status), callback_message);
        } else {
            GGML_LOG_ERROR("AI_CHAT_WEBGPU_WAIT_FAILED kind=%s reason=callback_error callbackKind=%s callbackStatus=%d message=%s\n",
                wait_name, failure_name, static_cast<int>(callback_status), callback_message);
        }
        return false;
    }
    return true;
}

static uint32_t ggml_backend_webgpu_get_max_inflight_batches() {
    return UINT32_MAX;
}

static uint32_t ggml_backend_webgpu_get_command_submit_batch_size() {
    return WEBGPU_DEFAULT_COMMAND_SUBMIT_BATCH_SIZE;
}

static bool ggml_backend_webgpu_wait_queue(webgpu_global_context & ctx) {
    if (ctx->device_error.load(std::memory_order_acquire) ||
        g_webgpu_graph_error.load(std::memory_order_acquire)) return false;
    auto result = std::make_shared<ai_chat_webgpu_wait_result>();
    const wgpu::WaitStatus wait_status = ctx->instance.WaitAny(
        ctx->queue.OnSubmittedWorkDone(
            wgpu::CallbackMode::AllowSpontaneous,
            [result](wgpu::QueueWorkDoneStatus status, wgpu::StringView message) {
                {
                    std::lock_guard<std::mutex> lock(result->message_mutex);
                    result->message = std::string(message);
                }
                result->status.store(static_cast<int>(status), std::memory_order_release);
            }),
        WEBGPU_RUNTIME_WAIT_TIMEOUT_NS);
    const int status_value = result->status.load(std::memory_order_acquire);
    std::string message;
    {
        std::lock_guard<std::mutex> lock(result->message_mutex);
        message = result->message;
    }
    if (status_value < 0) {
        g_webgpu_graph_error.store(true, std::memory_order_release);
        GGML_LOG_ERROR("AI_CHAT_WEBGPU_WAIT_FAILED kind=queue reason=callback_not_delivered waitStatus=%d\n",
            static_cast<int>(wait_status));
        return false;
    }
    const auto callback_status = static_cast<wgpu::QueueWorkDoneStatus>(status_value);
    return ggml_backend_webgpu_check_wait_status(wait_status, callback_status,
        wgpu::QueueWorkDoneStatus::Success, "Queue wait", "Queue work", message.c_str());
}

static bool ggml_backend_webgpu_map_buffer(webgpu_global_context & ctx,
                                           wgpu::Buffer & buffer,
                                           wgpu::MapMode mode,
                                           size_t offset,
                                           size_t size) {
    if (ctx->device_error.load(std::memory_order_acquire) ||
        g_webgpu_graph_error.load(std::memory_order_acquire)) return false;
    auto result = std::make_shared<ai_chat_webgpu_wait_result>();
    const wgpu::WaitStatus wait_status = ctx->instance.WaitAny(
        buffer.MapAsync(mode, offset, size, wgpu::CallbackMode::AllowSpontaneous,
            [result](wgpu::MapAsyncStatus status, wgpu::StringView message) {
                {
                    std::lock_guard<std::mutex> lock(result->message_mutex);
                    result->message = std::string(message);
                }
                result->status.store(static_cast<int>(status), std::memory_order_release);
            }),
        WEBGPU_RUNTIME_WAIT_TIMEOUT_NS);
    const int status_value = result->status.load(std::memory_order_acquire);
    std::string message;
    {
        std::lock_guard<std::mutex> lock(result->message_mutex);
        message = result->message;
    }
    if (status_value < 0) {
        g_webgpu_graph_error.store(true, std::memory_order_release);
        GGML_LOG_ERROR("AI_CHAT_WEBGPU_WAIT_FAILED kind=map reason=callback_not_delivered waitStatus=%d\n",
            static_cast<int>(wait_status));
        return false;
    }
    const auto callback_status = static_cast<wgpu::MapAsyncStatus>(status_value);
    return ggml_backend_webgpu_check_wait_status(wait_status, callback_status,
        wgpu::MapAsyncStatus::Success, "Buffer map wait", "Buffer map", message.c_str());
}
]=])
    set(_webgpu "${_webgpu_before_wait}${_wait_replacement}${_webgpu_after_wait}")
endif()

# Stop command submission after a failed wait; return the failure through graph_compute.
if (NOT _webgpu MATCHES "AI_CHAT_WEBGPU_SUBMIT_GUARD_V2")
    ai_chat_replace_once(_webgpu
        "    if (num_inflight_batches >= ctx->global_ctx->max_inflight_batches) {\n        ggml_backend_webgpu_wait_queue(ctx->global_ctx);\n        num_inflight_batches = 0;\n    }\n\n    ctx->global_ctx->queue.Submit(1, &commands);"
        "    if (ctx->global_ctx->device_error.load(std::memory_order_acquire) || g_webgpu_graph_error.load(std::memory_order_acquire)) return;\n    if (num_inflight_batches >= ctx->global_ctx->max_inflight_batches) {\n        if (!ggml_backend_webgpu_wait_queue(ctx->global_ctx)) return;\n        num_inflight_batches = 0;\n    }\n    if (ctx->global_ctx->device_error.load(std::memory_order_acquire) || g_webgpu_graph_error.load(std::memory_order_acquire)) return;\n\n    ctx->global_ctx->queue.Submit(1, &commands);"
        "stop command submission after a failed queue wait")
    string(APPEND _webgpu "\n// AI_CHAT_WEBGPU_SUBMIT_GUARD_V2\n")
endif()

# The bounded-allocation patch already marks device loss; give the event a clear marker.
if (NOT _webgpu MATCHES "AI_CHAT_WEBGPU_NONFATAL_CALLBACKS_V2")
    ai_chat_replace_once(_webgpu
        [=[            GGML_LOG_ERROR("ggml_webgpu: Device lost! Reason: %d, Message: %s\n", static_cast<int>(reason),
                           std::string(message).c_str());]=]
        [=[            GGML_LOG_ERROR("AI_CHAT_WEBGPU_DEVICE_LOST reason=%d message=%s\n", static_cast<int>(reason),
                           std::string(message).c_str());]=]
        "identify terminal device loss in diagnostics")
    ai_chat_replace_once(_webgpu
        [=[            GGML_LOG_ERROR("ggml_webgpu: uncaptured device error (non-fatal diagnostic): reason=%d message=%s\n",
                           static_cast<int>(reason), std::string(message).c_str());]=]
        [=[            g_webgpu_graph_error.store(true, std::memory_order_release);
            GGML_LOG_ERROR("AI_CHAT_WEBGPU_UNCAPTURED_ERROR reason=%d message=%s\n",
                           static_cast<int>(reason), std::string(message).c_str());]=]
        "record uncaptured Dawn validation errors as recoverable graph failures")
    string(APPEND _webgpu "\n// AI_CHAT_WEBGPU_NONFATAL_CALLBACKS_V2\n")
endif()

# Reset only transient graph errors before a new graph. Device loss remains sticky.
if (NOT _webgpu MATCHES "AI_CHAT_WEBGPU_GRAPH_FAILURE_PROPAGATION_V2")
    ai_chat_replace_once(_webgpu
        "    webgpu_context                ctx         = backend_ctx->webgpu_ctx;"
        "    webgpu_context                ctx         = backend_ctx->webgpu_ctx;\n    if (ctx->global_ctx->device_error.load(std::memory_order_acquire)) {\n        GGML_LOG_ERROR(\"AI_CHAT_WEBGPU_GRAPH_ABORTED reason=device_lost\\n\");\n        return GGML_STATUS_FAILED;\n    }\n    g_webgpu_graph_error.store(false, std::memory_order_release);"
        "reset transient graph errors without clearing device loss")
    ai_chat_replace_once(_webgpu
        "    while (node_idx < cgraph->n_nodes) {"
        "    while (node_idx < cgraph->n_nodes) {\n        if (ctx->global_ctx->device_error.load(std::memory_order_acquire) || g_webgpu_graph_error.load(std::memory_order_acquire)) {\n            GGML_LOG_ERROR(\"AI_CHAT_WEBGPU_GRAPH_ABORTED stage=encode node=%d totalNodes=%d\\n\", node_idx, cgraph->n_nodes);\n            WEBGPU_CPU_PROFILE_TOTAL_END(graph_compute, ctx->global_ctx);\n            return GGML_STATUS_FAILED;\n        }"
        "stop graph encoding after a callback or wait failure")
    ai_chat_replace_once(_webgpu
        "    WEBGPU_CPU_PROFILE_TOTAL_END(graph_compute, ctx->global_ctx);\n    return GGML_STATUS_SUCCESS;"
        "    if (!ggml_backend_webgpu_wait_queue(ctx->global_ctx)) {\n        GGML_LOG_ERROR(\"AI_CHAT_WEBGPU_GRAPH_ABORTED stage=queue_completion nodes=%d\\n\", cgraph->n_nodes);\n        WEBGPU_CPU_PROFILE_TOTAL_END(graph_compute, ctx->global_ctx);\n        return GGML_STATUS_FAILED;\n    }\n    GGML_LOG_INFO(\"AI_CHAT_WEBGPU_GRAPH_SYNC_COMPLETED nodes=%d\\n\", cgraph->n_nodes);\n    const bool ai_chat_graph_ok = !ctx->global_ctx->device_error.load(std::memory_order_acquire) && !g_webgpu_graph_error.load(std::memory_order_acquire);\n    WEBGPU_CPU_PROFILE_TOTAL_END(graph_compute, ctx->global_ctx);\n    if (!ai_chat_graph_ok) {\n        GGML_LOG_ERROR(\"AI_CHAT_WEBGPU_GRAPH_ABORTED stage=completion nodes=%d\\n\", cgraph->n_nodes);\n        return GGML_STATUS_FAILED;\n    }\n    return GGML_STATUS_SUCCESS;"
        "propagate WebGPU failures to the GGML scheduler")
    string(APPEND _webgpu "\n// AI_CHAT_WEBGPU_GRAPH_FAILURE_PROPAGATION_V2\n")
endif()

# Independently upgrade cached V2 graph-completion patches. This ensures a completed
# WebGPU queue is observed before the backend reports graph success.
if (NOT _webgpu MATCHES "AI_CHAT_WEBGPU_QUEUE_COMPLETION_GUARD_V1")
    if (NOT _webgpu MATCHES "AI_CHAT_WEBGPU_GRAPH_SYNC_COMPLETED")
        ai_chat_replace_once(_webgpu
            "    const bool ai_chat_graph_ok = !ctx->global_ctx->device_error.load(std::memory_order_acquire) && !g_webgpu_graph_error.load(std::memory_order_acquire);\n    WEBGPU_CPU_PROFILE_TOTAL_END(graph_compute, ctx->global_ctx);\n    if (!ai_chat_graph_ok) {\n        GGML_LOG_ERROR(\"AI_CHAT_WEBGPU_GRAPH_ABORTED stage=completion nodes=%d\\n\", cgraph->n_nodes);\n        return GGML_STATUS_FAILED;\n    }\n    return GGML_STATUS_SUCCESS;"
            "    if (!ggml_backend_webgpu_wait_queue(ctx->global_ctx)) {\n        GGML_LOG_ERROR(\"AI_CHAT_WEBGPU_GRAPH_ABORTED stage=queue_completion nodes=%d\\n\", cgraph->n_nodes);\n        WEBGPU_CPU_PROFILE_TOTAL_END(graph_compute, ctx->global_ctx);\n        return GGML_STATUS_FAILED;\n    }\n    GGML_LOG_INFO(\"AI_CHAT_WEBGPU_GRAPH_SYNC_COMPLETED nodes=%d\\n\", cgraph->n_nodes);\n    const bool ai_chat_graph_ok = !ctx->global_ctx->device_error.load(std::memory_order_acquire) && !g_webgpu_graph_error.load(std::memory_order_acquire);\n    WEBGPU_CPU_PROFILE_TOTAL_END(graph_compute, ctx->global_ctx);\n    if (!ai_chat_graph_ok) {\n        GGML_LOG_ERROR(\"AI_CHAT_WEBGPU_GRAPH_ABORTED stage=completion nodes=%d\\n\", cgraph->n_nodes);\n        return GGML_STATUS_FAILED;\n    }\n    return GGML_STATUS_SUCCESS;"
            "propagate queue completion failures from cached graph patch")
    endif()
    string(APPEND _webgpu "\n// AI_CHAT_WEBGPU_QUEUE_COMPLETION_GUARD_V1\n")
    file(WRITE "${_webgpu_file}" "${_webgpu}")
endif()

# SET_ROWS overflow is a backend limitation, not a reason to terminate the app.
if (NOT _webgpu MATCHES "AI_CHAT_WEBGPU_SET_ROWS_FAILURE_V2")
    ai_chat_replace_once(_webgpu
        "        GGML_ABORT(\"ggml_webgpu: SET_ROWS index > 2^32, unsupported.\");"
        "        g_webgpu_graph_error.store(true, std::memory_order_release);\n        GGML_LOG_ERROR(\"AI_CHAT_WEBGPU_SET_ROWS_UNSUPPORTED indexWidth=64\\n\");"
        "return an explicit graph failure for unsupported SET_ROWS index width")
    string(APPEND _webgpu "\n// AI_CHAT_WEBGPU_SET_ROWS_FAILURE_V2\n")
endif()

# A failed MapAsync must not be followed by GetConstMappedRange on an unmapped buffer.
if (NOT _webgpu MATCHES "AI_CHAT_WEBGPU_READBACK_GUARDS_V2")
ai_chat_replace_once(_webgpu
    "    ggml_backend_webgpu_map_buffer(ctx->global_ctx, ctx->set_rows_host_error_buf, wgpu::MapMode::Read, 0,\n                                   ctx->set_rows_host_error_buf.GetSize());\n    const uint32_t * error_data"
    "    ggml_backend_webgpu_map_buffer(ctx->global_ctx, ctx->set_rows_host_error_buf, wgpu::MapMode::Read, 0,\n                                   ctx->set_rows_host_error_buf.GetSize());\n    if (ctx->global_ctx->device_error.load(std::memory_order_acquire) || g_webgpu_graph_error.load(std::memory_order_acquire)) return;\n    const uint32_t * error_data"
    "guard SET_ROWS mapped-range access")
ai_chat_replace_once(_webgpu
    "    ggml_backend_webgpu_map_buffer(buf_ctx->global_ctx, buf_ctx->global_ctx->get_tensor_staging_buf,\n                                   wgpu::MapMode::Read, 0, final_size);\n    // Must specify size here"
    "    ggml_backend_webgpu_map_buffer(buf_ctx->global_ctx, buf_ctx->global_ctx->get_tensor_staging_buf,\n                                   wgpu::MapMode::Read, 0, final_size);\n    if (buf_ctx->global_ctx->device_error.load(std::memory_order_acquire) || g_webgpu_graph_error.load(std::memory_order_acquire)) return;\n    // Must specify size here"
    "guard tensor readback after a failed map")

ai_chat_replace_once(_webgpu
    "    ggml_backend_webgpu_map_buffer(ctx->global_ctx, ctx->profile_timestamp_host_buf, wgpu::MapMode::Read, 0,\n                                   mapped_size);\n    const uint64_t * ts_data"
    "    ggml_backend_webgpu_map_buffer(ctx->global_ctx, ctx->profile_timestamp_host_buf, wgpu::MapMode::Read, 0,\n                                   mapped_size);\n    if (ctx->global_ctx->device_error.load(std::memory_order_acquire) || g_webgpu_graph_error.load(std::memory_order_acquire)) return;\n    const uint64_t * ts_data"
    "guard timestamp readback after a failed map")

ai_chat_replace_once(_webgpu
    "    ggml_backend_webgpu_map_buffer(ctx, ctx->debug_host_buf, wgpu::MapMode::Read, 0, ctx->debug_host_buf.GetSize());\n    const float * debug_data"
    "    ggml_backend_webgpu_map_buffer(ctx, ctx->debug_host_buf, wgpu::MapMode::Read, 0, ctx->debug_host_buf.GetSize());\n    if (ctx->device_error.load(std::memory_order_acquire) || g_webgpu_graph_error.load(std::memory_order_acquire)) return;\n    const float * debug_data"
    "guard debug readback after a failed map")

ai_chat_replace_once(_webgpu
    "    if (status == wgpu::WaitStatus::TimedOut) {\n        GGML_ABORT(\"ggml_webgpu: event_synchronize timed out after %u ms\\n\", WEBGPU_RUNTIME_WAIT_TIMEOUT_MS);\n    }\n    event_ctx->recorded = false;"
    "    if (status != wgpu::WaitStatus::Success) {\n        g_webgpu_graph_error.store(true, std::memory_order_release);\n        GGML_LOG_ERROR(\"AI_CHAT_WEBGPU_EVENT_WAIT_FAILED status=%d timeoutMs=%u\\n\", static_cast<int>(status), WEBGPU_RUNTIME_WAIT_TIMEOUT_MS);\n    }\n    event_ctx->recorded = false;"
    "replace event synchronization abort with a graph failure")
string(APPEND _webgpu "\n// AI_CHAT_WEBGPU_READBACK_GUARDS_V2\n")
endif()

file(WRITE "${_webgpu_file}" "${_webgpu}")
message(STATUS "AI Chat: safe Dawn callback lifetime, recoverable waits, and graph failure propagation enabled")
