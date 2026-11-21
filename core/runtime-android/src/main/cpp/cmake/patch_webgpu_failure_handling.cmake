# Convert Dawn wait/callback failures into explicit GGML backend failures.
# Callback state is heap-owned so AllowSpontaneous callbacks can never reference
# stack variables after WaitAny returns (including timeout/error paths).
if (NOT DEFINED llama_cpp_SOURCE_DIR)
    message(FATAL_ERROR "llama_cpp_SOURCE_DIR is required")
endif()
set(_webgpu_file "${llama_cpp_SOURCE_DIR}/ggml/src/ggml-webgpu/ggml-webgpu.cpp")
if (NOT EXISTS "${_webgpu_file}")
    message(FATAL_ERROR "AI Chat WebGPU failure handling: pinned source is missing")
endif()

file(READ "${_webgpu_file}" _webgpu)
if (NOT _webgpu MATCHES "AI_CHAT_WEBGPU_WAIT_RESULT_V1")
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

// AI_CHAT_WEBGPU_WAIT_RESULT_V1
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
        g_webgpu_device_error.store(true, std::memory_order_release);
        if (wait_status == wgpu::WaitStatus::TimedOut) {
            GGML_LOG_ERROR("AI_CHAT_WEBGPU_WAIT_FAILED kind=%s reason=timeout timeoutMs=%u callbackStatus=%d message=%s
",
                wait_name, WEBGPU_RUNTIME_WAIT_TIMEOUT_MS, static_cast<int>(callback_status), callback_message);
        } else if (wait_status == wgpu::WaitStatus::Error) {
            GGML_LOG_ERROR("AI_CHAT_WEBGPU_WAIT_FAILED kind=%s reason=wait_error callbackStatus=%d message=%s
",
                wait_name, static_cast<int>(callback_status), callback_message);
        } else {
            GGML_LOG_ERROR("AI_CHAT_WEBGPU_WAIT_FAILED kind=%s reason=callback_error callbackKind=%s callbackStatus=%d message=%s
",
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
    if (ctx->device_error.load(std::memory_order_acquire)) return false;
    auto result = std::make_shared<ai_chat_webgpu_wait_result>();
    const wgpu::WaitStatus wait_status = ctx->instance.WaitAny(
        ctx->queue.OnSubmittedWorkDone(
            wgpu::CallbackMode::AllowSpontaneous,
            [result](wgpu::QueueWorkDoneStatus status, wgpu::StringView message) {
                {
                    std::lock_guard<std::mutex> lock(result->message_mutex);
                    result->message.assign(message.data ? message.data : "", message.length);
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
    const auto callback_status = status_value < 0
        ? wgpu::QueueWorkDoneStatus::Error
        : static_cast<wgpu::QueueWorkDoneStatus>(status_value);
    if (status_value < 0) {
        g_webgpu_device_error.store(true, std::memory_order_release);
        GGML_LOG_ERROR("AI_CHAT_WEBGPU_WAIT_FAILED kind=queue reason=callback_not_delivered waitStatus=%d
",
            static_cast<int>(wait_status));
        return false;
    }
    return ggml_backend_webgpu_check_wait_status(wait_status, callback_status,
        wgpu::QueueWorkDoneStatus::Success, "Queue wait", "Queue work", message.c_str());
}

static bool ggml_backend_webgpu_map_buffer(webgpu_global_context & ctx,
                                           wgpu::Buffer & buffer,
                                           wgpu::MapMode mode,
                                           size_t offset,
                                           size_t size) {
    if (ctx->device_error.load(std::memory_order_acquire)) return false;
    auto result = std::make_shared<ai_chat_webgpu_wait_result>();
    const wgpu::WaitStatus wait_status = ctx->instance.WaitAny(
        buffer.MapAsync(mode, offset, size, wgpu::CallbackMode::AllowSpontaneous,
            [result](wgpu::MapAsyncStatus status, wgpu::StringView message) {
                {
                    std::lock_guard<std::mutex> lock(result->message_mutex);
                    result->message.assign(message.data ? message.data : "", message.length);
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
    const auto callback_status = status_value < 0
        ? wgpu::MapAsyncStatus::Error
        : static_cast<wgpu::MapAsyncStatus>(status_value);
    if (status_value < 0) {
        g_webgpu_device_error.store(true, std::memory_order_release);
        GGML_LOG_ERROR("AI_CHAT_WEBGPU_WAIT_FAILED kind=map reason=callback_not_delivered waitStatus=%d
",
            static_cast<int>(wait_status));
        return false;
    }
    return ggml_backend_webgpu_check_wait_status(wait_status, callback_status,
        wgpu::MapAsyncStatus::Success, "Buffer map wait", "Buffer map", message.c_str());
}
]=])
    set(_webgpu "${_webgpu_before_wait}${_wait_replacement}${_webgpu_after_wait}")

    # Queue submission must stop immediately after a failed wait; graph_compute
    # observes the shared error flag and returns GGML_STATUS_FAILED to the scheduler.
    ai_chat_replace_once(_webgpu
        "static void ggml_backend_webgpu_submit_commands(ggml_backend_webgpu_context &"
        "static void ggml_backend_webgpu_submit_commands(ggml_backend_webgpu_context &"
        "verify submit helper signature")
endif()

# The pinned source uses webgpu_context in the submit helper. Keep the replacement
# narrow and independently verifiable.
if (NOT _webgpu MATCHES "AI_CHAT_WEBGPU_SUBMIT_GUARD_V1")
    ai_chat_replace_once(_webgpu
        "    if (num_inflight_batches >= ctx->global_ctx->max_inflight_batches) {\n        ggml_backend_webgpu_wait_queue(ctx->global_ctx);\n        num_inflight_batches = 0;\n    }\n\n    ctx->global_ctx->queue.Submit(1, &commands);"
        "    if (ctx->global_ctx->device_error.load(std::memory_order_acquire)) return;\n    if (num_inflight_batches >= ctx->global_ctx->max_inflight_batches) {\n        if (!ggml_backend_webgpu_wait_queue(ctx->global_ctx)) return;\n        num_inflight_batches = 0;\n    }\n    if (ctx->global_ctx->device_error.load(std::memory_order_acquire)) return;\n\n    ctx->global_ctx->queue.Submit(1, &commands);"
        "stop command submission after a failed queue wait")
    string(APPEND _webgpu "\n// AI_CHAT_WEBGPU_SUBMIT_GUARD_V1\n")
endif()

# Device callbacks must never abort the process. A device-loss/error flag prevents
# subsequent placement on a backend that is no longer safe to execute.
if (NOT _webgpu MATCHES "AI_CHAT_WEBGPU_NONFATAL_CALLBACKS_V1")
    ai_chat_replace_once(_webgpu
        "            GGML_LOG_ERROR(\"ggml_webgpu: Device lost! Reason: %d, Message: %s\\n\", static_cast<int>(reason), std::string(message).c_str());"
        "            g_webgpu_device_error.store(true, std::memory_order_release);\n            GGML_LOG_ERROR(\"AI_CHAT_WEBGPU_DEVICE_LOST reason=%d message=%s\\n\", static_cast<int>(reason), std::string(message).c_str());"
        "persist device loss before later graph placement")
    ai_chat_replace_once(_webgpu
        "            GGML_ABORT(\"ggml_webgpu: Device error! Reason: %d, Message: %s\\n\", static_cast<int>(reason),\n                       std::string(message).c_str());"
        "            g_webgpu_device_error.store(true, std::memory_order_release);\n            GGML_LOG_ERROR(\"AI_CHAT_WEBGPU_UNCAPTURED_ERROR reason=%d message=%s\\n\", static_cast<int>(reason), std::string(message).c_str());"
        "replace uncaptured-error abort with explicit backend failure state")
    string(APPEND _webgpu "\n// AI_CHAT_WEBGPU_NONFATAL_CALLBACKS_V1\n")
endif()

# Do not continue encoding/return success after a wait or callback has failed.
if (NOT _webgpu MATCHES "AI_CHAT_WEBGPU_GRAPH_FAILURE_PROPAGATION_V1")
    ai_chat_replace_once(_webgpu
        "    while (node_idx < cgraph->n_nodes) {"
        "    while (node_idx < cgraph->n_nodes) {\n        if (ctx->global_ctx->device_error.load(std::memory_order_acquire)) {\n            GGML_LOG_ERROR(\"AI_CHAT_WEBGPU_GRAPH_ABORTED stage=encode node=%d totalNodes=%d\\n\", node_idx, cgraph->n_nodes);\n            WEBGPU_CPU_PROFILE_TOTAL_END(graph_compute, ctx->global_ctx);\n            return GGML_STATUS_FAILED;\n        }"
        "stop graph encoding when the device enters an error state")
    ai_chat_replace_once(_webgpu
        "    WEBGPU_CPU_PROFILE_TOTAL_END(graph_compute, ctx->global_ctx);\n    return GGML_STATUS_SUCCESS;"
        "    const bool ai_chat_graph_ok = !ctx->global_ctx->device_error.load(std::memory_order_acquire);\n    WEBGPU_CPU_PROFILE_TOTAL_END(graph_compute, ctx->global_ctx);\n    if (!ai_chat_graph_ok) {\n        GGML_LOG_ERROR(\"AI_CHAT_WEBGPU_GRAPH_ABORTED stage=completion nodes=%d\\n\", cgraph->n_nodes);\n        return GGML_STATUS_FAILED;\n    }\n    return GGML_STATUS_SUCCESS;"
        "propagate WebGPU wait/callback errors to GGML scheduler")
    string(APPEND _webgpu "\n// AI_CHAT_WEBGPU_GRAPH_FAILURE_PROPAGATION_V1\n")
endif()

# Avoid process abort for the explicit SET_ROWS overflow diagnostic as well.
if (NOT _webgpu MATCHES "AI_CHAT_WEBGPU_SET_ROWS_FAILURE_V1")
    ai_chat_replace_once(_webgpu
        "        GGML_ABORT(\"ggml_webgpu: SET_ROWS index > 2^32, unsupported.\");"
        "        g_webgpu_device_error.store(true, std::memory_order_release);\n        GGML_LOG_ERROR(\"AI_CHAT_WEBGPU_SET_ROWS_UNSUPPORTED indexWidth=64\\n\");"
        "return a failed graph status instead of aborting on unsupported index width")
    string(APPEND _webgpu "\n// AI_CHAT_WEBGPU_SET_ROWS_FAILURE_V1\n")
endif()

file(WRITE "${_webgpu_file}" "${_webgpu}")
message(STATUS "AI Chat: Dawn callback lifetime, non-fatal errors, and GGML failure propagation enabled")
