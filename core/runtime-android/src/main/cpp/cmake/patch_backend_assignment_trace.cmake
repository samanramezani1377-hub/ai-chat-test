# Emit one bounded graph assignment trace so diagnostics can distinguish CPU and GPU ops.
if (NOT DEFINED llama_cpp_SOURCE_DIR)
    message(FATAL_ERROR "llama_cpp_SOURCE_DIR is required")
endif()

set(_sched_file "${llama_cpp_SOURCE_DIR}/ggml/src/ggml-backend.cpp")
if (NOT EXISTS "${_sched_file}")
    message(FATAL_ERROR "ggml backend scheduler source not found: ${_sched_file}")
endif()

file(READ "${_sched_file}" _sched)
if (_sched MATCHES "AI_CHAT_BACKEND_ASSIGNMENT_TRACE_V2")
    return()
endif()

set(_print_anchor "static void ggml_backend_sched_print_assignments(ggml_backend_sched_t sched, struct ggml_cgraph * graph) {")
string(FIND "${_sched}" "${_print_anchor}" _print_pos)
if (_print_pos LESS 0)
    message(FATAL_ERROR "AI Chat backend trace: scheduler print function anchor not found")
endif()

# Upgrade an already-patched V1 source tree in-place; clean trees get the complete patch.
if (_sched MATCHES "AI_CHAT_BACKEND_ASSIGNMENT_TRACE_V1")
    string(REPLACE "AI_CHAT_BACKEND_ASSIGNMENT_TRACE_V1" "AI_CHAT_BACKEND_ASSIGNMENT_TRACE_V2" _sched "${_sched}")
else()
    string(REPLACE "${_print_anchor}"
        "static bool ai_chat_backend_assignment_trace_emitted = false;\n// AI_CHAT_BACKEND_ASSIGNMENT_TRACE_V2\n${_print_anchor}"
        _sched "${_sched}")
endif()

set(_dispatch_anchor "    if (sched->debug) {\n        ggml_backend_sched_print_assignments(sched, graph);\n    }")
if (_sched MATCHES "if \\(sched->debug\\) \\{")
    string(REPLACE "${_dispatch_anchor}"
        "    if (sched->debug || (getenv(\"AI_CHAT_BACKEND_TRACE\") != nullptr && !ai_chat_backend_assignment_trace_emitted)) {\n        if (!sched->debug) ai_chat_backend_assignment_trace_emitted = true;\n        ggml_backend_sched_print_assignments(sched, graph);\n    }"
        _sched "${_sched}")
endif()

# Promote only the bounded assignment report to INFO and include per-node backend labels.
string(FIND "${_sched}" "${_print_anchor}" _print_start)
string(FIND "${_sched}" "static bool ggml_backend_sched_buffer_supported" _print_end)
if (_print_start LESS 0 OR _print_end LESS 0 OR _print_end LESS _print_start)
    message(FATAL_ERROR "AI Chat backend trace: assignment report boundaries not found")
endif()
math(EXPR _print_len "${_print_end} - ${_print_start}")
string(SUBSTRING "${_sched}" ${_print_start} ${_print_len} _print_block)
string(REPLACE "GGML_LOG_DEBUG(" "GGML_LOG_INFO(" _print_block "${_print_block}")
string(REPLACE "if (sched->debug > 1) {" "if (sched->debug > 1 || getenv(\"AI_CHAT_BACKEND_TRACE\") != nullptr) {" _print_block "${_print_block}")
string(SUBSTRING "${_sched}" 0 ${_print_start} _before_print)
string(LENGTH "${_sched}" _sched_len)
math(EXPR _after_pos "${_print_start} + ${_print_len}")
math(EXPR _after_len "${_sched_len} - ${_after_pos}")
string(SUBSTRING "${_sched}" ${_after_pos} ${_after_len} _after_print)
set(_sched "${_before_print}${_print_block}${_after_print}")

file(WRITE "${_sched_file}" "${_sched}")
message(STATUS "AI Chat: enabled one-shot CPU/GPU graph assignment diagnostics")
