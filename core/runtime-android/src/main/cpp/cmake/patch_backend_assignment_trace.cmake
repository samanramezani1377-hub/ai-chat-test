# Emit one bounded graph assignment trace plus sampled per-backend split timings.
if (NOT DEFINED llama_cpp_SOURCE_DIR)
    message(FATAL_ERROR "llama_cpp_SOURCE_DIR is required")
endif()

set(_sched_file "${llama_cpp_SOURCE_DIR}/ggml/src/ggml-backend.cpp")
if (NOT EXISTS "${_sched_file}")
    message(FATAL_ERROR "ggml backend scheduler source not found: ${_sched_file}")
endif()

file(READ "${_sched_file}" _sched)
if (_sched MATCHES "AI_CHAT_BACKEND_ASSIGNMENT_TRACE_V3")
    return()
endif()

if (NOT _sched MATCHES "#include <chrono>")
    string(REPLACE "#include <algorithm>" "#include <algorithm>\n#include <chrono>" _sched "${_sched}")
endif()

set(_print_anchor "static void ggml_backend_sched_print_assignments(ggml_backend_sched_t sched, struct ggml_cgraph * graph) {")
string(FIND "${_sched}" "${_print_anchor}" _print_pos)
if (_print_pos LESS 0)
    message(FATAL_ERROR "AI Chat backend trace: scheduler print function anchor not found")
endif()

# Upgrade cached V1/V2 source trees in-place; clean trees get the complete patch.
if (_sched MATCHES "AI_CHAT_BACKEND_ASSIGNMENT_TRACE_V[12]")
    string(REGEX REPLACE "AI_CHAT_BACKEND_ASSIGNMENT_TRACE_V[12]" "AI_CHAT_BACKEND_ASSIGNMENT_TRACE_V3" _sched "${_sched}")
else()
    string(REPLACE "${_print_anchor}"
        "static bool ai_chat_backend_assignment_trace_emitted = false;\n// AI_CHAT_BACKEND_ASSIGNMENT_TRACE_V3\n${_print_anchor}"
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

# Sample one in every eight CPU/OpenCL graph splits per backend. Assignment logs above
# identify the operators; this records split wall time without logging every token.
if (NOT _sched MATCHES "AI_CHAT_BACKEND_SPLIT_TIMING")
    set(_compute_anchor "            enum ggml_status ec = ggml_backend_graph_compute_async(split_backend, &split->graph);\n            if (ec != GGML_STATUS_SUCCESS) {")
    string(FIND "${_sched}" "${_compute_anchor}" _compute_pos)
    if (_compute_pos LESS 0)
        message(FATAL_ERROR "AI Chat backend trace: split compute anchor not found")
    endif()
    set(_compute_replacement [=[
            static std::unordered_map<std::string, uint64_t> ai_chat_backend_split_samples;
            const std::string ai_chat_backend_name(ggml_backend_name(split_backend));
            const bool ai_chat_sample_split = getenv("AI_CHAT_BACKEND_TRACE") != nullptr &&
                (++ai_chat_backend_split_samples[ai_chat_backend_name] % 8 == 1);
            const auto ai_chat_split_started = ai_chat_sample_split
                ? std::chrono::steady_clock::now()
                : std::chrono::steady_clock::time_point{};
            enum ggml_status ec = ggml_backend_graph_compute_async(split_backend, &split->graph);
            if (ai_chat_sample_split) {
                const auto ai_chat_split_finished = std::chrono::steady_clock::now();
                const struct ggml_tensor * ai_chat_first = split->graph.n_nodes > 0 ? split->graph.nodes[0] : nullptr;
                const struct ggml_tensor * ai_chat_last = split->graph.n_nodes > 0 ? split->graph.nodes[split->graph.n_nodes - 1] : nullptr;
                const double ai_chat_elapsed_ms = std::chrono::duration<double, std::milli>(
                    ai_chat_split_finished - ai_chat_split_started).count();
                GGML_LOG_INFO("AI_CHAT_BACKEND_SPLIT_TIMING backend=%s nodes=%d firstOp=%s firstName=%s lastOp=%s lastName=%s elapsedMs=%.3f sampleEvery=8",
                    ai_chat_backend_name.c_str(), split->graph.n_nodes,
                    ai_chat_first ? ggml_op_name(ai_chat_first->op) : "none",
                    ai_chat_first ? ai_chat_first->name : "none",
                    ai_chat_last ? ggml_op_name(ai_chat_last->op) : "none",
                    ai_chat_last ? ai_chat_last->name : "none", ai_chat_elapsed_ms);
            }
            if (ec != GGML_STATUS_SUCCESS) {
]=])
    string(REPLACE "${_compute_anchor}" "${_compute_replacement}" _sched "${_sched}")
endif()

file(WRITE "${_sched_file}" "${_sched}")
message(STATUS "AI Chat: enabled one-shot CPU/GPU assignments and sampled backend split timings")
