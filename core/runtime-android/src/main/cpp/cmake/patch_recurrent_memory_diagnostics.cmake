# Harden and instrument recurrent/hybrid state-cache allocation in the pinned llama.cpp build.
# This module is deliberately model-family agnostic: it preserves the upstream per-layer
# backend placement selected by offload_kqv and never forces recurrent models to CPU.
if (NOT DEFINED llama_cpp_SOURCE_DIR)
    message(FATAL_ERROR "llama_cpp_SOURCE_DIR is required")
endif()

set(_memory_file "${llama_cpp_SOURCE_DIR}/src/llama-memory-recurrent.cpp")
if (NOT EXISTS "${_memory_file}")
    message(FATAL_ERROR "AI Chat recurrent memory patch: pinned source is missing: ${_memory_file}")
endif()

file(READ "${_memory_file}" _memory)
if (NOT _memory MATCHES "AI_CHAT_RECURRENT_MEMORY_ALLOCATION_V2")
    # Timing support is used only for bounded constructor/allocation diagnostics.
    if (NOT _memory MATCHES "#include <chrono>")
        string(REPLACE "#include <cassert>" "#include <cassert>\n#include <chrono>" _memory "${_memory}")
    endif()

    ai_chat_replace_once(_memory
        "    const int32_t n_layer = hparams.n_layer();"
        "    const int32_t n_layer = hparams.n_layer();\n    const auto ai_chat_memory_started = std::chrono::steady_clock::now();\n    LLAMA_LOG_INFO(\"AI_CHAT_RECURRENT_MEMORY_BEGIN layers=%d cells=%u seqMax=%u rollbackSeq=%u offload=%d typeR=%s typeS=%s\\n\", n_layer, mem_size, n_seq_max, n_rs_seq, offload ? 1 : 0, ggml_type_name(type_r), ggml_type_name(type_s));"
        "record recurrent memory constructor inputs")

    ai_chat_replace_once(_memory
        "    std::map<ggml_backend_buffer_type_t, ggml_context_ptr, ggml_backend_buft_comparator> ctx_map;"
        "    std::map<ggml_backend_buffer_type_t, ggml_context_ptr, ggml_backend_buft_comparator> ctx_map;\n    std::map<ggml_backend_buffer_type_t, size_t, ggml_backend_buft_comparator> ai_chat_bytes_by_buft;\n    std::map<ggml_backend_buffer_type_t, size_t, ggml_backend_buft_comparator> ai_chat_tensors_by_buft;"
        "track expected cache bytes by actual buffer type")

    ai_chat_replace_once(_memory
        "            ggml_context * ctx = ggml_init(params);"
        "            LLAMA_LOG_INFO(\"AI_CHAT_RECURRENT_CONTEXT_CREATE_BEGIN buft=%s metadataBytes=%zu\\n\", ggml_backend_buft_name(buft), params.mem_size);\n            ggml_context * ctx = ggml_init(params);"
        "trace recurrent metadata context creation")

    ai_chat_replace_once(_memory
        "            ctx_map.emplace(buft, ctx);"
        "            LLAMA_LOG_INFO(\"AI_CHAT_RECURRENT_CONTEXT_CREATE_END buft=%s success=1\\n\", ggml_backend_buft_name(buft));\n            ctx_map.emplace(buft, ctx);"
        "trace recurrent metadata context success")

    ai_chat_replace_once(_memory
        "        LLAMA_LOG_DEBUG(\"%s, layer %3d: dev = %s\\n\", __func__, i, dev_name);"
        "        LLAMA_LOG_DEBUG(\"%s, layer %3d: dev = %s\\n\", __func__, i, dev_name);\n        LLAMA_LOG_INFO(\"AI_CHAT_RECURRENT_LAYER_PLACEMENT layer=%d device=%s buft=%s offload=%d\\n\", i, dev_name, ggml_backend_buft_name(buft), offload ? 1 : 0);"
        "trace the exact backend chosen for every recurrent layer")

    ai_chat_replace_once(_memory
        "        ggml_format_name(s, \"cache_s_l%d\", i);"
        "        ggml_format_name(s, \"cache_s_l%d\", i);\n        ai_chat_bytes_by_buft[buft] += ggml_nbytes(r) + ggml_nbytes(s);\n        ai_chat_tensors_by_buft[buft] += 2;"
        "account recurrent R and S tensor bytes before backend allocation")

    ai_chat_replace_once(_memory
        "            ggml_format_name(p, \"cache_ple_r_l%d\", i);"
        "            ggml_format_name(p, \"cache_ple_r_l%d\", i);\n            ai_chat_bytes_by_buft[buft] += ggml_nbytes(p);\n            ai_chat_tensors_by_buft[buft] += 1;"
        "account optional PLE recurrent tensor bytes")

    ai_chat_replace_once(_memory
        "    for (auto & [buft, ctx] : ctx_map) {\n        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx.get(), buft);"
        "    for (auto & [buft, ctx] : ctx_map) {\n        const size_t ai_chat_expected_bytes = ai_chat_bytes_by_buft[buft];\n        const size_t ai_chat_tensor_count = ai_chat_tensors_by_buft[buft];\n        const auto ai_chat_alloc_started = std::chrono::steady_clock::now();\n        LLAMA_LOG_INFO(\"AI_CHAT_RECURRENT_BUFFER_ALLOC_BEGIN buft=%s tensors=%zu expectedBytes=%zu\\n\", ggml_backend_buft_name(buft), ai_chat_tensor_count, ai_chat_expected_bytes);\n        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx.get(), buft);"
        "trace recurrent state buffer allocation before entering backend allocator")

    ai_chat_replace_once(_memory
        "        if (!buf) {\n            throw std::runtime_error(\"failed to allocate buffer for rs cache\");\n        }\n        ggml_backend_buffer_clear(buf, 0);"
        "        if (!buf) {\n            LLAMA_LOG_ERROR(\"AI_CHAT_RECURRENT_BUFFER_ALLOC_FAILED buft=%s tensors=%zu expectedBytes=%zu elapsedMs=%.3f\\n\", ggml_backend_buft_name(buft), ai_chat_tensor_count, ai_chat_expected_bytes, std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - ai_chat_alloc_started).count());\n            throw std::runtime_error(\"failed to allocate buffer for rs cache\");\n        }\n        LLAMA_LOG_INFO(\"AI_CHAT_RECURRENT_BUFFER_ALLOCATED buft=%s actualBytes=%zu expectedBytes=%zu tensors=%zu elapsedMs=%.3f\\n\", ggml_backend_buft_name(buft), ggml_backend_buffer_get_size(buf), ai_chat_expected_bytes, ai_chat_tensor_count, std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - ai_chat_alloc_started).count());\n        LLAMA_LOG_INFO(\"AI_CHAT_RECURRENT_BUFFER_CLEAR_BEGIN buft=%s bytes=%zu\\n\", ggml_backend_buft_name(buft), ggml_backend_buffer_get_size(buf));\n        ggml_backend_buffer_clear(buf, 0);\n        LLAMA_LOG_INFO(\"AI_CHAT_RECURRENT_BUFFER_CLEAR_END buft=%s success=1\\n\", ggml_backend_buft_name(buft));"
        "report allocation failure and isolate buffer clear")

    ai_chat_replace_once(_memory
        "    }\n\n    if (is_empty()) {"
        "    }\n    LLAMA_LOG_INFO(\"AI_CHAT_RECURRENT_MEMORY_READY buffers=%zu elapsedMs=%.3f\\n\", ctxs_bufs.size(), std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - ai_chat_memory_started).count());\n\n    if (is_empty()) {"
        "record successful recurrent memory initialization")

    string(APPEND _memory "\n// AI_CHAT_RECURRENT_MEMORY_ALLOCATION_V2\n")
    file(WRITE "${_memory_file}" "${_memory}")
endif()

message(STATUS "AI Chat: recurrent/hybrid state-cache allocation tracing enabled")
