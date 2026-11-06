# Adds app-visible OpenCL event profiling to the pinned llama.cpp source.
# Applied only to the profiling/debug native variant.
if (NOT DEFINED llama_cpp_SOURCE_DIR)
    message(FATAL_ERROR "llama_cpp_SOURCE_DIR is required")
endif()

set(_src_file "${llama_cpp_SOURCE_DIR}/ggml/src/ggml-opencl/ggml-opencl.cpp")
if (NOT EXISTS "${_src_file}")
    message(FATAL_ERROR "OpenCL source not found: ${_src_file}")
endif()

file(READ "${_src_file}" _src)

if (_src MATCHES "AI_CHAT_OPENCL_PROFILE_PATCH_V4" AND _src MATCHES "fperf_live" AND _src MATCHES "profiling_info.size\\(\\) >= 256")
    return()
endif()

# V3 may already be present in the persistent FetchContent tree. Upgrade it
# in-place instead of silently skipping the patch.
if (_src MATCHES "AI_CHAT_OPENCL_PROFILE_PATCH_V3")
    string(REPLACE
        "AI_CHAT_OPENCL_PROFILE_PATCH_V3"
        "AI_CHAT_OPENCL_PROFILE_PATCH_V4"
        _src "${_src}")
    string(REPLACE
        "if (profiling_info.size() >= 256) {"
        "if (profiling_info.size() >= 256) {"
        _src "${_src}")
    string(REPLACE
        "if (profiling_info.size() >= 2048) {"
        "if (profiling_info.size() >= 256) {"
        _src "${_src}")
else()
    string(REPLACE
        "#include \"cl-program-cache.h\""
        "#include \"cl-program-cache.h\"\n#include <cstdlib>\n\n// AI_CHAT_OPENCL_PROFILE_PATCH_V4"
        _src "${_src}")

    # Use the Android runtime temp directory when available; otherwise use the
    # app cache path and finally the current directory.
    string(REPLACE
        "FILE * fperf = fopen(\"cl_profiling.csv\", \"w\");"
        "const char * ai_chat_tmpdir = std::getenv(\"TMPDIR\");\n    const std::string ai_chat_profile_path =\n        ai_chat_tmpdir && *ai_chat_tmpdir\n            ? std::string(ai_chat_tmpdir) + \"/cl_profiling.csv\"\n            : std::string(\"/data/user/0/com.samanramezani.aichattest/cache/cl_profiling.csv\");\n    FILE * fperf = fopen(ai_chat_profile_path.c_str(), \"w\");\n    if (!fperf) { fperf = fopen(\"/data/data/com.samanramezani.aichattest/cache/cl_profiling.csv\", \"w\"); }\n    if (!fperf) { fperf = fopen(\"cl_profiling.csv\", \"w\"); }"
        _src "${_src}")

    string(REPLACE
        "FILE * ftrace = fopen(\"cl_trace.json\", \"w\");"
        "const std::string ai_chat_trace_path =\n        ai_chat_tmpdir && *ai_chat_tmpdir\n            ? std::string(ai_chat_tmpdir) + \"/cl_trace.json\"\n            : std::string(\"/data/user/0/com.samanramezani.aichattest/cache/cl_trace.json\");\n    FILE * ftrace = fopen(ai_chat_trace_path.c_str(), \"w\");\n    if (!ftrace) { ftrace = fopen(\"/data/data/com.samanramezani.aichattest/cache/cl_trace.json\", \"w\"); }\n    if (!ftrace) { ftrace = fopen(\"cl_trace.json\", \"w\"); }"
        _src "${_src}")

    string(REPLACE
        "fprintf(fperf, \"op name, kernel name, exec duration (ms), global size, local size, output size\\n\");"
        "fprintf(fperf, \"op name, kernel name, queue (ms), submit (ms), exec duration (ms), complete (ms), total (ms), global size, local size, output size\\n\");"
        _src "${_src}")

    string(REPLACE
        "fprintf(fperf, \"%s,%s,%f,%zux%zux%zu,%zux%zux%zu,%zux%zux%zux%zu\\n\",\n            info.op_name.c_str(), info.kernel_name.c_str(),\n            info.cmd_duration_ns/1.e6f,"
        "fprintf(fperf, \"%s,%s,%f,%f,%f,%f,%f,%zux%zux%zu,%zux%zux%zu,%zux%zux%zux%zu\\n\",\n            info.op_name.c_str(), info.kernel_name.c_str(),\n            info.cmd_queued_duration_ns/1.e6f,\n            info.cmd_submit_duration_ns/1.e6f,\n            info.cmd_duration_ns/1.e6f,\n            info.cmd_complete_duration_ns/1.e6f,\n            info.cmd_total_duration_ns/1.e6f,"
        _src "${_src}")

    string(REPLACE
        "profiling_results.insert(profiling_results.end(),\n            std::make_move_iterator(profiling_info.begin()),\n            std::make_move_iterator(profiling_info.end()));\n        profiling_info.clear();"
        "profiling_results.insert(profiling_results.end(),\n            std::make_move_iterator(profiling_info.begin()),\n            std::make_move_iterator(profiling_info.end()));\n        profiling_info.clear();\n\n        // Android keeps the OpenCL context alive across generations, so the\n        // upstream destructor-only CSV would otherwise remain unavailable.\n        {\n            const char * ai_chat_tmpdir = std::getenv(\"TMPDIR\");\n            const std::string ai_chat_profile_path =\n                ai_chat_tmpdir && *ai_chat_tmpdir\n                    ? std::string(ai_chat_tmpdir) + \"/cl_profiling.csv\"\n                    : std::string(\"/data/user/0/com.samanramezani.aichattest/cache/cl_profiling.csv\");\n            FILE * fperf_live = fopen(ai_chat_profile_path.c_str(), \"w\");\n            if (!fperf_live) { fperf_live = fopen(\"/data/data/com.samanramezani.aichattest/cache/cl_profiling.csv\", \"w\"); }\n            if (!fperf_live) { fperf_live = fopen(\"cl_profiling.csv\", \"w\"); }\n            if (fperf_live) {\n                fprintf(fperf_live, \"op name, kernel name, queue (ms), submit (ms), exec duration (ms), complete (ms), total (ms), global size, local size, output size\\n\");\n                for (const ProfilingInfo & info : profiling_results) {\n                    fprintf(fperf_live, \"%s,%s,%f,%f,%f,%f,%f,%zux%zux%zu,%zux%zux%zu,%zux%zux%zux%zu\\n\",\n                        info.op_name.c_str(), info.kernel_name.c_str(),\n                        info.cmd_queued_duration_ns/1.e6f,\n                        info.cmd_submit_duration_ns/1.e6f,\n                        info.cmd_duration_ns/1.e6f,\n                        info.cmd_complete_duration_ns/1.e6f,\n                        info.cmd_total_duration_ns/1.e6f,\n                        info.global_size[0], info.global_size[1], info.global_size[2],\n                        info.local_size[0], info.local_size[1], info.local_size[2],\n                        info.output_size[0], info.output_size[1], info.output_size[2], info.output_size[3]);\n                }\n                fclose(fperf_live);\n            }\n        }"
        _src "${_src}")

    string(REPLACE
        "if (profiling_info.size() >= 2048) {"
        "if (profiling_info.size() >= 256) {"
        _src "${_src}")
endif()



# V4 normalization also upgrades a pre-existing V3 patch.
if (NOT _src MATCHES "AI_CHAT_OPENCL_PROFILE_PATCH_V4")
    string(REPLACE
        "#include \"cl-program-cache.h\""
        "#include \"cl-program-cache.h\"\\n#include <cstdlib>\\n\\n// AI_CHAT_OPENCL_PROFILE_PATCH_V4"
        _src "${_src}")
endif()

# Pre-existing V3 trees may have the profiling metadata but no live writer.
# Insert the writer immediately after the unique profiling_info.clear() in
# flush_profiling_batch(). Using position-based insertion avoids depending on
# whitespace or upstream formatting.
if (NOT _src MATCHES "fperf_live")
    string(FIND "${_src}" "profiling_info.clear();" _clear_pos)
    if (_clear_pos LESS 0)
        message(FATAL_ERROR "AI Chat OpenCL profiling: profiling_info.clear() anchor not found")
    endif()
    string(LENGTH "profiling_info.clear();" _clear_len)
    math(EXPR _after_clear "${_clear_pos} + ${_clear_len}")
    string(SUBSTRING "${_src}" 0 ${_after_clear} _before_clear)
    string(LENGTH "${_src}" _src_len)
    math(EXPR _remaining_len "${_src_len} - ${_after_clear}")
    string(SUBSTRING "${_src}" ${_after_clear} ${_remaining_len} _after_clear_text)
    set(_live_writer [=[
        // AI_CHAT_OPENCL_PROFILE_PATCH_V4_LIVE_WRITER
        // Android keeps the OpenCL context alive across generations, so make
        // the accumulated event data visible before the backend is destroyed.
        {
            FILE * fperf_live = fopen("/data/user/0/com.samanramezi.aichattest/cache/cl_profiling.csv", "w");
            if (!fperf_live) { fperf_live = fopen("/data/data/com.samanramezi.aichattest/cache/cl_profiling.csv", "w"); }
            if (!fperf_live) { fperf_live = fopen("cl_profiling.csv", "w"); }
            if (fperf_live) {
                fprintf(fperf_live, "op name, kernel name, queue (ms), submit (ms), exec duration (ms), complete (ms), total (ms), global size, local size, output size\\n");
                for (const ProfilingInfo & info : profiling_results) {
                    fprintf(fperf_live, "%s,%s,%f,%f,%f,%f,%f,%zux%zux%zu,%zux%zux%zu,%zux%zux%zux%zu\\n",
                        info.op_name.c_str(), info.kernel_name.c_str(),
                        info.cmd_queued_duration_ns/1.e6f,
                        info.cmd_submit_duration_ns/1.e6f,
                        info.cmd_duration_ns/1.e6f,
                        info.cmd_complete_duration_ns/1.e6f,
                        info.cmd_total_duration_ns/1.e6f,
                        info.global_size[0], info.global_size[1], info.global_size[2],
                        info.local_size[0], info.local_size[1], info.local_size[2],
                        info.output_size[0], info.output_size[1], info.output_size[2], info.output_size[3]);
                }
                fclose(fperf_live);
            }
        }
]=])
    set(_src "${_before_clear}${_live_writer}${_after_clear_text}")
endif()

# Normalize any existing profiling batch threshold. Upstream revisions can use
# different formatting/thresholds; the runtime only needs a bounded live flush.
string(REGEX REPLACE
    "profiling_info\\.size\\(\\) *>= *[0-9]+"
    "profiling_info.size() >= 256"
    _src "${_src}")

# Normalize the CSV header for pre-existing V3 trees.
string(REPLACE
    "fprintf(fperf, \"op name, kernel name, exec duration (ms), global size, local size, output size\\n\");"
    "fprintf(fperf, \"op name, kernel name, queue (ms), submit (ms), exec duration (ms), complete (ms), total (ms), global size, local size, output size\\n\");"
    _src "${_src}")

# Do not allow a silent no-op patch. These checks make the Android build fail
# instead of shipping a debug APK that reports N/A.
if (NOT _src MATCHES "AI_CHAT_OPENCL_PROFILE_PATCH_V4")
    message(FATAL_ERROR "AI Chat OpenCL profiling V4 marker was not installed")
endif()
if (NOT _src MATCHES "fperf_live")
    message(FATAL_ERROR "AI Chat OpenCL live profiling writer was not installed")
endif()
if (NOT _src MATCHES "profiling_info\\.size\\(\\)")
    message(FATAL_ERROR "AI Chat OpenCL profiling batch control was not found")
endif()

file(WRITE "${_src_file}" "${_src}")
message(STATUS "AI Chat: patched llama.cpp OpenCL profiling for Android (V4, live)")
