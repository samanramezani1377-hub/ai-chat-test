# Adds app-visible OpenCL event profiling to the pinned llama.cpp source.
# This is intentionally applied only to the profiling/debug native variant.
if (NOT DEFINED llama_cpp_SOURCE_DIR)
    message(FATAL_ERROR "llama_cpp_SOURCE_DIR is required")
endif()

set(_src_file "${llama_cpp_SOURCE_DIR}/ggml/src/ggml-opencl/ggml-opencl.cpp")
if (NOT EXISTS "${_src_file}")
    message(FATAL_ERROR "OpenCL source not found: ${_src_file}")
endif()

file(READ "${_src_file}" _src)
if (_src MATCHES "AI_CHAT_OPENCL_PROFILE_PATCH_V1")
    return()
endif()

string(REPLACE
    "#include \"cl-program-cache.h\""
    "#include \"cl-program-cache.h\"\n#include <cstdlib>\n\n// AI_CHAT_OPENCL_PROFILE_PATCH_V1\n#ifdef GGML_OPENCL_PROFILING\nstatic cl_ulong ai_chat_transfer_exec_ns = 0;\nstatic size_t ai_chat_transfer_count = 0;\n\nstatic void ai_chat_profile_transfer_event(cl_event evt) {\n    if (!evt) return;\n    CL_CHECK(clWaitForEvents(1, &evt));\n    cl_ulong start = 0, end = 0;\n    CL_CHECK(clGetEventProfilingInfo(evt, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL));\n    CL_CHECK(clGetEventProfilingInfo(evt, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL));\n    ai_chat_transfer_exec_ns += end - start;\n    ++ai_chat_transfer_count;\n}\n\nstatic cl_int ai_chat_profile_write_buffer(cl_command_queue q, cl_mem b, cl_bool blocking, size_t off, size_t size, const void * ptr, cl_uint n, const cl_event * wl, cl_event * ev) {\n    cl_event local = nullptr;\n    cl_event * out = ev ? ev : &local;\n    cl_int rc = ::clEnqueueWriteBuffer(q, b, blocking, off, size, ptr, n, wl, out);\n    if (rc == CL_SUCCESS && *out) {\n        ai_chat_profile_transfer_event(*out);\n        if (!ev) CL_CHECK(clReleaseEvent(*out));\n    }\n    return rc;\n}\n\nstatic cl_int ai_chat_profile_read_buffer(cl_command_queue q, cl_mem b, cl_bool blocking, size_t off, size_t size, void * ptr, cl_uint n, const cl_event * wl, cl_event * ev) {\n    cl_event local = nullptr;\n    cl_event * out = ev ? ev : &local;\n    cl_int rc = ::clEnqueueReadBuffer(q, b, blocking, off, size, ptr, n, wl, out);\n    if (rc == CL_SUCCESS && *out) {\n        ai_chat_profile_transfer_event(*out);\n        if (!ev) CL_CHECK(clReleaseEvent(*out));\n    }\n    return rc;\n}\n\n#define clEnqueueWriteBuffer ai_chat_profile_write_buffer\n#define clEnqueueReadBuffer ai_chat_profile_read_buffer\n#endif"
    _src "${_src}")

string(REPLACE
    "void flush_profiling_batch() {"
    "void write_profiling_info();\n\nvoid flush_profiling_batch() {"
    _src "${_src}")

string(REPLACE
    "    profiling_info.clear();"
    "    profiling_info.clear();\n    // Persist accumulated results after every 2048-event batch so the Android UI\n    // can inspect the profile immediately after a generation without unloading.\n    write_profiling_info();"
    _src "${_src}")

string(REPLACE
    "FILE * fperf = fopen(\"cl_profiling.csv\", \"w\");"
    "const char * ai_chat_tmpdir = std::getenv(\"TMPDIR\");\n    const std::string ai_chat_profile_path =\n        ai_chat_tmpdir && *ai_chat_tmpdir\n            ? std::string(ai_chat_tmpdir) + \"/cl_profiling.csv\"\n            : std::string(\"cl_profiling.csv\");\n    FILE * fperf = fopen(ai_chat_profile_path.c_str(), \"w\");"
    _src "${_src}")

string(REPLACE
    "FILE * ftrace = fopen(\"cl_trace.json\", \"w\");"
    "const std::string ai_chat_trace_path =\n        ai_chat_tmpdir && *ai_chat_tmpdir\n            ? std::string(ai_chat_tmpdir) + \"/cl_trace.json\"\n            : std::string(\"cl_trace.json\");\n    FILE * ftrace = fopen(ai_chat_trace_path.c_str(), \"w\");"
    _src "${_src}")

string(REPLACE
    "fprintf(fperf, \"op name, kernel name, exec duration (ms), global size, local size, output size\\n\");"
    "fprintf(fperf, \"op name, kernel name, queue (ms), submit (ms), exec duration (ms), complete (ms), total (ms), global size, local size, output size\\n\");"
    _src "${_src}")

string(REPLACE
    "fprintf(fperf, \"%s,%s,%f,%zux%zux%zu,%zux%zux%zu,%zux%zux%zux%zu\\n\",\n            info.op_name.c_str(), info.kernel_name.c_str(),\n            info.cmd_duration_ns/1.e6f,"
    "fprintf(fperf, \"%s,%s,%f,%f,%f,%f,%f,%zux%zux%zu,%zux%zux%zu,%zux%zux%zux%zu\\n\",\n            info.op_name.c_str(), info.kernel_name.c_str(),\n            info.cmd_queued_duration_ns/1.e6f,\n            info.cmd_submit_duration_ns/1.e6f,\n            info.cmd_duration_ns/1.e6f,\n            info.cmd_complete_duration_ns/1.e6f,\n            info.cmd_total_duration_ns/1.e6f,"
    _src "${_src}")

file(WRITE "${_src_file}" "${_src}")
message(STATUS "AI Chat: patched llama.cpp OpenCL profiling for Android")

string(REPLACE
    "    fclose(ftrace);"
    "    fclose(ftrace);\n\n    const std::string ai_chat_transfer_path =\n        ai_chat_tmpdir && *ai_chat_tmpdir\n            ? std::string(ai_chat_tmpdir) + \"/cl_transfer_profile.txt\"\n            : std::string(\"cl_transfer_profile.txt\");\n    FILE * ftransfer = fopen(ai_chat_transfer_path.c_str(), \"w\");\n    if (ftransfer) {\n        fprintf(ftransfer, \"memory_transfer_ms=%f count=%zu\\n\", ai_chat_transfer_exec_ns/1.e6f, ai_chat_transfer_count);\n        fclose(ftransfer);\n    }"
    _src "${_src}")
