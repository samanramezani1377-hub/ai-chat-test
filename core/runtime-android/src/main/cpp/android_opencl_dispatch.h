#pragma once

// Android vendor OpenCL libraries commonly expose the complete OpenCL API
// directly but are not Khronos ICDs (they do not export
// clIcdGetPlatformIDsKHR). Calls from ggml-opencl must therefore be dispatched
// straight to the vendor implementation instead of through the bundled ICD
// loader. No vendor/GPU brand is assumed; the standard Android library paths
// are tried in order and the first library exporting the requested symbol wins.

#include <CL/cl.h>
#include <dlfcn.h>
#include <android/log.h>
#include <mutex>
#include <type_traits>
#include <utility>

namespace ai_chat_opencl_dispatch {
inline constexpr const char * kLogTag = "AIChatRuntime/OpenCL";

inline void * library_handle() {
    static void * handle = []() -> void * {
        const char * candidates[] = {
            "/vendor/lib64/libOpenCL.so",
            "/system/vendor/lib64/libOpenCL.so",
            "/vendor/lib/libOpenCL.so",
            "/system/lib64/libOpenCL.so",
            "libOpenCL.so",
        };
        for (const char * path : candidates) {
            (void) dlerror();
            void * candidate = dlopen(path, RTLD_NOW | RTLD_LOCAL);
            if (candidate) {
                __android_log_print(ANDROID_LOG_INFO, kLogTag,
                    "OPENCL_DIRECT_LIBRARY_SELECTED path=%s", path);
                return candidate;
            }
            const char * error = dlerror();
            __android_log_print(ANDROID_LOG_WARN, kLogTag,
                "OPENCL_DIRECT_LIBRARY_REJECTED path=%s error=%s",
                path, error ? error : "unknown");
        }
        __android_log_print(ANDROID_LOG_ERROR, kLogTag,
            "OPENCL_DIRECT_LIBRARY_UNAVAILABLE");
        return nullptr;
    }();
    return handle;
}

template <typename Fn>
inline Fn resolve(const char * symbol) {
    void * handle = library_handle();
    if (!handle) return nullptr;
    (void) dlerror();
    auto fn = reinterpret_cast<Fn>(dlsym(handle, symbol));
    const char * error = dlerror();
    if (!fn || error) {
        __android_log_print(ANDROID_LOG_ERROR, kLogTag,
            "OPENCL_DIRECT_SYMBOL_MISSING name=%s error=%s",
            symbol, error ? error : "symbol_not_found");
        return nullptr;
    }
    return fn;
}

template <typename Result>
inline Result missing_result() {
    if constexpr (std::is_pointer_v<Result>) {
        return nullptr;
    } else if constexpr (std::is_integral_v<Result>) {
        return static_cast<Result>(-1001); // CL_PLATFORM_NOT_FOUND_KHR / fail closed
    } else {
        return Result{};
    }
}

template <typename Fn, typename... Args>
inline auto call(const char * symbol, Args... args)
    -> decltype(std::declval<Fn>()(args...)) {
    using Result = decltype(std::declval<Fn>()(args...));
    const Fn fn = resolve<Fn>(symbol);
    if (!fn) {
        __android_log_print(ANDROID_LOG_ERROR, kLogTag,
            "OPENCL_DIRECT_CALL_BLOCKED name=%s", symbol);
        if constexpr (std::is_void_v<Result>) {
            return;
        } else {
            return missing_result<Result>();
        }
    }
    return fn(args...);
}
} // namespace ai_chat_opencl_dispatch

// Redirect every OpenCL API call in the Android runtime/backend translation
// units to the vendor implementation. These are function-like macros, so
// OpenCL declarations/types from cl.h remain untouched.
// Keep arguments in the original call expression. Some C OpenCL call sites use
// NULL as an integer null-pointer constant; capturing it in a template parameter
// turns it into an ordinary integer and breaks C++ overload conversion.
#define AI_CHAT_OPENCL_DIRECT_CALL(fn, ...) ([&]() { \
    auto ai_chat_opencl_fn = ai_chat_opencl_dispatch::resolve<decltype(&fn)>(#fn); \
    using AIChatOpenCLResult = decltype(ai_chat_opencl_fn(__VA_ARGS__)); \
    if (!ai_chat_opencl_fn) { \
        if constexpr (std::is_void_v<AIChatOpenCLResult>) return; \
        else return ai_chat_opencl_dispatch::missing_result<AIChatOpenCLResult>(); \
    } \
    return ai_chat_opencl_fn(__VA_ARGS__); \
}())
#define clBuildProgram(...) AI_CHAT_OPENCL_DIRECT_CALL(clBuildProgram, __VA_ARGS__)
#define clCreateBuffer(...) AI_CHAT_OPENCL_DIRECT_CALL(clCreateBuffer, __VA_ARGS__)
#define clCreateBufferWithProperties(...) AI_CHAT_OPENCL_DIRECT_CALL(clCreateBufferWithProperties, __VA_ARGS__)
#define clCreateCommandQueue(...) AI_CHAT_OPENCL_DIRECT_CALL(clCreateCommandQueue, __VA_ARGS__)
#define clCreateContext(...) AI_CHAT_OPENCL_DIRECT_CALL(clCreateContext, __VA_ARGS__)
#define clCreateImage(...) AI_CHAT_OPENCL_DIRECT_CALL(clCreateImage, __VA_ARGS__)
#define clCreateKernel(...) AI_CHAT_OPENCL_DIRECT_CALL(clCreateKernel, __VA_ARGS__)
#define clCreateProgramWithBinary(...) AI_CHAT_OPENCL_DIRECT_CALL(clCreateProgramWithBinary, __VA_ARGS__)
#define clCreateProgramWithSource(...) AI_CHAT_OPENCL_DIRECT_CALL(clCreateProgramWithSource, __VA_ARGS__)
#define clCreateSubBuffer(...) AI_CHAT_OPENCL_DIRECT_CALL(clCreateSubBuffer, __VA_ARGS__)
#define clEnqueueBarrierWithWaitList(...) AI_CHAT_OPENCL_DIRECT_CALL(clEnqueueBarrierWithWaitList, __VA_ARGS__)
#define clEnqueueCopyBuffer(...) AI_CHAT_OPENCL_DIRECT_CALL(clEnqueueCopyBuffer, __VA_ARGS__)
#define clEnqueueFillBuffer(...) AI_CHAT_OPENCL_DIRECT_CALL(clEnqueueFillBuffer, __VA_ARGS__)
#define clEnqueueMapBuffer(...) AI_CHAT_OPENCL_DIRECT_CALL(clEnqueueMapBuffer, __VA_ARGS__)
#define clEnqueueMarkerWithWaitList(...) AI_CHAT_OPENCL_DIRECT_CALL(clEnqueueMarkerWithWaitList, __VA_ARGS__)
#define clEnqueueNDRangeKernel(...) AI_CHAT_OPENCL_DIRECT_CALL(clEnqueueNDRangeKernel, __VA_ARGS__)
#define clEnqueueReadBuffer(...) AI_CHAT_OPENCL_DIRECT_CALL(clEnqueueReadBuffer, __VA_ARGS__)
#define clEnqueueUnmapMemObject(...) AI_CHAT_OPENCL_DIRECT_CALL(clEnqueueUnmapMemObject, __VA_ARGS__)
#define clEnqueueWriteBuffer(...) AI_CHAT_OPENCL_DIRECT_CALL(clEnqueueWriteBuffer, __VA_ARGS__)
#define clFinish(...) AI_CHAT_OPENCL_DIRECT_CALL(clFinish, __VA_ARGS__)
#define clFlush(...) AI_CHAT_OPENCL_DIRECT_CALL(clFlush, __VA_ARGS__)
#define clGetDeviceIDs(...) AI_CHAT_OPENCL_DIRECT_CALL(clGetDeviceIDs, __VA_ARGS__)
#define clGetDeviceInfo(...) AI_CHAT_OPENCL_DIRECT_CALL(clGetDeviceInfo, __VA_ARGS__)
#define clGetEventProfilingInfo(...) AI_CHAT_OPENCL_DIRECT_CALL(clGetEventProfilingInfo, __VA_ARGS__)
#define clGetKernelInfo(...) AI_CHAT_OPENCL_DIRECT_CALL(clGetKernelInfo, __VA_ARGS__)
#define clGetKernelSubGroupInfo(...) AI_CHAT_OPENCL_DIRECT_CALL(clGetKernelSubGroupInfo, __VA_ARGS__)
#define clGetKernelWorkGroupInfo(...) AI_CHAT_OPENCL_DIRECT_CALL(clGetKernelWorkGroupInfo, __VA_ARGS__)
#define clGetPlatformIDs(...) AI_CHAT_OPENCL_DIRECT_CALL(clGetPlatformIDs, __VA_ARGS__)
#define clGetPlatformInfo(...) AI_CHAT_OPENCL_DIRECT_CALL(clGetPlatformInfo, __VA_ARGS__)
#define clGetProgramBuildInfo(...) AI_CHAT_OPENCL_DIRECT_CALL(clGetProgramBuildInfo, __VA_ARGS__)
#define clGetProgramInfo(...) AI_CHAT_OPENCL_DIRECT_CALL(clGetProgramInfo, __VA_ARGS__)
#define clReleaseEvent(...) AI_CHAT_OPENCL_DIRECT_CALL(clReleaseEvent, __VA_ARGS__)
#define clReleaseKernel(...) AI_CHAT_OPENCL_DIRECT_CALL(clReleaseKernel, __VA_ARGS__)
#define clReleaseMemObject(...) AI_CHAT_OPENCL_DIRECT_CALL(clReleaseMemObject, __VA_ARGS__)
#define clReleaseProgram(...) AI_CHAT_OPENCL_DIRECT_CALL(clReleaseProgram, __VA_ARGS__)
#define clSetKernelArg(...) AI_CHAT_OPENCL_DIRECT_CALL(clSetKernelArg, __VA_ARGS__)
#define clWaitForEvents(...) AI_CHAT_OPENCL_DIRECT_CALL(clWaitForEvents, __VA_ARGS__)
