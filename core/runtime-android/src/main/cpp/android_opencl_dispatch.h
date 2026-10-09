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
#define clBuildProgram(...) ai_chat_opencl_dispatch::call<decltype(&clBuildProgram)>("clBuildProgram", __VA_ARGS__)
#define clCreateBuffer(...) ai_chat_opencl_dispatch::call<decltype(&clCreateBuffer)>("clCreateBuffer", __VA_ARGS__)
#define clCreateBufferWithProperties(...) ai_chat_opencl_dispatch::call<decltype(&clCreateBufferWithProperties)>("clCreateBufferWithProperties", __VA_ARGS__)
#define clCreateCommandQueue(...) ai_chat_opencl_dispatch::call<decltype(&clCreateCommandQueue)>("clCreateCommandQueue", __VA_ARGS__)
#define clCreateContext(...) ai_chat_opencl_dispatch::call<decltype(&clCreateContext)>("clCreateContext", __VA_ARGS__)
#define clCreateImage(...) ai_chat_opencl_dispatch::call<decltype(&clCreateImage)>("clCreateImage", __VA_ARGS__)
#define clCreateKernel(...) ai_chat_opencl_dispatch::call<decltype(&clCreateKernel)>("clCreateKernel", __VA_ARGS__)
#define clCreateProgramWithBinary(...) ai_chat_opencl_dispatch::call<decltype(&clCreateProgramWithBinary)>("clCreateProgramWithBinary", __VA_ARGS__)
#define clCreateProgramWithSource(...) ai_chat_opencl_dispatch::call<decltype(&clCreateProgramWithSource)>("clCreateProgramWithSource", __VA_ARGS__)
#define clCreateSubBuffer(...) ai_chat_opencl_dispatch::call<decltype(&clCreateSubBuffer)>("clCreateSubBuffer", __VA_ARGS__)
#define clEnqueueBarrierWithWaitList(...) ai_chat_opencl_dispatch::call<decltype(&clEnqueueBarrierWithWaitList)>("clEnqueueBarrierWithWaitList", __VA_ARGS__)
#define clEnqueueCopyBuffer(...) ai_chat_opencl_dispatch::call<decltype(&clEnqueueCopyBuffer)>("clEnqueueCopyBuffer", __VA_ARGS__)
#define clEnqueueFillBuffer(...) ai_chat_opencl_dispatch::call<decltype(&clEnqueueFillBuffer)>("clEnqueueFillBuffer", __VA_ARGS__)
#define clEnqueueMapBuffer(...) ai_chat_opencl_dispatch::call<decltype(&clEnqueueMapBuffer)>("clEnqueueMapBuffer", __VA_ARGS__)
#define clEnqueueMarkerWithWaitList(...) ai_chat_opencl_dispatch::call<decltype(&clEnqueueMarkerWithWaitList)>("clEnqueueMarkerWithWaitList", __VA_ARGS__)
#define clEnqueueNDRangeKernel(...) ai_chat_opencl_dispatch::call<decltype(&clEnqueueNDRangeKernel)>("clEnqueueNDRangeKernel", __VA_ARGS__)
#define clEnqueueReadBuffer(...) ai_chat_opencl_dispatch::call<decltype(&clEnqueueReadBuffer)>("clEnqueueReadBuffer", __VA_ARGS__)
#define clEnqueueUnmapMemObject(...) ai_chat_opencl_dispatch::call<decltype(&clEnqueueUnmapMemObject)>("clEnqueueUnmapMemObject", __VA_ARGS__)
#define clEnqueueWriteBuffer(...) ai_chat_opencl_dispatch::call<decltype(&clEnqueueWriteBuffer)>("clEnqueueWriteBuffer", __VA_ARGS__)
#define clFinish(...) ai_chat_opencl_dispatch::call<decltype(&clFinish)>("clFinish", __VA_ARGS__)
#define clFlush(...) ai_chat_opencl_dispatch::call<decltype(&clFlush)>("clFlush", __VA_ARGS__)
#define clGetDeviceIDs(...) ai_chat_opencl_dispatch::call<decltype(&clGetDeviceIDs)>("clGetDeviceIDs", __VA_ARGS__)
#define clGetDeviceInfo(...) ai_chat_opencl_dispatch::call<decltype(&clGetDeviceInfo)>("clGetDeviceInfo", __VA_ARGS__)
#define clGetEventProfilingInfo(...) ai_chat_opencl_dispatch::call<decltype(&clGetEventProfilingInfo)>("clGetEventProfilingInfo", __VA_ARGS__)
#define clGetKernelInfo(...) ai_chat_opencl_dispatch::call<decltype(&clGetKernelInfo)>("clGetKernelInfo", __VA_ARGS__)
#define clGetKernelSubGroupInfo(...) ai_chat_opencl_dispatch::call<decltype(&clGetKernelSubGroupInfo)>("clGetKernelSubGroupInfo", __VA_ARGS__)
#define clGetKernelWorkGroupInfo(...) ai_chat_opencl_dispatch::call<decltype(&clGetKernelWorkGroupInfo)>("clGetKernelWorkGroupInfo", __VA_ARGS__)
#define clGetPlatformIDs(...) ai_chat_opencl_dispatch::call<decltype(&clGetPlatformIDs)>("clGetPlatformIDs", __VA_ARGS__)
#define clGetPlatformInfo(...) ai_chat_opencl_dispatch::call<decltype(&clGetPlatformInfo)>("clGetPlatformInfo", __VA_ARGS__)
#define clGetProgramBuildInfo(...) ai_chat_opencl_dispatch::call<decltype(&clGetProgramBuildInfo)>("clGetProgramBuildInfo", __VA_ARGS__)
#define clGetProgramInfo(...) ai_chat_opencl_dispatch::call<decltype(&clGetProgramInfo)>("clGetProgramInfo", __VA_ARGS__)
#define clReleaseEvent(...) ai_chat_opencl_dispatch::call<decltype(&clReleaseEvent)>("clReleaseEvent", __VA_ARGS__)
#define clReleaseKernel(...) ai_chat_opencl_dispatch::call<decltype(&clReleaseKernel)>("clReleaseKernel", __VA_ARGS__)
#define clReleaseMemObject(...) ai_chat_opencl_dispatch::call<decltype(&clReleaseMemObject)>("clReleaseMemObject", __VA_ARGS__)
#define clReleaseProgram(...) ai_chat_opencl_dispatch::call<decltype(&clReleaseProgram)>("clReleaseProgram", __VA_ARGS__)
#define clSetKernelArg(...) ai_chat_opencl_dispatch::call<decltype(&clSetKernelArg)>("clSetKernelArg", __VA_ARGS__)
#define clWaitForEvents(...) ai_chat_opencl_dispatch::call<decltype(&clWaitForEvents)>("clWaitForEvents", __VA_ARGS__)
