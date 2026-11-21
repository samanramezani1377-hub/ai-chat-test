# Android OpenCL direct-dispatch integration and static backend registration.
# Android OpenCL vendor libraries are often complete OpenCL implementations but
# not ICDs. Patch only the pinned ggml OpenCL call sites to use our direct API
# dispatcher; the Khronos loader remains linked for ABI compatibility but is not
# used for driver dispatch. This avoids fabricating the ICD extension entrypoint.
set(AI_CHAT_OPENCL_SOURCE_FILES
    "${llama_cpp_SOURCE_DIR}/ggml/src/ggml-opencl/ggml-opencl.cpp"
    "${llama_cpp_SOURCE_DIR}/ggml/src/ggml-opencl/cl-program-cache.cpp"
)
foreach(AI_CHAT_OPENCL_SOURCE IN LISTS AI_CHAT_OPENCL_SOURCE_FILES)
    file(READ "${AI_CHAT_OPENCL_SOURCE}" AI_CHAT_OPENCL_SOURCE_TEXT)
    string(FIND "${AI_CHAT_OPENCL_SOURCE_TEXT}" "android_opencl_dispatch.h" AI_CHAT_OPENCL_ALREADY_PATCHED)
    if (AI_CHAT_OPENCL_ALREADY_PATCHED EQUAL -1)
        if (AI_CHAT_OPENCL_SOURCE MATCHES "ggml-opencl\\.cpp$")
            string(FIND "${AI_CHAT_OPENCL_SOURCE_TEXT}" "#include <CL/cl.h>" AI_CHAT_OPENCL_INCLUDE_POS)
            if (AI_CHAT_OPENCL_INCLUDE_POS EQUAL -1)
                message(FATAL_ERROR "Could not patch direct Android OpenCL dispatch in ${AI_CHAT_OPENCL_SOURCE}")
            endif()
            string(REPLACE "#include <CL/cl.h>"
                "#include <CL/cl.h>\n#include \"android_opencl_dispatch.h\""
                AI_CHAT_OPENCL_SOURCE_TEXT "${AI_CHAT_OPENCL_SOURCE_TEXT}")
        else()
            string(FIND "${AI_CHAT_OPENCL_SOURCE_TEXT}" "#include \"cl-program-cache.h\"" AI_CHAT_OPENCL_INCLUDE_POS)
            if (AI_CHAT_OPENCL_INCLUDE_POS EQUAL -1)
                message(FATAL_ERROR "Could not patch direct Android OpenCL dispatch in ${AI_CHAT_OPENCL_SOURCE}")
            endif()
            string(REPLACE "#include \"cl-program-cache.h\""
                "#include \"cl-program-cache.h\"\n#include \"android_opencl_dispatch.h\""
                AI_CHAT_OPENCL_SOURCE_TEXT "${AI_CHAT_OPENCL_SOURCE_TEXT}")
        endif()
        file(WRITE "${AI_CHAT_OPENCL_SOURCE}" "${AI_CHAT_OPENCL_SOURCE_TEXT}")
    endif()
endforeach()

# Do not spoof an unsupported GPU family by matching device-name substrings. The pinned
# upstream OpenCL backend contains family-specific kernels and subgroup assumptions.
# Device-agnostic support must be added through runtime capability checks and genuinely
# generic kernels, not by relabeling Mali/unknown devices as Adreno or Intel.
target_include_directories(ggml-opencl PRIVATE "${AI_CHAT_NATIVE_CMAKE_DIR}")

if (NOT TARGET ggml-opencl)
    message(FATAL_ERROR "OpenCL-preferred runtime requires the statically linked ggml-opencl target")
endif()
target_include_directories(ggml-opencl PRIVATE "${AI_CHAT_NATIVE_CMAKE_DIR}")

# Static backend hardening: make the OpenCL registration visible to the final
# runtime target as a real link dependency. This prevents linker GC from
# discarding the backend registration object and makes a missing OpenCL target
# a build-time error instead of a silent CPU-only runtime.
if (NOT TARGET ggml-opencl)
    message(FATAL_ERROR "OpenCL-preferred runtime requires the statically linked ggml-opencl target")
endif()
if (TARGET ggml)
    target_compile_definitions(ggml PUBLIC GGML_USE_OPENCL=1)
endif()

if (AI_CHAT_OPENCL_PROFILING)
    include("${AI_CHAT_NATIVE_CMAKE_DIR}/cmake/patch_opencl_profiling.cmake")
endif()
