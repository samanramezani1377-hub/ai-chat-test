# AndroidX WebGPU AARs package the Dawn C ABI shared library under jniLibs
# and the matching Dawn source SHA under assets. They do not publish a Prefab
# webgpuConfig.cmake package, so import the extracted AAR library explicitly.
if (NOT DEFINED AI_CHAT_WEBGPU_LIBRARY OR NOT EXISTS "${AI_CHAT_WEBGPU_LIBRARY}")
    message(FATAL_ERROR
        "AI_CHAT_WEBGPU_LIBRARY is missing. The Gradle extractWebGpuNativeLibrary task "
        "must extract libwebgpu_c_bundled.so before CMake configuration.")
endif()
if (NOT DEFINED AI_CHAT_DAWN_METADATA_FILE OR NOT EXISTS "${AI_CHAT_DAWN_METADATA_FILE}")
    message(FATAL_ERROR
        "AI_CHAT_DAWN_METADATA_FILE is missing. Extract assets/dawn_build_metadata.json "
        "from the same AndroidX WebGPU AAR as the native library.")
endif()

file(READ "${AI_CHAT_DAWN_METADATA_FILE}" _ai_chat_dawn_metadata)
string(JSON AI_CHAT_DAWN_SOURCE_COMMIT
    ERROR_VARIABLE _ai_chat_dawn_metadata_error
    GET "${_ai_chat_dawn_metadata}" dawn_source_commit_sha)
if (NOT _ai_chat_dawn_metadata_error STREQUAL "NOTFOUND")
    message(FATAL_ERROR
        "Could not read dawn_source_commit_sha from AndroidX metadata: ${_ai_chat_dawn_metadata_error}")
endif()
string(LENGTH "${AI_CHAT_DAWN_SOURCE_COMMIT}" _ai_chat_dawn_sha_length)
if (NOT _ai_chat_dawn_sha_length EQUAL 40 OR
    NOT AI_CHAT_DAWN_SOURCE_COMMIT MATCHES "^[0-9a-fA-F]+$")
    message(FATAL_ERROR "AndroidX AAR contains an invalid Dawn source SHA: ${AI_CHAT_DAWN_SOURCE_COMMIT}")
endif()

if (NOT TARGET webgpu::webgpu_c_bundled)
    add_library(webgpu::webgpu_c_bundled SHARED IMPORTED GLOBAL)
    set_target_properties(webgpu::webgpu_c_bundled PROPERTIES
        IMPORTED_LOCATION "${AI_CHAT_WEBGPU_LIBRARY}")
endif()

find_package(Python3 COMPONENTS Interpreter REQUIRED)
include(FetchContent)
FetchContent_Declare(ai_chat_dawn_headers
    URL "https://github.com/google/dawn/archive/${AI_CHAT_DAWN_SOURCE_COMMIT}.tar.gz"
)
FetchContent_GetProperties(ai_chat_dawn_headers)
if(NOT ai_chat_dawn_headers_POPULATED)
    FetchContent_Populate(ai_chat_dawn_headers)
endif()

# Generate C++ wrapper headers from the exact Dawn revision shipped in the AAR.
set(_ai_chat_dawn_generated "${CMAKE_CURRENT_BINARY_DIR}/androidx-dawn-generated")
file(MAKE_DIRECTORY "${_ai_chat_dawn_generated}")
execute_process(
    COMMAND "${Python3_EXECUTABLE}"
        "${ai_chat_dawn_headers_SOURCE_DIR}/generator/dawn_json_generator.py"
        --template-dir "${ai_chat_dawn_headers_SOURCE_DIR}/generator/templates"
        --output-dir "${_ai_chat_dawn_generated}"
        --dawn-json "${ai_chat_dawn_headers_SOURCE_DIR}/src/dawn/dawn.json"
        --targets headers,cpp_headers
    RESULT_VARIABLE _ai_chat_dawn_generator_result
    OUTPUT_VARIABLE _ai_chat_dawn_generator_stdout
    ERROR_VARIABLE _ai_chat_dawn_generator_stderr
)
if (NOT _ai_chat_dawn_generator_result EQUAL 0)
    message(FATAL_ERROR
        "Could not generate Dawn C++ WebGPU headers for the AndroidX AAR revision. "
        "The Python environment must include Dawn generator dependencies (notably Jinja2). "
        "stdout: ${_ai_chat_dawn_generator_stdout} stderr: ${_ai_chat_dawn_generator_stderr}"
    )
endif()

# The public webgpu/webgpu.h wrapper includes dawn/webgpu.h. The generator's
# C-header target is named "headers" (not "dawn_headers") and must emit it.
if (NOT EXISTS "${_ai_chat_dawn_generated}/include/dawn/webgpu.h")
    message(FATAL_ERROR
        "Dawn header generation completed but did not produce include/dawn/webgpu.h. "
        "Check the pinned Dawn generator target names and AAR source revision.")
endif()

if (NOT TARGET dawn::webgpu_dawn)
    add_library(dawn::webgpu_dawn INTERFACE IMPORTED GLOBAL)
    set_target_properties(dawn::webgpu_dawn PROPERTIES
        INTERFACE_LINK_LIBRARIES "webgpu::webgpu_c_bundled"
        INTERFACE_INCLUDE_DIRECTORIES
            "${ai_chat_dawn_headers_SOURCE_DIR}/include;${_ai_chat_dawn_generated}/include"
    )
endif()

set(Dawn_FOUND TRUE)
set(Dawn_VERSION "AndroidX-Jetpack-WebGPU-${AI_CHAT_DAWN_SOURCE_COMMIT}")
