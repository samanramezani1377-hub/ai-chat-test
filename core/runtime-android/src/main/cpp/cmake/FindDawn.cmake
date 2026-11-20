# AndroidX Jetpack WebGPU publishes the Dawn C ABI and native library as Prefab.
# GGML's upstream backend uses Dawn's C++ wrapper, which is generated from the exact
# Dawn revision recorded by AndroidX WebGPU 1.0.0-alpha06.
find_package(webgpu CONFIG REQUIRED)
find_package(Python3 COMPONENTS Interpreter REQUIRED)

if(TARGET webgpu::webgpu_c_bundled)
    set(_ai_chat_dawn_target webgpu::webgpu_c_bundled)
else()
    message(FATAL_ERROR "AndroidX WebGPU Prefab package did not export webgpu::webgpu_c_bundled target")
endif()

# AndroidX alpha06's meta/dawn_build_metadata.json pins this exact source revision.
# Only Dawn's generated C++ API headers are needed here; the implementation itself
# remains the prebuilt libwebgpu_c_bundled.so supplied by AndroidX.
set(AI_CHAT_DAWN_SOURCE_COMMIT "4b959fbe989d529e1eee839c48cda465cc1195fe")
include(FetchContent)
FetchContent_Declare(ai_chat_dawn_headers
    URL "https://github.com/google/dawn/archive/${AI_CHAT_DAWN_SOURCE_COMMIT}.tar.gz"
)
FetchContent_GetProperties(ai_chat_dawn_headers)
if(NOT ai_chat_dawn_headers_POPULATED)
    FetchContent_Populate(ai_chat_dawn_headers)
endif()

set(_ai_chat_dawn_generated "${CMAKE_CURRENT_BINARY_DIR}/androidx-dawn-generated")
file(MAKE_DIRECTORY "${_ai_chat_dawn_generated}")
execute_process(
    COMMAND "${Python3_EXECUTABLE}"
        "${ai_chat_dawn_headers_SOURCE_DIR}/generator/dawn_json_generator.py"
        --template-dir "${ai_chat_dawn_headers_SOURCE_DIR}/generator/templates"
        --output-dir "${_ai_chat_dawn_generated}"
        --dawn-json "${ai_chat_dawn_headers_SOURCE_DIR}/src/dawn/dawn.json"
        --targets cpp_headers
    RESULT_VARIABLE _ai_chat_dawn_generator_result
    OUTPUT_VARIABLE _ai_chat_dawn_generator_stdout
    ERROR_VARIABLE _ai_chat_dawn_generator_stderr
)
if(NOT _ai_chat_dawn_generator_result EQUAL 0)
    message(FATAL_ERROR
        "Could not generate Dawn C++ WebGPU headers for AndroidX alpha06. "
        "The Python environment must include Dawn generator dependencies (notably Jinja2). "
        "stdout: ${_ai_chat_dawn_generator_stdout} stderr: ${_ai_chat_dawn_generator_stderr}"
    )
endif()

if(NOT TARGET dawn::webgpu_dawn)
    add_library(dawn::webgpu_dawn INTERFACE IMPORTED)
    set_target_properties(dawn::webgpu_dawn PROPERTIES
        INTERFACE_LINK_LIBRARIES "${_ai_chat_dawn_target}"
        INTERFACE_INCLUDE_DIRECTORIES
            "${ai_chat_dawn_headers_SOURCE_DIR}/include;${_ai_chat_dawn_generated}/include"
    )
endif()

set(Dawn_FOUND TRUE)
set(Dawn_VERSION "AndroidX-Jetpack-WebGPU-1.0.0-alpha06")
