cmake_minimum_required(VERSION 3.22)
project(android_gpu_inference_engine LANGUAGES C CXX)

# Use ccache in CI for Dawn and engine translation units. Local Android builds
# remain independent of ccache and do not require an extra host tool.
if (DEFINED ENV{CI} AND DEFINED ENV{CCACHE_DIR})
    find_program(AI_CHAT_CCACHE_EXECUTABLE ccache REQUIRED)
    set(CMAKE_C_COMPILER_LAUNCHER "${AI_CHAT_CCACHE_EXECUTABLE}" CACHE STRING "C compiler launcher" FORCE)
    set(CMAKE_CXX_COMPILER_LAUNCHER "${AI_CHAT_CCACHE_EXECUTABLE}" CACHE STRING "C++ compiler launcher" FORCE)
endif()

if (NOT ANDROID)
    message(FATAL_ERROR "The Android GPU engine must be built with the Android NDK toolchain")
endif()
if (NOT CMAKE_ANDROID_ARCH_ABI STREQUAL "arm64-v8a")
    message(FATAL_ERROR "Only arm64-v8a is enabled in this Android build; got ${CMAKE_ANDROID_ARCH_ABI}")
endif()

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Dawn is the only fetched native GPU dependency. Its revision is pinned by the
# AndroidX WebGPU AAR metadata supplied by the Gradle build.
set(_ENGINE_CXX_STANDARD "${CMAKE_CXX_STANDARD}")
set(CMAKE_CXX_STANDARD 20)
include("${CMAKE_CURRENT_LIST_DIR}/cmake/FindDawn.cmake")
set(CMAKE_CXX_STANDARD "${_ENGINE_CXX_STANDARD}")
unset(_ENGINE_CXX_STANDARD)

if (NOT TARGET dawn::webgpu_dawn)
    message(FATAL_ERROR "Pinned Dawn did not provide dawn::webgpu_dawn")
endif()

set(ENGINE_DIR "${CMAKE_CURRENT_LIST_DIR}/gpu_engine")

add_library(ai_chat_gpu_engine STATIC
    ${ENGINE_DIR}/gpu_engine.cpp
    ${ENGINE_DIR}/execution_engine.cpp
    ${ENGINE_DIR}/gguf_reader.cpp
    ${ENGINE_DIR}/model_registry.cpp
    ${ENGINE_DIR}/model_graph.cpp
    ${ENGINE_DIR}/model_lowerings.cpp
    ${ENGINE_DIR}/operators.cpp
    ${ENGINE_DIR}/kernel_validation.cpp
    ${ENGINE_DIR}/weight_streamer.cpp
    ${ENGINE_DIR}/sampler.cpp
    ${ENGINE_DIR}/quantization.cpp
    ${ENGINE_DIR}/kv_cache.cpp
    ${ENGINE_DIR}/memory_planner.cpp
    ${ENGINE_DIR}/tokenizer.cpp
    ${ENGINE_DIR}/tokenizer_adapters.cpp
)
target_include_directories(ai_chat_gpu_engine PUBLIC "${CMAKE_CURRENT_LIST_DIR}")
target_compile_features(ai_chat_gpu_engine PUBLIC cxx_std_17)

add_library(ai_chat_gpu_backend_dawn STATIC ${ENGINE_DIR}/dawn_backend.cpp)
target_include_directories(ai_chat_gpu_backend_dawn PUBLIC "${CMAKE_CURRENT_LIST_DIR}")
target_compile_features(ai_chat_gpu_backend_dawn PUBLIC cxx_std_20)
target_link_libraries(ai_chat_gpu_backend_dawn PUBLIC ai_chat_gpu_engine PRIVATE dawn::webgpu_dawn)

add_library(ai_chat_gpu_backend_gles STATIC
    ${ENGINE_DIR}/universal_gles_backend.cpp
    ${ENGINE_DIR}/universal_adapter_bridge.cpp
    "${CMAKE_CURRENT_LIST_DIR}/universal_gpu_adapter.cpp"
)
target_include_directories(ai_chat_gpu_backend_gles PUBLIC "${CMAKE_CURRENT_LIST_DIR}")
target_compile_features(ai_chat_gpu_backend_gles PUBLIC cxx_std_17)
target_link_libraries(ai_chat_gpu_backend_gles PUBLIC ai_chat_gpu_engine PRIVATE EGL GLESv3 android log)

add_library(ai_chat_gpu_backends_android STATIC
    ${ENGINE_DIR}/android_backends.cpp
    ${ENGINE_DIR}/inference_session.cpp
)
target_include_directories(ai_chat_gpu_backends_android PUBLIC "${CMAKE_CURRENT_LIST_DIR}")
target_compile_features(ai_chat_gpu_backends_android PUBLIC cxx_std_17)
target_link_libraries(ai_chat_gpu_backends_android PUBLIC
    ai_chat_gpu_engine
    ai_chat_gpu_backend_dawn
    ai_chat_gpu_backend_gles
)

add_library(gpu_engine_android SHARED ${ENGINE_DIR}/android_jni.cpp)
target_include_directories(gpu_engine_android PRIVATE "${CMAKE_CURRENT_LIST_DIR}")
target_compile_options(gpu_engine_android PRIVATE -Os -gline-tables-only -ffunction-sections -fdata-sections)
target_link_options(gpu_engine_android PRIVATE -Wl,--gc-sections -Wl,--build-id=sha1)
# Extract debug information before stripping so native crash symbols remain available
# as a separate CI artifact, while the APK contains only runtime sections.
if (NOT CMAKE_STRIP OR NOT CMAKE_OBJCOPY)
    message(FATAL_ERROR "Android NDK strip and objcopy tools are required")
endif()
set(ENGINE_DEBUG_SYMBOLS "$<TARGET_FILE:gpu_engine_android>.debug")
add_custom_command(TARGET gpu_engine_android POST_BUILD
    COMMAND "${CMAKE_OBJCOPY}" --only-keep-debug
            "$<TARGET_FILE:gpu_engine_android>" "${ENGINE_DEBUG_SYMBOLS}"
    COMMAND "${CMAKE_STRIP}" --strip-debug "$<TARGET_FILE:gpu_engine_android>"
    COMMAND "${CMAKE_OBJCOPY}" --add-gnu-debuglink "${ENGINE_DEBUG_SYMBOLS}"
            "$<TARGET_FILE:gpu_engine_android>"
    COMMENT "Extracting native debug symbols and stripping packaged GPU engine"
    VERBATIM
)

target_link_libraries(gpu_engine_android PRIVATE
    ai_chat_gpu_backends_android
    ai_chat_gpu_engine
    ai_chat_gpu_backend_dawn
    ai_chat_gpu_backend_gles
    dawn::webgpu_dawn
    android log EGL GLESv3
)

target_compile_definitions(gpu_engine_android PRIVATE
    AI_CHAT_GPU_BACKEND_WEBGPU=1
    AI_CHAT_GPU_BACKEND_OPENGL_ES=1
    AI_CHAT_GPU_BACKEND_VULKAN=0
)

set_target_properties(gpu_engine_android PROPERTIES
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON
)
