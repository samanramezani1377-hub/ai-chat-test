# Android does not ship a linkable OpenCL SDK library in the NDK.
# The project already builds the Khronos ICD loader as the portable link-time
# implementation. The actual vendor OpenCL implementation is selected at runtime.
if(NOT TARGET OpenCL::OpenCL)
    if(TARGET OpenCL)
        add_library(OpenCL::OpenCL ALIAS OpenCL)
    else()
        message(FATAL_ERROR "OpenCL ICD loader target was not created")
    endif()
endif()

set(OpenCL_FOUND TRUE)
set(OpenCL_VERSION_STRING "3.0")
set(OpenCL_INCLUDE_DIRS "${opencl_headers_SOURCE_DIR}")
set(OpenCL_LIBRARIES OpenCL::OpenCL)
