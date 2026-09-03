#include "llama.h"

// Keep the JNI implementation in native_runtime.cpp and intercept only its
// model-load call so Android uses a bounded-memory loading policy.
extern "C" llama_model * llama_model_load_from_file_for_android(const char * path_model, llama_model_params params);
#define llama_model_load_from_file llama_model_load_from_file_for_android
#include "native_runtime.cpp"
#undef llama_model_load_from_file

extern "C" llama_model * llama_model_load_from_file_for_android(const char * path_model, llama_model_params params) {
    params.load_mode = LLAMA_LOAD_MODE_MMAP;
    params.check_tensors = false;
    return llama_model_load_from_file(path_model, params);
}
