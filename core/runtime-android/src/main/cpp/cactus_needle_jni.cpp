#include <jni.h>

#include <fstream>
#include <mutex>
#include <string>
#include <vector>

#include "needle/needle.h"

namespace {
std::mutex g_needle_mutex;
std::string g_loaded_model_path;
bool g_model_loaded = false;

void throwIllegalState(JNIEnv* env, const std::string& message) {
    jclass exception = env->FindClass("java/lang/IllegalStateException");
    if (exception != nullptr) env->ThrowNew(exception, message.c_str());
}

std::string needleError(const char* fallback) {
    const char* detail = needle_last_error();
    return detail != nullptr && detail[0] != '\0' ? std::string(detail) : std::string(fallback);
}
}  // namespace

extern "C" JNIEXPORT jboolean JNICALL
Java_com_samanramezani_aichattest_voice_CactusNeedleNative_loadModelNative(
    JNIEnv* env, jobject /* thiz */, jstring model_path) {
    std::lock_guard<std::mutex> guard(g_needle_mutex);
    const char* raw_path = env->GetStringUTFChars(model_path, nullptr);
    if (raw_path == nullptr) return JNI_FALSE;
    const std::string path(raw_path);
    env->ReleaseStringUTFChars(model_path, raw_path);

    if (g_model_loaded && g_loaded_model_path == path) return JNI_TRUE;

    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input) {
        throwIllegalState(env, "Cannot open Needle .cact model file.");
        return JNI_FALSE;
    }
    const std::streamsize size = input.tellg();
    if (size <= 0 || size > 128LL * 1024LL * 1024LL) {
        throwIllegalState(env, "Needle .cact model has an invalid size.");
        return JNI_FALSE;
    }
    input.seekg(0, std::ios::beg);
    std::vector<unsigned char> bytes(static_cast<size_t>(size));
    if (!input.read(reinterpret_cast<char*>(bytes.data()), size)) {
        throwIllegalState(env, "Cannot read the complete Needle .cact model.");
        return JNI_FALSE;
    }

    needle_reset();
    const int result = needle_load(bytes.data(), static_cast<unsigned long long>(bytes.size()));
    if (result < 0) {
        throwIllegalState(env, needleError("Needle model loading failed."));
        return JNI_FALSE;
    }
    g_loaded_model_path = path;
    g_model_loaded = true;
    return JNI_TRUE;
}

extern "C" JNIEXPORT jint JNICALL
Java_com_samanramezani_aichattest_voice_CactusNeedleNative_initNative(
    JNIEnv* env, jobject /* thiz */, jstring system_prompt, jstring tools_json) {
    std::lock_guard<std::mutex> guard(g_needle_mutex);
    if (!g_model_loaded) {
        throwIllegalState(env, "Needle model has not been loaded.");
        return -1;
    }

    const char* prompt = env->GetStringUTFChars(system_prompt, nullptr);
    const char* tools = env->GetStringUTFChars(tools_json, nullptr);
    if (prompt == nullptr || tools == nullptr) {
        if (prompt != nullptr) env->ReleaseStringUTFChars(system_prompt, prompt);
        if (tools != nullptr) env->ReleaseStringUTFChars(tools_json, tools);
        return -1;
    }

    needle_reset();
    const int result = needle_init(prompt, tools, nullptr);
    env->ReleaseStringUTFChars(system_prompt, prompt);
    env->ReleaseStringUTFChars(tools_json, tools);
    if (result < 0) throwIllegalState(env, needleError("Needle tool initialization failed."));
    return result;
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_samanramezani_aichattest_voice_CactusNeedleNative_completeNative(
    JNIEnv* env, jobject /* thiz */, jstring input, jint max_new_tokens) {
    std::lock_guard<std::mutex> guard(g_needle_mutex);
    if (!g_model_loaded) {
        throwIllegalState(env, "Needle model has not been loaded.");
        return nullptr;
    }

    const char* text = env->GetStringUTFChars(input, nullptr);
    if (text == nullptr) return nullptr;

    std::vector<char> output(64 * 1024, '\0');
    const int result = needle_complete(text, nullptr, 0, max_new_tokens,
                                       output.data(), static_cast<int>(output.size()));
    env->ReleaseStringUTFChars(input, text);
    if (result < 0) {
        throwIllegalState(env, needleError("Needle inference failed."));
        return nullptr;
    }
    output.back() = '\0';
    return env->NewStringUTF(output.data());
}

extern "C" JNIEXPORT void JNICALL
Java_com_samanramezani_aichattest_voice_CactusNeedleNative_resetNative(
    JNIEnv* /* env */, jobject /* thiz */) {
    std::lock_guard<std::mutex> guard(g_needle_mutex);
    if (g_model_loaded) needle_reset();
}
