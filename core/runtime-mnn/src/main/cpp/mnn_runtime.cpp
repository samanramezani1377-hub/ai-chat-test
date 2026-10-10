#include <jni.h>
#include <android/log.h>
#include <algorithm>
#include <atomic>
#include <mutex>
#include <sstream>
#include <string>
#include <llm/llm.hpp>
#define TAG "AIChatMNN"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, TAG, __VA_ARGS__)
#define LOGW(...) __android_log_print(ANDROID_LOG_WARN, TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, TAG, __VA_ARGS__)
using MNN::Transformer::Llm;
using MNN::Transformer::ChatMessages;
using MNN::Transformer::LlmStatus;
static std::mutex g_mutex;
static Llm* g_llm = nullptr;
static std::string g_config_path;
static std::string g_backend = "unloaded";
static int g_cpu_threads = 4;
static std::atomic_bool g_stop{false};
static std::string from_jstring(JNIEnv* env, jstring value) {
    if (!value) return {};
    const char* chars = env->GetStringUTFChars(value, nullptr);
    if (!chars) return {};
    std::string out(chars);
    env->ReleaseStringUTFChars(value, chars);
    return out;
}
static void destroy_llm_locked() {
    if (g_llm) { Llm::destroy(g_llm); g_llm = nullptr; }
}
static bool create_and_load_locked(const std::string& config, const char* backend, int threads) {
    destroy_llm_locked();
    Llm* candidate = Llm::createLLM(config);
    if (!candidate) { LOGE("MNN_CREATE_FAILED backend=%s", backend); return false; }
    const std::string overrideConfig = std::string("{\"backend_type\":\"") + backend +
        "\",\"thread_num\":" + std::to_string(threads) +
        ",\"precision\":\"low\",\"memory\":\"low\",\"use_mmap\":true}";
    if (!candidate->set_config(overrideConfig)) LOGW("MNN_SET_CONFIG_REJECTED backend=%s", backend);
    const bool loaded = candidate->load();
    const auto* context = candidate->getContext();
    const int status = context ? static_cast<int>(context->status) : -999;
    LOGI("MNN_LOAD_RESULT backend=%s loaded=%d status=%d", backend, loaded ? 1 : 0, status);
    if (!loaded || !context || context->status == LlmStatus::NOT_LOADED || context->status == LlmStatus::INTERNAL_ERROR) {
        Llm::destroy(candidate);
        return false;
    }
    g_llm = candidate;
    g_backend = backend;
    return true;
}
static bool load_with_fallback_locked(const std::string& config, int threads) {
    g_config_path = config;
    g_cpu_threads = std::max(1, std::min(threads, 8));
    g_stop.store(false);
    if (create_and_load_locked(config, "opencl", 68)) {
        LOGI("MNN_BACKEND_SELECTED backend=opencl policy=opencl-first");
        return true;
    }
    LOGW("MNN_OPENCL_LOAD_FAILED fallback=cpu");
    if (create_and_load_locked(config, "cpu", g_cpu_threads)) {
        LOGI("MNN_BACKEND_SELECTED backend=cpu reason=opencl_load_failed");
        return true;
    }
    g_backend = "failed";
    LOGE("MNN_BOTH_BACKENDS_FAILED config=%s", config.c_str());
    return false;
}
extern "C" JNIEXPORT jstring JNICALL
Java_com_woogit_aicore_runtime_mnn_MnnNative_nativeLoad(JNIEnv* env, jclass, jstring configPath, jint cpuThreads) {
    std::lock_guard<std::mutex> lock(g_mutex);
    const std::string config = from_jstring(env, configPath);
    if (config.empty()) return env->NewStringUTF("ERROR:empty_config_path");
    if (!load_with_fallback_locked(config, cpuThreads)) return env->NewStringUTF("ERROR:opencl_and_cpu_load_failed");
    return env->NewStringUTF(g_backend.c_str());
}
extern "C" JNIEXPORT jstring JNICALL
Java_com_woogit_aicore_runtime_mnn_MnnNative_nativeGenerate(JNIEnv* env, jclass, jobjectArray roles, jobjectArray contents, jint maxTokens) {
    std::lock_guard<std::mutex> lock(g_mutex);
    if (!g_llm) return env->NewStringUTF("ERROR:runtime_not_loaded");
    if (!roles || !contents || env->GetArrayLength(roles) != env->GetArrayLength(contents)) return env->NewStringUTF("ERROR:invalid_messages");
    ChatMessages messages;
    const jsize count = env->GetArrayLength(roles);
    messages.reserve(static_cast<size_t>(count));
    for (jsize i = 0; i < count; ++i) {
        auto role = static_cast<jstring>(env->GetObjectArrayElement(roles, i));
        auto content = static_cast<jstring>(env->GetObjectArrayElement(contents, i));
        messages.emplace_back(from_jstring(env, role), from_jstring(env, content));
        if (role) env->DeleteLocalRef(role);
        if (content) env->DeleteLocalRef(content);
    }
    auto run = [&](Llm* llm, std::string& output) -> bool {
        g_stop.store(false);
        std::ostringstream stream;
        llm->response(messages, &stream, nullptr, std::max(1, static_cast<int>(maxTokens)));
        output = stream.str();
        const auto* context = llm->getContext();
        return context && context->status != LlmStatus::INTERNAL_ERROR && context->status != LlmStatus::TIMEOUT && context->status != LlmStatus::USER_CANCEL && !output.empty();
    };
    std::string output;
    if (run(g_llm, output)) return env->NewStringUTF(output.c_str());
    if (g_backend == "opencl") {
        LOGW("MNN_OPENCL_GENERATION_FAILED retry=cpu partial_chars=%zu", output.size());
        if (create_and_load_locked(g_config_path, "cpu", g_cpu_threads)) {
            output.clear();
            if (run(g_llm, output)) {
                LOGI("MNN_GENERATION_RECOVERED backend=cpu reason=opencl_generation_failed");
                return env->NewStringUTF(output.c_str());
            }
        }
    }
    LOGE("MNN_GENERATION_FAILED backend=%s", g_backend.c_str());
    return env->NewStringUTF("ERROR:generation_failed");
}
extern "C" JNIEXPORT void JNICALL
Java_com_woogit_aicore_runtime_mnn_MnnNative_nativeStop(JNIEnv*, jclass) {
    // Do not acquire g_mutex here: nativeGenerate owns it while decoding. MNN checks
    // LlmContext status during generation, so request cancellation without blocking on
    // the generation lock. This pointer remains valid until the serialized unload.
    g_stop.store(true);
    Llm* active = g_llm;
    if (active) {
        auto* context = const_cast<MNN::Transformer::LlmContext*>(active->getContext());
        if (context) context->status = LlmStatus::USER_CANCEL;
    }
}
extern "C" JNIEXPORT void JNICALL
Java_com_woogit_aicore_runtime_mnn_MnnNative_nativeUnload(JNIEnv*, jclass) {
    std::lock_guard<std::mutex> lock(g_mutex);
    destroy_llm_locked();
    g_backend = "unloaded";
    g_config_path.clear();
}
extern "C" JNIEXPORT jstring JNICALL
Java_com_woogit_aicore_runtime_mnn_MnnNative_nativeBackend(JNIEnv* env, jclass) {
    std::lock_guard<std::mutex> lock(g_mutex);
    return env->NewStringUTF(g_backend.c_str());
}
