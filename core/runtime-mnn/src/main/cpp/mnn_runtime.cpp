#include <jni.h>
#include <android/log.h>
#include <algorithm>
#include <atomic>
#include <exception>
#include <filesystem>
#include <iomanip>
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
static std::string g_backend_reason = "not_loaded";
static std::string g_opencl_load_error = "not_attempted";
static std::string g_cpu_load_error = "not_attempted";
static std::string g_last_load_error = "not_attempted";
static int g_cpu_threads = 4;
static std::atomic_bool g_stop{false};

namespace fs = std::filesystem;
static std::string bounded(std::string value, size_t limit = 12000) {
    if (value.size() > limit) {
        value.resize(limit);
        value += "\\n[MNN_LOG_TRUNCATED]";
    }
    return value;
}
static std::string file_probe(const fs::path& path) {
    std::error_code ec;
    const bool exists = fs::is_regular_file(path, ec);
    if (ec) return path.string() + " [stat_error=" + ec.message() + "]";
    if (!exists) return path.string() + " [MISSING]";
    const auto size = fs::file_size(path, ec);
    if (ec) return path.string() + " [exists,size_error=" + ec.message() + "]";
    return path.string() + " [exists,size=" + std::to_string(size) + "]";
}
static std::string pack_diagnostics(const std::string& config) {
    const fs::path configPath(config);
    const fs::path dir = configPath.parent_path();
    std::ostringstream out;
    out << "config=" << file_probe(configPath)
        << "; llm_config=" << file_probe(dir / "llm_config.json")
        << "; model=" << file_probe(dir / "llm.mnn")
        << "; weight=" << file_probe(dir / "llm.mnn.weight")
        << "; tokenizer_txt=" << file_probe(dir / "tokenizer.txt")
        << "; tokenizer_mtok=" << file_probe(dir / "tokenizer.mtok")
        << "; context=" << file_probe(dir / "context.json");
    return out.str();
}
static std::string take_mnn_log(Llm* llm) {
    if (!llm) return {};
    try {
        return bounded(llm->getLog());
    } catch (...) {
        return "[MNN getLog() failed]";
    }
}

class StopAwareBuffer final : public std::stringbuf {
public:
    explicit StopAwareBuffer(Llm* llm) : llm_(llm) {}
protected:
    std::streamsize xsputn(const char* text, std::streamsize count) override {
        mark_cancel_if_requested();
        return std::stringbuf::xsputn(text, count);
    }
    int overflow(int character = traits_type::eof()) override {
        mark_cancel_if_requested();
        return std::stringbuf::overflow(character);
    }
    int sync() override {
        mark_cancel_if_requested();
        return std::stringbuf::sync();
    }
private:
    void mark_cancel_if_requested() {
        if (g_stop.load() && llm_) {
            auto* context = const_cast<MNN::Transformer::LlmContext*>(llm_->getContext());
            if (context) context->status = LlmStatus::USER_CANCEL;
        }
    }
    Llm* llm_;
};

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
    Llm* candidate = nullptr;
    try {
        candidate = Llm::createLLM(config);
        if (!candidate) {
            g_last_load_error = std::string("createLLM returned null; backend=") + backend +
                "; config=" + config + "; pack={" + pack_diagnostics(config) + "}";
            LOGE("MNN_CREATE_FAILED backend=%s", backend);
            if (std::string(backend) == "opencl") g_opencl_load_error = g_last_load_error;
            else g_cpu_load_error = g_last_load_error;
            return false;
        }
        const std::string overrideConfig = std::string(R"({"backend_type":")") + backend +
            R"(","thread_num":)" + std::to_string(threads) +
            R"(,"precision":"low","memory":"low","use_mmap":true})";
        if (!candidate->set_config(overrideConfig)) {
            g_last_load_error = std::string("set_config rejected override; backend=") + backend +
                "; override=" + overrideConfig + "; config=" + config +
                "; MNN_internal_log=" + take_mnn_log(candidate);
            LOGE("MNN_SET_CONFIG_REJECTED backend=%s; refusing to misreport the selected backend", backend);
            Llm::destroy(candidate);
            if (std::string(backend) == "opencl") g_opencl_load_error = g_last_load_error;
            else g_cpu_load_error = g_last_load_error;
            return false;
        }
        LOGI("MNN_LOAD_BEGIN backend=%s config=%s pack={%s}", backend, config.c_str(), pack_diagnostics(config).c_str());
        LOGI("MNN_EFFECTIVE_CONFIG backend=%s config_json=%s", backend, bounded(candidate->dump_config(), 6000).c_str());
        const bool loaded = candidate->load();
        const std::string mnnLog = take_mnn_log(candidate);
        if (!mnnLog.empty()) {
            LOGE("MNN_INTERNAL_LOG backend=%s detail=%s", backend, mnnLog.c_str());
        }
        const auto* context = candidate->getContext();
        const int status = context ? static_cast<int>(context->status) : -999;
        LOGI("MNN_LOAD_RESULT backend=%s loaded=%d status=%d", backend, loaded ? 1 : 0, status);
        if (!loaded || !context || context->status == LlmStatus::NOT_LOADED || context->status == LlmStatus::INTERNAL_ERROR) {
            g_last_load_error = std::string("load returned false or invalid status; backend=") + backend +
                "; loaded=" + (loaded ? "1" : "0") + "; context_status=" + std::to_string(status) +
                "; config=" + config + "; pack={" + pack_diagnostics(config) + "}" +
                (mnnLog.empty() ? "; MNN_internal_log=<empty; ensure LLM_LOG_TO_STRING is enabled>" : "; MNN_internal_log=" + mnnLog);
            LOGE("MNN_LOAD_REJECTED backend=%s loaded=%d context_status=%d", backend, loaded ? 1 : 0, status);
            Llm::destroy(candidate);
            if (std::string(backend) == "opencl") g_opencl_load_error = g_last_load_error;
            else g_cpu_load_error = g_last_load_error;
            return false;
        }
        g_llm = candidate;
        g_backend = backend;
        return true;
    } catch (const std::exception& error) {
        if (candidate) Llm::destroy(candidate);
        g_last_load_error = std::string("exception; backend=") + backend + "; message=" + error.what() +
            "; config=" + config + "; pack={" + pack_diagnostics(config) + "}" +
            "; MNN_internal_log=" + take_mnn_log(candidate);
        if (std::string(backend) == "opencl") g_opencl_load_error = g_last_load_error;
        else g_cpu_load_error = g_last_load_error;
        LOGE("MNN_LOAD_EXCEPTION backend=%s message=%s", backend, error.what());
        return false;
    } catch (...) {
        if (candidate) Llm::destroy(candidate);
        g_last_load_error = std::string("unknown exception; backend=") + backend +
            "; config=" + config + "; pack={" + pack_diagnostics(config) + "}" +
            "; MNN_internal_log=" + take_mnn_log(candidate);
        if (std::string(backend) == "opencl") g_opencl_load_error = g_last_load_error;
        else g_cpu_load_error = g_last_load_error;
        LOGE("MNN_LOAD_EXCEPTION backend=%s message=unknown", backend);
        return false;
    }
}
static bool load_with_fallback_locked(const std::string& config, int threads) {
    g_config_path = config;
    g_cpu_threads = std::max(1, std::min(threads, 8));
    g_stop.store(false);
    g_backend_reason = "none";
    g_opencl_load_error = "not_attempted";
    g_cpu_load_error = "not_attempted";
    g_last_load_error = "not_attempted";
    // MNN documents thread_num=68 for its OpenCL LLM backend. This is a
    // backend mode selector, not a CPU thread count; CPU uses g_cpu_threads.
    if (create_and_load_locked(config, "opencl", 68)) {
        LOGI("MNN_BACKEND_SELECTED backend=opencl policy=opencl-first");
        return true;
    }
    g_backend_reason = "opencl_load_failed";
    LOGW("MNN_OPENCL_LOAD_FAILED fallback=cpu");
    if (create_and_load_locked(config, "cpu", g_cpu_threads)) {
        LOGI("MNN_BACKEND_SELECTED backend=cpu reason=opencl_load_failed");
        return true;
    }
    g_backend = "failed";
    g_backend_reason = "opencl_and_cpu_load_failed";
    g_last_load_error = "OpenCL attempt: " + g_opencl_load_error + " | CPU attempt: " + g_cpu_load_error;
    LOGE("MNN_BOTH_BACKENDS_FAILED config=%s opencl_error=%s cpu_error=%s",
         config.c_str(), g_opencl_load_error.c_str(), g_cpu_load_error.c_str());
    return false;
}
extern "C" JNIEXPORT jstring JNICALL
Java_com_woogit_aicore_runtime_mnn_MnnNative_nativeLoad(JNIEnv* env, jclass, jstring configPath, jint cpuThreads) {
    std::lock_guard<std::mutex> lock(g_mutex);
    const std::string config = from_jstring(env, configPath);
    if (config.empty()) return env->NewStringUTF("ERROR:empty_config_path");
    if (!load_with_fallback_locked(config, cpuThreads)) {
        const std::string detail = "ERROR:opencl_and_cpu_load_failed; " + g_last_load_error;
        return env->NewStringUTF(detail.c_str());
    }
    return env->NewStringUTF((g_backend + ";" + g_backend_reason).c_str());
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
        StopAwareBuffer buffer(llm);
        std::ostream stream(&buffer);
        try {
            llm->response(messages, &stream, nullptr, std::max(1, static_cast<int>(maxTokens)));
        } catch (const std::exception& error) {
            output = buffer.str();
            LOGE("MNN_RESPONSE_EXCEPTION backend=%s message=%s", g_backend.c_str(), error.what());
            return false;
        } catch (...) {
            output = buffer.str();
            LOGE("MNN_RESPONSE_EXCEPTION backend=%s message=unknown", g_backend.c_str());
            return false;
        }
        output = buffer.str();
        const auto* context = llm->getContext();
        return context && context->status != LlmStatus::INTERNAL_ERROR && context->status != LlmStatus::TIMEOUT && context->status != LlmStatus::USER_CANCEL && !output.empty();
    };
    std::string output;
    if (run(g_llm, output)) return env->NewStringUTF(output.c_str());
    if (g_stop.load()) return env->NewStringUTF(output.c_str());
    if (g_backend == "opencl") {
        LOGW("MNN_OPENCL_GENERATION_FAILED retry=cpu partial_chars=%zu", output.size());
        g_backend_reason = "opencl_generation_failed";
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
    // The generation thread applies USER_CANCEL from StopAwareBuffer while streaming
    // output. This avoids racing on MNN's non-atomic context status from the UI thread.
    g_stop.store(true);
}
extern "C" JNIEXPORT void JNICALL
Java_com_woogit_aicore_runtime_mnn_MnnNative_nativeUnload(JNIEnv*, jclass) {
    std::lock_guard<std::mutex> lock(g_mutex);
    destroy_llm_locked();
    g_backend = "unloaded";
    g_backend_reason = "not_loaded";
    g_config_path.clear();
}
extern "C" JNIEXPORT jstring JNICALL
Java_com_woogit_aicore_runtime_mnn_MnnNative_nativeBackend(JNIEnv* env, jclass) {
    std::lock_guard<std::mutex> lock(g_mutex);
    return env->NewStringUTF((g_backend + ";" + g_backend_reason).c_str());
}
