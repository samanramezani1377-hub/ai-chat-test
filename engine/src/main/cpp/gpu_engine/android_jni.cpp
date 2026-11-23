#include <jni.h>

#include "android_backends.h"
#include "inference_session.h"
#include "execution_engine.h"
#include "model_lowerings.h"
#include "operators.h"
#include "model_registry.h"
#include "quantization.h"
#include "tokenizer.h"

#include <exception>
#include <cstdlib>
#include <sstream>
#include <stdexcept>
#include <cstdint>
#include <vector>
#include <memory>
#include <string>

namespace ai_chat::gpu_engine {
namespace {

const char* backend_name(BackendKind backend) {
    switch (backend) {
        case BackendKind::DawnWebGpu: return "DawnWebGPU";
        case BackendKind::OpenGles: return "UniversalGLES";
        default: return "Unknown";
    }
}

GpuInferenceSession& gpu_inference_session() {
    static GpuInferenceSession session;
    return session;
}

void append_utf8(uint32_t cp, std::string& out) {
    if (cp <= 0x7f) out.push_back(static_cast<char>(cp));
    else if (cp <= 0x7ff) {
        out.push_back(static_cast<char>(0xc0 | (cp >> 6)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3f)));
    } else if (cp <= 0xffff) {
        out.push_back(static_cast<char>(0xe0 | (cp >> 12)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3f)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3f)));
    } else {
        out.push_back(static_cast<char>(0xf0 | (cp >> 18)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3f)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3f)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3f)));
    }
}

std::string from_java_string(JNIEnv* env, jstring value) {
    if (!value) return {};
    const jsize length = env->GetStringLength(value);
    const jchar* chars = env->GetStringChars(value, nullptr);
    if (!chars) throw std::runtime_error("Unable to read Java UTF-16 string");
    std::string result;
    result.reserve(static_cast<size_t>(length) * 3);
    for (jsize i = 0; i < length; ++i) {
        uint32_t cp = chars[i];
        if (cp >= 0xd800 && cp <= 0xdbff && i + 1 < length &&
            chars[i + 1] >= 0xdc00 && chars[i + 1] <= 0xdfff) {
            cp = 0x10000 + ((cp - 0xd800) << 10) + (chars[++i] - 0xdc00);
        } else if (cp >= 0xd800 && cp <= 0xdfff) {
            cp = 0xfffd;
        }
        append_utf8(cp, result);
    }
    env->ReleaseStringChars(value, chars);
    return result;
}

// Hugging Face's Qwen3.5 and Qwen3/Qwen2 tokenizers apply Unicode NFC before pre-tokenization.
// Use Android's ICU-backed java.text.Normalizer so the complete Unicode tables
// are provided by the OS instead of a partial hand-maintained range table.
std::string normalized_java_text(JNIEnv* env, jstring value) {
    if (!value) return {};
    jclass normalizer = env->FindClass("java/text/Normalizer");
    if (!normalizer) throw std::runtime_error("java.text.Normalizer is unavailable");
    jclass form_class = env->FindClass("java/text/Normalizer$Form");
    if (!form_class) {
        env->DeleteLocalRef(normalizer);
        throw std::runtime_error("java.text.Normalizer.Form is unavailable");
    }
    jfieldID nfc_field = env->GetStaticFieldID(
        form_class, "NFC", "Ljava/text/Normalizer$Form;");
    jmethodID normalize = env->GetStaticMethodID(
        normalizer, "normalize",
        "(Ljava/lang/CharSequence;Ljava/text/Normalizer$Form;)Ljava/lang/String;");
    if (!nfc_field || !normalize) {
        env->DeleteLocalRef(form_class);
        env->DeleteLocalRef(normalizer);
        throw std::runtime_error("Unable to resolve NFC normalization API");
    }
    jobject nfc = env->GetStaticObjectField(form_class, nfc_field);
    auto normalized = static_cast<jstring>(env->CallStaticObjectMethod(
        normalizer, normalize, value, nfc));
    if (env->ExceptionCheck() || !normalized) {
        env->ExceptionClear();
        if (nfc) env->DeleteLocalRef(nfc);
        env->DeleteLocalRef(form_class);
        env->DeleteLocalRef(normalizer);
        throw std::runtime_error("Qwen3.5 NFC normalization failed");
    }
    std::string result = from_java_string(env, normalized);
    env->DeleteLocalRef(normalized);
    if (nfc) env->DeleteLocalRef(nfc);
    env->DeleteLocalRef(form_class);
    env->DeleteLocalRef(normalizer);
    return result;
}

Qwen35UnicodeCategories classify_qwen35_text(JNIEnv* env, const std::string& text) {
    Qwen35UnicodeCategories categories;
    jclass character = env->FindClass("java/lang/Character");
    if (!character) throw std::runtime_error("java.lang.Character is unavailable");
    jmethodID get_type = env->GetStaticMethodID(character, "getType", "(I)I");
    if (!get_type) {
        env->DeleteLocalRef(character);
        throw std::runtime_error("Character.getType(int) is unavailable");
    }
    auto is_unicode_space = [](uint32_t cp) {
        return (cp >= 0x0009 && cp <= 0x000d) || cp == 0x0020 || cp == 0x0085 ||
               cp == 0x00a0 || cp == 0x1680 || (cp >= 0x2000 && cp <= 0x200a) ||
               cp == 0x2028 || cp == 0x2029 || cp == 0x202f || cp == 0x205f ||
               cp == 0x3000;
    };
    for (size_t i = 0; i < text.size();) {
        const uint8_t first = static_cast<uint8_t>(text[i]);
        uint32_t cp = first;
        size_t width = 1;
        if ((first & 0xe0) == 0xc0 && i + 1 < text.size()) {
            cp = ((first & 0x1f) << 6) | (static_cast<uint8_t>(text[i+1]) & 0x3f); width = 2;
        } else if ((first & 0xf0) == 0xe0 && i + 2 < text.size()) {
            cp = ((first & 0x0f) << 12) | ((static_cast<uint8_t>(text[i+1]) & 0x3f) << 6) |
                 (static_cast<uint8_t>(text[i+2]) & 0x3f); width = 3;
        } else if ((first & 0xf8) == 0xf0 && i + 3 < text.size()) {
            cp = ((first & 0x07) << 18) | ((static_cast<uint8_t>(text[i+1]) & 0x3f) << 12) |
                 ((static_cast<uint8_t>(text[i+2]) & 0x3f) << 6) |
                 (static_cast<uint8_t>(text[i+3]) & 0x3f); width = 4;
        }
        i += width;
        uint8_t kind = 4; // Other
        if (cp == '\r' || cp == '\n') kind = 3; // Newline
        else if (is_unicode_space(cp)) kind = 2; // Space
        else {
            const jint type = env->CallStaticIntMethod(character, get_type, static_cast<jint>(cp));
            if (env->ExceptionCheck()) {
                env->ExceptionClear();
                env->DeleteLocalRef(character);
                throw std::runtime_error("Character.getType failed during Qwen3.5 tokenization");
            }
            if (type >= 1 && type <= 8) kind = 0; // Letter or Mark
            else if (type >= 9 && type <= 11) kind = 1; // Number
        }
        categories.emplace(cp, kind);
    }
    env->DeleteLocalRef(character);
    return categories;
}

class ScopedQwen35UnicodeCategories {
public:
    ScopedQwen35UnicodeCategories(JNIEnv* env, const std::string& normalized_text)
        : categories_(classify_qwen35_text(env, normalized_text)) {
        set_qwen35_unicode_categories(&categories_);
    }
    ~ScopedQwen35UnicodeCategories() { set_qwen35_unicode_categories(nullptr); }
    ScopedQwen35UnicodeCategories(const ScopedQwen35UnicodeCategories&) = delete;
    ScopedQwen35UnicodeCategories& operator=(const ScopedQwen35UnicodeCategories&) = delete;
private:
    Qwen35UnicodeCategories categories_;
};

size_t complete_utf8_prefix(const std::string& value) {
    size_t i = 0;
    while (i < value.size()) {
        const uint8_t lead = static_cast<uint8_t>(value[i]);
        size_t width = 1;
        if (lead >= 0xc2 && lead <= 0xdf) width = 2;
        else if (lead >= 0xe0 && lead <= 0xef) width = 3;
        else if (lead >= 0xf0 && lead <= 0xf4) width = 4;
        else if (lead >= 0x80) { ++i; continue; }
        if (i + width > value.size()) break;
        bool valid = true;
        for (size_t j = 1; j < width; ++j)
            if ((static_cast<uint8_t>(value[i+j]) & 0xc0) != 0x80) valid = false;
        if (valid && width == 3) {
            const uint8_t second = static_cast<uint8_t>(value[i+1]);
            if ((lead == 0xe0 && second < 0xa0) || (lead == 0xed && second >= 0xa0)) valid = false;
        }
        if (valid && width == 4) {
            const uint8_t second = static_cast<uint8_t>(value[i+1]);
            if ((lead == 0xf0 && second < 0x90) || (lead == 0xf4 && second >= 0x90)) valid = false;
        }
        i += valid ? width : 1;
    }
    return i;
}

jstring to_java_string(JNIEnv* env, const std::string& value) {
    std::vector<jchar> utf16;
    utf16.reserve(value.size());
    for (size_t i = 0; i < value.size();) {
        const uint8_t first = static_cast<uint8_t>(value[i]);
        uint32_t cp = 0xfffd;
        size_t width = 1;
        if (first < 0x80) cp = first;
        else if ((first & 0xe0) == 0xc0 && i + 1 < value.size()) {
            cp = ((first & 0x1f) << 6) | (static_cast<uint8_t>(value[i+1]) & 0x3f); width = 2;
        } else if ((first & 0xf0) == 0xe0 && i + 2 < value.size()) {
            cp = ((first & 0x0f) << 12) | ((static_cast<uint8_t>(value[i+1]) & 0x3f) << 6) |
                 (static_cast<uint8_t>(value[i+2]) & 0x3f); width = 3;
        } else if ((first & 0xf8) == 0xf0 && i + 3 < value.size()) {
            cp = ((first & 0x07) << 18) | ((static_cast<uint8_t>(value[i+1]) & 0x3f) << 12) |
                 ((static_cast<uint8_t>(value[i+2]) & 0x3f) << 6) |
                 (static_cast<uint8_t>(value[i+3]) & 0x3f); width = 4;
        }
        i += width;
        if (cp > 0x10ffff || (cp >= 0xd800 && cp <= 0xdfff)) cp = 0xfffd;
        if (cp <= 0xffff) utf16.push_back(static_cast<jchar>(cp));
        else {
            cp -= 0x10000;
            utf16.push_back(static_cast<jchar>(0xd800 + (cp >> 10)));
            utf16.push_back(static_cast<jchar>(0xdc00 + (cp & 0x3ff)));
        }
    }
    return env->NewString(utf16.data(), static_cast<jsize>(utf16.size()));
}

std::string probe_gpu_engine() {
    GpuEngine engine;
    OperatorRegistry operators;
    QuantizationRegistry quantizations;
    ArchitectureRegistry architectures;
    TokenizerRegistry tokenizers;
    ModelLoweringRegistry lowerings;

    const Status status = initialize_android_gpu_runtime(
        engine, operators, quantizations, architectures, tokenizers, lowerings);
    std::ostringstream report;
    report << "runtime=AndroidGpuInferenceEngine"
           << ";status=" << (status.ok() ? "READY" : "UNAVAILABLE");
    if (!status.ok()) {
        report << ";error=" << status.message;
        engine.shutdown();
        return report.str();
    }

    KernelValidationReport operator_report;
    const Status operator_status = validate_builtin_operator_kernels(
        *engine.active_backend(), operators, operator_report);
    const Status q8_status = validate_q8_0_matmul_kernel(
        *engine.active_backend(), operators, quantizations);
    const Status q4_status = validate_q4_0_matmul_kernel(
        *engine.active_backend(), operators, quantizations);
    const Status q4_0_embedding_status = validate_q4_0_embedding_kernel(
        *engine.active_backend(), operators);
    const Status q4_1_status = validate_q4_1_matmul_kernel(
        *engine.active_backend(), operators, quantizations);
    const Status q4_1_embedding_status = validate_q4_1_embedding_kernel(
        *engine.active_backend(), operators);
    const Status q6_k_embedding_status = validate_q6_k_embedding_kernel(
        *engine.active_backend(), operators, quantizations);
    const Status q6_k_matmul_status = validate_q6_k_matmul_kernel(
        *engine.active_backend(), operators, quantizations);

    const DeviceInfo* device = engine.active_device();
    report << ";backend=" << backend_name(engine.diagnostics().selected_backend)
           << ";operator_validation=" << (operator_status.ok() && operator_report.complete ? "COMPLETE" : "PARTIAL")
           << ";operators_validated=" << operator_report.validated.size()
           << ";operators_rejected=" << operator_report.rejected.size()
           << ";operators_unavailable=" << operator_report.unavailable.size()
           << ";q8_0_validation=" << (q8_status.ok() ? "VALIDATED" : "UNVERIFIED")
           << ";q4_0_validation=" << (q4_status.ok() ? "VALIDATED" : "UNVERIFIED")
           << ";q4_0_embedding_validation=" << (q4_0_embedding_status.ok() ? "VALIDATED" : "UNVERIFIED")
           << ";q4_1_validation=" << (q4_1_status.ok() ? "VALIDATED" : "UNVERIFIED")
           << ";q4_1_embedding_validation=" << (q4_1_embedding_status.ok() ? "VALIDATED" : "UNVERIFIED")
           << ";q6_k_embedding_validation=" << (q6_k_embedding_status.ok() ? "VALIDATED" : "UNVERIFIED")
           << ";q6_k_matmul_validation=" << (q6_k_matmul_status.ok() ? "VALIDATED" : "UNVERIFIED")
           << ";fallback_used=" << (engine.diagnostics().fallback_used ? "true" : "false")
           << ";fallback_reason=" << (engine.diagnostics().fallback_reason.empty()
                ? "none" : engine.diagnostics().fallback_reason);
    if (device) {
        report << ";device=" << (device->name.empty() ? "unknown" : device->name)
               << ";vendor=" << (device->vendor.empty() ? "unknown" : device->vendor)
               << ";architecture=" << (device->architecture.empty() ? "unknown" : device->architecture)
               << ";driver=" << (device->driver.empty() ? "unknown" : device->driver)
               << ";api=" << (device->api_version.empty() ? "unknown" : device->api_version)
               << ";compute=" << (device->compute_available ? "true" : "false")
               << ";self_test=" << (device->compute_self_test_passed ? "true" : "false")
               << ";max_buffer_bytes=" << device->limits.max_buffer_size
               << ";max_storage_binding_bytes=" << device->limits.max_storage_buffer_binding_size
               << ";max_workgroup_invocations=" << device->limits.max_compute_workgroup_invocations
               << ";max_workgroups_per_dimension=" << device->limits.max_compute_workgroups_per_dimension;
        for (const auto& feature : device->features) report << ";feature=" << feature;
    }
    report << ";inference_ready=false";
    engine.shutdown();
    return report.str();
}

} // namespace
} // namespace ai_chat::gpu_engine

extern "C" JNIEXPORT jstring JNICALL
Java_com_samanramezani_gpuengine_NativeGpuEngine_nativeGpuEngineProbe(
        JNIEnv* env, jclass) {
    std::string result;
    try {
        result = ai_chat::gpu_engine::probe_gpu_engine();
    } catch (const std::exception& exception) {
        result = std::string("runtime=AndroidGpuInferenceEngine;status=ERROR;error=") + exception.what();
    } catch (...) {
        result = "runtime=AndroidGpuInferenceEngine;status=ERROR;error=unknown";
    }
    return env->NewStringUTF(result.c_str());
}


extern "C" JNIEXPORT jint JNICALL
Java_com_samanramezani_gpuengine_NativeGpuEngine_nativeGpuEngineLoadModel(
        JNIEnv* env, jclass, jstring path, jint context_length, jlong memory_budget_bytes) {
    try {
        if (context_length <= 0 || memory_budget_bytes <= 0)
            return static_cast<jint>(ai_chat::gpu_engine::StatusCode::InvalidArgument);
        const std::string model_path = ai_chat::gpu_engine::from_java_string(env, path);
        const auto status = ai_chat::gpu_engine::gpu_inference_session().load_model(
            model_path, static_cast<uint32_t>(context_length),
            static_cast<uint64_t>(memory_budget_bytes));
        return status.ok() ? 0 : static_cast<jint>(status.code);
    } catch (...) {
        return static_cast<jint>(ai_chat::gpu_engine::StatusCode::BackendFailure);
    }
}

extern "C" JNIEXPORT jint JNICALL
Java_com_samanramezani_gpuengine_NativeGpuEngine_nativeGpuEngineUpdateMemoryBudget(
        JNIEnv*, jclass, jlong allocation_cap_bytes) {
    try {
        if (allocation_cap_bytes <= 0)
            return static_cast<jint>(ai_chat::gpu_engine::StatusCode::InvalidArgument);
        const auto status = ai_chat::gpu_engine::gpu_inference_session().update_memory_budget(
            static_cast<uint64_t>(allocation_cap_bytes));
        return status.ok() ? 0 : static_cast<jint>(status.code);
    } catch (...) {
        return static_cast<jint>(ai_chat::gpu_engine::StatusCode::BackendFailure);
    }
}

extern "C" JNIEXPORT jint JNICALL
Java_com_samanramezani_gpuengine_NativeGpuEngine_nativeGpuEngineGenerate(
        JNIEnv* env, jclass, jstring prompt, jint max_tokens, jfloat temperature,
        jint top_k, jfloat top_p, jfloat min_p, jobject listener) {
    try {
        if (max_tokens <= 0)
            return static_cast<jint>(ai_chat::gpu_engine::StatusCode::InvalidArgument);
        auto& session = ai_chat::gpu_engine::gpu_inference_session();
        const bool qwen35_tokenizer = session.uses_qwen35_tokenizer();
        const bool nfc_tokenizer = session.uses_nfc_tokenizer();
        const std::string prompt_text = nfc_tokenizer
            ? ai_chat::gpu_engine::normalized_java_text(env, prompt)
            : ai_chat::gpu_engine::from_java_string(env, prompt);
        std::unique_ptr<ai_chat::gpu_engine::ScopedQwen35UnicodeCategories> unicode_categories;
        if (qwen35_tokenizer) {
            unicode_categories = std::make_unique<ai_chat::gpu_engine::ScopedQwen35UnicodeCategories>(
                env, prompt_text);
        }
        ai_chat::gpu_engine::SamplingConfig sampling;
        sampling.temperature = temperature;
        sampling.top_k = top_k > 0 ? static_cast<uint32_t>(top_k) : 0u;
        sampling.top_p = top_p;
        sampling.min_p = min_p;
        sampling.repetition_penalty = 1.0f;

        jclass listener_class = listener ? env->GetObjectClass(listener) : nullptr;
        jmethodID on_token = listener_class
            ? env->GetMethodID(listener_class, "onToken", "(Ljava/lang/String;)V") : nullptr;
        if (listener && !on_token) return static_cast<jint>(ai_chat::gpu_engine::StatusCode::InvalidArgument);

        bool callback_failed = false;
        std::string utf8_pending;
        const auto callback = [env, listener, on_token, &callback_failed, &utf8_pending](const std::string& text) {
            if (!listener || !on_token || callback_failed) return;
            utf8_pending += text;
            const size_t complete = ai_chat::gpu_engine::complete_utf8_prefix(utf8_pending);
            if (complete == 0) return;
            jstring chunk = ai_chat::gpu_engine::to_java_string(env, utf8_pending.substr(0, complete));
            utf8_pending.erase(0, complete);
            if (!chunk) { callback_failed = true; return; }
            env->CallVoidMethod(listener, on_token, chunk);
            env->DeleteLocalRef(chunk);
            if (env->ExceptionCheck()) {
                env->ExceptionClear();
                callback_failed = true;
            }
        };
        const auto status = session.generate(
            prompt_text, static_cast<uint32_t>(max_tokens), sampling, callback);
        if (!utf8_pending.empty() && listener && on_token && !callback_failed) {
            jstring tail = ai_chat::gpu_engine::to_java_string(env, utf8_pending);
            if (tail) {
                env->CallVoidMethod(listener, on_token, tail);
                env->DeleteLocalRef(tail);
                if (env->ExceptionCheck()) { env->ExceptionClear(); callback_failed = true; }
            } else callback_failed = true;
        }
        if (callback_failed) return static_cast<jint>(ai_chat::gpu_engine::StatusCode::BackendFailure);
        if (status.code == ai_chat::gpu_engine::StatusCode::Cancelled) return 9;
        return status.ok() ? 0 : static_cast<jint>(status.code);
    } catch (...) {
        return static_cast<jint>(ai_chat::gpu_engine::StatusCode::BackendFailure);
    }
}

extern "C" JNIEXPORT jint JNICALL
Java_com_samanramezani_gpuengine_NativeGpuEngine_nativeGpuEngineCountTokens(
        JNIEnv* env, jclass, jstring text) {
    try {
        uint32_t count = 0;
        auto& session = ai_chat::gpu_engine::gpu_inference_session();
        const bool qwen35_tokenizer = session.uses_qwen35_tokenizer();
        const bool nfc_tokenizer = session.uses_nfc_tokenizer();
        const std::string input_text = nfc_tokenizer
            ? ai_chat::gpu_engine::normalized_java_text(env, text)
            : ai_chat::gpu_engine::from_java_string(env, text);
        std::unique_ptr<ai_chat::gpu_engine::ScopedQwen35UnicodeCategories> unicode_categories;
        if (qwen35_tokenizer) {
            unicode_categories = std::make_unique<ai_chat::gpu_engine::ScopedQwen35UnicodeCategories>(
                env, input_text);
        }
        const auto status = session.count_tokens(input_text, count);
        return status.ok() ? static_cast<jint>(count) : -1;
    } catch (...) {
        return -1;
    }
}

extern "C" JNIEXPORT jint JNICALL
Java_com_samanramezani_gpuengine_NativeGpuEngine_nativeGpuEngineContextLength(
        JNIEnv*, jclass) {
    try {
        const std::string diagnostics = ai_chat::gpu_engine::gpu_inference_session().diagnostics();
        const std::string key = ";context_length=";
        const size_t start = diagnostics.find(key);
        if (start == std::string::npos) return 0;
        return static_cast<jint>(std::strtol(diagnostics.c_str() + start + key.size(), nullptr, 10));
    } catch (...) {
        return 0;
    }
}

extern "C" JNIEXPORT void JNICALL
Java_com_samanramezani_gpuengine_NativeGpuEngine_nativeGpuEngineStop(
        JNIEnv*, jclass) {
    ai_chat::gpu_engine::gpu_inference_session().cancel();
}

extern "C" JNIEXPORT void JNICALL
Java_com_samanramezani_gpuengine_NativeGpuEngine_nativeGpuEngineUnload(
        JNIEnv*, jclass) {
    ai_chat::gpu_engine::gpu_inference_session().unload();
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_samanramezani_gpuengine_NativeGpuEngine_nativeGpuEngineDiagnostics(
        JNIEnv* env, jclass) {
    std::string result;
    try {
        result = ai_chat::gpu_engine::gpu_inference_session().diagnostics();
    } catch (const std::exception& exception) {
        result = std::string("runtime=AndroidGpuInferenceEngine;status=ERROR;error=") + exception.what();
    } catch (...) {
        result = "runtime=AndroidGpuInferenceEngine;status=ERROR;error=unknown";
    }
    return env->NewStringUTF(result.c_str());
}

extern "C" JNIEXPORT void JNICALL
Java_com_samanramezani_gpuengine_NativeGpuEngine_nativeGpuEngineResetStop(
        JNIEnv*, jclass) {
    ai_chat::gpu_engine::gpu_inference_session().reset_stop();
}
