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

std::string from_java_string(JNIEnv* env, jstring value) {
    if (!value) return {};
    const char* chars = env->GetStringUTFChars(value, nullptr);
    if (!chars) return {};
    std::string result(chars);
    env->ReleaseStringUTFChars(value, chars);
    return result;
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
    report << "runtime=AIChatGpuEngine"
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
        result = std::string("runtime=AIChatGpuEngine;status=ERROR;error=") + exception.what();
    } catch (...) {
        result = "runtime=AIChatGpuEngine;status=ERROR;error=unknown";
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
Java_com_samanramezani_gpuengine_NativeGpuEngine_nativeGpuEngineGenerate(
        JNIEnv* env, jclass, jstring prompt, jint max_tokens, jfloat temperature,
        jint top_k, jfloat top_p, jfloat min_p, jobject listener) {
    try {
        if (max_tokens <= 0)
            return static_cast<jint>(ai_chat::gpu_engine::StatusCode::InvalidArgument);
        const std::string prompt_text = ai_chat::gpu_engine::from_java_string(env, prompt);
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
        const auto callback = [env, listener, on_token, &callback_failed](const std::string& text) {
            if (!listener || !on_token || callback_failed) return;
            jstring chunk = env->NewStringUTF(text.c_str());
            if (!chunk) { callback_failed = true; return; }
            env->CallVoidMethod(listener, on_token, chunk);
            env->DeleteLocalRef(chunk);
            if (env->ExceptionCheck()) {
                env->ExceptionClear();
                callback_failed = true;
            }
        };
        const auto status = ai_chat::gpu_engine::gpu_inference_session().generate(
            prompt_text, static_cast<uint32_t>(max_tokens), sampling, callback);
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
        const auto status = ai_chat::gpu_engine::gpu_inference_session().count_tokens(
            ai_chat::gpu_engine::from_java_string(env, text), count);
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
        result = std::string("runtime=AIChatGpuEngine;status=ERROR;error=") + exception.what();
    } catch (...) {
        result = "runtime=AIChatGpuEngine;status=ERROR;error=unknown";
    }
    return env->NewStringUTF(result.c_str());
}

extern "C" JNIEXPORT void JNICALL
Java_com_samanramezani_gpuengine_NativeGpuEngine_nativeGpuEngineResetStop(
        JNIEnv*, jclass) {
    ai_chat::gpu_engine::gpu_inference_session().reset_stop();
}
