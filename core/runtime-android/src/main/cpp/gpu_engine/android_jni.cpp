#include <jni.h>

#include "android_backends.h"
#include "execution_engine.h"
#include "model_lowerings.h"
#include "operators.h"
#include "model_registry.h"
#include "quantization.h"
#include "tokenizer.h"

#include <exception>
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

    const DeviceInfo* device = engine.active_device();
    report << ";backend=" << backend_name(engine.diagnostics().selected_backend)
           << ";operator_validation=" << (operator_status.ok() && operator_report.complete ? "COMPLETE" : "PARTIAL")
           << ";operators_validated=" << operator_report.validated.size()
           << ";operators_rejected=" << operator_report.rejected.size()
           << ";operators_unavailable=" << operator_report.unavailable.size()
           << ";q8_0_validation=" << (q8_status.ok() ? "VALIDATED" : "UNVERIFIED")
           << ";q4_0_validation=" << (q4_status.ok() ? "VALIDATED" : "UNVERIFIED")
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
Java_com_woogit_aicore_runtime_android_NativeLlamaCpp_nativeGpuEngineProbe(
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
