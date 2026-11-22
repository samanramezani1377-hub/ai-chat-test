#include "runtime_router.h"

namespace ai_chat::gpu_engine {

RuntimeSelectionReport select_runtime(const RuntimeReadiness& readiness) {
    RuntimeSelectionReport report;
    report.cpu_fallback_available = readiness.cpu_fallback_available;
    if (!readiness.gpu_device_ready) report.blockers.emplace_back("GPU_DEVICE_NOT_READY");
    if (!readiness.tokenizer_validated) report.blockers.emplace_back("TOKENIZER_NOT_VALIDATED");
    if (!readiness.architecture_validated) report.blockers.emplace_back("ARCHITECTURE_NOT_VALIDATED");
    if (!readiness.quantization_validated) report.blockers.emplace_back("QUANTIZATION_NOT_VALIDATED");
    if (!readiness.operator_coverage_validated) report.blockers.emplace_back("OPERATOR_COVERAGE_NOT_VALIDATED");
    if (!readiness.model_graph_validated) report.blockers.emplace_back("MODEL_GRAPH_NOT_VALIDATED");
    if (!readiness.memory_preflight_passed) report.blockers.emplace_back("GPU_MEMORY_PREFLIGHT_FAILED");

    const bool ready = report.blockers.empty();
    if (ready) {
        report.selected = RuntimeSelection::DedicatedGpuEngine;
        report.gpu_selected = true;
    } else if (readiness.cpu_fallback_available) {
        report.selected = RuntimeSelection::CpuLlamaCpp;
        report.gpu_selected = false;
    } else {
        report.selected = RuntimeSelection::Unavailable;
        report.gpu_selected = false;
    }
    return report;
}

} // namespace ai_chat::gpu_engine
