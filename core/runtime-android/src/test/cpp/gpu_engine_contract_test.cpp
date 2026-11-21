#include "execution_engine.h"
#include "gguf_reader.h"
#include "model_registry.h"
#include "operators.h"
#include "weight_streamer.h"
#include "sampler.h"
#include "quantization.h"
#include "kv_cache.h"
#include "runtime_router.h"
#include "memory_planner.h"
#include "tokenizer.h"
#include "model_graph.h"

#include <algorithm>
#include <cmath>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <fstream>
#include <cstdio>
#include <memory>
#include <string>
#include <utility>
#include <vector>

using namespace ai_chat::gpu_engine;

namespace {

class FakeBuffer final : public Buffer {
public:
    explicit FakeBuffer(const BufferDesc& desc)
        : bytes(desc.size_bytes), domain(desc.memory) {}
    uint64_t size_bytes() const noexcept override { return bytes.size(); }
    MemoryDomain memory_domain() const noexcept override { return domain; }
    std::vector<uint8_t> bytes;
    MemoryDomain domain;
};

class FakeBatch final : public CommandBatch {
public:
    Status copy_buffer(const std::shared_ptr<Buffer>& source, uint64_t source_offset,
                       const std::shared_ptr<Buffer>& destination, uint64_t destination_offset,
                       uint64_t size_bytes) override {
        auto src = std::dynamic_pointer_cast<FakeBuffer>(source);
        auto dst = std::dynamic_pointer_cast<FakeBuffer>(destination);
        if (!src || !dst || source_offset + size_bytes > src->bytes.size() ||
            destination_offset + size_bytes > dst->bytes.size()) {
            return {StatusCode::InvalidArgument, "fake copy range invalid"};
        }
        std::memcpy(dst->bytes.data() + destination_offset,
                    src->bytes.data() + source_offset, static_cast<size_t>(size_bytes));
        return Status::success();
    }

    Status dispatch_compute(const ComputeDispatch& dispatch) override {
        if (dispatch.label.empty()) return {StatusCode::InvalidArgument, "dispatch label missing"};
        labels.push_back(dispatch.label);
        return Status::success();
    }

    std::vector<std::string> labels;
};

class FakeBackend final : public Backend {
public:
    explicit FakeBackend(BackendKind backend_kind = BackendKind::OpenGles, bool available = true)
        : backend_kind_(backend_kind), available_(available) {}
    BackendKind kind() const noexcept override { return backend_kind_; }
    Status enumerate_devices(std::vector<DeviceInfo>& out) override {
        out.clear();
        if (!available_) return {StatusCode::Unsupported, "fake backend intentionally unavailable"};
        DeviceInfo device;
        device.backend = backend_kind_;
        device.name = "fake-test-device";
        device.compute_available = true;
        out.push_back(device);
        return Status::success();
    }
    Status initialize(const DeviceInfo&) override { ready = true; return Status::success(); }
    Status allocate(const BufferDesc& desc, std::shared_ptr<Buffer>& out) override {
        out = std::make_shared<FakeBuffer>(desc);
        return Status::success();
    }
    Status begin_commands(std::unique_ptr<CommandBatch>& out) override {
        out = std::make_unique<FakeBatch>();
        return Status::success();
    }
    Status submit(std::unique_ptr<CommandBatch> commands) override {
        auto* batch = dynamic_cast<FakeBatch*>(commands.get());
        if (!batch) return {StatusCode::InvalidArgument, "wrong fake command batch"};
        submitted = batch->labels;
        ++submit_count;
        return Status::success();
    }
    Status synchronize() override { ++sync_count; return Status::success(); }
    Status upload(const std::shared_ptr<Buffer>& destination, uint64_t offset, const void* source, uint64_t size) override {
        auto buffer = std::dynamic_pointer_cast<FakeBuffer>(destination);
        if (!buffer || !source || offset + size > buffer->bytes.size()) return {StatusCode::InvalidArgument, "fake upload range invalid"};
        std::memcpy(buffer->bytes.data() + offset, source, static_cast<size_t>(size));
        return Status::success();
    }
    Status readback(const std::shared_ptr<Buffer>& source, uint64_t offset, void* destination, uint64_t size) override {
        auto buffer = std::dynamic_pointer_cast<FakeBuffer>(source);
        if (!buffer || !destination || offset + size > buffer->bytes.size()) return {StatusCode::InvalidArgument, "fake readback range invalid"};
        std::memcpy(destination, buffer->bytes.data() + offset, static_cast<size_t>(size));
        return Status::success();
    }
    Status shutdown() override { ready = false; return Status::success(); }
    bool healthy() const noexcept override { return ready; }
    BackendKind backend_kind_;
    bool available_;
    bool ready = true;
    int submit_count = 0;
    int sync_count = 0;
    std::vector<std::string> submitted;
};

void write_u32(std::ofstream& out, uint32_t value) {
    for (int i = 0; i < 4; ++i) out.put(static_cast<char>((value >> (i * 8)) & 0xff));
}
void write_u64(std::ofstream& out, uint64_t value) {
    for (int i = 0; i < 8; ++i) out.put(static_cast<char>((value >> (i * 8)) & 0xff));
}
void write_string(std::ofstream& out, const std::string& value) {
    write_u64(out, value.size());
    out.write(value.data(), static_cast<std::streamsize>(value.size()));
}
void test_independent_gguf_reader() {
    const std::string path = "/tmp/ai_chat_gpu_engine_minimal.gguf";
    {
        std::ofstream out(path, std::ios::binary | std::ios::trunc);
        out.write("GGUF", 4);
        write_u32(out, 3);
        write_u64(out, 1); // tensor count
        write_u64(out, 5); // metadata count
        write_string(out, "general.architecture");
        write_u32(out, 8); write_string(out, "llama");
        write_string(out, "general.alignment");
        write_u32(out, 4); write_u32(out, 32);
        write_string(out, "llama.context_length");
        write_u32(out, 4); write_u32(out, 4096);
        write_string(out, "tokenizer.ggml.model");
        write_u32(out, 8); write_string(out, "gpt2");
        write_string(out, "tokenizer.ggml.tokens");
        write_u32(out, 9); write_u32(out, 8); write_u64(out, 2);
        write_string(out, "hello"); write_string(out, "world");
        write_string(out, "test.weight");
        write_u32(out, 2);
        write_u64(out, 2); write_u64(out, 3);
        write_u32(out, 0); // F32
        write_u64(out, 0); // relative data offset
        const auto pos = static_cast<uint64_t>(out.tellp());
        const auto data_start = (pos + 31u) & ~31ULL;
        for (uint64_t i = pos; i < data_start; ++i) out.put('\0');
        const float values[] = {1, 2, 3, 4, 5, 6};
        out.write(reinterpret_cast<const char*>(values), sizeof(values));
    }

    GgufReader reader;
    const Status status = GgufReader::open(path, reader);
    assert(status.ok());
    assert(reader.info().version == 3);
    assert(reader.info().architecture == "llama");
    assert(reader.info().alignment == 32);
    assert(reader.info().tokenizer.model == "gpt2");
    assert(reader.info().tokenizer.tokens.size() == 2);
    assert(reader.info().tensors.size() == 1);
    const auto* tensor = reader.find_tensor("test.weight");
    assert(tensor && tensor->byte_size_known && tensor->byte_size == 24);
    float values[6] = {};
    assert(reader.read_tensor_chunk(*tensor, 0, values, sizeof(values)).ok());
    assert(values[0] == 1.0f && values[5] == 6.0f);
    std::remove(path.c_str());
}

void test_tensor_and_memory_lease() {
    FakeBackend backend;
    MemoryManager memory(backend, 128);
    BufferDesc allocation{64, 4, MemoryDomain::Device, false, "test"};
    MemoryLease lease;
    assert(memory.allocate(allocation, lease).ok());
    assert(memory.allocated_bytes() == 64);
    auto backing = lease.buffer();
    Buffer* original_buffer = backing.get();

    TensorDesc desc;
    desc.shape = {2, 3};
    desc.data_type = DataType::F32;
    desc.memory = MemoryDomain::Device;
    desc.alignment = 4;
    Tensor tensor;
    assert(Tensor::create(desc, backing, 0, tensor).ok());
    TensorDesc wrong_domain = desc;
    wrong_domain.memory = MemoryDomain::Host;
    Tensor rejected_tensor;
    assert(Tensor::create(wrong_domain, backing, 0, rejected_tensor).code == StatusCode::InvalidArgument);
    assert(tensor.element_count() == 6);
    assert(tensor.byte_size() == 24);

    lease = MemoryLease{};
    assert(memory.allocated_bytes() == 64); // Tensor still owns the aliasing lease
    backing.reset();
    assert(memory.allocated_bytes() == 64);
    tensor = Tensor{};
    assert(memory.allocated_bytes() == 0);
    assert(memory.available_bytes() == 64); // transient storage is retained in a bounded reusable pool
    MemoryLease reused_lease;
    assert(memory.allocate(allocation, reused_lease).ok());
    assert(reused_lease.buffer().get() == original_buffer); // best-fit pool reuse avoids a second GPU allocation
    assert(memory.allocated_bytes() == 64);

    reused_lease = MemoryLease{};
    assert(memory.allocated_bytes() == 0);
    desc.shape = {UINT64_MAX, 8};
    uint64_t bytes = 0;
    assert(Tensor::required_bytes(desc, bytes).code == StatusCode::InvalidArgument);
}


class FakeArchitecture final : public ArchitectureAdapter {
public:
    FakeArchitecture() {
        descriptor_.name = "test-architecture";
        descriptor_.family = ArchitectureFamily::DenseTransformer;
        descriptor_.validation = SupportState::Validated;
    }
    const ArchitectureDescriptor& descriptor() const noexcept override { return descriptor_; }
    Status validate_metadata(const GgufFileInfo&) const override { return Status::success(); }
    Status build_execution_plan(const GgufFileInfo&, ModelExecutionPlan&) const override {
        return {StatusCode::Unsupported, "test adapter has no execution plan"};
    }
private:
    ArchitectureDescriptor descriptor_;
};









void test_gpu_memory_planner_respects_streaming_and_reserve() {
    GpuMemoryPlanInput input;
    input.total_weight_bytes = 8ull * 1024 * 1024 * 1024;
    input.streaming_chunk_bytes = 8ull * 1024 * 1024;
    input.kv_cache_bytes = 128ull * 1024 * 1024;
    input.scratch_bytes = 32ull * 1024 * 1024;
    input.safety_reserve_bytes = 64ull * 1024 * 1024;
    input.device_budget_bytes = 512ull * 1024 * 1024;
    input.stream_weights = true;
    auto report = plan_gpu_memory(input);
    assert(report.fits);
    assert(report.resident_weight_bytes == input.streaming_chunk_bytes);
    assert(report.required_bytes == input.streaming_chunk_bytes +
           input.kv_cache_bytes + input.scratch_bytes);

    input.stream_weights = false;
    report = plan_gpu_memory(input);
    assert(!report.fits);
    assert(!report.blockers.empty());

    input.stream_weights = true;
    input.device_budget_bytes = 0;
    report = plan_gpu_memory(input);
    assert(!report.fits);
    assert(report.blockers.front() == "GPU_MEMORY_BUDGET_UNKNOWN");
}

void test_gpu_backend_fallback_order() {
    GpuEngine engine;
    assert(engine.register_backend(std::make_shared<FakeBackend>(BackendKind::DawnWebGpu, false)).ok());
    assert(engine.register_backend(std::make_shared<FakeBackend>(BackendKind::OpenGles, true)).ok());
    assert(engine.initialize_with_fallback({BackendKind::DawnWebGpu, BackendKind::OpenGles}).ok());
    assert(engine.ready());
    assert(engine.active_device() && engine.active_device()->backend == BackendKind::OpenGles);
    assert(engine.diagnostics().fallback_used);
    assert(!engine.diagnostics().fallback_reason.empty());
    assert(engine.shutdown().ok());
}
void test_runtime_router_fails_closed_to_cpu() {
    RuntimeReadiness readiness;
    auto report = select_runtime(readiness);
    assert(report.selected == RuntimeSelection::CpuLlamaCpp);
    assert(!report.gpu_selected);
    assert(report.cpu_fallback_available);
    assert(report.blockers.size() == 7);

    readiness.gpu_device_ready = true;
    readiness.tokenizer_validated = true;
    readiness.architecture_validated = true;
    readiness.quantization_validated = true;
    readiness.operator_coverage_validated = true;
    readiness.model_graph_validated = true;
    readiness.memory_preflight_passed = true;
    report = select_runtime(readiness);
    assert(report.selected == RuntimeSelection::DedicatedGpuEngine);
    assert(report.gpu_selected);
    assert(report.blockers.empty());

    readiness.cpu_fallback_available = false;
    report = select_runtime(readiness);
    assert(report.gpu_selected);
    assert(!report.cpu_fallback_available);

    readiness.gpu_device_ready = false;
    report = select_runtime(readiness);
    assert(report.selected == RuntimeSelection::Unavailable);
    assert(!report.gpu_selected);
}

void test_kv_cache_memory_and_sequential_append() {
    FakeBackend backend;
    MemoryManager memory(backend, 512);
    KvCacheConfig config;
    config.layers = 2;
    config.kv_heads = 2;
    config.head_dimension = 2;
    config.maximum_sequence_length = 4;
    config.data_type = DataType::F32;
    uint64_t estimate = 0;
    assert(KvCache::estimate_bytes(config, estimate).ok());
    assert(estimate == 256);
    std::unique_ptr<KvCache> cache;
    assert(KvCache::create(backend, memory, config, cache).ok());
    assert(cache && cache->total_bytes() == estimate);
    assert(memory.allocated_bytes() == estimate);
    assert(cache->bytes_per_token() == 16);

    const float keys[] = {1, 2, 3, 4, 5, 6, 7, 8};
    const float values[] = {9, 10, 11, 12, 13, 14, 15, 16};
    assert(cache->append(0, 0, 2, keys, values).ok());
    assert(cache->valid_tokens(0) == 2);
    assert(cache->valid_tokens(1) == 0);
    assert(cache->append(0, 1, 1, keys, values).code == StatusCode::InvalidArgument);
    assert(cache->append(1, 0, 2, keys, values).ok());
    float readback[8] = {};
    assert(backend.readback(cache->key_buffer(0), 0, readback, sizeof(readback)).ok());
    assert(readback[0] == 1.0f && readback[7] == 8.0f);
    assert(cache->reset_layer(0).ok());
    assert(cache->valid_tokens(0) == 0);
    cache.reset();
    assert(memory.allocated_bytes() == 0);

    config.data_type = DataType::I8;
    assert(KvCache::estimate_bytes(config, estimate).code == StatusCode::InvalidArgument);
}



void test_q5_decoders() {
    std::vector<uint8_t> q5_0(22, 0);
    q5_0[0] = 0x00; q5_0[1] = 0x38; // d = 0.5
    q5_0[2] = 0x01; q5_0[4] = 0x01; // high bits for q[0] and q[16]
    q5_0[6] = 0x10; // low nibble 0, high nibble 1
    float q5_0_output[32] = {};
    assert(decode_q5_0_blocks(q5_0.data(), q5_0.size(), q5_0_output, 32).ok());
    assert(q5_0_output[0] == 0.0f && q5_0_output[16] == 0.5f);

    std::vector<uint8_t> q5_1(24, 0);
    q5_1[0] = 0x00; q5_1[1] = 0x38; // d = 0.5
    q5_1[2] = 0x00; q5_1[3] = 0x3c; // m = 1.0
    q5_1[4] = 0x01; q5_1[6] = 0x01;
    q5_1[8] = 0x10;
    float q5_1_output[32] = {};
    assert(decode_q5_1_blocks(q5_1.data(), q5_1.size(), q5_1_output, 32).ok());
    assert(q5_1_output[0] == 9.0f && q5_1_output[16] == 9.5f);

    QuantizationRegistry registry;
    assert(register_builtin_quantization_formats(registry).ok());
    assert(registry.can_decode(6) && registry.can_decode(7));
    assert(!registry.can_execute_on_gpu(6) && !registry.can_execute_on_gpu(7));
}

void test_q4_decoders() {
    std::vector<uint8_t> q4_0(18, 0);
    q4_0[0] = 0x00; q4_0[1] = 0x38; // d = 0.5
    for (size_t i = 0; i < 16; ++i) q4_0[2 + i] = 0x98; // low=8, high=9
    float q4_0_output[32] = {};
    assert(decode_q4_0_blocks(q4_0.data(), q4_0.size(), q4_0_output, 32).ok());
    assert(q4_0_output[0] == 0.0f && q4_0_output[16] == 0.5f);

    std::vector<uint8_t> q4_1(20, 0);
    q4_1[0] = 0x00; q4_1[1] = 0x38; // d = 0.5
    q4_1[2] = 0x00; q4_1[3] = 0x3c; // m = 1.0
    for (size_t i = 0; i < 16; ++i) q4_1[4 + i] = 0xf0;
    float q4_1_output[32] = {};
    assert(decode_q4_1_blocks(q4_1.data(), q4_1.size(), q4_1_output, 32).ok());
    assert(q4_1_output[0] == 1.0f && q4_1_output[16] == 8.5f);
    assert(decode_q4_1_blocks(q4_1.data(), q4_1.size() - 1, q4_1_output, 32).code ==
           StatusCode::InvalidArgument);

    QuantizationRegistry registry;
    assert(register_builtin_quantization_formats(registry).ok());
    assert(registry.can_decode(2) && registry.can_decode(3) && registry.can_decode(6) &&
           registry.can_decode(7) && registry.can_decode(8));
    assert(!registry.can_execute_on_gpu(2) && !registry.can_execute_on_gpu(3) &&
           !registry.can_execute_on_gpu(8));
}


void test_dawn_q8_0_operator_registration_is_unverified() {
    OperatorRegistry registry;
    assert(register_builtin_quantized_operator_candidates(registry).ok());
    KernelDefinition selected;
    assert(registry.select_kernel("MatMul_Q8_0", BackendKind::DawnWebGpu,
        std::vector<DataType>{DataType::F32, DataType::U32}, DataType::F32, 2,
        selected).code == StatusCode::Unsupported);
}


void test_q8_0_gpu_validation_does_not_promote_fake_backend() {
    FakeBackend backend;
    OperatorRegistry operators;
    QuantizationRegistry quantizations;
    assert(register_builtin_gles_operator_candidates(operators).ok());
    assert(register_builtin_quantization_formats(quantizations).ok());
    assert(validate_q8_0_matmul_kernel(backend, operators, quantizations).code == StatusCode::NumericalFailure);
    const QuantizationFormat* format = quantizations.find(8);
    assert(format != nullptr && format->gpu_kernel_state == SupportState::Unverified);
    KernelDefinition selected;
    assert(operators.select_kernel("MatMul_Q8_0", BackendKind::OpenGles,
        std::vector<DataType>{DataType::F32, DataType::U32}, DataType::F32, 2,
        selected).code == StatusCode::Unsupported);
}

void test_q8_0_decoder_and_registry() {
    std::vector<uint8_t> encoded(34, 0);
    encoded[0] = 0x00;
    encoded[1] = 0x38; // FP16 0.5
    encoded[2] = static_cast<uint8_t>(-2);
    encoded[3] = static_cast<uint8_t>(-1);
    encoded[4] = 0;
    encoded[5] = 1;
    encoded[6] = 2;
    float decoded[32] = {};
    assert(decode_q8_0_blocks(encoded.data(), encoded.size(), decoded, 32).ok());
    assert(decoded[0] == -1.0f && decoded[1] == -0.5f &&
           decoded[2] == 0.0f && decoded[3] == 0.5f && decoded[4] == 1.0f);
    assert(decode_q8_0_blocks(encoded.data(), encoded.size() - 1, decoded, 32).code ==
           StatusCode::InvalidArgument);

    encoded[0] = 0x00; encoded[1] = 0x7e; // NaN scale
    assert(decode_q8_0_blocks(encoded.data(), encoded.size(), decoded, 32).code ==
           StatusCode::NumericalFailure);

    QuantizationRegistry registry;
    assert(register_builtin_quantization_formats(registry).ok());
    assert(registry.can_decode(8));
    assert(!registry.can_execute_on_gpu(8));

    FakeBackend backend;
    BufferDesc encoded_desc{36, 4, MemoryDomain::Device, false, "q8-encoded"};
    std::shared_ptr<Buffer> encoded_buffer;
    assert(backend.allocate(encoded_desc, encoded_buffer).ok());
    BufferDesc output_desc{128, 4, MemoryDomain::Device, false, "q8-output"};
    std::shared_ptr<Buffer> output_buffer;
    assert(backend.allocate(output_desc, output_buffer).ok());
    TensorDesc output_desc_tensor;
    output_desc_tensor.shape = {32};
    output_desc_tensor.data_type = DataType::F32;
    Tensor output_tensor;
    assert(Tensor::create(output_desc_tensor, output_buffer, 0, output_tensor).ok());
    ComputeDispatch dispatch;
    assert(build_q8_0_dequantize_candidate(encoded_buffer, 34, output_tensor, dispatch).ok());
    assert(dispatch.bindings.size() == 2);
    assert(dispatch.label.find("unverified") != std::string::npos);
    OperatorRegistry gles_registry;
    assert(register_builtin_gles_operator_candidates(gles_registry).ok());
    assert(dispatch.wgsl_source.find("const ELEMENT_COUNT: u32 = 32u;") != std::string::npos);

    BufferDesc q4_encoded_desc{20, 4, MemoryDomain::Device, false, "q4-encoded"};
    std::shared_ptr<Buffer> q4_encoded;
    assert(backend.allocate(q4_encoded_desc, q4_encoded).ok());
    BufferDesc q4_output_desc{128, 4, MemoryDomain::Device, false, "q4-output"};
    std::shared_ptr<Buffer> q4_output_buffer;
    assert(backend.allocate(q4_output_desc, q4_output_buffer).ok());
    Tensor q4_output_tensor;
    assert(Tensor::create(output_desc_tensor, q4_output_buffer, 0, q4_output_tensor).ok());
    ComputeDispatch q4_dispatch;
    assert(build_q4_0_dequantize_candidate(q4_encoded, 18, q4_output_tensor, q4_dispatch).ok());
    assert(q4_dispatch.bindings.size() == 2);
    assert(q4_dispatch.label.find("unverified") != std::string::npos);

    BufferDesc q4_1_encoded_desc{20, 4, MemoryDomain::Device, false, "q4-1-encoded"};
    std::shared_ptr<Buffer> q4_1_encoded;
    assert(backend.allocate(q4_1_encoded_desc, q4_1_encoded).ok());
    ComputeDispatch q4_1_dispatch;
    assert(build_q4_1_dequantize_candidate(q4_1_encoded, 20, q4_output_tensor, q4_1_dispatch).ok());
    assert(q4_1_dispatch.bindings.size() == 2);
    assert(q4_1_dispatch.label.find("unverified") != std::string::npos);

    BufferDesc q5_encoded_desc{24, 4, MemoryDomain::Device, false, "q5-encoded"};
    std::shared_ptr<Buffer> q5_encoded;
    assert(backend.allocate(q5_encoded_desc, q5_encoded).ok());
    ComputeDispatch q5_0_dispatch, q5_1_dispatch;
    assert(build_q5_0_dequantize_candidate(q5_encoded, 22, q4_output_tensor, q5_0_dispatch).ok());
    assert(build_q5_1_dequantize_candidate(q5_encoded, 24, q4_output_tensor, q5_1_dispatch).ok());
    assert(q5_0_dispatch.bindings.size() == 2 && q5_1_dispatch.bindings.size() == 2);

    std::vector<uint8_t> q8_weights(68, 0);
    q8_weights[0] = 0x00; q8_weights[1] = 0x38;
    q8_weights[34] = 0x00; q8_weights[35] = 0x38;
    for (size_t i = 0; i < 32; ++i) {
        q8_weights[2 + i] = 1;
        q8_weights[36 + i] = 2;
    }
    float activation_values[32];
    for (float& value : activation_values) value = 1.0f;
    float q8_matmul_output[2] = {};
    assert(reference_q8_0_matmul(activation_values, q8_weights.data(), q8_weights.size(),
                                 q8_matmul_output, 1, 2, 32).ok());
    assert(q8_matmul_output[0] == 16.0f && q8_matmul_output[1] == 32.0f);

    BufferDesc q8_weights_desc{68, 4, MemoryDomain::Device, false, "q8-matmul-weights"};
    std::shared_ptr<Buffer> q8_weights_buffer;
    assert(backend.allocate(q8_weights_desc, q8_weights_buffer).ok());
    BufferDesc activation_desc{128, 4, MemoryDomain::Device, false, "q8-matmul-activations"};
    std::shared_ptr<Buffer> activation_buffer;
    assert(backend.allocate(activation_desc, activation_buffer).ok());
    BufferDesc output_mat_desc{8, 4, MemoryDomain::Device, false, "q8-matmul-output"};
    std::shared_ptr<Buffer> output_mat_buffer;
    assert(backend.allocate(output_mat_desc, output_mat_buffer).ok());
    TensorDesc activation_tensor_desc;
    activation_tensor_desc.shape = {1, 32};
    activation_tensor_desc.data_type = DataType::F32;
    Tensor activation_tensor, output_mat_tensor;
    assert(Tensor::create(activation_tensor_desc, activation_buffer, 0, activation_tensor).ok());
    TensorDesc output_mat_tensor_desc;
    output_mat_tensor_desc.shape = {1, 2};
    output_mat_tensor_desc.data_type = DataType::F32;
    assert(Tensor::create(output_mat_tensor_desc, output_mat_buffer, 0, output_mat_tensor).ok());
    ComputeDispatch q8_matmul_dispatch;
    assert(build_q8_0_matmul_candidate(activation_tensor, q8_weights_buffer, 68,
                                       output_mat_tensor, q8_matmul_dispatch).ok());
    assert(q8_matmul_dispatch.bindings.size() == 3);
    assert(q8_matmul_dispatch.label.find("unverified") != std::string::npos);
    ComputeDispatch gles_q8_matmul_dispatch;
    assert(build_q8_0_matmul_gles_candidate(activation_tensor, q8_weights_buffer, 68,
                                             output_mat_tensor, gles_q8_matmul_dispatch).ok());
    assert(gles_q8_matmul_dispatch.backend == BackendKind::OpenGles);
    assert(gles_q8_matmul_dispatch.workgroup_size_x == 8 &&
           gles_q8_matmul_dispatch.workgroup_size_y == 8);
    assert(gles_q8_matmul_dispatch.glsl_source.find("const uint WEIGHT_ROW_BYTES = 34u;") != std::string::npos);
    KernelDefinition q8_gles_unverified;
    assert(gles_registry.select_kernel("MatMul_Q8_0", BackendKind::OpenGles,
        std::vector<DataType>{DataType::F32, DataType::U32}, DataType::F32, 2,
        q8_gles_unverified).code == StatusCode::Unsupported);
}

class FakeTokenizerAdapter final : public TokenizerAdapter {
public:
    FakeTokenizerAdapter() {
        descriptor_.model = "test-bpe";
        descriptor_.pre_tokenizer = "test-pretokenizer";
        descriptor_.algorithm = "test-only";
        descriptor_.validation = SupportState::Validated;
    }
    const TokenizerDescriptor& descriptor() const noexcept override { return descriptor_; }
    Status validate_metadata(const GgufTokenizerInfo& metadata) const override {
        return metadata.tokens.size() == 2 ? Status::success()
            : Status{StatusCode::InvalidArgument, "fake tokenizer expects exactly two tokens"};
    }
    Status encode(const GgufTokenizerInfo&, const std::string& text,
                  std::vector<uint32_t>& ids) const override {
        if (text == "a") { ids = {0}; return Status::success(); }
        if (text == "b") { ids = {1}; return Status::success(); }
        return {StatusCode::Unsupported, "fake tokenizer only accepts a or b"};
    }
    Status decode(const GgufTokenizerInfo& metadata, const std::vector<uint32_t>& ids,
                  std::string& text) const override {
        text.clear();
        for (uint32_t id : ids) text += metadata.tokens[id];
        return Status::success();
    }
private:
    TokenizerDescriptor descriptor_;
};

void test_tokenizer_engine_contracts() {
    GgufTokenizerInfo metadata;
    metadata.model = "test-bpe";
    metadata.pre_tokenizer = "test-pretokenizer";
    metadata.tokens = {"a", "b"};
    metadata.bos_token_id = 0;
    metadata.eos_token_id = 1;
    TokenizerRegistry registry;
    auto report = evaluate_tokenizer_preflight(metadata, registry);
    assert(report.metadata_valid);
    assert(!report.adapter_registered && !report.ready);
    assert(std::find(report.blockers.begin(), report.blockers.end(),
                     "TOKENIZER_ADAPTER_NOT_REGISTERED") != report.blockers.end());

    assert(registry.register_adapter(std::make_shared<FakeTokenizerAdapter>()).ok());
    report = evaluate_tokenizer_preflight(metadata, registry);
    assert(report.metadata_valid && report.adapter_registered &&
           report.adapter_validated && report.ready);
    TokenizerEngine tokenizer(registry);
    std::vector<uint32_t> ids;
    assert(tokenizer.encode(metadata, "a", ids).ok());
    assert(ids.size() == 1 && ids[0] == 0);
    std::string decoded;
    assert(tokenizer.decode(metadata, ids, decoded).ok() && decoded == "a");
    assert(tokenizer.encode(metadata, "invalid", ids).code == StatusCode::Unsupported);
    assert(ids.empty());

    metadata.bos_token_id = 9;
    report = evaluate_tokenizer_preflight(metadata, registry);
    assert(!report.metadata_valid && !report.ready);
    assert(std::find(report.blockers.begin(), report.blockers.end(),
                     "TOKENIZER_SPECIAL_TOKEN_ID_OUT_OF_RANGE") != report.blockers.end());

    TokenizerRegistry builtins;
    assert(register_builtin_tokenizer_adapters(builtins).ok());
    GgufTokenizerInfo sentencepiece;
    sentencepiece.model = "llama";
    sentencepiece.pre_tokenizer = "default";
    sentencepiece.tokens = {"<unk>", "▁", "a", "b", "▁a", "ab", "▁ab", "<s>", "</s>"};
    sentencepiece.scores = {-10.0f, -5.0f, -4.0f, -4.0f, 2.0f, 3.0f, 4.0f, 0.0f, 0.0f};
    sentencepiece.token_types = {2, 1, 1, 1, 1, 1, 1, 3, 3};
    sentencepiece.unknown_token_id = 0;
    sentencepiece.bos_token_id = 7;
    sentencepiece.eos_token_id = 8;
    sentencepiece.add_bos_token = true;
    sentencepiece.add_eos_token = true;
    sentencepiece.add_space_prefix = true;
    auto spm = builtins.find(sentencepiece);
    assert(spm && spm->descriptor().validation == SupportState::Unverified);
    assert(spm->validate_metadata(sentencepiece).ok());
    std::vector<uint32_t> spm_ids;
    assert(spm->encode(sentencepiece, "ab", spm_ids).ok());
    assert((spm_ids == std::vector<uint32_t>{7, 6, 8}));
    std::string spm_text;
    assert(spm->decode(sentencepiece, spm_ids, spm_text).ok() && spm_text == "ab");
    auto spm_report = evaluate_tokenizer_preflight(sentencepiece, builtins);
    assert(spm_report.metadata_valid && spm_report.adapter_registered && !spm_report.adapter_validated && !spm_report.ready);

    GgufTokenizerInfo bpe;
    bpe.model = "gpt2";
    bpe.pre_tokenizer = "gpt2";
    bpe.tokens = {"h", "i", "hi", "<unk>", "<|special|>"};
    bpe.merges = {"h i"};
    bpe.token_types = {1, 1, 1, 2, 3};
    bpe.unknown_token_id = 3;
    auto bpe_adapter = builtins.find(bpe);
    assert(bpe_adapter && bpe_adapter->validate_metadata(bpe).ok());
    std::vector<uint32_t> bpe_ids;
    assert(bpe_adapter->encode(bpe, "hi<|special|>", bpe_ids).ok());
    assert((bpe_ids == std::vector<uint32_t>{2, 4}));
    std::string bpe_text;
    assert(bpe_adapter->decode(bpe, bpe_ids, bpe_text).ok() && bpe_text == "hi");
    TokenizerEngine fail_closed_tokenizer(builtins);
    assert(fail_closed_tokenizer.encode(bpe, "hi", bpe_ids).code == StatusCode::Unsupported);
    assert(bpe_ids.empty());
}

void test_independent_sampler_and_stop_controller() {
    SamplingConfig greedy;
    greedy.temperature = 0.0f;
    greedy.seed = 42;
    Sampler sampler(greedy);
    int32_t token = -1;
    const std::vector<float> logits = {0.1f, 0.9f, 0.3f};
    assert(sampler.sample(logits, {}, token).ok());
    assert(token == 1);

    SamplingConfig random_config;
    random_config.temperature = 0.8f;
    random_config.top_k = 2;
    random_config.top_p = 0.9f;
    random_config.seed = 1234;
    Sampler first(random_config), second(random_config);
    int32_t first_token = -1, second_token = -1;
    assert(first.sample(logits, {}, first_token).ok());
    assert(second.sample(logits, {}, second_token).ok());
    assert(first_token == second_token);

    StopController stop(2, {{7, 8}});
    assert(!stop.push(7));
    assert(stop.push(8));
    stop.reset();
    assert(!stop.push(1));
    assert(stop.push(2));
}

void test_gguf_weight_streaming() {
    const std::string path = "/tmp/ai_chat_gpu_engine_weight.gguf";
    {
        std::ofstream out(path, std::ios::binary | std::ios::trunc);
        out.write("GGUF", 4);
        write_u32(out, 3);
        write_u64(out, 1);
        write_u64(out, 2);
        write_string(out, "general.architecture");
        write_u32(out, 8); write_string(out, "test-architecture");
        write_string(out, "general.alignment");
        write_u32(out, 4); write_u32(out, 32);
        write_string(out, "weight");
        write_u32(out, 1);
        write_u64(out, 8);
        write_u32(out, 0);
        write_u64(out, 0);
        const auto pos = static_cast<uint64_t>(out.tellp());
        const auto data_start = (pos + 31u) & ~31ULL;
        for (uint64_t i = pos; i < data_start; ++i) out.put('\0');
        const float values[] = {1,2,3,4,5,6,7,8};
        out.write(reinterpret_cast<const char*>(values), sizeof(values));
    }

    GgufReader reader;
    assert(GgufReader::open(path, reader).ok());
    FakeBackend backend;
    DeviceInfo device;
    device.backend = BackendKind::OpenGles;
    device.limits.max_buffer_size = 16;
    device.limits.max_storage_buffer_binding_size = 16;
    device.limits.min_storage_buffer_offset_alignment = 4;
    MemoryManager memory(backend, 64);
    WeightStreamingPolicy policy;
    policy.preferred_chunk_bytes = 16;
    LoadedWeightTensor loaded;
    assert(GgufWeightStreamer::load_tensor(reader, reader.info().tensors[0],
        backend, memory, device, policy, loaded).ok());
    assert(loaded.resident_chunks.size() == 2);
    assert(memory.allocated_bytes() == 32);
    float first[4] = {};
    assert(backend.readback(loaded.resident_chunks[0].memory.buffer(), 0, first, sizeof(first)).ok());
    assert(first[0] == 1.0f && first[3] == 4.0f);
    loaded = LoadedWeightTensor{};
    assert(memory.allocated_bytes() == 0);

    uint32_t streamed_chunks = 0;
    assert(GgufWeightStreamer::stream_tensor(reader, reader.info().tensors[0],
        backend, memory, device, policy,
        [&streamed_chunks](WeightChunk&& chunk) {
            assert(chunk.memory);
            ++streamed_chunks;
            return Status::success();
        }).ok());
    assert(streamed_chunks == 2);
    assert(memory.allocated_bytes() == 0); // each chunk is released after its consumer returns
    const Status consumer_failure = GgufWeightStreamer::stream_tensor(reader, reader.info().tensors[0],
        backend, memory, device, policy,
        [](WeightChunk&&) {
            return Status{StatusCode::BackendFailure, "intentional consumer failure"};
        });
    assert(consumer_failure.code == StatusCode::BackendFailure);
    assert(memory.allocated_bytes() == 0);
    std::remove(path.c_str());
}

void test_gpu_engine_backend_fallback() {
    GpuEngine engine;
    assert(engine.register_backend(std::make_shared<FakeBackend>(BackendKind::DawnWebGpu, false)).ok());
    assert(engine.register_backend(std::make_shared<FakeBackend>(BackendKind::OpenGles, true)).ok());
    assert(engine.initialize_with_fallback({BackendKind::DawnWebGpu, BackendKind::OpenGles}).ok());
    assert(engine.active_device() != nullptr);
    assert(engine.active_device()->backend == BackendKind::OpenGles);
    assert(engine.diagnostics().fallback_used);
    assert(!engine.diagnostics().fallback_reason.empty());
    assert(engine.diagnostics().last_error.empty());
    assert(engine.shutdown().ok());
}

void test_gpu_engine_lifecycle() {
    GpuEngine engine;
    auto backend = std::make_shared<FakeBackend>();
    assert(engine.register_backend(backend).ok());
    assert(engine.initialize(BackendKind::OpenGles).ok());
    assert(engine.ready());
    assert(engine.active_device());
    assert(engine.diagnostics().device_name == "fake-test-device");
    assert(!engine.active_device()->inference_ready);

        OperatorRegistry test_registry;
    auto register_test_kernel = [&test_registry](const std::string& id, BackendKind backend) {
        KernelDefinition definition;
        definition.operator_name = id;
        definition.kernel_id = id;
        definition.backend = backend;
        definition.input_type = DataType::F32;
        definition.output_type = DataType::F32;
        definition.validation = KernelValidation::Validated;
        if (backend == BackendKind::OpenGles) {
            definition.glsl_source = "#version 310 es\nlayout(local_size_x=1) in; void main() {}";
        } else {
            definition.wgsl_source = "@compute @workgroup_size(1) fn main() {}";
        }
        return test_registry.register_kernel(std::move(definition));
    };
    ComputeGraph graph;
    GraphNode node;
    node.name = "single";
    node.dispatch.label = "single";
    node.dispatch.kernel_id = "test-single";
    node.dispatch.backend = BackendKind::OpenGles;
    node.dispatch.validation = KernelValidation::Validated;
    assert(register_test_kernel("test-single", BackendKind::OpenGles).ok());
    size_t index = 0;
    GraphNode unverified_node;
    unverified_node.name = "unverified";
    assert(graph.add_node(unverified_node, test_registry, index).code == StatusCode::Unsupported);
    assert(graph.add_node(node, test_registry, index).ok());
    assert(engine.execute(graph).ok());
    assert(engine.shutdown().ok());
    assert(!engine.ready());
}


void test_dawn_add_validation_does_not_promote_fake_backend() {
    FakeBackend backend(BackendKind::DawnWebGpu, true);
    OperatorRegistry registry;
    assert(register_builtin_operator_candidates(registry).ok());
    assert(validate_dawn_add_f32_kernel(backend, registry).code == StatusCode::NumericalFailure);
    KernelDefinition selected;
    assert(registry.select_kernel("Add", BackendKind::DawnWebGpu, DataType::F32,
        DataType::F32, 2, 1, selected).code == StatusCode::Unsupported);
}

void test_add_operator_candidate_and_reference() {
    OperatorRegistry registry;
    assert(register_builtin_gles_operator_candidates(registry).ok());
    KernelDefinition selected;
    assert(registry.select_kernel("Add", BackendKind::DawnWebGpu, DataType::F32,
                                  DataType::F32, 2, 2, selected).code == StatusCode::Unsupported);
    KernelDefinition mixed_selected;
    assert(registry.select_kernel("Embedding", BackendKind::DawnWebGpu,
                                  std::vector<DataType>{DataType::F32, DataType::U32},
                                  DataType::F32, 2, mixed_selected).code == StatusCode::Unsupported);
    assert(registry.select_kernel("MatMul_Q8_0", BackendKind::DawnWebGpu,
                                  std::vector<DataType>{DataType::F32, DataType::U32},
                                  DataType::F32, 2, mixed_selected).code == StatusCode::Unsupported);

    FakeBackend backend;
    assert(validate_gles_add_f32_kernel(backend, registry).code == StatusCode::NumericalFailure);
    assert(registry.select_kernel("Add", BackendKind::OpenGles, DataType::F32,
                                  DataType::F32, 2, 2, selected).code == StatusCode::Unsupported);
    BufferDesc desc{24, 4, MemoryDomain::Device, false, "add"};
    std::shared_ptr<Buffer> a, b, out;
    assert(backend.allocate(desc, a).ok());
    assert(backend.allocate(desc, b).ok());
    assert(backend.allocate(desc, out).ok());
    TensorDesc tensor_desc;
    tensor_desc.shape = {2, 3};
    tensor_desc.data_type = DataType::F32;
    tensor_desc.memory = MemoryDomain::Device;
    tensor_desc.alignment = 4;
    Tensor lhs, rhs, output;
    assert(Tensor::create(tensor_desc, a, 0, lhs).ok());
    assert(Tensor::create(tensor_desc, b, 0, rhs).ok());
    assert(Tensor::create(tensor_desc, out, 0, output).ok());

    ComputeDispatch dispatch;
    assert(build_add_f32_candidate(lhs, rhs, output, dispatch).ok());
    assert(dispatch.bindings.size() == 3);
    assert(dispatch.workgroups_x == 1);
    assert(dispatch.label.find("unverified") != std::string::npos);

    const float left[] = {1, 2, 3};
    const float right[] = {4, 5, 6};
    float result[] = {0, 0, 0};
    assert(reference_add_f32(left, right, result, 3).ok());
    assert(result[0] == 5 && result[1] == 7 && result[2] == 9);
    const float embedding_weights[] = {1, 2, 3, 4, 5, 6};
    const std::vector<uint32_t> token_ids = {2, 0};
    float embedding_output[4] = {};
    assert(reference_embedding_f32(embedding_weights, 3, 2, token_ids, embedding_output).ok());
    assert(embedding_output[0] == 5 && embedding_output[1] == 6 &&
           embedding_output[2] == 1 && embedding_output[3] == 2);

    BufferDesc embedding_weights_desc{24, 4, MemoryDomain::Device, false, "embedding-weights"};
    BufferDesc embedding_ids_desc{8, 4, MemoryDomain::Device, false, "embedding-token-ids"};
    BufferDesc embedding_output_desc{16, 4, MemoryDomain::Device, false, "embedding-output"};
    std::shared_ptr<Buffer> embedding_weights_buffer, embedding_ids_buffer, embedding_output_buffer;
    assert(backend.allocate(embedding_weights_desc, embedding_weights_buffer).ok());
    assert(backend.allocate(embedding_ids_desc, embedding_ids_buffer).ok());
    assert(backend.allocate(embedding_output_desc, embedding_output_buffer).ok());
    TensorDesc embedding_weights_tensor_desc;
    embedding_weights_tensor_desc.shape = {3, 2};
    embedding_weights_tensor_desc.data_type = DataType::F32;
    Tensor embedding_weights_tensor, embedding_output_tensor;
    assert(Tensor::create(embedding_weights_tensor_desc, embedding_weights_buffer, 0,
                          embedding_weights_tensor).ok());
    TensorDesc embedding_output_tensor_desc;
    embedding_output_tensor_desc.shape = {2, 2};
    embedding_output_tensor_desc.data_type = DataType::F32;
    assert(Tensor::create(embedding_output_tensor_desc, embedding_output_buffer, 0,
                          embedding_output_tensor).ok());
    ComputeDispatch embedding_dispatch;
    assert(build_embedding_f32_candidate(embedding_weights_tensor, embedding_ids_buffer,
                                         token_ids, embedding_output_tensor,
                                         embedding_dispatch).ok());
    assert(embedding_dispatch.bindings.size() == 3);
    assert(embedding_dispatch.wgsl_source.find("const VOCABULARY_SIZE: u32 = 3u;") != std::string::npos);
    ComputeDispatch gles_embedding_dispatch;
    assert(build_embedding_f32_gles_candidate(embedding_weights_tensor, embedding_ids_buffer,
        token_ids, embedding_output_tensor, gles_embedding_dispatch).ok());
    assert(gles_embedding_dispatch.backend == BackendKind::OpenGles);
    assert(gles_embedding_dispatch.glsl_source.find("const uint VOCABULARY_SIZE = 3u;") != std::string::npos);
    assert(gles_embedding_dispatch.glsl_source.find("const uint EMBEDDING_DIMENSION = 2u;") != std::string::npos);
    assert(gles_embedding_dispatch.glsl_source.find("const uint TOKEN_COUNT = 2u;") != std::string::npos);
    const std::vector<uint32_t> invalid_token_ids = {3};
    assert(reference_embedding_f32(embedding_weights, 3, 2, invalid_token_ids, embedding_output).code ==
           StatusCode::InvalidArgument);
    float multiplied[3] = {};
    assert(reference_multiply_f32(left, right, multiplied, 3).ok());
    assert(multiplied[0] == 4.0f && multiplied[1] == 10.0f && multiplied[2] == 18.0f);

    const float matrix_a[] = {1, 2, 3, 4, 5, 6};
    const float matrix_b[] = {7, 8, 9, 10, 11, 12};
    float matrix_c[4] = {};
    assert(reference_matmul_f32(matrix_a, matrix_b, matrix_c, 2, 2, 3).ok());
    assert(matrix_c[0] == 58.0f && matrix_c[1] == 64.0f &&
           matrix_c[2] == 139.0f && matrix_c[3] == 154.0f);

    const float norm_input[] = {1, 2, 3, 4};
    const float norm_weight[] = {1, 1};
    float norm_output[4] = {};
    assert(reference_rms_norm_f32(norm_input, norm_weight, norm_output, 2, 2, 1e-5f).ok());
    assert(std::abs(norm_output[0] - 0.63245f) < 1e-4f);
    assert(std::abs(norm_output[1] - 1.26491f) < 1e-4f);

    const float activation_input[] = {-2.0f, 0.0f, 2.0f};
    float activation_output[3] = {};
    assert(reference_silu_f32(activation_input, activation_output, 3).ok());
    assert(activation_output[1] == 0.0f);
    assert(std::abs(activation_output[2] - 1.76159f) < 1e-4f);

    const float softmax_input[] = {1.0f, 2.0f, 3.0f, 0.0f};
    float softmax_output[4] = {};
    assert(reference_softmax_f32(softmax_input, softmax_output, 2, 2).ok());
    assert(std::abs(softmax_output[0] + softmax_output[1] - 1.0f) < 1e-6f);
    assert(std::abs(softmax_output[2] + softmax_output[3] - 1.0f) < 1e-6f);

    const float rope_input[] = {1.0f, 0.0f};
    float rope_output[2] = {};
    RopeConfig rope_config;
    rope_config.rotary_dimensions = 2;
    rope_config.start_position = 1;
    assert(reference_rope_f32(rope_input, rope_output, 1, 1, 2, rope_config).ok());
    assert(std::abs(rope_output[0] - std::cos(1.0f)) < 1e-5f);
    assert(std::abs(rope_output[1] - std::sin(1.0f)) < 1e-5f);

    BufferDesc rope_desc{8, 4, MemoryDomain::Device, false, "rope"};
    std::shared_ptr<Buffer> rope_input_buffer, rope_output_buffer;
    assert(backend.allocate(rope_desc, rope_input_buffer).ok());
    assert(backend.allocate(rope_desc, rope_output_buffer).ok());
    TensorDesc rope_tensor_desc;
    rope_tensor_desc.shape = {1, 1, 2};
    rope_tensor_desc.data_type = DataType::F32;
    Tensor rope_tensor, rope_output_tensor;
    assert(Tensor::create(rope_tensor_desc, rope_input_buffer, 0, rope_tensor).ok());
    assert(Tensor::create(rope_tensor_desc, rope_output_buffer, 0, rope_output_tensor).ok());
    ComputeDispatch rope_dispatch;
    assert(build_rope_f32_candidate(rope_tensor, rope_output_tensor, rope_config, rope_dispatch).ok());
    assert(rope_dispatch.bindings.size() == 2);
    assert(rope_dispatch.wgsl_source.find("const START_POSITION: u32 = 1u;") != std::string::npos);

    const float q_values[] = {1.0f, 0.0f};
    const float k_values[] = {1.0f, 0.0f};
    const float v_values[] = {2.0f, 4.0f};
    float attention_output[2] = {};
    assert(reference_attention_f32(q_values, k_values, v_values, attention_output,
                                   1, 1, 1, 2, 0).ok());
    assert(std::abs(attention_output[0] - 2.0f) < 1e-6f &&
           std::abs(attention_output[1] - 4.0f) < 1e-6f);

    const float gqa_query[] = {1, 0, 0, 1};
    const float gqa_key[] = {1, 0, 0, 1};
    const float gqa_value[] = {2, 4, 6, 8};
    float gqa_output[4] = {};
    assert(reference_attention_f32(gqa_query, gqa_key, gqa_value, gqa_output,
                                   1, 2, 1, 2, 1).ok());
    assert(std::abs(gqa_output[0] - 3.320954f) < 1e-4f);
    assert(std::abs(gqa_output[1] - 5.320954f) < 1e-4f);
    assert(std::abs(gqa_output[2] - 4.679046f) < 1e-4f);
    assert(std::abs(gqa_output[3] - 6.679046f) < 1e-4f);

    const float causal_query[] = {1, 1};
    const float causal_key[] = {1, 1};
    const float causal_value[] = {2, 6};
    float causal_output[2] = {};
    assert(reference_attention_f32(causal_query, causal_key, causal_value, causal_output,
                                   2, 1, 1, 1, 0).ok());
    assert(std::abs(causal_output[0] - 2.0f) < 1e-6f);
    assert(std::abs(causal_output[1] - 4.0f) < 1e-6f);

    BufferDesc gqa_query_desc{16, 4, MemoryDomain::Device, false, "gqa-query"};
    BufferDesc gqa_kv_desc{16, 4, MemoryDomain::Device, false, "gqa-kv"};
    std::shared_ptr<Buffer> gqa_query_buffer, gqa_key_buffer, gqa_value_buffer, gqa_output_buffer;
    assert(backend.allocate(gqa_query_desc, gqa_query_buffer).ok());
    assert(backend.allocate(gqa_kv_desc, gqa_key_buffer).ok());
    assert(backend.allocate(gqa_kv_desc, gqa_value_buffer).ok());
    assert(backend.allocate(gqa_query_desc, gqa_output_buffer).ok());
    TensorDesc gqa_query_tensor_desc;
    gqa_query_tensor_desc.shape = {1, 2, 2};
    gqa_query_tensor_desc.data_type = DataType::F32;
    Tensor gqa_query_tensor, gqa_output_tensor;
    assert(Tensor::create(gqa_query_tensor_desc, gqa_query_buffer, 0, gqa_query_tensor).ok());
    assert(Tensor::create(gqa_query_tensor_desc, gqa_output_buffer, 0, gqa_output_tensor).ok());
    TensorDesc gqa_kv_tensor_desc;
    gqa_kv_tensor_desc.shape = {2, 1, 2};
    gqa_kv_tensor_desc.data_type = DataType::F32;
    Tensor gqa_key_tensor, gqa_value_tensor;
    assert(Tensor::create(gqa_kv_tensor_desc, gqa_key_buffer, 0, gqa_key_tensor).ok());
    assert(Tensor::create(gqa_kv_tensor_desc, gqa_value_buffer, 0, gqa_value_tensor).ok());
    ComputeDispatch gqa_dispatch;
    assert(build_attention_f32_candidate(gqa_query_tensor, gqa_key_tensor, gqa_value_tensor,
                                         gqa_output_tensor, 1, gqa_dispatch).ok());
    assert(gqa_dispatch.wgsl_source.find("const QUERY_HEADS: u32 = 2u;") != std::string::npos);
    assert(gqa_dispatch.wgsl_source.find("const KV_HEADS: u32 = 1u;") != std::string::npos);
    assert(gqa_dispatch.workgroups_x == 1 && gqa_dispatch.workgroups_y == 2);

    BufferDesc attention_desc{8, 4, MemoryDomain::Device, false, "attention"};
    std::shared_ptr<Buffer> q_buffer, k_buffer, v_buffer, attention_output_buffer;
    assert(backend.allocate(attention_desc, q_buffer).ok());
    assert(backend.allocate(attention_desc, k_buffer).ok());
    assert(backend.allocate(attention_desc, v_buffer).ok());
    assert(backend.allocate(attention_desc, attention_output_buffer).ok());
    TensorDesc attention_tensor_desc;
    attention_tensor_desc.shape = {1, 1, 2};
    attention_tensor_desc.data_type = DataType::F32;
    Tensor q_tensor, k_tensor, v_tensor, attention_tensor;
    assert(Tensor::create(attention_tensor_desc, q_buffer, 0, q_tensor).ok());
    assert(Tensor::create(attention_tensor_desc, k_buffer, 0, k_tensor).ok());
    assert(Tensor::create(attention_tensor_desc, v_buffer, 0, v_tensor).ok());
    assert(Tensor::create(attention_tensor_desc, attention_output_buffer, 0, attention_tensor).ok());
    ComputeDispatch attention_dispatch;
    assert(build_attention_f32_candidate(q_tensor, k_tensor, v_tensor, attention_tensor,
                                         0, attention_dispatch).ok());
    assert(attention_dispatch.bindings.size() == 4);
    assert(attention_dispatch.wgsl_source.find("const SCALE: f32 = 0.707106781187;") != std::string::npos ||
           attention_dispatch.wgsl_source.find("const SCALE: f32 = 0.7071067812;") != std::string::npos);

    BufferDesc activation_desc{12, 4, MemoryDomain::Device, false, "silu"};
    std::shared_ptr<Buffer> activation_input_buffer, activation_output_buffer;
    assert(backend.allocate(activation_desc, activation_input_buffer).ok());
    assert(backend.allocate(activation_desc, activation_output_buffer).ok());
    TensorDesc activation_tensor_desc;
    activation_tensor_desc.shape = {3};
    activation_tensor_desc.data_type = DataType::F32;
    Tensor activation_tensor, activation_output_tensor;
    assert(Tensor::create(activation_tensor_desc, activation_input_buffer, 0, activation_tensor).ok());
    assert(Tensor::create(activation_tensor_desc, activation_output_buffer, 0, activation_output_tensor).ok());
    ComputeDispatch silu_dispatch;
    assert(build_silu_f32_candidate(activation_tensor, activation_output_tensor, silu_dispatch).ok());
    assert(silu_dispatch.bindings.size() == 2);
    assert(silu_dispatch.label.find("unverified") != std::string::npos);

    BufferDesc softmax_desc{16, 4, MemoryDomain::Device, false, "softmax"};
    std::shared_ptr<Buffer> softmax_input_buffer, softmax_output_buffer;
    assert(backend.allocate(softmax_desc, softmax_input_buffer).ok());
    assert(backend.allocate(softmax_desc, softmax_output_buffer).ok());
    TensorDesc softmax_tensor_desc;
    softmax_tensor_desc.shape = {2, 2};
    softmax_tensor_desc.data_type = DataType::F32;
    Tensor softmax_tensor, softmax_output_tensor;
    assert(Tensor::create(softmax_tensor_desc, softmax_input_buffer, 0, softmax_tensor).ok());
    assert(Tensor::create(softmax_tensor_desc, softmax_output_buffer, 0, softmax_output_tensor).ok());
    ComputeDispatch softmax_dispatch;
    assert(build_softmax_f32_candidate(softmax_tensor, softmax_output_tensor, softmax_dispatch).ok());
    assert(softmax_dispatch.bindings.size() == 2);
    assert(softmax_dispatch.wgsl_source.find("const COLUMNS: u32 = 2u;") != std::string::npos);

    BufferDesc mat_a_desc{24, 4, MemoryDomain::Device, false, "mat-a"};
    BufferDesc mat_b_desc{24, 4, MemoryDomain::Device, false, "mat-b"};
    BufferDesc mat_out_desc{16, 4, MemoryDomain::Device, false, "mat-out"};
    std::shared_ptr<Buffer> mat_a, mat_b, mat_out;
    assert(backend.allocate(mat_a_desc, mat_a).ok());
    assert(backend.allocate(mat_b_desc, mat_b).ok());
    assert(backend.allocate(mat_out_desc, mat_out).ok());
    TensorDesc a_desc; a_desc.shape = {2, 3}; a_desc.data_type = DataType::F32;
    TensorDesc b_desc; b_desc.shape = {3, 2}; b_desc.data_type = DataType::F32;
    TensorDesc c_desc; c_desc.shape = {2, 2}; c_desc.data_type = DataType::F32;
    Tensor a_tensor, b_tensor, c_tensor;
    assert(Tensor::create(a_desc, mat_a, 0, a_tensor).ok());
    assert(Tensor::create(b_desc, mat_b, 0, b_tensor).ok());
    assert(Tensor::create(c_desc, mat_out, 0, c_tensor).ok());
    ComputeDispatch mat_dispatch;
    assert(build_matmul_f32_candidate(a_tensor, b_tensor, c_tensor, mat_dispatch).ok());
    assert(mat_dispatch.bindings.size() == 3);
    assert(mat_dispatch.workgroup_size_x == 8 && mat_dispatch.workgroup_size_y == 8);
    assert(mat_dispatch.wgsl_source.find("@binding(3)") == std::string::npos);
    assert(mat_dispatch.wgsl_source.find("params.") == std::string::npos);
    assert(mat_dispatch.wgsl_source.find("const ROWS: u32 = 2u;") != std::string::npos);

    BufferDesc norm_x_desc{16, 4, MemoryDomain::Device, false, "norm-x"};
    BufferDesc norm_w_desc{8, 4, MemoryDomain::Device, false, "norm-w"};
    std::shared_ptr<Buffer> norm_x, norm_w, norm_y;
    assert(backend.allocate(norm_x_desc, norm_x).ok());
    assert(backend.allocate(norm_w_desc, norm_w).ok());
    assert(backend.allocate(norm_x_desc, norm_y).ok());
    TensorDesc nx_desc; nx_desc.shape = {2, 2}; nx_desc.data_type = DataType::F32;
    TensorDesc nw_desc; nw_desc.shape = {2}; nw_desc.data_type = DataType::F32;
    Tensor nx, nw, ny;
    assert(Tensor::create(nx_desc, norm_x, 0, nx).ok());
    assert(Tensor::create(nw_desc, norm_w, 0, nw).ok());
    assert(Tensor::create(nx_desc, norm_y, 0, ny).ok());
    ComputeDispatch norm_dispatch;
    assert(build_rms_norm_f32_candidate(nx, nw, ny, 1e-5f, norm_dispatch).ok());
    assert(norm_dispatch.bindings.size() == 3);
    assert(norm_dispatch.wgsl_source.find("@binding(3)") == std::string::npos);
    assert(norm_dispatch.wgsl_source.find("params.") == std::string::npos);
}


void test_builtin_architecture_adapter_is_fail_closed() {
    ArchitectureRegistry registry;
    assert(register_builtin_architecture_adapters(registry).ok());
    auto adapter = registry.find("llama");
    assert(adapter);
    assert(adapter->descriptor().validation == SupportState::Unverified);

    GgufFileInfo model;
    model.architecture = "llama";
    model.metadata["llama.block_count"] = uint64_t{24};
    model.metadata["llama.embedding_length"] = uint64_t{2048};
    model.metadata["llama.attention.head_count"] = uint64_t{16};
    model.metadata["llama.attention.head_count_kv"] = uint64_t{4};
    model.metadata["llama.feed_forward_length"] = uint64_t{5632};
    assert(adapter->validate_metadata(model).ok());
    auto valid_report = evaluate_model_preflight(model, registry, QuantizationRegistry{});
    assert(valid_report.architecture_known);
    assert(std::find(valid_report.blockers.begin(), valid_report.blockers.end(),
                     "UNVERIFIED_ARCHITECTURE_ADAPTER:llama") != valid_report.blockers.end());

    model.metadata["llama.attention.head_count_kv"] = uint64_t{32};
    assert(adapter->validate_metadata(model).code == StatusCode::InvalidArgument);
}


void test_model_plan_graph_compilation_is_fail_closed() {
    ModelExecutionPlan plan;
    plan.structurally_validated = true;
    plan.operator_coverage_validated = true;
    plan.operations = {
        {"add", "Add", {"input_a", "input_b"}, {}, "sum", {4}},
        {"mul", "Multiply", {"sum", "input_c"}, {}, "product", {4}}
    };
    OperatorRegistry operators;
    for (const auto& entry : std::vector<std::pair<std::string, std::string>>{
            {"Add", "test-add"}, {"Multiply", "test-mul"}}) {
        KernelDefinition kernel;
        kernel.operator_name = entry.first;
        kernel.kernel_id = entry.second;
        kernel.backend = BackendKind::OpenGles;
        kernel.input_type = DataType::F32;
        kernel.output_type = DataType::F32;
        kernel.min_input_count = 2;
        kernel.max_input_count = 2;
        kernel.min_rank = 1;
        kernel.max_rank = 1;
        kernel.validation = KernelValidation::Validated;
        kernel.glsl_source = "#version 310 es\nvoid main(){}";
        assert(operators.register_kernel(std::move(kernel)).ok());
    }
    ModelLoweringRegistry lowerings;
    for (const auto& item : std::vector<std::pair<std::string, std::string>>{
            {"Add", "test-add"}, {"Multiply", "test-mul"}}) {
        assert(lowerings.register_lowering(BackendKind::OpenGles, item.first,
            [kernel_id = item.second](const ModelPlanOperation&, ComputeDispatch& dispatch) {
                dispatch.backend = BackendKind::OpenGles;
                dispatch.kernel_id = kernel_id;
                dispatch.glsl_source = "#version 310 es\nvoid main(){}";
                dispatch.workgroups_x = 1;
                return Status::success();
            }).ok());
    }
    ComputeGraph graph;
    ModelGraphCompileReport report;
    assert(compile_model_execution_plan(plan, BackendKind::OpenGles, lowerings, operators,
        {"input_a", "input_b", "input_c"}, {}, graph, report).ok());
    assert(report.compiled && report.compiled_nodes == 2);
    assert(graph.node_count() == 2);
    std::vector<size_t> order;
    assert(graph.topological_order(order).ok());
    assert(order.size() == 2 && order[0] == 0 && order[1] == 1);
    assert(plan.graph_compiled && !plan.gpu_ready && !plan.execution_validated);

    ModelExecutionPlan blocked = plan;
    blocked.graph_compiled = false;
    blocked.operator_coverage_validated = false;
    ComputeGraph blocked_graph;
    assert(compile_model_execution_plan(blocked, BackendKind::OpenGles, lowerings, operators,
        {"input_a", "input_b", "input_c"}, {}, blocked_graph, report).code == StatusCode::NotReady);
    assert(blocked_graph.node_count() == 0);

    ModelExecutionPlan missing_input = plan;
    missing_input.graph_compiled = false;
    ComputeGraph incomplete_graph;
    assert(compile_model_execution_plan(missing_input, BackendKind::OpenGles, lowerings, operators,
        {"input_a", "input_b"}, {}, incomplete_graph, report).code == StatusCode::Unsupported);
    assert(!report.unresolved_inputs.empty());
    assert(incomplete_graph.node_count() == 0); // partial graphs are never published
}

void test_llama_model_execution_plan_is_structural_only() {
    ArchitectureRegistry registry;
    assert(register_builtin_architecture_adapters(registry).ok());
    const auto adapter = registry.find("llama");
    assert(adapter);
    GgufFileInfo model;
    model.architecture = "llama";
    model.metadata["llama.block_count"] = uint64_t{1};
    model.metadata["llama.embedding_length"] = uint64_t{8};
    model.metadata["llama.attention.head_count"] = uint64_t{2};
    model.metadata["llama.attention.head_count_kv"] = uint64_t{1};
    model.metadata["llama.feed_forward_length"] = uint64_t{16};
    const std::vector<std::string> names = {
        "token_embd.weight", "output.weight", "output_norm.weight",
        "blk.0.attn_norm.weight", "blk.0.attn_q.weight",
        "blk.0.attn_k.weight", "blk.0.attn_v.weight",
        "blk.0.attn_output.weight", "blk.0.ffn_norm.weight",
        "blk.0.ffn_gate.weight", "blk.0.ffn_up.weight", "blk.0.ffn_down.weight"
    };
    model.tokenizer.tokens = {"a", "b"};
    const auto ends_with = [](const std::string& value, const std::string& suffix) {
        return value.size() >= suffix.size() &&
               value.compare(value.size() - suffix.size(), suffix.size(), suffix) == 0;
    };
    for (const auto& name : names) {
        GgufTensorInfo tensor; tensor.name = name;
        if (name == "token_embd.weight" || name == "output.weight") tensor.dimensions = {8, 2};
        else if (name == "output_norm.weight" || ends_with(name, "attn_norm.weight") ||
                 ends_with(name, "ffn_norm.weight")) tensor.dimensions = {8};
        else if (ends_with(name, "attn_k.weight") || ends_with(name, "attn_v.weight"))
            tensor.dimensions = {8, 4};
        else if (ends_with(name, "ffn_gate.weight") || ends_with(name, "ffn_up.weight"))
            tensor.dimensions = {8, 16};
        else if (ends_with(name, "ffn_down.weight")) tensor.dimensions = {16, 8};
        else tensor.dimensions = {8, 8};
        uint64_t elements = 1;
        for (uint64_t dimension : tensor.dimensions) elements *= dimension;
        tensor.byte_size_known = true;
        tensor.byte_size = elements * 4;
        tensor.type_id = 0;
        model.tensors.push_back(tensor);
    }
    ModelExecutionPlan plan;
    assert(adapter->build_execution_plan(model, plan).ok());
    assert(plan.structurally_validated);
    assert(!plan.gpu_ready && !plan.execution_validated);
    assert(plan.layer_count == 1 && plan.embedding_length == 8);
    assert(plan.operations.size() == 19);
    assert(plan.operations.front().operator_name == "Embedding");
    assert(plan.operations.back().operator_name == "MatMul");
    assert(std::find(plan.blockers.begin(), plan.blockers.end(),
                     "MODEL_GRAPH_EXECUTION_NOT_VALIDATED") != plan.blockers.end());

    GgufFileInfo tied_model = model;
    tied_model.tensors.erase(std::remove_if(tied_model.tensors.begin(), tied_model.tensors.end(),
        [](const GgufTensorInfo& tensor) { return tensor.name == "output.weight"; }),
        tied_model.tensors.end());
    tied_model.metadata["llama.tie_word_embeddings"] = true;
    ModelExecutionPlan tied_plan;
    assert(adapter->build_execution_plan(tied_model, tied_plan).ok());
    GgufFileInfo ambiguous_output = tied_model;
    ambiguous_output.metadata.erase("llama.tie_word_embeddings");
    ModelExecutionPlan ambiguous_plan;
    assert(adapter->build_execution_plan(ambiguous_output, ambiguous_plan).code == StatusCode::Unsupported);

    OperatorRegistry operator_registry;
    ModelGraphCoverageReport coverage;
    assert(validate_model_plan_operator_coverage(plan, operator_registry,
        BackendKind::OpenGles, coverage).ok());
    assert(!coverage.complete && !plan.operator_coverage_validated);
    assert(std::find(coverage.unsupported_operators.begin(), coverage.unsupported_operators.end(),
                     "Attention") != coverage.unsupported_operators.end());

    model.tensors.pop_back();
    assert(adapter->build_execution_plan(model, plan).code == StatusCode::Unsupported);
    assert(!plan.structurally_validated);
}

void test_model_graph_compiler_requires_validated_dispatch_builders() {
    FakeBackend backend;
    TensorBindings tensors;
    const auto bind_tensor = [&](const std::string& name) {
        BufferDesc buffer_desc{16, 4, MemoryDomain::Device, false, name};
        std::shared_ptr<Buffer> buffer;
        assert(backend.allocate(buffer_desc, buffer).ok());
        TensorDesc tensor_desc; tensor_desc.shape = {4}; tensor_desc.data_type = DataType::F32;
        Tensor tensor;
        assert(Tensor::create(tensor_desc, buffer, 0, tensor).ok());
        tensors.emplace(name, tensor);
    };
    bind_tensor("input"); bind_tensor("weight"); bind_tensor("output");

    ModelExecutionPlan plan;
    plan.structurally_validated = true;
    plan.operator_coverage_validated = true;
    plan.execution_validated = true;
    plan.gpu_ready = true;
    plan.external_inputs = {"input"};
    plan.operations.push_back({"add", "Add", {"input"}, {"weight"}, "output", {4}});

    OperatorRegistry operators;
    KernelDefinition kernel;
    kernel.operator_name = "Add"; kernel.kernel_id = "add-f32-validated";
    kernel.backend = BackendKind::OpenGles; kernel.input_type = DataType::F32;
    kernel.output_type = DataType::F32; kernel.min_input_count = 1;
    kernel.max_input_count = 1; kernel.min_rank = 1; kernel.max_rank = 1;
    kernel.validation = KernelValidation::Validated;
    kernel.glsl_source = "#version 310 es\\nlayout(local_size_x=1) in; void main() {}";
    assert(operators.register_kernel(kernel).ok());

    OperatorDispatchRegistry builders;
    OperatorDispatchBuilderDefinition builder;
    builder.operator_name = "Add"; builder.backend = BackendKind::OpenGles;
    builder.validation = SupportState::Validated;
    builder.build = [](const ModelPlanOperation&, const std::vector<Tensor>& inputs,
                       const std::vector<Tensor>& weights, const Tensor& output,
                       ComputeDispatch& dispatch) {
        dispatch.backend = BackendKind::OpenGles;
        dispatch.kernel_id = "add-f32-validated";
        dispatch.validation = KernelValidation::Validated;
        dispatch.glsl_source = "#version 310 es\\nlayout(local_size_x=1) in; void main() {}";
        dispatch.workgroup_size_x = 1; dispatch.workgroup_size_y = 1; dispatch.workgroup_size_z = 1;
        dispatch.workgroups_x = 1; dispatch.workgroups_y = 1; dispatch.workgroups_z = 1;
        dispatch.bindings = {
            {0, inputs[0].buffer(), inputs[0].byte_offset(), inputs[0].byte_size()},
            {1, weights[0].buffer(), weights[0].byte_offset(), weights[0].byte_size()},
            {2, output.buffer(), output.byte_offset(), output.byte_size()}
        };
        return Status::success();
    };
    assert(builders.register_builder(builder).ok());
    ModelGraphCompiler compiler;
    ComputeGraph graph;
    assert(compiler.compile(plan, tensors, builders, operators, BackendKind::OpenGles, graph).ok());
    assert(graph.node_count() == 1);

    OperatorDispatchRegistry unverified;
    builder.validation = SupportState::Unverified;
    assert(unverified.register_builder(builder).ok());
    ComputeGraph rejected;
    assert(compiler.compile(plan, tensors, unverified, operators,
                            BackendKind::OpenGles, rejected).code == StatusCode::Unsupported);
    TensorBindings incomplete = tensors;
    incomplete.erase("weight");
    assert(compiler.compile(plan, incomplete, builders, operators,
                            BackendKind::OpenGles, rejected).code == StatusCode::InvalidArgument);
}

void test_qwen2_dense_adapter_builds_bias_aware_structural_plan() {
    ArchitectureRegistry registry;
    assert(register_builtin_architecture_adapters(registry).ok());
    const auto adapter = registry.find("qwen2");
    assert(adapter);
    GgufFileInfo model;
    model.architecture = "qwen2";
    model.metadata["qwen2.block_count"] = uint64_t{1};
    model.metadata["qwen2.embedding_length"] = uint64_t{8};
    model.metadata["qwen2.attention.head_count"] = uint64_t{2};
    model.metadata["qwen2.attention.head_count_kv"] = uint64_t{1};
    model.metadata["qwen2.feed_forward_length"] = uint64_t{16};
    model.metadata["qwen2.tie_word_embeddings"] = false;
    model.tokenizer.tokens = {"a", "b"};
    const std::vector<std::pair<std::string, std::vector<uint64_t>>> tensors = {
        {"token_embd.weight",{8,2}}, {"output.weight",{8,2}}, {"output_norm.weight",{8}},
        {"blk.0.attn_norm.weight",{8}}, {"blk.0.attn_q.weight",{8,8}},
        {"blk.0.attn_k.weight",{8,4}}, {"blk.0.attn_v.weight",{8,4}},
        {"blk.0.attn_output.weight",{8,8}}, {"blk.0.ffn_norm.weight",{8}},
        {"blk.0.ffn_gate.weight",{8,16}}, {"blk.0.ffn_up.weight",{8,16}},
        {"blk.0.ffn_down.weight",{16,8}}, {"blk.0.attn_q.bias",{8}},
        {"blk.0.attn_k.bias",{4}}, {"blk.0.attn_v.bias",{4}},
        {"blk.0.attn_output.bias",{8}}
    };
    for (const auto& item : tensors) {
        GgufTensorInfo tensor; tensor.name=item.first; tensor.dimensions=item.second;
        if (item.first.find(".bias") != std::string::npos) tensor.type.name = "F32";
        uint64_t elements=1; for(uint64_t d:tensor.dimensions) elements*=d;
        tensor.byte_size_known=true; tensor.byte_size=elements*4; model.tensors.push_back(tensor);
    }
    ModelExecutionPlan plan;
    assert(adapter->build_execution_plan(model, plan).ok());
    assert(plan.architecture == "qwen2" && plan.structurally_validated);
    assert(!plan.gpu_ready && !plan.execution_validated);
    assert(std::find_if(plan.operations.begin(), plan.operations.end(), [](const auto& op) {
        return op.id == "blk.0.attn_q.bias.apply" && op.operator_name == "Add";
    }) != plan.operations.end());
    model.tensors.pop_back();
    assert(adapter->build_execution_plan(model, plan).code == StatusCode::Unsupported);
}

void test_model_and_quantization_registry_fail_closed() {
    QuantizationRegistry quantizations;
    QuantizationFormat q4;
    q4.file_type_id = 2;
    q4.name = "Q4_0";
    q4.block_elements = 32;
    q4.bytes_per_block = 18;
    q4.quantized = true;
    assert(quantizations.register_format(q4).ok());
    assert(!quantizations.can_decode(2));
    assert(!quantizations.can_execute_on_gpu(2));

    ArchitectureRegistry architectures;
    assert(architectures.register_adapter(std::make_shared<FakeArchitecture>()).ok());
    GgufFileInfo model;
    model.architecture = "test-architecture";
    model.tensor_count = 1;
    GgufTensorInfo tensor;
    tensor.name = "weight";
    tensor.type_id = 2;
    tensor.type = GgufReader::tensor_type_info(2);
    tensor.byte_size_known = true;
    tensor.byte_size = 18;
    model.tensors.push_back(tensor);

    const auto report = evaluate_model_preflight(model, architectures, quantizations);
    assert(report.architecture_known);
    assert(!report.all_quantization_decoders_validated);
    assert(!report.all_quantization_gpu_kernels_validated);
    assert(!report.gpu_execution_ready);
    assert(report.unverified_quantized_tensors == 1);
}


void test_universal_gles_operator_candidate_is_fail_closed() {
    FakeBackend backend;
    BufferDesc desc{16, 4, MemoryDomain::Device, false, "gles-add"};
    std::shared_ptr<Buffer> lhs_buffer, rhs_buffer, output_buffer;
    assert(backend.allocate(desc, lhs_buffer).ok());
    assert(backend.allocate(desc, rhs_buffer).ok());
    assert(backend.allocate(desc, output_buffer).ok());
    TensorDesc tensor_desc; tensor_desc.shape = {4}; tensor_desc.data_type = DataType::F32;
    Tensor lhs, rhs, output;
    assert(Tensor::create(tensor_desc, lhs_buffer, 0, lhs).ok());
    assert(Tensor::create(tensor_desc, rhs_buffer, 0, rhs).ok());
    assert(Tensor::create(tensor_desc, output_buffer, 0, output).ok());
    ComputeDispatch dispatch;
    assert(build_add_f32_gles_candidate(lhs, rhs, output, dispatch).ok());
    assert(dispatch.backend == BackendKind::OpenGles);
    assert(dispatch.wgsl_source.empty());
    assert(dispatch.glsl_source.find("#version 310 es") != std::string::npos);
    assert(dispatch.glsl_source.find("output_values.length()") != std::string::npos);

    OperatorRegistry registry;
    assert(register_builtin_gles_operator_candidates(registry).ok());
    KernelDefinition selected;
    assert(registry.select_kernel("Add", BackendKind::OpenGles, DataType::F32,
                                  DataType::F32, 2, 1, selected).code == StatusCode::Unsupported);
    KernelDefinition validated;
    validated.operator_name = "Add";
    validated.kernel_id = "gles-add-test-validated";
    validated.backend = BackendKind::OpenGles;
    validated.input_type = DataType::F32;
    validated.output_type = DataType::F32;
    validated.min_input_count = 2;
    validated.max_input_count = 2;
    validated.min_rank = 1;
    validated.max_rank = 1;
    validated.validation = KernelValidation::Validated;
    validated.glsl_source = dispatch.glsl_source;
    assert(registry.register_kernel(validated).ok());
    dispatch.kernel_id = validated.kernel_id;
    assert(registry.authorize_dispatch(validated.kernel_id, BackendKind::OpenGles, dispatch).ok());
    assert(dispatch.validation == KernelValidation::Validated);
}

void test_operator_registry_fail_closed() {
    OperatorRegistry registry;
    KernelDefinition definition;
    definition.operator_name = "Add";
    definition.kernel_id = "add-f32-v1";
    definition.backend = BackendKind::DawnWebGpu;
    definition.input_type = DataType::F32;
    definition.output_type = DataType::F32;
    definition.min_input_count = 2;
    definition.max_input_count = 2;
    definition.min_rank = 2;
    definition.max_rank = 2;
    definition.wgsl_source = "@compute @workgroup_size(1) fn main() {}";
    assert(registry.register_kernel(definition).ok());

    KernelDefinition selected;
    assert(registry.select_kernel("Add", BackendKind::DawnWebGpu, DataType::F32,
                                  DataType::F32, 2, 2, selected).code == StatusCode::Unsupported);
    definition.kernel_id = "add-f32-validated-v1";
    definition.validation = KernelValidation::Validated;
    assert(registry.register_kernel(definition).ok());
    assert(registry.select_kernel("Add", BackendKind::DawnWebGpu, DataType::F32,
                                  DataType::F32, 2, 2, selected).ok());
    ComputeDispatch authorized;
    authorized.kernel_id = selected.kernel_id;
    authorized.wgsl_source = selected.wgsl_source;
    assert(registry.authorize_dispatch(selected.kernel_id, selected.backend, authorized).ok());
    assert(authorized.validation == KernelValidation::Validated);
    ComputeDispatch unknown;
    unknown.kernel_id = "not-registered";
    unknown.wgsl_source = selected.wgsl_source;
    assert(registry.authorize_dispatch(unknown.kernel_id, selected.backend, unknown).code ==
           StatusCode::Unsupported);
    OperatorRegistry gles_registry;
    KernelDefinition gles_definition;
    gles_definition.operator_name = "Add";
    gles_definition.kernel_id = "gles-add-f32-v1";
    gles_definition.backend = BackendKind::OpenGles;
    gles_definition.input_type = DataType::F32;
    gles_definition.output_type = DataType::F32;
    gles_definition.min_input_count = 2;
    gles_definition.max_input_count = 2;
    gles_definition.min_rank = 1;
    gles_definition.max_rank = 4;
    gles_definition.validation = KernelValidation::Validated;
    gles_definition.glsl_source = "#version 310 es\nvoid main(){}";
    assert(gles_registry.register_kernel(gles_definition).ok());
    KernelDefinition gles_selected;
    assert(gles_registry.select_kernel("Add", BackendKind::OpenGles,
        DataType::F32, DataType::F32, 2, 2, gles_selected).ok());
    ComputeDispatch gles_dispatch;
    gles_dispatch.kernel_id = gles_selected.kernel_id;
    gles_dispatch.glsl_source = gles_selected.glsl_source;
    assert(gles_registry.authorize_dispatch(gles_selected.kernel_id,
        BackendKind::OpenGles, gles_dispatch).ok());
    assert(gles_dispatch.backend == BackendKind::OpenGles);

    KernelDefinition mixed;
    mixed.operator_name = "Mixed";
    mixed.kernel_id = "mixed-f32-u32-v1";
    mixed.backend = BackendKind::DawnWebGpu;
    mixed.input_type = DataType::F32;
    mixed.input_types = {DataType::F32, DataType::U32};
    mixed.output_type = DataType::F32;
    mixed.min_input_count = 2;
    mixed.max_input_count = 2;
    mixed.min_rank = 2;
    mixed.max_rank = 2;
    mixed.validation = KernelValidation::Validated;
    mixed.wgsl_source = "@compute @workgroup_size(1) fn main() {}";
    assert(registry.register_kernel(mixed).ok());
    assert(registry.select_kernel("Mixed", BackendKind::DawnWebGpu,
                                  std::vector<DataType>{DataType::F32, DataType::U32},
                                  DataType::F32, 2, selected).ok());
    assert(registry.select_kernel("Mixed", BackendKind::DawnWebGpu,
                                  std::vector<DataType>{DataType::F32, DataType::F32},
                                  DataType::F32, 2, selected).code == StatusCode::Unsupported);
}

void test_graph_scheduler_order_and_cancel() {
    FakeBackend backend;
    OperatorRegistry test_registry;
    auto register_test_kernel = [&test_registry](const std::string& id, BackendKind backend) {
        KernelDefinition definition;
        definition.operator_name = id;
        definition.kernel_id = id;
        definition.backend = backend;
        definition.input_type = DataType::F32;
        definition.output_type = DataType::F32;
        definition.validation = KernelValidation::Validated;
        definition.wgsl_source = "@compute @workgroup_size(1) fn main() {}";
        return test_registry.register_kernel(std::move(definition));
    };
    ComputeGraph graph;
    size_t index = 0;
    GraphNode first;
    first.name = "first";
    first.dispatch.label = "first";
    first.dispatch.kernel_id = "test-first";
    first.dispatch.backend = BackendKind::OpenGles;
    first.dispatch.validation = KernelValidation::Validated;
    assert(register_test_kernel("test-first", BackendKind::OpenGles).ok());
    assert(graph.add_node(first, test_registry, index).ok());

    GraphNode second;
    second.name = "second";
    second.dependencies = {0};
    second.dispatch.label = "second";
    second.dispatch.kernel_id = "test-second";
    second.dispatch.backend = BackendKind::OpenGles;
    second.dispatch.validation = KernelValidation::Validated;
    assert(register_test_kernel("test-second", BackendKind::OpenGles).ok());
    assert(graph.add_node(second, test_registry, index).ok());

    GraphNode independent;
    independent.name = "independent";
    independent.dispatch.label = "independent";
    independent.dispatch.kernel_id = "test-independent";
    independent.dispatch.backend = BackendKind::OpenGles;
    independent.dispatch.validation = KernelValidation::Validated;
    assert(register_test_kernel("test-independent", BackendKind::OpenGles).ok());
    assert(graph.add_node(independent, test_registry, index).ok());

    std::vector<size_t> order;
    assert(graph.topological_order(order).ok());
    auto first_pos = std::find(order.begin(), order.end(), 0);
    auto second_pos = std::find(order.begin(), order.end(), 1);
    assert(first_pos < second_pos);

    ExecutionScheduler scheduler;
    ComputeGraph mismatched_graph;
    GraphNode mismatched;
    mismatched.name = "wrong-backend";
    mismatched.dispatch.kernel_id = "test-dawn-only";
    mismatched.dispatch.backend = BackendKind::DawnWebGpu;
    mismatched.dispatch.validation = KernelValidation::Validated;
    assert(register_test_kernel("test-dawn-only", BackendKind::DawnWebGpu).ok());
    size_t mismatch_index = 0;
    assert(mismatched_graph.add_node(mismatched, test_registry, mismatch_index).ok());
    assert(scheduler.execute(backend, mismatched_graph).code == StatusCode::Unsupported);
    assert(backend.submit_count == 0);

    assert(scheduler.execute(backend, graph).ok());
    assert(backend.submit_count == 1);
    assert(backend.sync_count == 1);
    assert(backend.submitted.size() == 3);

    std::atomic<bool> cancelled{true};
    assert(scheduler.execute(backend, graph, &cancelled).code == StatusCode::Cancelled);
    assert(backend.submit_count == 1); // cancelled work was never submitted
}

} // namespace

int main() {
    test_independent_gguf_reader();
    test_tensor_and_memory_lease();
    test_operator_registry_fail_closed();
    test_universal_gles_operator_candidate_is_fail_closed();
    test_model_plan_graph_compilation_is_fail_closed();
    test_llama_model_execution_plan_is_structural_only();
    test_model_graph_compiler_requires_validated_dispatch_builders();
    test_qwen2_dense_adapter_builds_bias_aware_structural_plan();
    test_model_and_quantization_registry_fail_closed();
    test_builtin_architecture_adapter_is_fail_closed();
    test_dawn_add_validation_does_not_promote_fake_backend();
    test_add_operator_candidate_and_reference();
    test_gpu_engine_lifecycle();
    test_gpu_engine_backend_fallback();
    test_gguf_weight_streaming();
    test_tokenizer_engine_contracts();
    test_independent_sampler_and_stop_controller();
    test_q4_decoders();
    test_q5_decoders();
    test_dawn_q8_0_operator_registration_is_unverified();
    test_q8_0_gpu_validation_does_not_promote_fake_backend();
    test_q8_0_decoder_and_registry();
    test_kv_cache_memory_and_sequential_append();
    test_gpu_backend_fallback_order();
    test_runtime_router_fails_closed_to_cpu();
    test_gpu_memory_planner_respects_streaming_and_reserve();
    test_graph_scheduler_order_and_cancel();
    std::cout << "GPU engine contract tests passed\n";
    return 0;
}
