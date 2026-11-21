#include "execution_engine.h"
#include "gguf_reader.h"
#include "model_registry.h"
#include "operators.h"
#include "weight_streamer.h"
#include "sampler.h"
#include "quantization.h"

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
    BackendKind kind() const noexcept override { return BackendKind::OpenGles; }
    Status enumerate_devices(std::vector<DeviceInfo>& out) override {
        out.clear();
        DeviceInfo device;
        device.backend = BackendKind::OpenGles;
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
    Status upload(const std::shared_ptr<Buffer>& destination, uint64_t offset,
                  const void* source, uint64_t size) override {
        auto buffer = std::dynamic_pointer_cast<FakeBuffer>(destination);
        if (!buffer || !source || offset + size > buffer->bytes.size())
            return {StatusCode::InvalidArgument, "fake upload range invalid"};
        std::memcpy(buffer->bytes.data() + offset, source, static_cast<size_t>(size));
        return Status::success();
    }
    Status readback(const std::shared_ptr<Buffer>& source, uint64_t offset,
                    void* destination, uint64_t size) override {
        auto buffer = std::dynamic_pointer_cast<FakeBuffer>(source);
        if (!buffer || !destination || offset + size > buffer->bytes.size())
            return {StatusCode::InvalidArgument, "fake readback range invalid"};
        std::memcpy(destination, buffer->bytes.data() + offset, static_cast<size_t>(size));
        return Status::success();
    }
    Status shutdown() override { ready = false; return Status::success(); }
    bool healthy() const noexcept override { return ready; }

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

    TensorDesc desc;
    desc.shape = {2, 3};
    desc.data_type = DataType::F32;
    desc.memory = MemoryDomain::Device;
    desc.alignment = 4;
    Tensor tensor;
    assert(Tensor::create(desc, backing, 0, tensor).ok());
    assert(tensor.element_count() == 6);
    assert(tensor.byte_size() == 24);

    lease = MemoryLease{};
    assert(memory.allocated_bytes() == 64); // aliased buffer keeps the lease alive
    backing.reset();
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
private:
    ArchitectureDescriptor descriptor_;
};






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
    assert(dispatch.wgsl_source.find("const ELEMENT_COUNT: u32 = 32u;") == std::string::npos);
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
    std::remove(path.c_str());
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

    ComputeGraph graph;
    GraphNode node;
    node.name = "single";
    node.dispatch.label = "single";
    size_t index = 0;
    assert(graph.add_node(node, index).ok());
    assert(engine.execute(graph).ok());
    assert(engine.shutdown().ok());
    assert(!engine.ready());
}

void test_add_operator_candidate_and_reference() {
    OperatorRegistry registry;
    assert(register_builtin_operator_candidates(registry).ok());
    KernelDefinition selected;
    assert(registry.select_kernel("Add", BackendKind::DawnWebGpu, DataType::F32,
                                  DataType::F32, 2, 2, selected).code == StatusCode::Unsupported);

    FakeBackend backend;
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
    assert(!report.gpu_execution_ready);
    assert(report.unverified_quantized_tensors == 1);
}

void test_operator_registry_fail_closed() {
    OperatorRegistry registry;
    KernelDefinition definition;
    definition.operator_name = "Add";
    definition.kernel_id = "add-f32-v1";
    definition.backend = BackendKind::DawnWebGpu;
    definition.input_type = DataType::F32;
    definition.output_type = DataType::F32;
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
}

void test_graph_scheduler_order_and_cancel() {
    FakeBackend backend;
    ComputeGraph graph;
    size_t index = 0;
    GraphNode first;
    first.name = "first";
    first.dispatch.label = "first";
    assert(graph.add_node(first, index).ok());

    GraphNode second;
    second.name = "second";
    second.dependencies = {0};
    second.dispatch.label = "second";
    assert(graph.add_node(second, index).ok());

    GraphNode independent;
    independent.name = "independent";
    independent.dispatch.label = "independent";
    assert(graph.add_node(independent, index).ok());

    std::vector<size_t> order;
    assert(graph.topological_order(order).ok());
    auto first_pos = std::find(order.begin(), order.end(), 0);
    auto second_pos = std::find(order.begin(), order.end(), 1);
    assert(first_pos < second_pos);

    ExecutionScheduler scheduler;
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
    test_model_and_quantization_registry_fail_closed();
    test_builtin_architecture_adapter_is_fail_closed();
    test_add_operator_candidate_and_reference();
    test_gpu_engine_lifecycle();
    test_gguf_weight_streaming();
    test_independent_sampler_and_stop_controller();
    test_q8_0_decoder_and_registry();
    test_graph_scheduler_order_and_cancel();
    std::cout << "GPU engine contract tests passed\n";
    return 0;
}
