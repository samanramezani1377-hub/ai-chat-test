#include "execution_engine.h"
#include "gguf_reader.h"
#include "model_registry.h"
#include "operators.h"
#include "weight_streamer.h"
#include "sampler.h"
#include "quantization.h"
#include "kv_cache.h"
#include "memory_planner.h"
#include "tokenizer.h"
#include "model_graph.h"
#include "model_lowerings.h"

#include <algorithm>
#include <cmath>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
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
    const GgufTensorInfo* streamed_tensor = reader.find_tensor("test.weight");
    assert(streamed_tensor && streamed_tensor->byte_size_known);
    std::vector<uint8_t> first_read(static_cast<size_t>(streamed_tensor->byte_size));
    std::vector<uint8_t> second_read(first_read.size());
    assert(reader.read_tensor_chunk(*streamed_tensor, 0, first_read.data(), first_read.size()).ok());
    assert(reader.read_tensor_chunk(*streamed_tensor, 0, second_read.data(), second_read.size()).ok());
    assert(first_read == second_read);
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

    MemoryManager bounded_pool(backend, 128);
    BufferDesc oversized_for_pool{80, 4, MemoryDomain::Device, false, "oversized-pool-entry"};
    MemoryLease oversized_lease;
    assert(bounded_pool.allocate(oversized_for_pool, oversized_lease).ok());
    oversized_lease = MemoryLease{};
    assert(bounded_pool.allocated_bytes() == 0);
    assert(bounded_pool.available_bytes() == 128); // An entry larger than half-budget is freed, never pooled.

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

    input.total_weight_bytes = 100;
    input.streaming_chunk_bytes = 104; // Per-buffer alignment padding is physically resident.
    input.kv_cache_bytes = 0;
    input.scratch_bytes = 0;
    input.safety_reserve_bytes = 0;
    input.device_budget_bytes = 200;
    report = plan_gpu_memory(input);
    assert(report.fits && report.resident_weight_bytes == 104 &&
           report.required_bytes == 104);

    input.total_weight_bytes = 8ull * 1024 * 1024 * 1024;
    input.streaming_chunk_bytes = 8ull * 1024 * 1024;
    input.kv_cache_bytes = 128ull * 1024 * 1024;
    input.scratch_bytes = 32ull * 1024 * 1024;
    input.safety_reserve_bytes = 64ull * 1024 * 1024;
    input.device_budget_bytes = 512ull * 1024 * 1024;
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

void test_model_gpu_memory_estimate_includes_kv_and_scratch() {
    ModelExecutionPlan model;
    model.structurally_validated = true;
    model.layer_count = 1;
    model.embedding_length = 8;
    model.attention_heads = 2;
    model.kv_heads = 1;
    model.head_dimension = 4;
    model.feed_forward_length = 16;
    ModelGpuMemoryEstimateInput input;
    input.total_weight_bytes = 1024;
    input.streaming_chunk_bytes = 256;
    input.token_count = 2;
    input.context_length = 8;
    input.bytes_per_element = 4;
    input.safety_reserve_bytes = 512;
    input.device_budget_bytes = 4096;
    const auto report = estimate_model_gpu_memory(model, input);
    assert(report.plan.fits);
    assert(report.estimated_kv_cache_bytes == 256);
    assert(report.estimated_scratch_bytes == 576);
    assert(report.plan.resident_weight_bytes == 256);

    input.context_length = UINT64_MAX;
    const auto overflow = estimate_model_gpu_memory(model, input);
    assert(!overflow.blockers.empty());
    assert(overflow.blockers.front() == "MODEL_KV_CACHE_SIZE_OVERFLOW");
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



void test_kv_cache_device_append_avoids_host_roundtrip() {
    FakeBackend backend;
    DeviceInfo device; device.backend = BackendKind::OpenGles;
    assert(backend.initialize(device).ok());
    MemoryManager memory(backend, 4096);
    KvCacheConfig config; config.layers = 1; config.kv_heads = 2;
    config.head_dimension = 2; config.maximum_sequence_length = 4;
    std::unique_ptr<KvCache> cache;
    assert(KvCache::create(backend, memory, config, cache).ok());

    BufferDesc source_desc{32, 4, MemoryDomain::Device, false, "source-kv"};
    std::shared_ptr<Buffer> keys, values;
    assert(backend.allocate(source_desc, keys).ok());
    assert(backend.allocate(source_desc, values).ok());
    auto key_bytes = std::dynamic_pointer_cast<FakeBuffer>(keys);
    auto value_bytes = std::dynamic_pointer_cast<FakeBuffer>(values);
    const float key_data[] = {1, 2, 3, 4};
    const float value_data[] = {5, 6, 7, 8};
    std::memcpy(key_bytes->bytes.data(), key_data, sizeof(key_data));
    std::memcpy(value_bytes->bytes.data(), value_data, sizeof(value_data));
    TensorDesc source_tensor_desc;
    source_tensor_desc.shape = {1, 2, 2};
    source_tensor_desc.data_type = DataType::F32;
    Tensor key_source_tensor, value_source_tensor;
    assert(Tensor::create(source_tensor_desc, keys, 0, key_source_tensor).ok());
    assert(Tensor::create(source_tensor_desc, values, 0, value_source_tensor).ok());
    assert(cache->append_device(0, 0, 1, key_source_tensor, value_source_tensor).ok());
    assert(cache->valid_tokens(0) == 1);
    Tensor cached_key_tensor, cached_value_tensor;
    assert(cache->key_tensor(0, cached_key_tensor).ok());
    assert(cache->value_tensor(0, cached_value_tensor).ok());
    assert(cached_key_tensor.descriptor().shape == std::vector<uint64_t>({1, 2, 2}));
    assert(cached_value_tensor.element_count() == 4);
    auto cached_keys = std::dynamic_pointer_cast<FakeBuffer>(cache->key_buffer(0));
    auto cached_values = std::dynamic_pointer_cast<FakeBuffer>(cache->value_buffer(0));
    assert(std::memcmp(cached_keys->bytes.data(), key_data, sizeof(key_data)) == 0);
    assert(std::memcmp(cached_values->bytes.data(), value_data, sizeof(value_data)) == 0);
    assert(cache->append_device(0, 2, 1, key_source_tensor, value_source_tensor).code == StatusCode::InvalidArgument);
    assert(cache->valid_tokens(0) == 1);
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


void test_f16_bf16_reference_matmul_decoding() {
    const float activations[2]={3.0f,4.0f};
    const uint8_t f16_weights[4]={0x00,0x3c,0x00,0x40};
    const uint8_t bf16_weights[4]={0x80,0x3f,0x00,0x40};
    float f16_output=0.0f,bf16_output=0.0f;
    assert(reference_f16_matmul(activations,f16_weights,4,&f16_output,1,1,2,false).ok());
    assert(reference_f16_matmul(activations,bf16_weights,4,&bf16_output,1,1,2,true).ok());
    assert(f16_output==11.0f&&bf16_output==11.0f);
    const uint8_t half_subnormal[2]={0x01,0x00};
    const float one[1]={1.0f};
    float subnormal_output=0.0f;
    assert(reference_f16_matmul(one,half_subnormal,2,&subnormal_output,1,1,1,false).ok());
    assert(std::fabs(subnormal_output-5.960464477539063e-8f)<1.0e-12f);
}




void test_q2_k_q3_k_decoder_layouts(){
    std::vector<uint8_t> q2(84,0);
    std::fill(q2.begin(),q2.begin()+16,0x01); // 16 scale/min pairs: scale 1, minimum 0
    std::fill(q2.begin()+16,q2.begin()+80,0x55); // packed 2-bit value 1
    q2[80]=0x00;q2[81]=0x3c; // d = 1.0; dmin = 0
    std::vector<float> q2_out(256);assert(decode_q2_k_blocks(q2.data(),q2.size(),q2_out.data(),q2_out.size()).ok());
    for(float value:q2_out)assert(value==1.0f);
    q2[0]=0x11;q2[83]=0x38; // first group uses min=1 and dmin=0.5
    assert(decode_q2_k_blocks(q2.data(),q2.size(),q2_out.data(),q2_out.size()).ok());
    for(size_t i=0;i<16;++i)assert(q2_out[i]==0.5f);
    for(size_t i=16;i<q2_out.size();++i)assert(q2_out[i]==1.0f);
    std::vector<uint8_t> q3(110,0);std::fill(q3.begin(),q3.begin()+32,0xff);
    std::fill(q3.begin()+32,q3.begin()+96,0x55);
    std::fill(q3.begin()+96,q3.begin()+104,0x11);
    std::fill(q3.begin()+104,q3.begin()+108,0xaa);
    q3[108]=0x00;q3[109]=0x3c;
    std::vector<float> q3_out(256);assert(decode_q3_k_blocks(q3.data(),q3.size(),q3_out.data(),q3_out.size()).ok());
    for(float value:q3_out)assert(value==1.0f);
    std::fill(q3.begin(),q3.begin()+32,0x00);
    assert(decode_q3_k_blocks(q3.data(),q3.size(),q3_out.data(),q3_out.size()).ok());
    for(float value:q3_out)assert(value==-3.0f);
}

void test_q4_k_q5_k_decoder_layouts(){
    std::vector<uint8_t> q4(144,0);q4[0]=0x00;q4[1]=0x3c;
    std::fill(q4.begin()+4,q4.begin()+8,1); // first four packed scale fields
    std::fill(q4.begin()+12,q4.begin()+16,1); // remaining four packed scale fields; minima stay zero
    std::fill(q4.begin()+16,q4.end(),0x11);
    std::vector<float> q4_out(256);assert(decode_q4_k_blocks(q4.data(),q4.size(),q4_out.data(),q4_out.size()).ok());
    for(float value:q4_out)assert(value==1.0f);
    std::vector<uint8_t> q5(176,0);q5[0]=0x00;q5[1]=0x3c;
    std::fill(q5.begin()+4,q5.begin()+8,1); // first four packed scale fields
    std::fill(q5.begin()+12,q5.begin()+16,1); // remaining four packed scale fields; minima stay zero
    std::fill(q5.begin()+16,q5.begin()+48,0xff);
    std::fill(q5.begin()+48,q5.end(),0x11);
    std::vector<float> q5_out(256);assert(decode_q5_k_blocks(q5.data(),q5.size(),q5_out.data(),q5_out.size()).ok());
    for(float value:q5_out)assert(value==17.0f);
}

void test_q6_k_decoder_layout() {
    std::vector<uint8_t> block(210,0);
    std::fill(block.begin(),block.begin()+128,0x11);
    std::fill(block.begin()+192,block.begin()+208,1);
    block[208]=0x00;block[209]=0x3c;
    std::vector<float> decoded(256);
    assert(decode_q6_k_blocks(block.data(),block.size(),decoded.data(),decoded.size()).ok());
    for(float value:decoded)assert(value==-31.0f);
    std::fill(block.begin()+128,block.begin()+192,0xff);
    assert(decode_q6_k_blocks(block.data(),block.size(),decoded.data(),decoded.size()).ok());
    for(float value:decoded)assert(value==17.0f);
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
    assert(registry.select_kernel("MatMul_Q4_0", BackendKind::DawnWebGpu,
        std::vector<DataType>{DataType::F32, DataType::U32}, DataType::F32, 2,
        selected).code == StatusCode::Unsupported);
    assert(registry.select_kernel("MatMul_Q4_1", BackendKind::DawnWebGpu,
        std::vector<DataType>{DataType::F32, DataType::U32}, DataType::F32, 2,
        selected).code == StatusCode::Unsupported);
    assert(registry.select_kernel("MatMul_Q5_0", BackendKind::DawnWebGpu,
        std::vector<DataType>{DataType::F32, DataType::U32}, DataType::F32, 2,
        selected).code == StatusCode::Unsupported);
    assert(registry.select_kernel("MatMul_Q5_1", BackendKind::DawnWebGpu,
        std::vector<DataType>{DataType::F32, DataType::U32}, DataType::F32, 2,
        selected).code == StatusCode::Unsupported);
    assert(registry.select_kernel("Embedding_Q4_0", BackendKind::DawnWebGpu,
        std::vector<DataType>{DataType::U32, DataType::U32}, DataType::F32, 2,
        selected).code == StatusCode::Unsupported);
    assert(registry.select_kernel("Embedding_Q4_1", BackendKind::DawnWebGpu,
        std::vector<DataType>{DataType::U32, DataType::U32}, DataType::F32, 2,
        selected).code == StatusCode::Unsupported);
}



void test_q5_matmul_lowerings_are_registered_for_both_backends() {
    ModelLoweringRegistry dawn;
    ModelLoweringRegistry gles;
    assert(register_builtin_model_lowerings(BackendKind::DawnWebGpu,dawn).ok());
    assert(register_builtin_model_lowerings(BackendKind::OpenGles,gles).ok());
    assert(dawn.find(BackendKind::DawnWebGpu,"MatMul_Q5_0") != nullptr);
    assert(dawn.find(BackendKind::DawnWebGpu,"MatMul_Q5_1") != nullptr);
    assert(gles.find(BackendKind::OpenGles,"MatMul_Q5_0") != nullptr);
    assert(gles.find(BackendKind::OpenGles,"MatMul_Q5_1") != nullptr);
    assert(dawn.find(BackendKind::DawnWebGpu,"Embedding_Q5_0") != nullptr);
    assert(dawn.find(BackendKind::DawnWebGpu,"Embedding_Q5_1") != nullptr);
    assert(gles.find(BackendKind::OpenGles,"Embedding_Q5_0") != nullptr);
    assert(gles.find(BackendKind::OpenGles,"Embedding_Q5_1") != nullptr);
    assert(dawn.find(BackendKind::DawnWebGpu,"MatMul_F16") != nullptr);
    assert(dawn.find(BackendKind::DawnWebGpu,"MatMul_BF16") != nullptr);
    assert(dawn.find(BackendKind::DawnWebGpu,"Embedding_F16") != nullptr);
    assert(dawn.find(BackendKind::DawnWebGpu,"Embedding_BF16") != nullptr);
    assert(dawn.find(BackendKind::DawnWebGpu,"MatMul_Q6_K") != nullptr);
    assert(gles.find(BackendKind::OpenGles,"MatMul_Q6_K") != nullptr);
    assert(dawn.find(BackendKind::DawnWebGpu,"MatMul_Q4_K") != nullptr);
    assert(dawn.find(BackendKind::DawnWebGpu,"MatMul_Q5_K") != nullptr);
    assert(gles.find(BackendKind::OpenGles,"MatMul_Q4_K") != nullptr);
    assert(gles.find(BackendKind::OpenGles,"MatMul_Q5_K") != nullptr);
    assert(dawn.find(BackendKind::DawnWebGpu,"MatMul_Q2_K") != nullptr);
    assert(dawn.find(BackendKind::DawnWebGpu,"MatMul_Q3_K") != nullptr);
    assert(gles.find(BackendKind::OpenGles,"MatMul_Q2_K") != nullptr);
    assert(gles.find(BackendKind::OpenGles,"MatMul_Q3_K") != nullptr);
    assert(gles.find(BackendKind::OpenGles,"MatMul_F16") != nullptr);
    assert(gles.find(BackendKind::OpenGles,"MatMul_BF16") != nullptr);
    assert(gles.find(BackendKind::OpenGles,"Embedding_F16") != nullptr);
    assert(gles.find(BackendKind::OpenGles,"Embedding_BF16") != nullptr);
    assert(dawn.find(BackendKind::DawnWebGpu,"MatMul_Q6_K") != nullptr);
    assert(gles.find(BackendKind::OpenGles,"MatMul_Q6_K") != nullptr);
    assert(dawn.find(BackendKind::DawnWebGpu,"MatMul_F16") != nullptr);
    assert(dawn.find(BackendKind::DawnWebGpu,"MatMul_BF16") != nullptr);
    assert(dawn.find(BackendKind::DawnWebGpu,"Embedding_F16") != nullptr);
    assert(dawn.find(BackendKind::DawnWebGpu,"Embedding_BF16") != nullptr);
    assert(gles.find(BackendKind::OpenGles,"MatMul_F16") != nullptr);
    assert(gles.find(BackendKind::OpenGles,"MatMul_BF16") != nullptr);
    assert(gles.find(BackendKind::OpenGles,"Embedding_F16") != nullptr);
    assert(gles.find(BackendKind::OpenGles,"Embedding_BF16") != nullptr);
}

void test_q8_0_gpu_validation_does_not_promote_fake_backend() {
    FakeBackend backend;
    OperatorRegistry operators;
    QuantizationRegistry quantizations;
    assert(register_builtin_gles_operator_candidates(operators).ok());
    assert(register_builtin_quantized_operator_candidates(operators).ok());
    assert(register_builtin_quantization_formats(quantizations).ok());
    KernelDefinition selected;
    assert(validate_q8_0_matmul_kernel(backend, operators, quantizations).code ==
           StatusCode::NumericalFailure);
    assert(validate_q8_0_embedding_kernel(backend, operators).code ==
           StatusCode::NumericalFailure);
    assert(validate_q4_0_matmul_kernel(backend, operators, quantizations).code ==
           StatusCode::NumericalFailure);
    assert(validate_q4_1_matmul_kernel(backend, operators, quantizations).code ==
           StatusCode::NumericalFailure);
    assert(validate_q5_0_matmul_kernel(backend, operators, quantizations).code ==
           StatusCode::NumericalFailure);
    assert(validate_q5_1_matmul_kernel(backend, operators, quantizations).code ==
           StatusCode::NumericalFailure);
    assert(validate_q5_0_embedding_kernel(backend, operators, quantizations).code == StatusCode::NumericalFailure);
    assert(validate_q5_1_embedding_kernel(backend, operators, quantizations).code == StatusCode::NumericalFailure);
    assert(validate_f16_matmul_kernel(backend,operators,quantizations).code==StatusCode::NumericalFailure);
    assert(validate_bf16_matmul_kernel(backend,operators,quantizations).code==StatusCode::NumericalFailure);
    assert(validate_f16_embedding_kernel(backend,operators,quantizations).code==StatusCode::NumericalFailure);
    assert(validate_bf16_embedding_kernel(backend,operators,quantizations).code==StatusCode::NumericalFailure);
    assert(validate_q6_k_matmul_kernel(backend,operators,quantizations).code==StatusCode::NumericalFailure);
    assert(validate_q4_k_matmul_kernel(backend,operators,quantizations).code==StatusCode::NumericalFailure);
    assert(validate_q5_k_matmul_kernel(backend,operators,quantizations).code==StatusCode::NumericalFailure);
    assert(validate_q2_k_matmul_kernel(backend,operators,quantizations).code==StatusCode::NumericalFailure);
    assert(validate_q3_k_matmul_kernel(backend,operators,quantizations).code==StatusCode::NumericalFailure);
    FakeBackend dawn_backend(BackendKind::DawnWebGpu);
    OperatorRegistry dawn_operators;
    QuantizationRegistry dawn_quantizations;
    assert(register_builtin_quantized_operator_candidates(dawn_operators).ok());
    assert(register_builtin_quantization_formats(dawn_quantizations).ok());
    assert(validate_q4_1_matmul_kernel(dawn_backend, dawn_operators, dawn_quantizations).code ==
           StatusCode::NumericalFailure);
    assert(validate_q8_0_embedding_kernel(dawn_backend, dawn_operators).code ==
           StatusCode::NumericalFailure);
    assert(validate_q4_0_embedding_kernel(backend, operators).code ==
           StatusCode::NumericalFailure);
    assert(validate_q4_1_embedding_kernel(backend, operators).code ==
           StatusCode::NumericalFailure);
    assert(validate_q4_1_matmul_kernel(backend, operators, quantizations).code ==
           StatusCode::NumericalFailure);
    const QuantizationFormat* q4_1_format = quantizations.find(3);
    assert(q4_1_format != nullptr && q4_1_format->gpu_kernel_state == SupportState::Unverified);
    assert(!quantizations.can_execute_on_gpu(3, BackendKind::DawnWebGpu));
    assert(!quantizations.can_execute_on_gpu(3, BackendKind::OpenGles));
    const QuantizationFormat* format = quantizations.find(8);
    const QuantizationFormat* q4_format = quantizations.find(2);
    assert(q4_format != nullptr && q4_format->gpu_kernel_state == SupportState::Unverified);
    assert(!quantizations.can_execute_on_gpu(2, BackendKind::DawnWebGpu));
    assert(!quantizations.can_execute_on_gpu(2, BackendKind::OpenGles));
    const QuantizationFormat* q5_0_format = quantizations.find(6);
    const QuantizationFormat* q5_1_format = quantizations.find(7);
    assert(q5_0_format && q5_0_format->gpu_kernel_state == SupportState::Unverified);
    assert(q5_1_format && q5_1_format->gpu_kernel_state == SupportState::Unverified);
    assert(!quantizations.can_execute_on_gpu(6, BackendKind::DawnWebGpu));
    assert(!quantizations.can_execute_on_gpu(6, BackendKind::OpenGles));
    assert(!quantizations.can_execute_on_gpu(7, BackendKind::DawnWebGpu));
    assert(!quantizations.can_execute_on_gpu(7, BackendKind::OpenGles));
    const QuantizationFormat* f16_format=quantizations.find(1);
    const QuantizationFormat* bf16_format=quantizations.find(30);
    assert(f16_format&&f16_format->gpu_kernel_state==SupportState::Unverified);
    assert(bf16_format&&bf16_format->gpu_kernel_state==SupportState::Unverified);
    assert(!quantizations.can_execute_on_gpu(1,BackendKind::DawnWebGpu));
    assert(!quantizations.can_execute_on_gpu(30,BackendKind::OpenGles));
    const QuantizationFormat* q6_k_format=quantizations.find(14);
    assert(q6_k_format&&q6_k_format->decoder_state==SupportState::Validated);
    assert(q6_k_format->gpu_kernel_state==SupportState::Unverified);
    assert(!quantizations.can_execute_on_gpu(14,BackendKind::DawnWebGpu));
    const QuantizationFormat* q4_k_format=quantizations.find(12);
    const QuantizationFormat* q5_k_format=quantizations.find(13);
    assert(q4_k_format&&q4_k_format->decoder_state==SupportState::Validated&&q4_k_format->gpu_kernel_state==SupportState::Unverified);
    assert(q5_k_format&&q5_k_format->decoder_state==SupportState::Validated&&q5_k_format->gpu_kernel_state==SupportState::Unverified);
    assert(!quantizations.can_execute_on_gpu(12,BackendKind::OpenGles));
    assert(!quantizations.can_execute_on_gpu(13,BackendKind::DawnWebGpu));
    const QuantizationFormat* q2_k_format=quantizations.find(10);
    const QuantizationFormat* q3_k_format=quantizations.find(11);
    assert(q2_k_format&&q2_k_format->decoder_state==SupportState::Validated&&q2_k_format->gpu_kernel_state==SupportState::Unverified);
    assert(q3_k_format&&q3_k_format->decoder_state==SupportState::Validated&&q3_k_format->gpu_kernel_state==SupportState::Unverified);
    assert(!quantizations.can_execute_on_gpu(10,BackendKind::DawnWebGpu));
    assert(!quantizations.can_execute_on_gpu(11,BackendKind::OpenGles));
    assert(operators.select_kernel("MatMul_Q6_K",BackendKind::DawnWebGpu,
        std::vector<DataType>{DataType::F32,DataType::U32},DataType::F32,2,selected).code==StatusCode::Unsupported);
    assert(operators.select_kernel("MatMul_Q4_K",BackendKind::DawnWebGpu,
        std::vector<DataType>{DataType::F32,DataType::U32},DataType::F32,2,selected).code==StatusCode::Unsupported);
    assert(operators.select_kernel("MatMul_Q5_K",BackendKind::OpenGles,
        std::vector<DataType>{DataType::F32,DataType::U32},DataType::F32,2,selected).code==StatusCode::Unsupported);
    assert(operators.select_kernel("MatMul_Q2_K",BackendKind::DawnWebGpu,
        std::vector<DataType>{DataType::F32,DataType::U32},DataType::F32,2,selected).code==StatusCode::Unsupported);
    assert(operators.select_kernel("MatMul_Q3_K",BackendKind::OpenGles,
        std::vector<DataType>{DataType::F32,DataType::U32},DataType::F32,2,selected).code==StatusCode::Unsupported);
    KernelDefinition q5_embedding_selected;
    assert(operators.select_kernel("Embedding_Q5_0", BackendKind::OpenGles,
        std::vector<DataType>{DataType::U32,DataType::U32}, DataType::F32, 2,
        q5_embedding_selected).code == StatusCode::Unsupported);
    assert(operators.select_kernel("Embedding_Q5_1", BackendKind::OpenGles,
        std::vector<DataType>{DataType::U32,DataType::U32}, DataType::F32, 2,
        q5_embedding_selected).code == StatusCode::Unsupported);
    assert(operators.select_kernel("MatMul_F16",BackendKind::DawnWebGpu,
        std::vector<DataType>{DataType::F32,DataType::U32},DataType::F32,2,selected).code==StatusCode::Unsupported);
    assert(operators.select_kernel("MatMul_BF16",BackendKind::DawnWebGpu,
        std::vector<DataType>{DataType::F32,DataType::U32},DataType::F32,2,selected).code==StatusCode::Unsupported);
    assert(operators.select_kernel("Embedding_F16",BackendKind::DawnWebGpu,
        std::vector<DataType>{DataType::U32,DataType::U32},DataType::F32,2,q5_embedding_selected).code==StatusCode::Unsupported);
    assert(operators.select_kernel("Embedding_BF16",BackendKind::DawnWebGpu,
        std::vector<DataType>{DataType::U32,DataType::U32},DataType::F32,2,q5_embedding_selected).code==StatusCode::Unsupported);
    assert(format != nullptr && format->gpu_kernel_state == SupportState::Unverified);
    assert(!quantizations.can_execute_on_gpu(8, BackendKind::DawnWebGpu));
    assert(!quantizations.can_execute_on_gpu(8, BackendKind::OpenGles));
    assert(operators.select_kernel("MatMul_Q8_0", BackendKind::OpenGles,
        std::vector<DataType>{DataType::F32, DataType::U32}, DataType::F32, 2,
        selected).code == StatusCode::Unsupported);
}


void test_quantization_backend_compatibility_is_backend_specific() {
    QuantizationRegistry registry;
    assert(register_builtin_quantization_formats(registry).ok());
    GgufFileInfo model;
    GgufTensorInfo tensor;
    tensor.name = "blk.0.attn_q.weight";
    tensor.type_id = 8;
    tensor.type = GgufReader::tensor_type_info(8);
    tensor.byte_size_known = true;
    tensor.byte_size = 68;
    model.tensors.push_back(tensor);
    const auto dawn = evaluate_model_quantization_backend(model, registry, BackendKind::DawnWebGpu);
    const auto gles = evaluate_model_quantization_backend(model, registry, BackendKind::OpenGles);
    assert(!dawn.compatible && !gles.compatible);
    assert(!dawn.blockers.empty() && !gles.blockers.empty());
    const auto unknown = evaluate_model_quantization_backend(model, registry, BackendKind::Unknown);
    assert(!unknown.compatible && !unknown.blockers.empty());

    QuantizationRegistry mixed_backend_registry;
    QuantizationFormat mixed_backend_q8;
    mixed_backend_q8.file_type_id = 8;
    mixed_backend_q8.name = "Q8_0";
    mixed_backend_q8.block_elements = 32;
    mixed_backend_q8.bytes_per_block = 34;
    mixed_backend_q8.quantized = true;
    mixed_backend_q8.decoder_state = SupportState::Validated;
    mixed_backend_q8.gpu_kernel_state = SupportState::Validated;
    mixed_backend_q8.backend_gpu_kernel_states[BackendKind::DawnWebGpu] = SupportState::Validated;
    mixed_backend_q8.backend_gpu_kernel_states[BackendKind::OpenGles] = SupportState::Unverified;
    assert(mixed_backend_registry.register_format(std::move(mixed_backend_q8)).ok());
    assert(mixed_backend_registry.can_execute_on_gpu(8, BackendKind::DawnWebGpu));
    assert(!mixed_backend_registry.can_execute_on_gpu(8, BackendKind::OpenGles));
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

    std::vector<uint8_t> subnormal_scale(34, 0);
    subnormal_scale[0] = 0x01; // smallest positive IEEE-754 binary16 subnormal
    subnormal_scale[2] = 1;
    float subnormal_decoded[32] = {};
    assert(decode_q8_0_blocks(subnormal_scale.data(), subnormal_scale.size(),
                              subnormal_decoded, 32).ok());
    assert(std::fabs(subnormal_decoded[0] - 5.960464477539063e-8f) < 1.0e-12f);

    encoded[0] = 0x00; encoded[1] = 0x7e; // NaN scale
    assert(decode_q8_0_blocks(encoded.data(), encoded.size(), decoded, 32).code ==
           StatusCode::NumericalFailure);

    QuantizationRegistry registry;
    assert(register_builtin_quantization_formats(registry).ok());
    assert(registry.can_decode(8));
    assert(!registry.can_execute_on_gpu(8));
    assert(registry.can_decode(2));
    assert(!registry.can_execute_on_gpu(2));
    assert(registry.can_decode(3));
    assert(!registry.can_execute_on_gpu(3));

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
    ComputeDispatch q8_gles_dequant;
    assert(build_q8_0_dequantize_gles_candidate(encoded_buffer, 34, output_tensor, q8_gles_dequant).ok());
    assert(q8_gles_dequant.backend == BackendKind::OpenGles);
    assert(q8_gles_dequant.glsl_source.find("#version 310 es") == 0);
    assert(dispatch.bindings.size() == 2);
    assert(dispatch.label.find("unverified") != std::string::npos);

    BufferDesc q4_matmul_weights_desc{72, 4, MemoryDomain::Device, true, "q4-matmul-weights"};
    std::shared_ptr<Buffer> q4_matmul_weights;
    assert(backend.allocate(q4_matmul_weights_desc, q4_matmul_weights).ok());
    BufferDesc q4_activation_desc{2 * 64 * sizeof(float), 4, MemoryDomain::Device, false, "q4-matmul-activation"};
    std::shared_ptr<Buffer> q4_activation_buffer;
    assert(backend.allocate(q4_activation_desc, q4_activation_buffer).ok());
    BufferDesc q4_matmul_output_desc{2 * 2 * sizeof(float), 4, MemoryDomain::Device, false, "q4-matmul-output"};
    std::shared_ptr<Buffer> q4_matmul_output_buffer;
    assert(backend.allocate(q4_matmul_output_desc, q4_matmul_output_buffer).ok());
    TensorDesc q4_activation_tensor_desc; q4_activation_tensor_desc.shape = {2, 64};
    q4_activation_tensor_desc.data_type = DataType::F32;
    TensorDesc q4_matmul_output_tensor_desc; q4_matmul_output_tensor_desc.shape = {2, 2};
    q4_matmul_output_tensor_desc.data_type = DataType::F32;
    Tensor q4_activation_tensor, q4_matmul_output_tensor;
    assert(Tensor::create(q4_activation_tensor_desc, q4_activation_buffer, 0, q4_activation_tensor).ok());
    assert(Tensor::create(q4_matmul_output_tensor_desc, q4_matmul_output_buffer, 0, q4_matmul_output_tensor).ok());
    ComputeDispatch q4_matmul_dispatch;
    assert(build_q4_0_matmul_candidate(q4_activation_tensor, q4_matmul_weights, 72,
                                       q4_matmul_output_tensor, q4_matmul_dispatch).ok());
    assert(q4_matmul_dispatch.kernel_id == "dawn-q4-0-matmul-f32-activation-v1");
    assert(q4_matmul_dispatch.wgsl_source.find("WEIGHT_ROW_BYTES: u32 = 36u") != std::string::npos);
    ComputeDispatch q4_gles_dispatch;
    assert(build_q4_0_matmul_gles_candidate(q4_activation_tensor, q4_matmul_weights, 72,
                                            q4_matmul_output_tensor, q4_gles_dispatch).ok());
    assert(q4_gles_dispatch.backend == BackendKind::OpenGles);
    assert(q4_gles_dispatch.glsl_source.find("WEIGHT_ROW_BYTES = 36u") != std::string::npos);

    BufferDesc q4_1_weights_desc{80, 4, MemoryDomain::Device, true, "q4-1-matmul-weights"};
    std::shared_ptr<Buffer> q4_1_weights;
    assert(backend.allocate(q4_1_weights_desc, q4_1_weights).ok());
    ComputeDispatch q4_1_dispatch;
    assert(build_q4_1_matmul_candidate(q4_activation_tensor, q4_1_weights, 80,
                                       q4_matmul_output_tensor, q4_1_dispatch).ok());
    assert(q4_1_dispatch.kernel_id == "dawn-q4-1-matmul-f32-activation-v1");
    assert(q4_1_dispatch.wgsl_source.find("WEIGHT_ROW_BYTES: u32 = 40u") != std::string::npos);
    ComputeDispatch q4_1_gles_dispatch;
    assert(build_q4_1_matmul_gles_candidate(q4_activation_tensor, q4_1_weights, 80,
                                            q4_matmul_output_tensor, q4_1_gles_dispatch).ok());
    assert(q4_1_gles_dispatch.backend == BackendKind::OpenGles);
    assert(q4_1_gles_dispatch.glsl_source.find("WEIGHT_ROW_BYTES = 40u") != std::string::npos);
    BufferDesc q5_0_weights_desc{88, 4, MemoryDomain::Device, true, "q5-0-matmul-weights"};
    BufferDesc q5_1_weights_desc{96, 4, MemoryDomain::Device, true, "q5-1-matmul-weights"};
    std::shared_ptr<Buffer> q5_0_weights, q5_1_weights;
    assert(backend.allocate(q5_0_weights_desc, q5_0_weights).ok());
    assert(backend.allocate(q5_1_weights_desc, q5_1_weights).ok());
    ComputeDispatch q5_0_matmul_dispatch, q5_1_matmul_dispatch;
    assert(build_q5_0_matmul_candidate(q4_activation_tensor, q5_0_weights, 88,
        q4_matmul_output_tensor, q5_0_matmul_dispatch).ok());
    assert(build_q5_1_matmul_candidate(q4_activation_tensor, q5_1_weights, 96,
        q4_matmul_output_tensor, q5_1_matmul_dispatch).ok());
    assert(q5_0_matmul_dispatch.kernel_id == "dawn-q5-0-matmul-f32-activation-v1");
    assert(q5_1_matmul_dispatch.kernel_id == "dawn-q5-1-matmul-f32-activation-v1");
    assert(q5_0_matmul_dispatch.wgsl_source.find("WEIGHT_ROW_BYTES: u32 = 44u") != std::string::npos);
    assert(q5_1_matmul_dispatch.wgsl_source.find("WEIGHT_ROW_BYTES: u32 = 48u") != std::string::npos);
    ComputeDispatch q5_0_matmul_gles_dispatch, q5_1_matmul_gles_dispatch;
    assert(build_q5_0_matmul_gles_candidate(q4_activation_tensor, q5_0_weights, 88,
        q4_matmul_output_tensor, q5_0_matmul_gles_dispatch).ok());
    assert(build_q5_1_matmul_gles_candidate(q4_activation_tensor, q5_1_weights, 96,
        q4_matmul_output_tensor, q5_1_matmul_gles_dispatch).ok());
    assert(q5_0_matmul_gles_dispatch.backend == BackendKind::OpenGles);
    assert(q5_1_matmul_gles_dispatch.backend == BackendKind::OpenGles);
    assert(q5_0_matmul_gles_dispatch.glsl_source.find("WEIGHT_ROW_BYTES = 44u") != std::string::npos);
    assert(q5_1_matmul_gles_dispatch.glsl_source.find("WEIGHT_ROW_BYTES = 48u") != std::string::npos);
    BufferDesc q6_k_weights_desc{420,4,MemoryDomain::Device,true,"q6-k-matmul-weights"};
    std::shared_ptr<Buffer> q6_k_weights;assert(backend.allocate(q6_k_weights_desc,q6_k_weights).ok());
    ComputeDispatch q6_k_dawn,q6_k_gles;
    assert(build_q6_k_matmul_candidate(q4_activation_tensor,q6_k_weights,420,q4_matmul_output_tensor,q6_k_dawn).code==StatusCode::InvalidArgument);
    TensorDesc q6_a_desc;q6_a_desc.shape={2,256};q6_a_desc.data_type=DataType::F32;
    TensorDesc q6_o_desc;q6_o_desc.shape={2,2};q6_o_desc.data_type=DataType::F32;
    BufferDesc q6_a_buf_desc{2*256*sizeof(float),4,MemoryDomain::Device,false,"q6-k-activations"};
    BufferDesc q6_o_buf_desc{2*2*sizeof(float),4,MemoryDomain::Device,false,"q6-k-output"};
    std::shared_ptr<Buffer> q6_a_buf,q6_o_buf;assert(backend.allocate(q6_a_buf_desc,q6_a_buf).ok());assert(backend.allocate(q6_o_buf_desc,q6_o_buf).ok());
    Tensor q6_a_tensor,q6_o_tensor;assert(Tensor::create(q6_a_desc,q6_a_buf,0,q6_a_tensor).ok());assert(Tensor::create(q6_o_desc,q6_o_buf,0,q6_o_tensor).ok());
    assert(build_q6_k_matmul_candidate(q6_a_tensor,q6_k_weights,420,q6_o_tensor,q6_k_dawn).ok());
    assert(build_q6_k_matmul_gles_candidate(q6_a_tensor,q6_k_weights,420,q6_o_tensor,q6_k_gles).ok());
    assert(q6_k_dawn.wgsl_source.find("WEIGHT_ROW_BYTES: u32 = 210u")!=std::string::npos);
    assert(q6_k_gles.backend==BackendKind::OpenGles&&q6_k_gles.glsl_source.find("#version 310 es")==0);
    BufferDesc q4k_weights_desc{288,4,MemoryDomain::Device,true,"q4-k-weights"};
    BufferDesc q5k_weights_desc{352,4,MemoryDomain::Device,true,"q5-k-weights"};
    std::shared_ptr<Buffer> q4k_weights,q5k_weights;assert(backend.allocate(q4k_weights_desc,q4k_weights).ok());assert(backend.allocate(q5k_weights_desc,q5k_weights).ok());
    ComputeDispatch q4k_dawn,q5k_dawn,q4k_gles,q5k_gles;
    assert(build_q4_k_matmul_candidate(q6_a_tensor,q4k_weights,288,q6_o_tensor,q4k_dawn).ok());
    assert(build_q5_k_matmul_candidate(q6_a_tensor,q5k_weights,352,q6_o_tensor,q5k_dawn).ok());
    assert(build_q4_k_matmul_gles_candidate(q6_a_tensor,q4k_weights,288,q6_o_tensor,q4k_gles).ok());
    assert(build_q5_k_matmul_gles_candidate(q6_a_tensor,q5k_weights,352,q6_o_tensor,q5k_gles).ok());
    assert(q4k_gles.backend==BackendKind::OpenGles&&q5k_gles.backend==BackendKind::OpenGles);
    assert(q4k_dawn.wgsl_source.find("WEIGHT_ROW_BYTES:u32=144u")!=std::string::npos);
    assert(q5k_dawn.wgsl_source.find("WEIGHT_ROW_BYTES:u32=176u")!=std::string::npos);
    BufferDesc q2k_weights_desc{168,4,MemoryDomain::Device,true,"q2-k-weights"};
    BufferDesc q3k_weights_desc{220,4,MemoryDomain::Device,true,"q3-k-weights"};
    std::shared_ptr<Buffer> q2k_weights,q3k_weights;assert(backend.allocate(q2k_weights_desc,q2k_weights).ok());assert(backend.allocate(q3k_weights_desc,q3k_weights).ok());
    ComputeDispatch q2k_dawn,q3k_dawn,q2k_gles,q3k_gles;
    assert(build_q2_k_matmul_candidate(q6_a_tensor,q2k_weights,168,q6_o_tensor,q2k_dawn).ok());
    assert(build_q3_k_matmul_candidate(q6_a_tensor,q3k_weights,220,q6_o_tensor,q3k_dawn).ok());
    assert(build_q2_k_matmul_gles_candidate(q6_a_tensor,q2k_weights,168,q6_o_tensor,q2k_gles).ok());
    assert(build_q3_k_matmul_gles_candidate(q6_a_tensor,q3k_weights,220,q6_o_tensor,q3k_gles).ok());
    assert(q2k_gles.backend==BackendKind::OpenGles&&q3k_gles.backend==BackendKind::OpenGles);
    assert(q2k_dawn.wgsl_source.find("WEIGHT_ROW_BYTES:u32=84u")!=std::string::npos);
    assert(q3k_dawn.wgsl_source.find("WEIGHT_ROW_BYTES:u32=110u")!=std::string::npos);
    BufferDesc f16_weights_desc{256,4,MemoryDomain::Device,true,"f16-matmul-weights"};
    BufferDesc bf16_weights_desc{256,4,MemoryDomain::Device,true,"bf16-matmul-weights"};
    std::shared_ptr<Buffer> f16_weights,bf16_weights;
    assert(backend.allocate(f16_weights_desc,f16_weights).ok());
    assert(backend.allocate(bf16_weights_desc,bf16_weights).ok());
    ComputeDispatch f16_dawn,bf16_dawn,f16_gles,bf16_gles;
    assert(build_f16_matmul_candidate(q4_activation_tensor,f16_weights,256,q4_matmul_output_tensor,f16_dawn).ok());
    assert(build_bf16_matmul_candidate(q4_activation_tensor,bf16_weights,256,q4_matmul_output_tensor,bf16_dawn).ok());
    assert(build_f16_matmul_gles_candidate(q4_activation_tensor,f16_weights,256,q4_matmul_output_tensor,f16_gles).ok());
    assert(build_bf16_matmul_gles_candidate(q4_activation_tensor,bf16_weights,256,q4_matmul_output_tensor,bf16_gles).ok());
    assert(f16_gles.backend==BackendKind::OpenGles&&bf16_gles.backend==BackendKind::OpenGles);
    assert(f16_gles.glsl_source.find("#version 310 es")==0&&bf16_gles.glsl_source.find("#version 310 es")==0);
    BufferDesc fp16_embed_weights_desc{32*2*2,4,MemoryDomain::Device,true,"f16-embedding-weights"};
    BufferDesc fp16_embed_tokens_desc{2*sizeof(uint32_t),4,MemoryDomain::Device,false,"f16-embedding-token-ids"};
    BufferDesc fp16_embed_output_desc{2*32*sizeof(float),4,MemoryDomain::Device,false,"f16-embedding-output"};
    std::shared_ptr<Buffer> fp16_embed_weights,fp16_embed_tokens,fp16_embed_output;
    assert(backend.allocate(fp16_embed_weights_desc,fp16_embed_weights).ok());
    assert(backend.allocate(fp16_embed_tokens_desc,fp16_embed_tokens).ok());
    assert(backend.allocate(fp16_embed_output_desc,fp16_embed_output).ok());
    TensorDesc fp16_embed_td;fp16_embed_td.shape={2,32};fp16_embed_td.data_type=DataType::F32;
    Tensor fp16_embed_tensor;assert(Tensor::create(fp16_embed_td,fp16_embed_output,0,fp16_embed_tensor).ok());
    const std::vector<uint32_t> fp16_embed_ids={1,0};
    ComputeDispatch f16_embed_dawn,bf16_embed_dawn,f16_embed_gles,bf16_embed_gles;
    assert(build_f16_embedding_candidate(fp16_embed_weights,128,2,32,fp16_embed_tokens,fp16_embed_ids,fp16_embed_tensor,f16_embed_dawn).ok());
    assert(build_bf16_embedding_candidate(fp16_embed_weights,128,2,32,fp16_embed_tokens,fp16_embed_ids,fp16_embed_tensor,bf16_embed_dawn).ok());
    assert(build_f16_embedding_gles_candidate(fp16_embed_weights,128,2,32,fp16_embed_tokens,fp16_embed_ids,fp16_embed_tensor,f16_embed_gles).ok());
    assert(build_bf16_embedding_gles_candidate(fp16_embed_weights,128,2,32,fp16_embed_tokens,fp16_embed_ids,fp16_embed_tensor,bf16_embed_gles).ok());
    assert(f16_embed_gles.backend==BackendKind::OpenGles&&bf16_embed_gles.backend==BackendKind::OpenGles);
    BufferDesc q5_embedding_weights_desc{48,4,MemoryDomain::Device,true,"q5-embedding-weights"};
    BufferDesc q5_embedding_tokens_desc{2*sizeof(uint32_t),4,MemoryDomain::Device,false,"q5-embedding-token-ids"};
    BufferDesc q5_embedding_output_desc{2*32*sizeof(float),4,MemoryDomain::Device,false,"q5-embedding-output"};
    std::shared_ptr<Buffer> q5_embedding_weights,q5_embedding_tokens,q5_embedding_output;
    assert(backend.allocate(q5_embedding_weights_desc,q5_embedding_weights).ok());
    assert(backend.allocate(q5_embedding_tokens_desc,q5_embedding_tokens).ok());
    assert(backend.allocate(q5_embedding_output_desc,q5_embedding_output).ok());
    TensorDesc q5_embedding_td;q5_embedding_td.shape={2,32};q5_embedding_td.data_type=DataType::F32;
    Tensor q5_embedding_tensor;assert(Tensor::create(q5_embedding_td,q5_embedding_output,0,q5_embedding_tensor).ok());
    const std::vector<uint32_t> q5_embedding_ids={1,0};
    ComputeDispatch q5_0_embedding_dispatch,q5_1_embedding_dispatch;
    assert(build_q5_0_embedding_candidate(q5_embedding_weights,44,2,32,q5_embedding_tokens,q5_embedding_ids,q5_embedding_tensor,q5_0_embedding_dispatch).ok());
    assert(build_q5_1_embedding_candidate(q5_embedding_weights,48,2,32,q5_embedding_tokens,q5_embedding_ids,q5_embedding_tensor,q5_1_embedding_dispatch).ok());
    ComputeDispatch q5_0_embedding_gles_dispatch,q5_1_embedding_gles_dispatch;
    assert(build_q5_0_embedding_gles_candidate(q5_embedding_weights,44,2,32,q5_embedding_tokens,q5_embedding_ids,q5_embedding_tensor,q5_0_embedding_gles_dispatch).ok());
    assert(build_q5_1_embedding_gles_candidate(q5_embedding_weights,48,2,32,q5_embedding_tokens,q5_embedding_ids,q5_embedding_tensor,q5_1_embedding_gles_dispatch).ok());
    assert(q5_0_embedding_gles_dispatch.backend==BackendKind::OpenGles&&q5_1_embedding_gles_dispatch.backend==BackendKind::OpenGles);
    assert(q5_0_embedding_gles_dispatch.glsl_source.find("#version 310 es")==0);
    assert(q5_1_embedding_gles_dispatch.glsl_source.find("#version 310 es")==0);



    BufferDesc q4_embedding_weights_desc{36, 4, MemoryDomain::Device, true, "q4-embedding-weights"};
    BufferDesc q4_embedding_tokens_desc{2 * sizeof(uint32_t), 4, MemoryDomain::Device, false, "q4-embedding-token-ids"};
    BufferDesc q4_embedding_output_desc{2 * 32 * sizeof(float), 4, MemoryDomain::Device, false, "q4-embedding-output"};
    std::shared_ptr<Buffer> q4_embedding_weights, q4_embedding_tokens, q4_embedding_output;
    assert(backend.allocate(q4_embedding_weights_desc, q4_embedding_weights).ok());
    assert(backend.allocate(q4_embedding_tokens_desc, q4_embedding_tokens).ok());
    assert(backend.allocate(q4_embedding_output_desc, q4_embedding_output).ok());
    TensorDesc q4_embedding_output_td; q4_embedding_output_td.shape = {2, 32};
    q4_embedding_output_td.data_type = DataType::F32;
    Tensor q4_embedding_output_tensor;
    assert(Tensor::create(q4_embedding_output_td, q4_embedding_output, 0,
                          q4_embedding_output_tensor).ok());
    const std::vector<uint32_t> q4_embedding_ids = {1, 0};
    ComputeDispatch q4_embedding_dispatch;
    assert(build_q4_0_embedding_candidate(q4_embedding_weights, 36, 2, 32,
        q4_embedding_tokens, q4_embedding_ids, q4_embedding_output_tensor,
        q4_embedding_dispatch).ok());
    assert(q4_embedding_dispatch.kernel_id == "dawn-q4-0-embedding-f32-v1");
    assert(q4_embedding_dispatch.wgsl_source.find("WEIGHT_ROW_BYTES: u32 = 18u") != std::string::npos);
    ComputeDispatch q4_embedding_gles_dispatch;
    assert(build_q4_0_embedding_gles_candidate(q4_embedding_weights, 36, 2, 32,
        q4_embedding_tokens, q4_embedding_ids, q4_embedding_output_tensor,
        q4_embedding_gles_dispatch).ok());
    assert(q4_embedding_gles_dispatch.backend == BackendKind::OpenGles);

    BufferDesc q4_1_embedding_weights_desc{40, 4, MemoryDomain::Device, true, "q4-1-embedding-weights"};
    std::shared_ptr<Buffer> q4_1_embedding_weights;
    assert(backend.allocate(q4_1_embedding_weights_desc, q4_1_embedding_weights).ok());
    ComputeDispatch q4_1_embedding_dispatch;
    assert(build_q4_1_embedding_candidate(q4_1_embedding_weights, 40, 2, 32,
        q4_embedding_tokens, q4_embedding_ids, q4_embedding_output_tensor,
        q4_1_embedding_dispatch).ok());
    assert(q4_1_embedding_dispatch.kernel_id == "dawn-q4-1-embedding-f32-v1");
    assert(q4_1_embedding_dispatch.wgsl_source.find("WEIGHT_ROW_BYTES: u32 = 20u") != std::string::npos);
    ComputeDispatch q4_1_embedding_gles_dispatch;
    assert(build_q4_1_embedding_gles_candidate(q4_1_embedding_weights, 40, 2, 32,
        q4_embedding_tokens, q4_embedding_ids, q4_embedding_output_tensor,
        q4_1_embedding_gles_dispatch).ok());
    assert(q4_1_embedding_gles_dispatch.backend == BackendKind::OpenGles);
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
    ComputeDispatch q4_1_dequant_dispatch;
    assert(build_q4_1_dequantize_candidate(q4_1_encoded, 20, q4_output_tensor, q4_1_dequant_dispatch).ok());
    assert(q4_1_dequant_dispatch.bindings.size() == 2);
    assert(q4_1_dequant_dispatch.label.find("unverified") != std::string::npos);
    ComputeDispatch q4_gles_dequant, q4_1_gles_dequant;
    assert(build_q4_0_dequantize_gles_candidate(q4_encoded, 18, q4_output_tensor, q4_gles_dequant).ok());
    assert(build_q4_1_dequantize_gles_candidate(q4_1_encoded, 20, q4_output_tensor, q4_1_gles_dequant).ok());
    assert(q4_gles_dequant.backend == BackendKind::OpenGles && q4_1_gles_dequant.backend == BackendKind::OpenGles);
    assert(q4_gles_dequant.glsl_source.find("#version 310 es") == 0);
    assert(q4_1_gles_dequant.glsl_source.find("#version 310 es") == 0);


    BufferDesc q5_encoded_desc{24, 4, MemoryDomain::Device, false, "q5-encoded"};
    std::shared_ptr<Buffer> q5_encoded;
    assert(backend.allocate(q5_encoded_desc, q5_encoded).ok());
    ComputeDispatch q5_0_dispatch, q5_1_dispatch;
    assert(build_q5_0_dequantize_candidate(q5_encoded, 22, q4_output_tensor, q5_0_dispatch).ok());
    assert(build_q5_1_dequantize_candidate(q5_encoded, 24, q4_output_tensor, q5_1_dispatch).ok());
    assert(q5_0_dispatch.bindings.size() == 2 && q5_1_dispatch.bindings.size() == 2);
    ComputeDispatch q5_0_gles_dispatch, q5_1_gles_dispatch;
    assert(build_q5_0_dequantize_gles_candidate(q5_encoded, 22, q4_output_tensor, q5_0_gles_dispatch).ok());
    assert(build_q5_1_dequantize_gles_candidate(q5_encoded, 24, q4_output_tensor, q5_1_gles_dispatch).ok());
    assert(q5_0_gles_dispatch.backend == BackendKind::OpenGles);
    assert(q5_1_gles_dispatch.backend == BackendKind::OpenGles);
    assert(q5_0_gles_dispatch.wgsl_source.empty() && q5_1_gles_dispatch.wgsl_source.empty());
    assert(q5_0_gles_dispatch.glsl_source.find("#version 310 es") == 0);
    assert(q5_1_gles_dispatch.glsl_source.find("#version 310 es") == 0);
    assert(q5_0_gles_dispatch.glsl_source.find("ELEMENT_COUNT = 32u") != std::string::npos);
    assert(q5_1_gles_dispatch.glsl_source.find("ELEMENT_COUNT = 32u") != std::string::npos);


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
    metadata.model = "qwen;profile";
    metadata.pre_tokenizer = "qwen=3";
    metadata.merges = {"a b"};
    const std::string diagnostic = describe_tokenizer_metadata(metadata, registry);
    assert(diagnostic.find("tokenizer_model=qwen%3Bprofile") != std::string::npos);
    assert(diagnostic.find("tokenizer_pre_tokenizer=qwen%3D3") != std::string::npos);
    assert(diagnostic.find("tokenizer_vocab_count=2") != std::string::npos);
    assert(diagnostic.find("tokenizer_merge_count=1") != std::string::npos);
    metadata.model = "test-bpe";
    metadata.pre_tokenizer = "test-pretokenizer";

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
    GgufTokenizerInfo unigram_choice = sentencepiece;
    unigram_choice.tokens = {"<unk>", "▁", "a", "b", "▁a", "ab", "▁ab"};
    unigram_choice.scores = {-100.0f, -20.0f, -20.0f, 4.0f, 4.0f, 6.0f, 7.0f};
    unigram_choice.token_types = {2, 1, 1, 1, 1, 1, 1};
    unigram_choice.bos_token_id = -1;
    unigram_choice.eos_token_id = -1;
    unigram_choice.add_bos_token = false;
    unigram_choice.add_eos_token = false;
    auto unigram_adapter = builtins.find(unigram_choice);
    assert(unigram_adapter && unigram_adapter->validate_metadata(unigram_choice).ok());
    std::vector<uint32_t> unigram_ids;
    assert(unigram_adapter->encode(unigram_choice, "ab", unigram_ids).ok());
    assert((unigram_ids == std::vector<uint32_t>{4, 3}));
    GgufTokenizerInfo unicode_spaces;
    unicode_spaces.model="llama";
    unicode_spaces.pre_tokenizer="default";
    unicode_spaces.tokens={"<unk>","▁","a","b","▁a","▁b"};
    unicode_spaces.scores={-100.0f,-20.0f,-20.0f,-20.0f,5.0f,5.0f};
    unicode_spaces.token_types={2,1,1,1,1,1};
    unicode_spaces.unknown_token_id=0;
    unicode_spaces.add_space_prefix=true;
    unicode_spaces.escape_whitespaces=true;
    unicode_spaces.remove_extra_whitespaces=true;
    auto unicode_space_adapter=builtins.find(unicode_spaces);
    assert(unicode_space_adapter&&unicode_space_adapter->validate_metadata(unicode_spaces).ok());
    std::vector<uint32_t> unicode_space_ids;
    assert(unicode_space_adapter->encode(unicode_spaces,"a\xe2\x80\x83" "b",unicode_space_ids).ok());
    assert((unicode_space_ids==std::vector<uint32_t>{4,5}));
    GgufTokenizerInfo byte_fallback;
    byte_fallback.model = "llama";
    byte_fallback.pre_tokenizer = "default";
    byte_fallback.tokens = {"<0xC3>", "<0xA9>"};
    byte_fallback.scores = {-1.0f, -1.0f};
    byte_fallback.token_types = {6, 6};
    auto byte_adapter = builtins.find(byte_fallback);
    assert(byte_adapter && byte_adapter->validate_metadata(byte_fallback).ok());
    std::vector<uint32_t> byte_ids;
    assert(byte_adapter->encode(byte_fallback, "\xc3\xa9", byte_ids).ok());
    assert((byte_ids == std::vector<uint32_t>{0, 1}));

    GgufTokenizerInfo invalid_scores = sentencepiece;
    invalid_scores.scores[2] = std::numeric_limits<float>::quiet_NaN();
    assert(spm->validate_metadata(invalid_scores).code == StatusCode::InvalidArgument);


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
    GgufTokenizerInfo malformed_bpe = bpe;
    malformed_bpe.merges = {"h x"};
    assert(bpe_adapter->validate_metadata(malformed_bpe).code == StatusCode::InvalidArgument);
    malformed_bpe = bpe;
    malformed_bpe.merges = {"h i", "h i"};
    assert(bpe_adapter->validate_metadata(malformed_bpe).code == StatusCode::InvalidArgument);
    malformed_bpe = bpe;
    malformed_bpe.tokens[2] = "h";
    assert(bpe_adapter->validate_metadata(malformed_bpe).code == StatusCode::InvalidArgument);
    std::string bpe_text;
    assert(bpe_adapter->decode(bpe, bpe_ids, bpe_text).ok() && bpe_text == "hi");

    GgufTokenizerInfo bpe_contractions;
    bpe_contractions.model = "gpt2";
    bpe_contractions.pre_tokenizer = "gpt2";
    bpe_contractions.tokens = {"w", "e", "'", "r", "'r", "'re", "we", "we're", "<unk>"};
    bpe_contractions.merges = {"w e", "' r", "'r e", "we 're"};
    bpe_contractions.token_types = {1, 1, 1, 1, 1, 1, 1, 1, 2};
    bpe_contractions.unknown_token_id = 8;
    auto contraction_adapter = builtins.find(bpe_contractions);
    assert(contraction_adapter && contraction_adapter->validate_metadata(bpe_contractions).ok());
    std::vector<uint32_t> contraction_ids;
    assert(contraction_adapter->encode(bpe_contractions, "we're", contraction_ids).ok());
    assert((contraction_ids == std::vector<uint32_t>{6, 5}));

    GgufTokenizerInfo gpt2_digits;
    gpt2_digits.model = "gpt2";
    gpt2_digits.pre_tokenizer = "gpt2";
    gpt2_digits.tokens = {"1", "2", "3", "4", "12", "123", "1234", "<unk>"};
    gpt2_digits.merges = {"1 2", "12 3", "123 4"};
    gpt2_digits.token_types = {1, 1, 1, 1, 1, 1, 1, 2};
    gpt2_digits.unknown_token_id = 7;
    auto gpt2_digit_adapter = builtins.find(gpt2_digits);
    assert(gpt2_digit_adapter && gpt2_digit_adapter->validate_metadata(gpt2_digits).ok());
    std::vector<uint32_t> digit_ids;
    assert(gpt2_digit_adapter->encode(gpt2_digits, "1234", digit_ids).ok());
    assert((digit_ids == std::vector<uint32_t>{5, 3}));

    GgufTokenizerInfo qwen2_digits = gpt2_digits;
    qwen2_digits.pre_tokenizer = "qwen2";
    auto qwen2_digit_adapter = builtins.find(qwen2_digits);
    assert(qwen2_digit_adapter && qwen2_digit_adapter->validate_metadata(qwen2_digits).ok());
    assert(qwen2_digit_adapter->encode(qwen2_digits, "1234", digit_ids).ok());
    assert((digit_ids == std::vector<uint32_t>{0, 1, 2, 3}));

    // Build a compact, complete byte-level vocabulary fixture for the Qwen3.5
    // adapter. The first 256 entries are GPT-2 byte symbols in byte order.
    GgufTokenizerInfo qwen35;
    qwen35.model = "gpt2";
    qwen35.pre_tokenizer = "qwen35";
    std::vector<std::string> byte_symbols(256);
    std::vector<uint32_t> printable;
    for (uint32_t b = 33; b <= 126; ++b) printable.push_back(b);
    for (uint32_t b = 161; b <= 172; ++b) printable.push_back(b);
    for (uint32_t b = 174; b <= 255; ++b) printable.push_back(b);
    uint32_t extra_symbol = 0;
    for (uint32_t b = 0; b < 256; ++b) {
        uint32_t cp = b;
        if (std::find(printable.begin(), printable.end(), b) == printable.end())
            cp = 256 + extra_symbol++;
        std::string encoded;
        if (cp < 0x80) encoded.push_back(static_cast<char>(cp));
        else if (cp < 0x800) {
            encoded.push_back(static_cast<char>(0xc0 | (cp >> 6)));
            encoded.push_back(static_cast<char>(0x80 | (cp & 0x3f)));
        } else {
            encoded.push_back(static_cast<char>(0xe0 | (cp >> 12)));
            encoded.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3f)));
            encoded.push_back(static_cast<char>(0x80 | (cp & 0x3f)));
        }
        byte_symbols[b] = encoded;
    }
    qwen35.tokens = byte_symbols;
    const auto append_qwen_token = [&](const std::string& token) {
        qwen35.tokens.push_back(token);
        return static_cast<uint32_t>(qwen35.tokens.size() - 1);
    };
    const uint32_t qwen_space = static_cast<uint32_t>(
        std::find(qwen35.tokens.begin(), qwen35.tokens.end(), byte_symbols[32]) - qwen35.tokens.begin());
    const uint32_t qwen_space_space = append_qwen_token(byte_symbols[32] + byte_symbols[32]);
    const uint32_t qwen_space_h = append_qwen_token(byte_symbols[32] + byte_symbols['h']);
    const uint32_t qwen_space_hi = append_qwen_token(byte_symbols[32] + byte_symbols['h'] + byte_symbols['i']);
    append_qwen_token(byte_symbols['1'] + byte_symbols['2']);
    append_qwen_token(byte_symbols['1'] + byte_symbols['2'] + byte_symbols['3']);
    append_qwen_token(byte_symbols['1'] + byte_symbols['2'] + byte_symbols['3'] + byte_symbols['4']);
    qwen35.merges = {
        byte_symbols[32] + " " + byte_symbols[32],
        byte_symbols[32] + " " + byte_symbols['h'],
        byte_symbols[32] + byte_symbols['h'] + " " + byte_symbols['i'],
        byte_symbols['1'] + " " + byte_symbols['2'],
        byte_symbols['1'] + byte_symbols['2'] + " " + byte_symbols['3'],
        byte_symbols['1'] + byte_symbols['2'] + byte_symbols['3'] + " " + byte_symbols['4']
    };
    qwen35.token_types.assign(qwen35.tokens.size(), 1);
    auto qwen35_adapter = builtins.find(qwen35);
    assert(qwen35_adapter && qwen35_adapter->descriptor().validation == SupportState::Validated);
    assert(qwen35_adapter->validate_metadata(qwen35).ok());
    auto qwen35_preflight = evaluate_tokenizer_preflight(qwen35, builtins);
    assert(qwen35_preflight.ready && qwen35_preflight.adapter_validated);
    std::vector<uint32_t> qwen35_ids;
    assert(qwen35_adapter->encode(qwen35, "  hi", qwen35_ids).ok());
    assert((qwen35_ids == std::vector<uint32_t>{qwen_space, qwen_space_hi}));
    assert(qwen35_adapter->encode(qwen35, "1234", qwen35_ids).ok());
    assert((qwen35_ids == std::vector<uint32_t>{49, 50, 51, 52}));
    std::string qwen35_text;
    assert(qwen35_adapter->decode(qwen35, qwen35_ids, qwen35_text).ok());
    assert(qwen35_text == "1234");
    GgufTokenizerInfo invalid_qwen35 = qwen35;
    invalid_qwen35.token_types.clear();
    assert(qwen35_adapter->validate_metadata(invalid_qwen35).code == StatusCode::Unsupported);
    invalid_qwen35 = qwen35;
    invalid_qwen35.add_space_prefix = true;
    assert(qwen35_adapter->validate_metadata(invalid_qwen35).code == StatusCode::Unsupported);
    (void)qwen_space_space;
    (void)qwen_space_h;

    GgufTokenizerInfo bpe_prefix;
    bpe_prefix.model = "gpt2";
    bpe_prefix.pre_tokenizer = "gpt2";
    bpe_prefix.tokens = {"(", "h", "i", "(h", "(hi", "hi", "<unk>"};
    bpe_prefix.merges = {"( h", "(h i", "h i"};
    bpe_prefix.token_types = {1, 1, 1, 1, 1, 1, 2};
    bpe_prefix.unknown_token_id = 6;
    auto prefix_adapter = builtins.find(bpe_prefix);
    assert(prefix_adapter && prefix_adapter->validate_metadata(bpe_prefix).ok());
    std::vector<uint32_t> prefix_ids;
    assert(prefix_adapter->encode(bpe_prefix, "(hi", prefix_ids).ok());
    assert((prefix_ids == std::vector<uint32_t>{4}));

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
    DeviceInfo model_device = device;
    model_device.limits.max_buffer_size = 64;
    model_device.limits.max_storage_buffer_binding_size = 64;
    MemoryManager model_memory(backend, 64);
    ModelExecutionPlan plan;
    ModelPlanOperation operation;
    operation.id = "weight-consumer";
    operation.operator_name = "MatMul";
    operation.weight_tensors = {"weight"};
    plan.operations.push_back(operation);
    ModelExecutionContext context;
    assert(load_model_execution_weights(reader, plan, backend, model_memory,
                                         model_device, context).ok());
    assert(context.tensors.at("weight").descriptor().shape == std::vector<uint64_t>({8}));
    assert(context.weight_leases.size() == 1);
    assert(model_memory.allocated_bytes() == 32);
    context = ModelExecutionContext{};
    assert(model_memory.allocated_bytes() == 0);

    const std::string q4_path = "/tmp/ai_chat_gpu_engine_q4_weight.gguf";
    {
        std::ofstream out(q4_path, std::ios::binary | std::ios::trunc);
        out.write("GGUF", 4);
        write_u32(out, 3);
        write_u64(out, 1);
        write_u64(out, 2);
        write_string(out, "general.architecture");
        write_u32(out, 8); write_string(out, "test-architecture");
        write_string(out, "general.alignment");
        write_u32(out, 4); write_u32(out, 32);
        write_string(out, "q4_weight");
        write_u32(out, 1);
        write_u64(out, 32);
        write_u32(out, 2);
        write_u64(out, 0);
        const auto pos = static_cast<uint64_t>(out.tellp());
        const auto data_start = (pos + 31u) & ~31ULL;
        for (uint64_t i = pos; i < data_start; ++i) out.put('\0');
        uint8_t q4_block[18] = {0x00, 0x38};
        for (uint32_t i = 0; i < 16; ++i)
            q4_block[2 + i] = static_cast<uint8_t>((8u + (i % 8u)) | ((8u + ((i + 2u) % 8u)) << 4u));
        out.write(reinterpret_cast<const char*>(q4_block), sizeof(q4_block));
    }
    GgufReader q4_reader;
    assert(GgufReader::open(q4_path, q4_reader).ok());
    assert(q4_reader.info().tensors.size() == 1);
    assert(q4_reader.info().tensors[0].type.name == "Q4_0");
    MemoryManager q4_memory(backend, 64);
    ModelExecutionPlan q4_plan;
    ModelPlanOperation q4_operation;
    q4_operation.id = "q4-weight-consumer";
    q4_operation.operator_name = "MatMul_Q4_0";
    q4_operation.weight_tensors = {"q4_weight"};
    q4_plan.operations.push_back(q4_operation);
    ModelExecutionContext q4_context;
    assert(load_model_execution_weights(q4_reader, q4_plan, backend, q4_memory, model_device,
                                         q4_context).ok());
    const auto q4_binding = q4_context.quantized_weights.find("q4_weight");
    assert(q4_binding != q4_context.quantized_weights.end());
    assert(q4_binding->second.file_type_id == 2);
    assert(q4_binding->second.byte_size == 18);
    assert(q4_binding->second.buffer->size_bytes() >= 20);
    assert(q4_context.weight_leases.size() == 1);
    q4_context = ModelExecutionContext{};
    assert(q4_memory.allocated_bytes() == 0);
    std::remove(q4_path.c_str());

    const std::string q4_1_path = "/tmp/ai_chat_gpu_engine_q4_1_weight.gguf";
    {
        std::ofstream out(q4_1_path, std::ios::binary | std::ios::trunc);
        out.write("GGUF", 4);
        write_u32(out, 3);
        write_u64(out, 1);
        write_u64(out, 2);
        write_string(out, "general.architecture");
        write_u32(out, 8); write_string(out, "test-architecture");
        write_string(out, "general.alignment");
        write_u32(out, 4); write_u32(out, 32);
        write_string(out, "q4_1_weight");
        write_u32(out, 1);
        write_u64(out, 32);
        write_u32(out, 3);
        write_u64(out, 0);
        const auto pos = static_cast<uint64_t>(out.tellp());
        const auto data_start = (pos + 31u) & ~31ULL;
        for (uint64_t i = pos; i < data_start; ++i) out.put('\0');
        uint8_t q4_1_block[20] = {0x00, 0x38, 0x00, 0xbc};
        for (uint32_t i = 0; i < 16; ++i)
            q4_1_block[4 + i] = static_cast<uint8_t>((i % 16u) | (((i + 2u) % 16u) << 4u));
        out.write(reinterpret_cast<const char*>(q4_1_block), sizeof(q4_1_block));
    }
    GgufReader q4_1_reader;
    assert(GgufReader::open(q4_1_path, q4_1_reader).ok());
    assert(q4_1_reader.info().tensors[0].type.name == "Q4_1");
    MemoryManager q4_1_memory(backend, 64);
    ModelExecutionPlan q4_1_plan;
    ModelPlanOperation q4_1_operation;
    q4_1_operation.id = "q4-1-weight-consumer";
    q4_1_operation.operator_name = "MatMul_Q4_1";
    q4_1_operation.weight_tensors = {"q4_1_weight"};
    q4_1_plan.operations.push_back(q4_1_operation);
    ModelExecutionContext q4_1_context;
    assert(load_model_execution_weights(q4_1_reader, q4_1_plan, backend, q4_1_memory, model_device,
                                         q4_1_context).ok());
    const auto q4_1_binding = q4_1_context.quantized_weights.find("q4_1_weight");
    assert(q4_1_binding != q4_1_context.quantized_weights.end());
    assert(q4_1_binding->second.file_type_id == 3);
    assert(q4_1_binding->second.byte_size == 20);
    assert(q4_1_binding->second.buffer->size_bytes() == 20);
    q4_1_context = ModelExecutionContext{};
    assert(q4_1_memory.allocated_bytes() == 0);
    std::remove(q4_1_path.c_str());

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
    BufferDesc bias_desc{12, 4, MemoryDomain::Device, false, "add-bias"};
    std::shared_ptr<Buffer> bias_buffer;
    assert(backend.allocate(bias_desc, bias_buffer).ok());
    TensorDesc bias_tensor_desc; bias_tensor_desc.shape = {3}; bias_tensor_desc.data_type = DataType::F32;
    Tensor bias_tensor;
    assert(Tensor::create(bias_tensor_desc, bias_buffer, 0, bias_tensor).ok());
    ComputeDispatch broadcast_dispatch;
    assert(build_add_f32_candidate(lhs, bias_tensor, output, broadcast_dispatch).ok());
    assert(broadcast_dispatch.bindings[1].size_bytes == 12);
    assert(broadcast_dispatch.wgsl_source.find("gid.x % arrayLength(&rhs)") != std::string::npos);
    float broadcast_result[6] = {};
    const float broadcast_lhs[] = {1, 2, 3, 4, 5, 6};
    const float broadcast_rhs[] = {10, 20, 30};
    assert(reference_add_f32_trailing_broadcast(broadcast_lhs, broadcast_rhs,
                                                broadcast_result, 6, 3).ok());
    assert(broadcast_result[0] == 11 && broadcast_result[1] == 22 &&
           broadcast_result[2] == 33 && broadcast_result[3] == 14 &&
           broadcast_result[4] == 25 && broadcast_result[5] == 36);

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
    BufferDesc mat3_out_desc{16, 4, MemoryDomain::Device, false, "mat-out-3d"};
    std::shared_ptr<Buffer> mat3_out;
    assert(backend.allocate(mat3_out_desc, mat3_out).ok());
    TensorDesc c3_desc; c3_desc.shape = {2, 1, 2}; c3_desc.data_type = DataType::F32;
    Tensor c3_tensor;
    assert(Tensor::create(c3_desc, mat3_out, 0, c3_tensor).ok());
    ComputeDispatch mat3_dispatch;
    assert(build_matmul_f32_candidate(a_tensor, b_tensor, c3_tensor, mat3_dispatch).ok());
    assert(mat3_dispatch.wgsl_source.find("const COLUMNS: u32 = 2u;") != std::string::npos);
    ComputeDispatch mat3_gles_dispatch;
    assert(build_matmul_f32_gles_candidate(a_tensor, b_tensor, c3_tensor, mat3_gles_dispatch).ok());
    assert(mat3_gles_dispatch.backend == BackendKind::OpenGles);
    assert(mat3_gles_dispatch.glsl_source.find("const uint COLUMNS = 2u;") != std::string::npos);


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







void test_dense_model_plan_selects_q8_0_matmul_adapter() {
    ArchitectureRegistry registry;
    assert(register_builtin_architecture_adapters(registry).ok());
    const auto adapter = registry.find("llama");
    assert(adapter);
    GgufFileInfo model;
    model.architecture = "llama";
    model.metadata["llama.block_count"] = uint64_t{1};
    model.metadata["llama.embedding_length"] = uint64_t{32};
    model.metadata["llama.attention.head_count"] = uint64_t{1};
    model.metadata["llama.feed_forward_length"] = uint64_t{32};
    model.tokenizer.tokens = {"a", "b"};
    const std::vector<std::pair<std::string, std::vector<uint64_t>>> tensors = {
        {"token_embd.weight", {32, 2}},
        {"output.weight", {32, 2}},
        {"output_norm.weight", {32}},
        {"blk.0.attn_norm.weight", {32}},
        {"blk.0.attn_q.weight", {32, 32}},
        {"blk.0.attn_k.weight", {32, 32}},
        {"blk.0.attn_v.weight", {32, 32}},
        {"blk.0.attn_output.weight", {32, 32}},
        {"blk.0.ffn_norm.weight", {32}},
        {"blk.0.ffn_gate.weight", {32, 32}},
        {"blk.0.ffn_up.weight", {32, 32}},
        {"blk.0.ffn_down.weight", {32, 32}},
    };
    for (const auto& item : tensors) {
        GgufTensorInfo tensor;
        tensor.name = item.first;
        tensor.dimensions = item.second;
        tensor.type_id = tensor.name == "blk.0.attn_q.weight" ? 8u : 0u;
        tensor.type = GgufReader::tensor_type_info(tensor.type_id);
        uint64_t elements = 1;
        for (uint64_t dimension : tensor.dimensions) elements *= dimension;
        tensor.byte_size_known = true;
        tensor.byte_size = tensor.type_id == 8 ? 32 * 34 : elements * sizeof(float);
        model.tensors.push_back(std::move(tensor));
    }
    ModelExecutionPlan plan;
    assert(adapter->build_execution_plan(model, plan).ok());
    const auto projection = std::find_if(plan.operations.begin(), plan.operations.end(),
        [](const ModelPlanOperation& operation) { return operation.id == "layer.0.q_proj"; });
    assert(projection != plan.operations.end());
    assert(projection->operator_name == "MatMul_Q8_0");
}


void test_dense_model_plan_selects_q4_0_matmul_adapter() {
    ArchitectureRegistry registry;
    assert(register_builtin_architecture_adapters(registry).ok());
    const auto adapter = registry.find("llama");
    assert(adapter);
    GgufFileInfo model;
    model.architecture = "llama";
    model.metadata["llama.block_count"] = uint64_t{1};
    model.metadata["llama.embedding_length"] = uint64_t{32};
    model.metadata["llama.attention.head_count"] = uint64_t{1};
    model.metadata["llama.feed_forward_length"] = uint64_t{32};
    model.tokenizer.tokens = {"a", "b"};
    const std::vector<std::pair<std::string, std::vector<uint64_t>>> tensors = {
        {"token_embd.weight", {32, 2}},
        {"output.weight", {32, 2}},
        {"output_norm.weight", {32}},
        {"blk.0.attn_norm.weight", {32}},
        {"blk.0.attn_q.weight", {32, 32}},
        {"blk.0.attn_k.weight", {32, 32}},
        {"blk.0.attn_v.weight", {32, 32}},
        {"blk.0.attn_output.weight", {32, 32}},
        {"blk.0.ffn_norm.weight", {32}},
        {"blk.0.ffn_gate.weight", {32, 32}},
        {"blk.0.ffn_up.weight", {32, 32}},
        {"blk.0.ffn_down.weight", {32, 32}},
    };
    for (const auto& item : tensors) {
        GgufTensorInfo tensor;
        tensor.name = item.first;
        tensor.dimensions = item.second;
        tensor.type_id = (tensor.name == "blk.0.attn_q.weight" || tensor.name == "token_embd.weight") ? 2u : 0u;
        tensor.type = GgufReader::tensor_type_info(tensor.type_id);
        uint64_t elements = 1;
        for (uint64_t dimension : tensor.dimensions) elements *= dimension;
        tensor.byte_size_known = true;
        tensor.byte_size = tensor.type_id == 2 ? (elements / 32) * 18 : elements * sizeof(float);
        model.tensors.push_back(std::move(tensor));
    }
    ModelExecutionPlan plan;
    assert(adapter->build_execution_plan(model, plan).ok());
    const auto projection = std::find_if(plan.operations.begin(), plan.operations.end(),
        [](const ModelPlanOperation& operation) { return operation.id == "layer.0.q_proj"; });
    assert(projection != plan.operations.end());
    assert(projection->operator_name == "MatMul_Q4_0");
    const auto embedding = std::find_if(plan.operations.begin(), plan.operations.end(),
        [](const ModelPlanOperation& operation) { return operation.id == "embedding"; });
    assert(embedding != plan.operations.end());
    assert(embedding->operator_name == "Embedding_Q4_0");
}







void test_dense_model_plan_selects_k23_matmul_adapter(uint32_t type_id,const std::string&expected){
    ArchitectureRegistry registry;assert(register_builtin_architecture_adapters(registry).ok());
    const auto adapter=registry.find("llama");assert(adapter);
    GgufFileInfo model;model.architecture="llama";
    model.metadata["llama.block_count"]=uint64_t{1};model.metadata["llama.embedding_length"]=uint64_t{256};
    model.metadata["llama.attention.head_count"]=uint64_t{1};model.metadata["llama.feed_forward_length"]=uint64_t{256};
    model.tokenizer.tokens={"a","b"};
    const std::vector<std::pair<std::string,std::vector<uint64_t>>> tensors={
        {"token_embd.weight",{256,2}},{"output.weight",{256,2}},{"output_norm.weight",{256}},
        {"blk.0.attn_norm.weight",{256}},{"blk.0.attn_q.weight",{256,256}},
        {"blk.0.attn_k.weight",{256,256}},{"blk.0.attn_v.weight",{256,256}},
        {"blk.0.attn_output.weight",{256,256}},{"blk.0.ffn_norm.weight",{256}},
        {"blk.0.ffn_gate.weight",{256,256}},{"blk.0.ffn_up.weight",{256,256}},
        {"blk.0.ffn_down.weight",{256,256}}
    };
    const uint64_t block_bytes=type_id==10?84:110;
    for(const auto&item:tensors){
        GgufTensorInfo tensor;tensor.name=item.first;tensor.dimensions=item.second;
        tensor.type_id=tensor.name=="blk.0.attn_q.weight"?type_id:0u;
        tensor.type=GgufReader::tensor_type_info(tensor.type_id);
        uint64_t elements=1;for(uint64_t d:tensor.dimensions)elements*=d;
        tensor.byte_size_known=true;tensor.byte_size=tensor.type_id==type_id?(elements/256)*block_bytes:elements*sizeof(float);
        model.tensors.push_back(std::move(tensor));
    }
    ModelExecutionPlan plan;assert(adapter->build_execution_plan(model,plan).ok());
    const auto projection=std::find_if(plan.operations.begin(),plan.operations.end(),
        [](const ModelPlanOperation&op){return op.id=="layer.0.q_proj";});
    assert(projection!=plan.operations.end()&&projection->operator_name==expected);
}

void test_dense_model_plan_selects_k_quant_matmul_adapter(uint32_t type_id,const std::string&expected){
    ArchitectureRegistry registry;assert(register_builtin_architecture_adapters(registry).ok());
    const auto adapter=registry.find("llama");assert(adapter);
    GgufFileInfo model;model.architecture="llama";
    model.metadata["llama.block_count"]=uint64_t{1};model.metadata["llama.embedding_length"]=uint64_t{256};
    model.metadata["llama.attention.head_count"]=uint64_t{1};model.metadata["llama.feed_forward_length"]=uint64_t{256};
    model.tokenizer.tokens={"a","b"};
    const std::vector<std::pair<std::string,std::vector<uint64_t>>> tensors={
        {"token_embd.weight",{256,2}},{"output.weight",{256,2}},{"output_norm.weight",{256}},
        {"blk.0.attn_norm.weight",{256}},{"blk.0.attn_q.weight",{256,256}},
        {"blk.0.attn_k.weight",{256,256}},{"blk.0.attn_v.weight",{256,256}},
        {"blk.0.attn_output.weight",{256,256}},{"blk.0.ffn_norm.weight",{256}},
        {"blk.0.ffn_gate.weight",{256,256}},{"blk.0.ffn_up.weight",{256,256}},
        {"blk.0.ffn_down.weight",{256,256}}
    };
    const uint64_t block_bytes=type_id==12?144:176;
    for(const auto&item:tensors){
        GgufTensorInfo tensor;tensor.name=item.first;tensor.dimensions=item.second;
        tensor.type_id=tensor.name=="blk.0.attn_q.weight"?type_id:0u;
        tensor.type=GgufReader::tensor_type_info(tensor.type_id);
        uint64_t elements=1;for(uint64_t d:tensor.dimensions)elements*=d;
        tensor.byte_size_known=true;tensor.byte_size=tensor.type_id==type_id?(elements/256)*block_bytes:elements*sizeof(float);
        model.tensors.push_back(std::move(tensor));
    }
    ModelExecutionPlan plan;assert(adapter->build_execution_plan(model,plan).ok());
    const auto projection=std::find_if(plan.operations.begin(),plan.operations.end(),
        [](const ModelPlanOperation&op){return op.id=="layer.0.q_proj";});
    assert(projection!=plan.operations.end()&&projection->operator_name==expected);
}

void test_dense_model_plan_selects_q6_k_matmul_adapter(){
    ArchitectureRegistry registry;assert(register_builtin_architecture_adapters(registry).ok());
    const auto adapter=registry.find("llama");assert(adapter);
    GgufFileInfo model;model.architecture="llama";
    model.metadata["llama.block_count"]=uint64_t{1};model.metadata["llama.embedding_length"]=uint64_t{256};
    model.metadata["llama.attention.head_count"]=uint64_t{1};model.metadata["llama.feed_forward_length"]=uint64_t{256};
    model.tokenizer.tokens={"a","b"};
    const std::vector<std::pair<std::string,std::vector<uint64_t>>> tensors={
        {"token_embd.weight",{256,2}},{"output.weight",{256,2}},{"output_norm.weight",{256}},
        {"blk.0.attn_norm.weight",{256}},{"blk.0.attn_q.weight",{256,256}},
        {"blk.0.attn_k.weight",{256,256}},{"blk.0.attn_v.weight",{256,256}},
        {"blk.0.attn_output.weight",{256,256}},{"blk.0.ffn_norm.weight",{256}},
        {"blk.0.ffn_gate.weight",{256,256}},{"blk.0.ffn_up.weight",{256,256}},
        {"blk.0.ffn_down.weight",{256,256}}
    };
    for(const auto&item:tensors){
        GgufTensorInfo tensor;tensor.name=item.first;tensor.dimensions=item.second;
        tensor.type_id=tensor.name=="blk.0.attn_q.weight"?14u:0u;
        tensor.type=GgufReader::tensor_type_info(tensor.type_id);
        uint64_t elements=1;for(uint64_t d:tensor.dimensions)elements*=d;
        tensor.byte_size_known=true;tensor.byte_size=tensor.type_id==14?(elements/256)*210:elements*sizeof(float);
        model.tensors.push_back(std::move(tensor));
    }
    ModelExecutionPlan plan;assert(adapter->build_execution_plan(model,plan).ok());
    const auto projection=std::find_if(plan.operations.begin(),plan.operations.end(),
        [](const ModelPlanOperation&op){return op.id=="layer.0.q_proj";});
    assert(projection!=plan.operations.end()&&projection->operator_name=="MatMul_Q6_K");
}

void test_dense_model_plan_selects_fp16_matmul_adapter(uint32_t file_type_id,const std::string&expected_operator){
    ArchitectureRegistry registry;assert(register_builtin_architecture_adapters(registry).ok());
    const auto adapter=registry.find("llama");assert(adapter);
    GgufFileInfo model;model.architecture="llama";
    model.metadata["llama.block_count"]=uint64_t{1};
    model.metadata["llama.embedding_length"]=uint64_t{32};
    model.metadata["llama.attention.head_count"]=uint64_t{1};
    model.metadata["llama.feed_forward_length"]=uint64_t{32};
    model.tokenizer.tokens={"a","b"};
    const std::vector<std::pair<std::string,std::vector<uint64_t>>> tensors={
        {"token_embd.weight",{32,2}},{"output.weight",{32,2}},{"output_norm.weight",{32}},
        {"blk.0.attn_norm.weight",{32}},{"blk.0.attn_q.weight",{32,32}},
        {"blk.0.attn_k.weight",{32,32}},{"blk.0.attn_v.weight",{32,32}},
        {"blk.0.attn_output.weight",{32,32}},{"blk.0.ffn_norm.weight",{32}},
        {"blk.0.ffn_gate.weight",{32,32}},{"blk.0.ffn_up.weight",{32,32}},
        {"blk.0.ffn_down.weight",{32,32}}
    };
    for(const auto&item:tensors){
        GgufTensorInfo tensor;tensor.name=item.first;tensor.dimensions=item.second;
        tensor.type_id=(tensor.name=="blk.0.attn_q.weight"||tensor.name=="token_embd.weight")?file_type_id:0u;
        tensor.type=GgufReader::tensor_type_info(tensor.type_id);
        uint64_t elements=1;for(uint64_t d:tensor.dimensions)elements*=d;
        tensor.byte_size_known=true;tensor.byte_size=tensor.type_id==file_type_id?elements*2:elements*sizeof(float);
        model.tensors.push_back(std::move(tensor));
    }
    ModelExecutionPlan plan;assert(adapter->build_execution_plan(model,plan).ok());
    const auto projection=std::find_if(plan.operations.begin(),plan.operations.end(),
        [](const ModelPlanOperation&op){return op.id=="layer.0.q_proj";});
    const auto embedding=std::find_if(plan.operations.begin(),plan.operations.end(),
        [](const ModelPlanOperation&op){return op.id=="embedding";});
    assert(projection!=plan.operations.end()&&projection->operator_name==expected_operator);
    assert(embedding!=plan.operations.end());
    assert(embedding->operator_name==(file_type_id==1?"Embedding_F16":"Embedding_BF16"));
}

void test_dense_model_plan_selects_q5_matmul_adapter(uint32_t q5_type_id, const std::string& expected_operator) {
    ArchitectureRegistry registry;
    assert(register_builtin_architecture_adapters(registry).ok());
    const auto adapter=registry.find("llama");
    assert(adapter);
    GgufFileInfo model;
    model.architecture="llama";
    model.metadata["llama.block_count"]=uint64_t{1};
    model.metadata["llama.embedding_length"]=uint64_t{32};
    model.metadata["llama.attention.head_count"]=uint64_t{1};
    model.metadata["llama.feed_forward_length"]=uint64_t{32};
    model.tokenizer.tokens={"a","b"};
    const std::vector<std::pair<std::string,std::vector<uint64_t>>> tensors={
        {"token_embd.weight",{32,2}},{"output.weight",{32,2}},{"output_norm.weight",{32}},
        {"blk.0.attn_norm.weight",{32}},{"blk.0.attn_q.weight",{32,32}},
        {"blk.0.attn_k.weight",{32,32}},{"blk.0.attn_v.weight",{32,32}},
        {"blk.0.attn_output.weight",{32,32}},{"blk.0.ffn_norm.weight",{32}},
        {"blk.0.ffn_gate.weight",{32,32}},{"blk.0.ffn_up.weight",{32,32}},
        {"blk.0.ffn_down.weight",{32,32}}
    };
    for(const auto& item:tensors){
        GgufTensorInfo tensor;tensor.name=item.first;tensor.dimensions=item.second;
        tensor.type_id=(tensor.name=="blk.0.attn_q.weight"||tensor.name=="token_embd.weight")?q5_type_id:0u;
        tensor.type=GgufReader::tensor_type_info(tensor.type_id);
        uint64_t elements=1;for(uint64_t dimension:tensor.dimensions)elements*=dimension;
        tensor.byte_size_known=true;
        const uint64_t block_bytes=q5_type_id==6?22u:24u;
        tensor.byte_size=tensor.type_id==q5_type_id?(elements/32)*block_bytes:elements*sizeof(float);
        model.tensors.push_back(std::move(tensor));
    }
    ModelExecutionPlan plan;
    assert(adapter->build_execution_plan(model,plan).ok());
    const auto projection=std::find_if(plan.operations.begin(),plan.operations.end(),
        [](const ModelPlanOperation& operation){return operation.id=="layer.0.q_proj";});
    assert(projection!=plan.operations.end());
    assert(projection->operator_name==expected_operator);
    const auto embedding=std::find_if(plan.operations.begin(),plan.operations.end(),
        [](const ModelPlanOperation& operation){return operation.id=="embedding";});
    assert(embedding!=plan.operations.end());
    assert(embedding->operator_name==(q5_type_id==6?"Embedding_Q5_0":"Embedding_Q5_1"));
}

void test_dense_model_plan_selects_q4_1_matmul_adapter() {
    ArchitectureRegistry registry;
    assert(register_builtin_architecture_adapters(registry).ok());
    const auto adapter = registry.find("llama");
    assert(adapter);
    GgufFileInfo model;
    model.architecture = "llama";
    model.metadata["llama.block_count"] = uint64_t{1};
    model.metadata["llama.embedding_length"] = uint64_t{32};
    model.metadata["llama.attention.head_count"] = uint64_t{1};
    model.metadata["llama.feed_forward_length"] = uint64_t{32};
    model.tokenizer.tokens = {"a", "b"};
    const std::vector<std::pair<std::string, std::vector<uint64_t>>> tensors = {
        {"token_embd.weight", {32, 2}},
        {"output.weight", {32, 2}},
        {"output_norm.weight", {32}},
        {"blk.0.attn_norm.weight", {32}},
        {"blk.0.attn_q.weight", {32, 32}},
        {"blk.0.attn_k.weight", {32, 32}},
        {"blk.0.attn_v.weight", {32, 32}},
        {"blk.0.attn_output.weight", {32, 32}},
        {"blk.0.ffn_norm.weight", {32}},
        {"blk.0.ffn_gate.weight", {32, 32}},
        {"blk.0.ffn_up.weight", {32, 32}},
        {"blk.0.ffn_down.weight", {32, 32}},
    };
    for (const auto& item : tensors) {
        GgufTensorInfo tensor;
        tensor.name = item.first;
        tensor.dimensions = item.second;
        tensor.type_id = (tensor.name == "blk.0.attn_q.weight" || tensor.name == "token_embd.weight") ? 3u : 0u;
        tensor.type = GgufReader::tensor_type_info(tensor.type_id);
        uint64_t elements = 1;
        for (uint64_t dimension : tensor.dimensions) elements *= dimension;
        tensor.byte_size_known = true;
        tensor.byte_size = tensor.type_id == 3 ? (elements / 32) * 20 : elements * sizeof(float);
        model.tensors.push_back(std::move(tensor));
    }
    ModelExecutionPlan plan;
    assert(adapter->build_execution_plan(model, plan).ok());
    const auto projection = std::find_if(plan.operations.begin(), plan.operations.end(),
        [](const ModelPlanOperation& operation) { return operation.id == "layer.0.q_proj"; });
    assert(projection != plan.operations.end());
    assert(projection->operator_name == "MatMul_Q4_1");
    const auto embedding = std::find_if(plan.operations.begin(), plan.operations.end(),
        [](const ModelPlanOperation& operation) { return operation.id == "embedding"; });
    assert(embedding != plan.operations.end());
    assert(embedding->operator_name == "Embedding_Q4_1");
}


void test_fake_backend_never_promotes_builtin_operator_kernels() {
    FakeBackend backend(BackendKind::OpenGles);
    OperatorRegistry registry;
    assert(register_builtin_gles_operator_candidates(registry).ok());
    KernelValidationReport report;
    assert(validate_builtin_operator_kernels(backend, registry, report).ok());
    assert(!report.complete);
    for (const char* kernel_id : {
        "gles-add-f32-generic-v1", "gles-multiply-f32-generic-v1",
        "gles-matmul-f32-generic-v1", "gles-rms-norm-f32-generic-v1",
        "gles-silu-f32-generic-v1", "gles-softmax-f32-generic-v1",
        "gles-rope-f32-generic-v1", "gles-attention-f32-causal-gqa-v1",
        "gles-embedding-f32-generic-v1"
    }) {
        assert(!registry.is_validated_kernel(kernel_id, BackendKind::OpenGles));
    }
}

void test_q8_0_model_lowering_supports_gles_3d_projection() {
    FakeBackend backend(BackendKind::OpenGles);
    ModelLoweringRegistry lowerings;
    assert(register_builtin_model_lowerings(BackendKind::OpenGles, lowerings).ok());
    BufferDesc activation_desc{2 * 32 * sizeof(float), 4, MemoryDomain::Device, false, "q8-activation"};
    BufferDesc output_desc{2 * 4 * sizeof(float), 4, MemoryDomain::Device, false, "q8-output"};
    BufferDesc weight_desc{4 * 34, 4, MemoryDomain::Device, true, "q8-weight"};
    std::shared_ptr<Buffer> activation_buffer, output_buffer, weight_buffer;
    assert(backend.allocate(activation_desc, activation_buffer).ok());
    assert(backend.allocate(output_desc, output_buffer).ok());
    assert(backend.allocate(weight_desc, weight_buffer).ok());
    TensorDesc activation_td; activation_td.shape = {2, 32}; activation_td.data_type = DataType::F32;
    TensorDesc output_td; output_td.shape = {2, 2, 2}; output_td.data_type = DataType::F32;
    Tensor activation, output;
    assert(Tensor::create(activation_td, activation_buffer, 0, activation).ok());
    assert(Tensor::create(output_td, output_buffer, 0, output).ok());
    ModelExecutionContext context;
    context.tensors.emplace("activation", activation);
    context.tensors.emplace("output", output);
    QuantizedWeightBinding binding;
    binding.buffer = weight_buffer;
    binding.byte_size = 4 * 34;
    binding.file_type_id = 8;
    binding.dimensions = {32, 4};
    context.quantized_weights.emplace("q.weight", binding);
    ModelPlanOperation operation;
    operation.id = "q8-matmul";
    operation.operator_name = "MatMul_Q8_0";
    operation.inputs = {"activation"};
    operation.weight_tensors = {"q.weight"};
    operation.output = "output";
    ComputeDispatch dispatch;
    assert((*lowerings.find(BackendKind::OpenGles, "MatMul_Q8_0"))(operation, context, dispatch).ok());
    assert(dispatch.backend == BackendKind::OpenGles);
    assert(dispatch.glsl_source.find("const uint OUTPUT_CHANNELS = 4u;") != std::string::npos);
    assert(dispatch.validation == KernelValidation::Unverified);
}

void test_q4_0_model_lowering_supports_gles_3d_projection() {
    FakeBackend backend(BackendKind::OpenGles);
    ModelLoweringRegistry lowerings;
    assert(register_builtin_model_lowerings(BackendKind::OpenGles, lowerings).ok());
    BufferDesc activation_desc{2 * 32 * sizeof(float), 4, MemoryDomain::Device, false, "q8-activation"};
    BufferDesc output_desc{2 * 4 * sizeof(float), 4, MemoryDomain::Device, false, "q8-output"};
    BufferDesc weight_desc{4 * 18, 4, MemoryDomain::Device, true, "q8-weight"};
    std::shared_ptr<Buffer> activation_buffer, output_buffer, weight_buffer;
    assert(backend.allocate(activation_desc, activation_buffer).ok());
    assert(backend.allocate(output_desc, output_buffer).ok());
    assert(backend.allocate(weight_desc, weight_buffer).ok());
    TensorDesc activation_td; activation_td.shape = {2, 32}; activation_td.data_type = DataType::F32;
    TensorDesc output_td; output_td.shape = {2, 2, 2}; output_td.data_type = DataType::F32;
    Tensor activation, output;
    assert(Tensor::create(activation_td, activation_buffer, 0, activation).ok());
    assert(Tensor::create(output_td, output_buffer, 0, output).ok());
    ModelExecutionContext context;
    context.tensors.emplace("activation", activation);
    context.tensors.emplace("output", output);
    QuantizedWeightBinding binding;
    binding.buffer = weight_buffer;
    binding.byte_size = 4 * 18;
    binding.file_type_id = 2;
    binding.dimensions = {32, 4};
    context.quantized_weights.emplace("q.weight", binding);
    ModelPlanOperation operation;
    operation.id = "q8-matmul";
    operation.operator_name = "MatMul_Q4_0";
    operation.inputs = {"activation"};
    operation.weight_tensors = {"q.weight"};
    operation.output = "output";
    ComputeDispatch dispatch;
    assert((*lowerings.find(BackendKind::OpenGles, "MatMul_Q4_0"))(operation, context, dispatch).ok());
    assert(dispatch.backend == BackendKind::OpenGles);
    assert(dispatch.glsl_source.find("const uint OUTPUT_CHANNELS = 4u;") != std::string::npos);
    assert(dispatch.validation == KernelValidation::Unverified);
}

void test_q4_1_model_lowering_supports_gles_3d_projection() {
    FakeBackend backend(BackendKind::OpenGles);
    ModelLoweringRegistry lowerings;
    assert(register_builtin_model_lowerings(BackendKind::OpenGles, lowerings).ok());
    BufferDesc activation_desc{2 * 32 * sizeof(float), 4, MemoryDomain::Device, false, "q8-activation"};
    BufferDesc output_desc{2 * 4 * sizeof(float), 4, MemoryDomain::Device, false, "q8-output"};
    BufferDesc weight_desc{4 * 20, 4, MemoryDomain::Device, true, "q8-weight"};
    std::shared_ptr<Buffer> activation_buffer, output_buffer, weight_buffer;
    assert(backend.allocate(activation_desc, activation_buffer).ok());
    assert(backend.allocate(output_desc, output_buffer).ok());
    assert(backend.allocate(weight_desc, weight_buffer).ok());
    TensorDesc activation_td; activation_td.shape = {2, 32}; activation_td.data_type = DataType::F32;
    TensorDesc output_td; output_td.shape = {2, 2, 2}; output_td.data_type = DataType::F32;
    Tensor activation, output;
    assert(Tensor::create(activation_td, activation_buffer, 0, activation).ok());
    assert(Tensor::create(output_td, output_buffer, 0, output).ok());
    ModelExecutionContext context;
    context.tensors.emplace("activation", activation);
    context.tensors.emplace("output", output);
    QuantizedWeightBinding binding;
    binding.buffer = weight_buffer;
    binding.byte_size = 4 * 20;
    binding.file_type_id = 3;
    binding.dimensions = {32, 4};
    context.quantized_weights.emplace("q.weight", binding);
    ModelPlanOperation operation;
    operation.id = "q8-matmul";
    operation.operator_name = "MatMul_Q4_1";
    operation.inputs = {"activation"};
    operation.weight_tensors = {"q.weight"};
    operation.output = "output";
    ComputeDispatch dispatch;
    assert((*lowerings.find(BackendKind::OpenGles, "MatMul_Q4_1"))(operation, context, dispatch).ok());
    assert(dispatch.backend == BackendKind::OpenGles);
    assert(dispatch.glsl_source.find("const uint OUTPUT_CHANNELS = 4u;") != std::string::npos);
    assert(dispatch.validation == KernelValidation::Unverified);
}

void test_q4_0_embedding_model_lowering_supports_gles() {
    FakeBackend backend(BackendKind::OpenGles);
    ModelLoweringRegistry lowerings;
    assert(register_builtin_model_lowerings(BackendKind::OpenGles, lowerings).ok());
    BufferDesc weight_desc{36, 4, MemoryDomain::Device, true, "q4-embedding-weight"};
    BufferDesc output_desc{2 * 32 * sizeof(float), 4, MemoryDomain::Device, false, "q4-embedding-output"};
    std::shared_ptr<Buffer> weight_buffer, output_buffer;
    assert(backend.allocate(weight_desc, weight_buffer).ok());
    assert(backend.allocate(output_desc, output_buffer).ok());
    TensorDesc output_td; output_td.shape = {2, 32}; output_td.data_type = DataType::F32;
    Tensor output;
    assert(Tensor::create(output_td, output_buffer, 0, output).ok());
    ModelExecutionContext context;
    context.backend = &backend;
    context.token_ids = {1, 0};
    context.tensors.emplace("embedding", output);
    QuantizedWeightBinding binding;
    binding.buffer = weight_buffer;
    binding.byte_size = 36;
    binding.file_type_id = 2;
    binding.dimensions = {32, 2};
    context.quantized_weights.emplace("token_embd.weight", binding);
    ModelPlanOperation operation;
    operation.id = "embedding";
    operation.operator_name = "Embedding_Q4_0";
    operation.inputs = {"token_ids"};
    operation.weight_tensors = {"token_embd.weight"};
    operation.output = "embedding";
    ComputeDispatch dispatch;
    assert((*lowerings.find(BackendKind::OpenGles, "Embedding_Q4_0"))(operation, context, dispatch).ok());
    assert(dispatch.backend == BackendKind::OpenGles);
    assert(dispatch.kernel_id == "gles-q4-0-embedding-f32-v1");
    assert(dispatch.glsl_source.find("VOCABULARY_SIZE = 2u") != std::string::npos);
    assert(dispatch.validation == KernelValidation::Unverified);
}

void test_q8_0_embedding_model_lowering_supports_gles() {
    FakeBackend backend(BackendKind::OpenGles);
    ModelLoweringRegistry lowerings;
    assert(register_builtin_model_lowerings(BackendKind::OpenGles, lowerings).ok());
    BufferDesc weight_desc{68, 4, MemoryDomain::Device, true, "q8-embedding-weight"};
    BufferDesc output_desc{2 * 32 * sizeof(float), 4, MemoryDomain::Device, false, "q8-embedding-output"};
    std::shared_ptr<Buffer> weight_buffer, output_buffer;
    assert(backend.allocate(weight_desc, weight_buffer).ok());
    assert(backend.allocate(output_desc, output_buffer).ok());
    TensorDesc output_td; output_td.shape = {2, 32}; output_td.data_type = DataType::F32;
    Tensor output;
    assert(Tensor::create(output_td, output_buffer, 0, output).ok());
    ModelExecutionContext context;
    context.backend = &backend;
    context.token_ids = {1, 0};
    context.tensors.emplace("embedding", output);
    QuantizedWeightBinding binding;
    binding.buffer = weight_buffer;
    binding.byte_size = 68;
    binding.file_type_id = 8;
    binding.dimensions = {32, 2};
    context.quantized_weights.emplace("token_embd.weight", binding);
    ModelPlanOperation operation;
    operation.id = "embedding";
    operation.operator_name = "Embedding_Q8_0";
    operation.inputs = {"token_ids"};
    operation.weight_tensors = {"token_embd.weight"};
    operation.output = "embedding";
    ComputeDispatch dispatch;
    assert((*lowerings.find(BackendKind::OpenGles, "Embedding_Q8_0"))(operation, context, dispatch).ok());
    assert(dispatch.backend == BackendKind::OpenGles);
    assert(dispatch.kernel_id == "gles-q8-0-embedding-f32-v1");
    assert(dispatch.glsl_source.find("VOCABULARY_SIZE = 2u") != std::string::npos);
    assert(dispatch.validation == KernelValidation::Unverified);
}

void test_q8_0_embedding_model_lowering_supports_dawn() {
    FakeBackend backend(BackendKind::DawnWebGpu);
    ModelLoweringRegistry lowerings;
    assert(register_builtin_model_lowerings(BackendKind::DawnWebGpu, lowerings).ok());
    BufferDesc weight_desc{68, 4, MemoryDomain::Device, true, "q8-embedding-weight"};
    BufferDesc output_desc{2 * 32 * sizeof(float), 4, MemoryDomain::Device, false, "q8-embedding-output"};
    std::shared_ptr<Buffer> weight_buffer, output_buffer;
    assert(backend.allocate(weight_desc, weight_buffer).ok());
    assert(backend.allocate(output_desc, output_buffer).ok());
    TensorDesc output_td; output_td.shape = {2, 32}; output_td.data_type = DataType::F32;
    Tensor output;
    assert(Tensor::create(output_td, output_buffer, 0, output).ok());
    ModelExecutionContext context;
    context.backend = &backend;
    context.token_ids = {1, 0};
    context.tensors.emplace("embedding", output);
    QuantizedWeightBinding binding;
    binding.buffer = weight_buffer;
    binding.byte_size = 68;
    binding.file_type_id = 8;
    binding.dimensions = {32, 2};
    context.quantized_weights.emplace("token_embd.weight", binding);
    ModelPlanOperation operation;
    operation.id = "embedding";
    operation.operator_name = "Embedding_Q8_0";
    operation.inputs = {"token_ids"};
    operation.weight_tensors = {"token_embd.weight"};
    operation.output = "embedding";
    ComputeDispatch dispatch;
    assert((*lowerings.find(BackendKind::DawnWebGpu, "Embedding_Q8_0"))(operation, context, dispatch).ok());
    assert(dispatch.backend == BackendKind::DawnWebGpu);
    assert(dispatch.kernel_id == "dawn-q8-0-embedding-f32-v1");
    assert(dispatch.wgsl_source.find("VOCABULARY_SIZE: u32 = 2u") != std::string::npos);
    assert(dispatch.validation == KernelValidation::Unverified);
}

void test_streamed_execution_loads_and_releases_quantized_weights_per_operation() {
    const std::string path = "/tmp/ai_chat_gpu_engine_streamed_q4.gguf";
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
        write_string(out, "q4_weight");
        write_u32(out, 2);
        write_u64(out, 32); write_u64(out, 2);
        write_u32(out, 2);
        write_u64(out, 0);
        const auto pos = static_cast<uint64_t>(out.tellp());
        const auto data_start = (pos + 31u) & ~31ULL;
        for (uint64_t i = pos; i < data_start; ++i) out.put('\0');
        uint8_t q4_weights[36] = {};
        for (uint32_t row = 0; row < 2; ++row) {
            const size_t offset = row * 18;
            q4_weights[offset] = 0x00;
            q4_weights[offset + 1] = 0x38;
            for (uint32_t i = 0; i < 16; ++i)
                q4_weights[offset + 2 + i] = static_cast<uint8_t>((8u + i % 8u) | ((8u + (i + 1u) % 8u) << 4u));
        }
        out.write(reinterpret_cast<const char*>(q4_weights), sizeof(q4_weights));
    }
    GgufReader reader;
    assert(GgufReader::open(path, reader).ok());

    FakeBackend backend(BackendKind::OpenGles);
    MemoryManager memory(backend, 1024);
    DeviceInfo device;
    device.limits.max_buffer_size = 1024;
    device.limits.max_storage_buffer_binding_size = 1024;
    device.limits.max_compute_workgroups_per_dimension = 65535;

    ModelExecutionPlan plan;
    plan.structurally_validated = true;
    plan.operator_coverage_validated = true;
    plan.operations.push_back({"q4-matmul", "MatMul_Q4_0", {"hidden"},
        {"q4_weight"}, "logits", {1, 2}, {}, {}, {}, {}, false});
    plan.retained_outputs = {"logits"};

    BufferDesc input_desc{2 * 32 * sizeof(float), alignof(float), MemoryDomain::Device, false, "streamed-hidden"};
    std::shared_ptr<Buffer> input_buffer;
    assert(backend.allocate(input_desc, input_buffer).ok());
    TensorDesc input_td; input_td.shape = {2, 32}; input_td.data_type = DataType::F32;
    Tensor input;
    assert(Tensor::create(input_td, input_buffer, 0, input).ok());

    ModelExecutionContext context;
    context.tensors.emplace("hidden", input);
    context.token_ids = {0, 1};
    context.prefill = true;

    OperatorRegistry operators;
    KernelDefinition q4_kernel;
    q4_kernel.operator_name = "MatMul_Q4_0";
    q4_kernel.backend = BackendKind::OpenGles;
    q4_kernel.input_type = DataType::F32;
    q4_kernel.input_types = {DataType::F32, DataType::U32};
    q4_kernel.output_type = DataType::F32;
    q4_kernel.min_input_count = q4_kernel.max_input_count = 2;
    q4_kernel.min_rank = 2;
    q4_kernel.max_rank = 3;
    q4_kernel.validation = KernelValidation::Validated;
    q4_kernel.kernel_id = "gles-q4-0-matmul-f32-activation-v1";
    q4_kernel.glsl_source = q4_0_matmul_glsl_source();
    assert(operators.register_kernel(std::move(q4_kernel)).ok());

    ModelLoweringRegistry lowerings;
    assert(register_builtin_model_lowerings(BackendKind::OpenGles, lowerings).ok());
    ModelGraphCompileReport report;
    const Status streamed_status = execute_model_execution_plan_streamed(
        plan, backend, lowerings, operators, {}, reader, memory, device, 2,
        nullptr, context, report);
    if (!streamed_status.ok()) {
        std::cerr << "streamed execution failed: " << streamed_status.message << '\n';
        for (const auto& error : report.errors) std::cerr << "  " << error << '\n';
        for (const auto& unresolved : report.unresolved_inputs) std::cerr << "  unresolved: " << unresolved << '\n';
        for (const auto& missing : report.missing_lowerings) std::cerr << "  missing lowering: " << missing << '\n';
        for (const auto& missing : report.missing_weight_tensors) std::cerr << "  missing weight: " << missing << '\n';
    }
    assert(streamed_status.ok());
    assert(report.compiled && report.executed && report.compiled_nodes == 1);
    assert(context.quantized_weights.empty());
    assert(context.tensors.find("q4_weight") == context.tensors.end());
    assert(context.tensors.at("logits").descriptor().shape == std::vector<uint64_t>({1, 2}));
    assert(context.weight_leases.empty());
    assert(memory.allocated_bytes() == 2 * sizeof(float));
    assert(plan.execution_validated && !plan.gpu_ready);
    std::remove(path.c_str());
}

void test_q4_1_embedding_model_lowering_supports_gles() {
    FakeBackend backend(BackendKind::OpenGles);
    ModelLoweringRegistry lowerings;
    assert(register_builtin_model_lowerings(BackendKind::OpenGles, lowerings).ok());
    BufferDesc weight_desc{40, 4, MemoryDomain::Device, true, "q4-1-embedding-weight"};
    BufferDesc output_desc{2 * 32 * sizeof(float), 4, MemoryDomain::Device, false, "q4-1-embedding-output"};
    std::shared_ptr<Buffer> weight_buffer, output_buffer;
    assert(backend.allocate(weight_desc, weight_buffer).ok());
    assert(backend.allocate(output_desc, output_buffer).ok());
    TensorDesc output_td; output_td.shape = {2, 32}; output_td.data_type = DataType::F32;
    Tensor output;
    assert(Tensor::create(output_td, output_buffer, 0, output).ok());
    ModelExecutionContext context;
    context.backend = &backend;
    context.token_ids = {1, 0};
    context.tensors.emplace("embedding", output);
    QuantizedWeightBinding binding;
    binding.buffer = weight_buffer;
    binding.byte_size = 40;
    binding.file_type_id = 3;
    binding.dimensions = {32, 2};
    context.quantized_weights.emplace("token_embd.weight", binding);
    ModelPlanOperation operation;
    operation.id = "embedding";
    operation.operator_name = "Embedding_Q4_1";
    operation.inputs = {"token_ids"};
    operation.weight_tensors = {"token_embd.weight"};
    operation.output = "embedding";
    ComputeDispatch dispatch;
    assert((*lowerings.find(BackendKind::OpenGles, "Embedding_Q4_1"))(operation, context, dispatch).ok());
    assert(dispatch.backend == BackendKind::OpenGles);
    assert(dispatch.kernel_id == "gles-q4-1-embedding-f32-v1");
    assert(dispatch.glsl_source.find("WEIGHT_ROW_BYTES = 20u") != std::string::npos);
}

void test_embedding_model_lowering_uploads_token_ids_and_binds_buffer() {
    FakeBackend backend(BackendKind::OpenGles);
    ModelLoweringRegistry lowerings;
    assert(register_builtin_model_lowerings(BackendKind::OpenGles, lowerings).ok());
    ModelExecutionContext context;
    context.backend = &backend;
    context.token_ids = {2, 0};
    BufferDesc weights_desc{24, 4, MemoryDomain::Device, false, "embedding-weights"};
    BufferDesc output_desc{16, 4, MemoryDomain::Device, false, "embedding-output"};
    std::shared_ptr<Buffer> weights_buffer, output_buffer;
    assert(backend.allocate(weights_desc, weights_buffer).ok());
    assert(backend.allocate(output_desc, output_buffer).ok());
    TensorDesc weights_td; weights_td.shape = {3, 2}; weights_td.data_type = DataType::F32;
    TensorDesc output_td; output_td.shape = {2, 2}; output_td.data_type = DataType::F32;
    Tensor weights, output;
    assert(Tensor::create(weights_td, weights_buffer, 0, weights).ok());
    assert(Tensor::create(output_td, output_buffer, 0, output).ok());
    context.tensors.emplace("token_embd.weight", weights);
    context.tensors.emplace("hidden", output);
    ModelPlanOperation operation;
    operation.id = "embedding";
    operation.operator_name = "Embedding";
    operation.inputs = {"token_ids"};
    operation.weight_tensors = {"token_embd.weight"};
    operation.output = "hidden";
    operation.output_shape = {0, 2};
    ComputeDispatch dispatch;
    assert((*lowerings.find(BackendKind::OpenGles, "Embedding"))(operation, context, dispatch).ok());
    assert(dispatch.backend == BackendKind::OpenGles);
    assert(dispatch.bindings.size() == 3);
    assert(dispatch.bindings[1].size_bytes == 2 * sizeof(uint32_t));
    assert(dispatch.glsl_source.find("const uint TOKEN_COUNT = 2u;") != std::string::npos);
}




void test_activation_allocator_reuses_non_overlapping_tensor_slots() {
    FakeBackend backend(BackendKind::OpenGles);
    MemoryManager memory(backend, 128);
    ModelExecutionPlan plan;
    plan.structurally_validated = true;
    ModelPlanOperation first; first.id = "first"; first.operator_name = "Add";
    first.inputs = {"a", "b"}; first.output = "x"; first.output_shape = {0, 4};
    ModelPlanOperation second; second.id = "second"; second.operator_name = "Add";
    second.inputs = {"x", "c"}; second.output = "y"; second.output_shape = {0, 4};
    ModelPlanOperation third; third.id = "third"; third.operator_name = "Add";
    third.inputs = {"y", "d"}; third.output = "z"; third.output_shape = {0, 4};
    ModelPlanOperation fourth; fourth.id = "fourth"; fourth.operator_name = "Add";
    fourth.inputs = {"z", "e"}; fourth.output = "w"; fourth.output_shape = {0, 4};
    ModelPlanOperation fifth; fifth.id = "fifth"; fifth.operator_name = "Add";
    fifth.inputs = {"w", "f"}; fifth.output = "q"; fifth.output_shape = {0, 4};
    plan.operations = {first, second, third, fourth, fifth};
    plan.retained_outputs = {"z"};
    ModelExecutionContext context;
    assert(allocate_model_activation_tensors(plan, memory, 2, context).ok());
    assert(context.tensors.at("x").descriptor().shape == std::vector<uint64_t>({2, 4}));
    assert(context.tensors.at("z").buffer().get() == context.tensors.at("x").buffer().get());
    assert(context.tensors.at("w").buffer().get() == context.tensors.at("y").buffer().get());
    assert(context.tensors.at("q").buffer().get() != context.tensors.at("z").buffer().get());
    assert(context.activation_leases.size() == 3);
    assert(memory.allocated_bytes() == 96);

    assert(allocate_model_activation_tensors(plan, memory, 1, context).ok());
    assert(context.tensors.at("x").descriptor().shape == std::vector<uint64_t>({1, 4}));
    assert(context.tensors.at("z").descriptor().shape == std::vector<uint64_t>({1, 4}));
    assert(context.activation_leases.size() == 3);
}

void test_model_kv_cache_view_binding_uses_planned_sequence_length() {
    FakeBackend backend(BackendKind::OpenGles);
    MemoryManager memory(backend, 4096);
    KvCacheConfig config; config.layers = 1; config.kv_heads = 2;
    config.head_dimension = 2; config.maximum_sequence_length = 8;
    std::unique_ptr<KvCache> cache;
    assert(KvCache::create(backend, memory, config, cache).ok());
    ModelExecutionPlan plan;
    ModelPlanOperation attention;
    attention.id = "layer.0.attention";
    attention.operator_name = "Attention";
    attention.inputs = {"layer.0.query", "layer.0.cache_key", "layer.0.cache_value"};
    attention.integer_attributes["layer_index"] = 0;
    plan.operations.push_back(attention);
    ModelExecutionContext context;
    assert(bind_model_kv_cache_views(plan, *cache, 3, 2, context).ok());
    assert(context.past_length == 3 && context.sequence_length == 2);
    assert(context.kv_cache == cache.get());
    assert(context.tensors.at("layer.0.cache_key").descriptor().shape ==
           std::vector<uint64_t>({5, 2, 2}));
    assert(context.tensors.at("layer.0.cache_value").element_count() == 20);
    assert(bind_model_kv_cache_views(plan, *cache, 2, 1, context).ok());
    assert(context.tensors.at("layer.0.cache_key").descriptor().shape ==
           std::vector<uint64_t>({3, 2, 2}));
    assert(context.past_length == 2 && context.sequence_length == 1);
    ModelExecutionContext invalid_context;
    assert(bind_model_kv_cache_views(plan, *cache, 7, 2, invalid_context).code ==
           StatusCode::InvalidArgument);
}

void test_model_graph_executes_kv_copy_node_before_committing_cache_state() {
    FakeBackend backend(BackendKind::OpenGles);
    MemoryManager memory(backend, 4096);
    KvCacheConfig config; config.layers = 1; config.kv_heads = 2;
    config.head_dimension = 2; config.maximum_sequence_length = 4;
    std::unique_ptr<KvCache> cache;
    assert(KvCache::create(backend, memory, config, cache).ok());

    BufferDesc source_desc{16, 4, MemoryDomain::Device, false, "graph-kv-source"};
    std::shared_ptr<Buffer> key_buffer, value_buffer;
    assert(backend.allocate(source_desc, key_buffer).ok());
    assert(backend.allocate(source_desc, value_buffer).ok());
    const float key_values[] = {1, 2, 3, 4};
    const float val_values[] = {5, 6, 7, 8};
    auto key_fake = std::dynamic_pointer_cast<FakeBuffer>(key_buffer);
    auto value_fake = std::dynamic_pointer_cast<FakeBuffer>(value_buffer);
    std::memcpy(key_fake->bytes.data(), key_values, sizeof(key_values));
    std::memcpy(value_fake->bytes.data(), val_values, sizeof(val_values));
    TensorDesc td; td.shape = {1, 2, 2}; td.data_type = DataType::F32;
    Tensor key_tensor, value_tensor;
    assert(Tensor::create(td, key_buffer, 0, key_tensor).ok());
    assert(Tensor::create(td, value_buffer, 0, value_tensor).ok());

    ModelExecutionPlan plan;
    plan.structurally_validated = true;
    plan.operator_coverage_validated = true;
    ModelPlanOperation append;
    append.id = "cache-append";
    append.operator_name = "KVCacheAppend";
    append.inputs = {"key", "value"};
    append.output = "cache-ready";
    append.side_effect = true;
    append.integer_attributes["layer_index"] = 0;
    plan.operations.push_back(append);
    ModelExecutionContext context;
    context.kv_cache = cache.get();
    context.past_length = 0;
    context.sequence_length = 1;
    context.tensors.emplace("key", key_tensor);
    context.tensors.emplace("value", value_tensor);

    ModelLoweringRegistry lowerings;
    assert(register_builtin_model_lowerings(BackendKind::OpenGles, lowerings).ok());
    OperatorRegistry operators;
    ModelGraphCompileReport report;
    assert(execute_model_execution_plan(plan, backend, lowerings, operators,
                                        {"key", "value"}, context, nullptr, report).ok());
    assert(report.compiled && report.executed && report.compiled_nodes == 1);
    assert(plan.execution_validated && !plan.gpu_ready);
    assert(cache->valid_tokens(0) == 1);
    auto stored_keys = std::dynamic_pointer_cast<FakeBuffer>(cache->key_buffer(0));
    auto stored_values = std::dynamic_pointer_cast<FakeBuffer>(cache->value_buffer(0));
    assert(std::memcmp(stored_keys->bytes.data(), key_values, sizeof(key_values)) == 0);
    assert(std::memcmp(stored_values->bytes.data(), val_values, sizeof(val_values)) == 0);
}

void test_model_execution_never_promotes_unvalidated_kernels() {
    FakeBackend backend(BackendKind::OpenGles);
    ModelExecutionPlan plan;
    plan.structurally_validated = true;
    plan.operator_coverage_validated = true;
    ModelPlanOperation operation;
    operation.id = "add";
    operation.operator_name = "Add";
    operation.inputs = {"lhs", "rhs"};
    operation.output = "sum";
    operation.output_shape = {4};
    plan.operations.push_back(operation);

    ModelExecutionContext context;
    for (const char* name : {"lhs", "rhs", "sum"}) {
        BufferDesc desc{16, 4, MemoryDomain::Device, false, name};
        std::shared_ptr<Buffer> buffer;
        assert(backend.allocate(desc, buffer).ok());
        TensorDesc td; td.shape = {4}; td.data_type = DataType::F32;
        Tensor tensor;
        assert(Tensor::create(td, buffer, 0, tensor).ok());
        context.tensors.emplace(name, tensor);
    }
    ModelLoweringRegistry lowerings;
    assert(register_builtin_model_lowerings(BackendKind::OpenGles, lowerings).ok());
    OperatorRegistry unvalidated;
    ModelGraphCompileReport report;
    const Status result = execute_model_execution_plan(
        plan, backend, lowerings, unvalidated, {}, context, nullptr, report);
    assert(result.code == StatusCode::Unsupported);
    assert(!report.compiled);
    assert(!report.executed);
    assert(!plan.execution_validated);
    assert(!plan.gpu_ready);
    assert(backend.submit_count == 0);
}

void test_builtin_model_lowerings_are_backend_scoped_and_fail_closed() {
    ModelLoweringRegistry dawn, gles;
    assert(register_builtin_model_lowerings(BackendKind::DawnWebGpu, dawn).ok());
    assert(register_builtin_model_lowerings(BackendKind::OpenGles, gles).ok());
    assert(dawn.find(BackendKind::DawnWebGpu, "Add") != nullptr);
    assert(gles.find(BackendKind::OpenGles, "MatMul") != nullptr);
    assert(gles.find(BackendKind::OpenGles, "Attention") != nullptr);
    assert(gles.find(BackendKind::OpenGles, "KVCacheAppend") != nullptr);
    assert(register_builtin_model_lowerings(BackendKind::Unknown, dawn).code == StatusCode::Unsupported);

    FakeBackend backend(BackendKind::OpenGles);
    ModelExecutionContext context;
    for (const char* name : {"a", "b", "out"}) {
        BufferDesc desc{16, 4, MemoryDomain::Device, false, name};
        std::shared_ptr<Buffer> buffer;
        assert(backend.allocate(desc, buffer).ok());
        TensorDesc td; td.shape = {4}; td.data_type = DataType::F32;
        Tensor tensor;
        assert(Tensor::create(td, buffer, 0, tensor).ok());
        context.tensors.emplace(name, tensor);
    }
    ModelPlanOperation operation;
    operation.id = "add";
    operation.operator_name = "Add";
    operation.inputs = {"a", "b"};
    operation.output = "out";
    ComputeDispatch dispatch;
    assert((*gles.find(BackendKind::OpenGles, "Add"))(operation, context, dispatch).ok());
    assert(dispatch.backend == BackendKind::OpenGles);
    assert(dispatch.kernel_id == "gles-add-f32-generic-v1");
    assert(dispatch.validation == KernelValidation::Unverified);
}

void test_model_plan_graph_compilation_is_fail_closed() {
    ModelExecutionPlan plan;
    plan.structurally_validated = true;
    plan.operator_coverage_validated = true;
    plan.operations = {
        {"add", "Add", {"input_a", "input_b"}, {}, "sum", {4}, {}, {}, {}, {}, false},
        {"mul", "Multiply", {"sum", "input_c"}, {}, "product", {4}, {}, {}, {}, {}, false}
    };
    FakeBackend backend;
    ModelExecutionContext context;
    const auto bind_tensor = [&](const std::string& name) {
        BufferDesc buffer_desc{16, 4, MemoryDomain::Device, false, name};
        std::shared_ptr<Buffer> buffer;
        assert(backend.allocate(buffer_desc, buffer).ok());
        TensorDesc tensor_desc; tensor_desc.shape = {4}; tensor_desc.data_type = DataType::F32;
        Tensor tensor;
        assert(Tensor::create(tensor_desc, buffer, 0, tensor).ok());
        context.tensors.emplace(name, tensor);
    };
    for (const char* name : {"input_a", "input_b", "input_c", "sum", "product"})
        bind_tensor(name);

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
            [kernel_id = item.second](const ModelPlanOperation& operation,
                                      const ModelExecutionContext& context,
                                      ComputeDispatch& dispatch) {
                const Tensor& lhs = context.tensors.at(operation.inputs[0]);
                const Tensor& rhs = context.tensors.at(operation.inputs[1]);
                const Tensor& output = context.tensors.at(operation.output);
                dispatch.backend = BackendKind::OpenGles;
                dispatch.kernel_id = kernel_id;
                dispatch.glsl_source = "#version 310 es\nvoid main(){}";
                dispatch.workgroups_x = 1;
                dispatch.bindings = {
                    {0, lhs.buffer(), lhs.byte_offset(), lhs.byte_size()},
                    {1, rhs.buffer(), rhs.byte_offset(), rhs.byte_size()},
                    {2, output.buffer(), output.byte_offset(), output.byte_size()}
                };
                return Status::success();
            }).ok());
    }
    ComputeGraph graph;
    ModelGraphCompileReport report;
    assert(compile_model_execution_plan(plan, BackendKind::OpenGles, lowerings, operators,
        {"input_a", "input_b", "input_c"}, context, graph, report).ok());
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
        {"input_a", "input_b", "input_c"}, context, blocked_graph, report).code == StatusCode::NotReady);
    assert(blocked_graph.node_count() == 0);

    ModelExecutionContext incomplete_context = context;
    incomplete_context.tensors.erase("input_c");
    ModelExecutionPlan missing_input = plan;
    missing_input.graph_compiled = false;
    ComputeGraph incomplete_graph;
    assert(compile_model_execution_plan(missing_input, BackendKind::OpenGles, lowerings, operators,
        {"input_a", "input_b"}, incomplete_context, incomplete_graph, report).code == StatusCode::Unsupported);
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
    assert(plan.operations.size() == 20);
    assert(plan.operations.front().operator_name == "Embedding");
    assert(plan.operations.front().output_shape == std::vector<uint64_t>({0, 8}));
    assert(plan.operations.back().operator_name == "MatMul");
    assert(plan.operations.back().output_shape == std::vector<uint64_t>({0, 2}));
    assert(plan.vocabulary_size == 2 && plan.head_dimension == 4);
    const auto rope_op = std::find_if(plan.operations.begin(), plan.operations.end(),
        [](const auto& op) { return op.operator_name == "RoPE"; });
    assert(rope_op != plan.operations.end());
    assert(rope_op->integer_attributes.at("rotary_dimensions") == 4);
    assert(rope_op->float_attributes.at("theta") == 10000.0);
    const auto attention_op = std::find_if(plan.operations.begin(), plan.operations.end(),
        [](const auto& op) { return op.operator_name == "Attention"; });
    assert(attention_op != plan.operations.end());
    assert(attention_op->integer_attributes.at("kv_heads") == 1);
    assert(attention_op->bool_attributes.at("causal"));
    assert(attention_op->inputs.size() == 3);
    assert(attention_op->control_inputs.size() == 1);
    const auto cache_append = std::find_if(plan.operations.begin(), plan.operations.end(),
        [](const auto& op) { return op.operator_name == "KVCacheAppend"; });
    assert(cache_append != plan.operations.end() && cache_append->side_effect);
    assert(cache_append->integer_attributes.at("layer_index") == 0);
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
    assert(plan.vocabulary_size == 2 && plan.head_dimension == 4);
    assert(std::find_if(plan.operations.begin(), plan.operations.end(), [](const auto& op) {
        return op.id == "layer.0.attention_output_bias.apply" && op.operator_name == "Add";
    }) != plan.operations.end());
    assert(std::find_if(plan.operations.begin(), plan.operations.end(), [](const auto& op) {
        return op.id == "layer.0.q_bias.apply" && op.operator_name == "Add";
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
    ComputeDispatch tampered;
    tampered.kernel_id = selected.kernel_id;
    tampered.wgsl_source = selected.wgsl_source + "\n// changed shader body";
    assert(registry.authorize_dispatch(selected.kernel_id, selected.backend, tampered).code ==
           StatusCode::InvalidArgument);

    KernelDefinition specialized;
    specialized.operator_name = "Specialized";
    specialized.kernel_id = "specialized-f32-v1";
    specialized.backend = BackendKind::DawnWebGpu;
    specialized.input_type = DataType::F32;
    specialized.output_type = DataType::F32;
    specialized.min_input_count = specialized.max_input_count = 1;
    specialized.min_rank = specialized.max_rank = 1;
    specialized.validation = KernelValidation::Validated;
    specialized.wgsl_source = "const ELEMENT_COUNT: u32 = 1u;\n@compute @workgroup_size(64) fn main() {}";
    assert(registry.register_kernel(specialized).ok());
    ComputeDispatch specialized_dispatch;
    specialized_dispatch.kernel_id = specialized.kernel_id;
    specialized_dispatch.wgsl_source = "const ELEMENT_COUNT: u32 = 4u;\n@compute @workgroup_size(64) fn main() {}";
    assert(registry.authorize_dispatch(specialized.kernel_id, specialized.backend,
                                       specialized_dispatch).ok());
    ComputeDispatch altered_specialization = specialized_dispatch;
    altered_specialization.wgsl_source = "const ELEMENT_COUNT: u32 = 4u;\n@compute @workgroup_size(64) fn main() { let x = 1u; }";
    assert(registry.authorize_dispatch(specialized.kernel_id, specialized.backend,
                                       altered_specialization).code == StatusCode::InvalidArgument);

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
        if (backend == BackendKind::OpenGles) {
            definition.glsl_source = "#version 310 es\nlayout(local_size_x=1) in; void main() {}";
        } else {
            definition.wgsl_source = "@compute @workgroup_size(1) fn main() {}";
        }
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

void test_sampler_min_p_filter() {
    SamplingConfig config;
    config.temperature = 1.0f;
    config.top_k = 0;
    config.top_p = 1.0f;
    config.min_p = 0.5f;
    Sampler sampler(config);
    const std::vector<float> logits = {8.0f, 2.0f, 1.0f};
    const std::vector<int32_t> history;
    for (int i = 0; i < 8; ++i) {
        int32_t token = -1;
        assert(sampler.sample(logits, history, token).ok());
        assert(token == 0);
    }
    config.min_p = 1.1f;
    Sampler invalid_sampler(config);
    int32_t ignored = -1;
    assert(invalid_sampler.sample(logits, history, ignored).code == StatusCode::InvalidArgument);
}

int main() {
    test_sampler_min_p_filter();
    test_independent_gguf_reader();
    test_tensor_and_memory_lease();
    test_operator_registry_fail_closed();
    test_universal_gles_operator_candidate_is_fail_closed();
    test_model_plan_graph_compilation_is_fail_closed();
    test_builtin_model_lowerings_are_backend_scoped_and_fail_closed();
    test_model_execution_never_promotes_unvalidated_kernels();
    test_model_graph_executes_kv_copy_node_before_committing_cache_state();
    test_model_kv_cache_view_binding_uses_planned_sequence_length();
    test_activation_allocator_reuses_non_overlapping_tensor_slots();
    test_q4_0_embedding_model_lowering_supports_gles();
    test_q8_0_embedding_model_lowering_supports_gles();
    test_q8_0_embedding_model_lowering_supports_dawn();
    test_q4_1_embedding_model_lowering_supports_gles();
    test_streamed_execution_loads_and_releases_quantized_weights_per_operation();
    test_embedding_model_lowering_uploads_token_ids_and_binds_buffer();
    test_q8_0_model_lowering_supports_gles_3d_projection();
    test_q4_0_model_lowering_supports_gles_3d_projection();
    test_q4_1_model_lowering_supports_gles_3d_projection();
    test_fake_backend_never_promotes_builtin_operator_kernels();
    test_dense_model_plan_selects_q8_0_matmul_adapter();
    test_dense_model_plan_selects_q4_0_matmul_adapter();
    test_dense_model_plan_selects_q4_1_matmul_adapter();
    test_dense_model_plan_selects_q5_matmul_adapter(6, "MatMul_Q5_0");
    test_dense_model_plan_selects_q5_matmul_adapter(7, "MatMul_Q5_1");
    test_dense_model_plan_selects_fp16_matmul_adapter(1,"MatMul_F16");
    test_dense_model_plan_selects_fp16_matmul_adapter(30,"MatMul_BF16");
    test_dense_model_plan_selects_q6_k_matmul_adapter();
    test_dense_model_plan_selects_k_quant_matmul_adapter(12,"MatMul_Q4_K");
    test_dense_model_plan_selects_k_quant_matmul_adapter(13,"MatMul_Q5_K");
    test_dense_model_plan_selects_k23_matmul_adapter(10,"MatMul_Q2_K");
    test_dense_model_plan_selects_k23_matmul_adapter(11,"MatMul_Q3_K");
    test_llama_model_execution_plan_is_structural_only();
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
    test_q6_k_decoder_layout();
    test_q4_k_q5_k_decoder_layouts();
    test_q2_k_q3_k_decoder_layouts();
    test_f16_bf16_reference_matmul_decoding();
    test_dawn_q8_0_operator_registration_is_unverified();
    test_q5_matmul_lowerings_are_registered_for_both_backends();
    test_q8_0_gpu_validation_does_not_promote_fake_backend();
    test_quantization_backend_compatibility_is_backend_specific();
    test_q8_0_decoder_and_registry();
    test_kv_cache_memory_and_sequential_append();
    test_kv_cache_device_append_avoids_host_roundtrip();
    test_model_gpu_memory_estimate_includes_kv_and_scratch();
    test_gpu_backend_fallback_order();
    test_gpu_memory_planner_respects_streaming_and_reserve();
    test_graph_scheduler_order_and_cancel();
    std::cout << "GPU engine contract tests passed\n";
    return 0;
}
