#include "model_graph.h"
#include "kv_cache.h"
#include "execution_engine.h"
#include "memory_planner.h"
#include "weight_streamer.h"
#include "quantization.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

namespace ai_chat::gpu_engine {

namespace {

void append_report_items(std::string& message, const char* label,
                         const std::vector<std::string>& items) {
    constexpr size_t kMaximumItems = 12;
    for (size_t index = 0; index < items.size() && index < kMaximumItems; ++index) {
        message += " | ";
        message += label;
        message += ":";
        message += items[index];
    }
    if (items.size() > kMaximumItems) {
        message += " | ";
        message += label;
        message += ":...(" + std::to_string(items.size() - kMaximumItems) + " more)";
    }
}

std::string describe_compile_failure(const ModelGraphCompileReport& report) {
    std::string message = "Model execution plan could not be fully lowered into a validated compute graph";
    message += " | compiled_nodes=" + std::to_string(report.compiled_nodes);
    append_report_items(message, "missing_lowering", report.missing_lowerings);
    append_report_items(message, "unresolved_input", report.unresolved_inputs);
    append_report_items(message, "missing_weight", report.missing_weight_tensors);
    append_report_items(message, "error", report.errors);
    return message;
}




uint64_t storage_binding_limit(const DeviceInfo& device) {
    const uint64_t storage = device.limits.max_storage_buffer_binding_size;
    const uint64_t buffer = device.limits.max_buffer_size;
    if (storage == 0) return buffer;
    if (buffer == 0) return storage;
    return std::min(storage, buffer);
}

bool checked_product(uint64_t a, uint64_t b, uint64_t& result) {
    if (a != 0 && b > std::numeric_limits<uint64_t>::max() / a) return false;
    result = a * b;
    return true;
}

Status allocate_chunk_buffer(MemoryManager& memory, const DeviceInfo& device,
                             uint64_t bytes, const std::string& label,
                             MemoryLease& lease) {
    if (bytes == 0 || bytes > UINT64_MAX - 3)
        return {StatusCode::InvalidArgument, "Chunk buffer size is zero or overflows"};
    BufferDesc descriptor;
    descriptor.size_bytes = (bytes + 3) & ~uint64_t{3};
    descriptor.alignment = std::max<uint64_t>(4, device.limits.min_storage_buffer_offset_alignment);
    descriptor.memory = MemoryDomain::Device;
    descriptor.persistent = false;
    descriptor.label = label;
    return memory.allocate(descriptor, lease);
}

Status verify_chunk_output_binding(const ComputeDispatch& dispatch, BackendKind backend,
                                   const std::shared_ptr<Buffer>& expected_buffer,
                                   uint64_t expected_bytes) {
    if (dispatch.copy_only || dispatch.backend != backend || dispatch.kernel_id.empty() ||
        dispatch.workgroups_x == 0 || dispatch.workgroups_y == 0 || dispatch.workgroups_z == 0)
        return {StatusCode::Unsupported, "Chunk lowering returned an incomplete or incompatible compute dispatch"};
    const auto binding = std::find_if(dispatch.bindings.begin(), dispatch.bindings.end(),
        [](const BufferBinding& item) { return item.binding == 2; });
    if (binding == dispatch.bindings.end() || binding->buffer.get() != expected_buffer.get() ||
        binding->offset_bytes != 0 || binding->size_bytes < expected_bytes)
        return {StatusCode::InvalidArgument, "Chunk lowering output binding does not target the allocated aggregation buffer"};
    return Status::success();
}

Status validate_resident_q6_k_matmul_sample(
        const ModelPlanOperation& operation, const GgufReader& reader,
        Backend& backend, const ModelExecutionContext& context,
        uint64_t* parity_checks) {
    if (!context.prefill || parity_checks == nullptr ||
        operation.operator_name != "MatMul_Q6_K" ||
        operation.inputs.size() != 1 || operation.weight_tensors.size() != 1)
        return Status::success();

    const GgufTensorInfo* source = reader.find_tensor(operation.weight_tensors.front());
    const auto input_it = context.tensors.find(operation.inputs.front());
    const auto output_it = context.tensors.find(operation.output);
    if (!source || source->type.name != "Q6_K" ||
        input_it == context.tensors.end() || output_it == context.tensors.end())
        return {StatusCode::InvalidArgument, "Real Q6_K MatMul validation could not resolve its GGUF weight or tensors"};

    const Tensor& input = input_it->second;
    const Tensor& output = output_it->second;
    const auto& input_shape = input.descriptor().shape;
    const auto& output_shape = output.descriptor().shape;
    if (!source->byte_size_known || source->dimensions.size() != 2 ||
        input.descriptor().data_type != DataType::F32 ||
        output.descriptor().data_type != DataType::F32 ||
        input_shape.size() < 2 || output_shape.size() < 2 ||
        source->dimensions[0] > UINT32_MAX || source->dimensions[1] == 0 ||
        source->dimensions[0] % 256 != 0)
        return {StatusCode::Unsupported, "Real Q6_K MatMul sample validation received incompatible tensor metadata"};

    // Keep parity validation consistent with the Q6_K lowering: preserve all
    // matching leading dimensions as rows, then flatten the remaining input
    // dimensions into K. This covers attention projections such as
    // [tokens, heads, head_dim] -> [tokens, embedding] without weakening checks.
    size_t leading_rank = 0;
    const size_t max_leading_rank = std::min(input_shape.size(), output_shape.size()) - 1;
    while (leading_rank < max_leading_rank &&
           input_shape[leading_rank] == output_shape[leading_rank]) {
        ++leading_rank;
    }
    if (leading_rank == 0)
        return {StatusCode::Unsupported, "Real Q6_K MatMul sample validation found no matching leading dimensions"};

    uint64_t rows = 1;
    for (size_t axis = 0; axis < leading_rank; ++axis) {
        if (input_shape[axis] == 0 ||
            !checked_product(rows, input_shape[axis], rows))
            return {StatusCode::Unsupported, "Real Q6_K MatMul sample validation found invalid or overflowing row dimensions"};
    }

    uint64_t inner = 1;
    for (size_t axis = leading_rank; axis < input_shape.size(); ++axis) {
        if (input_shape[axis] == 0 ||
            !checked_product(inner, input_shape[axis], inner))
            return {StatusCode::Unsupported, "Real Q6_K MatMul sample validation found invalid or overflowing flattened K dimensions"};
    }

    if (inner != source->dimensions[0] || rows == 0 ||
        output.element_count() == 0 || output.element_count() % rows != 0 ||
        output.element_count() / rows != source->dimensions[1])
        return {StatusCode::Unsupported, "Real Q6_K MatMul sample validation found GGUF dimensions inconsistent with flattened activation/output shapes"};

    const uint64_t channels = source->dimensions[1];
    const uint64_t row_bytes = (inner / 256) * 210;
    const uint32_t sample_channels = static_cast<uint32_t>(std::min<uint64_t>(4, channels));
    const uint64_t sample_weight_bytes = static_cast<uint64_t>(sample_channels) * row_bytes;
    std::vector<uint8_t> sample_weights(static_cast<size_t>(sample_weight_bytes));
    Status status = reader.read_tensor_chunk(*source, 0, sample_weights.data(), sample_weight_bytes);
    if (!status.ok()) return status;

    // Tensor storage is contiguous, so the first flattened row starts at the
    // tensor's byte offset even when its logical rank is greater than two.
    std::vector<float> activations(static_cast<size_t>(inner));
    std::vector<float> expected(sample_channels), actual(sample_channels);
    status = backend.readback(input.buffer(), input.byte_offset(), activations.data(),
                              activations.size() * sizeof(float));
    if (!status.ok()) return status;
    status = reference_q6_k_matmul(activations.data(), sample_weights.data(),
        sample_weight_bytes, expected.data(), 1, sample_channels,
        static_cast<uint32_t>(inner));
    if (!status.ok()) return status;
    status = backend.readback(output.buffer(), output.byte_offset(), actual.data(),
                              actual.size() * sizeof(float));
    if (!status.ok()) return status;

    for (size_t index = 0; index < actual.size(); ++index) {
        const float tolerance = 2.0e-3f +
            2.0e-3f * std::max(std::fabs(expected[index]), 1.0f);
        if (!std::isfinite(actual[index]) ||
            std::fabs(actual[index] - expected[index]) > tolerance) {
            return {StatusCode::NumericalFailure,
                "Resident real GGUF Q6_K MatMul parity failed for " + operation.id +
                " at sample_channel=" + std::to_string(index)};
        }
    }
    ++(*parity_checks);
    return Status::success();
}

Status execute_chunked_quantized_matmul_if_needed(
        const ModelPlanOperation& operation, Backend& backend,
        const ModelLoweringRegistry& lowerings, const OperatorRegistry& operators,
        const GgufReader& reader, MemoryManager& memory, const DeviceInfo& device,
        ModelExecutionContext& context, bool& handled, uint64_t* real_q6_k_parity_checks) {
    handled = false;
    if (operation.operator_name.rfind("MatMul_", 0) != 0 ||
        operation.inputs.size() != 1 || operation.weight_tensors.size() != 1)
        return Status::success();

    const std::string& weight_name = operation.weight_tensors.front();
    const auto weight_it = context.quantized_weights.find(weight_name);
    if (weight_it == context.quantized_weights.end() || weight_it->second.buffer)
        return Status::success();
    handled = true;

    const auto input_it = context.tensors.find(operation.inputs.front());
    const auto output_it = context.tensors.find(operation.output);
    const GgufTensorInfo* source = reader.find_tensor(weight_name);
    if (input_it == context.tensors.end() || output_it == context.tensors.end() ||
        !source || !source->byte_size_known)
        return {StatusCode::InvalidArgument, "Chunked MatMul requires bound activations, output and a known GGUF tensor"};

    const Tensor& input = input_it->second;
    const Tensor& output = output_it->second;
    const auto& dimensions = weight_it->second.dimensions;
    const auto& type = source->type;
    const auto& input_shape = input.descriptor().shape;
    const auto& output_shape = output.descriptor().shape;
    const auto describe_shape = [](const std::vector<uint64_t>& shape) {
        std::string value = "[";
        for (size_t index = 0; index < shape.size(); ++index) {
            if (index != 0) value += ",";
            value += std::to_string(shape[index]);
        }
        value += "]";
        return value;
    };
    if (input.descriptor().data_type != DataType::F32 ||
        output.descriptor().data_type != DataType::F32 ||
        input_shape.size() < 2 || input_shape.size() > 4 ||
        output_shape.empty() || output_shape.size() > 4 ||
        input.descriptor().layout != TensorLayout::Contiguous ||
        output.descriptor().layout != TensorLayout::Contiguous ||
        dimensions.size() != 2 || source->dimensions != dimensions ||
        source->type_id != weight_it->second.file_type_id ||
        !input.buffer() || !output.buffer() ||
        !type.storage_size_known || type.block_elements == 0 || type.bytes_per_block == 0)
        return {StatusCode::Unsupported,
            "Chunked MatMul metadata is incompatible: op=" + operation.id +
            " weight=" + weight_name + " quant_type=" + type.name +
            " input_shape=" + describe_shape(input_shape) +
            " output_shape=" + describe_shape(output_shape) +
            " gguf_dimensions=[" + std::to_string(dimensions[0]) + "," +
            std::to_string(dimensions[1]) + "]"};

    const uint64_t inner = dimensions[0];
    const uint64_t total_channels = dimensions[1];
    if (inner == 0 || total_channels == 0 || inner % type.block_elements != 0 ||
        total_channels > UINT32_MAX || inner > UINT32_MAX ||
        input.element_count() == 0 || input.element_count() % inner != 0)
        return {StatusCode::Unsupported,
            "Chunked MatMul activation elements do not form complete GGUF K rows: op=" +
            operation.id + " K=" + std::to_string(inner) +
            " activation_elements=" + std::to_string(input.element_count())};

    // Model plans may express activations as [batch, sequence, K] or
    // [tokens, heads, head_dim]. The quantized matrix is [K, N], so all
    // contiguous leading activation dimensions are flattened into M rows.
    // This is a view only: no host copy or data reordering is performed.
    const uint64_t rows = input.element_count() / inner;
    uint64_t expected_output_elements = 0;
    if (rows == 0 || rows > UINT32_MAX ||
        !checked_product(rows, total_channels, expected_output_elements) ||
        output.element_count() != expected_output_elements)
        return {StatusCode::Unsupported,
            "Chunked MatMul output element count does not match flattened MxN: op=" +
            operation.id + " M=" + std::to_string(rows) +
            " N=" + std::to_string(total_channels) +
            " output_elements=" + std::to_string(output.element_count())};

    uint64_t row_bytes = 0, expected_weight_bytes = 0, output_bytes_per_channel = 0;
    if (!checked_product(inner / type.block_elements, type.bytes_per_block, row_bytes) ||
        !checked_product(row_bytes, total_channels, expected_weight_bytes) ||
        !checked_product(rows, sizeof(float), output_bytes_per_channel) ||
        !checked_product(output_bytes_per_channel, total_channels, output_bytes_per_channel) ||
        row_bytes == 0 || expected_weight_bytes != source->byte_size ||
        weight_it->second.byte_size != source->byte_size)
        return {StatusCode::InvalidArgument, "Chunked MatMul GGUF row-byte calculation does not match tensor storage"};

    const uint64_t binding_limit = storage_binding_limit(device);
    if (binding_limit <= 3 || binding_limit < row_bytes || binding_limit < rows * sizeof(float))
        return {StatusCode::Unsupported, "Backend binding limit cannot hold one quantized weight row or one output channel"};
    const uint64_t output_channel_bytes = rows * sizeof(float);
    if (row_bytes > UINT64_MAX - output_channel_bytes)
        return {StatusCode::InvalidArgument, "Chunked MatMul per-channel memory estimate overflows"};
    const uint64_t bytes_per_channel = row_bytes + output_channel_bytes;
    const uint64_t padded_binding_limit = binding_limit - 3;

    const auto* lowering = lowerings.find(backend.kind(), operation.operator_name);
    if (!lowering)
        return {StatusCode::Unsupported, "No validated lowering exists for chunked MatMul: " + operation.operator_name};

    const bool validate_q6_k = context.prefill && real_q6_k_parity_checks != nullptr &&
        source->type.name == "Q6_K";
    std::vector<float> sample_activations;
    if (validate_q6_k) {
        sample_activations.resize(static_cast<size_t>(inner));
        Status status = backend.readback(input.buffer(), input.byte_offset(),
            sample_activations.data(), sample_activations.size() * sizeof(float));
        if (!status.ok()) return status;
    }

    struct PendingChunk {
        uint64_t channel_offset = 0;
        uint64_t chunk_channels = 0;
        MemoryLease weight_lease;
        MemoryLease output_lease;
        std::vector<uint8_t> sample_weights;
    };
    std::vector<PendingChunk> pending;
    std::unique_ptr<CommandBatch> commands;

    const auto flush_batch = [&]() -> Status {
        if (pending.empty()) return Status::success();
        if (!commands) return {StatusCode::BackendFailure, "Quantized MatMul batch lost its command buffer"};
        Status status = backend.submit(std::move(commands));
        if (!status.ok()) return status;
        status = backend.synchronize();
        if (!status.ok()) return status;

        if (validate_q6_k) {
            for (const auto& chunk : pending) {
                const uint32_t sample_channels = static_cast<uint32_t>(
                    std::min<uint64_t>(4, chunk.chunk_channels));
                std::vector<float> expected(sample_channels), actual(sample_channels);
                const uint64_t sample_weight_bytes = static_cast<uint64_t>(sample_channels) * row_bytes;
                status = reference_q6_k_matmul(sample_activations.data(), chunk.sample_weights.data(),
                    sample_weight_bytes, expected.data(), 1, sample_channels,
                    static_cast<uint32_t>(inner));
                if (!status.ok()) return status;
                status = backend.readback(chunk.output_lease.buffer(), 0,
                    actual.data(), actual.size() * sizeof(float));
                if (!status.ok()) return status;
                for (size_t index = 0; index < actual.size(); ++index) {
                    const float tolerance = 2.0e-3f +
                        2.0e-3f * std::max(std::fabs(expected[index]), 1.0f);
                    if (!std::isfinite(actual[index]) ||
                        std::fabs(actual[index] - expected[index]) > tolerance) {
                        return {StatusCode::NumericalFailure,
                            "Real GGUF Q6_K MatMul parity failed for " + operation.id +
                            " at channel_offset=" + std::to_string(chunk.channel_offset) +
                            " sample_channel=" + std::to_string(index)};
                    }
                }
                ++(*real_q6_k_parity_checks);
            }
        }
        pending.clear();
        return Status::success();
    };

    // Pressure mode changes how weights are sourced, not whether reusable
    // chunks may be retained. Admission remains opportunistic under live budget.
    bool cache_evicted_for_pressure = false;
    const auto reclaim_optional_memory = [&]() {
        bool had_optional_cached_weights = std::any_of(
            context.quantized_weights.begin(), context.quantized_weights.end(),
            [](const auto& entry) { return static_cast<bool>(entry.second.buffer); });
        if (context.weight_chunk_cache && !cache_evicted_for_pressure) {
            had_optional_cached_weights = had_optional_cached_weights ||
                context.weight_chunk_cache->stats().resident_bytes != 0;
            context.weight_chunk_cache->evict_all();
            cache_evicted_for_pressure = true;
        }
        if (had_optional_cached_weights) {
            context.weight_cache_pressure_mode = true;
            context.persistent_weight_cache_limit_bytes = 0;
        }
        memory.reclaim_transient_pool();
    };

    uint64_t channel_offset = 0;
    while (channel_offset < total_channels) {
        if (!backend.healthy())
            return {StatusCode::DeviceLost, "GPU backend became unhealthy during quantized MatMul chunking"};

        uint64_t available = memory.available_bytes();
        uint64_t reclaimed_optional_bytes = 0;
        // Optional cached weights yield before the planner rejects a legal tile.
        // Disable admissions for this operation so pressure cannot refill the cache.
        if (available < bytes_per_channel) {
            const uint64_t before_reclaim = available;
            reclaim_optional_memory();
            available = memory.available_bytes();
            reclaimed_optional_bytes = available > before_reclaim ? available - before_reclaim : 0;
        }
        const uint64_t available_for_chunks = available > 3 ? available - 3 : 0;
        uint64_t maximum_channels = std::min(padded_binding_limit / row_bytes,
                                             binding_limit / output_channel_bytes);
        maximum_channels = std::min(maximum_channels, available_for_chunks / bytes_per_channel);
        // Derive tile size from live headroom rather than a fixed model/GPU byte
        // constant. Bound each staging tile to a fraction of available memory.
        const uint64_t target_tile_bytes = std::max<uint64_t>(
            bytes_per_channel, available_for_chunks / 4);
        const uint64_t preferred_channels = std::max<uint64_t>(
            1, target_tile_bytes / bytes_per_channel);
        maximum_channels = std::min(maximum_channels, preferred_channels);
        maximum_channels = std::min<uint64_t>(maximum_channels, UINT32_MAX);
        if (maximum_channels == 0) {
            if (pending.empty())
                return {StatusCode::OutOfMemory,
                    "GPU budget cannot hold one quantized MatMul output-channel chunk: " +
                    operation.id + " weight=" + weight_name +
                    " required_min_bytes=" + std::to_string(bytes_per_channel + 3) +
                    " available_bytes=" + std::to_string(available) +
                    " reclaimed_optional_memory_bytes=" + std::to_string(reclaimed_optional_bytes) +
                    " rows=" + std::to_string(rows) +
                    " inner=" + std::to_string(inner) +
                    " weight_row_bytes=" + std::to_string(row_bytes) +
                    " output_channel_bytes=" + std::to_string(output_channel_bytes) +
                    " quant_type=" + type.name +
                    " chunk_cache_resident_bytes=" + std::to_string(context.weight_chunk_cache ? context.weight_chunk_cache->stats().resident_bytes : 0) +
                    " budget_bytes=" + std::to_string(memory.budget_bytes())};
            Status status = flush_batch();
            if (!status.ok()) return status;
            continue;
        }

        const uint64_t chunk_channels = std::min(maximum_channels, total_channels - channel_offset);
        uint64_t source_offset = 0, weight_bytes = 0, output_chunk_bytes = 0;
        if (!checked_product(channel_offset, row_bytes, source_offset) ||
            !checked_product(chunk_channels, row_bytes, weight_bytes) ||
            !checked_product(chunk_channels, output_channel_bytes, output_chunk_bytes) ||
            source_offset > source->byte_size || weight_bytes > source->byte_size - source_offset)
            return {StatusCode::InvalidArgument, "Quantized MatMul chunk byte range overflows GGUF tensor bounds"};

        PendingChunk chunk;
        chunk.channel_offset = channel_offset;
        chunk.chunk_channels = chunk_channels;
        const std::string chunk_cache_key = weight_name + "|" +
            std::to_string(source->type_id) + "|" + std::to_string(source_offset) + "|" +
            std::to_string(weight_bytes);
        bool chunk_cache_hit = context.weight_chunk_cache &&
            context.weight_chunk_cache->find(chunk_cache_key, chunk.weight_lease);
        Status status = Status::success();
        if (!chunk_cache_hit) {
            status = allocate_chunk_buffer(memory, device, weight_bytes,
                "quantized-matmul-weight-chunk:" + weight_name, chunk.weight_lease);
            if (!status.ok() && status.code == StatusCode::OutOfMemory) {
                reclaim_optional_memory();
                status = allocate_chunk_buffer(memory, device, weight_bytes,
                    "quantized-matmul-weight-chunk:" + weight_name, chunk.weight_lease);
            }
            if (!status.ok()) {
                if (pending.empty()) return status;
                status = flush_batch();
                if (!status.ok()) return status;
                continue;
            }
        }
        status = allocate_chunk_buffer(memory, device, output_chunk_bytes,
            "quantized-matmul-output-chunk:" + operation.id, chunk.output_lease);
        if (!status.ok() && status.code == StatusCode::OutOfMemory) {
            reclaim_optional_memory();
            status = allocate_chunk_buffer(memory, device, output_chunk_bytes,
                "quantized-matmul-output-chunk:" + operation.id, chunk.output_lease);
        }
        if (!status.ok()) {
            if (pending.empty()) return status;
            status = flush_batch();
            if (!status.ok()) return status;
            continue;
        }

        std::vector<uint8_t> staging;
        if (!chunk_cache_hit) {
            staging.resize(static_cast<size_t>((weight_bytes + 3) & ~uint64_t{3}), 0);
            status = reader.read_tensor_chunk(*source, source_offset, staging.data(), weight_bytes);
            if (!status.ok()) return status;
            status = backend.upload(chunk.weight_lease.buffer(), 0, staging.data(), staging.size());
            if (!status.ok()) return status;
            // Admission is opportunistic: keep a safety margin for subsequent
            // compute allocations instead of pinning every streamed slice.
            const uint64_t cache_headroom = memory.available_bytes();
            const uint64_t cache_reserve = std::max<uint64_t>(
                64ull * 1024ull * 1024ull, memory.budget_bytes() / 8);
            if (context.weight_chunk_cache &&
                cache_headroom >= std::max<uint64_t>(weight_bytes, cache_reserve))
                context.weight_chunk_cache->insert(chunk_cache_key, weight_bytes, chunk.weight_lease);
        }
        if (validate_q6_k) {
            const uint64_t sample_weight_bytes =
                static_cast<uint64_t>(std::min<uint64_t>(4, chunk_channels)) * row_bytes;
            if (chunk_cache_hit) {
                chunk.sample_weights.resize(static_cast<size_t>(sample_weight_bytes));
                status = reader.read_tensor_chunk(*source, source_offset,
                    chunk.sample_weights.data(), sample_weight_bytes);
                if (!status.ok()) return status;
            } else {
                chunk.sample_weights.assign(staging.begin(),
                    staging.begin() + static_cast<std::ptrdiff_t>(sample_weight_bytes));
            }
        }

        TensorDesc output_descriptor;
        output_descriptor.shape = {rows, chunk_channels};
        output_descriptor.data_type = DataType::F32;
        output_descriptor.layout = TensorLayout::Contiguous;
        output_descriptor.memory = MemoryDomain::Device;
        output_descriptor.alignment = 4;
        Tensor output_chunk;
        status = Tensor::create(output_descriptor, chunk.output_lease.buffer(), 0, output_chunk);
        if (!status.ok()) return status;

        ModelExecutionContext chunk_context = context;
        // Canonicalize contiguous activations to [M, K]. This supports rank-2,
        // rank-3 and rank-4 model-plan tensors without changing their byte order.
        TensorDesc activation_view_desc = input.descriptor();
        activation_view_desc.shape = {rows, inner};
        activation_view_desc.strides_bytes.clear();
        Tensor activation_view;
        status = Tensor::create(std::move(activation_view_desc), input.buffer(),
            input.byte_offset(), activation_view);
        if (!status.ok()) return status;
        chunk_context.tensors[operation.inputs.front()] = std::move(activation_view);

        QuantizedWeightBinding chunk_weight = weight_it->second;
        chunk_weight.buffer = chunk.weight_lease.buffer();
        chunk_weight.byte_size = weight_bytes;
        chunk_weight.dimensions = {inner, chunk_channels};
        chunk_weight.chunks.clear();
        chunk_context.quantized_weights[weight_name] = std::move(chunk_weight);
        chunk_context.tensors[operation.output] = std::move(output_chunk);

        ComputeDispatch dispatch;
        status = (*lowering)(operation, chunk_context, dispatch);
        if (!status.ok()) return status;
        status = operators.authorize_dispatch(dispatch.kernel_id, backend.kind(), dispatch);
        if (!status.ok()) return status;
        status = verify_chunk_output_binding(dispatch, backend.kind(), chunk.output_lease.buffer(),
                                             output_chunk_bytes);
        if (!status.ok()) return status;

        if (!commands) {
            status = backend.begin_commands(commands);
            if (!status.ok()) return status;
            if (!commands)
                return {StatusCode::BackendFailure, "Backend returned an empty quantized MatMul command batch"};
        }
        status = commands->dispatch_compute(dispatch);
        if (!status.ok()) return status;

        // Queue output aggregation immediately after its producer dispatch.
        // Both commands execute in-order on the same backend command stream.
        for (uint64_t row = 0; row < rows; ++row) {
            uint64_t source_row_offset = 0, destination_row_offset = 0, channel_byte_offset = 0;
            if (!checked_product(row, chunk_channels * sizeof(float), source_row_offset) ||
                !checked_product(row, total_channels * sizeof(float), destination_row_offset) ||
                !checked_product(chunk.channel_offset, sizeof(float), channel_byte_offset) ||
                output.byte_offset() > UINT64_MAX - destination_row_offset ||
                output.byte_offset() + destination_row_offset > UINT64_MAX - channel_byte_offset)
                return {StatusCode::InvalidArgument, "Quantized MatMul output aggregation offset overflows"};
            destination_row_offset += output.byte_offset() + channel_byte_offset;
            status = commands->copy_buffer(chunk.output_lease.buffer(), source_row_offset,
                output.buffer(), destination_row_offset, chunk_channels * sizeof(float));
            if (!status.ok()) return status;
        }

        pending.push_back(std::move(chunk));
        channel_offset += chunk_channels;
    }

    return flush_batch();
}

Status execute_chunked_quantized_embedding_if_needed(
        const ModelPlanOperation& operation, Backend& backend,
        const ModelLoweringRegistry& lowerings, const OperatorRegistry& operators,
        const GgufReader& reader, MemoryManager& memory, const DeviceInfo& device,
        ModelExecutionContext& context, bool& handled, uint64_t* real_q6_k_parity_checks) {
    handled = false;
    if (operation.operator_name.rfind("Embedding_", 0) != 0 ||
        operation.inputs.size() != 1 || operation.inputs.front() != "token_ids" ||
        operation.weight_tensors.size() != 1)
        return Status::success();

    const std::string& weight_name = operation.weight_tensors.front();
    const auto weight_it = context.quantized_weights.find(weight_name);
    if (weight_it == context.quantized_weights.end() || weight_it->second.buffer)
        return Status::success();
    handled = true;

    const GgufTensorInfo* source = reader.find_tensor(weight_name);
    const auto output_it = context.tensors.find(operation.output);
    if (!source || !source->byte_size_known || output_it == context.tensors.end())
        return {StatusCode::InvalidArgument, "Chunked Embedding requires GGUF weight metadata and a bound output"};
    const Tensor& output = output_it->second;
    const auto& dimensions = weight_it->second.dimensions;
    const auto& type = source->type;
    if (dimensions.size() != 2 || source->dimensions != dimensions ||
        source->type_id != weight_it->second.file_type_id ||
        !type.storage_size_known || type.block_elements == 0 || type.bytes_per_block == 0 ||
        output.descriptor().data_type != DataType::F32 ||
        output.descriptor().shape.size() != 2 || !output.buffer() ||
        output.descriptor().shape[0] != context.token_ids.size() ||
        output.descriptor().shape[1] != dimensions[0] ||
        context.token_ids.empty() || context.token_ids.size() > UINT32_MAX)
        return {StatusCode::Unsupported, "Chunked Embedding GGUF dimensions, token IDs or output shape are incompatible"};

    const uint64_t dimension = dimensions[0];
    const uint64_t vocabulary = dimensions[1];
    if (dimension == 0 || vocabulary == 0 || dimension % type.block_elements != 0)
        return {StatusCode::Unsupported, "Embedding dimension is not aligned to the GGUF quantization block"};
    uint64_t row_bytes = 0, expected_bytes = 0, output_bytes_per_token = 0;
    if (!checked_product(dimension / type.block_elements, type.bytes_per_block, row_bytes) ||
        !checked_product(row_bytes, vocabulary, expected_bytes) ||
        !checked_product(dimension, sizeof(float), output_bytes_per_token) ||
        expected_bytes != source->byte_size || weight_it->second.byte_size != source->byte_size)
        return {StatusCode::InvalidArgument, "Embedding row-byte calculation does not match GGUF tensor storage"};

    const uint64_t binding_limit = storage_binding_limit(device);
    if (row_bytes == 0 || output_bytes_per_token == 0 ||
        row_bytes > binding_limit || output_bytes_per_token > binding_limit)
        return {StatusCode::Unsupported, "Backend binding limit cannot hold one gathered embedding row"};
    if (row_bytes > UINT64_MAX - output_bytes_per_token ||
        row_bytes + output_bytes_per_token > UINT64_MAX - sizeof(uint32_t))
        return {StatusCode::InvalidArgument, "Embedding per-token chunk memory estimate overflows"};
    const uint64_t bytes_per_token = row_bytes + output_bytes_per_token + sizeof(uint32_t);
    const uint64_t available = memory.available_bytes();
    const uint64_t padded_binding_limit = binding_limit > 3 ? binding_limit - 3 : 0;
    const uint64_t available_for_chunks = available > 3 ? available - 3 : 0;
    uint64_t maximum_tokens = std::min(padded_binding_limit / row_bytes,
                                       binding_limit / output_bytes_per_token);
    maximum_tokens = std::min(maximum_tokens, binding_limit / sizeof(uint32_t));
    maximum_tokens = std::min(maximum_tokens, available_for_chunks / bytes_per_token);
    maximum_tokens = std::min<uint64_t>(maximum_tokens, UINT32_MAX);
    if (maximum_tokens == 0)
        return {StatusCode::OutOfMemory, "GPU budget cannot hold one gathered embedding row and output"};

    const auto* lowering = lowerings.find(backend.kind(), operation.operator_name);
    if (!lowering)
        return {StatusCode::Unsupported, "No validated lowering exists for chunked Embedding: " + operation.operator_name};

    uint64_t token_offset = 0;
    while (token_offset < context.token_ids.size()) {
        if (!backend.healthy())
            return {StatusCode::DeviceLost, "GPU backend became unhealthy during embedding row gathering"};
        const uint64_t batch_tokens = std::min<uint64_t>(
            maximum_tokens, context.token_ids.size() - token_offset);
        uint64_t weight_bytes = 0, output_chunk_bytes = 0;
        if (!checked_product(batch_tokens, row_bytes, weight_bytes) ||
            !checked_product(batch_tokens, output_bytes_per_token, output_chunk_bytes))
            return {StatusCode::InvalidArgument, "Embedding gather batch size overflows"};

        MemoryLease weight_lease, output_lease;
        Status status = allocate_chunk_buffer(memory, device, weight_bytes,
            "quantized-embedding-gather:" + weight_name, weight_lease);
        if (!status.ok()) return status;
        status = allocate_chunk_buffer(memory, device, output_chunk_bytes,
            "quantized-embedding-output:" + operation.id, output_lease);
        if (!status.ok()) return status;

        std::vector<uint8_t> staging(static_cast<size_t>((weight_bytes + 3) & ~uint64_t{3}), 0);
        std::vector<uint32_t> remapped_ids(static_cast<size_t>(batch_tokens));
        for (uint64_t index = 0; index < batch_tokens; ++index) {
            const uint32_t token_id = context.token_ids[static_cast<size_t>(token_offset + index)];
            if (token_id >= vocabulary)
                return {StatusCode::InvalidArgument, "Embedding token ID is outside the GGUF vocabulary"};
            const uint64_t source_offset = static_cast<uint64_t>(token_id) * row_bytes;
            status = reader.read_tensor_chunk(*source, source_offset,
                staging.data() + index * row_bytes, row_bytes);
            if (!status.ok()) return status;
            remapped_ids[static_cast<size_t>(index)] = static_cast<uint32_t>(index);
        }
        status = backend.upload(weight_lease.buffer(), 0, staging.data(), staging.size());
        if (!status.ok()) return status;
        TensorDesc output_descriptor;
        output_descriptor.shape = {batch_tokens, dimension};
        output_descriptor.data_type = DataType::F32;
        output_descriptor.layout = TensorLayout::Contiguous;
        output_descriptor.memory = MemoryDomain::Device;
        output_descriptor.alignment = 4;
        Tensor output_chunk;
        status = Tensor::create(output_descriptor, output_lease.buffer(), 0, output_chunk);
        if (!status.ok()) return status;

        ModelExecutionContext chunk_context = context;
        chunk_context.token_ids = std::move(remapped_ids);
        QuantizedWeightBinding chunk_weight = weight_it->second;
        chunk_weight.buffer = weight_lease.buffer();
        chunk_weight.byte_size = weight_bytes;
        chunk_weight.dimensions = {dimension, batch_tokens};
        chunk_weight.chunks.clear();
        chunk_context.quantized_weights[weight_name] = std::move(chunk_weight);
        chunk_context.tensors[operation.output] = std::move(output_chunk);
        // Embedding lowerings create their own token-ID buffer from token_ids.
        // The gathered rows are indexed from zero, so no source vocabulary IDs
        // are exposed to the backend kernel.

        ComputeDispatch dispatch;
        status = (*lowering)(operation, chunk_context, dispatch);
        if (!status.ok()) return status;
        status = operators.authorize_dispatch(dispatch.kernel_id, backend.kind(), dispatch);
        if (!status.ok()) return status;
        status = verify_chunk_output_binding(dispatch, backend.kind(), output_lease.buffer(),
                                             output_chunk_bytes);
        if (!status.ok()) return status;
        std::unique_ptr<CommandBatch> compute_commands;
        status = backend.begin_commands(compute_commands);
        if (!status.ok()) return status;
        if (!compute_commands)
            return {StatusCode::BackendFailure, "Backend returned an empty embedding command batch"};
        status = compute_commands->dispatch_compute(dispatch);
        if (!status.ok()) return status;
        status = backend.submit(std::move(compute_commands));
        if (!status.ok()) return status;
        status = backend.synchronize();
        if (!status.ok()) return status;

        // Validate real prompt-token embedding rows against the independent
        // Q6_K decoder before allowing the model graph to consume them.
        if (context.prefill && real_q6_k_parity_checks != nullptr &&
            source->type.name == "Q6_K") {
            std::vector<uint32_t> reference_ids(static_cast<size_t>(batch_tokens));
            for (uint32_t index = 0; index < reference_ids.size(); ++index)
                reference_ids[index] = index;
            std::vector<float> expected(static_cast<size_t>(batch_tokens * dimension));
            std::vector<float> actual(expected.size());
            status = reference_q6_k_embedding(staging.data(), weight_bytes,
                static_cast<uint32_t>(batch_tokens), static_cast<uint32_t>(dimension),
                reference_ids, expected.data());
            if (!status.ok()) return status;
            status = backend.readback(output_lease.buffer(), 0, actual.data(),
                actual.size() * sizeof(float));
            if (!status.ok()) return status;
            for (size_t index = 0; index < actual.size(); ++index) {
                const float tolerance = 2.0e-4f +
                    2.0e-4f * std::max(std::fabs(expected[index]), 1.0f);
                if (!std::isfinite(actual[index]) ||
                    std::fabs(actual[index] - expected[index]) > tolerance) {
                    return {StatusCode::NumericalFailure,
                        "Real GGUF Q6_K Embedding parity failed for " + operation.id +
                        " at token_offset=" + std::to_string(token_offset) +
                        " element=" + std::to_string(index)};
                }
            }
            ++(*real_q6_k_parity_checks);
        }

        std::unique_ptr<CommandBatch> copy_commands;
        status = backend.begin_commands(copy_commands);
        if (!status.ok()) return status;
        if (!copy_commands)
            return {StatusCode::BackendFailure, "Backend returned an empty embedding aggregation batch"};
        uint64_t destination_offset = 0;
        if (!checked_product(token_offset, output_bytes_per_token, destination_offset) ||
            output.byte_offset() > UINT64_MAX - destination_offset)
            return {StatusCode::InvalidArgument, "Embedding output aggregation offset overflows"};
        status = copy_commands->copy_buffer(output_lease.buffer(), 0,
            output.buffer(), output.byte_offset() + destination_offset, output_chunk_bytes);
        if (!status.ok()) return status;
        status = backend.submit(std::move(copy_commands));
        if (!status.ok()) return status;
        status = backend.synchronize();
        if (!status.ok()) return status;
        token_offset += batch_tokens;
    }
    return Status::success();
}


bool is_chunkable_elementwise(const ModelPlanOperation& operation) {
    if (!operation.weight_tensors.empty() || operation.side_effect) return false;
    const auto& name = operation.operator_name;
    return name == "SiLU" || name == "Sigmoid" || name == "Softplus" ||
           name == "Add" || name == "Multiply";
}

Status execute_chunked_elementwise_if_needed(
        const ModelPlanOperation& operation, Backend& backend,
        const ModelLoweringRegistry& lowerings, const OperatorRegistry& operators,
        const DeviceInfo& device, ModelExecutionContext& context, bool& handled) {
    handled = false;
    if (!is_chunkable_elementwise(operation)) return Status::success();
    const size_t expected_inputs =
        (operation.operator_name == "Add" || operation.operator_name == "Multiply") ? 2u : 1u;
    if (operation.inputs.size() != expected_inputs)
        return {StatusCode::InvalidArgument, "Chunkable elementwise operation has an unexpected input count"};
    const auto output_it = context.tensors.find(operation.output);
    if (output_it == context.tensors.end())
        return {StatusCode::InvalidArgument, "Chunkable elementwise output binding is missing"};
    const Tensor& output = output_it->second;
    if (output.descriptor().data_type != DataType::F32 ||
        output.descriptor().layout != TensorLayout::Contiguous || !output.buffer())
        return {StatusCode::Unsupported, "Chunked elementwise execution currently requires contiguous F32 output"};

    std::vector<const Tensor*> inputs;
    inputs.reserve(expected_inputs);
    bool oversized = output.byte_size() > device.limits.max_storage_buffer_binding_size;
    for (const auto& name : operation.inputs) {
        const auto input_it = context.tensors.find(name);
        if (input_it == context.tensors.end())
            return {StatusCode::InvalidArgument, "Chunkable elementwise input binding is missing: " + name};
        const Tensor& input = input_it->second;
        if (input.descriptor().data_type != DataType::F32 ||
            input.descriptor().layout != TensorLayout::Contiguous || !input.buffer() ||
            input.element_count() != output.element_count() ||
            input.descriptor().shape != output.descriptor().shape)
            return {StatusCode::Unsupported, "Chunked elementwise inputs must match the contiguous F32 output shape"};
        oversized = oversized || input.byte_size() > device.limits.max_storage_buffer_binding_size;
        inputs.push_back(&input);
    }
    if (!oversized) return Status::success();
    handled = true;

    const uint64_t binding_limit = device.limits.max_storage_buffer_binding_size;
    if (binding_limit < sizeof(float))
        return {StatusCode::Unsupported, "Backend storage binding limit is smaller than one F32 element"};
    const uint64_t alignment_bytes = std::max<uint64_t>(4, device.limits.min_storage_buffer_offset_alignment);
    const uint64_t alignment_elements = (alignment_bytes + sizeof(float) - 1) / sizeof(float);
    const uint64_t maximum_chunk_elements = binding_limit / sizeof(float);
    if (maximum_chunk_elements < alignment_elements)
        return {StatusCode::Unsupported, "Backend binding limit cannot hold one aligned F32 chunk"};

    const auto* lowering = lowerings.find(backend.kind(), operation.operator_name);
    if (!lowering)
        return {StatusCode::Unsupported, "No backend lowering exists for chunked elementwise operation: " + operation.operator_name};

    ChunkExecutionConfig config;
    config.total_elements = output.element_count();
    config.maximum_chunk_elements = maximum_chunk_elements;
    config.alignment_elements = alignment_elements;
    config.element_size_bytes = sizeof(float);
    config.output_base_offset_bytes = output.byte_offset();
    config.output_binding = expected_inputs == 1 ? 1u : 2u;
    config.output_buffer = output.buffer();

    const auto build_dispatch = [&](const ChunkExecutionRange& range, ComputeDispatch& dispatch) -> Status {
        if (range.byte_offset > UINT64_MAX - output.byte_offset())
            return {StatusCode::InvalidArgument, "Chunked output byte offset overflows"};
        ModelExecutionContext chunk_context = context;
        for (size_t index = 0; index < inputs.size(); ++index) {
            const Tensor& source = *inputs[index];
            if (range.byte_offset > UINT64_MAX - source.byte_offset())
                return {StatusCode::InvalidArgument, "Chunked input byte offset overflows"};
            TensorDesc descriptor = source.descriptor();
            descriptor.shape = {range.element_count};
            descriptor.strides_bytes.clear();
            Tensor view;
            Status result = Tensor::create(std::move(descriptor), source.buffer(),
                source.byte_offset() + range.byte_offset, view);
            if (!result.ok()) return result;
            chunk_context.tensors[operation.inputs[index]] = std::move(view);
        }
        TensorDesc output_descriptor = output.descriptor();
        output_descriptor.shape = {range.element_count};
        output_descriptor.strides_bytes.clear();
        Tensor output_view;
        Status result = Tensor::create(std::move(output_descriptor), output.buffer(),
            output.byte_offset() + range.byte_offset, output_view);
        if (!result.ok()) return result;
        chunk_context.tensors[operation.output] = std::move(output_view);

        result = (*lowering)(operation, chunk_context, dispatch);
        if (!result.ok()) return result;
        return operators.authorize_dispatch(dispatch.kernel_id, backend.kind(), dispatch);
    };
    return execute_chunked_compute(backend, config, build_dispatch, {});
}

} // namespace

std::string ModelLoweringRegistry::make_key(BackendKind backend, const std::string& name) {
    return std::to_string(static_cast<unsigned>(backend)) + ":" + name;
}

Status ModelLoweringRegistry::register_lowering(BackendKind backend,
                                                const std::string& operator_name,
                                                Lowering lowering) {
    if (backend == BackendKind::Unknown || operator_name.empty() || !lowering)
        return {StatusCode::InvalidArgument, "Model lowering requires a backend, operator name and callback"};
    const std::string key = make_key(backend, operator_name);
    if (lowerings_.find(key) != lowerings_.end())
        return {StatusCode::InvalidArgument, "Model lowering is already registered for this backend/operator"};
    lowerings_.emplace(key, std::move(lowering));
    return Status::success();
}

const ModelLoweringRegistry::Lowering* ModelLoweringRegistry::find(
        BackendKind backend, const std::string& operator_name) const {
    const auto found = lowerings_.find(make_key(backend, operator_name));
    return found == lowerings_.end() ? nullptr : &found->second;
}

Status compile_model_execution_plan(ModelExecutionPlan& plan, BackendKind backend,
                                    const ModelLoweringRegistry& lowerings,
                                    const OperatorRegistry& operators,
                                    const std::vector<std::string>& external_inputs,
                                    const ModelExecutionContext& context,
                                    ComputeGraph& graph, ModelGraphCompileReport& report) {
    report = ModelGraphCompileReport{};
    plan.graph_compiled = false;
    if (!plan.structurally_validated || !plan.operator_coverage_validated)
        return {StatusCode::NotReady, "Model graph compilation requires structural and operator-coverage validation"};
    if (backend == BackendKind::Unknown || graph.node_count() != 0)
        return {StatusCode::InvalidArgument, "Model graph compilation requires a backend and an empty destination graph"};

    const std::unordered_set<std::string> external(external_inputs.begin(), external_inputs.end());
    std::unordered_map<std::string, size_t> producers;
    std::unordered_set<std::string> operation_ids;
    ComputeGraph candidate;

    for (const auto& operation : plan.operations) {
        if (operation.id.empty() || operation.operator_name.empty() || operation.output.empty()) {
            report.errors.emplace_back("MODEL_OPERATION_HAS_EMPTY_ID_OPERATOR_OR_OUTPUT");
            continue;
        }
        if (!operation_ids.insert(operation.id).second) {
            report.errors.emplace_back("DUPLICATE_MODEL_OPERATION_ID:" + operation.id);
            continue;
        }
        bool inputs_ready = true;
        std::vector<size_t> dependencies;
        for (const auto& input : operation.inputs) {
            const auto producer = producers.find(input);
            if (producer != producers.end()) {
                if (std::find(dependencies.begin(), dependencies.end(), producer->second) == dependencies.end())
                    dependencies.push_back(producer->second);
            } else if (external.find(input) == external.end()) {
                report.unresolved_inputs.push_back(operation.id + ":" + input);
                inputs_ready = false;
            }
            const bool token_ids_external = input == "token_ids" &&
                external.find(input) != external.end() && !context.token_ids.empty();
            if (context.tensors.find(input) == context.tensors.end() && !token_ids_external) {
                report.unresolved_inputs.push_back(operation.id + ":UNBOUND:" + input);
                inputs_ready = false;
            }
        }
        for (const auto& control_input : operation.control_inputs) {
            const auto producer = producers.find(control_input);
            if (producer == producers.end()) {
                report.unresolved_inputs.push_back(operation.id + ":CONTROL:" + control_input);
                inputs_ready = false;
            } else if (std::find(dependencies.begin(), dependencies.end(), producer->second) == dependencies.end()) {
                dependencies.push_back(producer->second);
            }
        }
        for (const auto& weight : operation.weight_tensors) {
            const bool dense_bound = context.tensors.find(weight) != context.tensors.end();
            const bool quantized_bound = context.quantized_weights.find(weight) != context.quantized_weights.end();
            if (!dense_bound && !quantized_bound) {
                report.missing_weight_tensors.push_back(operation.id + ":" + weight);
                inputs_ready = false;
            }
        }
        if (operation.side_effect) {
            if (operation.output.empty()) {
                report.errors.push_back(operation.id + ":SIDE_EFFECT_OUTPUT_ID_REQUIRED");
                inputs_ready = false;
            }
        } else if (context.tensors.find(operation.output) == context.tensors.end()) {
            report.errors.push_back(operation.id + ":UNBOUND_OUTPUT:" + operation.output);
            inputs_ready = false;
        }
        if (!inputs_ready) continue;

        const auto* lowering = lowerings.find(backend, operation.operator_name);
        if (!lowering) {
            report.missing_lowerings.push_back(operation.operator_name);
            continue;
        }
        ComputeDispatch dispatch;
        Status status = (*lowering)(operation, context, dispatch);
        if (!status.ok()) {
            report.errors.push_back(operation.id + ":" + status.message);
            continue;
        }
        if (dispatch.backend != backend || (!dispatch.copy_only && dispatch.kernel_id.empty())) {
            report.errors.push_back(operation.id + ":LOWERING_RETURNED_WRONG_BACKEND_OR_EMPTY_KERNEL_ID");
            continue;
        }
        if (!dispatch.copy_only) {
            status = operators.authorize_dispatch(dispatch.kernel_id, backend, dispatch);
            if (!status.ok()) {
                report.errors.push_back(operation.id + ":" + status.message);
                continue;
            }
        }
        GraphNode node;
        node.name = operation.id;
        node.dependencies = std::move(dependencies);
        node.dispatch = std::move(dispatch);
        size_t index = 0;
        status = candidate.add_node(std::move(node), operators, index);
        if (!status.ok()) {
            report.errors.push_back(operation.id + ":" + status.message);
            continue;
        }
        producers[operation.output] = index;
    }

    if (!report.errors.empty() || !report.missing_lowerings.empty() ||
        !report.unresolved_inputs.empty() || !report.missing_weight_tensors.empty() ||
        candidate.node_count() != plan.operations.size()) {
        report.compiled_nodes = candidate.node_count();
        return {StatusCode::Unsupported, describe_compile_failure(report)};
    }
    graph = std::move(candidate);
    report.compiled = true;
    report.compiled_nodes = graph.node_count();
    plan.graph_compiled = true;
    plan.gpu_ready = false; // Graph compilation is not execution proof.
    return Status::success();
}


Status execute_model_execution_plan(ModelExecutionPlan& plan, Backend& backend,
                                    const ModelLoweringRegistry& lowerings,
                                    const OperatorRegistry& operators,
                                    const std::vector<std::string>& external_inputs,
                                    const ModelExecutionContext& context,
                                    const std::atomic<bool>* cancellation,
                                    ModelGraphCompileReport& report) {
    report = ModelGraphCompileReport{};
    plan.execution_validated = false;
    plan.gpu_ready = false;
    ComputeGraph graph;
    Status status = compile_model_execution_plan(plan, backend.kind(), lowerings,
                                                 operators, external_inputs, context,
                                                 graph, report);
    if (!status.ok()) return status;
    ExecutionScheduler scheduler;
    status = scheduler.execute(backend, graph, cancellation);
    if (!status.ok()) {
        report.errors.push_back("GRAPH_EXECUTION_FAILED:" + status.message);
        return status;
    }
    for (const auto& operation : plan.operations) {
        if (operation.operator_name != "KVCacheAppend") continue;
        if (!context.kv_cache) {
            report.errors.emplace_back("KV_CACHE_COMMIT_CONTEXT_MISSING");
            return {StatusCode::InvalidArgument, "KV cache context disappeared before commit"};
        }
        const auto layer_attr = operation.integer_attributes.find("layer_index");
        if (layer_attr == operation.integer_attributes.end() || layer_attr->second > UINT32_MAX ||
            operation.inputs.size() != 2) {
            report.errors.push_back(operation.id + ":KV_CACHE_COMMIT_METADATA_INVALID");
            return {StatusCode::InvalidArgument, "KV cache append commit metadata is invalid"};
        }
        const auto keys = context.tensors.find(operation.inputs[0]);
        if (keys == context.tensors.end() || keys->second.descriptor().shape.empty() ||
            keys->second.descriptor().shape[0] > UINT32_MAX) {
            report.errors.push_back(operation.id + ":KV_CACHE_COMMIT_TENSOR_MISSING");
            return {StatusCode::InvalidArgument, "KV cache append source tensor is missing at commit"};
        }
        status = context.kv_cache->commit_device_append(
            static_cast<uint32_t>(layer_attr->second),
            static_cast<uint32_t>(context.past_length),
            static_cast<uint32_t>(keys->second.descriptor().shape[0]));
        if (!status.ok()) {
            report.errors.push_back(operation.id + ":KV_CACHE_COMMIT_FAILED:" + status.message);
            return status;
        }
    }
    report.executed = true;
    plan.execution_validated = true;
    // One successful graph run is not a model-family or quantization certification.
    // The caller must still perform model-level parity and lifecycle qualification.
    plan.gpu_ready = false;
    return Status::success();
}



Status execute_model_execution_plan_streamed(ModelExecutionPlan& plan, Backend& backend,
                                    const ModelLoweringRegistry& lowerings,
                                    const OperatorRegistry& operators,
                                    const std::vector<std::string>& external_inputs,
                                    const GgufReader& reader, MemoryManager& memory,
                                    const DeviceInfo& device, uint64_t token_count,
                                    const std::atomic<bool>* cancellation,
                                    ModelExecutionContext& context,
                                    ModelGraphCompileReport& report) {
    report = ModelGraphCompileReport{};
    plan.graph_compiled = false;
    plan.execution_validated = false;
    plan.gpu_ready = false;
    if (!plan.structurally_validated || !plan.operator_coverage_validated ||
        plan.operations.empty() || backend.kind() == BackendKind::Unknown ||
        !backend.healthy() || token_count == 0)
        return {StatusCode::NotReady, "Streamed execution requires a non-empty validated plan, healthy backend and non-zero token count"};
    context.backend = &backend;
    if (context.token_ids.size() != token_count)
        return {StatusCode::InvalidArgument, "Streamed execution token count does not match the bound token IDs"};

    std::unordered_map<std::string, size_t> producer_indices;
    for (size_t index = 0; index < plan.operations.size(); ++index) {
        const auto& operation = plan.operations[index];
        if (operation.output.empty() || !producer_indices.emplace(operation.output, index).second)
            return {StatusCode::InvalidArgument, "Streamed model plan has an empty or duplicate output name"};
    }
    for (size_t index = 0; index < plan.operations.size(); ++index) {
        const auto& operation = plan.operations[index];
        for (const auto& input : operation.inputs) {
            const auto producer = producer_indices.find(input);
            if (producer != producer_indices.end() && producer->second >= index)
                return {StatusCode::InvalidArgument, "Streamed model plan is not in dependency order: " + operation.id};
        }
        for (const auto& control : operation.control_inputs) {
            const auto producer = producer_indices.find(control);
            if (producer == producer_indices.end() || producer->second >= index)
                return {StatusCode::InvalidArgument, "Streamed model plan has an unresolved or forward control dependency: " + operation.id};
        }
    }

    Status status = allocate_model_activation_tensors(plan, memory, token_count, context);
    if (!status.ok()) return status;
    const bool has_attention = std::any_of(plan.operations.begin(), plan.operations.end(),
        [](const ModelPlanOperation& operation) { return operation.operator_name == "Attention"; });
    const bool has_kv_append = std::any_of(plan.operations.begin(), plan.operations.end(),
        [](const ModelPlanOperation& operation) { return operation.operator_name == "KVCacheAppend"; });
    if ((has_attention || has_kv_append) &&
        (!context.kv_cache || context.sequence_length != token_count))
        return {StatusCode::NotReady, "Streamed attention execution requires a KV cache and a sequence length matching the invocation"};
    if (has_attention) {
        status = bind_model_kv_cache_views(plan, *context.kv_cache, context.past_length,
                                           context.sequence_length, context);
        if (!status.ok()) return status;
    }
    // The session may carry a bounded set of resident quantized MatMul weights
    // across token steps. All other weights remain streamed on demand.
    // Real GGUF parity validation performs GPU readbacks and CPU reference work;
    // keep it out of the normal hot path unless diagnostics explicitly request it.
    uint64_t* parity_counter = context.enable_runtime_quantization_parity_validation
        ? &report.real_q6_k_parity_checks : nullptr;

    ExecutionScheduler scheduler;
    std::vector<ComputeGraph> pending_graphs;
    const auto flush_pending_graphs = [&]() -> Status {
        if (pending_graphs.empty()) return Status::success();
        Status flush_status = scheduler.execute_sequence(backend, pending_graphs, cancellation);
        if (flush_status.ok()) pending_graphs.clear();
        return flush_status;
    };
    const uint64_t graph_binding_limit = storage_binding_limit(device);
    std::unordered_set<std::string> plan_weight_names;
    for (const auto& planned_operation : plan.operations)
        for (const auto& name : planned_operation.weight_tensors)
            plan_weight_names.insert(name);

    for (const auto& original_operation : plan.operations) {
        if (cancellation && cancellation->load(std::memory_order_relaxed)) {
            report.errors.emplace_back("STREAMED_EXECUTION_CANCELLED");
            return {StatusCode::Cancelled, "Streamed model execution was cancelled"};
        }
        if (!backend.healthy()) {
            report.errors.emplace_back("STREAMED_BACKEND_DEVICE_LOST");
            return {StatusCode::DeviceLost, "GPU backend became unavailable during streamed execution"};
        }

        ModelPlanOperation operation = original_operation;
        // Operations are executed serially in the already-validated plan order;
        // their control dependencies are therefore satisfied by prior submits.
        operation.control_inputs.clear();
        ModelExecutionPlan single;
        single.structurally_validated = true;
        single.operator_coverage_validated = true;
        single.external_inputs = external_inputs;
        single.operations.push_back(operation);

        for (const auto& name : operation.weight_tensors) {
            const auto cached = context.quantized_weights.find(name);
            if (cached != context.quantized_weights.end() && cached->second.buffer)
                ++context.persistent_weight_cache_hits;
            else
                ++context.persistent_weight_cache_misses;
        }

        // If the next streamed MatMul cannot fit even its smallest legal tile,
        // spill optional whole-tensor weights before copying them into the
        // operation context. This is a session-level mode: repeatedly rebuilding
        // the same oversized resident cache would otherwise recreate the OOM.
        if (!context.weight_cache_pressure_mode &&
            operation.operator_name.rfind("MatMul_", 0) == 0 &&
            operation.inputs.size() == 1 && operation.weight_tensors.size() == 1) {
            const std::string& streamed_weight_name = operation.weight_tensors.front();
            const auto weight_binding = context.quantized_weights.find(streamed_weight_name);
            const auto activation_binding = context.tensors.find(operation.inputs.front());
            const GgufTensorInfo* weight_info = reader.find_tensor(streamed_weight_name);
            const bool target_weight_not_resident = weight_binding == context.quantized_weights.end() ||
                !weight_binding->second.buffer;
            if (target_weight_not_resident && !context.quantized_weights.empty() &&
                activation_binding != context.tensors.end() && weight_info &&
                weight_info->dimensions.size() == 2 && weight_info->type.storage_size_known &&
                weight_info->type.block_elements != 0 && weight_info->type.bytes_per_block != 0 &&
                weight_info->dimensions[0] % weight_info->type.block_elements == 0) {
                const uint64_t inner_width = weight_info->dimensions[0];
                const uint64_t row_count = inner_width != 0
                    ? activation_binding->second.element_count() / inner_width : 0;
                uint64_t weight_row_bytes = 0, output_channel_bytes = 0, minimum_tile_bytes = 0;
                if (inner_width != 0 && activation_binding->second.element_count() % inner_width == 0 &&
                    checked_product(inner_width / weight_info->type.block_elements,
                                    weight_info->type.bytes_per_block, weight_row_bytes) &&
                    checked_product(row_count, sizeof(float), output_channel_bytes) &&
                    weight_row_bytes <= UINT64_MAX - output_channel_bytes &&
                    weight_row_bytes + output_channel_bytes <= UINT64_MAX - 3) {
                    minimum_tile_bytes = weight_row_bytes + output_channel_bytes + 3;
                    if (memory.available_bytes() < minimum_tile_bytes) {
                        status = flush_pending_graphs();
                        if (!status.ok()) return status;
                        for (auto& cached_weight : context.quantized_weights) {
                            cached_weight.second.buffer.reset();
                            cached_weight.second.chunks.clear();
                        }
                        for (const auto& cached_name : plan_weight_names)
                            context.tensors.erase(cached_name);
                        context.weight_leases.clear();
                        // Disable whole-tensor cache admission while retaining the
                        // explicit pressure-mode path to streamed MatMul weights.
                        context.persistent_weight_cache_limit_bytes = 0;
                        context.weight_cache_pressure_mode = true;
                        if (context.weight_chunk_cache)
                            context.weight_chunk_cache->evict_all();
                        memory.reclaim_transient_pool();
                    }
                }
            }
        }

        ModelExecutionContext operation_context;
        operation_context.tensors = context.tensors;
        operation_context.quantized_weights = context.quantized_weights;
        operation_context.weight_leases = context.weight_leases;
        operation_context.persistent_weight_cache_limit_bytes = context.persistent_weight_cache_limit_bytes;
        operation_context.weight_cache_pressure_mode = context.weight_cache_pressure_mode;
        // Share the session-owned LRU across per-operation contexts. Without
        // this assignment streamed MatMul silently bypasses the chunk cache.
        operation_context.weight_chunk_cache = context.weight_chunk_cache;
        operation_context.backend = &backend;
        operation_context.token_ids = context.token_ids;
        operation_context.kv_cache = context.kv_cache;
        operation_context.past_length = context.past_length;
        operation_context.sequence_length = context.sequence_length;
        operation_context.context_length = context.context_length;
        operation_context.prefill = context.prefill;

        // Only the final prompt position is consumed by the sampler. Slice the
        // final hidden-state row before output projection, avoiding [tokens,vocab]
        // logits storage during prefill.
        if (operation.output == "logits" && context.prefill &&
            context.token_ids.size() > 1 && !operation.inputs.empty()) {
            auto hidden = operation_context.tensors.find(operation.inputs.front());
            if (hidden == operation_context.tensors.end() ||
                hidden->second.descriptor().data_type != DataType::F32 ||
                hidden->second.descriptor().shape.size() != 2 ||
                hidden->second.descriptor().shape[0] != token_count) {
                report.errors.push_back(operation.id + ":LAST_TOKEN_LOGITS_INPUT_INVALID");
                return {StatusCode::InvalidArgument, "Final hidden-state tensor cannot be sliced for last-token logits"};
            }
            const Tensor& full_hidden = hidden->second;
            if (full_hidden.descriptor().shape[1] == 0 ||
                full_hidden.descriptor().shape[1] > UINT64_MAX / sizeof(float))
                return {StatusCode::InvalidArgument, "Final hidden-state row size overflowed"};
            const uint64_t row_bytes = full_hidden.descriptor().shape[1] * sizeof(float);
            if (token_count - 1 > UINT64_MAX / row_bytes)
                return {StatusCode::InvalidArgument, "Final hidden-state row offset overflowed"};
            const uint64_t row_offset = (token_count - 1) * row_bytes;
            if (full_hidden.byte_offset() > UINT64_MAX - row_offset)
                return {StatusCode::InvalidArgument, "Final hidden-state byte offset overflowed"};
            TensorDesc last_row_desc = full_hidden.descriptor();
            last_row_desc.shape[0] = 1;
            last_row_desc.strides_bytes.clear();
            Tensor last_hidden_row;
            status = Tensor::create(last_row_desc, full_hidden.buffer(),
                full_hidden.byte_offset() + row_offset, last_hidden_row);
            if (!status.ok()) return status;
            hidden->second = std::move(last_hidden_row);
        }

        if (!operation.weight_tensors.empty()) {
            status = load_model_execution_weights(reader, single, backend, memory,
                                                  device, operation_context);
            if (!status.ok()) {
                report.errors.push_back(operation.id + ":WEIGHT_STREAM_FAILED:" + status.message);
                for (const auto& name : operation.weight_tensors) {
                    operation_context.tensors.erase(name);
                    operation_context.quantized_weights.erase(name);
                }
                operation_context.weight_leases.clear();
                return status;
            }
        }

        const bool streamed_embedding_candidate =
            operation.operator_name.rfind("Embedding_", 0) == 0 &&
            !operation.weight_tensors.empty() &&
            operation_context.quantized_weights.count(operation.weight_tensors.front()) != 0 &&
            !operation_context.quantized_weights.at(operation.weight_tensors.front()).buffer;
        if (streamed_embedding_candidate) {
            status = flush_pending_graphs();
            if (!status.ok()) return status;
        }
        bool chunked_embedding = false;
        status = execute_chunked_quantized_embedding_if_needed(operation, backend, lowerings,
            operators, reader, memory, device, operation_context, chunked_embedding,
            parity_counter);
        if (!status.ok()) {
            report.errors.push_back(operation.id + ":CHUNKED_QUANTIZED_EMBEDDING_FAILED:" + status.message);
            for (const auto& name : operation.weight_tensors) {
                operation_context.tensors.erase(name);
                operation_context.quantized_weights.erase(name);
            }
            operation_context.weight_leases.clear();
            return status;
        }
        if (chunked_embedding) {
            ++report.compiled_nodes;
            for (const auto& name : operation.weight_tensors) {
                operation_context.tensors.erase(name);
                operation_context.quantized_weights.erase(name);
            }
            operation_context.weight_leases.clear();
            continue;
        }

        const bool streamed_matmul_candidate =
            operation.operator_name.rfind("MatMul_", 0) == 0 &&
            !operation.weight_tensors.empty() &&
            operation_context.quantized_weights.count(operation.weight_tensors.front()) != 0 &&
            !operation_context.quantized_weights.at(operation.weight_tensors.front()).buffer;
        if (streamed_matmul_candidate) {
            status = flush_pending_graphs();
            if (!status.ok()) return status;
        }
        bool chunked_matmul = false;
        status = execute_chunked_quantized_matmul_if_needed(operation, backend, lowerings,
            operators, reader, memory, device, operation_context, chunked_matmul,
            parity_counter);
        if (operation_context.weight_cache_pressure_mode) {
            if (!context.weight_cache_pressure_mode) {
                // The operation-local copy can still own resident whole weights.
                // Drop both copies only after the streamed operation has finished.
                operation_context.quantized_weights.clear();
                operation_context.weight_leases.clear();
                for (const auto& cached_name : plan_weight_names)
                    operation_context.tensors.erase(cached_name);
                for (auto& cached_weight : context.quantized_weights) {
                    cached_weight.second.buffer.reset();
                    cached_weight.second.chunks.clear();
                }
                for (const auto& cached_name : plan_weight_names)
                    context.tensors.erase(cached_name);
                context.weight_leases.clear();
            }
            context.weight_cache_pressure_mode = true;
            context.persistent_weight_cache_limit_bytes = 0;
        }
        if (!status.ok()) {
            report.errors.push_back(operation.id + ":CHUNKED_QUANTIZED_MATMUL_FAILED:" + status.message);
            for (const auto& name : operation.weight_tensors) {
                operation_context.tensors.erase(name);
                operation_context.quantized_weights.erase(name);
            }
            operation_context.weight_leases.clear();
            return status;
        }
        if (chunked_matmul) {
            ++report.compiled_nodes;
            for (const auto& name : operation.weight_tensors) {
                operation_context.tensors.erase(name);
                operation_context.quantized_weights.erase(name);
            }
            operation_context.weight_leases.clear();
            continue;
        }

        bool oversized_elementwise_candidate = false;
        if (operation.operator_name == "SiLU" || operation.operator_name == "Sigmoid" ||
            operation.operator_name == "Softplus" || operation.operator_name == "Add" ||
            operation.operator_name == "Multiply") {
            const auto out = operation_context.tensors.find(operation.output);
            oversized_elementwise_candidate = out != operation_context.tensors.end() &&
                out->second.byte_size() > graph_binding_limit;
            for (const auto& input_name : operation.inputs) {
                const auto in = operation_context.tensors.find(input_name);
                oversized_elementwise_candidate = oversized_elementwise_candidate ||
                    (in != operation_context.tensors.end() &&
                     in->second.byte_size() > graph_binding_limit);
            }
        }
        if (oversized_elementwise_candidate) {
            status = flush_pending_graphs();
            if (!status.ok()) return status;
        }
        bool chunked_elementwise = false;
        status = execute_chunked_elementwise_if_needed(operation, backend, lowerings,
            operators, device, operation_context, chunked_elementwise);
        if (!status.ok()) {
            report.errors.push_back(operation.id + ":CHUNKED_ELEMENTWISE_FAILED:" + status.message);
            for (const auto& name : operation.weight_tensors) {
                operation_context.tensors.erase(name);
                operation_context.quantized_weights.erase(name);
            }
            operation_context.weight_leases.clear();
            return status;
        }
        if (chunked_elementwise) {
            ++report.compiled_nodes;
            for (const auto& name : operation.weight_tensors) {
                operation_context.tensors.erase(name);
                operation_context.quantized_weights.erase(name);
            }
            operation_context.weight_leases.clear();
            continue;
        }

        {
            ComputeGraph graph;
            ModelGraphCompileReport compile_report;
            std::vector<std::string> operation_external_inputs = external_inputs;
            operation_external_inputs.reserve(operation_external_inputs.size() +
                                               operation_context.tensors.size() + 1);
            for (const auto& binding : operation_context.tensors)
                operation_external_inputs.push_back(binding.first);
            if (!operation_context.token_ids.empty())
                operation_external_inputs.emplace_back("token_ids");
            status = compile_model_execution_plan(single, backend.kind(), lowerings,
                operators, operation_external_inputs, operation_context, graph, compile_report);
            report.missing_lowerings.insert(report.missing_lowerings.end(),
                compile_report.missing_lowerings.begin(), compile_report.missing_lowerings.end());
            report.unresolved_inputs.insert(report.unresolved_inputs.end(),
                compile_report.unresolved_inputs.begin(), compile_report.unresolved_inputs.end());
            report.missing_weight_tensors.insert(report.missing_weight_tensors.end(),
                compile_report.missing_weight_tensors.begin(), compile_report.missing_weight_tensors.end());
            report.errors.insert(report.errors.end(), compile_report.errors.begin(), compile_report.errors.end());
            if (!status.ok()) {
                for (const auto& name : operation.weight_tensors) {
                    operation_context.tensors.erase(name);
                    operation_context.quantized_weights.erase(name);
                }
                operation_context.weight_leases.clear();
                return status;
            }
            pending_graphs.push_back(std::move(graph));
            report.compiled_nodes += compile_report.compiled_nodes;
            // Explicit numerical diagnostics need completed outputs before
            // readback. The normal path batches consecutive graph fragments.
            if (parity_counter != nullptr) {
                status = flush_pending_graphs();
                if (!status.ok()) {
                    report.errors.push_back(operation.id + ":STREAMED_DISPATCH_FAILED:" + status.message);
                    return status;
                }
                status = validate_resident_q6_k_matmul_sample(operation, reader, backend,
                    operation_context, parity_counter);
                if (!status.ok()) {
                    report.errors.push_back(operation.id + ":REAL_Q6_K_PARITY_FAILED:" + status.message);
                    return status;
                }
            }
        }

        // Keep only admitted resident quantized weights in the long-lived
        // session context. Metadata-only streamed bindings and temporary
        // operation references are discarded before the next operation.
        for (const auto& name : operation.weight_tensors) {
            operation_context.tensors.erase(name);
            const auto cached = operation_context.quantized_weights.find(name);
            if (cached != operation_context.quantized_weights.end() &&
                cached->second.buffer && context.persistent_weight_cache_limit_bytes != 0 &&
                cached->second.byte_size <= context.persistent_weight_cache_limit_bytes) {
                context.quantized_weights[name] = cached->second;
            } else if (cached != operation_context.quantized_weights.end()) {
                operation_context.quantized_weights.erase(cached);
            }
        }
        context.weight_leases = operation_context.weight_leases;
        operation_context.weight_leases.clear();
    }

    status = flush_pending_graphs();
    if (!status.ok()) {
        report.errors.push_back("STREAMED_BATCH_DISPATCH_FAILED:" + status.message);
        return status;
    }

    // All transient streamed weights are safe to release after the batch has
    // synchronized. Retain only leases backing explicitly admitted whole-tensor
    // cache entries; chunk-cache leases are owned by the LRU itself.
    std::unordered_set<const Buffer*> resident_weight_buffers;
    for (const auto& entry : context.quantized_weights) {
        if (entry.second.buffer) resident_weight_buffers.insert(entry.second.buffer.get());
    }
    context.weight_leases.erase(
        std::remove_if(context.weight_leases.begin(), context.weight_leases.end(),
            [&resident_weight_buffers](const MemoryLease& lease) {
                const auto buffer = lease.buffer();
                return !buffer || resident_weight_buffers.find(buffer.get()) ==
                    resident_weight_buffers.end();
            }),
        context.weight_leases.end());

    for (const auto& operation : plan.operations) {
        if (operation.operator_name != "KVCacheAppend") continue;
        if (!context.kv_cache || operation.inputs.size() != 2) {
            report.errors.push_back(operation.id + ":KV_CACHE_COMMIT_CONTEXT_INVALID");
            return {StatusCode::InvalidArgument, "KV cache append commit context is invalid"};
        }
        const auto layer = operation.integer_attributes.find("layer_index");
        const auto keys = context.tensors.find(operation.inputs[0]);
        if (layer == operation.integer_attributes.end() || layer->second > UINT32_MAX ||
            keys == context.tensors.end() || keys->second.descriptor().shape.empty() ||
            keys->second.descriptor().shape[0] > UINT32_MAX) {
            report.errors.push_back(operation.id + ":KV_CACHE_COMMIT_METADATA_INVALID");
            return {StatusCode::InvalidArgument, "KV cache append commit metadata is invalid"};
        }
        status = context.kv_cache->commit_device_append(static_cast<uint32_t>(layer->second),
            static_cast<uint32_t>(context.past_length),
            static_cast<uint32_t>(keys->second.descriptor().shape[0]));
        if (!status.ok()) {
            report.errors.push_back(operation.id + ":KV_CACHE_COMMIT_FAILED:" + status.message);
            return status;
        }
    }

    report.compiled = true;
    report.executed = true;
    plan.graph_compiled = true;
    plan.execution_validated = true;
    // Streaming is a valid execution strategy, not a model-family certification.
    plan.gpu_ready = false;
    return Status::success();
}

Status bind_model_kv_cache_views(const ModelExecutionPlan& plan, KvCache& cache,
                                 uint64_t past_length, uint64_t sequence_length,
                                 ModelExecutionContext& context) {
    if (sequence_length == 0 || past_length > UINT32_MAX ||
        sequence_length > UINT32_MAX - past_length)
        return {StatusCode::InvalidArgument, "KV cache view sequence range is invalid"};
    const uint32_t total_length = static_cast<uint32_t>(past_length + sequence_length);
    std::vector<std::pair<std::string, Tensor>> pending_views;
    for (const auto& operation : plan.operations) {
        if (operation.operator_name != "Attention") continue;
        if (operation.inputs.size() != 3)
            return {StatusCode::InvalidArgument, "Attention plan must reference Q and cache K/V tensor names"};
        const auto layer_attr = operation.integer_attributes.find("layer_index");
        if (layer_attr == operation.integer_attributes.end() || layer_attr->second > UINT32_MAX)
            return {StatusCode::InvalidArgument, "Attention plan is missing a valid KV cache layer index"};
        Tensor keys, values;
        Status status = cache.key_tensor_for_length(static_cast<uint32_t>(layer_attr->second),
                                                    total_length, keys);
        if (!status.ok()) return status;
        status = cache.value_tensor_for_length(static_cast<uint32_t>(layer_attr->second),
                                               total_length, values);
        if (!status.ok()) return status;
        pending_views.emplace_back(operation.inputs[1], std::move(keys));
        pending_views.emplace_back(operation.inputs[2], std::move(values));
    }

    // Rebind only cache-view names after all new views have been created
    // successfully. This supports repeated prefill/decode calls with a growing
    // valid prefix without disturbing weights or activation allocations.
    for (const auto& entry : pending_views) context.tensors.erase(entry.first);
    for (auto& entry : pending_views)
        context.tensors.emplace(std::move(entry.first), std::move(entry.second));
    context.kv_cache = &cache;
    context.past_length = past_length;
    context.sequence_length = sequence_length;
    return Status::success();
}

Status validate_model_plan_operator_coverage(ModelExecutionPlan& plan,
                                             const OperatorRegistry& registry,
                                             BackendKind backend,
                                             ModelGraphCoverageReport& report) {
    report = ModelGraphCoverageReport{};
    if (!plan.structurally_validated)
        return {StatusCode::InvalidArgument, "Cannot validate operator coverage for a structurally invalid model plan"};
    std::unordered_set<std::string> seen;
    for (const auto& operation : plan.operations) {
        if (!seen.insert(operation.operator_name).second) continue;
        if (operation.operator_name == "KVCacheAppend") {
            report.supported_operators.push_back(operation.operator_name);
            continue;
        }
        if (registry.has_validated_kernel(operation.operator_name, backend))
            report.supported_operators.push_back(operation.operator_name);
        else
            report.unsupported_operators.push_back(operation.operator_name);
    }
    std::sort(report.supported_operators.begin(), report.supported_operators.end());
    std::sort(report.unsupported_operators.begin(), report.unsupported_operators.end());
    report.complete = report.unsupported_operators.empty();
    const std::string blocker = "OPERATOR_KERNEL_COVERAGE_NOT_VALIDATED";
    auto found = std::find(plan.blockers.begin(), plan.blockers.end(), blocker);
    if (report.complete) {
        if (found != plan.blockers.end()) plan.blockers.erase(found);
    } else if (found == plan.blockers.end()) {
        plan.blockers.push_back(blocker);
    }
    plan.operator_coverage_validated = report.complete;
    plan.gpu_ready = plan.structurally_validated && plan.operator_coverage_validated &&
                     plan.execution_validated && plan.blockers.empty();
    return Status::success();
}

} // namespace ai_chat::gpu_engine
