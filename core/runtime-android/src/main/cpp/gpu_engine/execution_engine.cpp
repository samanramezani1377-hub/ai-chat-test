#include "execution_engine.h"

#include <algorithm>
#include <iterator>
#include <limits>
#include <mutex>
#include <unordered_set>
#include <utility>

namespace ai_chat::gpu_engine {
namespace {

bool multiply_overflow(uint64_t a, uint64_t b, uint64_t& result) {
    if (a != 0 && b > std::numeric_limits<uint64_t>::max() / a) return true;
    result = a * b;
    return false;
}

uint64_t element_size(DataType type) {
    switch (type) {
        case DataType::F32:
        case DataType::I32:
        case DataType::U32: return 4;
        case DataType::F16:
        case DataType::BF16:
        case DataType::I16:
        case DataType::U16: return 2;
        case DataType::I8:
        case DataType::U8: return 1;
        case DataType::Unknown: return 0;
    }
    return 0;
}

} // namespace

Status Tensor::required_bytes(const TensorDesc& descriptor, uint64_t& out_bytes) {
    out_bytes = 0;
    if (descriptor.shape.empty() || descriptor.shape.size() > 16) {
        return {StatusCode::InvalidArgument, "Tensor rank must be between 1 and 16"};
    }
    const uint64_t item_size = element_size(descriptor.data_type);
    if (item_size == 0) return {StatusCode::Unsupported, "Tensor data type has no scalar byte size"};
    if (descriptor.alignment == 0 || (descriptor.alignment & (descriptor.alignment - 1)) != 0) {
        return {StatusCode::InvalidArgument, "Tensor alignment must be a non-zero power of two"};
    }
    for (uint64_t dimension : descriptor.shape) {
        if (dimension == 0) return {StatusCode::InvalidArgument, "Tensor dimensions must be non-zero"};
    }

    if (descriptor.layout == TensorLayout::Contiguous && descriptor.strides_bytes.empty()) {
        uint64_t elements = 1;
        for (uint64_t dimension : descriptor.shape) {
            if (multiply_overflow(elements, dimension, elements)) {
                return {StatusCode::InvalidArgument, "Tensor element count overflows uint64"};
            }
        }
        if (multiply_overflow(elements, item_size, out_bytes)) {
            return {StatusCode::InvalidArgument, "Tensor byte size overflows uint64"};
        }
        return Status::success();
    }

    if (descriptor.strides_bytes.size() != descriptor.shape.size()) {
        return {StatusCode::InvalidArgument, "Strided tensor must provide one byte stride per dimension"};
    }
    uint64_t last_byte = item_size;
    for (size_t i = 0; i < descriptor.shape.size(); ++i) {
        uint64_t contribution = 0;
        if (multiply_overflow(descriptor.shape[i] - 1, descriptor.strides_bytes[i], contribution) ||
            contribution > std::numeric_limits<uint64_t>::max() - last_byte) {
            return {StatusCode::InvalidArgument, "Strided tensor byte range overflows uint64"};
        }
        last_byte += contribution;
    }
    out_bytes = last_byte;
    return Status::success();
}

Status Tensor::create(TensorDesc descriptor, std::shared_ptr<Buffer> buffer,
                      uint64_t byte_offset, Tensor& out) {
    out = Tensor{};
    if (!buffer) return {StatusCode::InvalidArgument, "Tensor requires a valid backing buffer"};
    if (descriptor.memory != buffer->memory_domain()) {
        return {StatusCode::InvalidArgument, "Tensor memory domain does not match its backing buffer"};
    }
    uint64_t required = 0;
    Status status = required_bytes(descriptor, required);
    if (!status.ok()) return status;
    if (byte_offset % descriptor.alignment != 0 ||
        byte_offset > buffer->size_bytes() || required > buffer->size_bytes() - byte_offset) {
        return {StatusCode::InvalidArgument, "Tensor storage range or alignment exceeds its backing buffer"};
    }
    uint64_t elements = 1;
    for (uint64_t dimension : descriptor.shape) {
        if (multiply_overflow(elements, dimension, elements)) {
            return {StatusCode::InvalidArgument, "Tensor element count overflows uint64"};
        }
    }
    out.descriptor_ = std::move(descriptor);
    out.buffer_ = std::move(buffer);
    out.byte_offset_ = byte_offset;
    out.byte_size_ = required;
    out.element_count_ = elements;
    return Status::success();
}

struct MemoryManager::State {
    struct PooledBuffer {
        std::shared_ptr<Buffer> buffer;
        uint64_t bytes = 0;
        uint64_t alignment = 1;
        MemoryDomain domain = MemoryDomain::Device;
    };

    explicit State(uint64_t maximum) : budget(maximum) {}
    const uint64_t budget;
    std::atomic<uint64_t> active_bytes{0};
    std::atomic<uint64_t> reserved_bytes{0};
    std::mutex pool_mutex;
    std::vector<PooledBuffer> pool;
    uint64_t pooled_bytes = 0;

    void release(std::shared_ptr<Buffer> buffer, uint64_t logical_bytes,
                 uint64_t physical_bytes, uint64_t alignment,
                 MemoryDomain domain, bool persistent) noexcept {
        if (logical_bytes != 0) active_bytes.fetch_sub(logical_bytes, std::memory_order_relaxed);
        if (!buffer || physical_bytes == 0) {
            if (physical_bytes != 0) reserved_bytes.fetch_sub(physical_bytes, std::memory_order_relaxed);
            return;
        }
        // Keep a bounded transient pool. Persistent model/KV allocations are
        // returned to the backend when their owner releases them.
        const uint64_t pool_limit = budget / 2;
        if (!persistent && physical_bytes <= pool_limit) {
            std::lock_guard<std::mutex> lock(pool_mutex);
            if (pooled_bytes <= pool_limit - physical_bytes) {
                try {
                    pool.push_back({buffer, physical_bytes, alignment, domain});
                    pooled_bytes += physical_bytes;
                    return;
                } catch (...) {
                    // A pool bookkeeping allocation must never terminate inference.
                }
            }
        }
        reserved_bytes.fetch_sub(physical_bytes, std::memory_order_relaxed);
    }
};

struct MemoryLease::Control {
    std::shared_ptr<Buffer> buffer;
    std::shared_ptr<MemoryManager::State> state;
    uint64_t logical_bytes = 0;
    uint64_t physical_bytes = 0;
    uint64_t alignment = 1;
    MemoryDomain domain = MemoryDomain::Device;
    bool persistent = false;
    ~Control() {
        if (state) state->release(std::move(buffer), logical_bytes, physical_bytes,
                                  alignment, domain, persistent);
    }
};

MemoryLease::MemoryLease(std::shared_ptr<Control> control) : control_(std::move(control)) {}
MemoryLease::operator bool() const noexcept { return control_ && control_->buffer; }
std::shared_ptr<Buffer> MemoryLease::buffer() const noexcept {
    return control_ ? std::shared_ptr<Buffer>(control_, control_->buffer.get()) : std::shared_ptr<Buffer>{};
}

MemoryManager::MemoryManager(Backend& backend, uint64_t budget_bytes)
    : backend_(backend), state_(std::make_shared<State>(budget_bytes)) {}

Status MemoryManager::allocate(const BufferDesc& descriptor, MemoryLease& out) {
    out = MemoryLease{};
    if (descriptor.size_bytes == 0 || descriptor.alignment == 0 ||
        (descriptor.alignment & (descriptor.alignment - 1)) != 0) {
        return {StatusCode::InvalidArgument, "GPU buffer size and alignment must be valid"};
    }

    std::shared_ptr<Buffer> buffer;
    uint64_t physical_bytes = descriptor.size_bytes;
    {
        std::lock_guard<std::mutex> lock(state_->pool_mutex);
        auto best = state_->pool.end();
        for (auto it = state_->pool.begin(); it != state_->pool.end(); ++it) {
            if (it->domain != descriptor.memory || it->alignment < descriptor.alignment ||
                it->bytes < descriptor.size_bytes) continue;
            if (best == state_->pool.end() || it->bytes < best->bytes) best = it;
        }
        if (best != state_->pool.end()) {
            physical_bytes = best->bytes;
            buffer = std::move(best->buffer);
            state_->pooled_bytes -= best->bytes;
            state_->pool.erase(best);
        }
    }

    if (buffer) {
        state_->active_bytes.fetch_add(descriptor.size_bytes, std::memory_order_relaxed);
    } else {
        uint64_t reserved = state_->reserved_bytes.load(std::memory_order_relaxed);
        for (;;) {
            if (descriptor.size_bytes > state_->budget ||
                reserved > state_->budget - descriptor.size_bytes) {
                return {StatusCode::OutOfMemory, "GPU engine memory budget would be exceeded"};
            }
            if (state_->reserved_bytes.compare_exchange_weak(reserved,
                    reserved + descriptor.size_bytes, std::memory_order_acq_rel,
                    std::memory_order_relaxed)) break;
        }
        state_->active_bytes.fetch_add(descriptor.size_bytes, std::memory_order_relaxed);
        Status status = backend_.allocate(descriptor, buffer);
        if (!status.ok() || !buffer) {
            state_->active_bytes.fetch_sub(descriptor.size_bytes, std::memory_order_relaxed);
            state_->reserved_bytes.fetch_sub(descriptor.size_bytes, std::memory_order_relaxed);
            return status.ok() ? Status{StatusCode::BackendFailure, "Backend returned success without a buffer"} : status;
        }
    }

    auto control = std::make_shared<MemoryLease::Control>();
    control->buffer = std::move(buffer);
    control->state = state_;
    control->logical_bytes = descriptor.size_bytes;
    control->physical_bytes = physical_bytes;
    control->alignment = descriptor.alignment;
    control->domain = descriptor.memory;
    control->persistent = descriptor.persistent;
    out = MemoryLease(std::move(control));
    return Status::success();
}

uint64_t MemoryManager::budget_bytes() const noexcept { return state_->budget; }
uint64_t MemoryManager::allocated_bytes() const noexcept { return state_->active_bytes.load(std::memory_order_relaxed); }
uint64_t MemoryManager::available_bytes() const noexcept {
    const uint64_t used = state_->reserved_bytes.load(std::memory_order_relaxed);
    return used >= state_->budget ? 0 : state_->budget - used;
}

Status OperatorRegistry::register_kernel(KernelDefinition definition) {
    const bool invalid_input_types = !definition.input_types.empty() &&
        (definition.input_types.size() < definition.min_input_count ||
         definition.input_types.size() > definition.max_input_count ||
         std::any_of(definition.input_types.begin(), definition.input_types.end(),
                     [](DataType type) { return type == DataType::Unknown; }));
    if (definition.operator_name.empty() || definition.kernel_id.empty() ||
        definition.backend == BackendKind::Unknown || definition.input_type == DataType::Unknown ||
        definition.output_type == DataType::Unknown || definition.min_input_count == 0 ||
        definition.min_input_count > definition.max_input_count ||
        definition.min_rank == 0 || definition.min_rank > definition.max_rank ||
        invalid_input_types) {
        return {StatusCode::InvalidArgument, "Kernel definition is missing required identity or valid constraints"};
    }
    if (definition.validation == KernelValidation::Validated) {
        const bool source_present = definition.backend == BackendKind::DawnWebGpu
            ? !definition.wgsl_source.empty()
            : (definition.backend == BackendKind::OpenGles && !definition.glsl_source.empty());
        if (!source_present)
            return {StatusCode::InvalidArgument, "A validated kernel must include source for its target backend"};
    }
    const auto duplicate = std::find_if(kernels_.begin(), kernels_.end(),
        [&definition](const KernelDefinition& current) {
            return current.kernel_id == definition.kernel_id && current.backend == definition.backend;
        });
    if (duplicate != kernels_.end()) return {StatusCode::InvalidArgument, "Duplicate kernel id for backend"};
    kernels_.push_back(std::move(definition));
    return Status::success();
}

Status OperatorRegistry::select_kernel(const std::string& operator_name, BackendKind backend,
                                            DataType input_type, DataType output_type,
                                            uint32_t input_count, uint32_t rank,
                                            KernelDefinition& out) const {
    return select_kernel(operator_name, backend,
                         std::vector<DataType>(input_count, input_type),
                         output_type, rank, out);
}

Status OperatorRegistry::select_kernel(const std::string& operator_name, BackendKind backend,
                                            const std::vector<DataType>& input_types,
                                            DataType output_type, uint32_t rank,
                                            KernelDefinition& out) const {
    out = KernelDefinition{};
    const auto found = std::find_if(kernels_.begin(), kernels_.end(),
        [&operator_name, backend, &input_types, output_type, rank](const KernelDefinition& definition) {
            if (definition.operator_name != operator_name || definition.backend != backend ||
                definition.output_type != output_type ||
                input_types.size() < definition.min_input_count ||
                input_types.size() > definition.max_input_count ||
                rank < definition.min_rank || rank > definition.max_rank ||
                definition.validation != KernelValidation::Validated) return false;
            if (!definition.input_types.empty()) return definition.input_types == input_types;
            return std::all_of(input_types.begin(), input_types.end(),
                [&definition](DataType type) { return type == definition.input_type; });
        });
    if (found == kernels_.end())
        return {StatusCode::Unsupported, "No validated kernel matches operator, backend, input types, output type and rank"};
    out = *found;
    return Status::success();
}

bool OperatorRegistry::is_validated_kernel(const std::string& kernel_id,
                                           BackendKind backend) const noexcept {
    return std::any_of(kernels_.begin(), kernels_.end(),
        [&kernel_id, backend](const KernelDefinition& definition) {
            return definition.kernel_id == kernel_id && definition.backend == backend &&
                   definition.validation == KernelValidation::Validated;
        });
}

Status OperatorRegistry::authorize_dispatch(const std::string& kernel_id, BackendKind backend,
                                             ComputeDispatch& dispatch) const {
    const auto found = std::find_if(kernels_.begin(), kernels_.end(),
        [&kernel_id, backend](const KernelDefinition& definition) {
            return definition.kernel_id == kernel_id && definition.backend == backend &&
                   definition.validation == KernelValidation::Validated;
        });
    if (found == kernels_.end())
        return {StatusCode::Unsupported, "Kernel is not registered as validated for this backend"};
    const bool source_present = backend == BackendKind::DawnWebGpu
        ? !dispatch.wgsl_source.empty()
        : (backend == BackendKind::OpenGles && !dispatch.glsl_source.empty());
    if (!source_present || dispatch.kernel_id != kernel_id)
        return {StatusCode::InvalidArgument, "Dispatch does not match the requested registered kernel source and identity"};
    dispatch.backend = backend;
    dispatch.validation = KernelValidation::Validated;
    return Status::success();
}

std::vector<std::string> OperatorRegistry::registered_operators() const {
    std::vector<std::string> result;
    for (const auto& kernel : kernels_) {
        if (std::find(result.begin(), result.end(), kernel.operator_name) == result.end()) {
            result.push_back(kernel.operator_name);
        }
    }
    return result;
}

Status ComputeGraph::add_node(GraphNode node, const OperatorRegistry& registry, size_t& out_index) {
    out_index = nodes_.size();
    for (size_t dependency : node.dependencies) {
        if (dependency >= nodes_.size()) {
            return {StatusCode::InvalidArgument, "Graph dependency must refer to an existing node"};
        }
    }
    if (node.name.empty()) return {StatusCode::InvalidArgument, "Graph node name cannot be empty"};
    if (node.dispatch.validation != KernelValidation::Validated || node.dispatch.kernel_id.empty() ||
        node.dispatch.backend == BackendKind::Unknown ||
        !registry.is_validated_kernel(node.dispatch.kernel_id, node.dispatch.backend)) {
        return {StatusCode::Unsupported, "Compute graph rejects kernels not authorized by the active registry"};
    }
    nodes_.push_back(std::move(node));
    return Status::success();
}

Status ComputeGraph::topological_order(std::vector<size_t>& out_order) const {
    out_order.clear();
    std::vector<size_t> indegree(nodes_.size(), 0);
    std::vector<std::vector<size_t>> dependents(nodes_.size());
    for (size_t i = 0; i < nodes_.size(); ++i) {
        std::unordered_set<size_t> unique;
        for (size_t dependency : nodes_[i].dependencies) {
            if (dependency >= nodes_.size() || !unique.insert(dependency).second) {
                return {StatusCode::InvalidArgument, "Graph contains an invalid or duplicate dependency"};
            }
            ++indegree[i];
            dependents[dependency].push_back(i);
        }
    }
    std::vector<size_t> ready;
    ready.reserve(nodes_.size());
    for (size_t i = 0; i < indegree.size(); ++i) if (indegree[i] == 0) ready.push_back(i);
    for (size_t cursor = 0; cursor < ready.size(); ++cursor) {
        const size_t current = ready[cursor];
        out_order.push_back(current);
        for (size_t dependent : dependents[current]) {
            if (--indegree[dependent] == 0) ready.push_back(dependent);
        }
    }
    if (out_order.size() != nodes_.size()) {
        out_order.clear();
        return {StatusCode::InvalidArgument, "Compute graph contains a dependency cycle"};
    }
    return Status::success();
}

size_t ComputeGraph::node_count() const noexcept { return nodes_.size(); }

Status ExecutionScheduler::execute(Backend& backend, const ComputeGraph& graph,
                                   const std::atomic<bool>* cancellation) const {
    if (!backend.healthy()) return {StatusCode::DeviceLost, "Cannot schedule work on an unhealthy backend"};
    std::vector<size_t> order;
    Status status = graph.topological_order(order);
    if (!status.ok()) return status;
    for (size_t index : order) {
        if (graph.nodes_[index].dispatch.backend != backend.kind()) {
            return {StatusCode::Unsupported, "Graph kernel backend does not match the active device backend"};
        }
    }
    std::unique_ptr<CommandBatch> batch;
    status = backend.begin_commands(batch);
    if (!status.ok()) return status;
    if (!batch) return {StatusCode::BackendFailure, "Backend returned an empty command batch"};

    for (size_t index : order) {
        if (cancellation && cancellation->load(std::memory_order_relaxed)) {
            return {StatusCode::Cancelled, "Graph execution was cancelled before submission"};
        }
        status = batch->dispatch_compute(graph.nodes_[index].dispatch);
        if (!status.ok()) return status;
    }
    if (cancellation && cancellation->load(std::memory_order_relaxed)) {
        return {StatusCode::Cancelled, "Graph execution was cancelled before submission"};
    }
    status = backend.submit(std::move(batch));
    if (!status.ok()) return status;
    if (!backend.healthy()) return {StatusCode::DeviceLost, "Backend was lost after command submission"};
    return backend.synchronize();
}

} // namespace ai_chat::gpu_engine
