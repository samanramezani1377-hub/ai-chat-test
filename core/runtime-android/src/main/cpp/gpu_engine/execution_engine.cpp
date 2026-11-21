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
    explicit State(uint64_t maximum) : budget(maximum) {}
    const uint64_t budget;
    std::atomic<uint64_t> allocated{0};
};

struct MemoryLease::Control {
    std::shared_ptr<Buffer> buffer;
    std::shared_ptr<MemoryManager::State> state;
    uint64_t bytes = 0;
    ~Control() {
        if (state && bytes != 0) state->allocated.fetch_sub(bytes, std::memory_order_relaxed);
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
    if (descriptor.size_bytes == 0) return {StatusCode::InvalidArgument, "Cannot allocate an empty GPU buffer"};
    uint64_t current = state_->allocated.load(std::memory_order_relaxed);
    for (;;) {
        if (descriptor.size_bytes > state_->budget || current > state_->budget - descriptor.size_bytes) {
            return {StatusCode::OutOfMemory, "GPU engine memory budget would be exceeded"};
        }
        if (state_->allocated.compare_exchange_weak(current, current + descriptor.size_bytes,
                std::memory_order_acq_rel, std::memory_order_relaxed)) break;
    }

    std::shared_ptr<Buffer> buffer;
    Status status = backend_.allocate(descriptor, buffer);
    if (!status.ok() || !buffer) {
        state_->allocated.fetch_sub(descriptor.size_bytes, std::memory_order_relaxed);
        return status.ok() ? Status{StatusCode::BackendFailure, "Backend returned success without a buffer"} : status;
    }
    auto control = std::make_shared<MemoryLease::Control>();
    control->buffer = std::move(buffer);
    control->state = state_;
    control->bytes = descriptor.size_bytes;
    out = MemoryLease(std::move(control));
    return Status::success();
}

uint64_t MemoryManager::budget_bytes() const noexcept { return state_->budget; }
uint64_t MemoryManager::allocated_bytes() const noexcept { return state_->allocated.load(std::memory_order_relaxed); }
uint64_t MemoryManager::available_bytes() const noexcept {
    const uint64_t used = allocated_bytes();
    return used >= state_->budget ? 0 : state_->budget - used;
}

Status OperatorRegistry::register_kernel(KernelDefinition definition) {
    if (definition.operator_name.empty() || definition.kernel_id.empty() ||
        definition.backend == BackendKind::Unknown || definition.input_type == DataType::Unknown ||
        definition.output_type == DataType::Unknown || definition.min_input_count == 0 ||
        definition.min_input_count > definition.max_input_count ||
        definition.min_rank == 0 || definition.min_rank > definition.max_rank) {
        return {StatusCode::InvalidArgument, "Kernel definition is missing required identity or valid constraints"};
    }
    if (definition.validation == KernelValidation::Validated && definition.wgsl_source.empty()) {
        return {StatusCode::InvalidArgument, "A validated Dawn kernel must include WGSL source"};
    }
    const auto duplicate = std::find_if(kernels_.begin(), kernels_.end(),
        [&definition](const KernelDefinition& current) {
            return current.kernel_id == definition.kernel_id &&
                   current.backend == definition.backend;
        });
    if (duplicate != kernels_.end()) return {StatusCode::InvalidArgument, "Duplicate kernel id for backend"};
    kernels_.push_back(std::move(definition));
    return Status::success();
}

Status OperatorRegistry::select_kernel(const std::string& operator_name, BackendKind backend,
                                       DataType input_type, DataType output_type,
                                       uint32_t input_count, uint32_t rank,
                                       KernelDefinition& out) const {
    const auto found = std::find_if(kernels_.begin(), kernels_.end(),
        [&](const KernelDefinition& candidate) {
            return candidate.operator_name == operator_name && candidate.backend == backend &&
                   candidate.input_type == input_type && candidate.output_type == output_type &&
                   input_count >= candidate.min_input_count && input_count <= candidate.max_input_count &&
                   rank >= candidate.min_rank && rank <= candidate.max_rank &&
                   candidate.validation == KernelValidation::Validated;
        });
    if (found == kernels_.end()) {
        return {StatusCode::Unsupported, "No validated kernel matches the requested operator signature"};
    }
    out = *found;
    return Status::success();
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
    if (dispatch.wgsl_source.empty() || dispatch.kernel_id != kernel_id)
        return {StatusCode::InvalidArgument, "Dispatch does not match the requested registered kernel"};
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

Status ComputeGraph::add_node(GraphNode node, size_t& out_index) {
    out_index = nodes_.size();
    for (size_t dependency : node.dependencies) {
        if (dependency >= nodes_.size()) {
            return {StatusCode::InvalidArgument, "Graph dependency must refer to an existing node"};
        }
    }
    if (node.name.empty()) return {StatusCode::InvalidArgument, "Graph node name cannot be empty"};
    if (node.dispatch.validation != KernelValidation::Validated || node.dispatch.kernel_id.empty()) {
        return {StatusCode::Unsupported, "Compute graph rejects unverified or unidentified kernels"};
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
