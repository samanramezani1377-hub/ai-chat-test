#pragma once

#include "gpu_engine.h"

#include <atomic>
#include <memory>
#include <string>
#include <vector>

namespace ai_chat::gpu_engine {

class Tensor {
public:
    static Status required_bytes(const TensorDesc& descriptor, uint64_t& out_bytes);
    static Status create(TensorDesc descriptor, std::shared_ptr<Buffer> buffer,
                         uint64_t byte_offset, Tensor& out);

    const TensorDesc& descriptor() const noexcept { return descriptor_; }
    const std::shared_ptr<Buffer>& buffer() const noexcept { return buffer_; }
    uint64_t byte_offset() const noexcept { return byte_offset_; }
    uint64_t byte_size() const noexcept { return byte_size_; }
    uint64_t element_count() const noexcept { return element_count_; }

private:
    TensorDesc descriptor_;
    std::shared_ptr<Buffer> buffer_;
    uint64_t byte_offset_ = 0;
    uint64_t byte_size_ = 0;
    uint64_t element_count_ = 0;
};

class MemoryLease {
public:
    MemoryLease() = default;
    explicit operator bool() const noexcept;
    std::shared_ptr<Buffer> buffer() const noexcept;

private:
    struct Control;
    explicit MemoryLease(std::shared_ptr<Control> control);
    std::shared_ptr<Control> control_;
    friend class MemoryManager;
};

class MemoryManager {
public:
    MemoryManager(Backend& backend, uint64_t budget_bytes);
    Status allocate(const BufferDesc& descriptor, MemoryLease& out);
    uint64_t budget_bytes() const noexcept;
    uint64_t allocated_bytes() const noexcept;
    uint64_t available_bytes() const noexcept;

private:
    struct State;
    friend class MemoryLease;
    Backend& backend_;
    std::shared_ptr<State> state_;
};

struct KernelDefinition {
    std::string operator_name;
    BackendKind backend = BackendKind::Unknown;
    DataType input_type = DataType::Unknown;
    std::vector<DataType> input_types;
    DataType output_type = DataType::Unknown;
    uint32_t min_input_count = 1;
    uint32_t max_input_count = 1;
    uint32_t min_rank = 1;
    uint32_t max_rank = 8;
    KernelValidation validation = KernelValidation::Unverified;
    std::string kernel_id;
    std::string wgsl_source;
    std::string glsl_source;
    std::string entry_point = "main";
};

class OperatorRegistry {
public:
    Status register_kernel(KernelDefinition definition);
    Status select_kernel(const std::string& operator_name, BackendKind backend,
                         DataType input_type, DataType output_type,
                         uint32_t input_count, uint32_t rank,
                         KernelDefinition& out) const;
    Status select_kernel(const std::string& operator_name, BackendKind backend,
                         const std::vector<DataType>& input_types, DataType output_type,
                         uint32_t rank, KernelDefinition& out) const;
    Status authorize_dispatch(const std::string& kernel_id, BackendKind backend,
                              ComputeDispatch& dispatch) const;
    bool is_validated_kernel(const std::string& kernel_id, BackendKind backend) const noexcept;
    std::vector<std::string> registered_operators() const;

private:
    std::vector<KernelDefinition> kernels_;
};

struct GraphNode {
    std::string name;
    std::vector<size_t> dependencies;
    ComputeDispatch dispatch;
};

class OperatorRegistry;

class ComputeGraph {
public:
    Status add_node(GraphNode node, const OperatorRegistry& registry, size_t& out_index);
    Status topological_order(std::vector<size_t>& out_order) const;
    size_t node_count() const noexcept;

private:
    std::vector<GraphNode> nodes_;
    friend class ExecutionScheduler;
};

class ExecutionScheduler {
public:
    Status execute(Backend& backend, const ComputeGraph& graph,
                   const std::atomic<bool>* cancellation = nullptr) const;
};

} // namespace ai_chat::gpu_engine
