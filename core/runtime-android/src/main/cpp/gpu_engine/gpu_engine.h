#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace ai_chat::gpu_engine {

enum class StatusCode : uint8_t {
    Ok = 0,
    NotReady,
    Unsupported,
    InvalidArgument,
    OutOfMemory,
    DeviceLost,
    BackendFailure,
    NumericalFailure,
    Cancelled,
};

struct Status {
    StatusCode code = StatusCode::Ok;
    std::string message;

    bool ok() const noexcept { return code == StatusCode::Ok; }
    static Status success() { return {}; }
};

enum class BackendKind : uint8_t {
    Unknown = 0,
    DawnWebGpu,
    OpenGles,
};

enum class CapabilityState : uint8_t {
    Unknown = 0,
    Discovered,
    SelfTested,
    BackendReady,
    OperatorReady,
    ModelValidated,
    DeviceValidated,
};

struct DeviceLimits {
    uint64_t max_buffer_size = 0;
    uint64_t max_storage_buffer_binding_size = 0;
    uint32_t max_compute_workgroup_size_x = 0;
    uint32_t max_compute_workgroup_size_y = 0;
    uint32_t max_compute_workgroup_size_z = 0;
    uint32_t max_compute_workgroup_invocations = 0;
    uint32_t max_compute_workgroup_storage_bytes = 0;
    uint32_t max_compute_workgroups_per_dimension = 0;
    uint32_t max_storage_buffers_per_shader_stage = 0;
    uint32_t min_storage_buffer_offset_alignment = 1;
};

struct DeviceInfo {
    BackendKind backend = BackendKind::Unknown;
    CapabilityState state = CapabilityState::Unknown;
    std::string name;
    std::string vendor;
    std::string architecture;
    std::string driver;
    std::string api_version;
    bool compute_available = false;
    bool compute_self_test_passed = false;
    bool inference_ready = false;
    DeviceLimits limits;
    std::vector<std::string> features;
    std::string diagnostic;
};

enum class DataType : uint8_t {
    Unknown = 0,
    F32,
    F16,
    BF16,
    I8,
    U8,
    I16,
    U16,
    I32,
    U32,
};

enum class KernelValidation : uint8_t {
    Unverified = 0,
    Validated,
    Rejected,
};

enum class MemoryDomain : uint8_t {
    Host = 0,
    Device,
    Shared,
};

enum class TensorLayout : uint8_t {
    Contiguous = 0,
    Strided,
    Opaque,
};

struct TensorDesc {
    std::vector<uint64_t> shape;
    std::vector<uint64_t> strides_bytes;
    DataType data_type = DataType::Unknown;
    TensorLayout layout = TensorLayout::Contiguous;
    MemoryDomain memory = MemoryDomain::Device;
    uint64_t alignment = 1;
};

struct BufferDesc {
    uint64_t size_bytes = 0;
    uint64_t alignment = 1;
    MemoryDomain memory = MemoryDomain::Device;
    bool persistent = false;
    std::string label;
};

class Buffer {
public:
    virtual ~Buffer() = default;
    virtual uint64_t size_bytes() const noexcept = 0;
    virtual MemoryDomain memory_domain() const noexcept = 0;
};

struct BufferCopy {
    std::shared_ptr<Buffer> source;
    uint64_t source_offset = 0;
    std::shared_ptr<Buffer> destination;
    uint64_t destination_offset = 0;
    uint64_t size_bytes = 0;
};

struct BufferBinding {
    uint32_t binding = 0;
    std::shared_ptr<Buffer> buffer;
    uint64_t offset_bytes = 0;
    uint64_t size_bytes = 0;
};

struct ComputeDispatch {
    std::string label;
    std::string kernel_id;
    BackendKind backend = BackendKind::Unknown;
    KernelValidation validation = KernelValidation::Unverified;
    std::string wgsl_source;
    std::string glsl_source;
    std::string entry_point = "main";
    std::vector<BufferBinding> bindings;
    std::vector<BufferCopy> buffer_copies;
    bool copy_only = false;
    uint32_t workgroups_x = 1;
    uint32_t workgroups_y = 1;
    uint32_t workgroups_z = 1;
    uint32_t workgroup_size_x = 1;
    uint32_t workgroup_size_y = 1;
    uint32_t workgroup_size_z = 1;
};

class CommandBatch {
public:
    virtual ~CommandBatch() = default;
    virtual Status copy_buffer(const std::shared_ptr<Buffer>& source,
                               uint64_t source_offset,
                               const std::shared_ptr<Buffer>& destination,
                               uint64_t destination_offset,
                               uint64_t size_bytes) = 0;
    virtual Status dispatch_compute(const ComputeDispatch& dispatch) = 0;
};

class Backend {
public:
    virtual ~Backend() = default;
    virtual BackendKind kind() const noexcept = 0;
    virtual Status enumerate_devices(std::vector<DeviceInfo>& out) = 0;
    virtual Status initialize(const DeviceInfo& device) = 0;
    virtual Status allocate(const BufferDesc& desc, std::shared_ptr<Buffer>& out) = 0;
    virtual Status begin_commands(std::unique_ptr<CommandBatch>& out) = 0;
    virtual Status submit(std::unique_ptr<CommandBatch> commands) = 0;
    virtual Status synchronize() = 0;
    virtual Status upload(const std::shared_ptr<Buffer>& destination,
                          uint64_t destination_offset,
                          const void* source,
                          uint64_t size_bytes) = 0;
    virtual Status readback(const std::shared_ptr<Buffer>& source,
                            uint64_t source_offset,
                            void* destination,
                            uint64_t size_bytes) = 0;
    virtual Status shutdown() = 0;
    virtual bool healthy() const noexcept = 0;
};

// Creates the first native Dawn/WebGPU backend. Device execution is still
// opt-in until the runtime router has validated model/operator support.
std::shared_ptr<Backend> create_dawn_backend();
std::shared_ptr<Backend> create_universal_gles_backend();

class BackendRegistry {
public:
    Status register_backend(std::shared_ptr<Backend> backend);
    std::vector<BackendKind> registered_backends() const;
    std::shared_ptr<Backend> find(BackendKind kind) const;

private:
    std::vector<std::shared_ptr<Backend>> backends_;
};

struct RuntimeDiagnostics {
    std::string runtime_name = "AIChatGpuEngine";
    std::string runtime_version = "0.1.0";
    BackendKind selected_backend = BackendKind::Unknown;
    CapabilityState capability_state = CapabilityState::Unknown;
    std::string device_name = "unknown";
    std::string model_architecture = "unknown";
    std::string quantization = "unknown";
    uint64_t host_memory_bytes = 0;
    uint64_t device_memory_bytes = 0;
    bool cpu_fallback_available = true;
    bool fallback_used = false;
    std::string fallback_reason;
    std::string last_error;
};

class ComputeGraph;

class GpuEngine {
public:
    Status register_backend(std::shared_ptr<Backend> backend);
    Status initialize(BackendKind preferred_backend = BackendKind::DawnWebGpu);
    Status initialize_with_fallback(const std::vector<BackendKind>& preference_order);
    Status execute(const ComputeGraph& graph, const std::atomic<bool>* cancellation = nullptr);
    Status shutdown();

    bool ready() const noexcept;
    const DeviceInfo* active_device() const noexcept;
    const RuntimeDiagnostics& diagnostics() const noexcept;

private:
    BackendRegistry registry_;
    std::shared_ptr<Backend> active_backend_;
    DeviceInfo active_device_;
    RuntimeDiagnostics diagnostics_;
};


} // namespace ai_chat::gpu_engine
