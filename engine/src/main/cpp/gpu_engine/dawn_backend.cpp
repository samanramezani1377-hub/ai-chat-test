#include "gpu_engine.h"

#include <webgpu/webgpu_cpp.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cstring>
#include <limits>
#include <iterator>
#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

namespace ai_chat::gpu_engine {
namespace {

struct AdapterRequestState {
    wgpu::RequestAdapterStatus status = wgpu::RequestAdapterStatus::Error;
    std::string message;
    wgpu::Adapter adapter;
};

struct DeviceRequestState {
    wgpu::RequestDeviceStatus status = wgpu::RequestDeviceStatus::Error;
    std::string message;
    wgpu::Device device;
};

struct QueueWorkState {
    wgpu::QueueWorkDoneStatus status = wgpu::QueueWorkDoneStatus::Error;
    std::string message;
};

struct MapState {
    wgpu::MapAsyncStatus status = wgpu::MapAsyncStatus::Error;
    std::string message;
};

Status backend_error(const char* operation, const std::string& detail) {
    return {StatusCode::BackendFailure, std::string(operation) + ": " + detail};
}

class DawnBuffer final : public Buffer {
public:
    DawnBuffer(wgpu::Buffer buffer, uint64_t size, MemoryDomain domain)
        : buffer_(std::move(buffer)), size_(size), domain_(domain) {}

    uint64_t size_bytes() const noexcept override { return size_; }
    MemoryDomain memory_domain() const noexcept override { return domain_; }
    wgpu::Buffer& handle() noexcept { return buffer_; }

private:
    wgpu::Buffer buffer_;
    uint64_t size_ = 0;
    MemoryDomain domain_ = MemoryDomain::Device;
};

class DawnCommandBatch final : public CommandBatch {
public:
    explicit DawnCommandBatch(wgpu::Device device, DeviceLimits limits)
        : device_(std::move(device)), limits_(limits), encoder_(device_.CreateCommandEncoder()) {}

    bool valid() const noexcept { return encoder_ != nullptr; }

    Status copy_buffer(const std::shared_ptr<Buffer>& source,
                       uint64_t source_offset,
                       const std::shared_ptr<Buffer>& destination,
                       uint64_t destination_offset,
                       uint64_t size_bytes) override {
        auto* src = dynamic_cast<DawnBuffer*>(source.get());
        auto* dst = dynamic_cast<DawnBuffer*>(destination.get());
        if (!src || !dst || size_bytes == 0 ||
            source_offset > src->size_bytes() || size_bytes > src->size_bytes() - source_offset ||
            destination_offset > dst->size_bytes() || size_bytes > dst->size_bytes() - destination_offset ||
            (source_offset % 4) != 0 || (destination_offset % 4) != 0 || (size_bytes % 4) != 0) {
            return {StatusCode::InvalidArgument, "Invalid Dawn buffer copy range or backend buffer"};
        }
        encoder_.CopyBufferToBuffer(src->handle(), source_offset, dst->handle(), destination_offset, size_bytes);
        return Status::success();
    }

    Status dispatch_compute(const ComputeDispatch& dispatch) override {
        if (dispatch.validation != KernelValidation::Validated || dispatch.kernel_id.empty() ||
            dispatch.backend != BackendKind::DawnWebGpu) {
            return {StatusCode::Unsupported, "Dawn refuses dispatches without a validated kernel identity"};
        }
        if (dispatch.wgsl_source.empty() ||
            dispatch.workgroups_x == 0 || dispatch.workgroups_y == 0 || dispatch.workgroups_z == 0 ||
            dispatch.bindings.empty()) {
            return {StatusCode::InvalidArgument, "Compute dispatch requires WGSL, entry point, bindings and non-zero workgroups"};
        }
        const uint64_t workgroup_invocations = static_cast<uint64_t>(dispatch.workgroup_size_x) *
            dispatch.workgroup_size_y * dispatch.workgroup_size_z;
        if (!dispatch.workgroup_size_x || !dispatch.workgroup_size_y || !dispatch.workgroup_size_z ||
            (limits_.max_compute_workgroup_size_x && dispatch.workgroup_size_x > limits_.max_compute_workgroup_size_x) ||
            (limits_.max_compute_workgroup_size_y && dispatch.workgroup_size_y > limits_.max_compute_workgroup_size_y) ||
            (limits_.max_compute_workgroup_size_z && dispatch.workgroup_size_z > limits_.max_compute_workgroup_size_z) ||
            (limits_.max_compute_workgroup_invocations && workgroup_invocations > limits_.max_compute_workgroup_invocations)) {
            return {StatusCode::Unsupported, "Compute dispatch exceeds Dawn workgroup-size or invocation limits"};
        }
        if (limits_.max_compute_workgroups_per_dimension != 0 &&
            (dispatch.workgroups_x > limits_.max_compute_workgroups_per_dimension ||
             dispatch.workgroups_y > limits_.max_compute_workgroups_per_dimension ||
             dispatch.workgroups_z > limits_.max_compute_workgroups_per_dimension)) {
            return {StatusCode::Unsupported, "Compute dispatch exceeds Dawn workgroup-count limits"};
        }
        const uint64_t local_invocations = static_cast<uint64_t>(dispatch.workgroup_size_x) *
            dispatch.workgroup_size_y * dispatch.workgroup_size_z;
        if (!dispatch.workgroup_size_x || !dispatch.workgroup_size_y || !dispatch.workgroup_size_z ||
            (limits_.max_compute_workgroup_size_x && dispatch.workgroup_size_x > limits_.max_compute_workgroup_size_x) ||
            (limits_.max_compute_workgroup_size_y && dispatch.workgroup_size_y > limits_.max_compute_workgroup_size_y) ||
            (limits_.max_compute_workgroup_size_z && dispatch.workgroup_size_z > limits_.max_compute_workgroup_size_z) ||
            (limits_.max_compute_workgroup_invocations && local_invocations > limits_.max_compute_workgroup_invocations)) {
            return {StatusCode::Unsupported, "Compute dispatch exceeds Dawn local workgroup-size or invocation limits"};
        }
        if (limits_.max_storage_buffers_per_shader_stage != 0 &&
            dispatch.bindings.size() > limits_.max_storage_buffers_per_shader_stage) {
            return {StatusCode::Unsupported, "Compute dispatch exceeds Dawn storage-buffer binding count"};
        }

        wgpu::ShaderSourceWGSL wgsl{};
        wgsl.code = dispatch.wgsl_source.c_str();
        wgpu::ShaderModuleDescriptor module_desc{};
        module_desc.nextInChain = &wgsl;
        module_desc.label = dispatch.label.c_str();
        wgpu::ShaderModule module = device_.CreateShaderModule(&module_desc);
        if (!module) return backend_error("CreateShaderModule", dispatch.label);

        wgpu::ComputePipelineDescriptor pipeline_desc{};
        pipeline_desc.label = dispatch.label.c_str();
        pipeline_desc.compute.module = module;
        pipeline_desc.compute.entryPoint = dispatch.entry_point.c_str();
        wgpu::ComputePipeline pipeline = device_.CreateComputePipeline(&pipeline_desc);
        if (!pipeline) return backend_error("CreateComputePipeline", dispatch.label);

        wgpu::BindGroupLayout layout = pipeline.GetBindGroupLayout(0);
        if (!layout) return backend_error("GetBindGroupLayout", dispatch.label);

        std::vector<wgpu::BindGroupEntry> entries;
        entries.reserve(dispatch.bindings.size());
        std::vector<uint32_t> seen_bindings;
        seen_bindings.reserve(dispatch.bindings.size());
        for (const auto& binding : dispatch.bindings) {
            auto* buffer = dynamic_cast<DawnBuffer*>(binding.buffer.get());
            if (!buffer || binding.offset_bytes > buffer->size_bytes()) {
                return {StatusCode::InvalidArgument, "Compute dispatch contains a non-Dawn buffer or invalid offset"};
            }
            const uint64_t size = binding.size_bytes == 0
                ? buffer->size_bytes() - binding.offset_bytes : binding.size_bytes;
            const uint32_t offset_alignment = std::max(1u, limits_.min_storage_buffer_offset_alignment);
            if (size == 0 || size > buffer->size_bytes() - binding.offset_bytes ||
                (binding.offset_bytes % offset_alignment) != 0 || (size % 4) != 0 ||
                (limits_.max_storage_buffer_binding_size != 0 &&
                 size > limits_.max_storage_buffer_binding_size)) {
                return {StatusCode::InvalidArgument, "Compute dispatch buffer range exceeds Dawn alignment or storage limits"};
            }
            if (std::find(seen_bindings.begin(), seen_bindings.end(), binding.binding) != seen_bindings.end()) {
                return {StatusCode::InvalidArgument, "Compute dispatch contains a duplicate binding index"};
            }
            seen_bindings.push_back(binding.binding);
            wgpu::BindGroupEntry entry{};
            entry.binding = binding.binding;
            entry.buffer = buffer->handle();
            entry.offset = binding.offset_bytes;
            entry.size = size;
            entries.push_back(entry);
        }

        wgpu::BindGroupDescriptor bind_group_desc{};
        bind_group_desc.layout = layout;
        bind_group_desc.entryCount = static_cast<uint32_t>(entries.size());
        bind_group_desc.entries = entries.data();
        wgpu::BindGroup bind_group = device_.CreateBindGroup(&bind_group_desc);
        if (!bind_group) return backend_error("CreateBindGroup", dispatch.label);

        wgpu::ComputePassEncoder pass = encoder_.BeginComputePass();
        if (!pass) return backend_error("BeginComputePass", dispatch.label);
        pass.SetPipeline(pipeline);
        pass.SetBindGroup(0, bind_group);
        pass.DispatchWorkgroups(dispatch.workgroups_x, dispatch.workgroups_y, dispatch.workgroups_z);
        pass.End();
        return Status::success();
    }

    wgpu::CommandEncoder finish() {
        return std::move(encoder_);
    }

private:
    wgpu::Device device_;
    DeviceLimits limits_;
    wgpu::CommandEncoder encoder_;
};

class DawnBackend final : public Backend {
public:
    BackendKind kind() const noexcept override { return BackendKind::DawnWebGpu; }

    Status enumerate_devices(std::vector<DeviceInfo>& out) override {
        std::lock_guard<std::mutex> lock(mutex_);
        out.clear();
        if (!instance_) {
            wgpu::InstanceDescriptor descriptor{};
            const wgpu::InstanceFeatureName required[] = {wgpu::InstanceFeatureName::TimedWaitAny};
            descriptor.requiredFeatures = required;
            descriptor.requiredFeatureCount = 1;
            instance_ = wgpu::CreateInstance(&descriptor);
        }
        if (!instance_) return backend_error("CreateInstance", "Dawn instance creation failed");

        wgpu::RequestAdapterOptions options{};
        options.backendType = wgpu::BackendType::OpenGLES;
        options.featureLevel = wgpu::FeatureLevel::Compatibility;

        AdapterRequestState request;
        const wgpu::Future future = instance_.RequestAdapter(
            &options, wgpu::CallbackMode::AllowSpontaneous,
            [](wgpu::RequestAdapterStatus status, wgpu::Adapter adapter,
               wgpu::StringView message, AdapterRequestState* state) {
                state->status = status;
                state->message = std::string(message);
                if (status == wgpu::RequestAdapterStatus::Success) state->adapter = std::move(adapter);
            },
            &request);
        const wgpu::WaitStatus wait = instance_.WaitAny(future, std::numeric_limits<uint64_t>::max());
        if (wait != wgpu::WaitStatus::Success || request.status != wgpu::RequestAdapterStatus::Success ||
            !request.adapter) {
            return backend_error("RequestAdapter", request.message.empty() ? "No OpenGL ES WebGPU adapter" : request.message);
        }
        adapter_ = std::move(request.adapter);

        wgpu::AdapterInfo info{};
        adapter_.GetInfo(&info);
        wgpu::Limits supported{};
        const bool has_limits = adapter_.GetLimits(&supported) == wgpu::Status::Success;
        device_limits_ = DeviceLimits{};

        DeviceInfo device;
        device.backend = BackendKind::DawnWebGpu;
        device.state = CapabilityState::Discovered;
        device.name = std::string(info.device);
        device.vendor = std::string(info.vendor);
        device.architecture = std::string(info.architecture);
        device.driver = "unknown";
        device.api_version = "WebGPU/Dawn Compatibility";
        device.compute_available = true;
        device.compute_self_test_passed = false;
        device.inference_ready = false;
        if (has_limits) {
            device_limits_.max_buffer_size = supported.maxBufferSize;
            device_limits_.max_storage_buffer_binding_size = supported.maxStorageBufferBindingSize;
            device_limits_.max_compute_workgroup_size_x = supported.maxComputeWorkgroupSizeX;
            device_limits_.max_compute_workgroup_size_y = supported.maxComputeWorkgroupSizeY;
            device_limits_.max_compute_workgroup_size_z = supported.maxComputeWorkgroupSizeZ;
            device_limits_.max_compute_workgroup_invocations = supported.maxComputeInvocationsPerWorkgroup;
            device_limits_.max_compute_workgroup_storage_bytes = supported.maxComputeWorkgroupStorageSize;
            device_limits_.max_compute_workgroups_per_dimension = supported.maxComputeWorkgroupsPerDimension;
            device_limits_.max_storage_buffers_per_shader_stage = supported.maxStorageBuffersPerShaderStage;
            device_limits_.min_storage_buffer_offset_alignment = supported.minStorageBufferOffsetAlignment;
            device.limits.max_buffer_size = supported.maxBufferSize;
            max_buffer_size_ = supported.maxBufferSize;
            device.limits.max_storage_buffer_binding_size = supported.maxStorageBufferBindingSize;
            device.limits.max_compute_workgroup_size_x = supported.maxComputeWorkgroupSizeX;
            device.limits.max_compute_workgroup_size_y = supported.maxComputeWorkgroupSizeY;
            device.limits.max_compute_workgroup_size_z = supported.maxComputeWorkgroupSizeZ;
            device.limits.max_compute_workgroup_invocations = supported.maxComputeInvocationsPerWorkgroup;
            device.limits.max_compute_workgroup_storage_bytes = supported.maxComputeWorkgroupStorageSize;
            device.limits.max_compute_workgroups_per_dimension = supported.maxComputeWorkgroupsPerDimension;
            device.limits.max_storage_buffers_per_shader_stage = supported.maxStorageBuffersPerShaderStage;
            device.limits.min_storage_buffer_offset_alignment = supported.minStorageBufferOffsetAlignment;
        }
        if (adapter_.HasFeature(wgpu::FeatureName::ShaderF16)) device.features.emplace_back("shader-f16");
        if (adapter_.HasFeature(wgpu::FeatureName::Subgroups)) device.features.emplace_back("subgroups");
        if (adapter_.HasFeature(wgpu::FeatureName::TimestampQuery)) device.features.emplace_back("timestamp-query");
        device.diagnostic = "source=DawnAdapterDiscovery backend=OpenGLES featureLevel=Compatibility inferenceReady=0";
        out.push_back(std::move(device));
        return Status::success();
    }

    Status initialize(const DeviceInfo& requested) override {
        std::lock_guard<std::mutex> lock(mutex_);
        if (requested.backend != BackendKind::DawnWebGpu) {
            return {StatusCode::InvalidArgument, "Dawn backend received a device from another backend"};
        }
        if (!instance_ || !adapter_) {
            return {StatusCode::NotReady, "Dawn adapter must be enumerated before device initialization"};
        }
        if (device_) return Status::success();

        wgpu::DeviceDescriptor descriptor{};
        descriptor.SetDeviceLostCallback(
            wgpu::CallbackMode::AllowSpontaneous,
            [](const wgpu::Device&, wgpu::DeviceLostReason reason, wgpu::StringView,
               std::atomic<bool>* health) {
                if (reason != wgpu::DeviceLostReason::Destroyed) health->store(false, std::memory_order_relaxed);
            },
            health_.get());
        descriptor.SetUncapturedErrorCallback(
            [](const wgpu::Device&, wgpu::ErrorType, wgpu::StringView, std::atomic<bool>* health) {
                health->store(false, std::memory_order_relaxed);
            },
            health_.get());

        DeviceRequestState request;
        const wgpu::Future future = adapter_.RequestDevice(
            &descriptor, wgpu::CallbackMode::AllowSpontaneous,
            [](wgpu::RequestDeviceStatus status, wgpu::Device device,
               wgpu::StringView message, DeviceRequestState* state) {
                state->status = status;
                state->message = std::string(message);
                if (status == wgpu::RequestDeviceStatus::Success) state->device = std::move(device);
            },
            &request);
        const wgpu::WaitStatus wait = instance_.WaitAny(future, std::numeric_limits<uint64_t>::max());
        if (wait != wgpu::WaitStatus::Success || request.status != wgpu::RequestDeviceStatus::Success ||
            !request.device) {
            return backend_error("RequestDevice", request.message.empty() ? "Dawn device creation failed" : request.message);
        }
        device_ = std::move(request.device);
        queue_ = device_.GetQueue();
        health_->store(true, std::memory_order_relaxed);
        const Status self_test = run_compute_self_test();
        if (!self_test.ok()) {
            health_->store(false, std::memory_order_relaxed);
            device_.Destroy();
            queue_ = nullptr;
            device_ = nullptr;
            return self_test;
        }
        return Status::success();
    }

    Status allocate(const BufferDesc& desc, std::shared_ptr<Buffer>& out) override {
        std::lock_guard<std::mutex> lock(mutex_);
        out.reset();
        if (!device_ || !healthy()) return {StatusCode::DeviceLost, "Dawn device is not ready"};
        if (desc.size_bytes == 0 || desc.size_bytes > static_cast<uint64_t>(std::numeric_limits<size_t>::max()) ||
            desc.size_bytes > max_buffer_size_) {
            return {StatusCode::InvalidArgument, "Dawn buffer size is zero or exceeds the validated device limit"};
        }
        if ((desc.size_bytes % 4) != 0) {
            return {StatusCode::InvalidArgument, "Dawn buffer size must be aligned to four bytes"};
        }
        wgpu::BufferDescriptor descriptor{};
        descriptor.label = desc.label.c_str();
        descriptor.size = desc.size_bytes;
        descriptor.usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc | wgpu::BufferUsage::CopyDst;
        descriptor.mappedAtCreation = false;
        wgpu::Buffer buffer = device_.CreateBuffer(&descriptor);
        if (!buffer) return {StatusCode::OutOfMemory, "Dawn CreateBuffer returned an empty buffer"};
        out = std::make_shared<DawnBuffer>(std::move(buffer), desc.size_bytes, desc.memory);
        return Status::success();
    }

    Status begin_commands(std::unique_ptr<CommandBatch>& out) override {
        std::lock_guard<std::mutex> lock(mutex_);
        out.reset();
        if (!device_ || !healthy()) return {StatusCode::DeviceLost, "Dawn device is not ready"};
        auto batch = std::make_unique<DawnCommandBatch>(device_, device_limits_);
        if (!batch->valid()) return backend_error("CreateCommandEncoder", "Dawn returned an empty encoder");
        out = std::move(batch);
        return Status::success();
    }

    Status submit(std::unique_ptr<CommandBatch> commands) override {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!device_ || !healthy()) return {StatusCode::DeviceLost, "Dawn device is not ready"};
        auto* batch = dynamic_cast<DawnCommandBatch*>(commands.get());
        if (!batch) return {StatusCode::InvalidArgument, "Command batch belongs to another backend"};
        wgpu::CommandEncoder encoder = batch->finish();
        if (!encoder) return backend_error("FinishCommandEncoder", "Dawn encoder is empty");
        wgpu::CommandBuffer command_buffer = encoder.Finish();
        if (!command_buffer) return backend_error("FinishCommandBuffer", "Dawn command buffer is empty");
        queue_.Submit(1, &command_buffer);
        return healthy() ? Status::success() : Status{StatusCode::DeviceLost, "Dawn device was lost during submit"};
    }

    Status synchronize() override {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!device_ || !healthy()) return {StatusCode::DeviceLost, "Dawn device is not ready"};
        QueueWorkState request;
        const wgpu::Future future = queue_.OnSubmittedWorkDone(
            wgpu::CallbackMode::AllowSpontaneous,
            [](wgpu::QueueWorkDoneStatus status, wgpu::StringView message, QueueWorkState* state) {
                state->status = status;
                state->message = std::string(message);
            },
            &request);
        const wgpu::WaitStatus wait = instance_.WaitAny(future, std::numeric_limits<uint64_t>::max());
        if (wait != wgpu::WaitStatus::Success || request.status != wgpu::QueueWorkDoneStatus::Success) {
            return backend_error("QueueWorkDone", request.message.empty() ? "Dawn queue wait failed" : request.message);
        }
        return Status::success();
    }

    Status upload(const std::shared_ptr<Buffer>& destination, uint64_t destination_offset,
                  const void* source, uint64_t size_bytes) override {
        std::lock_guard<std::mutex> lock(mutex_);
        auto* buffer = dynamic_cast<DawnBuffer*>(destination.get());
        if (!device_ || !healthy()) return {StatusCode::DeviceLost, "Dawn device is not ready"};
        if (!buffer || !source || size_bytes == 0 || (destination_offset % 4) != 0 || (size_bytes % 4) != 0 ||
            destination_offset > buffer->size_bytes() || size_bytes > buffer->size_bytes() - destination_offset) {
            return {StatusCode::InvalidArgument, "Invalid Dawn upload buffer or range"};
        }
        queue_.WriteBuffer(buffer->handle(), destination_offset, source, size_bytes);
        return healthy() ? Status::success() : Status{StatusCode::BackendFailure, "Dawn upload failed"};
    }

    Status readback(const std::shared_ptr<Buffer>& source, uint64_t source_offset,
                    void* destination, uint64_t size_bytes) override {
        std::lock_guard<std::mutex> lock(mutex_);
        auto* buffer = dynamic_cast<DawnBuffer*>(source.get());
        if (!device_ || !healthy()) return {StatusCode::DeviceLost, "Dawn device is not ready"};
        if (!buffer || !destination || size_bytes == 0 || (source_offset % 4) != 0 || (size_bytes % 4) != 0 ||
            source_offset > buffer->size_bytes() || size_bytes > buffer->size_bytes() - source_offset) {
            return {StatusCode::InvalidArgument, "Invalid Dawn readback buffer or range"};
        }

        wgpu::BufferDescriptor staging_desc{};
        staging_desc.size = size_bytes;
        staging_desc.usage = wgpu::BufferUsage::CopyDst | wgpu::BufferUsage::MapRead;
        wgpu::Buffer staging = device_.CreateBuffer(&staging_desc);
        if (!staging) return {StatusCode::OutOfMemory, "Dawn readback staging allocation failed"};

        wgpu::CommandEncoder encoder = device_.CreateCommandEncoder();
        encoder.CopyBufferToBuffer(buffer->handle(), source_offset, staging, 0, size_bytes);
        wgpu::CommandBuffer command_buffer = encoder.Finish();
        if (!command_buffer) return backend_error("ReadbackCommand", "Dawn command buffer creation failed");
        queue_.Submit(1, &command_buffer);

        MapState map_request;
        const wgpu::Future future = staging.MapAsync(
            wgpu::MapMode::Read, 0, size_bytes, wgpu::CallbackMode::AllowSpontaneous,
            [](wgpu::MapAsyncStatus status, wgpu::StringView message, MapState* state) {
                state->status = status;
                state->message = std::string(message);
            },
            &map_request);
        const wgpu::WaitStatus wait = instance_.WaitAny(future, std::numeric_limits<uint64_t>::max());
        if (wait != wgpu::WaitStatus::Success || map_request.status != wgpu::MapAsyncStatus::Success) {
            return backend_error("MapReadback", map_request.message.empty() ? "Dawn staging map failed" : map_request.message);
        }
        const void* mapped = staging.GetConstMappedRange(0, size_bytes);
        if (!mapped) {
            staging.Unmap();
            return backend_error("MapReadback", "Dawn returned a null mapped range");
        }
        std::memcpy(destination, mapped, static_cast<size_t>(size_bytes));
        staging.Unmap();
        return healthy() ? Status::success() : Status{StatusCode::DeviceLost, "Dawn device was lost during readback"};
    }

    Status shutdown() override {
        std::lock_guard<std::mutex> lock(mutex_);
        if (device_) device_.Destroy();
        queue_ = nullptr;
        device_ = nullptr;
        adapter_ = nullptr;
        instance_ = nullptr;
        health_->store(false, std::memory_order_relaxed);
        return Status::success();
    }

    bool healthy() const noexcept override {
        return health_->load(std::memory_order_relaxed);
    }

private:
    Status run_compute_self_test() {
        constexpr uint32_t input[] = {1u, 2u, 3u, 4u};
        constexpr uint32_t expected[] = {10u, 13u, 16u, 19u};
        constexpr uint64_t byte_size = sizeof(input);

        wgpu::BufferDescriptor data_desc{};
        data_desc.size = byte_size;
        data_desc.usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc |
                          wgpu::BufferUsage::CopyDst;
        wgpu::Buffer data = device_.CreateBuffer(&data_desc);
        if (!data) return backend_error("DawnSelfTest", "storage buffer allocation failed");
        queue_.WriteBuffer(data, 0, input, byte_size);

        static constexpr char shader[] = R"wgsl(
@group(0) @binding(0) var<storage, read_write> values: array<u32>;
@compute @workgroup_size(4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    values[gid.x] = values[gid.x] * 3u + 7u;
}
)wgsl";
        wgpu::ShaderSourceWGSL wgsl{};
        wgsl.code = shader;
        wgpu::ShaderModuleDescriptor module_desc{};
        module_desc.nextInChain = &wgsl;
        wgpu::ShaderModule module = device_.CreateShaderModule(&module_desc);
        if (!module) return backend_error("DawnSelfTest", "WGSL shader compilation failed");

        wgpu::ComputePipelineDescriptor pipeline_desc{};
        pipeline_desc.compute.module = module;
        pipeline_desc.compute.entryPoint = "main";
        pipeline_desc.layout = nullptr;
        wgpu::ComputePipeline pipeline = device_.CreateComputePipeline(&pipeline_desc);
        if (!pipeline) return backend_error("DawnSelfTest", "compute pipeline creation failed");

        wgpu::BindGroupLayout layout = pipeline.GetBindGroupLayout(0);
        wgpu::BindGroupEntry entry{};
        entry.binding = 0;
        entry.buffer = data;
        entry.offset = 0;
        entry.size = byte_size;
        wgpu::BindGroupDescriptor bind_desc{};
        bind_desc.layout = layout;
        bind_desc.entryCount = 1;
        bind_desc.entries = &entry;
        wgpu::BindGroup bind_group = device_.CreateBindGroup(&bind_desc);
        if (!bind_group) return backend_error("DawnSelfTest", "bind group creation failed");

        wgpu::BufferDescriptor staging_desc{};
        staging_desc.size = byte_size;
        staging_desc.usage = wgpu::BufferUsage::CopyDst | wgpu::BufferUsage::MapRead;
        wgpu::Buffer staging = device_.CreateBuffer(&staging_desc);
        if (!staging) return {StatusCode::OutOfMemory, "Dawn self-test readback allocation failed"};

        wgpu::CommandEncoder encoder = device_.CreateCommandEncoder();
        wgpu::ComputePassEncoder pass = encoder.BeginComputePass();
        if (!pass) return backend_error("DawnSelfTest", "compute pass creation failed");
        pass.SetPipeline(pipeline);
        pass.SetBindGroup(0, bind_group);
        pass.DispatchWorkgroups(1, 1, 1);
        pass.End();
        encoder.CopyBufferToBuffer(data, 0, staging, 0, byte_size);
        wgpu::CommandBuffer commands = encoder.Finish();
        if (!commands) return backend_error("DawnSelfTest", "command buffer creation failed");
        queue_.Submit(1, &commands);

        MapState map_request;
        const wgpu::Future future = staging.MapAsync(
            wgpu::MapMode::Read, 0, byte_size, wgpu::CallbackMode::AllowSpontaneous,
            [](wgpu::MapAsyncStatus status, wgpu::StringView message, MapState* state) {
                state->status = status;
                state->message = std::string(message);
            },
            &map_request);
        const wgpu::WaitStatus wait = instance_.WaitAny(future, std::numeric_limits<uint64_t>::max());
        if (wait != wgpu::WaitStatus::Success || map_request.status != wgpu::MapAsyncStatus::Success) {
            return backend_error("DawnSelfTest", map_request.message.empty() ? "readback map failed" : map_request.message);
        }
        const auto* output = static_cast<const uint32_t*>(staging.GetConstMappedRange(0, byte_size));
        if (!output) {
            staging.Unmap();
            return backend_error("DawnSelfTest", "mapped pointer is null");
        }
        const bool matches = std::equal(std::begin(expected), std::end(expected), output);
        staging.Unmap();
        if (!matches) return {StatusCode::NumericalFailure, "Dawn compute self-test output mismatch"};
        if (!healthy()) return {StatusCode::DeviceLost, "Dawn device reported an error during compute self-test"};
        return Status::success();
    }

    mutable std::mutex mutex_;
    wgpu::Instance instance_;
    wgpu::Adapter adapter_;
    wgpu::Device device_;
    wgpu::Queue queue_;
    std::shared_ptr<std::atomic<bool>> health_ = std::make_shared<std::atomic<bool>>(false);
    uint64_t max_buffer_size_ = std::numeric_limits<uint64_t>::max();
    DeviceLimits device_limits_;
};

} // namespace

std::shared_ptr<Backend> create_dawn_backend() {
    return std::make_shared<DawnBackend>();
}

} // namespace ai_chat::gpu_engine
