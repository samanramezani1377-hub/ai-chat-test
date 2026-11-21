#include "gpu_engine.h"
#include "../universal_gpu_adapter.h"
#include "universal_adapter_bridge.h"
#include <GLES3/gl31.h>
#include <algorithm>
#include <atomic>
#include <cstring>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <unordered_map>
#include <vector>

namespace ai_chat::gpu_engine {
namespace {
Status gl_error(const char* op, GLenum code) {
    return {code == GL_CONTEXT_LOST ? StatusCode::DeviceLost : StatusCode::BackendFailure, std::string(op) + " failed (GLES error " + std::to_string(code) + ")"};
}
void clear_errors() { while (glGetError() != GL_NO_ERROR) {} }

class GlesBuffer final : public Buffer {
public:
    GlesBuffer(GLuint id, uint64_t size, MemoryDomain domain) : id_(id), size_(size), domain_(domain) {}
    ~GlesBuffer() override {
        const GLuint id = id_;
        if (id) { std::string ignored; ai_chat::gpu::post_universal_gpu_task([id]() { glDeleteBuffers(1, &id); }, ignored); }
    }
    uint64_t size_bytes() const noexcept override { return size_; }
    MemoryDomain memory_domain() const noexcept override { return domain_; }
    GLuint id() const noexcept { return id_; }
private:
    GLuint id_ = 0;
    uint64_t size_ = 0;
    MemoryDomain domain_ = MemoryDomain::Device;
};

struct GlesCommand {
    enum class Type { Copy, Dispatch } type = Type::Copy;
    std::shared_ptr<Buffer> source, destination;
    uint64_t source_offset = 0, destination_offset = 0, size = 0;
    ComputeDispatch dispatch;
};

class GlesBatch final : public CommandBatch {
public:
    Status copy_buffer(const std::shared_ptr<Buffer>& src, uint64_t so,
                       const std::shared_ptr<Buffer>& dst, uint64_t dso, uint64_t size) override {
        auto* s = dynamic_cast<GlesBuffer*>(src.get());
        auto* d = dynamic_cast<GlesBuffer*>(dst.get());
        if (!s || !d || !size || so > s->size_bytes() || size > s->size_bytes()-so ||
            dso > d->size_bytes() || size > d->size_bytes()-dso || ((so|dso|size)&3u))
            return {StatusCode::InvalidArgument, "Invalid GLES buffer copy range"};
        GlesCommand c; c.type=GlesCommand::Type::Copy; c.source=src; c.destination=dst;
        c.source_offset=so; c.destination_offset=dso; c.size=size; commands_.push_back(std::move(c));
        return Status::success();
    }
    Status dispatch_compute(const ComputeDispatch& d) override {
        if (d.validation != KernelValidation::Validated || d.kernel_id.empty() ||
            d.backend != BackendKind::OpenGles)
            return {StatusCode::Unsupported, "GLES dispatch requires a validated GLES kernel identity"};
        if (d.glsl_source.empty() || d.bindings.empty() || !d.workgroups_x || !d.workgroups_y || !d.workgroups_z)
            return {StatusCode::InvalidArgument, "GLES dispatch requires GLSL ES source, bindings and non-zero workgroups"};
        GlesCommand c; c.type=GlesCommand::Type::Dispatch; c.dispatch=d;
        commands_.push_back(std::move(c)); return Status::success();
    }
    std::vector<GlesCommand> take() { return std::move(commands_); }
private:
    std::vector<GlesCommand> commands_;
};

Status compile_program(const std::string& source, GLuint& program) {
    program=0;
    const char* text=source.c_str();
    GLuint shader=glCreateShader(GL_COMPUTE_SHADER);
    if (!shader) return {StatusCode::BackendFailure,"glCreateShader returned zero"};
    glShaderSource(shader,1,&text,nullptr); glCompileShader(shader);
    GLint ok=GL_FALSE; glGetShaderiv(shader,GL_COMPILE_STATUS,&ok);
    if (ok!=GL_TRUE) {
        GLint n=0; glGetShaderiv(shader,GL_INFO_LOG_LENGTH,&n);
        std::string log(static_cast<size_t>(std::max(1,n)),'\0'); GLsizei written=0;
        glGetShaderInfoLog(shader,static_cast<GLsizei>(log.size()),&written,log.data());
        glDeleteShader(shader);
        return {StatusCode::BackendFailure,"GLES shader compile failed: "+log.substr(0,static_cast<size_t>(std::max<GLsizei>(0,written)))};
    }
    program=glCreateProgram();
    if (!program) { glDeleteShader(shader); return {StatusCode::BackendFailure,"glCreateProgram returned zero"}; }
    glAttachShader(program,shader); glLinkProgram(program); glDeleteShader(shader);
    glGetProgramiv(program,GL_LINK_STATUS,&ok);
    if (ok!=GL_TRUE) {
        GLint n=0; glGetProgramiv(program,GL_INFO_LOG_LENGTH,&n);
        std::string log(static_cast<size_t>(std::max(1,n)),'\0'); GLsizei written=0;
        glGetProgramInfoLog(program,static_cast<GLsizei>(log.size()),&written,log.data());
        glDeleteProgram(program); program=0;
        return {StatusCode::BackendFailure,"GLES program link failed: "+log.substr(0,static_cast<size_t>(std::max<GLsizei>(0,written)))};
    }
    return Status::success();
}

Status dispatch_gles(const ComputeDispatch& d, const ai_chat::gpu::UniversalGpuCapabilities& caps,
                     std::unordered_map<std::string, GLuint>* program_cache = nullptr) {
    if (d.backend != BackendKind::OpenGles || d.validation != KernelValidation::Validated ||
        d.kernel_id.empty() || d.glsl_source.empty() || !d.workgroups_x || !d.workgroups_y || !d.workgroups_z)
        return {StatusCode::Unsupported, "GLES dispatch requires a validated GLSL kernel and non-zero dispatch dimensions"};
    if (!d.workgroup_size_x || !d.workgroup_size_y || !d.workgroup_size_z ||
        d.workgroup_size_x > static_cast<uint32_t>(std::max(0, caps.max_compute_workgroup_size_x)) ||
        d.workgroup_size_y > static_cast<uint32_t>(std::max(0, caps.max_compute_workgroup_size_y)) ||
        d.workgroup_size_z > static_cast<uint32_t>(std::max(0, caps.max_compute_workgroup_size_z)) ||
        d.workgroups_x > static_cast<uint32_t>(std::max(0, caps.max_compute_workgroup_count_x)) ||
        d.workgroups_y > static_cast<uint32_t>(std::max(0, caps.max_compute_workgroup_count_y)) ||
        d.workgroups_z > static_cast<uint32_t>(std::max(0, caps.max_compute_workgroup_count_z)) ||
        d.bindings.empty() || d.bindings.size() > static_cast<size_t>(std::max(0, caps.max_shader_storage_buffer_bindings)))
        return {StatusCode::Unsupported, "GLES dispatch exceeds discovered workgroup or SSBO limits"};
    const uint64_t x = d.workgroup_size_x, y = d.workgroup_size_y, z = d.workgroup_size_z;
    if (x > UINT64_MAX / y || x * y > UINT64_MAX / z ||
        x * y * z > static_cast<uint64_t>(std::max(0, caps.max_compute_invocations)))
        return {StatusCode::Unsupported, "GLES local workgroup invocation count is invalid"};
    std::vector<uint32_t> used_bindings;
    for (const auto& binding : d.bindings) {
        if (binding.binding >= static_cast<uint32_t>(std::max(0, caps.max_shader_storage_buffer_bindings)) ||
            std::find(used_bindings.begin(), used_bindings.end(), binding.binding) != used_bindings.end())
            return {StatusCode::InvalidArgument, "GLES storage binding index is duplicated or out of range"};
        used_bindings.push_back(binding.binding);
        auto* buffer = dynamic_cast<GlesBuffer*>(binding.buffer.get());
        if (!buffer || binding.offset_bytes > buffer->size_bytes())
            return {StatusCode::InvalidArgument, "GLES dispatch references a foreign buffer or invalid offset"};
        const uint64_t range = binding.size_bytes ? binding.size_bytes : buffer->size_bytes() - binding.offset_bytes;
        const uint64_t alignment = static_cast<uint64_t>(std::max(4, caps.min_shader_storage_buffer_offset_alignment));
        if (!range || range > buffer->size_bytes() - binding.offset_bytes || binding.offset_bytes % alignment ||
            range > static_cast<uint64_t>(std::max<int64_t>(0, caps.max_shader_storage_block_bytes)))
            return {StatusCode::InvalidArgument, "GLES SSBO range violates alignment or size limits"};
    }
    GLuint program = 0;
    bool cache_hit = false;
    const std::string cache_key = d.kernel_id + "\n" + d.glsl_source;
    if (program_cache) {
        const auto found = program_cache->find(cache_key);
        if (found != program_cache->end()) { program = found->second; cache_hit = true; }
    }
    if (!cache_hit) {
        Status compile_status = compile_program(d.glsl_source, program);
        if (!compile_status.ok()) return compile_status;
        if (program_cache) {
            try { program_cache->emplace(cache_key, program); cache_hit = true; }
            catch (...) { /* Cache failure must not prevent inference. */ }
        }
    }
    clear_errors();
    glUseProgram(program);
    for (const auto& binding : d.bindings) {
        auto* buffer = static_cast<GlesBuffer*>(binding.buffer.get());
        const uint64_t range = binding.size_bytes ? binding.size_bytes : buffer->size_bytes() - binding.offset_bytes;
        glBindBufferRange(GL_SHADER_STORAGE_BUFFER, binding.binding, buffer->id(),
                          static_cast<GLintptr>(binding.offset_bytes), static_cast<GLsizeiptr>(range));
    }
    glDispatchCompute(d.workgroups_x, d.workgroups_y, d.workgroups_z);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT | GL_BUFFER_UPDATE_BARRIER_BIT);
    const GLenum error = glGetError();
    glUseProgram(0);
    for (const auto& binding : d.bindings) glBindBufferBase(GL_SHADER_STORAGE_BUFFER, binding.binding, 0);
    if (!cache_hit) glDeleteProgram(program);
    return error == GL_NO_ERROR ? Status::success() : gl_error("glDispatchCompute", error);
}
Status self_test_gles_backend(const ai_chat::gpu::UniversalGpuCapabilities& caps) {
    constexpr uint32_t input[4] = {1u, 2u, 3u, 4u};
    constexpr uint32_t expected[4] = {10u, 13u, 16u, 19u};
    GLuint id = 0;
    clear_errors();
    glGenBuffers(1, &id);
    if (!id) return {StatusCode::OutOfMemory, "GLES backend self-test buffer allocation failed"};
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, id);
    glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(input), input, GL_DYNAMIC_COPY);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
    const GLenum allocation_error = glGetError();
    Status status = allocation_error == GL_NO_ERROR ? Status::success() : gl_error("self-test buffer allocation", allocation_error);
    if (!status.ok()) { glDeleteBuffers(1, &id); return status; }
    auto buffer = std::make_shared<GlesBuffer>(id, sizeof(input), MemoryDomain::Device);
    ComputeDispatch dispatch;
    dispatch.label = "universal-gles-backend-self-test";
    dispatch.kernel_id = "internal-gles-u32-transform-v1";
    dispatch.backend = BackendKind::OpenGles;
    dispatch.validation = KernelValidation::Validated;
    dispatch.glsl_source = "#version 310 es\n"
        "layout(local_size_x=4) in;\n"
        "layout(std430,binding=0) buffer Data { uint values[]; };\n"
        "void main(){ uint i=gl_GlobalInvocationID.x; values[i]=values[i]*3u+7u; }";
    dispatch.bindings.push_back({0, buffer, 0, sizeof(input)});
    dispatch.workgroups_x = 1;
    dispatch.workgroup_size_x = 4;
    status = dispatch_gles(dispatch, caps);
    if (!status.ok()) return status;
    glFinish();
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, id);
    void* mapped = glMapBufferRange(GL_SHADER_STORAGE_BUFFER, 0, sizeof(input), GL_MAP_READ_BIT);
    if (!mapped) { glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0); return {StatusCode::BackendFailure, "GLES backend self-test readback map failed"}; }
    uint32_t output[4] = {};
    std::memcpy(output, mapped, sizeof(output));
    const GLboolean unmapped = glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
    if (unmapped != GL_TRUE || std::memcmp(output, expected, sizeof(expected)) != 0)
        return {StatusCode::NumericalFailure, "GLES backend self-test output mismatch"};
    return gl_status("GLES backend self-test");
}

class UniversalGlesBackend final : public Backend {
public:
    ~UniversalGlesBackend() override { shutdown(); }
    BackendKind kind() const noexcept override { return BackendKind::OpenGles; }
    Status enumerate_devices(std::vector<DeviceInfo>& out) override {
        out.clear();
        caps_ = ai_chat::gpu::probe_universal_gpu_adapter();
        if (!caps_.egl_context_created || !caps_.gles_compute_supported || !caps_.gles_compute_self_test_passed)
            return {StatusCode::Unsupported, caps_.failure_reason.empty() ? "Universal GPU Adapter GLES self-test failed" : caps_.failure_reason};
        DeviceInfo device = device_info_from_universal_adapter(caps_);
        out.push_back(std::move(device));
        return Status::success();
    }
    Status initialize(const DeviceInfo& d) override {
        if(d.backend!=BackendKind::OpenGles) return {StatusCode::InvalidArgument,"GLES backend received foreign device"};
        const auto c=ai_chat::gpu::probe_universal_gpu_adapter();
        if(!c.egl_context_created||!c.gles_compute_supported||!c.gles_compute_self_test_passed) { healthy_=false; return {StatusCode::DeviceLost,"Universal GLES context/self-test unavailable"}; }
        caps_=c;
        Status self_test = Status::success();
        std::string task_error;
        const bool ran = ai_chat::gpu::run_universal_gpu_task([&]() { self_test = self_test_gles_backend(caps_); }, task_error);
        if (!ran) { healthy_=false; return {StatusCode::BackendFailure, task_error}; }
        if (!self_test.ok()) { healthy_=false; return self_test; }
        healthy_=true;
        return Status::success();
    }
    Status allocate(const BufferDesc& d,std::shared_ptr<Buffer>& out) override {
        out.reset();
        if(!healthy()) return {StatusCode::DeviceLost,"GLES backend is not initialized"};
        if(d.memory==MemoryDomain::Host||!d.size_bytes||d.size_bytes%4||
           d.size_bytes>static_cast<uint64_t>(std::max<int64_t>(0,caps_.max_shader_storage_block_bytes))||
           d.size_bytes>static_cast<uint64_t>(std::numeric_limits<GLsizeiptr>::max()))
            return {StatusCode::InvalidArgument,"GLES buffer size/domain is unsupported or exceeds SSBO limit"};
        GLuint id=0; GLenum error=GL_NO_ERROR; std::string detail;
        const bool ran=ai_chat::gpu::run_universal_gpu_task([&](){
            clear_errors(); glGenBuffers(1,&id); glBindBuffer(GL_SHADER_STORAGE_BUFFER,id);
            glBufferData(GL_SHADER_STORAGE_BUFFER,static_cast<GLsizeiptr>(d.size_bytes),nullptr,d.persistent?GL_STATIC_DRAW:GL_DYNAMIC_DRAW);
            glBindBuffer(GL_SHADER_STORAGE_BUFFER,0); error=glGetError();
            if(error!=GL_NO_ERROR&&id){glDeleteBuffers(1,&id);id=0;}
        },detail);
        if(!ran) return {StatusCode::DeviceLost,detail};
        if(error!=GL_NO_ERROR) return gl_error("GLES buffer allocation",error);
        if(!id) return {StatusCode::BackendFailure,"GLES buffer allocation returned no buffer without a GL error"};
        out=std::make_shared<GlesBuffer>(id,d.size_bytes,d.memory); return Status::success();
    }
    Status begin_commands(std::unique_ptr<CommandBatch>& out) override {
        out.reset(); if(!healthy()) return {StatusCode::DeviceLost,"GLES backend is not initialized"};
        out=std::make_unique<GlesBatch>(); return Status::success();
    }
    Status submit(std::unique_ptr<CommandBatch> commands) override {
        auto* batch=dynamic_cast<GlesBatch*>(commands.get());
        if(!batch) return {StatusCode::InvalidArgument,"Command batch belongs to another backend"};
        if(!healthy()) return {StatusCode::DeviceLost,"GLES backend is not initialized"};
        auto pending=batch->take(); Status result=Status::success(); std::string detail;
        const bool ran=ai_chat::gpu::run_universal_gpu_task([&](){
            for(const auto& c:pending) {
                if(!result.ok()) break; clear_errors();
                if(c.type==GlesCommand::Type::Copy) {
                    auto* src=dynamic_cast<GlesBuffer*>(c.source.get()); auto* dst=dynamic_cast<GlesBuffer*>(c.destination.get());
                    if(!src||!dst){result={StatusCode::InvalidArgument,"GLES copy references foreign buffer"};break;}
                    glBindBuffer(GL_COPY_READ_BUFFER,src->id()); glBindBuffer(GL_COPY_WRITE_BUFFER,dst->id());
                    glCopyBufferSubData(GL_COPY_READ_BUFFER,GL_COPY_WRITE_BUFFER,static_cast<GLintptr>(c.source_offset),static_cast<GLintptr>(c.destination_offset),static_cast<GLsizeiptr>(c.size));
                    glBindBuffer(GL_COPY_READ_BUFFER,0); glBindBuffer(GL_COPY_WRITE_BUFFER,0);
                    const GLenum e=glGetError(); if(e!=GL_NO_ERROR) result=gl_error("glCopyBufferSubData",e);
                } else result=dispatch_gles(c.dispatch,caps_,&program_cache_);
            }
            glFlush();
        },detail);
        if(!ran){healthy_=false;return {StatusCode::DeviceLost,detail};}
        return result;
    }
    Status synchronize() override {
        if(!healthy()) return {StatusCode::DeviceLost,"GLES backend is not initialized"};
        GLenum e=GL_NO_ERROR; std::string detail;
        const bool ran=ai_chat::gpu::run_universal_gpu_task([&](){glFinish();e=glGetError();},detail);
        if(!ran){healthy_=false;return {StatusCode::DeviceLost,detail};}
        return e==GL_NO_ERROR?Status::success():gl_error("glFinish",e);
    }
    Status upload(const std::shared_ptr<Buffer>& dst,uint64_t offset,const void* src,uint64_t size) override {
        auto* b=dynamic_cast<GlesBuffer*>(dst.get());
        if(!b||!src||!size||offset>b->size_bytes()||size>b->size_bytes()-offset||
           offset>static_cast<uint64_t>(std::numeric_limits<GLintptr>::max())||size>static_cast<uint64_t>(std::numeric_limits<GLsizeiptr>::max()))
            return {StatusCode::InvalidArgument,"Invalid GLES upload range"};
        GLenum e=GL_NO_ERROR; std::string detail;
        const bool ran=ai_chat::gpu::run_universal_gpu_task([&](){clear_errors();glBindBuffer(GL_SHADER_STORAGE_BUFFER,b->id());glBufferSubData(GL_SHADER_STORAGE_BUFFER,static_cast<GLintptr>(offset),static_cast<GLsizeiptr>(size),src);glBindBuffer(GL_SHADER_STORAGE_BUFFER,0);e=glGetError();},detail);
        if(!ran)return {StatusCode::DeviceLost,detail}; return e==GL_NO_ERROR?Status::success():gl_error("GLES upload",e);
    }
    Status readback(const std::shared_ptr<Buffer>& src,uint64_t offset,void* dst,uint64_t size) override {
        auto* b=dynamic_cast<GlesBuffer*>(src.get());
        if(!b||!dst||!size||offset>b->size_bytes()||size>b->size_bytes()-offset||
           offset>static_cast<uint64_t>(std::numeric_limits<GLintptr>::max())||size>static_cast<uint64_t>(std::numeric_limits<GLsizeiptr>::max()))
            return {StatusCode::InvalidArgument,"Invalid GLES readback range"};
        GLenum e=GL_NO_ERROR; std::string detail;
        const bool ran=ai_chat::gpu::run_universal_gpu_task([&](){
            glFinish();clear_errors();glBindBuffer(GL_SHADER_STORAGE_BUFFER,b->id());
            void* mapped=glMapBufferRange(GL_SHADER_STORAGE_BUFFER,static_cast<GLintptr>(offset),static_cast<GLsizeiptr>(size),GL_MAP_READ_BIT);
            if(!mapped){e=glGetError();if(e==GL_NO_ERROR)e=GL_INVALID_OPERATION;}
            else {std::memcpy(dst,mapped,static_cast<size_t>(size));const GLboolean ok=glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);e=ok==GL_TRUE?glGetError():GL_INVALID_OPERATION;}
            glBindBuffer(GL_SHADER_STORAGE_BUFFER,0);
        },detail);
        if(!ran)return {StatusCode::DeviceLost,detail}; return e==GL_NO_ERROR?Status::success():gl_error("GLES readback",e);
    }
    Status shutdown() override {
        healthy_ = false;
        std::string detail;
        const bool ran = ai_chat::gpu::run_universal_gpu_task([&]() {
            for (const auto& entry : program_cache_) if (entry.second) glDeleteProgram(entry.second);
            program_cache_.clear();
        }, detail);
        if (!ran) return {StatusCode::DeviceLost, detail};
        return Status::success();
    }
    bool healthy() const noexcept override { return healthy_.load(std::memory_order_acquire); }
private:
    ai_chat::gpu::UniversalGpuCapabilities caps_;
    std::unordered_map<std::string, GLuint> program_cache_;
    std::atomic<bool> healthy_{false};
};
} // namespace

std::shared_ptr<Backend> create_universal_gles_backend() { return std::make_shared<UniversalGlesBackend>(); }
} // namespace ai_chat::gpu_engine
