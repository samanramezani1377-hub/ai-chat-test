#include "operators.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <utility>

namespace ai_chat::gpu_engine {
namespace {

struct TensorFixture {
    Tensor tensor;
    std::shared_ptr<Buffer> buffer;
};

Status make_f32_tensor(Backend& backend, const std::vector<uint64_t>& shape,
                       const std::vector<float>& values, const char* label,
                       TensorFixture& out) {
    uint64_t elements = 1;
    for (uint64_t dimension : shape) {
        if (dimension == 0 || elements > UINT64_MAX / dimension)
            return {StatusCode::InvalidArgument, "Kernel validation tensor shape is invalid"};
        elements *= dimension;
    }
    if (!values.empty() && values.size() != elements)
        return {StatusCode::InvalidArgument, "Kernel validation input data does not match its shape"};
    if (elements > UINT64_MAX / sizeof(float))
        return {StatusCode::InvalidArgument, "Kernel validation tensor byte size overflows"};

    BufferDesc buffer_desc;
    buffer_desc.size_bytes = elements * sizeof(float);
    buffer_desc.alignment = alignof(float);
    buffer_desc.memory = MemoryDomain::Device;
    buffer_desc.persistent = false;
    buffer_desc.label = label;
    Status status = backend.allocate(buffer_desc, out.buffer);
    if (!status.ok()) return status;
    if (!out.buffer || out.buffer->size_bytes() < buffer_desc.size_bytes)
        return {StatusCode::BackendFailure, "Backend returned an undersized validation buffer"};
    if (!values.empty()) {
        status = backend.upload(out.buffer, 0, values.data(), buffer_desc.size_bytes);
        if (!status.ok()) return status;
    }

    TensorDesc tensor_desc;
    tensor_desc.shape = shape;
    tensor_desc.data_type = DataType::F32;
    tensor_desc.layout = TensorLayout::Contiguous;
    tensor_desc.memory = MemoryDomain::Device;
    tensor_desc.alignment = alignof(float);
    return Tensor::create(std::move(tensor_desc), out.buffer, 0, out.tensor);
}

Status execute_and_compare(Backend& backend, const ComputeDispatch& dispatch,
                           const Tensor& output, const std::vector<float>& expected) {
    if (expected.size() != output.element_count() || !output.buffer())
        return {StatusCode::InvalidArgument, "Kernel validation output shape mismatch"};
    std::unique_ptr<CommandBatch> commands;
    Status status = backend.begin_commands(commands);
    if (!status.ok()) return status;
    if (!commands) return {StatusCode::BackendFailure, "Backend returned an empty validation command batch"};
    // Candidate kernels are intentionally Unverified until this exact GPU
    // execution/readback succeeds. The backend requires a validated dispatch
    // token even during the isolated validation run; do not mutate the registry
    // here. Promotion happens only in run_one() after numerical parity passes.
    ComputeDispatch validation_dispatch = dispatch;
    if (validation_dispatch.backend != backend.kind() || validation_dispatch.kernel_id.empty())
        return {StatusCode::InvalidArgument, "Kernel validation dispatch identity does not match the selected backend"};
    validation_dispatch.validation = KernelValidation::Validated;
    status = commands->dispatch_compute(validation_dispatch);
    if (!status.ok()) return status;
    status = backend.submit(std::move(commands));
    if (!status.ok()) return status;
    if (!backend.healthy()) return {StatusCode::DeviceLost, "Backend was lost during kernel validation"};
    status = backend.synchronize();
    if (!status.ok()) return status;

    std::vector<float> actual(expected.size());
    status = backend.readback(output.buffer(), output.byte_offset(),
                               actual.data(), actual.size() * sizeof(float));
    if (!status.ok()) return status;
    constexpr float kAbsoluteTolerance = 2.0e-4f;
    constexpr float kRelativeTolerance = 2.0e-4f;
    for (size_t index = 0; index < actual.size(); ++index) {
        if (!std::isfinite(actual[index]) || !std::isfinite(expected[index]))
            return {StatusCode::NumericalFailure, "Kernel validation produced a non-finite value"};
        const float tolerance = kAbsoluteTolerance +
            kRelativeTolerance * std::max(std::fabs(expected[index]), 1.0f);
        if (std::fabs(actual[index] - expected[index]) > tolerance)
            return {StatusCode::NumericalFailure, "Kernel output differs from the CPU reference"};
    }
    return Status::success();
}

template <typename Validate>
Status validate_add(Backend& backend, Validate&& build) {
    TensorFixture lhs, rhs, output;
    const std::vector<float> a = {1.0f, -2.0f, 0.5f, 4.0f, 3.0f, -1.0f, 2.0f, 0.25f};
    const std::vector<float> b = {0.5f, 3.0f, -4.0f, 0.25f, -2.0f, 0.5f, 1.5f, 8.0f};
    Status status = make_f32_tensor(backend, {2, 4}, a, "validate-add-a", lhs);
    if (!status.ok()) return status;
    status = make_f32_tensor(backend, {2, 4}, b, "validate-add-b", rhs);
    if (!status.ok()) return status;
    status = make_f32_tensor(backend, {2, 4}, {}, "validate-add-out", output);
    if (!status.ok()) return status;
    std::vector<float> expected(a.size());
    status = reference_add_f32(a.data(), b.data(), expected.data(), a.size());
    if (!status.ok()) return status;
    ComputeDispatch dispatch;
    status = build(lhs.tensor, rhs.tensor, output.tensor, dispatch);
    if (!status.ok()) return status;
    status = execute_and_compare(backend, dispatch, output.tensor, expected);
    if (!status.ok()) return status;

    TensorFixture lhs2, rhs2, output2;
    std::vector<float> a2(129), b2(129);
    for (size_t i = 0; i < a2.size(); ++i) {
        a2[i] = static_cast<float>(static_cast<int>(i % 17) - 8) * 0.125f;
        b2[i] = static_cast<float>(static_cast<int>(i % 11) - 5) * 0.2f;
    }
    status = make_f32_tensor(backend, {1, 129}, a2, "validate-add-a-odd", lhs2);
    if (!status.ok()) return status;
    status = make_f32_tensor(backend, {1, 129}, b2, "validate-add-b-odd", rhs2);
    if (!status.ok()) return status;
    status = make_f32_tensor(backend, {1, 129}, {}, "validate-add-out-odd", output2);
    if (!status.ok()) return status;
    std::vector<float> expected2(a2.size());
    status = reference_add_f32(a2.data(), b2.data(), expected2.data(), a2.size());
    if (!status.ok()) return status;
    status = build(lhs2.tensor, rhs2.tensor, output2.tensor, dispatch);
    if (!status.ok()) return status;
    return execute_and_compare(backend, dispatch, output2.tensor, expected2);
}

template <typename Validate>
Status validate_multiply(Backend& backend, Validate&& build) {
    TensorFixture lhs, rhs, output;
    const std::vector<float> a = {1.0f, -2.0f, 0.5f, 4.0f, 3.0f, -1.0f, 2.0f, 0.25f};
    const std::vector<float> b = {0.5f, 3.0f, -4.0f, 0.25f, -2.0f, 0.5f, 1.5f, 8.0f};
    Status status = make_f32_tensor(backend, {2, 4}, a, "validate-multiply-a", lhs);
    if (!status.ok()) return status;
    status = make_f32_tensor(backend, {2, 4}, b, "validate-multiply-b", rhs);
    if (!status.ok()) return status;
    status = make_f32_tensor(backend, {2, 4}, {}, "validate-multiply-out", output);
    if (!status.ok()) return status;
    std::vector<float> expected(a.size());
    status = reference_multiply_f32(a.data(), b.data(), expected.data(), a.size());
    if (!status.ok()) return status;
    ComputeDispatch dispatch;
    status = build(lhs.tensor, rhs.tensor, output.tensor, dispatch);
    if (!status.ok()) return status;
    status = execute_and_compare(backend, dispatch, output.tensor, expected);
    if (!status.ok()) return status;

    TensorFixture lhs2, rhs2, output2;
    std::vector<float> a2(129), b2(129);
    for (size_t i = 0; i < a2.size(); ++i) {
        a2[i] = static_cast<float>(static_cast<int>(i % 17) - 8) * 0.125f;
        b2[i] = static_cast<float>(static_cast<int>(i % 11) - 5) * 0.2f;
    }
    status = make_f32_tensor(backend, {1, 129}, a2, "validate-multiply-a-odd", lhs2);
    if (!status.ok()) return status;
    status = make_f32_tensor(backend, {1, 129}, b2, "validate-multiply-b-odd", rhs2);
    if (!status.ok()) return status;
    status = make_f32_tensor(backend, {1, 129}, {}, "validate-multiply-out-odd", output2);
    if (!status.ok()) return status;
    std::vector<float> expected2(a2.size());
    status = reference_multiply_f32(a2.data(), b2.data(), expected2.data(), a2.size());
    if (!status.ok()) return status;
    status = build(lhs2.tensor, rhs2.tensor, output2.tensor, dispatch);
    if (!status.ok()) return status;
    return execute_and_compare(backend, dispatch, output2.tensor, expected2);
}

template <typename Validate>
Status validate_matmul(Backend& backend, Validate&& build) {
    TensorFixture lhs, rhs, output;
    const std::vector<float> a = {1.0f, 2.0f, -1.0f, 0.5f, -2.0f, 3.0f};
    const std::vector<float> b = {2.0f, -1.0f, 0.25f, 3.0f, -2.0f, 0.5f};
    Status status = make_f32_tensor(backend, {2, 3}, a, "validate-matmul-a", lhs);
    if (!status.ok()) return status;
    status = make_f32_tensor(backend, {3, 2}, b, "validate-matmul-b", rhs);
    if (!status.ok()) return status;
    status = make_f32_tensor(backend, {2, 2}, {}, "validate-matmul-out", output);
    if (!status.ok()) return status;
    std::vector<float> expected(4);
    status = reference_matmul_f32(a.data(), b.data(), expected.data(), 2, 2, 3);
    if (!status.ok()) return status;
    ComputeDispatch dispatch;
    status = build(lhs.tensor, rhs.tensor, output.tensor, dispatch);
    if (!status.ok()) return status;
    status = execute_and_compare(backend, dispatch, output.tensor, expected);
    if (!status.ok()) return status;

    TensorFixture lhs2, rhs2, output2;
    std::vector<float> a2(3 * 7), b2(7 * 5);
    for (size_t i = 0; i < a2.size(); ++i) a2[i] = static_cast<float>(static_cast<int>(i % 13) - 6) * 0.15f;
    for (size_t i = 0; i < b2.size(); ++i) b2[i] = static_cast<float>(static_cast<int>(i % 9) - 4) * 0.2f;
    status = make_f32_tensor(backend, {3, 7}, a2, "validate-matmul-a-irregular", lhs2);
    if (!status.ok()) return status;
    status = make_f32_tensor(backend, {7, 5}, b2, "validate-matmul-b-irregular", rhs2);
    if (!status.ok()) return status;
    status = make_f32_tensor(backend, {3, 5}, {}, "validate-matmul-out-irregular", output2);
    if (!status.ok()) return status;
    std::vector<float> expected2(15);
    status = reference_matmul_f32(a2.data(), b2.data(), expected2.data(), 3, 5, 7);
    if (!status.ok()) return status;
    status = build(lhs2.tensor, rhs2.tensor, output2.tensor, dispatch);
    if (!status.ok()) return status;
    return execute_and_compare(backend, dispatch, output2.tensor, expected2);
}

template <typename Validate>
Status validate_rms_norm(Backend& backend, Validate&& build) {
    TensorFixture input, weight, output;
    const std::vector<float> x = {1.0f, -2.0f, 0.5f, 3.0f, -1.0f, 0.25f, 2.0f, -0.5f};
    const std::vector<float> w = {1.0f, 0.75f, 1.25f, -0.5f};
    constexpr float epsilon = 1.0e-5f;
    Status status = make_f32_tensor(backend, {2, 4}, x, "validate-rms-x", input);
    if (!status.ok()) return status;
    status = make_f32_tensor(backend, {4}, w, "validate-rms-w", weight);
    if (!status.ok()) return status;
    status = make_f32_tensor(backend, {2, 4}, {}, "validate-rms-out", output);
    if (!status.ok()) return status;
    std::vector<float> expected(x.size());
    status = reference_rms_norm_f32(x.data(), w.data(), expected.data(), 2, 4, epsilon);
    if (!status.ok()) return status;
    ComputeDispatch dispatch;
    status = build(input.tensor, weight.tensor, output.tensor, epsilon, dispatch);
    if (!status.ok()) return status;
    status = execute_and_compare(backend, dispatch, output.tensor, expected);
    if (!status.ok()) return status;

    TensorFixture input2, weight2, output2;
    std::vector<float> x2(65), w2(65);
    for (size_t i = 0; i < x2.size(); ++i) {
        x2[i] = static_cast<float>(static_cast<int>(i % 19) - 9) * 0.1f;
        w2[i] = 0.5f + static_cast<float>(i % 7) * 0.125f;
    }
    status = make_f32_tensor(backend, {1, 65}, x2, "validate-rms-x-wide", input2);
    if (!status.ok()) return status;
    status = make_f32_tensor(backend, {65}, w2, "validate-rms-w-wide", weight2);
    if (!status.ok()) return status;
    status = make_f32_tensor(backend, {1, 65}, {}, "validate-rms-out-wide", output2);
    if (!status.ok()) return status;
    std::vector<float> expected2(x2.size());
    status = reference_rms_norm_f32(x2.data(), w2.data(), expected2.data(), 1, 65, epsilon);
    if (!status.ok()) return status;
    status = build(input2.tensor, weight2.tensor, output2.tensor, epsilon, dispatch);
    if (!status.ok()) return status;
    return execute_and_compare(backend, dispatch, output2.tensor, expected2);
}

template <typename Validate>
Status validate_unary(Backend& backend, Validate&& build, bool softmax) {
    TensorFixture input, output;
    const std::vector<float> x = {-3.0f, -1.0f, 0.0f, 2.0f, 1.0f, 0.5f, -0.5f, 4.0f};
    Status status = make_f32_tensor(backend, {2, 4}, x, "validate-unary-input", input);
    if (!status.ok()) return status;
    status = make_f32_tensor(backend, {2, 4}, {}, "validate-unary-output", output);
    if (!status.ok()) return status;
    std::vector<float> expected(x.size());
    status = softmax
        ? reference_softmax_f32(x.data(), expected.data(), 2, 4)
        : reference_silu_f32(x.data(), expected.data(), x.size());
    if (!status.ok()) return status;
    ComputeDispatch dispatch;
    status = build(input.tensor, output.tensor, dispatch);
    if (!status.ok()) return status;
    status = execute_and_compare(backend, dispatch, output.tensor, expected);
    if (!status.ok()) return status;

    TensorFixture input2, output2;
    std::vector<float> x2(129);
    for (size_t i = 0; i < x2.size(); ++i)
        x2[i] = static_cast<float>(static_cast<int>(i % 23) - 11) * 0.17f;
    status = make_f32_tensor(backend, {1, 129}, x2, "validate-unary-input-odd", input2);
    if (!status.ok()) return status;
    status = make_f32_tensor(backend, {1, 129}, {}, "validate-unary-output-odd", output2);
    if (!status.ok()) return status;
    std::vector<float> expected2(x2.size());
    status = softmax
        ? reference_softmax_f32(x2.data(), expected2.data(), 1, 129)
        : reference_silu_f32(x2.data(), expected2.data(), x2.size());
    if (!status.ok()) return status;
    status = build(input2.tensor, output2.tensor, dispatch);
    if (!status.ok()) return status;
    return execute_and_compare(backend, dispatch, output2.tensor, expected2);
}

template <typename Validate, typename Reference>
Status validate_elementwise(Backend& backend, Validate&& build, Reference&& reference,
                            const char* label) {
    TensorFixture input, output;
    const std::vector<float> x = {-100.0f, -20.0f, -3.0f, -1.0f,
                                   0.0f, 0.5f, 2.0f, 100.0f};
    Status status = make_f32_tensor(backend, {2, 4}, x, label, input);
    if (!status.ok()) return status;
    status = make_f32_tensor(backend, {2, 4}, {}, "validate-elementwise-output", output);
    if (!status.ok()) return status;
    std::vector<float> expected(x.size());
    status = reference(x.data(), expected.data(), x.size());
    if (!status.ok()) return status;
    ComputeDispatch dispatch;
    status = build(input.tensor, output.tensor, dispatch);
    if (!status.ok()) return status;
    status = execute_and_compare(backend, dispatch, output.tensor, expected);
    if (!status.ok()) return status;

    TensorFixture input2, output2;
    std::vector<float> x2(129);
    for (size_t i = 0; i < x2.size(); ++i)
        x2[i] = static_cast<float>(static_cast<int>(i % 31) - 15) * 2.0f;
    status = make_f32_tensor(backend, {1, 129}, x2, "validate-elementwise-input-odd", input2);
    if (!status.ok()) return status;
    status = make_f32_tensor(backend, {1, 129}, {}, "validate-elementwise-output-odd", output2);
    if (!status.ok()) return status;
    std::vector<float> expected2(x2.size());
    status = reference(x2.data(), expected2.data(), x2.size());
    if (!status.ok()) return status;
    status = build(input2.tensor, output2.tensor, dispatch);
    if (!status.ok()) return status;
    return execute_and_compare(backend, dispatch, output2.tensor, expected2);
}

template <typename Validate>
Status validate_split(Backend& backend, Validate&& build) {
    TensorFixture input, output;
    std::vector<float> values(2 * 3 * 4);
    for (size_t i = 0; i < values.size(); ++i) values[i] = static_cast<float>(i) * 0.25f;
    Status status = make_f32_tensor(backend, {2, 3, 4}, values, "validate-split-input", input);
    if (!status.ok()) return status;
    status = make_f32_tensor(backend, {2, 3, 2}, {}, "validate-split-output", output);
    if (!status.ok()) return status;
    std::vector<float> expected(2 * 3 * 2);
    status = reference_split_f32(values.data(), expected.data(), {2, 3, 4}, {2, 3, 2}, 2, 1);
    if (!status.ok()) return status;
    ComputeDispatch dispatch;
    status = build(input.tensor, output.tensor, 2, 1, dispatch);
    if (!status.ok()) return status;
    status = execute_and_compare(backend, dispatch, output.tensor, expected);
    if (!status.ok()) return status;

    TensorFixture input2, output2;
    std::vector<float> values2(3 * 5);
    for (size_t i = 0; i < values2.size(); ++i) values2[i] = static_cast<float>(i) - 4.0f;
    status = make_f32_tensor(backend, {3, 5}, values2, "validate-split-input-axis1", input2);
    if (!status.ok()) return status;
    status = make_f32_tensor(backend, {3, 2}, {}, "validate-split-output-axis1", output2);
    if (!status.ok()) return status;
    std::vector<float> expected2(3 * 2);
    status = reference_split_f32(values2.data(), expected2.data(), {3, 5}, {3, 2}, 1, 2);
    if (!status.ok()) return status;
    status = build(input2.tensor, output2.tensor, 1, 2, dispatch);
    if (!status.ok()) return status;
    return execute_and_compare(backend, dispatch, output2.tensor, expected2);
}

template <typename Validate>
Status validate_rope(Backend& backend, Validate&& build) {
    TensorFixture input, output;
    const std::vector<float> x = {
        1.0f, 0.5f, -1.0f, 2.0f,
        -0.25f, 3.0f, 0.75f, -2.0f
    };
    RopeConfig config;
    config.rotary_dimensions = 4;
    config.start_position = 2;
    config.theta = 10000.0f;
    config.neox_style = false;
    Status status = make_f32_tensor(backend, {2, 1, 4}, x, "validate-rope-input", input);
    if (!status.ok()) return status;
    status = make_f32_tensor(backend, {2, 1, 4}, {}, "validate-rope-output", output);
    if (!status.ok()) return status;
    std::vector<float> expected(x.size());
    status = reference_rope_f32(x.data(), expected.data(), 2, 1, 4, config);
    if (!status.ok()) return status;
    ComputeDispatch dispatch;
    status = build(input.tensor, output.tensor, config, dispatch);
    if (!status.ok()) return status;
    status = execute_and_compare(backend, dispatch, output.tensor, expected);
    if (!status.ok()) return status;

    TensorFixture input2, output2;
    std::vector<float> x2(3 * 2 * 6);
    for (size_t i = 0; i < x2.size(); ++i)
        x2[i] = static_cast<float>(static_cast<int>(i % 15) - 7) * 0.13f;
    RopeConfig config2;
    config2.rotary_dimensions = 4;
    config2.start_position = 7;
    config2.theta = 10000.0f;
    config2.neox_style = true;
    status = make_f32_tensor(backend, {3, 2, 6}, x2, "validate-rope-input-neox", input2);
    if (!status.ok()) return status;
    status = make_f32_tensor(backend, {3, 2, 6}, {}, "validate-rope-output-neox", output2);
    if (!status.ok()) return status;
    std::vector<float> expected2(x2.size());
    status = reference_rope_f32(x2.data(), expected2.data(), 3, 2, 6, config2);
    if (!status.ok()) return status;
    status = build(input2.tensor, output2.tensor, config2, dispatch);
    if (!status.ok()) return status;
    return execute_and_compare(backend, dispatch, output2.tensor, expected2);
}

template <typename Validate>
Status validate_attention(Backend& backend, Validate&& build) {
    TensorFixture query, key, value, output;
    const std::vector<float> q = {1.0f, 0.5f, -1.0f, 2.0f, -0.25f, 3.0f, 0.75f, -2.0f};
    const std::vector<float> k = {0.5f, -1.0f, 2.0f, 0.25f};
    const std::vector<float> v = {1.0f, 2.0f, -0.5f, 0.75f};
    Status status = make_f32_tensor(backend, {1, 2, 4}, q, "validate-attn-q", query);
    if (!status.ok()) return status;
    status = make_f32_tensor(backend, {1, 1, 4}, k, "validate-attn-k", key);
    if (!status.ok()) return status;
    status = make_f32_tensor(backend, {1, 1, 4}, v, "validate-attn-v", value);
    if (!status.ok()) return status;
    status = make_f32_tensor(backend, {1, 2, 4}, {}, "validate-attn-out", output);
    if (!status.ok()) return status;
    std::vector<float> expected(q.size());
    status = reference_attention_f32(q.data(), k.data(), v.data(), expected.data(),
                                     1, 2, 1, 4, 0);
    if (!status.ok()) return status;
    ComputeDispatch dispatch;
    status = build(query.tensor, key.tensor, value.tensor, output.tensor, 0, dispatch);
    if (!status.ok()) return status;
    status = execute_and_compare(backend, dispatch, output.tensor, expected);
    if (!status.ok()) return status;

    TensorFixture query2, key2, value2, output2;
    std::vector<float> q2(2 * 4 * 4), k2(3 * 2 * 4), v2(3 * 2 * 4);
    for (size_t i = 0; i < q2.size(); ++i) q2[i] = static_cast<float>(static_cast<int>(i % 11) - 5) * 0.2f;
    for (size_t i = 0; i < k2.size(); ++i) k2[i] = static_cast<float>(static_cast<int>(i % 7) - 3) * 0.25f;
    for (size_t i = 0; i < v2.size(); ++i) v2[i] = static_cast<float>(static_cast<int>(i % 13) - 6) * 0.1f;
    status = make_f32_tensor(backend, {2, 4, 4}, q2, "validate-attn-q-gqa", query2);
    if (!status.ok()) return status;
    status = make_f32_tensor(backend, {3, 2, 4}, k2, "validate-attn-k-gqa", key2);
    if (!status.ok()) return status;
    status = make_f32_tensor(backend, {3, 2, 4}, v2, "validate-attn-v-gqa", value2);
    if (!status.ok()) return status;
    status = make_f32_tensor(backend, {2, 4, 4}, {}, "validate-attn-out-gqa", output2);
    if (!status.ok()) return status;
    std::vector<float> expected2(q2.size());
    status = reference_attention_f32(q2.data(), k2.data(), v2.data(), expected2.data(),
                                     2, 4, 2, 4, 1);
    if (!status.ok()) return status;
    status = build(query2.tensor, key2.tensor, value2.tensor, output2.tensor, 1, dispatch);
    if (!status.ok()) return status;
    return execute_and_compare(backend, dispatch, output2.tensor, expected2);
}

template <typename Validate>
Status validate_embedding(Backend& backend, Validate&& build) {
    TensorFixture weights, output;
    const std::vector<float> table = {
        1.0f, 0.5f, -1.0f, 2.0f,
        -0.25f, 3.0f, 0.75f, -2.0f,
        2.5f, -0.5f, 1.5f, 0.25f
    };
    const std::vector<uint32_t> ids = {2, 0};
    Status status = make_f32_tensor(backend, {3, 4}, table, "validate-embedding-weight", weights);
    if (!status.ok()) return status;
    status = make_f32_tensor(backend, {2, 4}, {}, "validate-embedding-output", output);
    if (!status.ok()) return status;
    BufferDesc token_desc;
    token_desc.size_bytes = ids.size() * sizeof(uint32_t);
    token_desc.alignment = alignof(uint32_t);
    token_desc.memory = MemoryDomain::Device;
    token_desc.label = "validate-embedding-token-ids";
    std::shared_ptr<Buffer> token_buffer;
    status = backend.allocate(token_desc, token_buffer);
    if (!status.ok()) return status;
    status = backend.upload(token_buffer, 0, ids.data(), token_desc.size_bytes);
    if (!status.ok()) return status;
    std::vector<float> expected(ids.size() * 4);
    status = reference_embedding_f32(table.data(), 3, 4, ids, expected.data());
    if (!status.ok()) return status;
    ComputeDispatch dispatch;
    status = build(weights.tensor, token_buffer, ids, output.tensor, dispatch);
    if (!status.ok()) return status;
    status = execute_and_compare(backend, dispatch, output.tensor, expected);
    if (!status.ok()) return status;

    TensorFixture weights2, output2;
    std::vector<float> table2(5 * 7);
    const std::vector<uint32_t> ids2 = {4, 0, 2};
    for (size_t i = 0; i < table2.size(); ++i)
        table2[i] = static_cast<float>(static_cast<int>(i % 17) - 8) * 0.11f;
    status = make_f32_tensor(backend, {5, 7}, table2, "validate-embedding-weight-odd", weights2);
    if (!status.ok()) return status;
    status = make_f32_tensor(backend, {3, 7}, {}, "validate-embedding-output-odd", output2);
    if (!status.ok()) return status;
    BufferDesc token_desc2;
    token_desc2.size_bytes = ids2.size() * sizeof(uint32_t);
    token_desc2.alignment = alignof(uint32_t);
    token_desc2.memory = MemoryDomain::Device;
    token_desc2.label = "validate-embedding-token-ids-odd";
    std::shared_ptr<Buffer> token_buffer2;
    status = backend.allocate(token_desc2, token_buffer2);
    if (!status.ok()) return status;
    status = backend.upload(token_buffer2, 0, ids2.data(), token_desc2.size_bytes);
    if (!status.ok()) return status;
    std::vector<float> expected2(ids2.size() * 7);
    status = reference_embedding_f32(table2.data(), 5, 7, ids2, expected2.data());
    if (!status.ok()) return status;
    status = build(weights2.tensor, token_buffer2, ids2, output2.tensor, dispatch);
    if (!status.ok()) return status;
    return execute_and_compare(backend, dispatch, output2.tensor, expected2);
}

template <typename Validate>
void run_one(KernelDefinition* definition, KernelValidationReport& report,
             const char* operator_name, const char* kernel_id, Validate&& validate) {
    const std::string label = std::string(operator_name) + ":" + kernel_id;
    if (!definition) {
        report.unavailable.push_back(label + ":NOT_REGISTERED");
        return;
    }
    if (definition->validation == KernelValidation::Validated) {
        report.validated.push_back(label + ":ALREADY_VALIDATED");
        return;
    }
    const Status status = validate();
    if (status.ok()) {
        definition->validation = KernelValidation::Validated;
        report.validated.push_back(label);
    } else if (status.code == StatusCode::NumericalFailure) {
        definition->validation = KernelValidation::Rejected;
        report.rejected.push_back(label + ":" + status.message);
    } else {
        report.unavailable.push_back(label + ":" + status.message);
    }
}

} // namespace

Status validate_builtin_operator_kernels(Backend& backend, OperatorRegistry& registry,
                                         KernelValidationReport& report) {
    report = KernelValidationReport{};
    if (!backend.healthy() || backend.kind() == BackendKind::Unknown)
        return {StatusCode::NotReady, "Kernel validation requires a healthy initialized backend"};

    auto run = [&](const char* operator_name, const char* kernel_id, auto&& validate) {
        auto found = std::find_if(registry.kernels_.begin(), registry.kernels_.end(),
            [operator_name, kernel_id, &backend](const KernelDefinition& definition) {
                return definition.operator_name == operator_name &&
                       definition.kernel_id == kernel_id && definition.backend == backend.kind();
            });
        run_one(found == registry.kernels_.end() ? nullptr : &*found,
                report, operator_name, kernel_id, std::forward<decltype(validate)>(validate));
    };

    const bool gles = backend.kind() == BackendKind::OpenGles;
    run("Add", gles ? "gles-add-f32-generic-v1" : "dawn-add-f32-generic-v1",
        [&] {
            return validate_add(backend, [&](const Tensor& a, const Tensor& b,
                                             const Tensor& out, ComputeDispatch& dispatch) {
                return gles ? build_add_f32_gles_candidate(a, b, out, dispatch)
                            : build_add_f32_candidate(a, b, out, dispatch);
            });
        });

    run( "Multiply",
            gles ? "gles-multiply-f32-generic-v1" : "dawn-multiply-f32-generic-v1",
            [&] {
                return validate_multiply(backend, [&](const Tensor& a, const Tensor& b,
                                                      const Tensor& out, ComputeDispatch& dispatch) {
                    return gles ? build_multiply_f32_gles_candidate(a, b, out, dispatch)
                                : build_multiply_f32_candidate(a, b, out, dispatch);
                });
            });
    run( "MatMul",
            gles ? "gles-matmul-f32-generic-v1" : "dawn-matmul-f32-generic-v1",
            [&] {
                return validate_matmul(backend, [&](const Tensor& a, const Tensor& b,
                                                    const Tensor& out, ComputeDispatch& dispatch) {
                    return gles ? build_matmul_f32_gles_candidate(a, b, out, dispatch)
                                : build_matmul_f32_candidate(a, b, out, dispatch);
                });
            });
    run( "RMSNorm",
            gles ? "gles-rms-norm-f32-generic-v1" : "dawn-rms-norm-f32-generic-v1",
            [&] {
                return validate_rms_norm(backend, [&](const Tensor& input, const Tensor& weight,
                                                      const Tensor& out, float epsilon, ComputeDispatch& dispatch) {
                    return gles ? build_rms_norm_f32_gles_candidate(input, weight, out, epsilon, dispatch)
                                : build_rms_norm_f32_candidate(input, weight, out, epsilon, dispatch);
                });
            });
    run( "SiLU",
            gles ? "gles-silu-f32-generic-v1" : "dawn-silu-f32-generic-v1",
            [&] {
                return validate_unary(backend, [&](const Tensor& input, const Tensor& out,
                                                   ComputeDispatch& dispatch) {
                    return gles ? build_silu_f32_gles_candidate(input, out, dispatch)
                                : build_silu_f32_candidate(input, out, dispatch);
                }, false);
            });
    run("Split", gles ? "gles-split-f32-v1" : "dawn-split-f32-v1",
        [&] {
            return validate_split(backend, [&](const Tensor& input, const Tensor& output,
                                               uint32_t axis, uint32_t offset,
                                               ComputeDispatch& dispatch) {
                return gles ? build_split_f32_gles_candidate(input, output, axis, offset, dispatch)
                            : build_split_f32_candidate(input, output, axis, offset, dispatch);
            });
        });
    run("Sigmoid", gles ? "gles-sigmoid-f32-v1" : "dawn-sigmoid-f32-v1",
        [&] {
            return validate_elementwise(backend, [&](const Tensor& input, const Tensor& output,
                                                      ComputeDispatch& dispatch) {
                return gles ? build_sigmoid_f32_gles_candidate(input, output, dispatch)
                            : build_sigmoid_f32_candidate(input, output, dispatch);
            }, reference_sigmoid_f32, "validate-sigmoid-input");
        });
    run("Softplus", gles ? "gles-softplus-f32-v1" : "dawn-softplus-f32-v1",
        [&] {
            return validate_elementwise(backend, [&](const Tensor& input, const Tensor& output,
                                                      ComputeDispatch& dispatch) {
                return gles ? build_softplus_f32_gles_candidate(input, output, dispatch)
                            : build_softplus_f32_candidate(input, output, dispatch);
            }, reference_softplus_f32, "validate-softplus-input");
        });
    run( "Softmax",
            gles ? "gles-softmax-f32-generic-v1" : "dawn-softmax-f32-generic-v1",
            [&] {
                return validate_unary(backend, [&](const Tensor& input, const Tensor& out,
                                                   ComputeDispatch& dispatch) {
                    return gles ? build_softmax_f32_gles_candidate(input, out, dispatch)
                                : build_softmax_f32_candidate(input, out, dispatch);
                }, true);
            });
    run( "RoPE",
            gles ? "gles-rope-f32-generic-v1" : "dawn-rope-f32-generic-v1",
            [&] {
                return validate_rope(backend, [&](const Tensor& input, const Tensor& out,
                                                  const RopeConfig& config, ComputeDispatch& dispatch) {
                    return gles ? build_rope_f32_gles_candidate(input, out, config, dispatch)
                                : build_rope_f32_candidate(input, out, config, dispatch);
                });
            });
    run( "Attention",
            gles ? "gles-attention-f32-causal-gqa-v1" : "dawn-attention-f32-causal-gqa-v1",
            [&] {
                return validate_attention(backend, [&](const Tensor& q, const Tensor& k,
                    const Tensor& v, const Tensor& out, uint32_t past, ComputeDispatch& dispatch) {
                    return gles ? build_attention_f32_gles_candidate(q, k, v, out, past, dispatch)
                                : build_attention_f32_candidate(q, k, v, out, past, dispatch);
                });
            });
    run( "Embedding",
            gles ? "gles-embedding-f32-generic-v1" : "dawn-embedding-f32-generic-v1",
            [&] {
                return validate_embedding(backend, [&](const Tensor& weights,
                    const std::shared_ptr<Buffer>& token_buffer, const std::vector<uint32_t>& ids,
                    const Tensor& out, ComputeDispatch& dispatch) {
                    return gles ? build_embedding_f32_gles_candidate(weights, token_buffer, ids, out, dispatch)
                                : build_embedding_f32_candidate(weights, token_buffer, ids, out, dispatch);
                });
            });

    report.complete = report.unavailable.empty() && report.rejected.empty();
    return Status::success();
}

} // namespace ai_chat::gpu_engine
