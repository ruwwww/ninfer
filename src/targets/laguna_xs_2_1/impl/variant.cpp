#include "targets/laguna_xs_2_1/impl/variant.h"

#include "ninfer/ops/linear.h"
#include "ninfer/ops/linear_add.h"
#include "ninfer/ops/linear_swiglu.h"
#include "ninfer/ops/rmsnorm.h"
#include "ninfer/ops/rope.h"
#include "ninfer/ops/silu_mul.h"
#include "ninfer/ops/sparse_moe.h"

#include <algorithm>
#include <cmath>
#include <cuda_bf16.h>
#include <stdexcept>
#include <vector>

namespace ninfer::targets::laguna_xs_2_1::detail {
namespace {

std::vector<GraphFrontierRange>
graph_ranges_through(std::uint32_t max_frontier, const std::vector<std::uint32_t>& preferred_ends) {
    std::vector<GraphFrontierRange> out;
    std::uint32_t min_range = 0;
    for (const std::uint32_t preferred_end : preferred_ends) {
        if (min_range > max_frontier) { break; }
        const std::uint32_t max_range = std::min(preferred_end, max_frontier);
        out.push_back({min_range, max_range});
        if (max_range == max_frontier) { return out; }
        min_range = max_range + 1;
    }
    if (min_range <= max_frontier) { out.push_back({min_range, max_frontier}); }
    return out;
}

} // namespace

// ---- Graph frontier ranges ----

std::vector<GraphFrontierRange> Variant::ordinary_graph_ranges(std::uint32_t capacity) noexcept {
    return graph_ranges_through(capacity, {127, 511, 2047, 4095});
}

// ---- Attention projection ----

void Variant::attention_projection(
    const Tensor& hidden,
    const FullAttentionProjectionWeights& weights,
    Tensor& query, Tensor& gate, Tensor& key, Tensor& value,
    TextPhase /*phase*/,
    WorkspaceArena& workspace,
    cudaStream_t stream)
{
    const int T = static_cast<int>(hidden.ne[1]);
    auto scope = workspace.scope();

    ops::linear(hidden, weights.q_proj, query, stream);
    ops::linear(hidden, weights.k_proj, key, stream);
    ops::linear(hidden, weights.v_proj, value, stream);
    ops::linear(hidden, weights.g_proj, gate, stream);
}

void Variant::attention_output_projection(const Tensor& attention, const Weight& weight,
                                          Tensor& residual, TextPhase,
                                          WorkspaceArena& workspace, cudaStream_t stream) {
    ops::linear_add(attention, weight, residual, workspace, stream);
}

// ---- Post-mixer ----

void Variant::post_mixer(
    const Tensor& hidden,
    const PostMixerWeights& weights,
    Tensor& residual,
    TextPhase /*phase*/,
    WorkspaceArena& workspace,
    cudaStream_t stream)
{
    auto scope = workspace.scope();

    if (std::holds_alternative<DenseMlpPayload>(weights)) {
        // Layer 0: Dense SwiGLU MLP (intermediate_size = 8192)
        const auto& payload = std::get<DenseMlpPayload>(weights);
        const int intermediate = 8192;
        const int T = static_cast<int>(hidden.ne[1]);

        Tensor gate_up = workspace.alloc(DType::BF16, {2 * intermediate, T});
        ops::linear(hidden, payload.gate_proj, gate_up, stream);

        Tensor activation = workspace.alloc(DType::BF16, {intermediate, T});
        ops::silu_mul(gate_up.slice(0, 0, intermediate),
                      gate_up.slice(0, intermediate, intermediate), activation, stream);

        ops::linear_add(activation, payload.down_proj, residual, workspace, stream);
    } else {
        // Layers 1-39: MoE with sigmoid router
        run_sparse_moe(hidden, std::get<SparseMoePayload>(weights).op, residual, workspace, stream);
    }

    (void)workspace;
}

void Variant::run_sparse_moe(const Tensor& hidden, const ops::SparseMoeWeights& weights,
                              Tensor& residual, WorkspaceArena& workspace, cudaStream_t stream)
{
    auto scope = workspace.scope();
    const int T = static_cast<int>(hidden.ne[1]);
    const std::size_t capacity = ops::sparse_moe_workspace_capacity_bytes(
        weights.routed_gate_up.qtype, weights.routed_down.qtype, T, T);
    const DeviceSpan storage = workspace.alloc_bytes(capacity);
    WorkspaceArena leaf_workspace(storage);

    ops::sparse_moe(hidden, weights, ops::SparseMoeEpilogue::AddResidual, residual,
                    leaf_workspace, stream);
}

// ---- Workspace capacity ----

std::size_t Variant::attention_projection_workspace_capacity_bytes(
    WeightsProfile /*profile*/, TextPhase /*phase*/, int first, int last)
{
    const int max_tokens = last;
    const int max_q_rows = swa_q_rows;
    const int kv_rows = kv_heads * head_dim;

    std::size_t bytes = 0;
    bytes += static_cast<std::size_t>(max_q_rows) * max_tokens * sizeof(float);
    bytes += static_cast<std::size_t>(kv_rows) * max_tokens * sizeof(float);
    bytes += static_cast<std::size_t>(kv_rows) * max_tokens * sizeof(float);
    return bytes;
}

std::size_t Variant::post_mixer_workspace_capacity_bytes(
    WeightsProfile /*profile*/, TextPhase phase, int first, int last)
{
    if (phase == TextPhase::Prefill) {
        return ops::sparse_moe_workspace_capacity_bytes(
            QType::Q4G64_F16S, QType::Q5G64_F16S, first, last);
    }
    return ops::sparse_moe_workspace_capacity_bytes(
        QType::Q4G64_F16S, QType::Q5G64_F16S, 1, 1);
}

// ---- Stub workspace capacity functions for unsupported features ----
// Laguna does not use GDN, MTP, DFlash, or attention output projection.
// These stubs exist so the shared runtime code compiles; they are never
// called at runtime because the corresponding feature flags are zero.

std::size_t Variant::attention_output_projection_workspace_capacity_bytes(
    WeightsProfile /*profile*/, TextPhase /*phase*/, int /*first*/, int /*last*/) {
    return 0;
}

std::size_t Variant::gdn_norm_control_projection_workspace_capacity_bytes(
    int /*first*/, int /*last*/) {
    return 0;
}

std::size_t Variant::gdn_input_projection_workspace_capacity_bytes(
    WeightsProfile /*profile*/, TextPhase /*phase*/, int /*first*/, int /*last*/) {
    return 0;
}

std::size_t Variant::gdn_input_projection_snapshot_workspace_capacity_bytes(
    WeightsProfile /*profile*/, TextPhase /*phase*/, int /*first*/, int /*last*/) {
    return 0;
}

std::size_t Variant::gdn_output_projection_workspace_capacity_bytes(
    WeightsProfile /*profile*/, TextPhase /*phase*/, int /*first*/, int /*last*/) {
    return 0;
}

std::size_t Variant::mtp_attention_projection_workspace_capacity_bytes(
    int /*first*/, int /*last*/) {
    return 0;
}

std::size_t Variant::mtp_kv_projection_workspace_capacity_bytes(
    int /*first*/, int /*last*/) {
    return 0;
}

std::size_t Variant::mtp_q_gate_projection_workspace_capacity_bytes(
    int /*first*/, int /*last*/) {
    return 0;
}

std::size_t Variant::mtp_post_mixer_workspace_capacity_bytes(
    int /*first*/, int /*last*/) {
    return 0;
}

}  // namespace ninfer::targets::laguna_xs_2_1::detail

// ---- Runtime template instantiation ----
// Laguna-specific instantiation using custom runtime files.
// Laguna does not use GDN, Vision, MTP, or DFlash; those features are
// gated by compile-time constants (all zero) and never executed.

#define NINFER_QWEN36_VARIANT        ::ninfer::targets::laguna_xs_2_1::detail::Variant
#define NINFER_QWEN36_RUNTIME_NS     laguna_runtime
#include "targets/laguna_xs_2_1/impl/runtime/instantiate.h"

#undef NINFER_QWEN36_VARIANT
#undef NINFER_QWEN36_RUNTIME_NS