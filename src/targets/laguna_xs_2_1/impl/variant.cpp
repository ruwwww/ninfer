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
    std::uint32_t begin = 0;
    for (const std::uint32_t preferred_end : preferred_ends) {
        if (begin > max_frontier) { break; }
        const std::uint32_t end = std::min(preferred_end, max_frontier);
        out.push_back({begin, end});
        if (end == max_frontier) { return out; }
        begin = end + 1;
    }
    if (begin <= max_frontier) { out.push_back({begin, max_frontier}); }
    return out;
}

} // namespace

// ---- Graph frontier ranges ----

std::vector<GraphFrontierRange> Variant::ordinary_graph_ranges() noexcept {
    return graph_ranges_through(static_cast<std::uint32_t>(maximum_context - 1),
                                {127, 511, 2047, 4095});
}

// ---- Attention projection ----

void Variant::attention_projection(
    const Tensor& hidden,
    const FullAttentionProjectionWeights& weights,
    int layer,
    TextPhase phase,
    Tensor& q_out, Tensor& k_out, Tensor& v_out,
    WorkspaceArena& workspace,
    cudaStream_t stream)
{
    const int T = static_cast<int>(hidden.ne[1]);
    const int q_row_count = q_rows(layer);
    const int kv_rows = kv_heads * head_dim;
    const bool is_full = is_full_attention(layer);

    // Step 1: QKV GEMM projections
    auto scope = workspace.scope();

    Tensor q_proj = workspace.alloc(DType::BF16, {q_row_count, T});
    Tensor k_proj = workspace.alloc(DType::BF16, {kv_rows, T});
    Tensor v_proj = workspace.alloc(DType::BF16, {kv_rows, T});

    ops::linear(hidden, weights.q_proj, q_proj, stream);
    ops::linear(hidden, weights.k_proj, k_proj, stream);
    ops::linear(hidden, weights.v_proj, v_proj, stream);

    // Step 2: QK RMSNorm (per-head, before RoPE)
    Tensor q_normed = workspace.alloc(DType::BF16, {head_dim, q_heads(layer), T});
    Tensor k_normed = workspace.alloc(DType::BF16, {head_dim, kv_heads, T});

    for (int h = 0; h < q_heads(layer); ++h) {
        Tensor q_head = q_proj.slice(1, h, 1).view({head_dim, T});
        Tensor q_norm_head = q_normed.slice(1, h, 1).view({head_dim, T});
        ops::rmsnorm(q_head, weights.q_norm.tensor, TextConfig::rms_norm_eps, false, q_norm_head, stream);
    }

    for (int h = 0; h < kv_heads; ++h) {
        Tensor k_head = k_proj.slice(1, h, 1).view({head_dim, T});
        Tensor k_norm_head = k_normed.slice(1, h, 1).view({head_dim, T});
        ops::rmsnorm(k_head, weights.k_norm.tensor, TextConfig::rms_norm_eps, false, k_norm_head, stream);
    }

    // Step 3: RoPE dispatch based on layer type
    if (is_full) {
        ops::rope(/*positions=*/{}, /*rotary_dim=*/64, TextConfig::RopeConfig::full_theta,
                  q_normed, k_normed, stream);
    } else {
        ops::rope(/*positions=*/{}, /*rotary_dim=*/128, TextConfig::RopeConfig::swa_theta,
                  q_normed, k_normed, stream);
    }

    q_out = q_normed;
    k_out = k_normed;
    v_out = v_proj.view({head_dim, kv_heads, T});

    (void)phase;
}

// ---- Post-mixer ----

void Variant::post_mixer(
    const Tensor& hidden,
    const PostMixerWeights& weights,
    int layer,
    TextPhase /*phase*/,
    Tensor& residual,
    WorkspaceArena& workspace,
    cudaStream_t stream)
{
    auto scope = workspace.scope();

    if (is_dense_mlp(layer)) {
        // Layer 0: Dense SwiGLU MLP (intermediate_size = 8192)
        const int intermediate = 8192;
        const int T = static_cast<int>(hidden.ne[1]);

        Tensor gate_up = workspace.alloc(DType::BF16, {2 * intermediate, T});
        ops::linear(hidden, std::get<DenseMlpPayload>(weights).gate_proj, gate_up, stream);

        Tensor activation = workspace.alloc(DType::BF16, {intermediate, T});
        ops::silu_mul(gate_up.slice(0, 0, intermediate),
                      gate_up.slice(0, intermediate, intermediate), activation, stream);

        ops::linear_add(activation, std::get<DenseMlpPayload>(weights).down_proj, residual, stream);

    } else {
        // Layers 1-39: MoE with sigmoid router
        run_sparse_moe(hidden, std::get<SparseMoePayload>(weights).op, residual, workspace, stream);
    }

    (void)stream;
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

}  // namespace ninfer::targets::laguna_xs_2_1::detail