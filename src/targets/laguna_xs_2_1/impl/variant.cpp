#include "targets/laguna_xs_2_1/impl/variant.h"

#include "ninfer/ops/linear.h"
#include "ninfer/ops/linear_add.h"
#include "ninfer/ops/sparse_moe.h"

#include <algorithm>
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
    TextPhase /*phase*/,
    Tensor& q_out, Tensor& k_out, Tensor& v_out,
    WorkspaceArena& /*workspace*/,
    cudaStream_t stream)
{
    const int T = static_cast<int>(hidden.ne[1]);
    const int q_rows = q_rows(layer);
    const int kv_rows = kv_heads * head_dim;

    // QKV projections: separate weights
    Tensor q_work = hidden.view({q_rows, T});
    Tensor k_work = hidden.view({kv_rows, T});
    Tensor v_work = hidden.view({kv_rows, T});

    // TODO: Actually compute QKV via GEMM. For now, these are view placeholders.
    // The real implementation needs:
    //   ops::linear(hidden, weights.q_proj, q_work, stream);
    //   ops::linear(hidden, weights.k_proj, k_work, stream);
    //   ops::linear(hidden, weights.v_proj, v_work, stream);

    // Reshape to [head_dim, q_heads, T] / [head_dim, kv_heads, T]
    const int q_heads = q_heads(layer);
    q_out = q_work.view({head_dim, q_heads, T});
    k_out = k_work.view({head_dim, kv_heads, T});
    v_out = v_work.view({head_dim, kv_heads, T});

    // TODO: QK normalization (RMSNorm per-head, before RoPE)
    // TODO: RoPE (YARN for full, standard for SWA)
    (void)weights;
    (void)layer;
    (void)stream;
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

        Tensor gate_work = workspace.alloc(DType::BF16, {intermediate, T});
        Tensor up_work = workspace.alloc(DType::BF16, {intermediate, T});

        ops::linear(hidden, std::get<DenseMlpPayload>(weights).gate_proj, gate_work, stream);
        ops::linear(hidden, std::get<DenseMlpPayload>(weights).up_proj, up_work, stream);

        // SwiGLU: activation = silu(gate) * up
        // TODO: Implement or use ops::linear_swiglu
        Tensor activation = workspace.alloc(DType::BF16, {intermediate, T});
        (void)activation;

        // down = down_proj @ activation
        ops::linear_add(activation, std::get<DenseMlpPayload>(weights).down_proj, residual, stream);

    } else {
        // Layers 1-39: MoE
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
    bytes += static_cast<std::size_t>(max_q_rows) * max_tokens * sizeof(__nv_bfloat16);
    bytes += static_cast<std::size_t>(kv_rows) * max_tokens * sizeof(__nv_bfloat16);
    bytes += static_cast<std::size_t>(kv_rows) * max_tokens * sizeof(__nv_bfloat16);
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