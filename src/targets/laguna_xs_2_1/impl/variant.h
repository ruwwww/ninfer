#pragma once

// Variant declaration for Laguna XS 2.1.

#include "targets/laguna_xs_2_1/impl/config.h"
#include "targets/laguna_xs_2_1/impl/load/bindings.h"

#include "core/arena.h"
#include "core/tensor.h"
#include <ninfer/targets/qwen3_6/diagnostics.h>

#include <cmath>
#include <cstdint>
#include <cstddef>
#include <cuda_runtime.h>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

namespace ninfer::targets::laguna_xs_2_1::detail {

struct GraphFrontierRange {
    std::uint32_t min = 0;
    std::uint32_t max = 0;
};

// Use the shared qwen3_6 TextPhase so the shared runtime code compiles correctly
using TextPhase = qwen3_6::TextPhase;

// ---- Variant struct ----

struct Variant {
    using TextConfig                       = detail::TextConfig;
    using VisionConfig                     = detail::VisionConfig;
    using DFlashConfig                     = detail::DFlashConfig;
    using ModelView                        = detail::LoadedModelData;
    using WeightsProfile                   = detail::WeightsProfile;
    using FullAttentionProjectionWeights   = AttentionProjectionPayload;
    using GdnProjectionWeights             = GdnProjectionPayload;
    using PostMixerWeights                 = std::variant<DenseMlpPayload, SparseMoePayload>;
    using GraphFrontierRange               = detail::GraphFrontierRange;
    using VisionWeights                    = void;

    // ---- Identity ----
    [[nodiscard]] static constexpr std::string_view model_id() noexcept {
        return "laguna-xs-2.1";
    }
    [[nodiscard]] static constexpr std::string_view target_key() noexcept {
        return "laguna_xs_2_1";
    }

    // ---- Model dimensions ----
    static constexpr int hidden            = 2048;
    static constexpr int layers            = 40;
    static constexpr int head_dim          = 128;
    static constexpr int kv_heads          = 8;
    static constexpr int num_experts       = 256;
    static constexpr int topk              = 8;
    static constexpr int sliding_window    = 512;
    static constexpr int maximum_context   = 262144;
    static constexpr int output_rows       = 100352;

    static constexpr int full_q_heads = 48;
    static constexpr int swa_q_heads  = 64;
    static constexpr int full_q_rows  = 6144;
    static constexpr int swa_q_rows   = 8192;
    static constexpr int kv_rows      = 1024;

    static constexpr float attention_scale = 0.08838834764831845F;

    // Shared runtime constexpr requirements
    static constexpr std::uint32_t draft_head_rows            = 0;
    static constexpr std::uint32_t maximum_mtp_draft_tokens   = 0;
    static constexpr std::uint32_t maximum_dflash_draft_tokens = 0;
    static constexpr std::uint32_t prefill_chunk_alignment    = 128;
    static constexpr float       gdn_scale                    = 0.0F;

    static constexpr bool supports_vision       = false;
    static constexpr bool supports_mtp          = false;
    static constexpr bool supports_dflash       = false;
    static constexpr bool supports_fp8_kv       = true;
    static constexpr bool uses_sliding_window   = true;

    // ---- Static leaf function declarations ----

static void attention_projection(
        const Tensor& hidden,
        const FullAttentionProjectionWeights& weights,
        Tensor& query, Tensor& gate, Tensor& key, Tensor& value,
        TextPhase phase,
        WorkspaceArena& workspace,
        cudaStream_t stream);

    static void post_mixer(
        const Tensor& hidden,
        const PostMixerWeights& weights,
        Tensor& residual,
        TextPhase phase,
        WorkspaceArena& workspace,
        cudaStream_t stream);

    static void run_sparse_moe(const Tensor& hidden, const ops::SparseMoeWeights& weights,
        Tensor& residual, WorkspaceArena& workspace, cudaStream_t stream);

    // Stub methods for unsupported features (MTP, GDN, attention output proj)
    // These are never called at runtime because the corresponding feature flags are zero.
    static void mtp_attention_projection(
        const Tensor&, const void*, Tensor&, Tensor&, Tensor&, Tensor&,
        WorkspaceArena&, cudaStream_t) {}
    static void mtp_kv_projection(
        const Tensor&, const void*, Tensor&, Tensor&,
        WorkspaceArena&, cudaStream_t) {}
    static void mtp_q_gate_projection(
        const Tensor&, const void*, Tensor&, Tensor&,
        WorkspaceArena&, cudaStream_t) {}
    static void mtp_post_mixer(
        const Tensor&, const void*, Tensor&,
        WorkspaceArena&, cudaStream_t) {}
    static void gdn_norm_control_projection(
        const Tensor&, const Tensor&, float, const void*,
        Tensor&, Tensor&, Tensor&, WorkspaceArena&, cudaStream_t) {}
    static void gdn_input_projection(
        const Tensor&, const void*, Tensor&, Tensor&, TextPhase,
        WorkspaceArena&, cudaStream_t) {}
    static void gdn_input_projection_snapshot(
        const Tensor&, const void*, const Tensor&, Tensor&,
        const Tensor&, Tensor&, Tensor&, Tensor&, TextPhase,
        WorkspaceArena&, cudaStream_t) {}
    static void gdn_output_projection(
        const Tensor&, const Weight&, Tensor&, TextPhase,
        WorkspaceArena&, cudaStream_t) {}
    static void attention_output_projection(
        const Tensor& attention, const Weight& weight, Tensor& residual, TextPhase phase,
        WorkspaceArena& workspace, cudaStream_t stream);

    // ---- Graph capture frontier ranges ----
    static std::vector<GraphFrontierRange> ordinary_graph_ranges(std::uint32_t capacity) noexcept;
    static std::vector<GraphFrontierRange> mtp_graph_ranges(std::uint32_t, std::uint32_t) noexcept {
        return {};
    }
    static std::vector<GraphFrontierRange> dflash_graph_ranges(std::uint32_t, std::uint32_t) noexcept {
        return {};
    }

    // ---- Workspace capacity functions ----
    [[nodiscard]] static std::size_t attention_projection_workspace_capacity_bytes(
        WeightsProfile profile, TextPhase phase, int first, int last);

    [[nodiscard]] static std::size_t post_mixer_workspace_capacity_bytes(
        WeightsProfile profile, TextPhase phase, int first, int last);

    // Stub capacity functions for unsupported features (GDN, MTP, DFlash, attn output proj)
    [[nodiscard]] static std::size_t attention_output_projection_workspace_capacity_bytes(
        WeightsProfile profile, TextPhase phase, int first, int last);
    [[nodiscard]] static std::size_t gdn_norm_control_projection_workspace_capacity_bytes(
        int first, int last);
    [[nodiscard]] static std::size_t gdn_input_projection_workspace_capacity_bytes(
        WeightsProfile profile, TextPhase phase, int first, int last);
    [[nodiscard]] static std::size_t gdn_input_projection_snapshot_workspace_capacity_bytes(
        WeightsProfile profile, TextPhase phase, int first, int last);
    [[nodiscard]] static std::size_t gdn_output_projection_workspace_capacity_bytes(
        WeightsProfile profile, TextPhase phase, int first, int last);
    [[nodiscard]] static std::size_t mtp_attention_projection_workspace_capacity_bytes(
        int first, int last);
    [[nodiscard]] static std::size_t mtp_kv_projection_workspace_capacity_bytes(
        int first, int last);
    [[nodiscard]] static std::size_t mtp_q_gate_projection_workspace_capacity_bytes(
        int first, int last);
    [[nodiscard]] static std::size_t mtp_post_mixer_workspace_capacity_bytes(
        int first, int last);

    // ---- Layer-type classification ----
    [[nodiscard]] static constexpr bool is_full_attention(int layer) noexcept {
        return (layer % 4) == 0;
    }
    [[nodiscard]] static constexpr bool is_swa(int layer) noexcept {
        return (layer % 4) != 0;
    }
    [[nodiscard]] static constexpr bool is_dense_mlp(int layer) noexcept {
        return layer == 0;
    }
    [[nodiscard]] static constexpr bool is_moe(int layer) noexcept {
        return layer > 0;
    }
    [[nodiscard]] static constexpr int q_heads(int layer) noexcept {
        return is_full_attention(layer) ? full_q_heads : swa_q_heads;
    }
    [[nodiscard]] static constexpr int q_rows(int layer) noexcept {
        return is_full_attention(layer) ? full_q_rows : swa_q_rows;
    }
};

}  // namespace ninfer::targets::laguna_xs_2_1::detail