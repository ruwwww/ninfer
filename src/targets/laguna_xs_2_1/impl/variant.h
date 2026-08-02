#pragma once

// Variant declaration for Laguna XS 2.1.

#include "targets/laguna_xs_2_1/impl/config.h"
#include "targets/laguna_xs_2_1/impl/load/bindings.h"

#include "core/arena.h"
#include "core/tensor.h"

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
    std::uint32_t begin = 0;
    std::uint32_t end   = 0;
};

enum class TextPhase : std::uint8_t {
    Prefill = 0,
    Decode  = 1,
};

// ---- Variant struct ----

struct Variant {
    using ModelView                      = detail::LoadedModelData;
    using WeightsProfile                 = detail::WeightsProfile;
    using FullAttentionProjectionWeights = AttentionProjectionPayload;
    using GdnProjectionWeights           = GdnProjectionPayload;
    using PostMixerWeights               = std::variant<DenseMlpPayload, SparseMoePayload>;

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

    static constexpr float attention_scale = 1.0F / sqrtf(static_cast<float>(head_dim));

    static constexpr bool supports_vision       = false;
    static constexpr bool supports_mtp          = false;
    static constexpr bool supports_dflash       = false;
    static constexpr bool supports_fp8_kv       = true;
    static constexpr bool uses_sliding_window   = true;

    // ---- Static leaf function declarations ----

    static void attention_projection(
        const Tensor& hidden,
        const FullAttentionProjectionWeights& weights,
        int layer,
        TextPhase phase,
        Tensor& q_out, Tensor& k_out, Tensor& v_out,
        WorkspaceArena& workspace,
        cudaStream_t stream);

    static void post_mixer(
        const Tensor& hidden,
        const PostMixerWeights& weights,
        int layer,
        TextPhase phase,
        Tensor& residual,
        WorkspaceArena& workspace,
        cudaStream_t stream);

    static void run_sparse_moe(const Tensor& hidden, const ops::SparseMoeWeights& weights,
                                Tensor& residual, WorkspaceArena& workspace, cudaStream_t stream);

    // ---- Graph capture frontier ranges ----
    static std::vector<GraphFrontierRange> ordinary_graph_ranges() noexcept;

    // ---- Workspace capacity functions ----
    [[nodiscard]] static std::size_t attention_projection_workspace_capacity_bytes(
        WeightsProfile profile, TextPhase phase, int first, int last);

    [[nodiscard]] static std::size_t post_mixer_workspace_capacity_bytes(
        WeightsProfile profile, TextPhase phase, int first, int last);

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