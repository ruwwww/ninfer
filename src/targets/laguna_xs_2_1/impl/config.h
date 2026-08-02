#pragma once

#include <array>
#include <cmath>
#include <string_view>

// Compile-time model geometry for Laguna XS 2.1.
//
// Laguna uses a 1:3 alternating pattern of global (full) attention and
// sliding-window attention (SWA), with per-layer varying head counts.
// Layer 0 uses a dense SwiGLU MLP; layers 1-39 use MoE with 256 experts.

namespace ninfer::targets::laguna_xs_2_1::detail {

struct TextConfig {
    // ---- Identity ----
    static constexpr std::string_view model_id   = "laguna-xs-2.1";
    static constexpr std::string_view target_key = "laguna_xs_2_1";

    // ---- Shared dimensions ----
    static constexpr int hidden            = 2048;
    static constexpr int layers            = 40;
    static constexpr int head_dim          = 128;
    static constexpr int kv_heads          = 8;
    static constexpr int moe_intermediate  = 512;       // per-expert FFN intermediate dim
    static constexpr int shared_expert_int = 512;
    static constexpr int num_experts       = 256;
    static constexpr int topk              = 8;
    static constexpr int sliding_window    = 512;
    static constexpr int maximum_context   = 262144;
    static constexpr float rms_norm_eps    = 1e-6F;
    static constexpr float moe_scaling     = 2.5F;

    // ---- Vocabulary ----
    static constexpr int output_rows = 100352;

    // ---- Per-layer head counts ----
    static constexpr int full_q_heads = 48;    // global/full attention layers
    static constexpr int swa_q_heads  = 64;    // sliding window attention layers

    // ---- Derived dimensions ----
    static constexpr int full_q_rows = full_q_heads * head_dim;   // 6144
    static constexpr int swa_q_rows  = swa_q_heads * head_dim;   // 8192
    static constexpr int kv_rows     = kv_heads * head_dim;      // 1024

    // ---- Layer-type classification ----
    // Global/full-attention layers: 0, 4, 8, ..., 36
    // Sliding-window layers: all others
    [[nodiscard]] static constexpr bool is_full_attention(int layer) noexcept {
        return (layer % 4) == 0;
    }

    [[nodiscard]] static constexpr bool is_swa(int layer) noexcept {
        return (layer % 4) != 0;
    }

    // Dense MLP: only layer 0
    [[nodiscard]] static constexpr bool is_dense_mlp(int layer) noexcept {
        return layer == 0;
    }

    // MoE layers: 1-39
    [[nodiscard]] static constexpr bool is_moe(int layer) noexcept {
        return layer > 0;
    }

    // ---- Attention query heads for a given layer ----
    [[nodiscard]] static constexpr int q_heads(int layer) noexcept {
        return is_full_attention(layer) ? full_q_heads : swa_q_heads;
    }

    // ---- Attention query rows for a given layer ----
    [[nodiscard]] static constexpr int q_rows(int layer) noexcept {
        return is_full_attention(layer) ? full_q_rows : swa_q_rows;
    }

    // ---- Full-attention layer indices ----
    static constexpr std::array<int, 10> full_attention_layers = []() constexpr {
        std::array<int, 10> out{};
        for (int i = 0; i < 10; ++i) out[i] = i * 4;  // 0, 4, 8, ..., 36
        return out;
    }();

    static constexpr int k_full_attention_count = 10;

    // ---- MoE layer indices (1-39) ----
    static constexpr std::array<int, 39> moe_layers = []() constexpr {
        std::array<int, 39> out{};
        for (int i = 0; i < 39; ++i) out[i] = i + 1;
        return out;
    }();

    static constexpr int k_moe_count = 39;

    // ---- RoPE configuration ----
    // Full attention: YARN-extrapolated, theta=500000, partial_rotary_factor=0.5
    // SWA: standard, theta=10000, partial_rotary_factor=1.0
    struct RopeConfig {
        // Full attention RoPE
        static constexpr float full_theta      = 500000.0F;
        static constexpr float full_factor     = 32.0F;
        static constexpr int   full_orig_ctx   = 8192;
        static constexpr float full_beta_slow  = 1.0F;
        static constexpr float full_beta_fast  = 64.0F;
        static constexpr float full_attn_factor = 1.3465735902799727F;
        static constexpr float full_partial_rf  = 0.5F;

        // SWA RoPE
        static constexpr float swa_theta       = 10000.0F;
        static constexpr float swa_partial_rf  = 1.0F;
    };

    // ---- Attention scale ----
    static constexpr float attention_scale = 1.0F / sqrtf(static_cast<float>(head_dim));
};

}  // namespace ninfer::targets::laguna_xs_2_1::detail