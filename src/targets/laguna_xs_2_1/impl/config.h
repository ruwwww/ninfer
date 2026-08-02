#pragma once

#include <array>
#include <cmath>
#include <cstdint>
#include <string_view>

// Compile-time model geometry for Laguna XS 2.1.
//
// Laguna uses a 1:3 alternating pattern of global (full) attention and
// sliding-window attention (SWA), with per-layer varying head counts.
// Layer 0 uses a dense SwiGLU MLP; layers 1-39 use MoE with 256 experts.

namespace ninfer::targets::laguna_xs_2_1::detail {

struct TextConfig {
    // ---- Shared dimensions ----
    static constexpr int hidden            = 2048;
    static constexpr int layers            = 40;
    static constexpr int head_dim          = 128;
    static constexpr int kv_heads          = 8;
    static constexpr int moe_intermediate  = 512;
    static constexpr int shared_expert_int = 512;
    static constexpr int num_experts       = 256;
    static constexpr int topk              = 8;
    static constexpr int sliding_window    = 512;
    static constexpr int maximum_context   = 262144;
    static constexpr float rms_epsilon     = 1e-6F;
    // Alias expected by shared code
    static constexpr float rms_norm_eps    = 1e-6F;
    static constexpr float moe_scaling     = 2.5F;

    // ---- Vocabulary ----
    static constexpr int output_rows = 100352;
    static constexpr int token_domain = output_rows;

    // ---- Per-layer head counts ----
    static constexpr int full_q_heads = 48;
    static constexpr int swa_q_heads  = 64;

    // ---- Derived dimensions ----
    static constexpr int full_q_rows = full_q_heads * head_dim;
    static constexpr int swa_q_rows  = swa_q_heads * head_dim;
    static constexpr int kv_rows     = kv_heads * head_dim;

    // ---- Intermediate dimension ----
    static constexpr int intermediate = 8192;

    // ---- GDN fields — all zero because Laguna has no GDN layers ----
    static constexpr int gdn_conv_kernel         = 0;
    static constexpr int gdn_conv_state_width    = 0;
    static constexpr int gdn_key_heads           = 0;
    static constexpr int gdn_key_head_dim        = 0;
    static constexpr int gdn_value_heads         = 0;
    static constexpr int gdn_value_head_dim      = 0;
    static constexpr int key_dim                 = gdn_key_heads * gdn_key_head_dim;
    static constexpr int value_dim               = gdn_value_heads * gdn_value_head_dim;
    static constexpr int convolution_dim         = 2 * key_dim + value_dim;

    // ---- Rotary embedding ----
    static constexpr int rotary_dim              = 128;
    static constexpr float rope_theta            = 500000.0F;

    // ---- Query/key projection dimensions ----
    static constexpr int query_size              = swa_q_heads * head_dim;
    // Alias expected by shared code (uses max query heads across layer types)
    static constexpr int query_heads             = swa_q_heads;
    static constexpr int kv_size                 = kv_heads * head_dim;
    static constexpr int query_projection_rows   = 2 * query_size;

    // ---- MTP fields — zero because Laguna does not support MTP ----
    static constexpr int mtp_layers              = 0;
    static constexpr int mtp_input_rows          = 0;
    static constexpr int mtp_attention_input_rows = 0;
    static constexpr int mtp_mlp_gate_up_rows    = 0;

    // ---- Attention scale ----
    static constexpr float attention_scale = 0.08838834764831845F;

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

    // ---- Shared runtime helper methods ----

    [[nodiscard]] static constexpr int full_attention_layers() noexcept {
        return 10;
    }

    [[nodiscard]] static constexpr int gdn_layers() noexcept {
        return 0;
    }

    [[nodiscard]] static constexpr int full_attention_index(int layer) noexcept {
        return layer / 4;
    }

    [[nodiscard]] static constexpr int gdn_index(int layer) noexcept {
        (void)layer;
        return 0;
    }

    // ---- RoPE configuration (Laguna-specific, not used by shared code) ----
    struct RopeConfig {
        static constexpr float full_theta      = 500000.0F;
        static constexpr float full_factor     = 32.0F;
        static constexpr int   full_orig_ctx   = 8192;
        static constexpr float full_beta_slow  = 1.0F;
        static constexpr float full_beta_fast  = 64.0F;
        static constexpr float full_attn_factor = 1.3465735902799727F;
        static constexpr float full_partial_rf  = 0.5F;
        static constexpr float swa_theta       = 10000.0F;
        static constexpr float swa_partial_rf  = 1.0F;
    };
};

// Laguna does not support Vision — stub fields required by shared runtime headers
struct VisionConfig {
    static constexpr int layers              = 0;
    static constexpr int hidden              = 0;
    static constexpr int intermediate        = 0;
    static constexpr int heads               = 0;
    static constexpr int head_dim            = 0;
    static constexpr int value_heads         = 0;
    static constexpr int value_head_dim      = 0;
    static constexpr int key_head_dim        = 0;
    static constexpr int conv_state_width    = 0;
    static constexpr int output_hidden       = 0;
    // Additional VisionContext fields expected by shared code
    static constexpr int patch_dim           = 0;
    static constexpr int merge_unit          = 0;
    static constexpr int merger_hidden       = 0;
    static constexpr int position_embeddings = 0;
    static constexpr int rotary_dim          = 0;
    static constexpr float rope_theta        = 0.0F;
    static constexpr float norm_epsilon      = 1e-6F;
};

// Laguna does not support DFlash
struct DFlashConfig {
    static constexpr bool supported     = false;
    static constexpr int local_layers   = 0;
    static constexpr int local_capacity = 0;
    static constexpr int kv_heads       = 0;
    static constexpr int head_dim       = 0;
    static constexpr int feature_rows   = 0;
    static constexpr int hidden         = 0;
    static constexpr int intermediate   = 0;
    static constexpr int query_size     = 0;
    static constexpr int kv_size        = 0;
};

inline constexpr std::uint32_t kPrefillChunkAlignment    = 128;
inline constexpr std::uint32_t kMaximumMtpDraftTokens    = 0;
inline constexpr std::uint32_t kMaximumDFlashDraftTokens = 0;

}  // namespace ninfer::targets::laguna_xs_2_1::detail