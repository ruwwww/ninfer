#pragma once

// Laguna XS 2.1 hybrid KV cache with per-layer sliding-window and global storage.
//
// Laguna uses a 1:3 alternating pattern: every 4th layer (0, 4, 8, ...) is global
// attention with full-context cyclic caching, while the other 30 layers use sliding
// window attention with a 512-token circular buffer.
//
// This file defines the planning and view types for that mixed layout. FP8 quantization
// is supported via scale tensors stored per token-head group.

#include "core/kv_cache.h"
#include "core/layout.h"
#include "core/tensor.h"

#include <cstdint>
#include <cstddef>
#include <vector>

namespace ninfer::targets::laguna_xs_2_1 {

inline constexpr std::int32_t kLagunaLayers          = 40;
inline constexpr std::int32_t kLagunaGlobalLayers    = 10;  // 0, 4, 8, ..., 36
inline constexpr std::int32_t kLagunaSWALayers       = 30;  // all others
inline constexpr std::int32_t kLagunaKVHeads         = 8;
inline constexpr std::int32_t kLagunaHeadDim         = 128;
inline constexpr std::int32_t kLagunaSlidingWindow   = 512;
inline constexpr std::int32_t kLagunaMaxContext      = 262144;

// ---- Layer-type classification ----

inline constexpr bool is_laguna_global_layer(std::int32_t layer) noexcept {
    return (layer % 4) == 0;
}

inline constexpr bool is_laguna_swa_layer(std::int32_t layer) noexcept {
    return (layer % 4) != 0;
}

// ---- Per-layer cache views ----

/**
 * View for a global-attention layer: standard cyclic cache with full context capacity.
 */
struct LagunaGlobalCacheView {
    Tensor k;
    Tensor v;
    Tensor k_scale;
    Tensor v_scale;
    std::uint32_t capacity        = 0;
    std::uint32_t padded_capacity = 0;
    std::int32_t  num_kv_heads    = 0;
    std::int32_t  head_dim        = 0;
    DType         dtype           = DType::BF16;
    std::int32_t  quant_group     = 0;
};

/**
 * View for a sliding-window layer: circular buffer with fixed window size.
 * Absolute position p maps to physical slot p % window_size.
 */
struct LagunaSWACacheView {
    Tensor k;
    Tensor v;
    Tensor k_scale;
    Tensor v_scale;
    std::uint32_t window_size     = 0;
    std::int32_t  num_kv_heads    = 0;
    std::int32_t  head_dim        = 0;
    DType         dtype           = DType::BF16;
    std::int32_t  quant_group     = 0;
};

// ---- Layout planning ----

/**
 * Plan the physical layout for Laguna's mixed KV cache.
 *
 * Global layers (10): cyclic cache with capacity = max_context per layer.
 * SWA layers (30): circular buffer with capacity = sliding_window per layer.
 *
 * For FP8 KV cache, scale tensors are stored separately. Each scale covers
 * one token per kv-head, grouped by quant_group (default 64).
 */
struct LagunaKVCacheLayout {
    std::uint32_t max_context       = 0;
    std::uint32_t sliding_window    = 0;
    std::int32_t  num_kv_heads      = 0;
    std::int32_t  head_dim          = 0;
    DType         dtype             = DType::BF16;
    std::int32_t  quant_group       = 0;

    // Per-layer regions — indexed by layer number (0..39)
    std::vector<TensorRegion> global_k;
    std::vector<TensorRegion> global_v;
    std::vector<TensorRegion> global_k_scale;
    std::vector<TensorRegion> global_v_scale;

    std::vector<TensorRegion> swa_k;
    std::vector<TensorRegion> swa_v;
    std::vector<TensorRegion> swa_k_scale;
    std::vector<TensorRegion> swa_v_scale;

    [[nodiscard]] std::size_t payload_bytes() const noexcept;
};

/**
 * Plan Laguna KV cache layout within a LayoutBuilder.
 *
 * Returns the computed layout; regions are appended to builder.
 */
[[nodiscard]] LagunaKVCacheLayout plan_laguna_kv_cache(
    LayoutBuilder& builder,
    std::uint32_t max_context,
    std::uint32_t sliding_window,
    std::int32_t num_kv_heads,
    std::int32_t head_dim,
    DType dtype = DType::BF16,
    std::int32_t quant_group = 0);

// ---- Runtime cache ----

/**
 * Laguna KV cache with per-layer views for mixed global/SWA storage.
 *
 * Provides unified access: layer_view(i) returns the appropriate view type
 * based on whether layer i is global or SWA.
 */
struct LagunaKVCache {
    std::uint32_t max_context       = 0;
    std::uint32_t sliding_window    = 0;
    std::int32_t  num_kv_heads      = 0;
    std::int32_t  head_dim          = 0;
    DType         dtype             = DType::BF16;
    std::int32_t  quant_group       = 0;

    // Backing store for all layers
    DeviceSpan backing;

    // Global layer views (indices into layers[] where layer%4==0)
    std::vector<LagunaGlobalCacheView> global_views;
    // SWA layer views (indices into layers[] where layer%4!=0)
    std::vector<LagunaSWACacheView> swa_views;

    // Flat per-layer array for uniform indexing
    struct LayerView {
        bool is_swa = false;
        union PtrUnion {
            LagunaGlobalCacheView* global;
            LagunaSWACacheView* swa;
            PtrUnion() : global(nullptr) {}
        } ptr;
    };
    std::vector<LayerView> layers;

    LagunaKVCache() = default;
    LagunaKVCache(DeviceSpan backing, const LagunaKVCacheLayout& layout);

    std::uint32_t layer_count() const noexcept;
    [[nodiscard]] LayerView layer_view(std::uint32_t layer) const;
};

} // namespace ninfer::targets::laguna_xs_2_1