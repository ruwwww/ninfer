#include "targets/laguna_xs_2_1/impl/kv_cache.h"

#include "core/layout.h"

#include <algorithm>
#include <stdexcept>

namespace ninfer::targets::laguna_xs_2_1 {
namespace {

inline std::size_t kv_layer_bytes(std::int32_t num_kv_heads, std::int32_t head_dim,
                                   std::uint32_t capacity, DType dtype) {
    std::size_t elem_bytes = (dtype == DType::FP8_E4M3) ? sizeof(std::uint8_t) : sizeof(__nv_bfloat16);
    return static_cast<std::size_t>(capacity) * static_cast<std::size_t>(num_kv_heads) *
           static_cast<std::size_t>(head_dim) * elem_bytes;
}

inline std::vector<LayoutRegion> make_kv_regions(LayoutBuilder& builder,
                                                   std::int32_t layer_count,
                                                   std::int32_t num_kv_heads,
                                                   std::int32_t head_dim,
                                                   std::uint32_t capacity,
                                                   DType dtype) {
    std::vector<LayoutRegion> regions(layer_count);
    for (int i = 0; i < layer_count; ++i) {
        regions[i] = builder.alloc_tensor({head_dim, capacity, num_kv_heads}, dtype);
    }
    return regions;
}

inline std::vector<LayoutRegion> make_scale_regions(LayoutBuilder& builder,
                                                     std::int32_t layer_count,
                                                     std::int32_t num_kv_heads,
                                                     std::uint32_t capacity,
                                                     std::int32_t quant_group) {
    // FP8 scale: one scale per (quant_group, kv_head) per token position
    // For cyclic cache: capacity positions; for SWA: window_size positions
    std::int32_t scale_len = (capacity + quant_group - 1) / quant_group * num_kv_heads;
    std::vector<LayoutRegion> regions(layer_count);
    for (int i = 0; i < layer_count; ++i) {
        regions[i] = builder.alloc_tensor({scale_len, 1}, DType::FP32);
    }
    return regions;
}

}  // namespace

std::size_t LagunaKVCacheLayout::payload_bytes() const noexcept {
    std::size_t total = 0;
    auto add = [&](const std::vector<LayoutRegion>& regions) {
        for (auto& r : regions) total += r.size_bytes;
    };
    add(global_k); add(global_v);
    add(global_k_scale); add(global_v_scale);
    add(swa_k); add(swa_v);
    add(swa_k_scale); add(swa_v_scale);
    return total;
}

LagunaKVCacheLayout plan_laguna_kv_cache(
    LayoutBuilder& builder,
    std::uint32_t max_context,
    std::uint32_t sliding_window,
    std::int32_t num_kv_heads,
    std::int32_t head_dim,
    DType dtype,
    std::int32_t quant_group)
{
    if (num_kv_heads <= 0 || head_dim <= 0) {
        throw std::invalid_argument("Laguna KV cache: invalid dimensions");
    }

    LagunaKVCacheLayout layout;
    layout.max_context       = max_context;
    layout.sliding_window    = sliding_window;
    layout.num_kv_heads      = num_kv_heads;
    layout.head_dim          = head_dim;
    layout.dtype             = dtype;
    layout.quant_group       = quant_group;

    // Global layers: 10 layers (0, 4, 8, ..., 36) with full context capacity
    layout.global_k     = make_kv_regions(builder, kLagunaGlobalLayers, num_kv_heads, head_dim,
                                           max_context, dtype);
    layout.global_v     = make_kv_regions(builder, kLagunaGlobalLayers, num_kv_heads, head_dim,
                                           max_context, dtype);
    layout.global_k_scale = make_scale_regions(builder, kLagunaGlobalLayers, num_kv_heads,
                                                max_context, quant_group);
    layout.global_v_scale = make_scale_regions(builder, kLagunaGlobalLayers, num_kv_heads,
                                                max_context, quant_group);

    // SWA layers: 30 layers with window-sized circular buffer
    layout.swa_k     = make_kv_regions(builder, kLagunaSWALayers, num_kv_heads, head_dim,
                                        sliding_window, dtype);
    layout.swa_v     = make_kv_regions(builder, kLagunaSWALayers, num_kv_heads, head_dim,
                                        sliding_window, dtype);
    layout.swa_k_scale = make_scale_regions(builder, kLagunaSWALayers, num_kv_heads,
                                             sliding_window, quant_group);
    layout.swa_v_scale = make_scale_regions(builder, kLagunaSWALayers, num_kv_heads,
                                             sliding_window, quant_group);

    return layout;
}

LagunaKVCache::LagunaKVCache(DeviceSpan backing, const LagunaKVCacheLayout& layout)
    : max_context(layout.max_context),
      sliding_window(layout.sliding_window),
      num_kv_heads(layout.num_kv_heads),
      head_dim(layout.head_dim),
      dtype(layout.dtype),
      quant_group(layout.quant_group),
      backing(backing)
{
    global_views.resize(kLagunaGlobalLayers);
    swa_views.resize(kLagunaSWALayers);
    layers.resize(kLagunaLayers);

    // Fill global views
    for (int i = 0; i < kLagunaGlobalLayers; ++i) {
        auto& gv = global_views[i];
        gv.k          = Tensor(backing, layout.global_k[i].offset,
                               {head_dim, max_context, num_kv_heads}, dtype);
        gv.v          = Tensor(backing, layout.global_v[i].offset,
                               {head_dim, max_context, num_kv_heads}, dtype);
        gv.k_scale    = Tensor(backing, layout.global_k_scale[i].offset,
                                {(max_context + quant_group - 1) / quant_group * num_kv_heads, 1}, DType::FP32);
        gv.v_scale    = Tensor(backing, layout.global_v_scale[i].offset,
                                {(max_context + quant_group - 1) / quant_group * num_kv_heads, 1}, DType::FP32);
        gv.capacity        = max_context;
        gv.padded_capacity = max_context;
        gv.num_kv_heads    = num_kv_heads;
        gv.head_dim        = head_dim;
        gv.dtype           = dtype;
        gv.quant_group     = quant_group;

        // Assign to the correct layer index (0, 4, 8, ...)
        int layer_idx = i * 4;
        layers[layer_idx].is_swa = false;
        layers[layer_idx].ptr.global = &global_views[i];
    }

    // Fill SWA views
    for (int i = 0; i < kLagunaSWALayers; ++i) {
        auto& sv = swa_views[i];
        // Find the next SWA layer index
        int layer_idx = 0;
        while (layer_idx < kLagunaLayers && is_laguna_global_layer(layer_idx)) {
            ++layer_idx;
        }
        if (layer_idx >= kLagunaLayers) break;

        sv.k          = Tensor(backing, layout.swa_k[i].offset,
                                {head_dim, sliding_window, num_kv_heads}, dtype);
        sv.v          = Tensor(backing, layout.swa_v[i].offset,
                                {head_dim, sliding_window, num_kv_heads}, dtype);
        sv.k_scale    = Tensor(backing, layout.swa_k_scale[i].offset,
                                {(sliding_window + quant_group - 1) / quant_group * num_kv_heads, 1},
                                DType::FP32);
        sv.v_scale    = Tensor(backing, layout.swa_v_scale[i].offset,
                                {(sliding_window + quant_group - 1) / quant_group * num_kv_heads, 1},
                                DType::FP32);
        sv.window_size = sliding_window;
        sv.num_kv_heads = num_kv_heads;
        sv.head_dim    = head_dim;
        sv.dtype       = dtype;
        sv.quant_group = quant_group;

        layers[layer_idx].is_swa = true;
        layers[layer_idx].ptr.swa = &swa_views[i];

        // Advance to next SWA layer
        ++layer_idx;
    }
}

std::uint32_t LagunaKVCache::layer_count() const noexcept {
    return static_cast<std::uint32_t>(layers.size());
}

LagunaKVCache::LayerView LagunaKVCache::layer_view(std::uint32_t layer) const {
    if (layer >= layers.size()) {
        throw std::out_of_range("Laguna KV cache: layer index out of range");
    }
    return layers[layer];
}

}  // namespace ninfer::targets::laguna_xs_2_1