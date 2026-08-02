#pragma once

// Exact grouped-query head geometries served by the Qwen3.6 GQA kernels. Head
// dimension, cache format, and tile policy are shared; head mapping remains a
// compile-time property so each registered shape gets an independent kernel.
//
// Laguna XS 2.1 adds <48,8> and <64,8> geometries with head_dim=128.

namespace ninfer::ops {

template <int QHeadsValue, int KVHeadsValue, int DecodeSplitScaleValue>
struct GqaGeometry {
    static_assert(QHeadsValue > 0 && KVHeadsValue > 0);
    static_assert(QHeadsValue % KVHeadsValue == 0);
    static_assert(DecodeSplitScaleValue > 0);

    static constexpr int QHeads           = QHeadsValue;
    static constexpr int KVHeads          = KVHeadsValue;
    static constexpr int GroupSize        = QHeads / KVHeads;
    static constexpr int DecodeSplitScale = DecodeSplitScaleValue;
    static constexpr int DecodeSplits     = 85 * DecodeSplitScale;
};

using Gqa27Geometry = GqaGeometry<24, 4, 1>;
using Gqa35Geometry = GqaGeometry<16, 2, 2>;

// Laguna XS 2.1 full-attention layers: 48 query heads, 8 KV heads
using GqaLagunaFullGeometry = GqaGeometry<48, 8, 1>;

// Laguna XS 2.1 sliding-window layers: 64 query heads, 8 KV heads
using GqaLagunaSwaGeometry = GqaGeometry<64, 8, 1>;

} // namespace ninfer::ops
