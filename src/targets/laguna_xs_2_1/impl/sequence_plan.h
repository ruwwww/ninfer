#pragma once

// Laguna XS 2.1 sequence planning: KV cache layout, arena allocation, and
// per-layer geometry dispatch for the mixed global/SWA attention pattern.

#include "targets/laguna_xs_2_1/impl/config.h"
#include "targets/laguna_xs_2_1/impl/kv_cache.h"
#include "targets/laguna_xs_2_1/impl/variant.h"

#include "core/device.h"
#include "core/layout.h"
#include "runtime/engine_options.h"

#include <cstdint>
#include <cstddef>

namespace ninfer::targets::laguna_xs_2_1 {

/**
 * Sequence plan for Laguna XS 2.1.
 *
 * Contains:
 * - KV cache layout (mixed global/SWA)
 * - Arena allocation for transient workspace
 * - Per-layer geometry information for the Program scheduler
 */
struct SequencePlan {
    std::uint32_t capacity() const noexcept;
    std::size_t device_reservation_bytes() const noexcept;
    std::size_t request_transient_capacity_bytes() const noexcept;

    // ---- Configuration ----
    std::uint32_t max_context       = 0;
    std::uint32_t sliding_window    = detail::TextConfig::sliding_window;

    // ---- KV cache ----
    LagunaKVCacheLayout kv_layout;
    std::size_t kv_cache_bytes = 0;

    // ---- Arena ----
    // Peak workspace needed across all layers during a decode round
    std::size_t peak_workspace_bytes = 0;

private:
    std::size_t reservation_bytes_ = 0;
};

/**
 * Plan the full sequence layout for Laguna XS 2.1.
 *
 * This computes:
 * 1. KV cache layout (10 global + 30 SWA layers)
 * 2. Workspace capacity estimates from Variant methods
 * 3. Total device reservation
 */
inline SequencePlan plan_sequence(DeviceContext& device,
                                   const EngineOptions& options,
                                   detail::WeightsProfile /*profile*/)
{
    SequencePlan plan;
    plan.max_context = options.max_context;

    // Plan KV cache layout
    LayoutBuilder builder(device);
    plan.kv_layout = plan_laguna_kv_cache(
        builder,
        options.max_context,
        detail::TextConfig::sliding_window,
        detail::TextConfig::kv_heads,
        detail::TextConfig::head_dim,
        DType::BF16,  // TODO: FP8 support when quantized KV cache is implemented
        kKvQuantGroup
    );
    plan.kv_cache_bytes = plan.kv_layout.payload_bytes();

    // Estimate workspace capacity from the largest layer (SWA with 64 heads)
    // This is a conservative estimate; actual per-layer workspace varies
    plan.peak_workspace_bytes =
        detail::Variant::attention_projection_workspace_capacity_bytes(
            detail::WeightsProfile::kGroupwiseInt,
            TextPhase::Decode,
            1, static_cast<int>(options.max_context)) +
        detail::Variant::post_mixer_workspace_capacity_bytes(
            detail::WeightsProfile::kGroupwiseInt,
            TextPhase::Decode,
            1, 1);

    // Total reservation: KV cache + one-time graph overhead + workspace
    plan.reservation_bytes_ = plan.kv_cache_bytes +
                              (plan.peak_workspace_bytes * 2);  // buffer for peaks

    return plan;
}

}  // namespace ninfer::targets::laguna_xs_2_1