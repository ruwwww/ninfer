#pragma once

// Laguna XS 2.1 runtime layout planning.
//
// Provides SequencePlanImpl<Variant> specialization for the shared
// qwen3_6 runtime template infrastructure.

#include "targets/laguna_xs_2_1/impl/config.h"
#include "targets/laguna_xs_2_1/impl/variant.h"

#include "core/layout.h"
#include "ninfer/targets/qwen3_6/runtime.h"
#include "ninfer/types.h"

#include <cstddef>
#include <cstdint>
#include <memory>

namespace ninfer::targets::laguna_runtime {

// Minimal PersistentLayout for Laguna (no GDN, no MTP, no Vision, no DFlash).
struct PersistentLayout {
    std::size_t bytes            = 0;
    std::size_t kv_payload_bytes = 0;
};

// Minimal WorkspacePlan for Laguna.
struct WorkspacePlan {
    std::size_t text_prefill    = 0;
    std::size_t ordinary_round  = 0;
    std::size_t capacity        = 0;
};

} // namespace ninfer::targets::laguna_runtime

namespace ninfer::targets::qwen3_6::detail {

// Laguna-specific SequencePlanImpl specialization
template <>
struct SequencePlanImpl< ::ninfer::targets::laguna_xs_2_1::detail::Variant> {
    ::ninfer::targets::laguna_xs_2_1::detail::WeightsProfile weights_profile;
    std::uint32_t capacity                 = 0;
    std::uint32_t prefill_chunk            = 0;
    std::uint32_t draft_window             = 0;
    SpeculativeBackend speculative_backend = SpeculativeBackend::None;
    DType kv_dtype                         = DType::BF16;
    std::int32_t kv_quant_group            = 0;
    ProposalHead proposal_head             = ProposalHead::Full;
    StartupFeatures features;
    bool use_cuda_graph = true;
    int device          = 0;
    laguna_runtime::PersistentLayout persistent;
    laguna_runtime::WorkspacePlan workspace;
    std::size_t request_transient_capacity_bytes = 0;
    std::size_t graph_allowance_bytes            = 0;
    std::size_t device_reservation_bytes         = 0;
};

} // namespace ninfer::targets::qwen3_6::detail

namespace ninfer::targets::laguna_xs_2_1 {

inline constexpr std::int32_t kKvQuantGroup = 64;

/**
 * Plan the full sequence layout for Laguna XS 2.1.
 *
 * Returns a qwen3_6::SequencePlan<Variant> suitable for constructing
 * the shared Program instance.
 */
inline qwen3_6::SequencePlan<detail::Variant> plan_sequence(/*DeviceContext& device,**/
                                                    const EngineOptions& options,
                                                    detail::WeightsProfile weights_profile) {
    using SeqImpl = qwen3_6::detail::SequencePlanImpl<detail::Variant>;

    auto impl = std::make_unique<SeqImpl>();
    impl->weights_profile = weights_profile;
    impl->capacity = options.max_context;
    impl->prefill_chunk = options.prefill_chunk;
    impl->draft_window = 0;
    impl->speculative_backend = SpeculativeBackend::None;
    impl->kv_dtype = DType::BF16;
    impl->kv_quant_group = kKvQuantGroup;
    impl->proposal_head = ProposalHead::Full;
    impl->features.vision = false;
    impl->features.speculative = SpeculativeBackend::None;
    impl->use_cuda_graph = true;
    impl->device = 0;
    impl->persistent = laguna_runtime::PersistentLayout{};
    impl->workspace = laguna_runtime::WorkspacePlan{};
    impl->request_transient_capacity_bytes = 0;
    impl->graph_allowance_bytes = 0;
    impl->device_reservation_bytes = 0;

    return qwen3_6::SequencePlan<detail::Variant>(std::move(impl));
}

}  // namespace ninfer::targets::laguna_xs_2_1