#pragma once

// Laguna-specific text_prefill_impl.h stubs

#include "targets/qwen3_6/impl/runtime/instance.h"
#include "targets/qwen3_6/impl/runtime/schedule.h"
#include "targets/qwen3_6/impl/runtime/text_context.h"

#include "ninfer/ops/scalar.h"
#include "ninfer/ops/linear.h"
#include "ninfer/ops/sampling.h"
#include "ninfer/ops/position.h"

#include <cuda_runtime.h>
#include <optional>
#include <span>
#include <stdexcept>

namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS::schedule {

void configure_text_card(TextContext& card, const State& state) {
    card.set_sampling(state.sampling);
    if (state.proposal_head == ProposalHead::Full) {
        card.set_proposal_head(nullptr, nullptr, 0);
        return;
    }
    if (card.proposal_head() == nullptr || card.proposal_head_ids() == nullptr ||
        card.proposal_head_n() <= 0) {
        throw std::runtime_error("optimized proposal head is unavailable");
    }
}

bool prefill_text(State& state, std::span<const TokenId> ids,
                   std::optional<std::uint32_t> snapshot_boundary, bool /*prepare_mtp*/) {
    (void)snapshot_boundary;
    TextContext card(state.device, state.model, state.work, state.text_kv, state.gdn, state.io,
                      state.prefill_hidden, state.prefill_chunk, state.text_kv_base,
                      nullptr);
    configure_text_card(card, state);

    if (ids.empty()) {
        throw std::invalid_argument("prefill requires tokens");
    }

    // Handle prefix reuse: continue from existing cache
    if (state.text_kv_base > 0 && !ids.empty()) {
        std::vector<int> suffix_ids(ids.begin(), ids.end());
        card.prefill(suffix_ids);
    } else {
        std::vector<int> all_ids(ids.begin(), ids.end());
        card.prefill(all_ids);
    }

    return false;
}

void sample_from_hidden(State& state, const Tensor& hidden, std::int32_t absolute_position,
                         std::int32_t purpose) {
    if (hidden.dtype != DType::BF16 || hidden.ne[0] != TextConfig::hidden || hidden.ne[1] != 1 ||
        hidden.ne[2] != 1 || hidden.ne[3] != 1 || hidden.data == nullptr) {
        throw std::invalid_argument("sample_from_hidden requires BF16 [hidden,1]");
    }
    state.work.reset();
    Tensor logits = state.io.logits.slice(1, 0, 1);
    ops::linear(hidden, state.model.output_head, logits, state.device.stream);
    CUDA_CHECK(cudaMemcpyAsync(state.io.pos.data, &absolute_position, sizeof(absolute_position),
                                cudaMemcpyHostToDevice, state.device.stream));
    ops::sample(logits, state.io.token, TextConfig::token_domain, state.sampling,
                static_cast<const std::int32_t*>(state.io.pos.data), purpose, state.work,
                state.device.stream);
    state.work.reset();
}

} // namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS::schedule