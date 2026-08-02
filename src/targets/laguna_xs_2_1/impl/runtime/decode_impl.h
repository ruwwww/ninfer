#pragma once

// Laguna-specific decode_impl.h stub
// Laguna does not support speculative decoding; ordinary_round uses target_verify.

#include "targets/qwen3_6/impl/runtime/instance.h"
#include "targets/qwen3_6/impl/runtime/schedule.h"

#include "ninfer/ops/sampling.h"
#include "ninfer/ops/scalar.h"

#include <stdexcept>

namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS::schedule {
namespace {

auto ordinary_body(State& state, bool /*align_mtp*/, ops::GqaExecutionEnvelope envelope) {
    auto record = [&state, envelope] {
        TextContext card(state.device, state.model, state.work, state.text_kv, state.gdn, state.io,
                         state.prefill_hidden, state.prefill_chunk, state.text_kv_base,
                         nullptr);
        configure_text_card(card, state);

        Tensor verify_id = state.io.speculative.target_input_ids.slice(0, 0, 1);
        Tensor position  = state.io.speculative.target_positions.slice(0, 0, 1);
        ops::assign_i32_scalar(state.io.token, verify_id, state.device.stream);
        ops::assign_i32_scalar(state.io.pos, position, state.device.stream);
        target_verify(card, state, verify_id, position, envelope);

        Tensor logits = state.io.logits.slice(1, 0, 1);
        ops::sample(logits, state.io.token, TextConfig::token_domain, state.sampling,
                    static_cast<const std::int32_t*>(state.io.pos.data), ops::kSamplePurposeDecode,
                    state.work, state.device.stream);

        ops::increment_i32_scalar(state.io.pos, state.device.stream);
        ops::increment_i32_scalar(state.io.rope_pos, state.device.stream);
        ops::set_i32_scalar(state.io.gdn_initial_slot, 0, state.device.stream);
    };
    return record;
}

} // namespace

void warm_capture_ordinary_round(State& state, bool /*align_mtp*/, ops::GqaExecutionEnvelope envelope,
                                  const GraphPrepare& prepare, DecodeGraph& graph) {
    auto body = ordinary_body(state, false, envelope);
    warm_capture(state, graph, prepare, body);
}

void ordinary_round(State& state, bool /*align_mtp*/, ops::GqaExecutionEnvelope envelope,
                     DecodeGraph* graph) {
    auto body = ordinary_body(state, false, envelope);
    run_prepared(state, graph, body);
}

} // namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS::schedule