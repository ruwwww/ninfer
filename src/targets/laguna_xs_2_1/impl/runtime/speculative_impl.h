#pragma once

// Laguna-specific speculative_impl.h stub
// Provides target_verify for use by decode_impl.h.

#include "targets/qwen3_6/impl/runtime/instance.h"
#include "targets/qwen3_6/impl/runtime/text_context.h"
#include "targets/qwen3_6/impl/runtime/schedule.h"

namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS::schedule {

void target_verify(TextContext& card, State& /*state*/, const Tensor& ids, const Tensor& positions,
                    ops::GqaExecutionEnvelope envelope) {
    card.target_verify(ids, positions, envelope);
}

} // namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS::schedule