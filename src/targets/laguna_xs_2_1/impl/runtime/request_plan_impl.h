#pragma once

// Laguna-specific request_plan_impl.h

#include "targets/qwen3_6/impl/runtime/instance.h"
#include "targets/qwen3_6/impl/runtime/program.h"
#include "targets/qwen3_6/impl/runtime/schedule.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS {
namespace {

void validate_sampling(const SamplingParameters& sampling) {
    if (!std::isfinite(sampling.temperature) || !std::isfinite(sampling.top_p) ||
        !std::isfinite(sampling.min_p) || !std::isfinite(sampling.presence_penalty) ||
        !std::isfinite(sampling.frequency_penalty)) {
        throw std::invalid_argument("sampling parameters must be finite");
    }
    if (sampling.top_p < 0.0F || sampling.top_p > 1.0F) {
        throw std::invalid_argument("top_p must be in [0,1]");
    }
    if (sampling.min_p < 0.0F || sampling.min_p > 1.0F) {
        throw std::invalid_argument("min_p must be in [0,1]");
    }
}

ops::SamplingConfig translate_sampling(const SamplingParameters& source) {
    ops::SamplingConfig out;
    out.temperature       = source.temperature;
    out.top_k             = source.top_k;
    out.top_p             = source.top_p;
    out.min_p             = source.min_p;
    out.presence_penalty  = source.presence_penalty;
    out.frequency_penalty = source.frequency_penalty;
    out.seed              = source.seed;
    out.token_counts      = nullptr;
    return out;
}

} // namespace

RequestPlan ProgramImplCore::plan_request(const PreparedPromptData& prompt,
                                           const ExecutionOptions& options) const {
    if (lifecycle == Lifecycle::Active || lifecycle == Lifecycle::Pending) {
        throw std::logic_error("cannot plan a request while Program is active or pending");
    }
    if (prompt.token_ids.empty()) { throw std::invalid_argument("prompt must contain tokens"); }
    if (prompt.token_ids.size() > capacity) {
        throw std::invalid_argument("prompt exceeds configured context capacity");
    }
    if (prompt.token_ids.size() > std::numeric_limits<std::uint32_t>::max()) {
        throw std::overflow_error("prompt token count exceeds uint32");
    }
    for (const TokenId id : prompt.token_ids) {
        if (id < 0 || id >= TextConfig::token_domain) {
            throw std::invalid_argument("prompt contains token outside the vocabulary domain");
        }
    }
    if (prompt.token_types.size() != prompt.token_ids.size() ||
        prompt.positions.size() != 3ULL * prompt.token_ids.size()) {
        throw std::invalid_argument("prepared prompt token metadata has an invalid shape");
    }
    if (prompt.has_media() != !prompt.patches.empty()) {
        throw std::invalid_argument("prepared prompt media payload is incomplete");
    }
    if (prompt.has_media() && !vision_enabled) {
        throw std::invalid_argument("Vision is disabled for this Engine");
    }
    validate_sampling(options.sampling);

    auto plan                             = std::make_unique<RequestPlanImpl>();
    plan->summary.prompt_tokens           = static_cast<std::uint32_t>(prompt.token_ids.size());
    plan->summary.requested_output_tokens = options.requested_output_tokens;
    const std::uint32_t capacity_output =
        capacity - plan->summary.prompt_tokens + static_cast<std::uint32_t>(1);
    plan->summary.effective_output_tokens =
        std::min(options.requested_output_tokens, capacity_output);
    plan->summary.effective_limit_reason = options.requested_output_tokens <= capacity_output
                                            ? FinishReason::OutputLimit
                                            : FinishReason::ContextCapacity;
    plan->summary.transient_alignment    = 1;
    plan->summary.transient_bytes        = 0;
    plan->sampling                       = translate_sampling(options.sampling);

    if (options.allow_prefix_reuse && prompt.identity.reusable &&
        lifecycle == Lifecycle::Resident) {
        if (E != 0 &&
            qwen3_6::detail::prefix_matches(prompt, ledger, prefix_identity, E)) {
            plan->reuse      = ReusePath::AppendAtFrontier;
            plan->reuse_base = E;
        } else if (boundary.valid && boundary.boundary != 0 &&
                   boundary.boundary < prompt.token_ids.size() &&
                   qwen3_6::detail::prefix_matches(prompt, ledger, prefix_identity,
                                                    boundary.boundary)) {
            plan->reuse      = ReusePath::RestoreBoundary;
            plan->reuse_base = boundary.boundary;
        }
    }

    plan->summary.reusable_prompt_tokens = plan->reuse_base;
    // Laguna does not support MTP or DFlash speculative decoding

    if (prompt.identity.assistant_content_boundary) {
        const std::uint32_t candidate = *prompt.identity.assistant_content_boundary;
        if (candidate > plan->reuse_base && candidate <= plan->summary.prompt_tokens) {
            plan->snapshot_boundary = candidate;
        }
    }
    return RequestPlan(std::move(plan));
}

} // namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS