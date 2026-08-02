#pragma once

// Laguna-specific program_impl.h
// Provides ProgramImplCore method definitions for Laguna.

#include "targets/qwen3_6/impl/runtime/instance.h"
#include "targets/qwen3_6/impl/runtime/program.h"
#include "targets/qwen3_6/impl/runtime/schedule.h"

#include "core/nvtx.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS {
namespace {

using Clock = std::chrono::steady_clock;

std::int32_t checked_i32(std::uint32_t value, const char* label) {
    if (value > static_cast<std::uint32_t>(std::numeric_limits<std::int32_t>::max())) {
        throw std::overflow_error(label);
    }
    return static_cast<std::int32_t>(value);
}

template <typename Variant>
Variant& select_graph_variant(std::vector<Variant>& variants, std::uint32_t frontier,
                              const char* label) {
    const auto it = std::lower_bound(variants.begin(), variants.end(), frontier,
                                      [](const Variant& variant, std::uint32_t value) {
                                          return variant.max_execution_frontier < value;
                                      });
    if (it == variants.end() || frontier < it->min_execution_frontier) {
        throw std::logic_error(std::string(label) + " CUDA Graph coverage is incomplete");
    }
    return *it;
}

void validate_graph_ranges(const std::vector<GraphFrontierRange>& ranges,
                           std::uint32_t max_frontier, const char* label) {
    if (ranges.empty() || ranges.front().min != 0 || ranges.back().max != max_frontier) {
        throw std::logic_error(std::string(label) + " CUDA Graph coverage has invalid endpoints");
    }
    for (std::size_t i = 0; i < ranges.size(); ++i) {
        if (ranges[i].min > ranges[i].max || (i != 0 && ranges[i].min != ranges[i - 1].max + 1)) {
            throw std::logic_error(std::string(label) + " CUDA Graph coverage has a gap");
        }
    }
}

std::uint32_t final_prefill_chunk_length(std::uint32_t base, std::uint32_t end, std::uint32_t chunk,
                                          std::optional<std::uint32_t> boundary) {
    std::uint32_t cursor = base;
    std::uint32_t last   = 0;
    while (cursor < end) {
        last = std::min(chunk, end - cursor);
        if (boundary && *boundary > cursor && *boundary < cursor + last) {
            last = *boundary - cursor;
        }
        cursor += last;
    }
    if (last == 0) { throw std::logic_error("prefill suffix is empty"); }
    return last;
}

} // namespace

ProgramImplCore::ProgramImplCore(const LoadedModelData& model_in, const SequencePlanImpl& plan,
                                  DeviceContext& device_in)
    : model(model_in), device(device_in), capacity(plan.capacity),
      prefill_chunk(plan.prefill_chunk), draft_window(plan.draft_window),
      speculative_backend(SpeculativeBackend::None), kv_dtype(plan.kv_dtype),
      kv_quant_group(plan.kv_quant_group), proposal_head(ProposalHead::Full),
      vision_enabled(false), use_cuda_graph(plan.use_cuda_graph),
      kv_payload_bytes(plan.persistent.kv_payload_bytes),
      graph_allowance_bytes(plan.graph_allowance_bytes), workspace_plan(plan.workspace),
      persistent(plan.persistent.bytes), workspace_storage(plan.workspace.capacity),
      work(DeviceSpan{workspace_storage.base(), workspace_storage.capacity()}),
      round_host((static_cast<std::size_t>(draft_window) + 2ULL) * sizeof(std::int32_t)) {
    if (model.weights_arena == nullptr) {
        throw std::invalid_argument("Laguna model view has no owning weight arena");
    }
    if (model.features != plan.features) {
        throw std::invalid_argument(
            "Laguna loaded weights do not match the frozen startup features");
    }
    const DeviceSpan backing = persistent.alloc_bytes(plan.persistent.bytes, 256);
    decoder = std::make_unique<qwen3_6::DecoderState>(backing, plan.persistent.decoder);

    io = qwen3_6::RoundState(backing, plan.persistent.round);
    prefill_hidden  = plan.persistent.prefill_hidden.bind(backing);
    token_counts    = plan.persistent.token_counts.bind(backing);
    sampling_config = plan.persistent.sampling_config.bind(backing);
    tail_hidden     = plan.persistent.tail_hidden.bind(backing);
    boundary_hidden = plan.persistent.boundary_hidden.bind(backing);

    host_count  = static_cast<std::int32_t*>(round_host.data());
    host_tokens = reinterpret_cast<TokenId*>(host_count + 1);
    ledger.reserve(static_cast<std::size_t>(capacity) + 1ULL);
    prefix_identity.reserve(static_cast<std::size_t>(capacity) + 1ULL);

    CUDA_CHECK(cudaMemsetAsync(io.speculative.produced_count.data, 0,
                                io.speculative.produced_count.bytes(), device.stream));
    CUDA_CHECK(cudaMemsetAsync(io.rope_delta.data, 0, io.rope_delta.bytes(), device.stream));
    CUDA_CHECK(cudaMemsetAsync(io.speculative.accepted_drafts.data, 0,
                                io.speculative.accepted_drafts.bytes(), device.stream));
    CUDA_CHECK(
        cudaMemsetAsync(io.speculative.stats.data, 0, io.speculative.stats.bytes(), device.stream));
    CUDA_CHECK(cudaMemsetAsync(token_counts.data, 0, token_counts.bytes(), device.stream));
    sampling_host = {};
    CUDA_CHECK(cudaMemcpyAsync(sampling_config.data, &sampling_host, sizeof(sampling_host),
                                cudaMemcpyHostToDevice, device.stream));
    device.synchronize();
    prepare_graphs();
    work.reset();
    work.reset_peak();
    workspace_logical_peak_bytes = 0;
}

ProgramImplCore::~ProgramImplCore() noexcept {
    if (device.stream != nullptr) { (void)cudaStreamSynchronize(device.stream); }
}

void ProgramImplCore::make_invalid() noexcept {
    lifecycle = Lifecycle::Invalid;
    E         = 0;
    S         = 0;
    ledger.clear();
    prefix_identity.clear();
    ordinary_tail = false;
    drafts_ready = false;
    tail_hidden_valid = false;
    pending = {};
}

void ProgramImplCore::set_device_i32(Tensor& tensor, std::int32_t value) {
    CUDA_CHECK(
        cudaMemcpyAsync(tensor.data, &value, sizeof(value), cudaMemcpyHostToDevice, device.stream));
}

void ProgramImplCore::ordered_reset() {
    decoder->gdn.reset_running(device.stream);
    work.reset();
    set_device_i32(io.pos, 0);
    set_device_i32(io.rope_pos, 0);
    set_device_i32(io.rope_delta, 0);
    set_device_i32(io.speculative.accepted_drafts, 0);
    ordinary_tail = false;
    drafts_ready = false;
}

void ProgramImplCore::prepare_graphs() {
    if (!use_cuda_graph) { return; }

    std::size_t free_before = 0;
    std::size_t total_bytes = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_before, &total_bytes));

    const auto clear_stable_controls = [&] {
        std::vector<Tensor> controls{
            io.token,
            io.pos,
            io.rope_pos,
            io.rope_delta,
            io.speculative.target_argmax,
            io.speculative.draft_tokens,
            io.speculative.round_tokens,
            io.speculative.produced_count,
            io.speculative.target_input_ids,
            io.speculative.target_positions,
            io.speculative.accepted_drafts,
            io.speculative.stats,
        };
        for (const Tensor& tensor : controls) {
            CUDA_CHECK(cudaMemsetAsync(tensor.data, 0, tensor.bytes(), device.stream));
        }
    };
    const auto initialize_cache = [&](KVCache& cache) {
        for (Tensor& tensor : cache.k) {
            CUDA_CHECK(cudaMemsetAsync(tensor.data, 0, tensor.bytes(), device.stream));
        }
        for (Tensor& tensor : cache.v) {
            CUDA_CHECK(cudaMemsetAsync(tensor.data, 0, tensor.bytes(), device.stream));
        }
        for (Tensor& tensor : cache.k_scale) {
            CUDA_CHECK(cudaMemsetAsync(tensor.data, 0, tensor.bytes(), device.stream));
        }
        for (Tensor& tensor : cache.v_scale) {
            CUDA_CHECK(cudaMemsetAsync(tensor.data, 0, tensor.bytes(), device.stream));
        }
    };
    initialize_cache(decoder->text_kv);
    device.synchronize();

    const auto prepare_representative = [&](std::uint32_t frontier) {
        work.reset();
        clear_stable_controls();
        decoder->gdn.reset_running(device.stream);
        set_device_i32(io.pos, checked_i32(frontier, "graph representative position"));
        set_device_i32(io.rope_pos, checked_i32(frontier, "graph representative rope position"));
    };
    const auto state = [&](std::uint32_t frontier) {
        return schedule::State{device,
                               model,
                               work,
                               decoder->text_kv,
                               nullptr,
                               nullptr,
                               decoder->gdn,
                               io,
                               prefill_hidden,
                               prefill_chunk,
                               frontier,
                               static_cast<const ops::SamplingConfig*>(sampling_config.data),
                               proposal_head,
                               &tail_hidden,
                               &boundary_hidden,
                               nullptr,
                               nullptr,
                               nullptr};
    };

    const auto ordinary_ranges = ordinary_graph_ranges(capacity);
    validate_graph_ranges(ordinary_ranges, capacity - 1, "ordinary");
    ordinary_graphs.reserve(ordinary_ranges.size());
    for (const GraphFrontierRange range : ordinary_ranges) {
        ordinary_graphs.emplace_back();
        OrdinaryGraphVariant& variant      = ordinary_graphs.back();
        variant.min_execution_frontier     = range.min;
        variant.max_execution_frontier     = range.max;
        const std::uint32_t representative = range.min;
        const ops::GqaExecutionEnvelope envelope{range.min + 1, range.max + 1};
        const auto prepare = [&, representative] { prepare_representative(representative); };

        auto ordinary_state = state(representative);
        schedule::warm_capture_ordinary_round(ordinary_state, false, envelope, prepare,
                                               variant.ordinary);
    }

    ordered_reset();
    clear_stable_controls();
    CUDA_CHECK(cudaMemsetAsync(token_counts.data, 0, token_counts.bytes(), device.stream));
    device.synchronize();

    std::size_t free_after = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_after, &total_bytes));
    const std::size_t consumed = free_before > free_after ? free_before - free_after : 0;
    if (consumed > graph_allowance_bytes) {
        throw std::runtime_error("CUDA Graph warm/capture consumed " + std::to_string(consumed) +
                                 " bytes, exceeding the planned allowance of " +
                                 std::to_string(graph_allowance_bytes) + " bytes");
    }
}

void ProgramImplCore::install_sampling(const ops::SamplingConfig& config) {
    CUDA_CHECK(cudaMemsetAsync(token_counts.data, 0, token_counts.bytes(), device.stream));
    CUDA_CHECK(
        cudaMemsetAsync(io.speculative.stats.data, 0, io.speculative.stats.bytes(), device.stream));
    sampling_host = config;
    sampling_host.token_counts =
        sampling_host.presence_penalty != 0.0F || sampling_host.frequency_penalty != 0.0F
            ? static_cast<std::int32_t*>(token_counts.data)
            : nullptr;
    CUDA_CHECK(cudaMemcpyAsync(sampling_config.data, &sampling_host, sizeof(sampling_host),
                                cudaMemcpyHostToDevice, device.stream));
}

void ProgramImplCore::copy_tail(const Tensor& source) {
    if (source.dtype != DType::BF16 || source.ne[0] != TextConfig::hidden || source.ne[1] != 1) {
        throw std::logic_error("target tail hidden has an invalid shape");
    }
    CUDA_CHECK(cudaMemcpyAsync(tail_hidden.data, source.data, tail_hidden.bytes(),
                                cudaMemcpyDeviceToDevice, device.stream));
    tail_hidden_valid = true;
}

void ProgramImplCore::copy_round_token() {
    CUDA_CHECK(cudaMemcpyAsync(host_tokens, io.token.data, sizeof(TokenId), cudaMemcpyDeviceToHost,
                                device.stream));
}

void ProgramImplCore::mark_workspace_usage(std::size_t phase_bytes) noexcept {
    workspace_logical_peak_bytes = std::max(workspace_logical_peak_bytes, phase_bytes);
}

void ProgramImplCore::validate_licensed_tokens(std::span<const TokenId> tokens) const {
    for (const TokenId token : tokens) {
        if (token < 0 || token >= TextConfig::token_domain) {
            throw std::runtime_error("target returned a token outside the vocabulary domain");
        }
    }
}

runtime::BeginResult ProgramImplCore::begin(PreparedPromptData&& prompt, RequestPlan&& request_plan,
                                             runtime::TransientRegion transient) {
    if (request_plan.impl_ == nullptr) { throw std::invalid_argument("request plan is empty"); }
    const RequestPlanImpl& plan = *request_plan.impl_;
    if (lifecycle == Lifecycle::Active || lifecycle == Lifecycle::Pending) {
        throw std::logic_error("begin requires Empty, Resident, or Invalid Program state");
    }
    const std::uint32_t prompt_tokens = static_cast<std::uint32_t>(prompt.token_ids.size());
    if (prompt_tokens != plan.summary.prompt_tokens) {
        throw std::invalid_argument("request plan does not describe the prepared prompt");
    }
    if (plan.summary.transient_bytes != 0 &&
        (transient.data == nullptr || transient.size < plan.summary.transient_bytes ||
         transient.alignment < plan.summary.transient_alignment)) {
        throw std::invalid_argument("request transient region does not satisfy the plan");
    }
    if (plan.reuse != ReusePath::FullReset) {
        if (lifecycle != Lifecycle::Resident ||
            !qwen3_6::detail::prefix_matches(prompt, ledger, prefix_identity, plan.reuse_base)) {
            throw std::logic_error("planned resident prefix is no longer reusable");
        }
    }

    const std::uint32_t base              = plan.reuse_base;
    const bool had_suffix                 = prompt_tokens > base;
    const std::int32_t request_rope_delta = prompt.rope_delta;
    const auto snapshot_boundary          = plan.snapshot_boundary;
    const auto begin_start                = Clock::now();
    const auto text_start                 = Clock::now();

    lifecycle = Lifecycle::Invalid;
    try {
        if (plan.reuse == ReusePath::FullReset) {
            ordered_reset();
            ledger.clear();
        } else if (plan.reuse == ReusePath::AppendAtFrontier) {
            if (text_kv_valid < base) {
                throw std::logic_error("resident Text KV is shorter than E");
            }
            text_kv_valid = base;
            ledger.resize(base);
        } else {
            if (text_kv_valid < base) {
                throw std::logic_error("resident Text KV is shorter than boundary");
            }
            text_kv_valid = base;
            ledger.resize(base);
        }

        install_sampling(plan.sampling);
        rope_delta = request_rope_delta;
        set_device_i32(io.rope_delta, rope_delta);
        boundary                 = {};
        timings                  = {};
        ++request_epoch;
        drafts_ready          = false;
        ordinary_tail         = false;

        ledger.assign(prompt.token_ids.begin(), prompt.token_ids.end());
        prefix_identity.assign(prompt);

        schedule::State schedule_state{
            device,
            model,
            work,
            decoder->text_kv,
            nullptr,
            nullptr,
            decoder->gdn,
            io,
            prefill_hidden,
            prefill_chunk,
            base,
            static_cast<const ops::SamplingConfig*>(sampling_config.data),
            proposal_head,
            &tail_hidden,
            &boundary_hidden,
            diagnostic_context,
            diagnostic_text_tap,
            diagnostic_vision_tap};

        if (had_suffix) {
            mark_workspace_usage(workspace_plan.text_prefill);
            bool mtp_prepared = schedule::prefill_text(
                schedule_state, std::span<const TokenId>(prompt.token_ids).subspan(base),
                snapshot_boundary, false);
            const std::uint32_t final_length = final_prefill_chunk_length(
                base, prompt_tokens, prefill_chunk, snapshot_boundary);
            copy_tail(prefill_hidden.slice(1, static_cast<int>(final_length) - 1, 1));
            mtp_prepared = false;
        } else {
            mark_workspace_usage(workspace_plan.ordinary_round);
            if (!tail_hidden_valid) {
                throw std::logic_error("zero-suffix reuse has no target tail hidden");
            }
            schedule::sample_from_hidden(schedule_state, tail_hidden,
                                          checked_i32(prompt_tokens, "sample position"),
                                          ops::kSamplePurposePrefill);
            set_device_i32(io.rope_pos,
                            checked_i32(prompt_tokens, "rope position") + rope_delta);
        }
        copy_round_token();
        device.synchronize();
        timings.prefill_seconds =
            std::chrono::duration<double>(Clock::now() - text_start).count();

        validate_licensed_tokens(std::span<const TokenId>(host_tokens, 1));
        if (ledger.size() != prompt_tokens) {
            throw std::logic_error("candidate token ledger does not match prompt length");
        }
        ledger.push_back(host_tokens[0]);
        prefix_identity.append_generated(1, rope_delta);
        text_kv_valid = prompt_tokens;
        tail_hidden_valid = true;

        pending   = PendingCandidate{.kind          = PendingKind::Begin,
                                      .base_E        = 0,
                                      .base_S        = 0,
                                      .prompt_tokens = prompt_tokens,
                                      .produced      = 1};
        lifecycle = Lifecycle::Pending;
        timings.prefill_seconds =
            std::max(timings.prefill_seconds,
                      std::chrono::duration<double>(Clock::now() - begin_start).count());
        return runtime::BeginResult{
            .summary =
                runtime::BeginSummary{.prompt_tokens = prompt_tokens, .reused_prompt_tokens = base},
            .round = runtime::GeneratedRound{.tokens = std::span<const TokenId>(host_tokens, 1)},
        };
    } catch (...) {
        try { device.synchronize(); } catch (...) {}
        make_invalid();
        throw;
    }
}

runtime::GeneratedRound ProgramImplCore::decode_round(runtime::RoundBudget budget) {
    if (lifecycle != Lifecycle::Active) {
        throw std::logic_error("decode_round requires Active Program state");
    }
    if (budget.generated_tokens_remaining == 0) {
        throw std::invalid_argument("decode round budget must be nonzero");
    }
    if (E >= capacity) { throw std::out_of_range("Text execution context is full"); }
    if (S != E + 1 || ledger.size() != S || prefix_identity.size() != S) {
        throw std::logic_error("Active frontier is inconsistent");
    }

    const std::uint32_t base_E = E;
    const std::uint32_t base_S = S;
    nvtx::ScopedRange round_range(nvtx::Name::DecodeOrdinaryRound, nvtx::Category::Decode,
                                   base_E);
    try {
        set_device_i32(io.gdn_initial_slot, 0);
        schedule::State schedule_state{
            device,
            model,
            work,
            decoder->text_kv,
            nullptr,
            nullptr,
            decoder->gdn,
            io,
            prefill_hidden,
            prefill_chunk,
            base_E,
            static_cast<const ops::SamplingConfig*>(sampling_config.data),
            proposal_head,
            &tail_hidden,
            &boundary_hidden,
            diagnostic_context,
            diagnostic_text_tap,
            diagnostic_vision_tap};

        mark_workspace_usage(workspace_plan.ordinary_round);
        DecodeGraph* graph = nullptr;
        ops::GqaExecutionEnvelope envelope{base_E + 1, base_E + 1};
        if (use_cuda_graph && diagnostic_text_tap == nullptr) {
            OrdinaryGraphVariant& variant =
                select_graph_variant(ordinary_graphs, base_E, "ordinary");
            graph    = &variant.ordinary;
            envelope = {variant.min_execution_frontier + 1, variant.max_execution_frontier + 1};
        }
        {
            nvtx::ScopedRange submit_range(nvtx::Name::DecodeOrdinarySubmit,
                                            nvtx::Category::Decode, base_E);
            schedule::ordinary_round(schedule_state, false, envelope, graph);
            copy_tail(io.verify_hidden.slice(1, 0, 1));
            copy_round_token();
        }
        {
            nvtx::ScopedRange wait_range(nvtx::Name::DecodeOrdinaryWait,
                                          nvtx::Category::Control, base_E);
            device.synchronize();
        }
        text_kv_valid    = base_E + 1;
        drafts_ready     = false;
        tail_hidden_valid = true;

        validate_licensed_tokens(std::span<const TokenId>(host_tokens, 1));
        ledger.insert(ledger.end(), host_tokens, host_tokens + 1);
        prefix_identity.append_generated(1, rope_delta);
        pending   = PendingCandidate{.kind          = PendingKind::Ordinary,
                                      .base_E        = base_E,
                                      .base_S        = base_S,
                                      .prompt_tokens = 0,
                                      .produced      = 1};
        lifecycle = Lifecycle::Pending;
        return runtime::GeneratedRound{.tokens = std::span<const TokenId>(host_tokens, 1)};
    } catch (...) {
        try { device.synchronize(); } catch (...) {}
        make_invalid();
        throw;
    }
}

void ProgramImplCore::resolve_pending(std::uint32_t accepted_tokens, bool terminal) {
    if (lifecycle != Lifecycle::Pending) {
        throw std::logic_error("resolve_pending requires a pending generated round");
    }
    if (accepted_tokens == 0 || accepted_tokens > pending.produced) {
        throw std::out_of_range("accepted prefix is outside the pending generated round");
    }
    if (!terminal && accepted_tokens != pending.produced) {
        throw std::logic_error("a continuing round must accept every licensed token");
    }
    if (accepted_tokens != pending.produced) {
        throw std::logic_error("a non-speculative terminal round must accept its only token");
    }

    switch (pending.kind) {
    case PendingKind::Begin:
        E = pending.prompt_tokens;
        S = pending.prompt_tokens + 1;
        break;
    case PendingKind::Ordinary:
    case PendingKind::Speculative:
        E = pending.base_E + pending.produced;
        S = pending.base_S + pending.produced;
        break;
    case PendingKind::None:
        break;
    }
    drafts_ready = false;
    lifecycle = Lifecycle::Resident;
    pending = {};
}

void ProgramImplCore::finish_active() {
    if (lifecycle != Lifecycle::Active) { return; }
    resolve_pending(1, true);
}

void ProgramImplCore::abort_request() noexcept {
    make_invalid();
}

std::uint32_t ProgramImplCore::materialized_tokens() const noexcept {
    return S;
}

MemorySummary ProgramImplCore::memory_summary() const noexcept {
    MemorySummary out;
    out.device      = device.device;
    out.max_context = capacity;
    out.kv_cache = kv_dtype == DType::BF16 ? KvCacheStorage::BFloat16 : KvCacheStorage::Int8Group64;
    out.workspace_logical_peak_bytes = workspace_logical_peak_bytes;
    out.cuda_graph_allowance_bytes   = graph_allowance_bytes;
    out.kv_payload_bytes             = kv_payload_bytes;
    return out;
}

SpeculativeStats ProgramImplCore::speculative_stats() const {
    return SpeculativeStats{};
}

void ProgramImplCore::reset_memory_peaks() noexcept {
    workspace_logical_peak_bytes = 0;
}

} // namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS