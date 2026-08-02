#pragma once

// Laguna-specific layouts_impl.h
// Provides plan_sequence_impl for the Laguna target.

#include "targets/qwen3_6/impl/runtime/instance.h"
#include "targets/qwen3_6/impl/runtime/layouts.h"
#include "targets/qwen3_6/impl/runtime/workspace_recipe.h"

#include "core/device.h"
#include "ninfer/ops/sampling.h"
#include "ninfer/ops/gqa_attention.h"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <string>

namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS {
namespace {

constexpr std::size_t kMiB = 1024ULL * 1024ULL;
constexpr std::size_t kArenaAlign = 256ULL;

std::size_t checked_add(std::size_t a, std::size_t b, const char* label) {
    if (b > std::numeric_limits<std::size_t>::max() - a) { throw std::overflow_error(label); }
    return a + b;
}

std::size_t checked_mul(std::size_t a, std::size_t b, const char* label) {
    if (b != 0 && a > std::numeric_limits<std::size_t>::max() / b) {
        throw std::overflow_error(label);
    }
    return a * b;
}

TensorLayout add_tensor(LayoutBuilder& builder, DType dtype,
                        std::initializer_list<std::int32_t> shape, const char* label) {
    return builder.add_tensor(dtype, shape, kArenaAlign, label);
}

PersistentLayout persistent_layout(const SequencePlanImpl& plan) {
    const std::size_t columns = plan.draft_window + 1ULL;
    const std::size_t slots   = columns + 1ULL;
    const auto effective_prefill_chunk =
        static_cast<std::int32_t>(std::min(plan.prefill_chunk, plan.capacity));
    LayoutBuilder builder;
    PersistentLayout out;

    // Decoder state (Laguna has no GDN, MTP, or DFlash)
    out.decoder = qwen3_6::plan_decoder_state(
        builder, qwen3_6::DecoderStateSpec{
            .full_attention_layers = TextConfig::layers,
            .mtp_layers            = 0,
            .capacity              = plan.capacity,
            .kv_heads              = TextConfig::kv_heads,
            .attention_head_dim    = TextConfig::head_dim,
            .kv_dtype              = plan.kv_dtype,
            .kv_quant_group        = plan.kv_quant_group,
            .enable_mtp            = false,
            .gdn = {
                // Family contract requires >= 1 GDN layer; never executed here
                // because TextConfig::gdn_layers() == 0 drives the schedule.
                .layers         = 1,
                .conv_dim       = 1,
                .conv_width     = 1,
                .value_heads    = 1,
                .value_head_dim = 1,
                .key_head_dim   = 1,
                .snapshot_slots = static_cast<std::int32_t>(slots),
                .conv_dtype     = DType::BF16,
            },
        });

    out.round = qwen3_6::begin_round_state_layout(
        builder, qwen3_6::RoundStateSpec{
            .hidden       = TextConfig::hidden,
            .output_rows  = TextConfig::output_rows,
            .draft_window = plan.draft_window,
            .enable_mtp   = false});
    out.prefill_hidden = add_tensor(
        builder, DType::BF16, {TextConfig::hidden, effective_prefill_chunk}, "step prefill hidden");
    qwen3_6::complete_round_state_layout(builder, out.round);

    const auto i32 = [&](std::size_t n, const char* label) {
        return add_tensor(builder, DType::I32, {static_cast<std::int32_t>(n)}, label);
    };
    out.token_counts        = i32(TextConfig::token_domain, "sampling token counts");
    const auto config_words = static_cast<std::int32_t>(
        (sizeof(ops::SamplingConfig) + sizeof(std::int32_t) - 1) / sizeof(std::int32_t));
    out.sampling_config = add_tensor(builder, DType::I32, {config_words}, "sampling config");
    out.tail_hidden     = add_tensor(builder, DType::BF16, {TextConfig::hidden, 1}, "tail hidden");
    out.boundary_hidden =
        add_tensor(builder, DType::BF16, {TextConfig::hidden, 1}, "boundary hidden");
    out.bytes = builder.finish(kArenaAlign, "persistent layout");
    out.kv_payload_bytes = out.decoder.kv_payload_bytes();
    return out;
}

WorkspacePlan build_workspace_plan(const SequencePlanImpl& plan) {
    const std::uint32_t chunk_u32 = std::min(plan.prefill_chunk, plan.capacity);
    if (chunk_u32 == 0 ||
        chunk_u32 > static_cast<std::uint32_t>(std::numeric_limits<std::int32_t>::max()) ||
        plan.draft_window >= static_cast<std::uint32_t>(std::numeric_limits<std::int32_t>::max())) {
        throw std::invalid_argument("sequence workspace dimensions are invalid");
    }
    const auto chunk  = static_cast<std::int32_t>(chunk_u32);
    const ops::GqaExecutionEnvelope text_envelope{1, plan.capacity};

    const auto matrix  = [](WorkspaceLayoutBuilder& layout, DType dtype, std::int32_t rows,
                           std::int32_t tokens) { (void)layout.alloc(dtype, {rows, tokens}); };
    const auto scratch = [](WorkspaceLayoutBuilder& layout, std::size_t bytes) {
        if (bytes == 0) { return; }
        auto scope = layout.scope();
        (void)layout.alloc_bytes(bytes);
    };
    const auto finish = [](const WorkspaceLayoutBuilder& layout) { return layout.peak_bytes(1); };

    const auto text_common_root = [&](WorkspaceLayoutBuilder& layout, std::int32_t tokens) {
        (void)workspace_recipe::text_prefill_roots<TextConfig>(layout, tokens, 0, 0);
    };

    const auto attention_stage = [&](WorkspaceLayoutBuilder& layout, std::int32_t first,
                                      std::int32_t last, qwen3_6::TextPhase phase,
                                      ops::GqaExecutionEnvelope envelope) {
        auto stage = layout.scope();
        (void)workspace_recipe::text_attention_projection<TextConfig>(layout, last);
        scratch(layout, Variant::attention_projection_workspace_capacity_bytes(
                             plan.weights_profile, phase, first, last));
        (void)workspace_recipe::text_attention_results<TextConfig>(layout, last);
        scratch(layout, ops::gqa_attention_workspace_capacity_bytes(
                             TextConfig::query_heads, TextConfig::head_dim, plan.kv_dtype,
                             envelope, first, last));
        scratch(layout, Variant::attention_output_projection_workspace_capacity_bytes(
                             plan.weights_profile, phase, first, last));
    };

    const auto post_mixer_stage = [&](WorkspaceLayoutBuilder& layout, std::int32_t first,
                                       std::int32_t last, qwen3_6::TextPhase phase) {
        auto stage = layout.scope();
        (void)workspace_recipe::post_mixer_hidden<TextConfig>(layout, last);
        scratch(layout, Variant::post_mixer_workspace_capacity_bytes(
                            plan.weights_profile, phase, first, last));
    };

    const auto target_body = [&](WorkspaceLayoutBuilder& layout, std::int32_t first,
                                  std::int32_t last, qwen3_6::TextPhase phase, bool /*snapshot*/,
                                  ops::GqaExecutionEnvelope envelope) {
        attention_stage(layout, first, last, phase, envelope);
        post_mixer_stage(layout, first, last, phase);
    };

    WorkspacePlan out;
    WorkspaceLayoutBuilder text_prefill;
    text_common_root(text_prefill, chunk);
    target_body(text_prefill, 1, chunk, qwen3_6::TextPhase::Prefill, false, text_envelope);
    scratch(text_prefill, ops::sampling_workspace_capacity_bytes(TextConfig::token_domain, 1, 1));
    out.text_prefill = finish(text_prefill);

    const std::size_t verify_one = ops::sampling_workspace_capacity_bytes(
        TextConfig::token_domain, 1, 1);
    out.ordinary_round = verify_one;

    out.capacity = std::max(out.text_prefill, out.ordinary_round);
    return out;
}

} // namespace

void validate_target_options(DeviceContext& device, const EngineOptions& options) {
    if (options.max_context == 0 || options.max_context > Variant::maximum_context) {
        throw std::invalid_argument("max_context exceeds the variant native context capacity");
    }
    if (options.prefill_chunk == 0 || options.prefill_chunk % kPrefillChunkAlignment != 0) {
        throw std::invalid_argument("prefill_chunk must be a nonzero multiple of 128");
    }
    switch (options.speculative.backend) {
    case SpeculativeBackend::None:
        if (options.speculative.draft_tokens != 0 ||
            options.speculative.proposal_head != ProposalHead::Full) {
            throw std::invalid_argument(
                "disabled speculative decoding requires draft_tokens=0 and the full proposal head");
        }
        break;
    default:
        throw std::invalid_argument("Laguna does not support speculative decoding");
    }
    if (device.sm() != 120) {
        throw std::invalid_argument("Laguna runtime requires compute capability 12.0");
    }
}

std::unique_ptr<SequencePlanImpl> plan_sequence_impl(DeviceContext& device,
                                                      const EngineOptions& options,
                                                      WeightsProfile weights_profile) {
    validate_target_options(device, options);

    auto impl                 = std::make_unique<SequencePlanImpl>();
    impl->weights_profile     = weights_profile;
    impl->capacity            = options.max_context;
    impl->prefill_chunk       = std::min(options.prefill_chunk, options.max_context);
    impl->draft_window        = options.speculative.draft_tokens;
    impl->speculative_backend = SpeculativeBackend::None;
    impl->proposal_head       = ProposalHead::Full;
    impl->features            = qwen3_6::startup_features(options);
    impl->use_cuda_graph      = options.use_cuda_graph;
    impl->device              = options.device;
    impl->kv_dtype       = options.kv_cache == KvCacheStorage::BFloat16 ? DType::BF16 : DType::I8;
    impl->kv_quant_group = impl->kv_dtype == DType::I8 ? kKvQuantGroup : 0;
    impl->persistent     = persistent_layout(*impl);
    impl->workspace      = build_workspace_plan(*impl);
    impl->request_transient_capacity_bytes = 0;

    if (impl->use_cuda_graph) {
        const std::size_t ordinary_variants = ordinary_graph_ranges(impl->capacity).size();
        impl->graph_allowance_bytes = ordinary_variants * 12ULL * kMiB;
    }

    impl->device_reservation_bytes = checked_add(
        checked_add(
            checked_add(impl->persistent.bytes, impl->workspace.capacity, "sequence memory plan"),
            impl->request_transient_capacity_bytes, "request transient reservation"),
        impl->graph_allowance_bytes, "sequence graph allowance");
    return impl;
}

} // namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS