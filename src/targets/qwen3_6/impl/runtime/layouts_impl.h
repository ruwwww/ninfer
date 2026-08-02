#include "targets/qwen3_6/impl/runtime/instance.h"
#include "targets/qwen3_6/impl/runtime/layouts.h"
#include "targets/qwen3_6/impl/runtime/vision_context.h"
#include "targets/qwen3_6/impl/runtime/workspace_recipe.h"

#include "core/device.h"
#include "ninfer/ops/gated_delta_net.h"
#include "ninfer/ops/gdn_gating_proj.h"
#include "ninfer/ops/gdn_input_proj.h"
#include "ninfer/ops/linear_add.h"
#include "ninfer/ops/linear_swiglu.h"
#include "ninfer/ops/sampling.h"
#include "ninfer/ops/speculative_round.h"
#include "ninfer/ops/gqa_attention.h"
#include "ninfer/ops/bidirectional_gqa_attention.h"
#include "ninfer/ops/swa.h"

#include <algorithm>
#include <initializer_list>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS {
namespace {

constexpr std::size_t kMiB        = 1024ULL * 1024ULL;
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

std::size_t align_up(std::size_t value, std::size_t alignment, const char* label) {
    const std::size_t padding = (alignment - value % alignment) % alignment;
    return checked_add(value, padding, label);
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
    out.decoder = qwen3_6::plan_decoder_state(
        builder, qwen3_6::DecoderStateSpec{
                     .full_attention_layers = TextConfig::full_attention_layers(),
                     .mtp_layers            = TextConfig::mtp_layers,
                     .capacity              = plan.capacity,
                     .kv_heads              = TextConfig::kv_heads,
                     .attention_head_dim    = TextConfig::head_dim,
                     .kv_dtype              = plan.kv_dtype,
                     .kv_quant_group        = plan.kv_quant_group,
                     .enable_mtp            = plan.features.mtp(),
                     .gdn =
                         {
                             .layers         = TextConfig::gdn_layers(),
                             .conv_dim       = TextConfig::convolution_dim,
                             .conv_width     = TextConfig::gdn_conv_state_width,
                             .value_heads    = TextConfig::gdn_value_heads,
                             .value_head_dim = TextConfig::gdn_value_head_dim,
                             .key_head_dim   = TextConfig::gdn_key_head_dim,
                             .snapshot_slots = static_cast<std::int32_t>(slots),
                             .conv_dtype     = DType::BF16,
                         },
                 });
    if constexpr (Variant::supports_dflash) {
        if (plan.features.dflash()) {
            DFlashPersistentLayout& dflash = out.dflash.emplace();
            dflash.local =
                plan_kv_cache(builder, DFlashConfig::local_layers, DFlashConfig::local_capacity,
                              DFlashConfig::kv_heads, DFlashConfig::head_dim, DType::BF16);
            dflash.boundary_local =
                plan_kv_cache(builder, DFlashConfig::local_layers, DFlashConfig::local_capacity,
                              DFlashConfig::kv_heads, DFlashConfig::head_dim, DType::BF16);
            dflash.full         = plan_kv_cache(builder, 1, plan.capacity, DFlashConfig::kv_heads,
                                                DFlashConfig::head_dim, DType::BF16);
            dflash.commit_count = add_tensor(builder, DType::I32, {1}, "DFlash exact commit count");
            dflash.target_features = add_tensor(
                builder, DType::BF16, {DFlashConfig::feature_rows, effective_prefill_chunk},
                "DFlash retained target features");
            dflash.feature_positions = add_tensor(builder, DType::I32, {effective_prefill_chunk},
                                                  "DFlash retained target positions");
        }
    }

    out.round = qwen3_6::begin_round_state_layout(
        builder, qwen3_6::RoundStateSpec{.hidden       = TextConfig::hidden,
                                         .output_rows  = TextConfig::output_rows,
                                         .draft_window = plan.draft_window,
                                         .enable_mtp   = plan.features.mtp()});
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
    out.kv_payload_bytes =
        out.decoder.kv_payload_bytes() + (out.dflash ? out.dflash->kv_payload_bytes() : 0);
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
    const auto drafts = static_cast<std::int32_t>(plan.draft_window);
    const auto verify = drafts + 1;
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
        (void)workspace_recipe::text_prefill_roots<TextConfig>(
            layout, tokens, plan.features.vision ? 3 : 0, plan.features.vision ? tokens : 0);
    };
    const auto attention_stage = [&](WorkspaceLayoutBuilder& layout, std::int32_t first,
                                     std::int32_t last, qwen3_6::TextPhase phase,
                                     ops::GqaExecutionEnvelope envelope) {
        auto stage = layout.scope();
        (void)workspace_recipe::text_attention_projection<TextConfig>(layout, last);
        scratch(layout, Variant::attention_projection_workspace_capacity_bytes(plan.weights_profile,
                                                                               phase, first, last));
        (void)workspace_recipe::text_attention_results<TextConfig>(layout, last);
scratch(layout, ops::gqa_attention_workspace_capacity_bytes(
                             TextConfig::query_heads, 256, plan.kv_dtype, envelope, first, last));
        scratch(layout, Variant::attention_output_projection_workspace_capacity_bytes(
                            plan.weights_profile, phase, first, last));
    };
    const auto gdn_stage = [&](WorkspaceLayoutBuilder& layout, std::int32_t first,
                               std::int32_t last, qwen3_6::TextPhase phase, bool snapshot) {
        auto stage = layout.scope();
        (void)workspace_recipe::gdn_control<TextConfig>(layout, last);
        scratch(layout, Variant::gdn_norm_control_projection_workspace_capacity_bytes(first, last));
        (void)workspace_recipe::gdn_projection<TextConfig>(layout, last);
        if (snapshot) {
            scratch(layout, Variant::gdn_input_projection_snapshot_workspace_capacity_bytes(
                                plan.weights_profile, phase, first, last));
        } else {
            (void)workspace_recipe::gdn_prefill_conv<TextConfig>(layout, last);
            scratch(layout, Variant::gdn_input_projection_workspace_capacity_bytes(
                                plan.weights_profile, phase, first, last));
        }
        (void)workspace_recipe::gdn_recurrent_output<TextConfig>(layout, last);
        if (!snapshot) {
            scratch(layout,
                    ops::gated_delta_net_workspace_capacity_bytes(
                        TextConfig::gdn_key_heads, TextConfig::gdn_value_heads, true, first, last));
        }
        (void)workspace_recipe::gdn_normalized_output<TextConfig>(layout, last);
        scratch(layout, Variant::gdn_output_projection_workspace_capacity_bytes(
                            plan.weights_profile, phase, first, last));
    };
    const auto post_mixer_stage = [&](WorkspaceLayoutBuilder& layout, std::int32_t first,
                                      std::int32_t last, qwen3_6::TextPhase phase) {
        auto stage = layout.scope();
        (void)workspace_recipe::post_mixer_hidden<TextConfig>(layout, last);
        scratch(layout, Variant::post_mixer_workspace_capacity_bytes(plan.weights_profile, phase,
                                                                     first, last));
    };
    const auto target_body = [&](WorkspaceLayoutBuilder& layout, std::int32_t first,
                                 std::int32_t last, qwen3_6::TextPhase phase, bool snapshot,
                                 ops::GqaExecutionEnvelope envelope) {
        attention_stage(layout, first, last, phase, envelope);
        gdn_stage(layout, first, last, phase, snapshot);
        post_mixer_stage(layout, first, last, phase);
    };
    const auto target_verify = [&](std::int32_t tokens, ops::GqaExecutionEnvelope envelope) {
        WorkspaceLayoutBuilder layout;
        matrix(layout, DType::I32, 1, tokens);
        matrix(layout, DType::BF16, TextConfig::hidden, tokens);
        target_body(layout, tokens, tokens, qwen3_6::TextPhase::Verify, true, envelope);
        return finish(layout);
    };

    const auto proposal_scratch = [&](WorkspaceLayoutBuilder& layout, std::int32_t columns) {
        if (plan.proposal_head == ProposalHead::Optimized) {
            matrix(layout, DType::BF16, Variant::draft_head_rows, columns);
        }
    };
    const auto mtp_stem = [&](WorkspaceLayoutBuilder& layout, std::int32_t tokens,
                              bool preembedded) {
        (void)workspace_recipe::mtp_stem<TextConfig>(layout, tokens, !preembedded);
    };
    const auto mtp_full_core = [&](WorkspaceLayoutBuilder& layout, std::int32_t tokens,
                                   ops::GqaExecutionEnvelope envelope) {
        auto core = layout.scope();
        mtp_stem(layout, tokens, false);
        (void)workspace_recipe::mtp_attention_projection<TextConfig>(layout, tokens);
        scratch(layout, Variant::mtp_attention_projection_workspace_capacity_bytes(tokens, tokens));
        (void)workspace_recipe::mtp_attention_results<TextConfig>(layout, tokens);
scratch(layout, ops::gqa_attention_workspace_capacity_bytes(
                             TextConfig::query_heads, 256, plan.kv_dtype, envelope, tokens, tokens));
        (void)workspace_recipe::mtp_post_attention<TextConfig>(layout, tokens);
        scratch(layout, Variant::mtp_post_mixer_workspace_capacity_bytes(tokens, tokens));
    };
    const auto mtp_full_call = [&](WorkspaceLayoutBuilder& layout, std::int32_t tokens,
                                   ops::GqaExecutionEnvelope envelope, bool build_proposal) {
        auto call = layout.scope();
        matrix(layout, DType::I32, 1, tokens);
        mtp_full_core(layout, tokens, envelope);
        if (build_proposal) {
            auto proposal = layout.scope();
            proposal_scratch(layout, 1);
        }
    };
    const auto mtp_prefill_chunk = [&](WorkspaceLayoutBuilder& layout, std::int32_t first,
                                       std::int32_t last, bool preembedded) {
        auto call = layout.scope();
        matrix(layout, DType::BF16, TextConfig::hidden, 1);
        matrix(layout, DType::BF16, TextConfig::hidden, 1);
        {
            auto bulk = layout.scope();
            mtp_stem(layout, last, preembedded);
            matrix(layout, DType::BF16, TextConfig::kv_size, last);
            matrix(layout, DType::BF16, TextConfig::kv_size, last);
            scratch(layout, Variant::mtp_kv_projection_workspace_capacity_bytes(first, last));
            matrix(layout, DType::BF16, TextConfig::kv_size, last);
        }
        matrix(layout, DType::BF16, TextConfig::query_size, 1);
        matrix(layout, DType::BF16, TextConfig::query_size, 1);
        scratch(layout, Variant::mtp_q_gate_projection_workspace_capacity_bytes(1, 1));
        matrix(layout, DType::BF16, TextConfig::query_size, 1);
        matrix(layout, DType::I32, 3, 1);
        matrix(layout, DType::BF16, TextConfig::query_size, 1);
scratch(layout, ops::gqa_attention_workspace_capacity_bytes(
                             TextConfig::query_heads, 256, plan.kv_dtype, text_envelope, 1, 1));
        matrix(layout, DType::BF16, TextConfig::hidden, 1);
        matrix(layout, DType::BF16, TextConfig::hidden, 1);
        scratch(layout, Variant::mtp_post_mixer_workspace_capacity_bytes(1, 1));
        proposal_scratch(layout, 1);
    };

    WorkspacePlan out;
    WorkspaceLayoutBuilder text_prefill;
    text_common_root(text_prefill, chunk);
    target_body(text_prefill, 1, chunk, qwen3_6::TextPhase::Prefill, false, text_envelope);
    scratch(text_prefill, ops::sampling_workspace_capacity_bytes(TextConfig::token_domain, 1, 1));
    out.text_prefill = finish(text_prefill);

    const std::size_t verify_one = target_verify(1, text_envelope);
    const std::size_t sample_one =
        ops::sampling_workspace_capacity_bytes(TextConfig::token_domain, 1, 1);
    out.ordinary_round = std::max(verify_one, sample_one);

    if (plan.features.mtp()) {
        WorkspaceLayoutBuilder mtp_prefill;
        text_common_root(mtp_prefill, chunk);
        target_body(mtp_prefill, 1, chunk, qwen3_6::TextPhase::Prefill, false, text_envelope);
        matrix(mtp_prefill, DType::I32, 1, chunk);
        if (plan.features.vision) {
            matrix(mtp_prefill, DType::BF16, TextConfig::hidden, chunk);
            (void)workspace_recipe::visual_scatter_indices(mtp_prefill, chunk);
        }
        mtp_prefill_chunk(mtp_prefill, 1, chunk, plan.features.vision);
        for (std::int32_t i = 1; i < drafts; ++i) {
            matrix(mtp_prefill, DType::BF16, TextConfig::hidden, 1);
            mtp_full_call(mtp_prefill, 1, text_envelope, true);
        }
        out.mtp_prefill = finish(mtp_prefill);

        WorkspaceLayoutBuilder mtp_batch;
        mtp_full_call(mtp_batch, verify, text_envelope, false);
        WorkspaceLayoutBuilder mtp_ar;
        mtp_full_call(mtp_ar, 1, text_envelope, true);
        WorkspaceLayoutBuilder mtp_align;
        mtp_full_call(mtp_align, 1, text_envelope, false);
        WorkspaceLayoutBuilder mtp_proposal;
        proposal_scratch(mtp_proposal, 1);
        const std::size_t accept = ops::speculative_accept_greedy_drafts_workspace_capacity_bytes(
            TextConfig::token_domain, drafts, drafts);
        out.mtp_round = std::max({target_verify(verify, text_envelope), accept, finish(mtp_batch),
                                  finish(mtp_ar), finish(mtp_proposal)});
        out.ordinary_round = std::max(out.ordinary_round, finish(mtp_align));
    }

    if (plan.features.dflash()) {
        if constexpr (!Variant::supports_dflash) {
            throw std::logic_error("unsupported target reached DFlash scratch planning");
        } else {
            const auto dflash_context_capacity = [&](std::int32_t tokens) {
                WorkspaceLayoutBuilder layout;
                (void)workspace_recipe::dflash_context<DFlashConfig>(layout, tokens);
                {
                    auto layer = layout.scope();
                    (void)workspace_recipe::dflash_context_layer<DFlashConfig>(layout, tokens);
                }
                return finish(layout);
            };
            const auto dflash_proposal_capacity = [&](std::int32_t tokens) {
                WorkspaceLayoutBuilder layout;
                (void)workspace_recipe::dflash_proposal<DFlashConfig>(layout, tokens);
                {
                    auto attention = layout.scope();
                    (void)workspace_recipe::dflash_attention<DFlashConfig>(layout, tokens);
                    scratch(layout,
                            std::max(ops::swa_workspace_capacity_bytes({0, plan.capacity}, tokens,
                                                                       tokens),
                                     ops::bidirectional_gqa_attention_workspace_capacity_bytes(
                                         {0, plan.capacity}, tokens, tokens)));
                    scratch(layout, ops::linear_add_workspace_capacity_bytes(
                                        QType::W8G32_F16S, DFlashConfig::hidden,
                                        DFlashConfig::query_size, tokens, tokens));
                }
                {
                    auto mlp = layout.scope();
                    (void)workspace_recipe::dflash_mlp<DFlashConfig>(layout, tokens);
                    scratch(layout, ops::linear_swiglu_workspace_capacity_bytes(
                                        QType::W8G32_F16S, 2 * DFlashConfig::intermediate,
                                        DFlashConfig::hidden, tokens, tokens));
                    scratch(layout, ops::linear_add_workspace_capacity_bytes(
                                        QType::W8G32_F16S, DFlashConfig::hidden,
                                        DFlashConfig::intermediate, tokens, tokens));
                }
                matrix(layout, DType::BF16, DFlashConfig::hidden, drafts);
                if (plan.proposal_head == ProposalHead::Optimized) {
                    matrix(layout, DType::BF16, Variant::draft_head_rows, drafts);
                }
                return finish(layout);
            };

            out.dflash_context  = dflash_context_capacity(chunk);
            out.dflash_proposal = dflash_proposal_capacity(verify);
            const std::size_t accept =
                ops::speculative_accept_greedy_drafts_workspace_capacity_bytes(
                    TextConfig::token_domain, drafts, drafts);
            out.dflash_round   = std::max({target_verify(verify, text_envelope), accept,
                                           dflash_context_capacity(verify), out.dflash_proposal});
            out.ordinary_round = std::max(out.ordinary_round, dflash_context_capacity(1));
        }
    }

    if (plan.features.vision) {
        constexpr std::uint32_t kFrontendMergedLimit  = 32768;
        constexpr std::uint32_t kFrontendSegmentLimit = 768 / 2;
        const std::uint32_t merged = std::min(plan.capacity, kFrontendMergedLimit);
        out.vision_encode          = schedule::VisionContext::workspace_capacity_bytes(
            merged, std::min(merged, kFrontendSegmentLimit));
    }

    out.capacity =
        std::max({out.text_prefill, out.ordinary_round, out.mtp_prefill, out.mtp_round,
                  out.dflash_context, out.dflash_proposal, out.dflash_round, out.vision_encode});
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
    case SpeculativeBackend::Mtp:
        if (options.speculative.draft_tokens == 0 ||
            options.speculative.draft_tokens > kMaximumMtpDraftTokens) {
            throw std::invalid_argument("MTP draft window must be in [1,5]");
        }
        break;
    case SpeculativeBackend::DFlash:
        if (kMaximumDFlashDraftTokens == 0) {
            throw std::invalid_argument("DFlash is not supported by this target");
        }
        if (options.speculative.draft_tokens == 0 ||
            options.speculative.draft_tokens > kMaximumDFlashDraftTokens) {
            throw std::invalid_argument("DFlash draft window must be in [1,15]");
        }
        if (options.enable_vision) {
            throw std::invalid_argument("DFlash and Vision cannot be enabled together");
        }
        break;
    }
    if (device.sm() != 120) {
        throw std::invalid_argument("Qwen3.6 family runtime requires compute capability 12.0");
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
    impl->speculative_backend = options.speculative.backend;
    impl->proposal_head       = options.speculative.proposal_head;
    impl->features            = qwen3_6::startup_features(options);
    impl->use_cuda_graph      = options.use_cuda_graph;
    impl->device              = options.device;
    impl->kv_dtype       = options.kv_cache == KvCacheStorage::BFloat16 ? DType::BF16 : DType::I8;
    impl->kv_quant_group = impl->kv_dtype == DType::I8 ? kKvQuantGroup : 0;
    impl->persistent     = persistent_layout(*impl);
    impl->workspace      = build_workspace_plan(*impl);
    if (impl->features.vision) {
        constexpr std::size_t kFrontendMergedLimit = 32768;
        const std::size_t merged = std::min<std::size_t>(impl->capacity, kFrontendMergedLimit);
        impl->request_transient_capacity_bytes =
            align_up(checked_mul(checked_mul(static_cast<std::size_t>(VisionConfig::output_hidden),
                                             merged, "request transient elements"),
                                 dtype_size(DType::BF16), "request transient bytes"),
                     kArenaAlign, "request transient alignment");
    }
    if (impl->use_cuda_graph) {
        const std::size_t ordinary_variants = ordinary_graph_ranges(impl->capacity).size();
        const std::size_t ordinary_graphs =
            ordinary_variants *
            (impl->speculative_backend == SpeculativeBackend::Mtp ? 2ULL : 1ULL);
        // Cold full-model capture also materializes lazy CUDA module state. The 35B
        // K=3/C=4096 public benchmark measured 123,277,312 bytes across its 12 ordinary/aligned
        // and short-window executables. Keep one conservative family allowance of 12 MiB per
        // executable. Long MTP executables also trigger substantially larger driver allocations.
        impl->graph_allowance_bytes = ordinary_graphs * 12ULL * kMiB;
        if (impl->speculative_backend == SpeculativeBackend::Mtp) {
            for (const GraphFrontierRange range :
                 mtp_graph_ranges(impl->capacity, impl->draft_window)) {
                const std::uint64_t final_visible =
                    static_cast<std::uint64_t>(range.max) + 2ULL * impl->draft_window;
                impl->graph_allowance_bytes += (final_visible <= 4096 ? 12ULL : 82ULL) * kMiB;
            }
        } else if (impl->speculative_backend == SpeculativeBackend::DFlash) {
            for (const GraphFrontierRange range :
                 dflash_graph_ranges(impl->capacity, impl->draft_window)) {
                const std::uint64_t final_visible =
                    static_cast<std::uint64_t>(range.max) + impl->draft_window + 1ULL;
                const std::size_t per_graph = final_visible <= 4096 ? 64ULL : 96ULL;
                impl->graph_allowance_bytes = checked_add(
                    impl->graph_allowance_bytes, 2ULL * per_graph * kMiB, "DFlash graph allowance");
            }
        }
    }

    impl->device_reservation_bytes = checked_add(
        checked_add(
            checked_add(impl->persistent.bytes, impl->workspace.capacity, "sequence memory plan"),
            impl->request_transient_capacity_bytes, "request transient reservation"),
        impl->graph_allowance_bytes, "sequence graph allowance");
    return impl;
}

} // namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS
