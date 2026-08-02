#pragma once

// Laguna-specific text_context_impl.h stubs
// Laguna does not use GDN, MTP, or Vision features. The shared code paths for these
// are never reached because gdn_layers() == 0 and mtp_enabled() == false.

#include "targets/qwen3_6/impl/runtime/instance.h"
#include "targets/qwen3_6/impl/runtime/text_context.h"
#include "targets/qwen3_6/impl/runtime/workspace_recipe.h"

#include "core/nvtx.h"
#include <ninfer/targets/qwen3_6/vision_control.h>
#include "ninfer/ops/argmax.h"
#include "ninfer/ops/embedding.h"
#include "ninfer/ops/gqa_attention.h"
#include "ninfer/ops/linear.h"
#include "ninfer/ops/linear_add.h"
#include "ninfer/ops/linear_swiglu.h"
#include "ninfer/ops/position.h"
#include "ninfer/ops/residual_add.h"
#include "ninfer/ops/rmsnorm.h"
#include "ninfer/ops/rope.h"
#include "ninfer/ops/rope_yarn.h"
#include "ninfer/ops/sampling.h"
#include "ninfer/ops/sigmoid_mul.h"
#include "ninfer/ops/silu_mul.h"
#include "ninfer/ops/softplus_mul.h"

#include <cuda_runtime.h>
#include <algorithm>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS::schedule {
namespace {

void copy_i32(const std::int32_t* source, Tensor& destination, cudaStream_t stream) {
    if (source == nullptr || destination.dtype != DType::I32 || !destination.is_contiguous() ||
        destination.data == nullptr) {
        throw std::invalid_argument("copy_i32: invalid host source or I32 destination");
    }
    CUDA_CHECK(cudaMemcpyAsync(destination.data, source, destination.bytes(),
                                cudaMemcpyHostToDevice, stream));
}

void require_tensor_shape(const Tensor& t, DType dtype, std::initializer_list<std::int32_t> shape,
                           const char* label) {
    if (t.dtype != dtype) { throw std::invalid_argument(std::string(label) + " dtype mismatch"); }
    int i = 0;
    for (const std::int32_t dim : shape) {
        if (t.ne[i] != dim) { throw std::invalid_argument(std::string(label) + " shape mismatch"); }
        ++i;
    }
    for (; i < 4; ++i) {
        if (t.ne[i] != 1) { throw std::invalid_argument(std::string(label) + " shape mismatch"); }
    }
    if (!t.is_contiguous()) {
        throw std::invalid_argument(std::string(label) + " must be contiguous");
    }
    if (t.data == nullptr) { throw std::invalid_argument(std::string(label) + " data is null"); }
}

void require_tensor_window(const Tensor& t, DType dtype, std::int32_t rows, std::int32_t cols,
                             const char* label) {
    if (cols <= 0) { throw std::invalid_argument(std::string(label) + " cols must be positive"); }
    if (t.dtype != dtype) { throw std::invalid_argument(std::string(label) + " dtype mismatch"); }
    if (t.ne[0] != rows || t.ne[1] < cols || t.ne[2] != 1 || t.ne[3] != 1) {
        throw std::invalid_argument(std::string(label) + " shape mismatch");
    }
    if (!t.is_contiguous()) {
        throw std::invalid_argument(std::string(label) + " must be contiguous");
    }
    if (t.data == nullptr) { throw std::invalid_argument(std::string(label) + " data is null"); }
}

void require_vector_window(const Tensor& t, DType dtype, std::int32_t cols, const char* label) {
    if (cols <= 0) { throw std::invalid_argument(std::string(label) + " cols must be positive"); }
    if (t.dtype != dtype) { throw std::invalid_argument(std::string(label) + " dtype mismatch"); }
    if (t.ne[0] < cols || t.ne[1] != 1 || t.ne[2] != 1 || t.ne[3] != 1) {
        throw std::invalid_argument(std::string(label) + " shape mismatch");
    }
    if (!t.is_contiguous()) {
        throw std::invalid_argument(std::string(label) + " must be contiguous");
    }
    if (t.data == nullptr) { throw std::invalid_argument(std::string(label) + " data is null"); }
}

Tensor matrix_window(Tensor& t, std::int32_t cols) {
    if (cols <= 0) { throw std::invalid_argument("matrix_window cols must be positive"); }
    if (t.ne[1] < cols || t.ne[2] != 1 || t.ne[3] != 1) {
        throw std::invalid_argument("matrix_window shape mismatch");
    }
    return t.slice(1, 0, cols);
}

Tensor vector_window(Tensor& t, std::int32_t cols) {
    if (cols <= 0) { throw std::invalid_argument("vector_window cols must be positive"); }
    if (t.ne[0] < cols || t.ne[1] != 1 || t.ne[2] != 1 || t.ne[3] != 1) {
        throw std::invalid_argument("vector_window shape mismatch");
    }
    return t.slice(0, 0, cols);
}

class ScopedPositions {
public:
    ScopedPositions(const Tensor*& slot, const Tensor& positions) : slot_(slot) {
        slot_ = &positions;
    }
    ScopedPositions(const ScopedPositions&)            = delete;
    ScopedPositions& operator=(const ScopedPositions&) = delete;
    ~ScopedPositions() { slot_ = nullptr; }
private:
    const Tensor*& slot_;
};

class ScopedEnvelope {
public:
    ScopedEnvelope(const ops::GqaExecutionEnvelope*& slot,
                   const ops::GqaExecutionEnvelope& envelope)
        : slot_(slot) {
        slot_ = &envelope;
    }
    ScopedEnvelope(const ScopedEnvelope&)            = delete;
    ScopedEnvelope& operator=(const ScopedEnvelope&) = delete;
    ~ScopedEnvelope() { slot_ = nullptr; }
private:
    const ops::GqaExecutionEnvelope*& slot_;
};

struct CallbackTap {
    static constexpr bool enabled = true;
    void* context            = nullptr;
    TextTapCallback callback = nullptr;
    void operator()(TapId id, int layer, Phase phase, const Tensor& value,
                    cudaStream_t stream) const {
        callback(context, id, layer, phase, value, stream);
    }
};

} // namespace

TextContext::TextContext(DeviceContext& ctx, const LoadedModelData& weights, WorkspaceArena& work,
                         KVCache& kv, qwen3_6::GdnStateStore& state, qwen3_6::RoundState& io,
                         Tensor& prefill_hidden, std::uint32_t prefill_chunk,
                         std::uint32_t text_kv_base, KVCache* /*mtp_kv*/)
    : ctx_(ctx), weights_(weights), work_(work), kv_(kv), mtp_kv_(nullptr), state_(state), io_(io),
      prefill_hidden_(prefill_hidden), prefill_chunk_(prefill_chunk), text_kv_base_(text_kv_base) {
    if (prefill_chunk_ == 0 ||
        prefill_chunk_ > static_cast<std::uint32_t>(std::numeric_limits<std::int32_t>::max())) {
        throw std::invalid_argument("TextContext effective prefill chunk must fit positive int32");
    }
    bind();
}

TextContext::~TextContext() = default;

void TextContext::bind() {
    using TargetMlp = typename Variant::PostMixerWeights;
    const auto bind_mlp = [](const TargetMlp& source) { return MlpW{&source}; };

    embed_      = &weights_.token_embedding;
    final_norm_ = &weights_.final_norm;
    lm_head_    = &weights_.output_head;

    for (int layer = 0; layer < kCfg.n_layers; ++layer) {
        if (TextConfig::is_full_attention(layer)) {
            const int fidx = layer / 4;
            FullLayerW& out = full_.at(static_cast<std::size_t>(fidx));
            const auto& source = weights_.full_layers[fidx];
            out.input_norm     = &source.input_norm;
            out.projection     = &source.projection;
            out.o_proj         = &source.output;
            out.q_norm         = &source.query_norm;
            out.k_norm         = &source.key_norm;
            out.post_attn_norm = &source.post_attention_norm;
            out.mlp            = bind_mlp(source.post_mixer);
        } else {
            const int sidx = layer - layer / 4 - 1;
            FullLayerW& out = swa_.at(static_cast<std::size_t>(sidx));
            const auto& source = weights_.swa_layers[sidx];
            out.input_norm     = &source.input_norm;
            out.projection     = &source.projection;
            out.o_proj         = &source.output;
            out.q_norm         = &source.query_norm;
            out.k_norm         = &source.key_norm;
            out.post_attn_norm = &source.post_attention_norm;
            out.mlp            = bind_mlp(source.post_mixer);
        }
    }
}

void TextContext::attn_mix(const FullLayerW& w, Tensor& x, int layer, Phase ph) {
    cudaStream_t s = ctx_.stream;
    const int T    = static_cast<int>(x.ne[1]);
    if (active_gqa_envelope_ == nullptr) {
        throw std::logic_error("Text GQA execution envelope is not set");
    }

    const int q_heads = TextConfig::q_heads(layer);
    const int q_rows  = q_heads * TextConfig::head_dim;
    const int kv_rows = TextConfig::kv_rows;

    Tensor h = work_.alloc(DType::BF16, {TextConfig::hidden, T});
    ops::rmsnorm(x, *w.input_norm, kCfg.rms_eps, true, h, s);

    Tensor q_flat    = work_.alloc(DType::BF16, {q_rows, T});
    Tensor gate      = work_.alloc(DType::BF16, {q_heads, T});
    Tensor k_flat    = work_.alloc(DType::BF16, {kv_rows, T});
    Tensor v_flat    = work_.alloc(DType::BF16, {kv_rows, T});

    Variant::attention_projection(h, *w.projection, q_flat, gate, k_flat, v_flat, ph, work_, s);

    Tensor qn = work_.alloc(DType::BF16, {TextConfig::head_dim, q_heads, T});
    Tensor kn = work_.alloc(DType::BF16, {TextConfig::head_dim, TextConfig::kv_heads, T});
    ops::rmsnorm(q_flat.view({TextConfig::head_dim, q_heads, T}), *w.q_norm, kCfg.rms_eps, true, qn, s);
    ops::rmsnorm(k_flat.view({TextConfig::head_dim, TextConfig::kv_heads, T}), *w.k_norm, kCfg.rms_eps, true, kn, s);

    const Tensor& cache_positions =
        active_cache_positions_ != nullptr ? *active_cache_positions_ : io_.pos;
    const Tensor& rope_positions =
        active_rope_positions_ != nullptr ? *active_rope_positions_ : io_.rope_pos;

    if (TextConfig::is_full_attention(layer)) {
        ops::rope_yarn(rope_positions, 64,
                       ops::YarnParameters{
                           500000.0F, 32.0F, 8192.0F, 1.0F, 64.0F, 1.3465735902799727F},
                       qn, kn, s);
    } else {
        ops::rope(rope_positions, TextConfig::head_dim, 10000.0F, qn, kn, s);
    }

    Tensor a = work_.alloc(DType::BF16, {TextConfig::head_dim, q_heads, T});
    int window = TextConfig::is_swa(layer) ? TextConfig::sliding_window : 0;
    ops::gqa_attention(qn, kn, v_flat.view({TextConfig::head_dim, TextConfig::kv_heads, T}),
                       cache_positions, kAttnScale, window,
                       kv_.layer_view(static_cast<std::uint32_t>(layer)),
                       *active_gqa_envelope_, work_, a, TextConfig::head_dim, s);

    {
        ops::softplus_mul(gate, a, s);
    }

    Variant::attention_output_projection(a.reshape({q_rows, T}), *w.o_proj, x, ph, work_, s);
}

void TextContext::mlp_tail(const Tensor* post_norm, const MlpW& m, Tensor& x, Phase ph) {
    cudaStream_t s = ctx_.stream;
    const int T    = static_cast<int>(x.ne[1]);
    Tensor h       = workspace_recipe::post_mixer_hidden<TextConfig>(work_, T);
    ops::rmsnorm(x, *post_norm, kCfg.rms_eps, true, h, s);
    Variant::post_mixer(h, *m.payload, x, ph, work_, s);
}

template <class Tap>
void TextContext::run_layers(Tensor& x, Phase ph, Tap& tap) {
    const bool prefill = ph == Phase::Prefill;
    for (int layer = 0; layer < kCfg.n_layers; ++layer) {
        if (TextConfig::is_full_attention(layer)) {
            const int fidx = layer / 4;
            const FullLayerW& full = full_.at(static_cast<std::size_t>(fidx));
            nvtx::ScopedRange layer_range(
                prefill ? nvtx::Name::PrefillLayerFull : nvtx::Name::VerifyLayerFull,
                nvtx::Category::Attention, static_cast<std::uint64_t>(layer));
            {
                nvtx::ScopedRange mixer_range(
                    prefill ? nvtx::Name::PrefillAttention : nvtx::Name::VerifyAttention,
                    nvtx::Category::Attention, static_cast<std::uint64_t>(layer));
                auto mixer_scope = work_.scope();
                attn_mix(full, x, layer, ph);
                if constexpr (Tap::enabled) { tap(TapId::AfterMixer, layer, ph, x, ctx_.stream); }
            }
            {
                nvtx::ScopedRange post_mixer_range(
                    prefill ? nvtx::Name::PrefillPostMixer : nvtx::Name::VerifyPostMixer,
                    nvtx::Category::PostMixer, static_cast<std::uint64_t>(layer));
                auto mlp_scope = work_.scope();
                mlp_tail(full.post_attn_norm, full.mlp, x, ph);
                if constexpr (Tap::enabled) { tap(TapId::AfterMlp, layer, ph, x, ctx_.stream); }
            }
        } else if (TextConfig::is_swa(layer)) {
            const int sidx = layer - layer / 4 - 1;
            const FullLayerW& swa = swa_.at(static_cast<std::size_t>(sidx));
            nvtx::ScopedRange layer_range(
                prefill ? nvtx::Name::PrefillLayerFull : nvtx::Name::VerifyLayerFull,
                nvtx::Category::Attention, static_cast<std::uint64_t>(layer));
            {
                nvtx::ScopedRange mixer_range(
                    prefill ? nvtx::Name::PrefillAttention : nvtx::Name::VerifyAttention,
                    nvtx::Category::Attention, static_cast<std::uint64_t>(layer));
                auto mixer_scope = work_.scope();
                attn_mix(swa, x, layer, ph);
                if constexpr (Tap::enabled) { tap(TapId::AfterMixer, layer, ph, x, ctx_.stream); }
            }
            {
                nvtx::ScopedRange post_mixer_range(
                    prefill ? nvtx::Name::PrefillPostMixer : nvtx::Name::VerifyPostMixer,
                    nvtx::Category::PostMixer, static_cast<std::uint64_t>(layer));
                auto mlp_scope = work_.scope();
                mlp_tail(swa.post_attn_norm, swa.mlp, x, ph);
                if constexpr (Tap::enabled) { tap(TapId::AfterMlp, layer, ph, x, ctx_.stream); }
            }
        }
    }
}

void TextContext::run_layers(Tensor& x, Phase ph) {
    NullTap tap;
    run_layers(x, ph, tap);
}

template <class Tap>
void TextContext::prefill_impl(std::span<const int> ids, const MultimodalPrefill* /*multimodal*/,
                                 Tap& tap) {
    if (ids.empty()) { throw std::invalid_argument("TextContext::prefill requires tokens"); }
    if (ids.size() > static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max())) {
        throw std::overflow_error("TextContext::prefill token count exceeds int32");
    }
    cudaStream_t s           = ctx_.stream;
    const int T              = static_cast<int>(ids.size());
    const int chunk          = static_cast<int>(prefill_chunk_);
    const std::uint32_t base = text_kv_base_;

    if (text_kv_base_ == 0) {
        rope_delta_ = 0;
    }
    ops::set_i32_scalar(io_.rope_delta, rope_delta_, s);

    if (static_cast<std::uint64_t>(base) + static_cast<std::uint64_t>(T) >
        static_cast<std::uint64_t>(std::numeric_limits<std::int32_t>::max())) {
        throw std::overflow_error("TextContext::prefill absolute position exceeds int32");
    }
    const int base_i = static_cast<int>(base);

    for (int t0 = 0; t0 < T;) {
        int len = std::min(chunk, T - t0);
        work_.reset();

        const std::uint32_t prompt_t0 = base + static_cast<std::uint32_t>(t0);
        const bool is_last = (t0 + len == T);
        if (is_last) { last_prefill_chunk_length_ = static_cast<std::uint32_t>(len); }
        nvtx::ScopedRange chunk_range(nvtx::Name::PrefillChunk, nvtx::Category::Prefill,
                                       static_cast<std::uint64_t>(len));

        {
            const std::int32_t rope_axes = (rope_delta_ != 0 ? 1 : 0);
            const auto roots             = workspace_recipe::text_prefill_roots<TextConfig>(
                work_, len, rope_axes, 0);
            Tensor ids_device = roots.ids;
            copy_i32(ids.data() + t0, ids_device, s);

            Tensor positions = roots.positions;
            ops::fill_i32_positions(positions, base_i + t0, s);

            Tensor rope_positions = positions;
            if (rope_delta_ != 0) {
                rope_positions = roots.rope_positions;
                ops::offset_i32_positions(positions, io_.rope_delta, rope_positions, s);
            }
            ScopedPositions scoped_cache(active_cache_positions_, positions);
            ScopedPositions scoped_rope(active_rope_positions_, rope_positions);
            const auto visible = static_cast<std::uint32_t>(base_i + t0 + len);
            const ops::GqaExecutionEnvelope chunk_envelope{visible, visible};
            ScopedEnvelope scoped_envelope(active_gqa_envelope_, chunk_envelope);

            Tensor x = roots.residual;
            ops::embedding(ids_device, *embed_, x, s);
            if constexpr (Tap::enabled) { tap(TapId::AfterEmbed, -1, Phase::Prefill, x, s); }
            run_layers(x, Phase::Prefill, tap);

            Tensor xf = prefill_hidden_.data != nullptr
                            ? matrix_window(prefill_hidden_, len)
                            : work_.alloc(DType::BF16, {kCfg.hidden, len});
            ops::rmsnorm(x, *final_norm_, kCfg.rms_eps, true, xf, s);
            if constexpr (Tap::enabled) { tap(TapId::AfterFinalNorm, -1, Phase::Prefill, xf, s); }

            if (is_last) {
                Tensor last_xf = xf.slice(1, len - 1, 1);
                Tensor logits  = matrix_window(io_.logits, 1);
                ops::linear(last_xf, *lm_head_, logits, s);
                if constexpr (Tap::enabled) {
                    tap(TapId::AfterLogits, -1, Phase::Prefill, logits, s);
                }
                ops::set_i32_scalar(io_.pos, base_i + T, s);
                ops::set_i32_scalar(io_.rope_pos, base_i + T + rope_delta_, s);
                if (sampling_config_ != nullptr) {
                    ops::sample(logits, io_.token, kCfg.token_domain, sampling_config_,
                                static_cast<const std::int32_t*>(io_.pos.data),
                                ops::kSamplePurposePrefill, work_, s);
                }
            }
        }
        t0 += len;
    }
}

void TextContext::prefill(std::span<const int> ids) {
    NullTap tap;
    prefill_impl(ids, nullptr, tap);
}

void TextContext::diagnostic_prefill(std::span<const int> ids, void* context,
                                       TextTapCallback callback) {
    if (callback == nullptr) {
        throw std::invalid_argument("diagnostic prefill callback is null");
    }
    CallbackTap tap{context, callback};
    prefill_impl(ids, nullptr, tap);
}

void TextContext::target_verify(const Tensor& ids, const Tensor& positions,
                                 ops::GqaExecutionEnvelope envelope) {
    NullTap tap;
    target_verify_impl(ids, positions, envelope, tap);
}

void TextContext::diagnostic_target_verify(const Tensor& ids, const Tensor& positions,
                                             ops::GqaExecutionEnvelope envelope, void* context,
                                             TextTapCallback callback) {
    if (callback == nullptr) {
        throw std::invalid_argument("diagnostic target verify callback is null");
    }
    CallbackTap tap{context, callback};
    target_verify_impl(ids, positions, envelope, tap);
}

template <class Tap>
void TextContext::target_verify_impl(const Tensor& ids, const Tensor& positions,
                                       ops::GqaExecutionEnvelope envelope, Tap& tap) {
    const int T = ids.ne[0];
    if (T <= 0) { throw std::invalid_argument("target_verify T must be positive"); }
    require_tensor_shape(ids, DType::I32, {T}, "target_verify ids");
    require_tensor_shape(positions, DType::I32, {T}, "target_verify positions");
    require_tensor_window(io_.verify_hidden, DType::BF16, kCfg.hidden, T, "target_verify hidden");
    require_tensor_window(io_.logits, DType::BF16, kCfg.vocab, T, "target_verify logits");
    require_vector_window(io_.speculative.target_argmax, DType::I32, T,
                          "target_verify target_tokens");

    cudaStream_t s = ctx_.stream;
    work_.reset();

    {
        Tensor rope_positions = work_.alloc(DType::I32, {T});
        ops::offset_i32_positions(positions, io_.rope_delta, rope_positions, ctx_.stream);
        Tensor x = work_.alloc(DType::BF16, {kCfg.hidden, T});
        ops::embedding(ids, *embed_, x, s);
        if constexpr (Tap::enabled) { tap(TapId::AfterEmbed, -1, Phase::Verify, x, s); }
        ScopedPositions scoped_cache(active_cache_positions_, positions);
        ScopedPositions scoped_rope(active_rope_positions_, rope_positions);
        ScopedEnvelope scoped_envelope(active_gqa_envelope_, envelope);
        run_layers(x, Phase::Verify, tap);

        Tensor hidden = matrix_window(io_.verify_hidden, T);
        Tensor logits = matrix_window(io_.logits, T);
        Tensor target = vector_window(io_.speculative.target_argmax, T);
        ops::rmsnorm(x, *final_norm_, kCfg.rms_eps, true, hidden, s);
        if constexpr (Tap::enabled) { tap(TapId::AfterFinalNorm, -1, Phase::Verify, hidden, s); }
        ops::linear(hidden, *lm_head_, logits, s);
        if constexpr (Tap::enabled) { tap(TapId::AfterLogits, -1, Phase::Verify, logits, s); }
        ops::argmax(logits, target, kCfg.token_domain, s);
    }
    work_.reset();
}

} // namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS::schedule
