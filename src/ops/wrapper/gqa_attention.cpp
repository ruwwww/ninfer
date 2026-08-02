// ninfer::ops - GQA A1/A2/A3 validation and finite route dispatch.
#include "ninfer/ops/gqa_attention.h"

#include "core/layout.h"
#include "ops/launcher/gqa_attention.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace ninfer::ops {
namespace {

constexpr std::int32_t kHeadDim                      = 256;
constexpr std::int32_t kLagunaHeadDim                = 128;
constexpr float kExpectedScale                       = 0.0625f;
constexpr float kLagunaExpectedScale                 = 0.07905694f;  // 1/sqrt(128)
constexpr std::int32_t kSmallTChunkTokens            = 6;
constexpr std::int32_t kMaximumVerifyTokens          = 16;
constexpr std::uint32_t kTwoChunkPromptVisibleKeys   = 512;
constexpr std::uint32_t kThreeChunkPromptVisibleKeys = 1024;

std::int32_t kv_heads_for_q_heads(std::int32_t q_heads, std::int32_t head_dim, const char* op) {
    if (head_dim == 256) {
        if (q_heads == 24) { return 4; }
        if (q_heads == 16) { return 2; }
    } else if (head_dim == 128) {
        if (q_heads == 48) { return 8; }
        if (q_heads == 64) { return 8; }
    }
    throw std::invalid_argument(std::string(op) + ": unsupported Q/KV head geometry");
}

void require_kv_heads(std::int32_t kv_heads, std::int32_t head_dim, const char* op) {
    if (head_dim == 256) {
        if (kv_heads != 4 && kv_heads != 2) {
            throw std::invalid_argument(std::string(op) + ": unsupported KV head geometry");
        }
    } else if (head_dim == 128) {
        if (kv_heads != 8) {
            throw std::invalid_argument(std::string(op) + ": unsupported KV head geometry");
        }
    }
}

std::int32_t checked_i32(std::uint32_t value, const char* op, const char* name) {
    if (value > static_cast<std::uint32_t>(std::numeric_limits<std::int32_t>::max())) {
        throw std::overflow_error(std::string(op) + ": " + name + " exceeds int32");
    }
    return static_cast<std::int32_t>(value);
}

void require_shape(const Tensor& tensor, std::int32_t n0, std::int32_t n1, std::int32_t n2,
                   std::int32_t n3, const char* op, const char* name) {
    if (tensor.ne[0] != n0 || tensor.ne[1] != n1 || tensor.ne[2] != n2 || tensor.ne[3] != n3) {
        throw std::invalid_argument(std::string(op) + ": invalid shape for " + name);
    }
}

void require_contiguous_nonnull(const Tensor& tensor, const char* op, const char* name) {
    if (!tensor.is_contiguous()) {
        throw std::invalid_argument(std::string(op) + ": " + name + " must be contiguous");
    }
    if (tensor.data == nullptr) {
        throw std::invalid_argument(std::string(op) + ": " + name + " data must be non-null");
    }
}

void validate_cache(const KVCacheLayerView& cache, std::int32_t kv_heads, std::int32_t head_dim,
                    const char* op) {
    const bool is_laguna = (head_dim == 128);
    const int expected_dim = is_laguna ? kLagunaHeadDim : kHeadDim;
    if ((cache.dtype != DType::BF16 && cache.dtype != DType::I8) ||
        cache.num_kv_heads != kv_heads || cache.head_dim != expected_dim) {
        throw std::invalid_argument(std::string(op) + ": invalid KV cache geometry or dtype");
    }
    if (cache.max_context == 0 || cache.padded_context < cache.max_context) {
        throw std::invalid_argument(std::string(op) + ": invalid KV cache capacity");
    }
    if (cache.dtype == DType::BF16 && cache.quant_group != 0) {
        throw std::invalid_argument(std::string(op) + ": BF16 KV cache must not have quant_group");
    }
    if (cache.dtype == DType::I8 && cache.quant_group != kKvQuantGroup) {
        throw std::invalid_argument(std::string(op) + ": I8 KV cache must use quant_group 64");
    }

    const std::int32_t padded = checked_i32(cache.padded_context, op, "padded_context");
    const DType code_dtype    = cache.dtype == DType::I8 ? DType::I8 : DType::BF16;
    if (cache.k.dtype != code_dtype || cache.v.dtype != code_dtype) {
        throw std::invalid_argument(std::string(op) + ": invalid KV cache code dtype");
    }
    require_shape(cache.k, expected_dim, padded, kv_heads, 1, op, "cache k");
    require_shape(cache.v, expected_dim, padded, kv_heads, 1, op, "cache v");
    require_contiguous_nonnull(cache.k, op, "cache k");
    require_contiguous_nonnull(cache.v, op, "cache v");

    if (cache.dtype == DType::BF16) {
        if (cache.k_scale.data != nullptr || cache.v_scale.data != nullptr) {
            throw std::invalid_argument(std::string(op) + ": BF16 KV cache must not have scales");
        }
        return;
    }

    const int groups = expected_dim / kKvQuantGroup;
    if (cache.k_scale.dtype != DType::FP16 || cache.v_scale.dtype != DType::FP16) {
        throw std::invalid_argument(std::string(op) + ": invalid KV cache scale dtype");
    }
    require_shape(cache.k_scale, groups, padded, kv_heads, 1, op, "cache k scale");
    require_shape(cache.v_scale, groups, padded, kv_heads, 1, op, "cache v scale");
    require_contiguous_nonnull(cache.k_scale, op, "cache k scale");
    require_contiguous_nonnull(cache.v_scale, op, "cache v scale");
}

void validate_envelope(GqaExecutionEnvelope envelope, const KVCacheLayerView& cache,
                       std::int32_t tokens, const char* op) {
    if (envelope.min_visible_keys == 0 || envelope.min_visible_keys > envelope.max_visible_keys ||
        envelope.max_visible_keys > cache.max_context) {
        throw std::invalid_argument(std::string(op) + ": invalid execution envelope");
    }
    if (envelope.max_visible_keys < static_cast<std::uint32_t>(tokens)) {
        throw std::invalid_argument(std::string(op) + ": execution envelope is shorter than T");
    }
}

void validate_attention_tensors(const Tensor& q, const Tensor& positions, const Tensor& out,
                                const KVCacheLayerView& cache, GqaExecutionEnvelope envelope,
                                float scale, std::int32_t head_dim, const char* op) {
    const bool is_laguna = (head_dim == 128);
    const int expected_dim = is_laguna ? kLagunaHeadDim : kHeadDim;
    const float expected_scale = is_laguna ? kLagunaExpectedScale : kExpectedScale;
    if (q.dtype != DType::BF16 || out.dtype != DType::BF16) {
        throw std::invalid_argument(std::string(op) + ": q/out must be BF16");
    }
    if (positions.dtype != DType::I32) {
        throw std::invalid_argument(std::string(op) + ": positions must be I32");
    }
    if (!std::isfinite(scale) || std::abs(scale - expected_scale) > 1.0e-5f) {
        throw std::invalid_argument(std::string(op) + ": scale mismatch for head_dim=" +
                                    std::to_string(head_dim));
    }
    const std::int32_t q_heads  = q.ne[1];
    const std::int32_t kv_heads = kv_heads_for_q_heads(q_heads, head_dim, op);
    const std::int32_t tokens   = q.ne[2];
    if (tokens <= 0) { throw std::invalid_argument(std::string(op) + ": T must be positive"); }
    require_shape(q, expected_dim, q_heads, tokens, 1, op, "q");
    require_shape(positions, tokens, 1, 1, 1, op, "positions");
    require_shape(out, expected_dim, q_heads, tokens, 1, op, "out");
    require_contiguous_nonnull(q, op, "q");
    require_contiguous_nonnull(positions, op, "positions");
    require_contiguous_nonnull(out, op, "out");
    validate_cache(cache, kv_heads, head_dim, op);
    validate_envelope(envelope, cache, tokens, op);
}

struct SmallTWorkspace {
    Tensor acc;
    Tensor m;
    Tensor l;
};

template <class Allocator>
SmallTWorkspace allocate_small_t_workspace(Allocator& workspace, std::int32_t q_heads,
                                            std::int32_t head_dim, std::int32_t tokens,
                                            std::int32_t splits) {
    return {
        workspace.alloc(DType::BF16, {head_dim, q_heads, tokens, splits}),
        workspace.alloc(DType::FP32, {q_heads, tokens, splits}),
        workspace.alloc(DType::FP32, {q_heads, tokens, splits}),
    };
}

template <typename Launch>
void for_each_small_t_chunk(const Tensor& q, const Tensor& positions, WorkspaceArena& workspace,
                            DType cache_dtype, GqaExecutionEnvelope envelope, Tensor& out,
                            std::int32_t head_dim, Launch&& launch) {
    const int expected_dim = (head_dim == 128) ? kLagunaHeadDim : kHeadDim;
    for (std::int32_t begin = 0; begin < q.ne[2]; begin += kSmallTChunkTokens) {
        const std::int32_t count = std::min(kSmallTChunkTokens, q.ne[2] - begin);
        auto chunk_scope         = workspace.scope();
        const std::int32_t splits =
            detail::gqa_attention_split_capacity(q.ne[1], count, cache_dtype, envelope);
        SmallTWorkspace partial = allocate_small_t_workspace(workspace, q.ne[1], expected_dim,
                                                             count, splits);
        Tensor q_chunk          = q.slice(2, begin, count);
        Tensor position_chunk   = positions.slice(0, begin, count);
        Tensor out_chunk        = out.slice(2, begin, count);
        launch(begin, count, q_chunk, position_chunk, partial, out_chunk);
    }
}

void launch_chunked_small_t(const Tensor& q, const Tensor& k, const Tensor& v,
                            const Tensor& positions, float scale, KVCacheLayerView cache,
                            GqaExecutionEnvelope envelope, WorkspaceArena& workspace, Tensor& out,
                            std::int32_t head_dim, cudaStream_t stream) {
    for_each_small_t_chunk(
        q, positions, workspace, cache.dtype, envelope, out, head_dim,
        [&](std::int32_t begin, std::int32_t count, const Tensor& q_chunk,
            const Tensor& position_chunk, SmallTWorkspace& partial, Tensor& out_chunk) {
            Tensor k_chunk = k.slice(2, begin, count);
            Tensor v_chunk = v.slice(2, begin, count);
            detail::gqa_attention_small_t_launch(q_chunk, k_chunk, v_chunk, position_chunk, scale,
                                                  cache, envelope, partial.acc, partial.m, partial.l,
                                                  out_chunk, stream);
        });
}

void launch_cached_chunked_small_t(const Tensor& q, const Tensor& positions, float scale,
                                    const KVCacheLayerView& cache, GqaExecutionEnvelope envelope,
                                    WorkspaceArena& workspace, Tensor& out, std::int32_t head_dim,
                                    cudaStream_t stream) {
    for_each_small_t_chunk(
        q, positions, workspace, cache.dtype, envelope, out, head_dim,
        [&](std::int32_t, std::int32_t, const Tensor& q_chunk, const Tensor& position_chunk,
            SmallTWorkspace& partial, Tensor& out_chunk) {
            detail::gqa_attention_cached_small_t_launch(q_chunk, position_chunk, scale, cache,
                                                         envelope, partial.acc, partial.m, partial.l,
                                                         out_chunk, stream);
        });
}

} // namespace

namespace detail {

GqaAttentionRoute gqa_attention_resolve_route(std::int32_t q_heads, std::int32_t head_dim,
                                               std::int32_t tokens,
                                               GqaExecutionEnvelope envelope) {
    (void)envelope;
    if (tokens >= 1 && tokens <= kSmallTChunkTokens) { return GqaAttentionRoute::SmallT; }
    const std::uint32_t prompt_visible_keys = tokens <= 2 * kSmallTChunkTokens
                                                  ? kTwoChunkPromptVisibleKeys
                                                  : kThreeChunkPromptVisibleKeys;
    if (q_heads == 16 && tokens <= kMaximumVerifyTokens &&
        envelope.max_visible_keys > prompt_visible_keys) {
        return GqaAttentionRoute::ChunkedSmallT;
    }
    // Laguna geometries use chunked_small_t for prompt lengths at verify-scale token counts
    if ((head_dim == 128) && tokens <= kMaximumVerifyTokens) {
        return GqaAttentionRoute::ChunkedSmallT;
    }
    return GqaAttentionRoute::Prompt;
}

const char* gqa_attention_route_name(GqaAttentionRoute route) {
    switch (route) {
    case GqaAttentionRoute::SmallT:
        return "small_t";
    case GqaAttentionRoute::ChunkedSmallT:
        return "chunked_small_t";
    case GqaAttentionRoute::Prompt:
        return "prompt";
    }
    return "unknown";
}

} // namespace detail

std::size_t gqa_attention_workspace_capacity_bytes(std::int32_t q_heads, std::int32_t head_dim,
                                                    DType cache_dtype,
                                                    GqaExecutionEnvelope envelope,
                                                    std::int32_t min_tokens,
                                                    std::int32_t max_tokens) {
    const int expected_dim = (head_dim == 128) ? kLagunaHeadDim : kHeadDim;
    (void)kv_heads_for_q_heads(q_heads, head_dim, "gqa_attention workspace");
    if ((cache_dtype != DType::BF16 && cache_dtype != DType::I8) || min_tokens <= 0 ||
        max_tokens < min_tokens || envelope.min_visible_keys == 0 ||
        envelope.min_visible_keys > envelope.max_visible_keys ||
        envelope.max_visible_keys >
            static_cast<std::uint32_t>(std::numeric_limits<std::int32_t>::max()) ||
        envelope.max_visible_keys < static_cast<std::uint32_t>(max_tokens)) {
        throw std::invalid_argument("gqa_attention workspace: invalid profile or interval");
    }

    const auto chunk_capacity = [&](std::int32_t tokens) {
        const std::int32_t splits =
            detail::gqa_attention_split_capacity(q_heads, tokens, cache_dtype, envelope);
        WorkspaceLayoutBuilder layout;
        (void)allocate_small_t_workspace(layout, q_heads, expected_dim, tokens, splits);
        return layout.peak_bytes(1);
    };
    const auto exact_capacity = [&](std::int32_t tokens) {
        const detail::GqaAttentionRoute route =
            detail::gqa_attention_resolve_route(q_heads, expected_dim, tokens, envelope);
        if (route == detail::GqaAttentionRoute::Prompt) { return std::size_t{0}; }
        if (route == detail::GqaAttentionRoute::SmallT) { return chunk_capacity(tokens); }
        std::size_t maximum = 0;
        for (std::int32_t begin = 0; begin < tokens; begin += kSmallTChunkTokens) {
            maximum =
                std::max(maximum, chunk_capacity(std::min(kSmallTChunkTokens, tokens - begin)));
        }
        return maximum;
    };

    std::size_t maximum = 0;
    if (min_tokens <= kMaximumVerifyTokens) {
        const std::int32_t last = std::min(max_tokens, kMaximumVerifyTokens);
        for (std::int32_t tokens = min_tokens; tokens <= last; ++tokens) {
            maximum = std::max(maximum, exact_capacity(tokens));
        }
    }
    return maximum;
}

void gqa_attention(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& positions,
                   float scale, KVCacheLayerView cache, GqaExecutionEnvelope envelope,
                   WorkspaceArena& workspace, Tensor& out, std::int32_t head_dim,
                   cudaStream_t stream) {
    constexpr const char* op = "gqa_attention";
    const int expected_dim = (head_dim == 128) ? kLagunaHeadDim : kHeadDim;
    validate_attention_tensors(q, positions, out, cache, envelope, scale, head_dim, op);
    if (k.dtype != DType::BF16 || v.dtype != DType::BF16) {
        throw std::invalid_argument("gqa_attention: k/v must be BF16");
    }
    const std::int32_t tokens   = q.ne[2];
    const std::int32_t kv_heads = kv_heads_for_q_heads(q.ne[1], head_dim, op);
    require_shape(k, expected_dim, kv_heads, tokens, 1, op, "k");
    require_shape(v, expected_dim, kv_heads, tokens, 1, op, "v");
    require_contiguous_nonnull(k, op, "k");
    require_contiguous_nonnull(v, op, "v");

    auto scope = workspace.scope();
    if (detail::gqa_attention_resolve_route(q.ne[1], expected_dim, tokens, envelope) ==
        detail::GqaAttentionRoute::ChunkedSmallT) {
        launch_chunked_small_t(q, k, v, positions, scale, cache, envelope, workspace, out,
                               static_cast<std::int32_t>(head_dim), stream);
        return;
    }
    if (detail::gqa_attention_uses_small_t(tokens)) {
        const std::int32_t splits =
            detail::gqa_attention_split_capacity(q.ne[1], head_dim, cache.dtype, envelope);
        SmallTWorkspace partial = allocate_small_t_workspace(workspace, q.ne[1], expected_dim,
                                                             tokens, splits);
        detail::gqa_attention_launch(q, k, v, positions, scale, cache, envelope, &partial.acc,
                                      &partial.m, &partial.l, out, stream);
        return;
    }
    detail::gqa_attention_launch(q, k, v, positions, scale, cache, envelope, nullptr, nullptr,
                                  nullptr, out, stream);
}

void gqa_kv_append(const Tensor& k, const Tensor& v, const Tensor& positions,
                   KVCacheLayerView cache, std::int32_t head_dim, cudaStream_t stream) {
    constexpr const char* op = "gqa_kv_append";
    if (k.dtype != DType::BF16 || v.dtype != DType::BF16) {
        throw std::invalid_argument("gqa_kv_append: k/v must be BF16");
    }
    if (positions.dtype != DType::I32) {
        throw std::invalid_argument("gqa_kv_append: positions must be I32");
    }
    const std::int32_t kv_heads = k.ne[1];
    require_kv_heads(kv_heads, head_dim, op);
    const std::int32_t tokens = k.ne[2];
    if (tokens <= 0) { throw std::invalid_argument("gqa_kv_append: T must be positive"); }
    const int expected_dim = (head_dim == 128) ? kLagunaHeadDim : kHeadDim;
    require_shape(k, expected_dim, kv_heads, tokens, 1, op, "k");
    require_shape(v, expected_dim, kv_heads, tokens, 1, op, "v");
    require_shape(positions, tokens, 1, 1, 1, op, "positions");
    require_contiguous_nonnull(k, op, "k");
    require_contiguous_nonnull(v, op, "v");
    require_contiguous_nonnull(positions, op, "positions");
    validate_cache(cache, kv_heads, head_dim, op);
    if (static_cast<std::uint32_t>(tokens) > cache.max_context) {
        throw std::invalid_argument("gqa_kv_append: T exceeds KV cache capacity");
    }
    detail::gqa_kv_append_launch(k, v, positions, cache, stream);
}

void gqa_attention_cached(const Tensor& q, const Tensor& positions, float scale,
                          const KVCacheLayerView& cache, GqaExecutionEnvelope envelope,
                          WorkspaceArena& workspace, Tensor& out, std::int32_t head_dim,
                          cudaStream_t stream) {
    constexpr const char* op = "gqa_attention_cached";
    validate_attention_tensors(q, positions, out, cache, envelope, scale, head_dim, op);

    auto scope = workspace.scope();
    const int expected_dim = (head_dim == 128) ? kLagunaHeadDim : kHeadDim;
    if (detail::gqa_attention_resolve_route(q.ne[1], expected_dim, q.ne[2], envelope) ==
        detail::GqaAttentionRoute::ChunkedSmallT) {
        launch_cached_chunked_small_t(q, positions, scale, cache, envelope, workspace, out,
                                       static_cast<std::int32_t>(head_dim), stream);
        return;
    }
    if (detail::gqa_attention_uses_small_t(q.ne[2])) {
        const std::int32_t splits =
            detail::gqa_attention_split_capacity(q.ne[1], head_dim, cache.dtype, envelope);
        SmallTWorkspace partial = allocate_small_t_workspace(workspace, q.ne[1], expected_dim,
                                                             q.ne[2], splits);
        detail::gqa_attention_cached_small_t_launch(q, positions, scale, cache, envelope,
                                                     partial.acc, partial.m, partial.l, out, stream);
        return;
    }
    detail::gqa_attention_prompt_attention_launch(q, positions, scale, cache, out, stream);
}

} // namespace ninfer::ops
