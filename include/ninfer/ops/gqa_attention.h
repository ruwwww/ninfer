#pragma once

#include "core/kv_cache.h"
#include "core/tensor.h"

#include <cuda_runtime.h> // cudaStream_t

#include <cstddef>
#include <cstdint>

namespace ninfer::ops {

struct GqaExecutionEnvelope {
    std::uint32_t min_visible_keys = 0;
    std::uint32_t max_visible_keys = 0;
};

/**
 * Shared numerical contract for A1/A2/A3.
 *
 * Public q/k/v inputs and BF16 cache values are interpreted after their BF16 storage boundary.
 * INT8-G64 cache rows use one FP16 scale for each contiguous 64-element group. For BF16 source
 * values x, their exact observable encoding is:
 *
 *   a          = max_i abs(FP32(x[i]))
 *   scale_bits = FP16_RNE(a / 127)
 *   s          = FP32(scale_bits)
 *   inv        = s == 0 ? 0 : FP32(1 / s)
 *   code[i]    = s == 0 ? 0 : I8(clamp(RNE_even(FP32(x[i]) * inv), -127, 127))
 *   decode[i]  = FP32(code[i]) * s
 *
 * A1 and A2 produce identical code and scale bits. The common ideal attention oracle uses BF16 Q
 * and logical cache values (BF16 values for a BF16 cache, FP32 decode above for INT8-G64), then
 * evaluates score dot products, stable softmax, and value reduction in FP64. The BF16 Op output is
 * promoted to FP64 for comparison with that result.
 *
 * The registered INT8 implementation defines Q8-G64, paired with INT8-G64 K, as its native query
 * compute profile. Its profile-defined query quantization and any narrower staging do not replace
 * BF16 Q in the ideal oracle. BF16-cache and INT8-cache compute profiles therefore have separate
 * named numerical criteria owned by the GQA conformance test. Those envelopes apply to the
 * registered geometries, tested token extents, conformance matrix, and target-representative
 * activation range; they are not a universal error bound for arbitrary adversarial BF16 tensors.
 * A1 and A3 are each qualified directly against the ideal oracle. A1-versus-A3 parity is only an
 * additional consistency check.
 */

/**
 * Returns the transient arena capacity required for every T in the inclusive interval. Head
 * geometry, cache dtype, and execution envelope are the fixed implementation profile. Invalid
 * profiles or intervals throw; a legal prompt route may return zero.
 */
[[nodiscard]] std::size_t gqa_attention_workspace_capacity_bytes(std::int32_t q_heads,
                                                                  std::int32_t head_dim,
                                                                  DType cache_dtype,
                                                                  GqaExecutionEnvelope envelope,
                                                                  std::int32_t min_tokens,
                                                                  std::int32_t max_tokens);

/**
 * A1: append K/V at the supplied absolute positions and compute causal grouped-query attention.
 * For query head h, kvh=floor(h/group), p=positions[t], and populated cache history [0,p]:
 *
 *   score[j]    = scale * dot(q[:,h,t], K_cache[:,j,kvh]), 0 <= j <= p
 *   probability = softmax_j(score)
 *   ideal[:,h,t] = sum_j probability[j] * V_cache[:,j,kvh].
 *
 * Supported geometries:
 *   - head_dim=256: [256,24|4,T] group 6 (Qwen3.6 27B), [256,16|2,T] group 8 (Qwen3.6 35B)
 *   - head_dim=128: [128,48|8,T] group 6 (Laguna XS 2.1 full), [128,64|8,T] group 8 (Laguna SWA)
 * q/k/v/out are contiguous BF16, positions is contiguous sequential I32 [T].
 * T may be any positive value that fits the declared cache and execution envelope.
 * Cache storage is BF16 or INT8-G64 under the shared numerical contract above.
 */
void gqa_attention(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& positions,
                   float scale, KVCacheLayerView cache, GqaExecutionEnvelope envelope,
                   WorkspaceArena& workspace, Tensor& out, std::int32_t head_dim,
                   cudaStream_t stream);

/**
 * A2: perform only the cache-write part of A1. k/v are contiguous BF16, positions is
 * contiguous sequential I32 [T], and every addressed code and INT8 scale is overwritten.
 * Supported head_dim: 256 (Qwen3.6) or 128 (Laguna XS 2.1).
 */
void gqa_kv_append(const Tensor& k, const Tensor& v, const Tensor& positions,
                    KVCacheLayerView cache, std::int32_t head_dim, cudaStream_t stream);

/**
 * A3: compute causal attention from an already populated cache without accepting new K/V or
 * mutating any cache plane. q/out are contiguous BF16, positions is contiguous sequential I32 [T].
 * Supported head_dim: 256 (Qwen3.6) or 128 (Laguna XS 2.1).
 */
void gqa_attention_cached(const Tensor& q, const Tensor& positions, float scale,
                          const KVCacheLayerView& cache, GqaExecutionEnvelope envelope,
                          WorkspaceArena& workspace, Tensor& out, std::int32_t head_dim,
                          cudaStream_t stream);

} // namespace ninfer::ops