#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h> // cudaStream_t

namespace ninfer::ops {

/**
 * YaRN (rope theta, factor, original_max_position_embeddings, and beta bounds) parameters for
 * the rotary frequency ramp. The registered geometry is theta=500000, factor=32,
 * original_max_position_embeddings=8192, beta_slow=1, beta_fast=64, attention_factor
 * =1.3465735902799727.
 */
struct YarnParameters {
    float theta;
    float factor;
    float original_max_position_embeddings;
    float beta_slow;
    float beta_fast;
    float attention_factor;
};

/**
 * Applies split-half NeoX RoPE with YaRN frequency scaling in place. With pairs=rotary_dim/2,
 * correction dimensions low=floor(find_correction_dim(beta_fast)) and
 * high=ceil(find_correction_dim(beta_slow)) clamped to [0,rotary_dim-1] where
 * find_correction_dim(r) = (rotary_dim * ln(ompe / (r * 2*pi))) / (2 * ln(theta)), and per
 * pair i in [0,pairs):
 *
 *   pos_freqs(i)   = theta^(2*i/rotary_dim)
 *   inv_freq(i)    = (1/(factor*pos_freqs(i)))*ramp(i) + (1/pos_freqs(i))*(1-ramp(i))
 *   ramp(i)        = clamp((i-low)/(high-low), 0, 1)
 *   angle(i,t)     = positions[t] * inv_freq(i)
 *   ideal[i]       = attention_factor * (x[i] * cos(angle) - x[i+pairs] * sin(angle))
 *   ideal[i+pairs] = attention_factor * (x[i+pairs] * cos(angle) + x[i] * sin(angle))
 *
 * The 32-value inv_freq array is computed on the host in FP64 from the parameters once per
 * call; it is independent of position. Dimensions [rotary_dim,head_dim) are unchanged. The
 * supported mode is Text 1-D with head_dim=128 and rotary_dim=64 (the registered YARN decode
 * geometry); any positive head counts are accepted.
 *
 * positions is contiguous I32 [T]. Q/K tensors are BF16 [head_dim,heads,T] with contiguous
 * head features and heads and an optional padded token stride. q and k must not overlap one
 * another or positions. The Op mutates only dimensions [0,rotary_dim) of the supplied Q/K
 * tensor storage. The oracle evaluates the rotated dimensions naively in FP64 from the
 * represented inputs. The updated BF16 values are promoted and compared directly with that
 * result; output storage rounding belongs to the Op's numerical criterion, not the oracle.
 * Unrotated dimensions remain bit-exact. Private kernel arithmetic is implementation-defined.
 * The Op uses no workspace or persistent state.
 */
void rope_yarn(const Tensor& positions, int rotary_dim, const YarnParameters& yarn, Tensor& q,
               Tensor& k, cudaStream_t stream);

} // namespace ninfer::ops
