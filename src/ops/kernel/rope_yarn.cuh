#pragma once

// Implements: include/ninfer/ops/rope_yarn.h
// YaRN split-half NeoX rotation at the registered Text D128/R64 geometry, with the 32-value
// FP64 inv_freq array and attention_factor delivered through the kernel parameter struct.
// One CTA owns one token and shares its rotary coefficients across heads.

#include <cuda_bf16.h>

#include <cmath>
#include <cstdint>

namespace ninfer::ops {

inline constexpr int kRopeYarnHalf = 32;

struct RopeYarnParams {
    double inv_freq[kRopeYarnHalf];
    float attention_factor;
};

__global__ void rope_yarn_kernel(const std::int32_t* positions, __nv_bfloat16* q,
                                 __nv_bfloat16* k, std::int32_t head_dim, std::int32_t q_heads,
                                 std::int32_t k_heads, std::int32_t tokens,
                                 std::int64_t q_token_stride, std::int64_t k_token_stride,
                                 RopeYarnParams params) {
    const int token = static_cast<int>(blockIdx.x);
    if (token >= tokens) { return; }

    __shared__ float cos_cache[kRopeYarnHalf];
    __shared__ float sin_cache[kRopeYarnHalf];
    if (threadIdx.x < static_cast<unsigned>(kRopeYarnHalf)) {
        const int pair = static_cast<int>(threadIdx.x);
        constexpr double kInvTwoPi = 1.59154943091895336e-01;
        constexpr double kTwoPi    = 6.28318530717958648e+00;
        const double angle = static_cast<double>(positions[token]) * params.inv_freq[pair];
        const double turns = angle * kInvTwoPi;
        const float reduced = static_cast<float>(angle - nearbyint(turns) * kTwoPi);
        sincosf(reduced, &sin_cache[pair], &cos_cache[pair]);
        sin_cache[pair] *= params.attention_factor;
        cos_cache[pair] *= params.attention_factor;
    }
    __syncthreads();

    const int lane        = static_cast<int>(threadIdx.x) & 31;
    const int warp        = static_cast<int>(threadIdx.x) >> 5;
    const int block_warps = static_cast<int>(blockDim.x) >> 5;
    for (int combined_head = warp; combined_head < q_heads + k_heads;
         combined_head += block_warps) {
        const bool is_q             = combined_head < q_heads;
        const int head              = is_q ? combined_head : combined_head - q_heads;
        __nv_bfloat16* data         = is_q ? q : k;
        const std::int64_t stride_t = is_q ? q_token_stride : k_token_stride;
        const std::int64_t base     = static_cast<std::int64_t>(token) * stride_t +
                                  static_cast<std::int64_t>(head) * head_dim;
        for (int pair = lane; pair < kRopeYarnHalf; pair += 32) {
            const float first        = __bfloat162float(data[base + pair]);
            const float second       = __bfloat162float(data[base + pair + kRopeYarnHalf]);
            const float c            = cos_cache[pair];
            const float s            = sin_cache[pair];
            data[base + pair]        = __float2bfloat16_rn(first * c - second * s);
            data[base + pair + kRopeYarnHalf] = __float2bfloat16_rn(second * c + first * s);
        }
    }
}

} // namespace ninfer::ops
