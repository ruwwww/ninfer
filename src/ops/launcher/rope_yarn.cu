// ninfer::ops - rope_yarn launcher: generic per-token kernel at the registered geometry.
#include "ops/launcher/rope_yarn.h"

#include "core/device.h" // CUDA_CHECK
#include "ops/kernel/rope_yarn.cuh"

#include <cstdint>

namespace ninfer::ops::detail {
namespace {

std::int64_t token_stride(const Tensor& tensor) {
    return tensor.nb[2] / static_cast<std::int64_t>(sizeof(__nv_bfloat16));
}

} // namespace

void rope_yarn_launch(const Tensor& positions, int rotary_dim, float attention_factor,
                      const double* inv_freq, Tensor& q, Tensor& k, cudaStream_t stream) {
    constexpr int block = 128;
    const int tokens    = positions.ne[0];
    RopeYarnParams params;
    params.attention_factor = attention_factor;
    for (int pair = 0; pair < kRopeYarnHalf; ++pair) {
        params.inv_freq[pair] = inv_freq[pair];
    }
    rope_yarn_kernel<<<tokens, block, 0, stream>>>(
        static_cast<const std::int32_t*>(positions.data),
        static_cast<__nv_bfloat16*>(q.data), static_cast<__nv_bfloat16*>(k.data), q.ne[0],
        q.ne[1], k.ne[1], tokens, token_stride(q), token_stride(k), params);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
