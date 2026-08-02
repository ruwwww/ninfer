// Implements: include/ninfer/ops/softplus_mul.h
// Broadcast softplus gate multiply: gate [G,T] is broadcast over leading dim of x [D,G,T].
#include "ops/launcher/softplus_gate_mul.h"

#include "ops/kernel/softplus_gate_mul.cuh"
#include "core/device.h"

#include <algorithm>
#include <cstdint>

namespace ninfer::ops::detail {

void softplus_gate_mul_launch(const Tensor& gate, Tensor& x, cudaStream_t stream) {
    const std::int32_t G = gate.ne[0];
    const std::int32_t T = gate.ne[2];
    const std::int32_t D = x.ne[0];

    constexpr int kTileT = 16;
    constexpr int kTileG = 16;
    dim3 block(kTileT, kTileG, 1);
    dim3 grid((T + kTileT - 1) / kTileT, (G + kTileG - 1) / kTileG, D);
    softplus_gate_mul_kernel<<<grid, block, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(gate.data),
        static_cast<__nv_bfloat16*>(x.data),
        G, D, T);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
