// Implements: src/ops/launcher/softplus_gate_mul.h
// Broadcast softplus gate multiply kernel. gate [G,T] broadcast over leading D of x [D,G,T].
#include <cuda_bf16.h>

#include <cstdint>

namespace ninfer::ops {

__global__ void softplus_gate_mul_kernel(const __nv_bfloat16* __restrict__ gate,
                                         __nv_bfloat16* __restrict__ x,
                                         int G, int D, int T) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    int g = blockIdx.y * blockDim.y + threadIdx.y;
    int d = blockIdx.z;
    if (t < T && g < G && d < D) {
        float v = __bfloat162float(gate[g * T + t]);
        v = logf(1.0f + expf(v));
        x[(d * G + g) * T + t] = __float2bfloat16_rn(
            __bfloat162float(x[(d * G + g) * T + t]) * v);
    }
}

} // namespace ninfer::ops
