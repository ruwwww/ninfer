#pragma once

// ninfer::ops::detail - private launch prototype for rope_yarn. Included by the wrapper
// and defined by the CUDA launcher.

#include "core/tensor.h"

#include <cuda_runtime.h>

namespace ninfer::ops::detail {

// inv_freq holds rotary_dim/2 host-computed FP64 frequencies and attention_factor is the
// YaRN rotation scale.
void rope_yarn_launch(const Tensor& positions, int rotary_dim, float attention_factor,
                      const double* inv_freq, Tensor& q, Tensor& k, cudaStream_t stream);

} // namespace ninfer::ops::detail
