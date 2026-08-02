#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h>

namespace ninfer::ops {

/**
 * Elementwise softplus gate with broadcast:
 *
 *   ideal[d,g,t] = x[d,g,t] * log(1 + exp(gate[g,t])).
 *
 * `gate` is BF16 [G, T] (one value per group per token). `x` is BF16 [D, G, T] with the
 * same G and T as gate; the gate value is broadcast over the leading D dimension. The oracle
 * evaluates `ideal` in FP64 from the represented inputs. The updated BF16 x is promoted and
 * compared directly with that result; output storage rounding belongs to the Op's numerical
 * criterion, not the oracle. Private kernel arithmetic is implementation-defined. The Op uses
 * no workspace or other persistent state.
 */
void softplus_mul(const Tensor& gate, Tensor& x, cudaStream_t stream);

} // namespace ninfer::ops
