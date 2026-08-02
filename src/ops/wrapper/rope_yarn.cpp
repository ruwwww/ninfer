#include "ninfer/ops/rope_yarn.h"

#include "ops/launcher/rope_yarn.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace ninfer::ops {
namespace {

constexpr std::int32_t kTextHeadDim   = 128;
constexpr std::int32_t kTextRotaryDim = 64;
constexpr std::int32_t kHalf          = kTextRotaryDim / 2;
constexpr double kTwoPi               = 6.28318530717958648e+00;

std::int64_t numel_allow_zero(const Tensor& tensor, const char* label) {
    bool zero      = false;
    std::int64_t n = 1;
    for (int dim = 0; dim < 4; ++dim) {
        if (tensor.ne[dim] < 0) {
            throw std::invalid_argument(std::string("rope_yarn: ") + label +
                                        " dimensions must be nonnegative");
        }
        if (tensor.ne[dim] == 0) {
            zero = true;
            continue;
        }
        if (n > std::numeric_limits<std::int64_t>::max() / tensor.ne[dim]) {
            throw std::overflow_error("rope_yarn: tensor size overflows int64");
        }
        n *= tensor.ne[dim];
    }
    return zero ? 0 : n;
}

void require_positions_shape(const Tensor& positions, std::int32_t tokens) {
    if (positions.dtype != DType::I32) {
        throw std::invalid_argument("rope_yarn: positions must be I32");
    }
    if (positions.ne[0] != tokens || positions.ne[1] != 1 || positions.ne[2] != 1 ||
        positions.ne[3] != 1) {
        throw std::invalid_argument("rope_yarn: positions must have shape [T]");
    }
}

void require_positions_storage(const Tensor& positions) {
    if (!positions.is_contiguous()) {
        throw std::invalid_argument("rope_yarn: positions must be contiguous");
    }
    if (positions.data == nullptr) {
        throw std::invalid_argument("rope_yarn: positions data must be non-null");
    }
}

void require_tensor_layout(const Tensor& tensor, const char* label, std::int32_t heads,
                           std::int32_t tokens) {
    if (heads <= 0) {
        throw std::invalid_argument(std::string("rope_yarn: ") + label +
                                    " must have positive heads");
    }
    if (tensor.ne[0] != kTextHeadDim || tensor.ne[1] != heads || tensor.ne[2] != tokens ||
        tensor.ne[3] != 1) {
        throw std::invalid_argument(std::string("rope_yarn: invalid ") + label + " shape");
    }
    constexpr std::int64_t elem = 2;
    if (tensor.nb[0] != elem || tensor.nb[1] != elem * kTextHeadDim ||
        tensor.nb[2] < elem * static_cast<std::int64_t>(kTextHeadDim) * heads ||
        (tensor.nb[2] % elem) != 0) {
        throw std::invalid_argument(std::string("rope_yarn: invalid ") + label + " strides");
    }
}

void require_yarn(const YarnParameters& yarn) {
    const auto positive_finite = [](float value) { return value > 0.0F && std::isfinite(value); };
    if (!positive_finite(yarn.theta) || !positive_finite(yarn.factor) ||
        !positive_finite(yarn.original_max_position_embeddings) ||
        !positive_finite(yarn.beta_slow) || !positive_finite(yarn.beta_fast) ||
        !positive_finite(yarn.attention_factor)) {
        throw std::invalid_argument("rope_yarn: yarn parameters must be positive and finite");
    }
}

void require_model_mode(std::int32_t rotary_dim) {
    if (rotary_dim != kTextRotaryDim) {
        throw std::invalid_argument("rope_yarn: Text YARN mode requires head_dim=128, "
                                    "rotary_dim=64");
    }
}

double correction_dim(float proposed_dim, const YarnParameters& yarn) {
    return (static_cast<double>(kTextRotaryDim) *
            std::log(static_cast<double>(yarn.original_max_position_embeddings) /
                     (static_cast<double>(proposed_dim) * kTwoPi))) /
           (2.0 * std::log(static_cast<double>(yarn.theta)));
}

void compute_inv_freq(const YarnParameters& yarn, double* inv_freq) {
    const std::int32_t low = static_cast<std::int32_t>(std::clamp(
        std::floor(correction_dim(yarn.beta_fast, yarn)), 0.0,
        static_cast<double>(kTextRotaryDim - 1)));
    const std::int32_t high = static_cast<std::int32_t>(std::clamp(
        std::ceil(correction_dim(yarn.beta_slow, yarn)), 0.0,
        static_cast<double>(kTextRotaryDim - 1)));
    for (int pair = 0; pair < kHalf; ++pair) {
        const double pos_freq =
            std::pow(static_cast<double>(yarn.theta), static_cast<double>(2 * pair) / kTextRotaryDim);
        const double inv_extrap = 1.0 / pos_freq;
        const double inv_interp = 1.0 / (static_cast<double>(yarn.factor) * pos_freq);
        double ramp             = 0.0;
        if (high != low) {
            ramp = std::clamp(static_cast<double>(pair - low) /
                                  static_cast<double>(high - low),
                              0.0, 1.0);
        }
        inv_freq[pair] = inv_interp * ramp + inv_extrap * (1.0 - ramp);
    }
}

} // namespace

void rope_yarn(const Tensor& positions, int rotary_dim, const YarnParameters& yarn, Tensor& q,
               Tensor& k, cudaStream_t stream) {
    require_yarn(yarn);
    require_model_mode(rotary_dim);
    if (q.dtype != DType::BF16 || k.dtype != DType::BF16) {
        throw std::invalid_argument("rope_yarn: q/k must be BF16");
    }
    (void)numel_allow_zero(positions, "positions");
    const std::int64_t q_numel = numel_allow_zero(q, "q");
    (void)numel_allow_zero(k, "k");
    const std::int32_t tokens   = q.ne[2];
    const std::int32_t q_heads  = q.ne[1];
    const std::int32_t k_heads  = k.ne[1];
    require_positions_shape(positions, tokens);
    require_tensor_layout(q, "q", q_heads, tokens);
    require_tensor_layout(k, "k", k_heads, tokens);
    if (q_numel == 0) { return; }
    require_positions_storage(positions);
    if (q.data == nullptr || k.data == nullptr) {
        throw std::invalid_argument("rope_yarn: q/k data must be non-null");
    }
    double inv_freq[kHalf];
    compute_inv_freq(yarn, inv_freq);
    detail::rope_yarn_launch(positions, rotary_dim, yarn.attention_factor, inv_freq, q, k, stream);
}

} // namespace ninfer::ops
