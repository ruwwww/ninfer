#include "ninfer/ops/rope_yarn.h"
#include "ops/op_tester.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <random>
#include <string>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;

namespace {

constexpr ops::YarnParameters kYarn{
    500'000.0F,                 // theta
    32.0F,                      // factor
    8192.0F,                    // original_max_position_embeddings
    1.0F,                       // beta_slow
    64.0F,                      // beta_fast
    1.3465735902799727F,        // attention_factor
};

constexpr int kHeadDim    = 128;
constexpr int kRotaryDim  = 64;
constexpr int kHalf       = kRotaryDim / 2;

// Either member of a rotated pair can approach zero through cancellation. The scale-invariant
// RoPE BF16 profile therefore bounds each output by the FP64 norm of its public input pair rather
// than by the cancelled output value.
constexpr double kRopePointwisePairRtol = 6.9e-3;

std::size_t dense_elements(int heads, int tokens) {
    return static_cast<std::size_t>(kHeadDim) * static_cast<std::size_t>(heads) *
           static_cast<std::size_t>(tokens);
}

std::size_t dense_index(int heads, int token, int head, int dim) {
    return (static_cast<std::size_t>(token) * static_cast<std::size_t>(heads) +
            static_cast<std::size_t>(head)) *
               static_cast<std::size_t>(kHeadDim) +
           static_cast<std::size_t>(dim);
}

std::vector<float> make_bf16_input(std::size_t elements, std::uint32_t seed) {
    std::vector<float> input(elements);
    std::mt19937 generator(seed);
    std::uniform_real_distribution<float> distribution(-4.0F, 4.0F);
    for (float& value : input) { value = bf16_to_f32(f32_to_bf16(distribution(generator))); }
    return input;
}

std::vector<std::uint16_t> to_bf16_bits(const std::vector<float>& values) {
    std::vector<std::uint16_t> bits(values.size());
    std::transform(values.begin(), values.end(), bits.begin(),
                   [](float value) { return f32_to_bf16(value); });
    return bits;
}

std::vector<int> make_positions(int tokens) {
    static constexpr int kSpecials[] = {0, 1, 8191, 8192, 262143, 131071};
    std::vector<int> positions(static_cast<std::size_t>(tokens));
    for (int token = 0; token < tokens; ++token) {
        if (token < 6) {
            positions[static_cast<std::size_t>(token)] = kSpecials[token];
        } else {
            positions[static_cast<std::size_t>(token)] = 4096 + 7 * (token - 6);
        }
    }
    return positions;
}

// This is the sole rope_yarn oracle. It evaluates the documented YaRN inv_freq ramp and the
// split-half rotation naively in FP64 from the represented public inputs. It does not reproduce
// output storage rounding, production staging, coefficient tables, or kernel decomposition.
std::vector<double> yarn_inv_freq_oracle(const ops::YarnParameters& yarn) {
    std::vector<double> inv_freq(static_cast<std::size_t>(kHalf));
    const auto correction_dim = [&yarn](double proposed_dim) {
        return (static_cast<double>(kRotaryDim) *
                std::log(static_cast<double>(yarn.original_max_position_embeddings) /
                         (proposed_dim * 2.0 * 3.14159265358979323846))) /
               (2.0 * std::log(static_cast<double>(yarn.theta)));
    };
    const double low_raw  = std::floor(correction_dim(yarn.beta_fast));
    const double high_raw = std::ceil(correction_dim(yarn.beta_slow));
    const int low         = static_cast<int>(
        std::min(std::max(low_raw, 0.0), static_cast<double>(kRotaryDim - 1)));
    const int high = static_cast<int>(
        std::min(std::max(high_raw, 0.0), static_cast<double>(kRotaryDim - 1)));
    for (int pair = 0; pair < kHalf; ++pair) {
        const double pos_freq =
            std::pow(static_cast<double>(yarn.theta), static_cast<double>(2 * pair) / kRotaryDim);
        const double inv_extrap = 1.0 / pos_freq;
        const double inv_interp = 1.0 / (static_cast<double>(yarn.factor) * pos_freq);
        double ramp             = 0.0;
        if (high != low) {
            ramp = std::min(std::max(static_cast<double>(pair - low) /
                                         static_cast<double>(high - low),
                                     0.0),
                            1.0);
        }
        inv_freq[static_cast<std::size_t>(pair)] =
            inv_interp * ramp + inv_extrap * (1.0 - ramp);
    }
    return inv_freq;
}

std::vector<double> rope_yarn_oracle(const std::vector<float>& input,
                                     const std::vector<int>& positions,
                                     const std::vector<double>& inv_freq, int heads,
                                     float attention_factor) {
    std::vector<double> output(input.begin(), input.end());
    const int tokens = positions.size();
    for (int token = 0; token < tokens; ++token) {
        for (int head = 0; head < heads; ++head) {
            for (int pair = 0; pair < kHalf; ++pair) {
                const double angle =
                    static_cast<double>(positions[static_cast<std::size_t>(token)]) *
                    inv_freq[static_cast<std::size_t>(pair)];
                const double cosine    = std::cos(angle);
                const double sine      = std::sin(angle);
                const std::size_t lo   = dense_index(heads, token, head, pair);
                const std::size_t hi   = dense_index(heads, token, head, pair + kHalf);
                const double first     = static_cast<double>(input[lo]);
                const double second    = static_cast<double>(input[hi]);
                const double scale     = static_cast<double>(attention_factor);
                output[lo]             = scale * (first * cosine - second * sine);
                output[hi]             = scale * (second * cosine + first * sine);
            }
        }
    }
    return output;
}

std::vector<std::uint16_t> make_strided_storage(const std::vector<std::uint16_t>& dense,
                                                int dense_token_elements, int token_stride,
                                                int tokens, std::uint16_t padding) {
    std::vector<std::uint16_t> storage(static_cast<std::size_t>(token_stride) * tokens, padding);
    for (int token = 0; token < tokens; ++token) {
        std::copy_n(dense.data() + static_cast<std::size_t>(token) * dense_token_elements,
                    dense_token_elements,
                    storage.data() + static_cast<std::size_t>(token) * token_stride);
    }
    return storage;
}

std::vector<double> gather_dense(const std::vector<std::uint16_t>& storage,
                                 int dense_token_elements, int token_stride, int tokens) {
    std::vector<double> dense(static_cast<std::size_t>(dense_token_elements) * tokens);
    for (int token = 0; token < tokens; ++token) {
        for (int element = 0; element < dense_token_elements; ++element) {
            dense[static_cast<std::size_t>(token) * dense_token_elements + element] =
                static_cast<double>(
                    bf16_to_f32(storage[static_cast<std::size_t>(token) * token_stride + element]));
        }
    }
    return dense;
}

int verify_rope_profile(const std::string& label, const std::vector<double>& got,
                        const std::vector<double>& expected, const std::vector<float>& input,
                        int heads, int tokens) {
    if (got.size() != expected.size() || got.size() != input.size()) {
        std::cerr << label << ": profile input size mismatch\n";
        return 1;
    }

    double case_max_abs            = 0.0;
    double case_required_pair_rtol = 0.0;
    int violations                 = 0;
    for (int token = 0; token < tokens; ++token) {
        for (int head = 0; head < heads; ++head) {
            for (int dim = 0; dim < kRotaryDim; ++dim) {
                const int pair          = dim < kHalf ? dim : dim - kHalf;
                const std::size_t index = dense_index(heads, token, head, dim);
                const std::size_t lo    = dense_index(heads, token, head, pair);
                const std::size_t hi    = dense_index(heads, token, head, pair + kHalf);
                if (!std::isfinite(got[index]) || !std::isfinite(expected[index])) {
                    std::cerr << label << ": non-finite output at index=" << index << '\n';
                    return 1;
                }
                const double abs_error = std::abs(got[index] - expected[index]);
                const double pair_norm =
                    std::hypot(static_cast<double>(input[lo]), static_cast<double>(input[hi]));
                const double required_pair_rtol =
                    pair_norm == 0.0
                        ? (abs_error == 0.0 ? 0.0 : std::numeric_limits<double>::infinity())
                        : abs_error / pair_norm;
                if (required_pair_rtol > case_required_pair_rtol) {
                    case_required_pair_rtol = required_pair_rtol;
                }
                case_max_abs = std::max(case_max_abs, abs_error);
                if (abs_error > kRopePointwisePairRtol * pair_norm) {
                    ++violations;
                    if (violations == 1) {
                        std::cerr << label << ": pointwise BF16 profile mismatch at index=" << index
                                  << " abs_error=" << abs_error << " pair_norm=" << pair_norm
                                  << '\n';
                    }
                }
            }
        }
    }

    report_scaled_pointwise_stats(
        label, static_cast<std::int64_t>(tokens) * heads * kRotaryDim, case_max_abs,
        case_required_pair_rtol, kRopePointwisePairRtol);
    if (violations != 0) {
        std::cerr << label << ": " << violations << " values exceed the unified RoPE profile\n";
        return 1;
    }
    return 0;
}

int verify_passthrough(const std::string& label, const std::vector<std::uint16_t>& got_storage,
                       const std::vector<std::uint16_t>& before_dense, int heads, int tokens,
                       int token_stride) {
    for (int token = 0; token < tokens; ++token) {
        for (int head = 0; head < heads; ++head) {
            for (int dim = kRotaryDim; dim < kHeadDim; ++dim) {
                const std::size_t dense  = dense_index(heads, token, head, dim);
                const std::size_t stored = static_cast<std::size_t>(token) * token_stride +
                                           static_cast<std::size_t>(head) * kHeadDim + dim;
                if (got_storage[stored] != before_dense[dense]) {
                    std::cerr << label << ": non-rotary dimension changed at token=" << token
                              << " head=" << head << " dim=" << dim << '\n';
                    return 1;
                }
            }
        }
    }
    return 0;
}

int verify_padding(const std::string& label, const std::vector<std::uint16_t>& storage,
                   int dense_token_elements, int token_stride, int tokens, std::uint16_t padding) {
    for (int token = 0; token < tokens; ++token) {
        for (int element = dense_token_elements; element < token_stride; ++element) {
            if (storage[static_cast<std::size_t>(token) * token_stride + element] != padding) {
                std::cerr << label << ": token padding changed at token=" << token
                          << " element=" << element << '\n';
                return 1;
            }
        }
    }
    return 0;
}

int run_case(const std::string& label, int q_heads, int k_heads, int tokens, int q_padding = 0,
             int k_padding = 0) {
    constexpr std::uint16_t kPadding = 0x3f81U;
    const int q_dense_per_token      = kHeadDim * q_heads;
    const int k_dense_per_token      = kHeadDim * k_heads;
    const int q_stride               = q_dense_per_token + q_padding;
    const int k_stride               = k_dense_per_token + k_padding;
    const auto inv_freq              = yarn_inv_freq_oracle(kYarn);

    const auto q = make_bf16_input(dense_elements(q_heads, tokens), 0x1001U);
    const auto k = make_bf16_input(dense_elements(k_heads, tokens), 0x2001U);
    const auto q_before = to_bf16_bits(q);
    const auto k_before = to_bf16_bits(k);
    const auto q_storage =
        make_strided_storage(q_before, q_dense_per_token, q_stride, tokens, kPadding);
    const auto k_storage =
        make_strided_storage(k_before, k_dense_per_token, k_stride, tokens, kPadding);
    const auto positions = make_positions(tokens);
    const auto q_expected =
        rope_yarn_oracle(q, positions, inv_freq, q_heads, kYarn.attention_factor);
    const auto k_expected =
        rope_yarn_oracle(k, positions, inv_freq, k_heads, kYarn.attention_factor);

    GuardedDeviceBuffer q_device(q_storage.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer k_device(k_storage.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer position_device(positions.size() * sizeof(int));
    q_device.copy_from_host(q_storage.data(), q_device.bytes());
    k_device.copy_from_host(k_storage.data(), k_device.bytes());
    position_device.copy_from_host(positions.data(), position_device.bytes());

    Tensor position_tensor(position_device.data(), DType::I32, {tokens});
    Tensor q_tensor(q_device.data(), DType::BF16, {kHeadDim, q_heads, tokens});
    Tensor k_tensor(k_device.data(), DType::BF16, {kHeadDim, k_heads, tokens});
    q_tensor.nb[2] = static_cast<std::int64_t>(q_stride) * sizeof(std::uint16_t);
    k_tensor.nb[2] = static_cast<std::int64_t>(k_stride) * sizeof(std::uint16_t);

    ops::rope_yarn(position_tensor, kRotaryDim, kYarn, q_tensor, k_tensor, nullptr);
    cuda_synchronize();

    const auto q_got = from_device<std::uint16_t>(q_device.data(), q_storage.size());
    const auto k_got = from_device<std::uint16_t>(k_device.data(), k_storage.size());
    int failures     = 0;
    failures += verify_rope_profile(
        label + " q", gather_dense(q_got, q_dense_per_token, q_stride, tokens), q_expected, q,
        q_heads, tokens);
    failures += verify_rope_profile(
        label + " k", gather_dense(k_got, k_dense_per_token, k_stride, tokens), k_expected, k,
        k_heads, tokens);
    failures += verify_passthrough(label + " q", q_got, q_before, q_heads, tokens, q_stride);
    failures += verify_passthrough(label + " k", k_got, k_before, k_heads, tokens, k_stride);
    failures +=
        verify_padding(label + " q", q_got, q_dense_per_token, q_stride, tokens, kPadding);
    failures +=
        verify_padding(label + " k", k_got, k_dense_per_token, k_stride, tokens, kPadding);
    failures += verify_exact((label + " positions").c_str(),
                             from_device<int>(position_device.data(), positions.size()), positions);
    failures += q_device.verify_guards((label + " q guards").c_str());
    failures += k_device.verify_guards((label + " k guards").c_str());
    failures += position_device.verify_guards((label + " position guards").c_str());
    return failures;
}

} // namespace

int main() {
    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }

    int failures = 0;

    // Registered Laguna decode geometry: 48 Q heads / 8 K heads across short, padded, and long
    // token runs, with the special positions {0, 1, 8191, 8192, 262143, 131071} covered.
    failures += run_case("yarn 48q/8k decode single", 48, 8, 1);
    failures += run_case("yarn 48q/8k decode padded", 48, 8, 16, 16, 8);
    failures += run_case("yarn 48q/8k prefill", 48, 8, 128);

    // Small synthetic geometry exercises the generic head loop at 4 Q heads / 2 K heads.
    failures += run_case("yarn synthetic", 4, 2, 16, 8, 0);

    std::cout << (failures == 0 ? "OK" : "FAIL") << " rope_yarn correctness\n";
    return failures == 0 ? 0 : 1;
}
