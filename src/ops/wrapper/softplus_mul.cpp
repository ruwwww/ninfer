// ninfer::ops - softplus_mul wrapper: implements the public api, validates parameters,
// and dispatches to the launcher. Host-compiled; never includes the kernel header.
// See docs/op-development.md §2.
#include "ninfer/ops/softplus_mul.h"

#include "ops/launcher/softplus_gate_mul.h"

#include <cstdint>
#include <limits>
#include <stdexcept>

namespace ninfer::ops {
namespace {

std::int64_t numel_allow_zero(const Tensor& t, const char* label) {
    std::int64_t total = 1;
    for (int d = 0; d < 4; ++d) {
        if (t.ne[d] < 0) {
            throw std::invalid_argument(std::string("softplus_mul: ") + label +
                                        " dimensions must be nonnegative");
        }
        if (t.ne[d] == 0) { return 0; }
        if (total > std::numeric_limits<std::int64_t>::max() / t.ne[d]) {
            throw std::overflow_error(std::string("softplus_mul: ") + label + " size overflows int64");
        }
        total *= t.ne[d];
    }
    return total;
}

} // namespace

void softplus_mul(const Tensor& gate, Tensor& x, cudaStream_t stream) {
    if (gate.dtype != DType::BF16 || x.dtype != DType::BF16) {
        throw std::invalid_argument("softplus_mul: gate/x must be BF16");
    }
    if (gate.ne[0] != x.ne[1] || gate.ne[2] != x.ne[2]) {
        throw std::invalid_argument("softplus_mul: gate [G,T] and x [D,G,T] shapes must match on G and T");
    }
    if (gate.ne[1] != 1 || x.ne[3] != 1) {
        throw std::invalid_argument("softplus_mul: gate must be [G,1,T,1] and x must be [D,G,T,1]");
    }
    if (numel_allow_zero(x, "x") == 0) { return; }
    if (!gate.is_contiguous() || !x.is_contiguous()) {
        throw std::invalid_argument("softplus_mul: gate/x must be contiguous");
    }
    if (gate.data == nullptr || x.data == nullptr) {
        throw std::invalid_argument("softplus_mul: gate/x data must be non-null");
    }

    detail::softplus_gate_mul_launch(gate, x, stream);
}

} // namespace ninfer::ops
