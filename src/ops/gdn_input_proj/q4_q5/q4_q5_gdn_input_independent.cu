#include "ops/gdn_input_proj/q4_q5/q4_q5_gdn_input_kernels.h"

#include "core/device.h"
#include "core/pdl.cuh"
#include "ops/common/math.h"
#include "ops/linear/q4/q4_rowsplit_gemm_simt.cuh"
#include "ops/linear/q4/q4_rowsplit_gemv.cuh"
#include "ops/linear/q5/q5_rowsplit_gemm_simt.cuh"
#include "ops/linear/q5/q5_rowsplit_gemv.cuh"

#include <cuda_bf16.h>

#include <cstdint>
#include <stdexcept>
#include <type_traits>

namespace ninfer::ops::detail {
namespace {

template <std::int32_t QkRows, std::int32_t ValueRows, std::int32_t ZRows,
          std::int32_t Hidden, std::int32_t FullSlabs>
struct GdnInputGeometry {
    static constexpr std::int32_t kQkRows     = QkRows;
    static constexpr std::int32_t kValueRows  = ValueRows;
    static constexpr std::int32_t kValueZRows = ValueRows + ZRows;
    static constexpr std::int32_t kHidden     = Hidden;
    static constexpr std::int32_t kFullSlabs  = FullSlabs;
};

using GdnInputGeometry27 =
    GdnInputGeometry<4096, 6144, 6144, 5120, 5>;
using GdnInputGeometry9 =
    GdnInputGeometry<4096, 4096, 4096, 4096, 4>;

using Q4GdnSimtR8C4Schedule = Q4RowSplitSimtGemmSchedule<8, 4, 16, 2, Cache::ca, 1>;
using Q4GdnSimtR8C8Schedule = Q4RowSplitSimtGemmSchedule<8, 8, 16, 2, Cache::ca, 1>;

template <class Geometry>
void launch_q4_gemv(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    using Schedule = std::conditional_t<Geometry::kHidden == 4096,
                                        Q4GemvR1W8DirectK64Schedule, Q4GemvR1W8DirectSchedule>;
    constexpr std::int32_t kQkRows = Geometry::kQkRows;
    constexpr std::int32_t kHidden = Geometry::kHidden;
    const dim3 grid(static_cast<unsigned>(div_up(kQkRows, Schedule::kRowsPerCta)), 1u, 1u);
    constexpr dim3 block(static_cast<unsigned>(Schedule::kThreads), 1u, 1u);
    q4_rowsplit_gemv_kernel<Schedule><<<grid, block, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), static_cast<const std::uint8_t*>(weight.qdata),
        static_cast<const std::uint8_t*>(weight.scales), static_cast<__nv_bfloat16*>(out.data),
        nullptr, kQkRows, kHidden);
    CUDA_CHECK(cudaGetLastError());
}

template <class Geometry, class Schedule, bool Full>
void launch_q4_simt(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    constexpr std::int32_t kQkRows = Geometry::kQkRows;
    constexpr std::int32_t kHidden = Geometry::kHidden;
    const std::int32_t cols   = x.ne[1];
    const std::int32_t out_ld = static_cast<std::int32_t>(out.nb[1] / sizeof(__nv_bfloat16));
    const dim3 grid(static_cast<unsigned>(div_up(kQkRows, Schedule::kRowsPerCta)),
                    static_cast<unsigned>(div_up(cols, Schedule::kColsPerTile)), 1u);
    q4_rowsplit_gemm_simt_kernel<Schedule, Full><<<grid, Schedule::kThreads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), static_cast<const std::uint8_t*>(weight.qdata),
        static_cast<const std::uint8_t*>(weight.scales), static_cast<__nv_bfloat16*>(out.data),
        nullptr, out_ld, 0, kQkRows, kHidden, cols, weight.padded_shape[1]);
    CUDA_CHECK(cudaGetLastError());
}

template <class Geometry, class Schedule>
void launch_q4_simt_route(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    constexpr std::int32_t kQkRows = Geometry::kQkRows;
    constexpr std::int32_t kHidden = Geometry::kHidden;
    const bool full = (kQkRows % Schedule::kRowsPerCta) == 0 &&
                      ((kHidden / Q4RowSplitStorage::kGroupK) % Schedule::kGroupsPerStage) == 0 &&
                      (x.ne[1] % Schedule::kColsPerTile) == 0;
    if (full) {
        launch_q4_simt<Geometry, Schedule, true>(x, weight, out, stream);
    } else {
        launch_q4_simt<Geometry, Schedule, false>(x, weight, out, stream);
    }
}

template <class Geometry>
void launch_q4(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    if (x.ne[1] == 1) {
        launch_q4_gemv<Geometry>(x, weight, out, stream);
        return;
    }
    if (x.ne[1] <= 4) {
        launch_q4_simt_route<Geometry, Q4GdnSimtR8C4Schedule>(x, weight, out, stream);
        return;
    }
    if (x.ne[1] <= 16) {
        launch_q4_simt_route<Geometry, Q4GdnSimtR8C8Schedule>(x, weight, out, stream);
        return;
    }
    throw std::invalid_argument("Q4/Q5 GDN independent launch requires T in [1,15]");
}

template <class Geometry>
void launch_q5_gemv(const Tensor& x, const Weight& weight, Tensor& value, Tensor& z,
                    cudaStream_t stream) {
    constexpr std::int32_t kValueZRows = Geometry::kValueZRows;
    constexpr std::int32_t kValueRows  = Geometry::kValueRows;
    constexpr std::int32_t kHidden     = Geometry::kHidden;
    constexpr int kRowsPerBlock = 16;
    constexpr int kThreads      = kRowsPerBlock * 32;
    q5_rowsplit_gemv_kernel<kValueZRows, kHidden, kRowsPerBlock, 2, true, false, true, kValueRows>
        <<<kValueZRows / kRowsPerBlock, kThreads, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data),
            static_cast<const std::uint8_t*>(weight.qdata),
            static_cast<const std::uint8_t*>(weight.qhigh),
            static_cast<const std::uint8_t*>(weight.scales),
            static_cast<__nv_bfloat16*>(value.data), static_cast<__nv_bfloat16*>(z.data));
    CUDA_CHECK(cudaGetLastError());
}

template <class Geometry, int Cols>
void launch_q5_split4(const Tensor& x, const Weight& weight, Tensor& value, Tensor& z,
                      cudaStream_t stream) {
    constexpr std::int32_t kValueZRows = Geometry::kValueZRows;
    constexpr std::int32_t kValueRows  = Geometry::kValueRows;
    constexpr std::int32_t kHidden     = Geometry::kHidden;
    constexpr std::int32_t kFullSlabs  = Geometry::kFullSlabs;
    constexpr int kThreads    = 4 * 32;
    const std::int32_t out_ld = static_cast<std::int32_t>(value.nb[1] / sizeof(__nv_bfloat16));
    const dim3 grid(static_cast<unsigned>(kValueZRows), 1u, 1u);
    q5_rowsplit_gemm_simt_split4_kernel<Q5RowSplitSimtSchedule, Cols, kFullSlabs, kHidden, true,
                                        kValueRows>
        <<<grid, kThreads, 0, stream>>>(static_cast<const __nv_bfloat16*>(x.data),
                                        static_cast<const std::uint8_t*>(weight.qdata),
                                        static_cast<const std::uint8_t*>(weight.qhigh),
                                        static_cast<const std::uint8_t*>(weight.scales),
                                        static_cast<__nv_bfloat16*>(value.data),
                                        static_cast<__nv_bfloat16*>(z.data), kValueZRows, out_ld,
                                        kHidden, Cols, weight.padded_shape[1], kFullSlabs);
    CUDA_CHECK(cudaGetLastError());
}

template <class Geometry>
void launch_q5_split4_exact(const Tensor& x, const Weight& weight, Tensor& value, Tensor& z,
                            cudaStream_t stream) {
    switch (x.ne[1]) {
    case 2:
        launch_q5_split4<Geometry, 2>(x, weight, value, z, stream);
        return;
    case 3:
        launch_q5_split4<Geometry, 3>(x, weight, value, z, stream);
        return;
    case 4:
        launch_q5_split4<Geometry, 4>(x, weight, value, z, stream);
        return;
    case 5:
        launch_q5_split4<Geometry, 5>(x, weight, value, z, stream);
        return;
    case 6:
        launch_q5_split4<Geometry, 6>(x, weight, value, z, stream);
        return;
    default:
        throw std::invalid_argument("GDN Q5 split4 requires T in [2,6]");
    }
}

template <class Geometry>
void launch_q5_simt_r8_c8(const Tensor& x, const Weight& weight, Tensor& value, Tensor& z,
                          cudaStream_t stream) {
    constexpr std::int32_t kValueZRows = Geometry::kValueZRows;
    constexpr std::int32_t kValueRows  = Geometry::kValueRows;
    constexpr std::int32_t kHidden     = Geometry::kHidden;
    constexpr std::int32_t kFullSlabs  = Geometry::kFullSlabs;
    constexpr int kColsPerTile  = 8;
    constexpr int kRowsPerBlock = 8;
    constexpr int kStages       = 2;
    constexpr int kThreads      = kRowsPerBlock * 32;
    const std::int32_t cols     = x.ne[1];
    const std::int32_t out_ld   = static_cast<std::int32_t>(value.nb[1] / sizeof(__nv_bfloat16));
    const dim3 grid(static_cast<unsigned>(div_up(kValueZRows, kRowsPerBlock)),
                    static_cast<unsigned>(div_up(cols, kColsPerTile)), 1u);
    q5_rowsplit_gemm_simt_kernel<Q5RowSplitSimtSchedule, kColsPerTile, kRowsPerBlock, kStages, true,
                                 kValueRows><<<grid, kThreads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), static_cast<const std::uint8_t*>(weight.qdata),
        static_cast<const std::uint8_t*>(weight.qhigh),
        static_cast<const std::uint8_t*>(weight.scales), static_cast<__nv_bfloat16*>(value.data),
        static_cast<__nv_bfloat16*>(z.data), kValueZRows, out_ld, kHidden, cols,
        weight.padded_shape[1], kFullSlabs);
    CUDA_CHECK(cudaGetLastError());
}

template <class Geometry>
void launch_q5(const Tensor& x, const Weight& weight, Tensor& value, Tensor& z,
               cudaStream_t stream) {
    if (x.ne[1] == 1) {
        launch_q5_gemv<Geometry>(x, weight, value, z, stream);
        return;
    }
    if (x.ne[1] <= 6) {
        launch_q5_split4_exact<Geometry>(x, weight, value, z, stream);
        return;
    }
    if (x.ne[1] <= 16) {
        launch_q5_simt_r8_c8<Geometry>(x, weight, value, z, stream);
        return;
    }
    throw std::invalid_argument("Q4/Q5 GDN independent launch requires T in [1,15]");
}

template <class Geometry>
void launch_t4_pdl(const Tensor& x, const Weight& qk_weight, const Weight& value_z_weight,
                   Tensor& qk, Tensor& value, Tensor& z, cudaStream_t stream) {
    using Q4Schedule         = Q4GdnSimtR8C4Schedule;
    constexpr int kQ5Threads = 4 * 32;
    const dim3 q4_grid(Geometry::kQkRows / Q4Schedule::kRowsPerCta, 1u, 1u);
    const dim3 q5_grid(Geometry::kValueZRows, 1u, 1u);
    const std::int32_t q4_out_ld = static_cast<std::int32_t>(qk.nb[1] / sizeof(__nv_bfloat16));
    const std::int32_t q5_out_ld = static_cast<std::int32_t>(value.nb[1] / sizeof(__nv_bfloat16));

    // Q5 and Q4 publish disjoint row ranges. Q4 can execute while Q5 drains and joins Q5 only at
    // exit, before the following convolution/snapshot kernel becomes runnable.
    q5_rowsplit_gemm_simt_split4_kernel<Q5RowSplitSimtSchedule, 4, Geometry::kFullSlabs,
                                        Geometry::kHidden, true, Geometry::kValueRows,
                                        Q5Split4StoreEpilogue, true, false>
        <<<q5_grid, kQ5Threads, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data),
            static_cast<const std::uint8_t*>(value_z_weight.qdata),
            static_cast<const std::uint8_t*>(value_z_weight.qhigh),
            static_cast<const std::uint8_t*>(value_z_weight.scales),
            static_cast<__nv_bfloat16*>(value.data), static_cast<__nv_bfloat16*>(z.data),
            Geometry::kValueZRows, q5_out_ld, Geometry::kHidden, 4,
            value_z_weight.padded_shape[1], Geometry::kFullSlabs);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(pdl::launch_dependent(
        {q4_grid, dim3(Q4Schedule::kThreads), 0, stream},
        q4_rowsplit_gemm_simt_kernel<Q4Schedule, true, false, 0, Q4SimtStoreEpilogue, false, true>,
        static_cast<const __nv_bfloat16*>(x.data),
        static_cast<const std::uint8_t*>(qk_weight.qdata),
        static_cast<const std::uint8_t*>(qk_weight.scales), static_cast<__nv_bfloat16*>(qk.data),
        nullptr, q4_out_ld, 0, Geometry::kQkRows, Geometry::kHidden, 4,
        qk_weight.padded_shape[1], Q4SimtStoreEpilogue{}));
}

template <class Geometry>
void launch_geometry(const Tensor& x, const Weight& qk_weight, const Weight& value_z_weight,
                     Tensor& qk, Tensor& value, Tensor& z, cudaStream_t stream) {
    if (x.ne[1] == 4) {
        launch_t4_pdl<Geometry>(x, qk_weight, value_z_weight, qk, value, z, stream);
        return;
    }
    launch_q4<Geometry>(x, qk_weight, qk, stream);
    launch_q5<Geometry>(x, value_z_weight, value, z, stream);
}

} // namespace

void q4_q5_gdn_input_independent_launch(const Tensor& x, const Weight& qk_weight,
                                        const Weight& value_z_weight, Tensor& qk, Tensor& value,
                                        Tensor& z, cudaStream_t stream) {
    switch (x.ne[0]) {
    case 5120:
        launch_geometry<GdnInputGeometry27>(x, qk_weight, value_z_weight, qk, value, z, stream);
        return;
    case 4096:
        launch_geometry<GdnInputGeometry9>(x, qk_weight, value_z_weight, qk, value, z, stream);
        return;
    default:
        throw std::invalid_argument("GDN Q4/Q5 independent launch: unsupported input width");
    }
}

} // namespace ninfer::ops::detail
