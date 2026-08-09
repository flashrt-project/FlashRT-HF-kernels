// SPDX-License-Identifier: Apache-2.0
// Native NVFP4 GEMM variants with BF16 output for SM100-family devices.

#include "gemm/fp4/cutlass_fp4_gemm_bf16_variants_sm100.cuh"

#include "cutlass/cutlass.h"
#include "cutlass/detail/sm100_blockscaled_layout.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/util/packed_stride.hpp"
#include "cute/tensor.hpp"

namespace flash_rt::fp4 {
namespace {

using namespace cute;

template <class MmaTile>
struct Bf16Variant {
  using ElementA = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
  using ElementB = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
  using ElementC = cutlass::bfloat16_t;
  using ElementD = cutlass::bfloat16_t;
  using Accumulator = float;
  using Arch = cutlass::arch::Sm100;
  using OpClass = cutlass::arch::OpClassBlockScaledTensorOp;
  using Cluster = Shape<_1, _1, _1>;

  static constexpr int AlignmentA = 32;
  static constexpr int AlignmentB = 32;
  static constexpr int AlignmentC = 8;
  static constexpr int AlignmentD = 8;

  using Epilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      Arch, OpClass, MmaTile, Cluster,
      cutlass::epilogue::collective::EpilogueTileAuto,
      Accumulator, Accumulator,
      ElementC, cutlass::layout::RowMajor, AlignmentC,
      ElementD, cutlass::layout::RowMajor, AlignmentD,
      cutlass::epilogue::collective::EpilogueScheduleAuto>::CollectiveOp;

  using Mainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      Arch, OpClass,
      ElementA, cutlass::layout::RowMajor, AlignmentA,
      ElementB, cutlass::layout::ColumnMajor, AlignmentB,
      Accumulator, MmaTile, Cluster,
      cutlass::gemm::collective::StageCountAutoCarveout<
          static_cast<int>(sizeof(typename Epilogue::SharedStorage))>,
      cutlass::gemm::collective::KernelScheduleAuto>::CollectiveOp;

  using Kernel = cutlass::gemm::kernel::GemmUniversal<
      Shape<int, int, int, int>, Mainloop, Epilogue, void>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<Kernel>;
  using StrideA = typename Kernel::StrideA;
  using StrideB = typename Kernel::StrideB;
  using StrideC = typename Kernel::StrideC;
  using StrideD = typename Kernel::StrideD;
  using ScaleConfig = typename Mainloop::Sm1xxBlkScaledConfig;

  static int run(
      const void* a_packed, const void* sfa, const void* b_packed,
      const void* sfb, void* out, int m, int n, int k, float alpha,
      float beta, cudaStream_t stream) {
    auto stride_a = cutlass::make_cute_packed_stride(StrideA{}, {m, k, 1});
    auto stride_b = cutlass::make_cute_packed_stride(StrideB{}, {n, k, 1});
    auto stride_c = cutlass::make_cute_packed_stride(StrideC{}, {m, n, 1});
    auto stride_d = cutlass::make_cute_packed_stride(StrideD{}, {m, n, 1});
    auto layout_sfa = ScaleConfig::tile_atom_to_shape_SFA(
        make_shape(m, n, k, 1));
    auto layout_sfb = ScaleConfig::tile_atom_to_shape_SFB(
        make_shape(m, n, k, 1));

    using DataA = typename ElementA::DataType;
    using ScaleA = typename ElementA::ScaleFactorType;
    using DataB = typename ElementB::DataType;
    using ScaleB = typename ElementB::ScaleFactorType;
    typename Gemm::Arguments args{
        cutlass::gemm::GemmUniversalMode::kGemm, {m, n, k, 1},
        {reinterpret_cast<const DataA*>(a_packed), stride_a,
         reinterpret_cast<const DataB*>(b_packed), stride_b,
         reinterpret_cast<const ScaleA*>(sfa), layout_sfa,
         reinterpret_cast<const ScaleB*>(sfb), layout_sfb},
        {{alpha, beta}, reinterpret_cast<ElementC*>(out), stride_c,
         reinterpret_cast<ElementD*>(out), stride_d}};

    Gemm gemm;
    auto status = gemm.can_implement(args);
    if (status != cutlass::Status::kSuccess) {
      return static_cast<int>(status) | 0x10000;
    }
    const size_t workspace_size = Gemm::get_workspace_size(args);
    // These persistent PI0.5 paths must remain CUDA Graph capture-safe. The
    // selected kernels require no workspace; reject an unexpected CUTLASS
    // configuration instead of allocating from the runtime hot path.
    if (workspace_size != 0) return -1;
    status = gemm.initialize(args, nullptr, stream);
    if (status == cutlass::Status::kSuccess) status = gemm.run(stream);
    return status == cutlass::Status::kSuccess
        ? 0 : (static_cast<int>(status) | 0x20000);
  }
};

using Variant7 = Bf16Variant<Shape<_128, _128, _256>>;
using Variant10 = Bf16Variant<Shape<_128, _64, _256>>;

}  // namespace

int cutlass_fp4_gemm_bf16_variant(
    int variant, const void* a_packed, const void* sfa,
    const void* b_packed, const void* sfb, void* out_bf16,
    int m, int n, int k, float alpha, float beta, cudaStream_t stream) {
  switch (variant) {
    case 7:
      return Variant7::run(a_packed, sfa, b_packed, sfb, out_bf16,
                           m, n, k, alpha, beta, stream);
    case 10:
      return Variant10::run(a_packed, sfa, b_packed, sfb, out_bf16,
                            m, n, k, alpha, beta, stream);
    default:
      return -99;
  }
}

}  // namespace flash_rt::fp4
