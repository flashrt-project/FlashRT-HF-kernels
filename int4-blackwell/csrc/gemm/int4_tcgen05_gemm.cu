// ============================================================================
//  FlashRT native E0M3/INT4 GEMM for the SM100 tcgen05 family. CUTLASS owns
//  the block-scaled data movement; the package-local descriptor header maps
//  the 4-bit operands to the hardware-validated E0M3 format value zero.
//
//  Build target guard: only compiled when ENABLE_SM100_CUTLASS=ON
//  (i.e. Thor SM110 or Blackwell SM100+). Arch flag must be -arch=sm_110a
//  or -arch=sm_100a because TCGEN05_MXF4_MMA requires the 'a' suffix.
// ============================================================================

#include "gemm/int4_tcgen05_gemm.cuh"

#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED) || defined(__CUDA_ARCH__)
#  include "cutlass/cutlass.h"
#  include "cutlass/tensor_ref.h"
#  include "cutlass/epilogue/thread/linear_combination.h"
#  include "cutlass/gemm/dispatch_policy.hpp"
#  include "cutlass/gemm/collective/collective_builder.hpp"
#  include "cutlass/epilogue/collective/collective_builder.hpp"
#  include "cutlass/gemm/device/gemm_universal_adapter.h"
#  include "cutlass/gemm/kernel/gemm_universal.hpp"
#  include "cutlass/detail/sm100_blockscaled_layout.hpp"
#  include "cute/tensor.hpp"
#  define FLASHRT_INT4_HAVE_CUTLASS 1
#else
#  define FLASHRT_INT4_HAVE_CUTLASS 0
#endif

namespace flashrt_int4 {

bool has_tcgen05_int4() {
#if FLASHRT_INT4_HAVE_CUTLASS
  return true;
#else
  return false;
#endif
}

#if FLASHRT_INT4_HAVE_CUTLASS

using namespace cute;

// Kernel configuration derived from CUTLASS example 72a.
using ElementA   = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
using LayoutATag = cutlass::layout::RowMajor;
constexpr int AlignmentA = 32;

using ElementB   = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
using LayoutBTag = cutlass::layout::ColumnMajor;
constexpr int AlignmentB = 32;

using ElementD   = cutlass::bfloat16_t;
using ElementC   = cutlass::bfloat16_t;
using LayoutCTag = cutlass::layout::RowMajor;
using LayoutDTag = cutlass::layout::RowMajor;
constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;
constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;

using ElementAccumulator = float;
using ArchTag            = cutlass::arch::Sm100;
using OperatorClass      = cutlass::arch::OpClassBlockScaledTensorOp;

// Tuned for Pi0.5 decoder: small M (SQ=10), wide N (2H=16384 for Gate+Up, or
// H=8192 for Down). MmaTileShape picked to cover small-M case efficiently.
// We may add shape-specific kernel instantiations (e.g. _wide, _sq) later —
// for now single "sq" variant to validate end-to-end numerics.
using MmaTileShape  = Shape<_128, _128, _128>;
using ClusterShape  = Shape<_2, _1, _1>;

using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    MmaTileShape, ClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccumulator, ElementAccumulator,
    ElementC, LayoutCTag, AlignmentC,
    ElementD, LayoutDTag, AlignmentD,
    cutlass::epilogue::collective::EpilogueScheduleAuto
>::CollectiveOp;

using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    ElementA, LayoutATag, AlignmentA,
    ElementB, LayoutBTag, AlignmentB,
    ElementAccumulator,
    MmaTileShape, ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<
        static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
    cutlass::gemm::collective::KernelScheduleAuto
>::CollectiveOp;

using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    Shape<int, int, int, int>,
    CollectiveMainloop,
    CollectiveEpilogue,
    void>;

using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

using StrideA   = typename Gemm::GemmKernel::StrideA;
using StrideB   = typename Gemm::GemmKernel::StrideB;
using StrideC   = typename Gemm::GemmKernel::StrideC;
using StrideD   = typename Gemm::GemmKernel::StrideD;
using LayoutSFA = typename Gemm::GemmKernel::CollectiveMainloop::LayoutSFA;
using LayoutSFB = typename Gemm::GemmKernel::CollectiveMainloop::LayoutSFB;
using Sm1xxBlkScaledConfig =
    typename Gemm::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;

#endif // FLASHRT_INT4_HAVE_CUTLASS

// ──────────────────────────────────────────────────────────────────────────────
int tcgen05_int4_gemm_bf16(
    void const* A_fp4_packed,
    void const* SFA,
    void const* B_fp4_packed,
    void const* SFB,
    void* D_bf16,
    int M, int N, int K,
    float alpha, float beta,
    cudaStream_t stream)
{
#if FLASHRT_INT4_HAVE_CUTLASS
  // Derive strides and layouts for this call.
  StrideA stride_A{K, cute::Int<1>{}, int64_t{0}};
  StrideB stride_B{K, cute::Int<1>{}, int64_t{0}};
  StrideC stride_C{N, cute::Int<1>{}, int64_t{0}};
  StrideD stride_D{N, cute::Int<1>{}, int64_t{0}};

  auto layout_SFA =
      Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(cute::make_shape(M, N, K, 1));
  auto layout_SFB =
      Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(cute::make_shape(M, N, K, 1));

  // The CUTLASS kernel expects typed pointers. A_fp4_packed is a packed
  // byte stream (2 int4 per byte); CUTLASS treats this as an ElementA::DataType
  // array which is `cutlass::float_e2m1_t` stored at 4 bits each. We just
  // reinterpret_cast the device byte pointer.
  using EA_data = typename ElementA::DataType;
  using EA_sf   = typename ElementA::ScaleFactorType;
  using EB_data = typename ElementB::DataType;
  using EB_sf   = typename ElementB::ScaleFactorType;

  typename Gemm::Arguments args{
      cutlass::gemm::GemmUniversalMode::kGemm,
      {M, N, K, 1},
      { // Mainloop
          reinterpret_cast<EA_data const*>(A_fp4_packed), stride_A,
          reinterpret_cast<EB_data const*>(B_fp4_packed), stride_B,
          reinterpret_cast<EA_sf   const*>(SFA),          layout_SFA,
          reinterpret_cast<EB_sf   const*>(SFB),          layout_SFB
      },
      { // Epilogue
          {alpha, beta},
          // For beta=0 we don't read C. The CUTLASS API still requires a ptr,
          // so reuse D for C (unused when beta=0).
          reinterpret_cast<ElementC*>(D_bf16), stride_C,
          reinterpret_cast<ElementD*>(D_bf16), stride_D
      }
  };

  Gemm gemm;

  cutlass::Status st = gemm.can_implement(args);
  if (st != cutlass::Status::kSuccess) {
    return static_cast<int>(st) | 0x10000;
  }

  size_t ws_sz = Gemm::get_workspace_size(args);
  // This validated 128-aligned kernel requires no workspace. Do not introduce
  // a hidden allocation or synchronization into the public Tensor op.
  if (ws_sz != 0) return -1;
  void* ws_ptr = nullptr;

  st = gemm.initialize(args, ws_ptr, stream);
  if (st != cutlass::Status::kSuccess) {
    return static_cast<int>(st) | 0x20000;
  }

  st = gemm.run(stream);
  if (st != cutlass::Status::kSuccess) {
    return static_cast<int>(st) | 0x30000;
  }
  return 0;
#else
  // Build-time: not compiled with CUTLASS SM100 support. Fail loud.
  (void)A_fp4_packed; (void)SFA; (void)B_fp4_packed; (void)SFB;
  (void)D_bf16; (void)M; (void)N; (void)K; (void)alpha; (void)beta; (void)stream;
  return -2;
#endif
}

} // namespace flashrt_int4
