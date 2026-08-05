#include "sm110_fp4_dispatch.cuh"

#include "dit_norm_fp4_sfa.cuh"

namespace flash_rt::adaln_producers::hub {
namespace {

struct Registration {
  Registration() {
    ada_layer_norm_fp4_dispatch =
        &flash_rt::fused_fp4::ada_layer_norm_fp4_sfa_bf16;
    layer_norm_fp4_dispatch =
        &flash_rt::fused_fp4::layer_norm_no_affine_fp4_sfa_bf16;
  }
};

Registration registration;

}  // namespace
}  // namespace flash_rt::adaln_producers::hub
