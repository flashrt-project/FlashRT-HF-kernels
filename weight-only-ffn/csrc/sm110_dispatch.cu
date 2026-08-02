#include "sm110_dispatch.cuh"
#include "weight_only_ffn.cuh"

namespace flashrt_weight_only::hub {
namespace {

const Sm110Dispatch dispatch{
    &flashrt_weight_only::quantize_w4_weight_bf16,
    &flashrt_weight_only::dequantize_w4_weight_bf16,
    &flashrt_weight_only::w4a16_linear_bf16,
    &flashrt_weight_only::quantize_w8_weight_bf16,
    &flashrt_weight_only::dequantize_w8_weight_bf16,
    &flashrt_weight_only::w8a16_linear_bf16,
    &flashrt_weight_only::gated_activation_bf16,
    &flashrt_weight_only::gelu_activation_bf16,
    &flashrt_weight_only::add_bias_bf16,
};

struct RegisterDispatch {
  RegisterDispatch() { sm110_dispatch = &dispatch; }
};

RegisterDispatch registration;

}  // namespace
}  // namespace flashrt_weight_only::hub
