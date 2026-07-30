#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

#include "world_model_conv.cuh"

namespace {

cudaStream_t stream_for(const torch::Tensor& tensor) {
  return at::cuda::getCurrentCUDAStream(tensor.get_device()).stream();
}

void causal_conv3d(
    torch::Tensor cache_x,
    torch::Tensor new_x,
    torch::Tensor weight,
    torch::Tensor bias,
    double alpha,
    torch::Tensor out) {
  const int co = weight.size(0);
  const int status =
      co % 8 == 0
          ? flash_rt::conv::fp8_conv3d_v17_ndhwc_bf16out(
                cache_x.data_ptr(), new_x.data_ptr(), weight.data_ptr(),
                out.data_ptr(), bias.data_ptr(), new_x.size(0), cache_x.size(1),
                new_x.size(1), new_x.size(2), new_x.size(3), new_x.size(4), co,
                static_cast<float>(alpha), stream_for(new_x))
          : flash_rt::conv::fp8_conv3d_v17_anyco_ndhwc_bf16out(
                cache_x.data_ptr(), new_x.data_ptr(), weight.data_ptr(),
                out.data_ptr(), bias.data_ptr(), new_x.size(0), cache_x.size(1),
                new_x.size(1), new_x.size(2), new_x.size(3), new_x.size(4), co,
                static_cast<float>(alpha), stream_for(new_x));
  TORCH_CHECK(status == 0);
}

void causal_conv3d_residual(
    torch::Tensor cache_x,
    torch::Tensor new_x,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor residual,
    double alpha,
    torch::Tensor out) {
  TORCH_CHECK(
      flash_rt::conv::fp8_conv3d_v18_ncdhw_res_bf16out(
          cache_x.data_ptr(), new_x.data_ptr(), weight.data_ptr(),
          out.data_ptr(), bias.data_ptr(), residual.data_ptr(), new_x.size(0),
          cache_x.size(1), new_x.size(1), new_x.size(2), new_x.size(3),
          new_x.size(4), weight.size(0), static_cast<float>(alpha),
          stream_for(new_x)) == 0);
}

void conv2d(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    double alpha,
    torch::Tensor out) {
  TORCH_CHECK(
      flash_rt::conv::fp8_conv2d_3x3_v2_nhwc_bf16out(
          input.data_ptr(), weight.data_ptr(), out.data_ptr(), bias.data_ptr(),
          input.size(0), input.size(1), input.size(2), input.size(3),
          weight.size(0), static_cast<float>(alpha), stream_for(input)) == 0);
}

void nvfp4_causal_conv3d(
    torch::Tensor cache,
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor cache_sf,
    torch::Tensor input_sf,
    torch::Tensor weight_sf,
    torch::Tensor bias,
    double alpha,
    torch::Tensor out) {
  TORCH_CHECK(
      flash_rt::conv::motus_fp4_conv3d_v19sf_ndhwc_bf16out(
          cache.data_ptr(), input.data_ptr(), weight.data_ptr(),
          cache_sf.data_ptr(), input_sf.data_ptr(), weight_sf.data_ptr(),
          out.data_ptr(), bias.data_ptr(), input.size(0), cache.size(1),
          input.size(1), input.size(2), input.size(3),
          input.size(4) * 2, weight.size(0), static_cast<float>(alpha),
          stream_for(input)) == 0);
}

void nvfp4_causal_conv3d_residual(
    torch::Tensor cache,
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor cache_sf,
    torch::Tensor input_sf,
    torch::Tensor weight_sf,
    torch::Tensor bias,
    torch::Tensor residual,
    double alpha,
    torch::Tensor out) {
  const int ci = input.size(4) * 2;
  const int status = ci % 128 == 0
      ? flash_rt::conv::motus_fp4_conv3d_v19sfbk128_ncdhw_res_bf16out(
            cache.data_ptr(), input.data_ptr(), weight.data_ptr(),
            cache_sf.data_ptr(), input_sf.data_ptr(), weight_sf.data_ptr(),
            out.data_ptr(), bias.data_ptr(), residual.data_ptr(),
            input.size(0), cache.size(1), input.size(1), input.size(2),
            input.size(3), ci, weight.size(0), static_cast<float>(alpha),
            stream_for(input))
      : flash_rt::conv::motus_fp4_conv3d_v19sfb_ncdhw_res_bf16out(
            cache.data_ptr(), input.data_ptr(), weight.data_ptr(),
            cache_sf.data_ptr(), input_sf.data_ptr(), weight_sf.data_ptr(),
            out.data_ptr(), bias.data_ptr(), residual.data_ptr(),
            input.size(0), cache.size(1), input.size(1), input.size(2),
            input.size(3), ci, weight.size(0), static_cast<float>(alpha),
            stream_for(input));
  TORCH_CHECK(status == 0);
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("causal_conv3d", &causal_conv3d);
  module.def("causal_conv3d_residual", &causal_conv3d_residual);
  module.def("conv2d", &conv2d);
  module.def("nvfp4_causal_conv3d", &nvfp4_causal_conv3d);
  module.def(
      "nvfp4_causal_conv3d_residual",
      &nvfp4_causal_conv3d_residual);
}
