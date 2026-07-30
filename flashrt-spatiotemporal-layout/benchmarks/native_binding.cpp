#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

#include "bf16_ndhwc_to_ncdhw_transpose.cuh"
#include "bf16_quant_fp8_ncdhw_to_ndhwc.cuh"
#include "spatiotemporal_layout.cuh"

namespace {

cudaStream_t stream_for(const torch::Tensor& tensor) {
  return at::cuda::getCurrentCUDAStream(tensor.get_device()).stream();
}

void ndhwc_to_ncdhw(torch::Tensor x, torch::Tensor out) {
  TORCH_CHECK(
      flash_rt::quantize::bf16_ndhwc_to_ncdhw_transpose(
          x.data_ptr(), out.data_ptr(), x.size(0), x.size(4), x.size(1),
          x.size(2), x.size(3), stream_for(x)) == 0);
}

void ndhwc_to_ncdhw_bias(
    torch::Tensor x, torch::Tensor bias, torch::Tensor out) {
  TORCH_CHECK(
      flash_rt::quantize::bf16_ndhwc_to_ncdhw_bias_bf16(
          x.data_ptr(), bias.data_ptr(), out.data_ptr(), x.size(0), x.size(4),
          x.size(1), x.size(2), x.size(3), stream_for(x)) == 0);
}

void ndhwc_to_ncdhw_add(
    torch::Tensor x, torch::Tensor residual, torch::Tensor out) {
  const auto rs = residual.strides();
  TORCH_CHECK(
      flash_rt::quantize::bf16_ndhwc_to_ncdhw_add_bf16(
          x.data_ptr(), residual.data_ptr(), out.data_ptr(), x.size(0),
          x.size(4), x.size(1), x.size(2), x.size(3), rs[0], rs[1], rs[2],
          rs[3], rs[4], stream_for(x)) == 0);
}

void ncdhw_quantize(torch::Tensor x, double scale, torch::Tensor out) {
  TORCH_CHECK(
      flash_rt::quantize::bf16_quant_fp8_ncdhw_to_ndhwc(
          x.data_ptr(), out.data_ptr(), x.size(0), x.size(1), x.size(2),
          x.size(3), x.size(4), static_cast<float>(scale), stream_for(x)) == 0);
}

void upsample2x_quantize(torch::Tensor x, double scale, torch::Tensor out) {
  TORCH_CHECK(
      flash_rt::quantize::bf16_upsample2x_quant_fp8_nchw_to_nhwc(
          x.data_ptr(), out.data_ptr(), x.size(0), x.size(1), x.size(2),
          x.size(3), static_cast<float>(scale), stream_for(x)) == 0);
}

void ncdhw_to_blc(torch::Tensor x, torch::Tensor out) {
  flash_rt::spatiotemporal_layout::ncdhw_to_blc_bf16(
      x.data_ptr(), out.data_ptr(), x.size(0), x.size(1), x.size(2),
      x.size(3), x.size(4), stream_for(x));
}

void time_unshuffle2(torch::Tensor x, torch::Tensor out) {
  flash_rt::spatiotemporal_layout::time_unshuffle2_bf16(
      x.data_ptr(), out.data_ptr(), x.size(0), x.size(1) / 2, x.size(2),
      x.size(3), x.size(4), stream_for(x));
}

void add_bias(torch::Tensor out, torch::Tensor bias) {
  flash_rt::spatiotemporal_layout::add_bias_ncdhw_bf16(
      out.data_ptr(), bias.data_ptr(), out.size(0), out.size(1), out.size(2),
      out.size(3), out.size(4), stream_for(out));
}

void update_cache2(
    torch::Tensor cur, torch::Tensor prev, torch::Tensor out) {
  flash_rt::spatiotemporal_layout::update_cache2_ncdhw_bf16(
      cur.data_ptr(), prev.data_ptr(), out.data_ptr(), cur.size(0),
      cur.size(1), cur.size(2), cur.size(3), cur.size(4), stream_for(cur));
}

void channel_to_space3d(
    torch::Tensor x,
    int64_t out_channels,
    int64_t temporal_factor,
    int64_t spatial_factor,
    int64_t repeats,
    bool first_chunk,
    torch::Tensor out) {
  flash_rt::spatiotemporal_layout::channel_to_space3d_bf16(
      x.data_ptr(), out.data_ptr(), x.size(0), x.size(1), out_channels,
      x.size(2), x.size(3), x.size(4), temporal_factor, spatial_factor,
      repeats, first_chunk, stream_for(x));
}

void pack_causal_cache3_nhwc(
    torch::Tensor previous, torch::Tensor current, torch::Tensor out) {
  flash_rt::spatiotemporal_layout::pack_causal_cache3_nhwc_bf16(
      previous.data_ptr(), current.data_ptr(), out.data_ptr(),
      current.size(0), current.size(1), current.size(3), current.size(4),
      stream_for(current));
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("ndhwc_to_ncdhw", &ndhwc_to_ncdhw);
  module.def("ndhwc_to_ncdhw_bias", &ndhwc_to_ncdhw_bias);
  module.def("ndhwc_to_ncdhw_add", &ndhwc_to_ncdhw_add);
  module.def("ncdhw_quantize", &ncdhw_quantize);
  module.def("upsample2x_quantize", &upsample2x_quantize);
  module.def("ncdhw_to_blc", &ncdhw_to_blc);
  module.def("time_unshuffle2", &time_unshuffle2);
  module.def("add_bias", &add_bias);
  module.def("update_cache2", &update_cache2);
  module.def("channel_to_space3d", &channel_to_space3d);
  module.def("pack_causal_cache3_nhwc", &pack_causal_cache3_nhwc);
}
