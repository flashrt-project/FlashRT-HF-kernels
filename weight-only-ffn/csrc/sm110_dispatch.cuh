#pragma once

#include <cuda_runtime.h>

namespace flashrt_weight_only::hub {

struct Sm110Dispatch {
  int (*quantize_w4)(const void*, void*, void*, int, int, cudaStream_t);
  int (*dequantize_w4)(const void*, const void*, void*, int, int, cudaStream_t);
  int (*linear_w4)(const void*, const void*, const void*, void*, int, int, int,
                   float, int, cudaStream_t);
  int (*quantize_w8)(const void*, void*, void*, int, int, cudaStream_t);
  int (*dequantize_w8)(const void*, const void*, void*, int, int, cudaStream_t);
  int (*linear_w8)(const void*, const void*, const void*, void*, int, int, int,
                   int, cudaStream_t);
  void (*gated_activation)(const void*, const void*, void*, int, int, bool,
                           cudaStream_t);
  void (*gelu_activation)(const void*, const void*, void*, int, int,
                          cudaStream_t);
  void (*add_bias)(void*, const void*, int, int, cudaStream_t);
};

extern const Sm110Dispatch* sm110_dispatch;

}  // namespace flashrt_weight_only::hub
