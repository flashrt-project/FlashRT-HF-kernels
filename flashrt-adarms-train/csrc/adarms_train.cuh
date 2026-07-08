// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cuda_runtime.h>

namespace flashrt_hub {
namespace adarms_train {

// Row-parallel AdaRMS training kernels. All launchers operate on flattened
// rows = B*T with hidden size H (contiguous last dim), fp32 internal math.
// scale/shift are addressed as base + b*batch_stride + t*token_stride so
// chunked modulation views ((B, 3H) or (B, T, 3H)) are consumed zero-copy;
// token_stride == 0 expresses per-sample broadcast.

struct ModView {
  const void* ptr;        // nullptr => non-adaptive (use weight)
  long batch_stride;      // elements
  long token_stride;      // elements (0 = broadcast over T)
};

// Adaptive backward writes per-row (rows, H) dscale elements (the caller
// reduces broadcasts). Weight-mode backward writes per-CTA fp32 partial
// weight-grad rows of shape (bwd_weight_grid(rows), H) into the same output
// slot; the caller sums dim 0.
inline int bwd_weight_grid(int rows) { return rows < 1536 ? rows : 1536; }

void adarms_fwd_launch(const void* x, ModView scale, ModView shift,
                       const void* weight, void* y, float* rstd, int rows,
                       int tokens, int h, float eps, bool bf16,
                       cudaStream_t stream);

void adarms_bwd_launch(const void* dy, const void* x, ModView scale,
                       const void* weight, const float* rstd, void* dx,
                       void* dscale_elem,  // (rows, H) io dtype
                       int rows, int tokens, int h, bool bf16,
                       cudaStream_t stream);

void resgate_adarms_fwd_launch(const void* x, const void* hbr, const void* gate,
                               ModView scale, ModView shift, const void* weight,
                               void* r, void* y, float* rstd, int rows,
                               int tokens, int h, float eps, bool bf16,
                               cudaStream_t stream);

void resgate_adarms_bwd_launch(const void* dy, const void* dyr,  // dyr may be nullptr
                               const void* r, const void* hbr, const void* gate,
                               ModView scale, const void* weight,
                               const float* rstd, void* dr_total, void* dh,
                               void* dg,  // nullptr if gate == nullptr
                               void* dscale_elem, int rows,
                               int tokens, int h, bool bf16,
                               cudaStream_t stream);

}  // namespace adarms_train
}  // namespace flashrt_hub
