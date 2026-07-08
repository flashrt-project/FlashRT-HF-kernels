// SPDX-License-Identifier: Apache-2.0

#include <torch/all.h>
#include <torch/library.h>

#include "registration.h"
#include "torch_binding.h"

torch::Tensor flashrt_training_package_marker(torch::Tensor x) { return x; }

TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, ops) {
  ops.def("_flashrt_training_package_marker(Tensor x) -> Tensor");
  ops.impl("_flashrt_training_package_marker", c10::DispatchKey::CompositeExplicitAutograd, &flashrt_training_package_marker);
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)
