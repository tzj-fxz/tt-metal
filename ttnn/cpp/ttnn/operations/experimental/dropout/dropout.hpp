
// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

namespace ttnn::operations::experimental {

struct DropoutOperation {
    static Tensor invoke(const Tensor& input_tensor, float prob, float scale, uint32_t seed);
};
}  // namespace ttnn::operations::experimental
namespace ttnn::experimental {
constexpr auto dropout = ttnn::register_operation_with_auto_launch_op<
    "ttnn::experimental::dropout",
    ttnn::operations::experimental::DropoutOperation>();
}  // namespace ttnn::experimental