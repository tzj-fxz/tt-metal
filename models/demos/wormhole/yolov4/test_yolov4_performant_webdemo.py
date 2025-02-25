# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import ttnn
import time
import torch

from models.utility_functions import run_for_wormhole_b0
from models.demos.yolov4.tests.yolov4_perfomant_webdemo import Yolov4Trace2CQ


@run_for_wormhole_b0()
@pytest.mark.parametrize(
    "device_params", [{"l1_small_size": 24576, "trace_region_size": 3211264, "num_command_queues": 2}], indirect=True
)
@pytest.mark.parametrize(
    "batch_size, act_dtype, weight_dtype",
    ((1, ttnn.bfloat16, ttnn.bfloat16),),
)
@pytest.mark.parametrize("enable_async_mode", (False, True), indirect=True)
def test_run_yolov4_trace_2cqs_inference(
    device,
    use_program_cache,
    batch_size,
    act_dtype,
    weight_dtype,
    enable_async_mode,
    model_location_generator,
):
    yolov4_trac2_2cq = Yolov4Trace2CQ()

    yolov4_trac2_2cq.initialize_yolov4_trace_2cqs_inference(
        device,
        batch_size,
        act_dtype,
        weight_dtype,
        model_location_generator=None,
    )
    for iter in range(0, 10):
        input_shape = (1, 3, 320, 320)
        torch_input_tensor = torch.randn(input_shape, dtype=torch.float32)
        n, c, h, w = torch_input_tensor.shape
        torch_input_tensor = torch_input_tensor.permute(0, 2, 3, 1)
        torch_input_tensor = torch_input_tensor.reshape(1, 1, h * w * n, c)
        tt_inputs_host = ttnn.from_torch(torch_input_tensor, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
        tt_inputs_host = ttnn.pad(tt_inputs_host, [1, 1, n * h * w, 16], [0, 0, 0, 0], 0)

        t0 = time.time()
        output = yolov4_trac2_2cq.execute_yolov4_trace_2cqs_inference(tt_inputs_host)
        t1 = time.time()
        print("TIME", t1 - t0)

    yolov4_trac2_2cq.release_yolov4_trace_2cqs_inference()
