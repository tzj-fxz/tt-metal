# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
from models.utility_functions import skip_for_grayskull
from tests.ttnn.utils_for_testing import assert_with_pcc
from models.demos.yolov4.ttnn.genboxes import TtGenBoxes
from models.demos.yolov4.demo.demo import YoloLayer, get_region_boxes, gen_yolov4_boxes_confs

import pytest
import os


@skip_for_grayskull()
@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
def test_yolov4_post_processing(device, reset_seeds, model_location_generator):
    torch.manual_seed(0)

    torch_input_1 = torch.randn((1, 1, 1600, 256), dtype=torch.bfloat16)
    ttnn_input_1 = ttnn.from_torch(
        torch_input_1, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.L1_MEMORY_CONFIG
    )
    torch_input_2 = torch.randn((1, 1, 400, 256), dtype=torch.bfloat16)
    ttnn_input_2 = ttnn.from_torch(
        torch_input_2, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.L1_MEMORY_CONFIG
    )
    torch_input_3 = torch.randn((1, 1, 100, 256), dtype=torch.bfloat16)
    ttnn_input_3 = ttnn.from_torch(
        torch_input_3, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.L1_MEMORY_CONFIG
    )

    torch_input_1 = torch_input_1[:, :, :, :255]
    torch_input_1 = torch_input_1.reshape(1, 40, 40, 255)
    torch_input_1 = torch.permute(torch_input_1, (0, 3, 1, 2))
    torch_input_2 = torch_input_2[:, :, :, :255]
    torch_input_2 = torch_input_2.reshape(1, 20, 20, 255)
    torch_input_2 = torch.permute(torch_input_2, (0, 3, 1, 2))
    torch_input_3 = torch_input_3[:, :, :, :255]
    torch_input_3 = torch_input_3.reshape(1, 10, 10, 255)
    torch_input_3 = torch.permute(torch_input_3, (0, 3, 1, 2))

    ref1, ref2, ref3 = gen_yolov4_boxes_confs([torch_input_1, torch_input_2, torch_input_3])

    boxes_confs_1 = TtGenBoxes(device)
    boxes_confs_2 = TtGenBoxes(device)
    boxes_confs_3 = TtGenBoxes(device)

    result_1 = boxes_confs_1(device, ttnn_input_1)
    result_2 = boxes_confs_2(device, ttnn_input_2)
    result_3 = boxes_confs_3(device, ttnn_input_3)

    result_1_bb = ttnn.to_torch(result_1[0])
    result_2_bb = ttnn.to_torch(result_2[0])
    result_3_bb = ttnn.to_torch(result_3[0])

    result_1_bb = result_1_bb.permute(0, 2, 3, 1)
    result_2_bb = result_2_bb.permute(0, 2, 3, 1)
    result_3_bb = result_3_bb.permute(0, 2, 3, 1)

    result_1_bb = result_1_bb.reshape(1, 4800, 1, 4)
    result_2_bb = result_2_bb.reshape(1, 1200, 1, 4)
    result_3_bb = result_3_bb.reshape(1, 300, 1, 4)

    result_1_conf = ttnn.to_torch(result_1[1])
    result_2_conf = ttnn.to_torch(result_2[1])
    result_3_conf = ttnn.to_torch(result_3[1])

    assert_with_pcc(ref1[0], result_1_bb, 0.99)
    assert_with_pcc(ref2[0], result_2_bb, 0.99)
    assert_with_pcc(ref3[0], result_3_bb, 0.99)

    assert_with_pcc(ref1[1], result_1_conf, 0.99)
    assert_with_pcc(ref2[1], result_2_conf, 0.99)
    assert_with_pcc(ref3[1], result_3_conf, 0.99)

    output = get_region_boxes(
        [(result_1_bb, result_1_conf), (result_2_bb, result_2_conf), (result_3_bb, result_3_conf)]
    )
