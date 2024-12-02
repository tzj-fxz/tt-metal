# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import time

from loguru import logger
import csv
import pytest
import torch
import ttnn
from models.utility_functions import run_for_wormhole_b0, is_grayskull, profiler
from tests.ttnn.utils_for_testing import assert_with_pcc
from pathlib import Path
import os


SUBBLOCK_HW_CHOICES = [
    (4, 2),
    (2, 4),
    (8, 1),
    (1, 8),  # subblock_hw = 8
    (7, 1),
    (1, 7),  # subblock_hw = 7
    (3, 2),
    (2, 3),
    (6, 1),
    (1, 6),  # subblock_hw = 6
    (5, 1),
    (1, 5),  # subblock_hw = 5
    (2, 2),
    (4, 1),
    (1, 4),  # subblock_hw = 4
    (3, 1),
    (1, 3),  # subblock_hw = 3
    (2, 1),
    (1, 2),  # subblock_hw = 2
    (1, 1),  # subblock_hw = 1
]


def get_subblock_sizes(m_tiles_per_core, n_tiles_per_core, out_sharded=False, fp32_dest_acc_en=False):
    for subblock_hw in SUBBLOCK_HW_CHOICES:
        out_subblock_h = subblock_hw[0]
        out_subblock_w = subblock_hw[1]

        if fp32_dest_acc_en:
            if (out_subblock_h * out_subblock_w) > 4:
                continue

        if out_sharded:
            if n_tiles_per_core % out_subblock_w != 0 or out_subblock_h != 1:
                continue

        if m_tiles_per_core % out_subblock_h == 0 and n_tiles_per_core % out_subblock_w == 0:
            return (out_subblock_h, out_subblock_w)

    return (1, 1)


# This test runs different shapes for matmul_2d, with possibly the best configurations for performance.
#
# The inputs include:
#   - m, k, n: Dimensions of the input tensors.
#   - in0_sharded, out_sharded: Flags indicating whether the in0 (activation) and output tensors are sharded or not.
#   - in0_block_w_div: A parameter to divide an in0 block into multiple chunks, helping to reduce L1 cache usage.
#   - num_out_blocks_h: A parameter to divide an output block into multiple chunks on height dim, helping to reduce L1 cache usage.
#   - num_out_blocks_w: A parameter to divide an output block into multiple chunks on width dim, helping to reduce L1 cache usage.

from tt_metal.tools.profiler.process_device_log import import_log_run_stats
import tt_metal.tools.profiler.device_post_proc_config as device_post_proc_config
from tt_metal.tools.profiler.common import PROFILER_LOGS_DIR, PROFILER_DEVICE_SIDE_LOG

profiler_log_path = PROFILER_LOGS_DIR / PROFILER_DEVICE_SIDE_LOG


def get_device_freq():
    setup = device_post_proc_config.default_setup()
    setup.deviceInputLog = profiler_log_path
    deviceData = import_log_run_stats(setup)
    freq = deviceData["deviceInfo"]["freq"]
    return freq


# m, k, n, strategy, in0_sharded, out_sharded, in0_block_w_div, num_out_blocks_h, num_out_blocks_w 
matmul_shapes_bfloat16 = [
    (512, 512, 512, True, True, 1, 1, 1),
    (512, 1024, 1024, True, True, 1, 1, 1),
    (512, 1024, 2048, True, True, 1, 1, 1),
    (1024, 1024, 1024, True, True, 1, 1, 1),
    (1024, 1024, 2048, True, True, 1, 1, 1),
    (1024, 2048, 2048, True, True, 1, 1, 1),
    (2048, 2048, 2048, True, True, 1, 1, 1),
    (2048, 2048, 3072, True, True, 1, 1, 1),
    (2048, 3072, 3072, True, True, 2, 1, 1),
    (3072, 3072, 3072, True, True, 4, 1, 1),
    (3072, 3072, 4096, False, False, 2, 1, 1),
    (3072, 4096, 4096, False, False, 2, 1, 1),
    (4096, 4096, 4096, False, False, 1, 2, 2),
    (8192, 8192, 8192, False, False, 2, 4, 4),
    (16384, 16384, 16384, False, False, 4, 8, 8),
]

# m, k, n, strategy, in0_sharded, out_sharded, in0_block_w_div, num_out_blocks_h, num_out_blocks_w 
matmul_shapes_bfloat16_square = [
    (512, 512, 512, "NonOptimized", True, True, 1, 1, 1),
    (1024, 1024, 1024, "NonOptimized", True, True, 1, 1, 1),
    (2048, 2048, 2048, "NonOptimized", True, True, 1, 1, 1),
    (4096, 4096, 4096, "NonOptimized", False, False, 1, 2, 2),
    (8192, 8192, 8192, "NonOptimized", False, False, 2, 4, 4),
    (16384, 16384, 16384, "NonOptimized", False, False, 4, 8, 8),
    (512, 512, 512, "Reuse", True, True, 1, 1, 1),
    (1024, 1024, 1024, "Reuse", True, True, 1, 1, 1),
    (2048, 2048, 2048, "Reuse", True, True, 1, 1, 1),
    (4096, 4096, 4096, "Reuse", False, False, 1, 2, 2),
    (8192, 8192, 8192, "Reuse", False, False, 2, 4, 4),
    (16384, 16384, 16384, "Reuse", False, False, 4, 8, 8),
    (512, 512, 512, "ReuseMultiCast", True, True, 1, 1, 1),
    (1024, 1024, 1024, "ReuseMultiCast", True, True, 1, 1, 1),
    (2048, 2048, 2048, "ReuseMultiCast", True, True, 1, 1, 1),
    (4096, 4096, 4096, "ReuseMultiCast", False, False, 1, 2, 2),
    (8192, 8192, 8192, "ReuseMultiCast", False, False, 2, 4, 4),
    (16384, 16384, 16384, "ReuseMultiCast", False, False, 4, 8, 8),
]

matmul_shapes_bfloat16_cannon = [
    (512, 512, 512, "Cannon", True, True, 1, 1, 1),
    (1024, 1024, 1024, "Cannon", True, True, 1, 1, 1),
    (2048, 2048, 2048, "Cannon", True, True, 1, 1, 1),
    (4096, 4096, 4096, "Cannon", False, False, 1, 2, 2),
    (8192, 8192, 8192, "Cannon", False, False, 2, 4, 4),
    (16384, 16384, 16384, "Cannon", False, False, 4, 8, 8),
]

matmul_shapes_bfloat8_b = [
    (512, 512, 512, True, True, 1, 1, 1),
    (512, 1024, 1024, True, True, 1, 1, 1),
    (512, 1024, 2048, True, True, 1, 1, 1),
    (1024, 1024, 1024, True, True, 1, 1, 1),
    (1024, 1024, 2048, True, True, 1, 1, 1),
    (1024, 2048, 2048, True, True, 1, 1, 1),
    (2048, 2048, 2048, True, True, 1, 1, 1),
    (2048, 2048, 3072, True, True, 1, 1, 1),
    (2048, 3072, 3072, True, True, 1, 1, 1),
    (3072, 3072, 3072, True, True, 2, 1, 1),
    (3072, 3072, 4096, True, True, 2, 1, 1),
    (3072, 4096, 4096, True, True, 1, 2, 2),
    (4096, 4096, 4096, False, False, 1, 2, 2),
    (8192, 8192, 8192, False, False, 2, 4, 4),
    (16384, 16384, 16384, False, False, 4, 8, 8),
]

matmul_shapes_bfloat4_b = [
    (512, 512, 512, True, True, 1, 1, 1),
    (512, 1024, 1024, True, True, 1, 1, 1),
    (512, 1024, 2048, True, True, 1, 1, 1),
    (1024, 1024, 1024, True, True, 1, 1, 1),
    (1024, 1024, 2048, True, True, 1, 1, 1),
    (1024, 2048, 2048, True, True, 1, 1, 1),
    (2048, 2048, 2048, True, True, 1, 1, 1),
    (2048, 2048, 3072, True, True, 1, 1, 1),
    (2048, 3072, 3072, True, True, 1, 1, 1),
    (3072, 3072, 3072, True, True, 1, 1, 1),
    (3072, 3072, 4096, True, True, 1, 1, 1),
    (3072, 4096, 4096, True, True, 2, 1, 1),
    (4096, 4096, 4096, True, True, 2, 1, 1),
    (8192, 8192, 8192, False, False, 2, 2, 2),
    (16384, 16384, 16384, False, False, 4, 4, 4),
]

# dtype, math_fidelity, use_trace
matmul_configs_square = [
    (ttnn.bfloat16, ttnn.MathFidelity.HiFi2, False),
    (ttnn.bfloat16, ttnn.MathFidelity.HiFi4, False),
    (ttnn.bfloat16, ttnn.MathFidelity.HiFi2, True),
    (ttnn.bfloat16, ttnn.MathFidelity.HiFi4, True),
]

matmul_configs = [
    (ttnn.bfloat16, ttnn.MathFidelity.HiFi2, False),
    (ttnn.bfloat16, ttnn.MathFidelity.HiFi4, False),
    (ttnn.bfloat8_b, ttnn.MathFidelity.HiFi2, False),
    (ttnn.bfloat8_b, ttnn.MathFidelity.LoFi, False),
    (ttnn.bfloat4_b, ttnn.MathFidelity.LoFi, False),
    (ttnn.bfloat16, ttnn.MathFidelity.HiFi2, True),
    (ttnn.bfloat16, ttnn.MathFidelity.HiFi4, True),
    (ttnn.bfloat8_b, ttnn.MathFidelity.HiFi2, True),
    (ttnn.bfloat8_b, ttnn.MathFidelity.LoFi, True),
    (ttnn.bfloat4_b, ttnn.MathFidelity.LoFi, True),
]


# @pytest.mark.skip(reason="WH didt hang, need to skip CI and run locally only")
@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576, "trace_region_size": 3855488}], indirect=True)
@pytest.mark.parametrize("grid_size", [(8, 8)])
@pytest.mark.parametrize("tile_h", [32])
@pytest.mark.parametrize("tile_w", [32])
@pytest.mark.parametrize("num_warmup_iterations", [5])
@pytest.mark.parametrize("num_measurement_iterations", [100])
def test_matmul_2d_host_perf(
    device,
    grid_size,
    tile_h,
    tile_w,
    num_warmup_iterations,
    num_measurement_iterations,
    use_program_cache,
):
    ENVS = dict(os.environ)
    TT_METAL_HOME = Path(ENVS["TT_METAL_HOME"])
    ARTIFACTS_DIR = TT_METAL_HOME / "generated"
    FILE_NAME = ARTIFACTS_DIR / "matmul_2d_host_perf_report.csv"

    LoFi_cycle = 16
    HiFi2_cycle = LoFi_cycle * 2
    HiFi3_cycle = LoFi_cycle * 3
    HiFi4_cycle = LoFi_cycle * 4

    with open(FILE_NAME, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "m",
                "k",
                "n",
                "use_trace",
                "grid_size",
                "in0_sharded",
                "out_sharded",
                "in0_storage_type",
                "in1_storage_type",
                "out_storage_type",
                "dtype",
                "math_fidelity",
                "inference_time_avg (ns)",
                "TFLOPs (avg)",
                "Utilization (vs user grid)",
                "Utilization (vs 8x8 full grid)",
            ]
        )

        for dtype, math_fidelity, use_trace in matmul_configs_square:
            if dtype == ttnn.bfloat16:
                matmul_shapes = matmul_shapes_bfloat16_square + matmul_shapes_bfloat16_cannon
            else:
                raise NotImplementedError
            # elif dtype == ttnn.bfloat8_b:
            #     matmul_shapes = matmul_shapes_bfloat8_b
            # elif dtype == ttnn.bfloat4_b:
            #     matmul_shapes = matmul_shapes_bfloat4_b
            for m, k, n, strategy, in0_sharded, out_sharded, in0_block_w_div, num_out_blocks_h, num_out_blocks_w in matmul_shapes:
                profiler.clear()

                in0_shape = [1, 1, m, k]
                in1_shape = [1, 1, k, n]

                in0_block_w = k // grid_size[0] // 32 // in0_block_w_div
                per_core_M = m // grid_size[1] // tile_h
                per_core_N = n // grid_size[0] // tile_w
                out_block_h = per_core_M // num_out_blocks_h
                out_block_w = per_core_N // num_out_blocks_w
                out_subblock_h, out_subblock_w = get_subblock_sizes(out_block_h, out_block_w, out_sharded)

                logger.info(f"M*K*N = {m}*{k}*{n} out_subblock_h: {out_subblock_h}, out_subblock_w: {out_subblock_w}")

                in0 = torch.ones(in0_shape).bfloat16()
                in1 = torch.randn(in1_shape).bfloat16()

                if in0_sharded:
                    in0_storage_type = "L1"
                else:
                    in0_storage_type = "DRAM"
                in1_storage_type = "DRAM"
                if out_sharded:
                    out_storage_type = "L1"
                else:
                    out_storage_type = "DRAM"

                if in0_sharded:
                    in0_memory_config = ttnn.create_sharded_memory_config(
                        (1, 1, m, k),
                        core_grid=ttnn.CoreGrid(y=grid_size[1], x=grid_size[0]),
                        strategy=ttnn.ShardStrategy.BLOCK,
                        orientation=ttnn.ShardOrientation.ROW_MAJOR,
                    )
                else:
                    in0_memory_config = ttnn.DRAM_MEMORY_CONFIG
                in0_t = ttnn.from_torch(
                    in0,
                    tile=ttnn.Tile((tile_h, 32)),
                    dtype=dtype,
                    layout=ttnn.TILE_LAYOUT,
                    device=device,
                    memory_config=in0_memory_config,
                )
                in1_t = ttnn.from_torch(
                    in1,
                    tile=ttnn.Tile((32, tile_w)),
                    dtype=dtype,
                    layout=ttnn.TILE_LAYOUT,
                    device=device,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
                
                if strategy == "NonOptimized":
                    program_config = ttnn.MatmulMultiCoreReuseProgramConfig()
                elif strategy == "Reuse":
                    program_config = ttnn.MatmulMultiCoreReuseProgramConfig(
                        compute_with_storage_grid_size=grid_size,
                        in0_block_w=in0_block_w,
                        out_subblock_h=out_subblock_h,
                        out_subblock_w=out_subblock_w,
                        per_core_M=per_core_M,
                        per_core_N=per_core_N,
                    )
                elif strategy == "ReuseMultiCast":
                    program_config = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                        compute_with_storage_grid_size=grid_size,
                        in0_block_w=in0_block_w,
                        out_subblock_h=out_subblock_h,
                        out_subblock_w=out_subblock_w,
                        out_block_h=out_block_h,
                        out_block_w=out_block_w,
                        per_core_M=per_core_M,
                        per_core_N=per_core_N,
                        transpose_mcast=False,
                        fused_activation=None,
                    )
                elif strategy == "Cannon":
                    program_config = ttnn.MatmulMultiCoreReuseCannonProgramConfig(
                        compute_with_storage_grid_size=grid_size,
                        out_subblock_h=out_subblock_h,
                        out_subblock_w=out_subblock_w,
                        per_core_M=per_core_M,
                        per_core_N=per_core_N,
                        # TODO: per_core_K is specified in cannon
                        per_core_K=per_core_M,
                    )

                if is_grayskull():
                    compute_kernel_config = ttnn.GrayskullComputeKernelConfig(
                        math_fidelity=math_fidelity,
                        math_approx_mode=True,
                    )
                else:
                    compute_kernel_config = ttnn.WormholeComputeKernelConfig(
                        math_fidelity=math_fidelity,
                        math_approx_mode=True,
                        fp32_dest_acc_en=False,
                        packer_l1_acc=True,
                    )
                if out_sharded:
                    out_mem_config = ttnn.MemoryConfig(
                        memory_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
                        buffer_type=ttnn.BufferType.L1,
                    )
                else:
                    out_mem_config = ttnn.DRAM_MEMORY_CONFIG
                if out_sharded:
                    output_tile = ttnn.Tile([tile_h, 32]) if tile_h <= 16 else ttnn.Tile([tile_h, tile_w])
                else:
                    output_tile = ttnn.Tile([tile_h, tile_w])

                output_t = ttnn.matmul(
                    in0_t,
                    in1_t,
                    program_config=program_config,
                    memory_config=out_mem_config,
                    dtype=dtype,
                    compute_kernel_config=compute_kernel_config,
                    output_tile=output_tile,
                )

                for iter in range(0, num_warmup_iterations):
                    output_t = ttnn.matmul(
                        in0_t,
                        in1_t,
                        program_config=program_config,
                        memory_config=out_mem_config,
                        dtype=dtype,
                        compute_kernel_config=compute_kernel_config,
                        output_tile=output_tile,
                    )

                if use_trace:
                    tid = ttnn.begin_trace_capture(device, cq_id=0)
                    for iter in range(0, num_measurement_iterations):
                        output_t = ttnn.matmul(
                            in0_t,
                            in1_t,
                            program_config=program_config,
                            memory_config=out_mem_config,
                            dtype=dtype,
                            compute_kernel_config=compute_kernel_config,
                            output_tile=output_tile,
                        )
                    ttnn.end_trace_capture(device, tid, cq_id=0)
                    profiler.start(f"run")
                    ttnn.execute_trace(device, tid, cq_id=0, blocking=False)
                    ttnn.synchronize_device(device)
                    profiler.end(f"run")
                    ttnn.release_trace(device, tid)
                else:
                    profiler.start(f"run")
                    for iter in range(0, num_measurement_iterations):
                        output_t = ttnn.matmul(
                            in0_t,
                            in1_t,
                            program_config=program_config,
                            memory_config=out_mem_config,
                            dtype=dtype,
                            compute_kernel_config=compute_kernel_config,
                            output_tile=output_tile,
                        )
                    ttnn.synchronize_device(device)
                    profiler.end(f"run")

                ttnn.DumpDeviceProfiler(device)

                inference_time_avg = profiler.get("run") / num_measurement_iterations
                tflops = 2 * m * k * n / 1e12 / inference_time_avg
                if math_fidelity == ttnn.MathFidelity.LoFi:
                    cycle_per_tile = LoFi_cycle
                elif math_fidelity == ttnn.MathFidelity.HiFi2:
                    cycle_per_tile = HiFi2_cycle
                elif math_fidelity == ttnn.MathFidelity.HiFi3:
                    cycle_per_tile = HiFi3_cycle
                elif math_fidelity == ttnn.MathFidelity.HiFi4:
                    cycle_per_tile = HiFi4_cycle
                num_cores_user_grid = grid_size[0] * grid_size[1]
                compute_grid_size = device.compute_with_storage_grid_size()
                num_cores_full_grid = compute_grid_size.x * compute_grid_size.y
                ideal_cycle_full_grid = m * k * n / tile_h / tile_w / 32 * cycle_per_tile / num_cores_full_grid
                ideal_cycle_user_grid = m * k * n / tile_h / tile_w / 32 * cycle_per_tile / num_cores_user_grid
                inference_cycle = inference_time_avg * get_device_freq() * 1e6
                utilization_full_grid = ideal_cycle_full_grid / inference_cycle
                utilization_user_grid = ideal_cycle_user_grid / inference_cycle
                utilization_full_grid_percentage = f"{utilization_full_grid * 100:.2f}%"
                utilization_user_grid_percentage = f"{utilization_user_grid * 100:.2f}%"
                logger.info(
                    f"M*K*N = {m}*{k}*{n} == inference time (avg): {inference_time_avg}, tflops (avg): {tflops}, utilization (vs user grid): {utilization_user_grid_percentage}, utilization (vs 8x8 grid): {utilization_full_grid_percentage}"
                )

                output_tensor = ttnn.to_torch(output_t)
                ttnn.deallocate(output_t)
                ttnn.deallocate(in0_t)
                ttnn.deallocate(in1_t)
                writer.writerow(
                    [
                        m,
                        k,
                        n,
                        f"{True}" if use_trace else f"{False}",
                        grid_size,
                        in0_sharded,
                        out_sharded,
                        in0_storage_type,
                        in1_storage_type,
                        out_storage_type,
                        dtype,
                        math_fidelity,
                        f"{inference_time_avg * 1e9:.2f}",
                        f"{tflops:.2f}",
                        utilization_user_grid_percentage,
                        utilization_full_grid_percentage,
                    ]
                )
