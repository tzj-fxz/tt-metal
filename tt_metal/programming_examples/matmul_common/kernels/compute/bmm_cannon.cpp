// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/matmul.h"
#include "tools/profiler/kernel_profiler.hpp"

using std::uint32_t;

// matmul C=A*B using dims MK*KN = MN (row major order)
//
namespace NAMESPACE {
void MAIN {

    uint32_t Mt = get_compile_time_arg_val(0);
    uint32_t Nt = get_compile_time_arg_val(1);
    uint32_t Kt = get_compile_time_arg_val(2);
    uint32_t batch = get_compile_time_arg_val(3);
    uint32_t per_core_M = get_compile_time_arg_val(4);
    uint32_t per_core_N = get_compile_time_arg_val(5);
    uint32_t per_core_K = get_compile_time_arg_val(6);

    uint32_t num_block_x = Mt / per_core_M;
    uint32_t num_block_y = Nt / per_core_N;

    uint32_t core_x = get_arg_val(0);
    uint32_t core_y = get_arg_val(1);

    uint32_t single_tile_size_bytes = get_tile_size(tt::CB::c_in0);

    mm_init();
    DeviceZoneScopedN("TEST-bmm-start");

    uint32_t in0_num_tiles = per_core_M * per_core_K;
    uint32_t in1_num_tiles = per_core_K * per_core_N;
    uint32_t out_num_tiles = per_core_M * per_core_N;

    for (uint32_t nb = 0; nb < batch; ++nb) {
        cb_wait_front(tt::CB::c_in0, in0_num_tiles * 2);
        cb_wait_front(tt::CB::c_in1, in1_num_tiles * 2);
        // pop the unskewed block first
        cb_pop_front(tt::CB::c_in0, in0_num_tiles);
        cb_pop_front(tt::CB::c_in1, in1_num_tiles);

        acquire_dst();
        int dst_index = 0;
        for (uint32_t block_m = 0; block_m < per_core_M; ++block_m) {
            int in0_offset = block_m * per_core_K;
            for (uint32_t block_n = 0; block_n < per_core_N; ++block_n) {
                int in1_offset = block_n;
                for (uint32_t block_k = 0; block_k < per_core_K; ++block_k) {
                    matmul_tiles(tt::CB::c_in0, tt::CB::c_in1, in0_offset + block_k, block_k * per_core_N + in1_offset, dst_index, false);
                }
                dst_index++;
            }
        }
        cb_reserve_back(tt::CB::c_out0, out_num_tiles);
        for (uint32_t i = 0; i < out_num_tiles; ++i) {
            pack_tile(i, tt::CB::c_out0);
        }
        cb_push_back(tt::CB::c_out0, out_num_tiles);
        release_dst();

        // after compute, the input tiles should be transfered to neighbor as cannon does
        // use dataflow CB
        uint32_t cb_in0_ptr = get_ptr(tt::CB::c_in0);
        uint32_t cb_in1_ptr = get_ptr(tt::CB::c_in1);
        uint32_t cb_dataflow0_ptr = get_ptr(tt::CB::dataflow0);
        uint32_t cb_dataflow1_ptr = get_ptr(tt::CB::dataflow1);
        uint64_t cb_dataflow0_noc = get_noc_addr((core_x + 1) % num_block_x, core_y, cb_dataflow0_ptr);
        uint64_t cb_dataflow1_noc = get_noc_addr(core_x, (core_y + 1) % num_block_y, cb_dataflow1_ptr);
        noc_async_write(cb_in0_ptr, cb_dataflow0_noc, in0_num_tiles * single_tile_size_bytes);
        noc_async_write_barrier();
        noc_async_write(cb_in1_ptr, cb_dataflow1_noc, in1_num_tiles * single_tile_size_bytes);
        noc_async_write_barrier();
        cb_pop_front(tt::CB::c_in0, in0_num_tiles);
        cb_pop_front(tt::CB::c_in1, in1_num_tiles);
    }

}
} // NAMESPACE
