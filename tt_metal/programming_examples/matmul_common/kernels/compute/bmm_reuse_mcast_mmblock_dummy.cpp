// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/matmul.h"
#include "tools/profiler/kernel_profiler.hpp"

namespace NAMESPACE {
void MAIN {
    DeviceZoneScopedN("reuse-mcast-bmm-block-dummy");

    uint32_t in0_block_w = get_compile_time_arg_val(0); // inner block size in tiles
    uint32_t in0_num_subblocks = get_compile_time_arg_val(1); // outer row block size (in inner row blocks)
    uint32_t in0_block_num_tiles = get_compile_time_arg_val(2); // out_subblock_h*in0_block_w*in0_num_subblocks;
    uint32_t in0_subblock_num_tiles = get_compile_time_arg_val(3);  // out_subblock_h*in0_block_w
    uint32_t in1_num_subblocks = get_compile_time_arg_val(4); // outer column block size (in inner column blocks)
    uint32_t in1_block_num_tiles = get_compile_time_arg_val(5); //out_subblock_w*in0_block_w* in1_num_subblocks;
    uint32_t in1_per_core_w = get_compile_time_arg_val(6); // out_subblock_w*in1_num_subblocks
    uint32_t num_blocks = get_compile_time_arg_val(7);  // outer inner dim (in inner dim blocks)
    uint32_t out_subblock_h = get_compile_time_arg_val(8); // inner row block size in tiles
    uint32_t out_subblock_w = get_compile_time_arg_val(9); // inner column block size in tiles
    uint32_t out_subblock_num_tiles = get_compile_time_arg_val(10); // out_subblock_h * out_subblock_w;
    uint32_t batch = get_compile_time_arg_val(11); // batch dim

    // mm_init();

    for (uint32_t b = 0; b < batch; b++){
        bool spill = num_blocks > 1;
        bool enable_reload = false;
        uint32_t out_num_tiles_to_wait = out_subblock_num_tiles;

        for(uint32_t block = 0; block < num_blocks; block++)
        {
            bool last_out = block == (num_blocks-1);

            cb_wait_front(tt::CB::c_in0, in0_block_num_tiles);
            cb_wait_front(tt::CB::c_in1, in1_block_num_tiles);
            if (last_out) {
                for (uint32_t in0_subblock = 0; in0_subblock < in0_num_subblocks; in0_subblock++) {
                    for (uint32_t in1_subblock = 0; in1_subblock < in1_num_subblocks; in1_subblock++) {
                        uint32_t index = in0_subblock * in1_num_subblocks + in1_subblock;
                        // copy_tile_init();
                        // tile_regs_acquire();
                        // copy_tile(tt::CB::c_in0, index * out_subblock_num_tiles, 0);
                        // tile_regs_commit();
                        cb_reserve_back(tt::CB::c_out0, out_subblock_num_tiles);
                        // tile_regs_wait();
                        // pack_tile(0, tt::CB::c_out0);
                        // tile_regs_release();
                        cb_push_back(tt::CB::c_out0, out_subblock_num_tiles);
                    }
                }
            }
            cb_pop_front(tt::CB::c_in0, in0_block_num_tiles);
            cb_pop_front(tt::CB::c_in1, in1_block_num_tiles);
        }
    }
}
}
