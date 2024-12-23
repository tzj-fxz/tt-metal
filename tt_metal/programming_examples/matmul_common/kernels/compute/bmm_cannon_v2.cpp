// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/matmul.h"
#include "tools/profiler/kernel_profiler.hpp"
#include "debug/dprint.h"

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
    uint32_t subblock_size_h = get_compile_time_arg_val(7); // for each subblock matrix
    uint32_t subblock_size_w = get_compile_time_arg_val(8); // for each subblock matrix
    uint32_t subblock_h = per_core_M / subblock_size_h; // in0_num_subblock
    uint32_t subblock_w = per_core_N / subblock_size_w; // in1_num_subblock
    uint32_t subblock_tiles = subblock_size_h * subblock_size_w;

    uint32_t num_block_x = Mt / per_core_M;
    uint32_t num_block_y = Nt / per_core_N;

    uint32_t core_x = get_arg_val<uint32_t>(0);
    uint32_t core_y = get_arg_val<uint32_t>(1);
    // uint32_t in0_sender_semaphore_id = get_semaphore(get_arg_val<uint32_t>(2));
    // uint32_t in1_sender_semaphore_id = get_semaphore(get_arg_val<uint32_t>(3));
    // volatile tt_l1_ptr uint32_t* in0_sender_semaphore_addr_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(in0_sender_semaphore_id);
    // volatile tt_l1_ptr uint32_t* in1_sender_semaphore_addr_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(in1_sender_semaphore_id);

    // uint32_t single_tile_size_bytes = get_tile_size(tt::CB::c_in0);


    mm_init();
    DeviceZoneScopedN("TEST-bmm-cannon");

    uint32_t in0_num_tiles = per_core_M * per_core_K;
    uint32_t in1_num_tiles = per_core_K * per_core_N;
    uint32_t out_num_tiles = per_core_M * per_core_N;

    for (uint32_t nb = 0; nb < batch; ++nb) {
        // each shift should do a matmul which contains many subblock matmul
        // TODO currently assume num_block_x == num_block_y
        bool spill = num_block_x > 0;
        bool enable_reload = false;
        for (uint32_t shift_num = 0; shift_num < num_block_x; ++shift_num) {
            cb_wait_front(tt::CB::c_in0, in0_num_tiles);
            cb_wait_front(tt::CB::c_in1, in1_num_tiles);
            // DPRINT << "bmm cannon start " << shift_num << ENDL();
            bool last_out = (shift_num == (num_block_x - 1));
            for (uint32_t subblock_m = 0; subblock_m < subblock_h; ++subblock_m) {
                for (uint32_t subblock_n = 0; subblock_n < subblock_w; ++subblock_n) {
                    acquire_dst();

                    if (enable_reload) {
                        copy_tile_to_dst_init_short();
                        cb_wait_front(tt::CB::c_intermed0, subblock_tiles);
                        for (uint32_t i = 0; i < subblock_tiles; i++) {
                            // in_tile_index=i is because the cb_pop_front below
                            copy_tile(tt::CB::c_intermed0, i, i);
                        }
                        cb_pop_front(tt::CB::c_intermed0, subblock_tiles);
                        mm_init_short();
                    }

                    int dst_index = 0;
                    int in0_subblock_offset = subblock_m * per_core_K * subblock_size_h;
                    int in1_subblock_offset = subblock_n * subblock_size_w;

                    for (uint32_t h = 0; h < subblock_size_h; ++h) {
                        for (uint32_t w = 0; w < subblock_size_w; ++w) {
                            // TODO The following per_core_K may be finer-grained
                            for (uint32_t inner_dim = 0; inner_dim < per_core_K; ++inner_dim) {
                                int in0_offset = in0_subblock_offset + h * per_core_K + inner_dim;
                                int in1_offset = in1_subblock_offset + w + inner_dim * per_core_N;
                                matmul_tiles(tt::CB::c_in0, tt::CB::c_in1, in0_offset, in1_offset, dst_index, false);
                            }
                            dst_index++;
                        }
                    }
                    
                    if (last_out) {
                        // DPRINT << "last out: use c_out0" << ENDL();
                        cb_reserve_back(tt::CB::c_out0, subblock_tiles);
                        for (uint32_t i = 0; i < subblock_tiles; ++i) {
                            pack_tile(i, tt::CB::c_out0);
                        }
                        cb_push_back(tt::CB::c_out0, subblock_tiles);
                    } else {
                        cb_reserve_back(tt::CB::c_intermed0, subblock_tiles);
                        for (uint32_t i = 0; i < subblock_tiles; ++i) {
                            pack_tile(i, tt::CB::c_intermed0);
                        }
                        cb_push_back(tt::CB::c_intermed0, subblock_tiles);
                    }

                    release_dst();
                }
            }
            if (spill) {
                enable_reload = true;
            }
            // DPRINT << "bmm cannon pop " << shift_num << ENDL();
            cb_pop_front(tt::CB::c_in0, in0_num_tiles);
            cb_pop_front(tt::CB::c_in1, in1_num_tiles);
            // after compute, signal reader kernel to send data at the front of "in" CB
            // noc_semaphore_set(in0_sender_semaphore_addr_ptr, 1);
            // noc_semaphore_set(in1_sender_semaphore_addr_ptr, 1);
        }
    }
}
} // NAMESPACE
