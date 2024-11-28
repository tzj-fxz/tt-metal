// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "debug/dprint.h"
#include <stdint.h>
#include "dataflow_api.h"

void kernel_main() {
    const uint32_t src_addr = get_compile_time_arg_val<uint32_t>(0);
    const uint32_t single_tile_size = get_compile_time_arg_val<uint32_t>(1);
    const uint32_t sharded_height = get_compile_time_arg_val<uint32_t>(2);
    const uint32_t sharded_width = get_compile_time_arg_val<uint32_t>(3);
    const uint32_t Mt = get_compile_time_arg_val<uint32_t>(4);
    const uint32_t Nt = get_compile_time_arg_val<uint32_t>(5);
    const uint32_t core_x = get_compile_time_arg_val<uint32_t>(6);
    const uint32_t core_y = get_compile_time_arg_val<uint32_t>(7);
    const uint32_t curr_idx_h = get_arg_val<uint32_t>(0);
    const uint32_t curr_idx_w = get_arg_val<uint32_t>(1);
    const uint32_t curr_core_x = get_arg_val<uint32_t>(2);
    const uint32_t curr_core_y = get_arg_val<uint32_t>(3);

    // read from src_addr into CB::c_in0
    const InterleavedAddrGen<true> src = {
        .bank_base_address = src_addr,
        .page_size = single_tile_size
    };
    cb_reserve_back(tt::CB::c_in0, sharded_height * sharded_width);
    uint32_t l1_addr_read_in0 = get_write_ptr(tt::CB::c_in0);
    uint32_t l1_addr_read_in0_start = l1_addr_read_in0;
    for (uint32_t h = 0; h < sharded_height; h++) {
        for (uint32_t w = 0; w < sharded_width; w++) {
            uint32_t tile_id = (curr_idx_h + h) * Nt + (curr_idx_w + w);
            uint64_t src_noc_addr = get_noc_addr(tile_id, src);
            noc_async_read(src_noc_addr, l1_addr_read_in0, single_tile_size);
            l1_addr_read_in0 += single_tile_size;
        }
    }
    noc_async_read_barrier();
    cb_push_back(tt::CB::c_in0, sharded_height * sharded_width);
    
    // write to other cores' CB::c_out0
    uint32_t all2all_tile_on_height = sharded_height / (core_x);
    uint32_t all2all_tile_on_width = sharded_width / (core_y);
    uint32_t l1_addr_write_out0 = get_write_ptr(tt::CB::c_out0);
    for (uint32_t i = 0; i < core_x; i++) {
        for (uint32_t j = 0; j < core_y; j++) {
            if (i == curr_core_x && j == curr_core_y) {
                continue;
            }
            uint32_t from_l1_addr_base = l1_addr_read_in0_start + (i * all2all_tile_on_height * sharded_width + j * all2all_tile_on_width) * single_tile_size;
            uint64_t send_to_noc_addr_base = get_noc_addr(i, j, l1_addr_write_out0 + (curr_core_x * all2all_tile_on_height * sharded_width + curr_core_y * all2all_tile_on_width) * single_tile_size);
            for (uint32_t h = 0; h < all2all_tile_on_height; h++) {
                for (uint32_t w = 0; w < all2all_tile_on_width; w++) {
                    uint32_t from_l1_addr = from_l1_addr_base + (h * sharded_width + w) * single_tile_size;
                    uint64_t send_to_noc_addr = send_to_noc_addr_base + (h * sharded_width + w) * single_tile_size;
                    noc_async_write(from_l1_addr, send_to_noc_addr, single_tile_size);
                }
            }
        }
    }
    noc_async_write_barrier();
    cb_push_back(tt::CB::c_out0, sharded_height * sharded_width);
    cb_pop_front(tt::CB::c_in0, sharded_height * sharded_width);
}
