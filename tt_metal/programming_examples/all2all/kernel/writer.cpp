// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"

void kernel_main() {
    uint32_t dst_addr = get_arg_compile_time_val<uint32_t>(0);
    uint32_t single_tile_size = get_arg_compile_time_val<uint32_t>(1);
    uint32_t sharded_height = get_arg_compile_time_val<uint32_t>(2);
    uint32_t sharded_width = get_arg_compile_time_val<uint32_t>(3);
    uint32_t Mt = get_arg_compile_time_val<uint32_t>(4);
    uint32_t Nt = get_arg_compile_time_val<uint32_t>(5);
    const uint32_t curr_core_x = get_arg_val<uint32_t>(0);
    const uint32_t curr_core_y = get_arg_val<uint32_t>(1);
    uint32_t all2all_tile_on_height = sharded_height / (core_x);
    uint32_t all2all_tile_on_width = sharded_width / (core_y);
    {
        cb_wait_front(tt::CB::c_out0, sharded_height * sharded_width);
    }

    const InterleavedAddrGenFast<true> dst = {
        .bank_base_address = dst_addr,
        .page_size = single_tile_size,
        .data_format = get_dataformat(tt::CB::c_out0)
    };

    uint32_t l1_addr_read_out0 = get_read_ptr(tt::CB::c_out0);
    uint32_t scatter_core_x = curr_core_x * all2all_tile_on_height;
    uint32_t scatter_core_y = curr_core_y * all2all_tile_on_width;
    for (uint32_t i = 0; i < core_x; i++) {
        for (uint32_t j = 0; j < core_y; j++) {
            uint32_t src_tile_start_id = i * all2all_tile_on_height * sharded_width + j * all2all_tile_on_width;
            uint32_t dst_tile_start_id = i * sharded_height * Nt + j * sharded_width;
            for (uint32_t h = 0; h < all2all_tile_on_height; h++) {
                for (uint32_t w = 0; w < all2all_tile_on_width; w++) {
                    uint32_t src_tile_id = src_tile_start_id + h * sharded_width + w;
                    uint32_t dst_tile_id = dst_tile_start_id + h * Nt + w;
                    uint64_t dst_noc_addr = get_noc_addr(dst_tile_id, dst);
                    noc_async_write(l1_addr_read_out0 + src_tile_id * single_tile_size, dst_noc_addr, single_tile_size);
                }
            }
            noc_async_write_barrier();
        }
    }
    cb_pop_front(tt::CB::c_out0, sharded_height * sharded_width);
}

