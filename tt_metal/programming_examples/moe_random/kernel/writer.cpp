// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"
#include "tools/profiler/kernel_profiler.hpp"

void kernel_main() {
    uint32_t dst_addr = get_arg_val<uint32_t>(0);
    uint32_t single_tile_size = get_arg_val<uint32_t>(1);
    uint32_t M = get_arg_val<uint32_t>(2);
    uint32_t core_x = get_arg_val<uint32_t>(3);
    uint32_t core_y = get_arg_val<uint32_t>(4);
    uint32_t curr_core_x_logical = get_arg_val<uint32_t>(5);
    uint32_t curr_core_y_logical = get_arg_val<uint32_t>(6);
    uint32_t K = get_arg_val<uint32_t>(7);

    // wait from sender cores sending over
    {
        DeviceZoneScopedN("writer_wait_for_noc");
        cb_wait_front(tt::CB::c_out0, M * core_x * core_y);
    }
    uint32_t l1_addr_read_out0 = get_read_ptr(tt::CB::c_out0);
    const uint32_t single_tile_size_bytes = get_tile_size(tt::CB::c_out0);
    const DataFormat dst_data_format = get_dataformat(tt::CB::c_out0);

    const InterleavedAddrGenFast<true> dst = {
        .bank_base_address = dst_addr,
        .page_size = single_tile_size_bytes,
        .data_format = dst_data_format
    };

    uint32_t core_offset = curr_core_x_logical * core_y + curr_core_y_logical;
    uint32_t tile_id = core_offset * M;
    {
        DeviceZoneScopedN("writer_write_to_DRAM");
        for (uint32_t i = 0; i < M; ++i) {
            noc_async_write_tile(tile_id + i, dst, l1_addr_read_out0);
            l1_addr_read_out0 += single_tile_size_bytes;
        }
        noc_async_write_barrier();
    }

    cb_pop_front(tt::CB::c_out0, M * core_x * core_y);
}

