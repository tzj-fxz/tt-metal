// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "debug/dprint.h"
#include <stdint.h>
// vector is not supported in reader kernel
// #include <vector>
#include "dataflow_api.h"
#include "tools/profiler/kernel_profiler.hpp"

void kernel_main() {
    uint32_t src_addr = get_arg_val<uint32_t>(0);
    uint32_t single_tile_size = get_arg_val<uint32_t>(1);
    uint32_t Mt = get_arg_val<uint32_t>(2);
    uint32_t Nt = get_arg_val<uint32_t>(3);
    uint32_t core_x = get_arg_val<uint32_t>(4);
    uint32_t core_y = get_arg_val<uint32_t>(5);
    uint32_t curr_core_x_logical = get_arg_val<uint32_t>(6);
    uint32_t curr_core_y_logical = get_arg_val<uint32_t>(7);
    uint32_t sender_semaphore_addr = get_semaphore(get_arg_val<uint32_t>(8));
    // logical-physical mapping of core coordinate, hard code here for wormhole_b0
    // std::vector<uint32_t> core_x_list = {1, 2, 3, 4, 5, 7, 8};
    // std::vector<uint32_t> core_y_list = {1, 2, 3, 4, 6, 7, 8};

    volatile tt_l1_ptr uint32_t *sender_semaphore_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t *>(sender_semaphore_addr);
    noc_semaphore_set(sender_semaphore_ptr, 0);
    const uint32_t single_tile_size_bytes = get_tile_size(tt::CB::c_in0);
    const DataFormat src_data_format = get_dataformat(tt::CB::c_in0);

    // read from src_addr into CB::c_in0
    const InterleavedAddrGenFast<true> src = {
        .bank_base_address = src_addr,
        .page_size = single_tile_size_bytes,
        .data_format = src_data_format
    };
    cb_reserve_back(tt::CB::c_in0, Mt * Nt);
    uint32_t l1_addr_read_in0 = get_write_ptr(tt::CB::c_in0);
    uint32_t l1_addr_read_in0_start = l1_addr_read_in0;
    uint32_t core_offset = curr_core_x_logical * core_y + curr_core_y_logical;
    uint32_t tile_id = core_offset * (Mt * Nt);
    {
        DeviceZoneScopedN("reader_read_from_DRAM");
        for (uint32_t h = 0; h < Mt; h++) {
            for (uint32_t w = 0; w < Nt; w++) {
                noc_async_read_tile(tile_id + (h * Nt + w), src, l1_addr_read_in0);
                l1_addr_read_in0 += single_tile_size_bytes;
            }
        }
        noc_async_read_barrier();
    }
    cb_push_back(tt::CB::c_in0, Mt * Nt);
    
    cb_reserve_back(tt::CB::c_out0, Mt * Nt);

    // write to other cores' SRAM
    // how many tiles should write to each core
    uint32_t row_stride = Mt / core_x;
    uint32_t col_stride = Nt / core_y;
    uint32_t l1_addr_write_out0 = get_write_ptr(tt::CB::c_out0);
    uint32_t l1_addr_write_out0_start = l1_addr_write_out0;
    uint32_t l1_addr_write_out0_offset = (curr_core_x_logical * row_stride * Nt + curr_core_y_logical * col_stride) * single_tile_size;
    {
        DeviceZoneScopedN("reader_write_to_noc");
        for (uint32_t i = 0; i < core_x; ++i) {
            for (uint32_t j = 0; j < core_y; ++j) {
                DeviceZoneScopedN("reader_write_to_noc_per_core");
                // hard code for logical-physical core-NoC mapping in wormhole_b0
                uint32_t dst_core_x_physical = i + 1 + (i >= 5);
                uint32_t dst_core_y_physical = j + 1 + (j >= 4);
                for (uint32_t h = 0; h < row_stride; ++h) {
                    for (uint32_t w = 0; w < col_stride; ++w) {
                        uint32_t l1_addr_read_in0_offset = ((i * row_stride + h) * Nt + (j * col_stride + w)) * single_tile_size;
                        uint32_t dst_noc_offset = (h * Nt + w) * single_tile_size;
                        uint64_t dst_core_addr = get_noc_addr(dst_core_x_physical, dst_core_y_physical, l1_addr_write_out0_start + l1_addr_write_out0_offset + dst_noc_offset);
                        noc_async_write(l1_addr_read_in0_start + l1_addr_read_in0_offset, dst_core_addr, single_tile_size);
                    }
                }
            }
        }
        noc_async_write_barrier();
    }
    // Note: Use semaphore to wait for other cores sending over, then push back
    {
        DeviceZoneScopedN("reader_semaphore");
        for (uint32_t i = 0; i < core_x; ++i) {
            for (uint32_t j = 0; j < core_y; ++j) {
                DeviceZoneScopedN("reader_semaphore_per_core");
                uint32_t dst_core_x_physical = i + 1 + (i >= 5);
                uint32_t dst_core_y_physical = j + 1 + (j >= 4);
                uint64_t dst_semaphore_addr = get_noc_addr(dst_core_x_physical, dst_core_y_physical, sender_semaphore_addr);
                noc_semaphore_inc(dst_semaphore_addr, 1);
            }
        }
        noc_semaphore_wait(sender_semaphore_ptr, core_x * core_y);
        cb_push_back(tt::CB::c_out0, Mt * Nt);
    }
    // {
    //     DeviceZoneScopedN("reader_push_to_writer");
    // }

    cb_wait_front(tt::CB::c_in0, Mt * Nt);
    cb_pop_front(tt::CB::c_in0, Mt * Nt);
}
