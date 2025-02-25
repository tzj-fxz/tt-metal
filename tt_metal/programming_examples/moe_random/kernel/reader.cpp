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
    uint32_t M = get_arg_val<uint32_t>(2);
    uint32_t core_x = get_arg_val<uint32_t>(3);
    uint32_t core_y = get_arg_val<uint32_t>(4);
    uint32_t curr_core_x_logical = get_arg_val<uint32_t>(5);
    uint32_t curr_core_y_logical = get_arg_val<uint32_t>(6);
    uint32_t sender_semaphore_addr = get_semaphore(get_arg_val<uint32_t>(7));
    uint32_t K = get_arg_val<uint32_t>(8);
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
    cb_reserve_back(tt::CB::c_in0, M);
    uint32_t l1_addr_read_in0 = get_write_ptr(tt::CB::c_in0);
    uint32_t l1_addr_read_in0_start = l1_addr_read_in0;
    uint32_t core_offset = curr_core_x_logical * core_y + curr_core_y_logical;
    uint32_t tile_id = core_offset * M;
    {
        DeviceZoneScopedN("reader_read_from_DRAM");
        for (uint32_t h = 0; h < M; h++) {
            noc_async_read_tile(tile_id + h, src, l1_addr_read_in0);
            l1_addr_read_in0 += single_tile_size_bytes;
        }
        noc_async_read_barrier();
    }
    cb_push_back(tt::CB::c_in0, M);

    // Since the sending process is random, we must reserve enough space for receiving data
    cb_reserve_back(tt::CB::c_out0, M * core_x * core_y);
    uint32_t l1_addr_write_out0 = get_write_ptr(tt::CB::c_out0);
    uint32_t l1_addr_write_out0_start = l1_addr_write_out0;

    bool core_used[128] = {0};
    for (uint32_t i = 0; i < K; ++i) {
        DeviceZoneScopedN("reader_random_send");
        // Get random K core to send tokens, imitating top-k experts
        uint64_t timestamp = reg_read(RISCV_DEBUG_REG_WALL_CLOCK_L);
        uint32_t core_index = timestamp % (core_x * core_y);
        while (core_used[core_index]) {
            timestamp = reg_read(RISCV_DEBUG_REG_WALL_CLOCK_L);
            core_index = timestamp % (core_x * core_y);
        }
        core_used[core_index] = true;
        
        uint32_t dst_core_x = core_index / core_y;
        uint32_t dst_core_y = core_index % core_y;
        uint32_t dst_core_x_physical = dst_core_x + 1 + (dst_core_x >= 4);
        uint32_t dst_core_y_physical = dst_core_y + 1 + (dst_core_y >= 5);
        // The remote NoC address is bound to the offset calculated by current core
        uint32_t l1_addr_write_out0_offset = M * single_tile_size * core_offset;
        uint64_t dst_core_addr = get_noc_addr(dst_core_x_physical, dst_core_y_physical, l1_addr_write_out0_start + l1_addr_write_out0_offset);
        noc_async_write(l1_addr_read_in0_start, dst_core_addr, M * single_tile_size);
    }
    noc_async_write_barrier();

    // Note: Use semaphore to wait for other cores sending over, then push back
    {
        DeviceZoneScopedN("reader_semaphore");
        for (uint32_t i = 0; i < core_x; ++i) {
            for (uint32_t j = 0; j < core_y; ++j) {
                DeviceZoneScopedN("reader_semaphore_per_core");
                uint32_t dst_core_x_physical = i + 1 + (i >= 4);
                uint32_t dst_core_y_physical = j + 1 + (j >= 5);
                uint64_t dst_semaphore_addr = get_noc_addr(dst_core_x_physical, dst_core_y_physical, sender_semaphore_addr);
                noc_semaphore_inc(dst_semaphore_addr, 1);
            }
        }
        noc_semaphore_wait(sender_semaphore_ptr, core_x * core_y);
        cb_push_back(tt::CB::c_out0, M * core_x * core_y);
    }
    // {
    //     DeviceZoneScopedN("reader_push_to_writer");
    // }

    cb_wait_front(tt::CB::c_in0, M);
    cb_pop_front(tt::CB::c_in0, M);
}
