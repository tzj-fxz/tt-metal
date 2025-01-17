// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "dataflow_api.h"
#include "hostdevcommon/common_values.hpp"
#include "tools/profiler/kernel_profiler.hpp"
#include "debug/dprint.h"

inline uint64_t get_timestamp() {
    uint32_t timestamp_low = reg_read(RISCV_DEBUG_REG_WALL_CLOCK_L);
    uint32_t timestamp_high = reg_read(RISCV_DEBUG_REG_WALL_CLOCK_H);
    return (((uint64_t)timestamp_high) << 32) | timestamp_low;
}

void kernel_main() {
    std::uint32_t input_address = get_arg_val<uint32_t>(0);
    std::uint32_t dst_core_x = get_arg_val<uint32_t>(1);
    std::uint32_t dst_core_y = get_arg_val<uint32_t>(2);
    std::uint32_t dram_tiles = get_arg_val<uint32_t>(3);
    std::uint32_t cb_tiles = get_arg_val<uint32_t>(4);
    std::uint32_t repeat = get_arg_val<uint32_t>(5);
    std::uint32_t dram_src_noc_x = get_arg_val<uint32_t>(6);
    std::uint32_t dram_src_noc_y = get_arg_val<uint32_t>(7);
    std::uint32_t bandwidth_size = get_arg_val<uint32_t>(8);
    std::uint32_t dst_core_x_down = get_arg_val<uint32_t>(9);
    std::uint32_t dst_core_y_down = get_arg_val<uint32_t>(10);
    std::uint32_t core_x = get_arg_val<uint32_t>(11);
    std::uint32_t core_y = get_arg_val<uint32_t>(12);

    const uint32_t src_tile_bytes = get_tile_size(tt::CB::c_in0);
    // std::uint32_t batch = dram_tiles * src_tile_bytes / bandwidth_size;
    std::uint32_t batch = 1;

    cb_reserve_back(tt::CB::c_in0, cb_tiles);
    std::uint64_t dram_src_noc = get_noc_addr(dram_src_noc_x, dram_src_noc_y, input_address);

    std::uint32_t l1_write_addr_in0 = get_write_ptr(tt::CB::c_in0);
    std::uint32_t in0_start_addr = l1_write_addr_in0;

    for (uint32_t i = 0; i < bandwidth_size; ++i) {
        *(uint32_t *)(l1_write_addr_in0) = (uint16_t)1;
        l1_write_addr_in0 += 2;
    }

    // {
    //     DeviceZoneScopedN("TEST-NoC-sender_dram");
    //     noc_async_read(dram_src_noc, l1_write_addr_in0, bandwidth_size);
    //     noc_async_read_barrier();
    //     l1_write_addr_in0 += bandwidth_size;
    // }

    // warm-up
    {
        DeviceZoneScopedN("TEST-NoC-sender-warmup");
        uint64_t start = get_timestamp();
        for (uint32_t r = 0; r < repeat; ++r) {
            for (uint32_t b = 0; b < batch; ++b) {
                uint64_t noc_addr = get_noc_addr(dst_core_x, dst_core_y, l1_write_addr_in0);
                noc_async_write(l1_write_addr_in0, noc_addr, bandwidth_size);
                // uint64_t noc_addr_down = get_noc_addr(dst_core_x_down, dst_core_y_down, in0_start_addr + bandwidth_size);
                // noc_async_write(l1_write_addr_in0, noc_addr_down, bandwidth_size);
            }
        }
        noc_async_write_barrier();
        uint64_t end = get_timestamp();
        DPRINT << end - start << ENDL();
        l1_write_addr_in0 += bandwidth_size;
    }

    // test bandwidth
    {
        DeviceZoneScopedN("TEST-NoC-sender-bandwidth");
        cb_reserve_back(tt::CB::c_in0, cb_tiles);
        for (uint32_t r = 0; r < repeat; ++r) {
            for (uint32_t b = 0; b < batch; ++b) {
                uint64_t noc_addr = get_noc_addr(dst_core_x, dst_core_y, l1_write_addr_in0);
                noc_async_write(l1_write_addr_in0, noc_addr, bandwidth_size);
                // uint64_t noc_addr_down = get_noc_addr(dst_core_x_down, dst_core_y_down, in0_start_addr + bandwidth_size);
                // noc_async_write(l1_write_addr_in0, noc_addr_down, bandwidth_size);
            }
        }

        noc_async_write_barrier();
        cb_push_back(tt::CB::c_in0, cb_tiles);
    }
}
