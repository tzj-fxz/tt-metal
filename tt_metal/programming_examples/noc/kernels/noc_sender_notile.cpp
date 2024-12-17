// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "dataflow_api.h"
#include "hostdevcommon/common_values.hpp"
#include "tools/profiler/kernel_profiler.hpp"
#include "debug/dprint.h"

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

    const uint32_t src_tile_bytes = get_tile_size(tt::CB::c_in0);
    // std::uint32_t batch = dram_tiles * src_tile_bytes / bandwidth_size;
    std::uint32_t batch = 1 << 10;

    cb_reserve_back(tt::CB::c_in0, cb_tiles);
    std::uint64_t dram_src_noc = get_noc_addr(dram_src_noc_x, dram_src_noc_y, input_address);
    std::uint32_t l1_write_addr_in0 = get_write_ptr(tt::CB::c_in0);
    std::uint32_t in0_start_addr = l1_write_addr_in0;
    noc_async_read(dram_src_noc, l1_write_addr_in0, bandwidth_size);
    noc_async_read_barrier();
    l1_write_addr_in0 += bandwidth_size;

    DeviceZoneScopedN("TEST-NoC-sender-notile");

    for (uint32_t r = 0; r < repeat; ++r) {
        for (uint32_t b = 0; b < batch; ++b) {
            uint64_t noc_addr = get_noc_addr(dst_core_x, dst_core_y, in0_start_addr);
            noc_async_write(in0_start_addr, noc_addr, bandwidth_size);
            // noc_async_write_barrier();
        }
    }
}
