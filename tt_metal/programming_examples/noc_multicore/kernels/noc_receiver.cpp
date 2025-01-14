// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "dataflow_api.h"
#include "hostdevcommon/common_values.hpp"
#include "tools/profiler/kernel_profiler.hpp"

void kernel_main() {
    std::uint32_t input_address = get_arg_val<uint32_t>(0);
    std::uint32_t src_core_x = get_arg_val<uint32_t>(1);
    std::uint32_t src_core_y = get_arg_val<uint32_t>(2);
    std::uint32_t dram_tiles = get_arg_val<uint32_t>(3);
    std::uint32_t cb_tiles = get_arg_val<uint32_t>(4);
    std::uint32_t repeat = get_arg_val<uint32_t>(5);

    const uint32_t src_tile_bytes = get_tile_size(tt::CB::c_in0);
    const DataFormat src_data_format = get_dataformat(tt::CB::c_in0);

    const InterleavedAddrGenFast<true> s0 = {
        .bank_base_address = input_address,
        .page_size = src_tile_bytes,
        .data_format = src_data_format
    };

    std::uint32_t batch = dram_tiles / cb_tiles;

    {
        // cb_wait_front(tt::CB::c_in0, cb_tiles);
        DeviceZoneScopedN("TEST-NoC-receiver");
    }
}
