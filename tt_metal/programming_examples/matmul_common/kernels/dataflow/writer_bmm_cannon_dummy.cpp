// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"
#include "tools/profiler/kernel_profiler.hpp"
#include "debug/dprint.h"

void kernel_main() {
    DeviceZoneScopedN("TEST-writer_bmm_cannon");

    uint32_t Mt = get_arg_val<uint32_t>(0);
    uint32_t Nt = get_arg_val<uint32_t>(1);
    uint32_t Kt = get_arg_val<uint32_t>(2);
    uint32_t batch = get_arg_val<uint32_t>(3);
    uint32_t core_x = get_arg_val<uint32_t>(4);
    uint32_t core_y = get_arg_val<uint32_t>(5);
    uint32_t per_core_M = get_arg_val<uint32_t>(6);
    uint32_t per_core_N = get_arg_val<uint32_t>(7);
    uint32_t per_core_K = get_arg_val<uint32_t>(8);
    uint32_t subblock_size_h = get_arg_val<uint32_t>(9);
    uint32_t subblock_size_w = get_arg_val<uint32_t>(10);
    uint32_t dst_addr = get_arg_val<uint32_t>(11);
    uint32_t dram_shard_x = get_arg_val<uint32_t>(12);
    uint32_t dram_shard_y = get_arg_val<uint32_t>(13);
    uint32_t subblock_h = dram_shard_x / subblock_size_h;
    uint32_t subblock_w = dram_shard_y / subblock_size_w;
    uint32_t subblock_tiles = subblock_size_h * subblock_size_w;

    constexpr bool out_is_dram = get_compile_time_arg_val(0) == 1;
    uint32_t output_tiles = dram_shard_x * dram_shard_y;
    uint32_t output_index = core_x * per_core_M * Nt + core_y * per_core_N;

    // single-tile
    const uint32_t single_tile_size_bytes = get_tile_size(tt::CB::c_out0);
    const DataFormat data_format = get_dataformat(tt::CB::c_out0);

    const InterleavedAddrGenFast<out_is_dram> s = {
        .bank_base_address = dst_addr,
        .page_size = single_tile_size_bytes,
        .data_format = data_format
    };

    for (uint32_t b = 0; b < batch; b++) {
        for (uint32_t dram_shard_h = 0; dram_shard_h < per_core_M / dram_shard_x; ++dram_shard_h) {
            for (uint32_t dram_shard_w = 0; dram_shard_w < per_core_N / dram_shard_y; ++dram_shard_w) {
                uint32_t output_shard_index = output_index + dram_shard_h * dram_shard_x * Nt + dram_shard_w * dram_shard_y;
                for (uint32_t subblock_m = 0; subblock_m < subblock_h; ++subblock_m) {
                    for (uint32_t subblock_n = 0; subblock_n < subblock_w; ++subblock_n) {
                        cb_wait_front(tt::CB::c_out0, subblock_tiles);
                        // uint32_t l1_read_addr_out = get_read_ptr(tt::CB::c_out0);
                        // uint32_t output_offset = output_shard_index + subblock_m * subblock_size_h * Nt + subblock_n * subblock_size_w;
                        // for (uint32_t h = 0; h < subblock_size_h; ++h) {
                        //     for (uint32_t w = 0; w < subblock_size_w; ++w) {
                        //         noc_async_write_tile(output_offset + h * Nt + w, s, l1_read_addr_out);
                        //         l1_read_addr_out += single_tile_size_bytes;
                        //     }
                        // }
                        // noc_async_write_barrier();
                        cb_pop_front(tt::CB::c_out0, subblock_tiles);
                    }
                }                
            }
        }
    }
}
