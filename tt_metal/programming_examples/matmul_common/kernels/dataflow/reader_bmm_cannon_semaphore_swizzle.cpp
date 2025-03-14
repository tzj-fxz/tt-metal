// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"
#include "hostdevcommon/common_values.hpp"
#include "tools/profiler/kernel_profiler.hpp"
#include "debug/dprint.h"

void kernel_main() {
    DeviceZoneScopedN("TEST-reader_bmm_all");

    constexpr bool src0_is_dram = get_compile_time_arg_val(0) == 1;
    constexpr bool src1_is_dram = get_compile_time_arg_val(1) == 1;

    uint32_t Mt = get_arg_val<uint32_t>(0);
    uint32_t Nt = get_arg_val<uint32_t>(1);
    uint32_t Kt = get_arg_val<uint32_t>(2);
    uint32_t batch = get_arg_val<uint32_t>(3);
    uint32_t bcast_B = get_arg_val<uint32_t>(4);
    uint32_t core_x = get_arg_val<uint32_t>(5);
    uint32_t core_y = get_arg_val<uint32_t>(6);
    uint32_t src0_addr = get_arg_val<uint32_t>(7);
    uint32_t src0_start_tile_id = get_arg_val<uint32_t>(8);
    uint32_t src0_block_tiles = get_arg_val<uint32_t>(9);
    uint32_t src1_addr = get_arg_val<uint32_t>(10);
    uint32_t src1_start_tile_id = get_arg_val<uint32_t>(11);
    uint32_t src1_block_tiles = get_arg_val<uint32_t>(12);
    uint32_t per_core_M = get_arg_val<uint32_t>(13);
    uint32_t per_core_N = get_arg_val<uint32_t>(14);
    uint32_t per_core_K = get_arg_val<uint32_t>(15);
    uint32_t in0_sender_semaphore_addr = get_semaphore(get_arg_val<uint32_t>(16));
    uint32_t in0_receiver_semaphore_addr = get_semaphore(get_arg_val<uint32_t>(17));
    uint32_t in1_sender_semaphore_addr = get_semaphore(get_arg_val<uint32_t>(18));
    uint32_t in1_receiver_semaphore_addr = get_semaphore(get_arg_val<uint32_t>(19));
    uint32_t src0_next_core_physical_x = get_arg_val<uint32_t>(20);
    uint32_t src0_next_core_physical_y = get_arg_val<uint32_t>(21);
    uint32_t src1_next_core_physical_x = get_arg_val<uint32_t>(22);
    uint32_t src1_next_core_physical_y = get_arg_val<uint32_t>(23);
    uint32_t src0_prev_core_physical_x = get_arg_val<uint32_t>(24);
    uint32_t src0_prev_core_physical_y = get_arg_val<uint32_t>(25);
    uint32_t src1_prev_core_physical_x = get_arg_val<uint32_t>(26);
    uint32_t src1_prev_core_physical_y = get_arg_val<uint32_t>(27);
    uint32_t dram_shard_x = get_arg_val<uint32_t>(28);
    uint32_t dram_shard_y = get_arg_val<uint32_t>(29);
    uint32_t core_physical_x = get_arg_val<uint32_t>(30);
    uint32_t core_physical_y = get_arg_val<uint32_t>(31);

    // DPRINT << "core: " << core_x + 1 << ", " << core_y + 1 << ENDL();
    volatile tt_l1_ptr uint32_t* in0_sender_semaphore_addr_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(in0_sender_semaphore_addr);
    volatile tt_l1_ptr uint32_t* in0_receiver_semaphore_addr_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(in0_receiver_semaphore_addr);
    volatile tt_l1_ptr uint32_t* in1_sender_semaphore_addr_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(in1_sender_semaphore_addr);
    volatile tt_l1_ptr uint32_t* in1_receiver_semaphore_addr_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(in1_receiver_semaphore_addr);
    noc_semaphore_set(in0_sender_semaphore_addr_ptr, 0);
    noc_semaphore_set(in1_sender_semaphore_addr_ptr, 0);
    noc_semaphore_set(in0_receiver_semaphore_addr_ptr, 0);
    noc_semaphore_set(in1_receiver_semaphore_addr_ptr, 0);

    uint32_t num_block_x = Mt / per_core_M;
    uint32_t num_block_y = Nt / per_core_N;

    constexpr uint32_t onetile = 1;
    const uint32_t src0_tile_bytes = get_tile_size(tt::CB::c_in0);
    const DataFormat src0_data_format = get_dataformat(tt::CB::c_in0);
    const uint32_t src1_tile_bytes = get_tile_size(tt::CB::c_in1);
    const DataFormat src1_data_format = get_dataformat(tt::CB::c_in1);

    uint32_t itileA_batch = 0;
    uint32_t itileB_batch = 0;

    const InterleavedAddrGenFast<src0_is_dram> s0 = {
        .bank_base_address = src0_addr,
        .page_size = src0_tile_bytes,
        .data_format = src0_data_format
    };

    const InterleavedAddrGenFast<src1_is_dram> s1 = {
        .bank_base_address = src1_addr,
        .page_size = src1_tile_bytes,
        .data_format = src1_data_format
    };

    uint32_t l1_write_addr_in0;
    uint32_t l1_write_addr_in1;
    uint32_t l1_write_addr_in2;
    uint32_t l1_write_addr_in3;
    uint32_t src0_sharded_tiles = dram_shard_x * per_core_K;
    uint32_t src1_sharded_tiles = per_core_K * dram_shard_y;

    // swizzle round
    uint32_t dram_shard_h = per_core_M / dram_shard_x;
    uint32_t dram_shard_w = per_core_N / dram_shard_y;
    uint32_t round = dram_shard_h * dram_shard_w;

    // TODO currenty batch = 1
    for (uint32_t nbatch = 0; nbatch < batch; ++nbatch) {
        uint32_t dram_shard_hi = 0;
        uint32_t dram_shard_wi = 0;
        uint32_t in0_start_addr;
        uint32_t in1_start_addr;
        uint32_t in2_start_addr;
        uint32_t in3_start_addr;
        // To pipelining DRAM and NoC, two set of CBs should be used in turn
        uint32_t noc_cb_A = tt::CB::c_in0;
        uint32_t noc_cb_B = tt::CB::c_in1;
        uint32_t dram_cb_A = tt::CB::c_in2;
        uint32_t dram_cb_B = tt::CB::c_in3;
        // initially read A and B
        for (uint32_t r = 0; r < round; ++r) {
            cb_reserve_back(dram_cb_A, src0_sharded_tiles);
            cb_reserve_back(dram_cb_B, src1_sharded_tiles);
            DeviceZoneScopedN("TEST-reader_bmm_cannon_initial");
            // cb_reserve_back(tt::CB::c_in0, src0_block_tiles);
            // cb_reserve_back(tt::CB::c_in1, src1_block_tiles);
            // For DRAM independent buffer
            bool swizzleA = true;
            bool swizzleB = true;
            if (r != 0) {
                if (dram_shard_hi % 2 == 0) {
                    if (dram_shard_wi < dram_shard_w - 1) {
                        ++dram_shard_wi;
                        swizzleA = false;
                    } else {
                        ++dram_shard_hi;
                        swizzleB = false;
                    }
                } else {
                    if (dram_shard_wi > 0) {
                        --dram_shard_wi;
                        swizzleA = false;
                    } else {
                        ++dram_shard_hi;
                        swizzleB = false;
                    }
                }
            }
            // currently only support square, means per_core_M = per_core_N (= per_core_K)
            // now, we manually skew src_start_tile_id to let reader kernel read the correct initial block.
            if (swizzleA) {
                l1_write_addr_in2 = get_write_ptr(dram_cb_A);
                in2_start_addr = l1_write_addr_in2;
                src0_start_tile_id = core_x * Kt * per_core_M + ((core_y + core_x) % num_block_y) * per_core_K;
                src0_start_tile_id += dram_shard_hi * dram_shard_x * Kt;
                for (uint32_t h = 0; h < dram_shard_x; ++h) {
                    for (uint32_t w = 0; w < per_core_K; ++w) {
                        noc_async_read_tile(src0_start_tile_id + h * Kt + w, s0, l1_write_addr_in2);
                        l1_write_addr_in2 += src0_tile_bytes;
                    }
                }
            }
            if (swizzleB) {
                l1_write_addr_in3 = get_write_ptr(dram_cb_B);
                in3_start_addr = l1_write_addr_in3;
                src1_start_tile_id = ((core_x + core_y) % num_block_x) * Nt * per_core_K + core_y * per_core_N;
                src1_start_tile_id += dram_shard_wi * dram_shard_y;
                for (uint32_t h = 0; h < per_core_K; ++h) {
                    for (uint32_t w = 0; w < dram_shard_y; ++w) {
                        noc_async_read_tile(src1_start_tile_id + h * Nt + w, s1, l1_write_addr_in3);
                        l1_write_addr_in3 += src1_tile_bytes;
                    }
                }
            }
            noc_async_read_barrier();
            
            // Copy from dataflow CB to IN CB: currently use "NoC" operation
            cb_reserve_back(noc_cb_A, src0_sharded_tiles);
            cb_reserve_back(noc_cb_B, src1_sharded_tiles);
            l1_write_addr_in0 = get_write_ptr(noc_cb_A);
            l1_write_addr_in1 = get_write_ptr(noc_cb_B);
            // pay attention to the start address, need for later shift
            in0_start_addr = l1_write_addr_in0;
            in1_start_addr = l1_write_addr_in1;
            uint64_t dataflow_to_in0 = get_noc_addr(core_physical_x, core_physical_y, l1_write_addr_in0);
            uint64_t dataflow_to_in1 = get_noc_addr(core_physical_x, core_physical_y, l1_write_addr_in1);
            noc_async_write(in2_start_addr, dataflow_to_in0, src0_tile_bytes * dram_shard_x * per_core_K);
            noc_async_write(in3_start_addr, dataflow_to_in1, src1_tile_bytes * per_core_K * dram_shard_y);
            noc_async_write_barrier();
            cb_push_back(noc_cb_A, src0_sharded_tiles);
            cb_push_back(noc_cb_B, src1_sharded_tiles);

            // in cannon process, do not need to read one block from neighbor compute kernel
            // instead, directly write tile in current core into next core (shifted)
            // pay attention to the CB capacity: we first write into CB, then pop from CB, and finally push into CB for compute kernel to process
            DPRINT << "src0 to core: " << src0_next_core_physical_x << ", " << src0_next_core_physical_y << ENDL();
            DPRINT << "src1 to core: " << src1_next_core_physical_x << ", " << src1_next_core_physical_y << ENDL();

            for (uint32_t shift_num = 0; shift_num < Mt / per_core_M - 1; ++shift_num) {
                DeviceZoneScopedN("TEST-reader_bmm_cannon_shift");
                // cb_reserve_back(tt::CB::c_in0, src0_block_tiles);
                // cb_reserve_back(tt::CB::c_in1, src1_block_tiles);
                cb_reserve_back(noc_cb_A, src0_sharded_tiles);
                cb_reserve_back(noc_cb_B, src1_sharded_tiles);
                l1_write_addr_in0 = get_write_ptr(noc_cb_A);
                l1_write_addr_in1 = get_write_ptr(noc_cb_B);
                // l1 address should be updated
                uint64_t src0_shift_to_addr_in0 = get_noc_addr(src0_next_core_physical_x, src0_next_core_physical_y, l1_write_addr_in0);
                uint64_t src1_shift_to_addr_in1 = get_noc_addr(src1_next_core_physical_x, src1_next_core_physical_y, l1_write_addr_in1);

                // First, wait for all semaphore to be set, then write data tile to next core's CB

                uint64_t in0_sender_semaphore_noc_addr = get_noc_addr(src0_prev_core_physical_x, src0_prev_core_physical_y, in0_sender_semaphore_addr);
                uint64_t in1_sender_semaphore_noc_addr = get_noc_addr(src1_prev_core_physical_x, src1_prev_core_physical_y, in1_sender_semaphore_addr);
                // DPRINT << "Signal (" << src0_prev_core_physical_x << ", " << src0_prev_core_physical_y << ") to send data" << ENDL();
                noc_semaphore_inc(in0_sender_semaphore_noc_addr, 1);
                noc_semaphore_inc(in1_sender_semaphore_noc_addr, 1);
                // DPRINT << "Wait for signal from (" << src0_next_core_physical_x << ", " << src0_next_core_physical_y << ") to send data" << ENDL();
                noc_semaphore_wait(in0_sender_semaphore_addr_ptr, 1);
                // DPRINT << "Get signal from (" << src0_next_core_physical_x << ", " << src0_next_core_physical_y << ") to send data" << ENDL();
                noc_semaphore_set(in0_sender_semaphore_addr_ptr, 0);
                noc_semaphore_wait(in1_sender_semaphore_addr_ptr, 1);
                noc_semaphore_set(in1_sender_semaphore_addr_ptr, 0);
                noc_async_write(in0_start_addr, src0_shift_to_addr_in0, src0_tile_bytes * dram_shard_x * per_core_K);
                // DPRINT << "Send data" << ENDL();
                // DPRINT << "Current in0_start_addr=" << in0_start_addr << "; l1_write_addr_in0=" << l1_write_addr_in0 << "; The difference=" << l1_write_addr_in0 - in0_start_addr << ENDL();
                noc_async_write(in1_start_addr, src1_shift_to_addr_in1, src1_tile_bytes * per_core_K * dram_shard_y);
                noc_async_write_barrier();
                in0_start_addr = l1_write_addr_in0;
                in1_start_addr = l1_write_addr_in1;
                uint64_t in0_receiver_semaphore_noc_addr = get_noc_addr(src0_next_core_physical_x, src0_next_core_physical_y, in0_receiver_semaphore_addr);
                uint64_t in1_receiver_semaphore_noc_addr = get_noc_addr(src1_next_core_physical_x, src1_next_core_physical_y, in1_receiver_semaphore_addr);
                // DPRINT << "Signal (" << src0_next_core_physical_x << ", " << src0_next_core_physical_y << ") to receive data" << ENDL();
                noc_semaphore_inc(in0_receiver_semaphore_noc_addr, 1);
                noc_semaphore_inc(in1_receiver_semaphore_noc_addr, 1);
                // DPRINT << "Wait for signal from (" << src0_prev_core_physical_x << ", " << src0_prev_core_physical_y << ") to receive data" << ENDL();
                noc_semaphore_wait(in0_receiver_semaphore_addr_ptr, 1);
                // DPRINT << "Get signal from (" << src0_prev_core_physical_x << ", " << src0_prev_core_physical_y << ") to receive data" << ENDL();
                noc_semaphore_set(in0_receiver_semaphore_addr_ptr, 0);
                noc_semaphore_wait(in1_receiver_semaphore_addr_ptr, 1);
                noc_semaphore_set(in1_receiver_semaphore_addr_ptr, 0);
                // DPRINT << "Receive data" << ENDL();
                // DPRINT << TSLICE(tt::CB::c_in0, (shift_num+1)*(per_core_M*per_core_K), SliceRange::hw0_32_16(), TSLICE_INPUT_CB, TSLICE_RD_PTR, true, false) << ENDL();
                cb_push_back(noc_cb_A, src0_sharded_tiles);
                cb_push_back(noc_cb_B, src1_sharded_tiles);

                // DPRINT << "Signal prev core (" << src1_prev_core_physical_x << ", " << src1_prev_core_physical_y << ") to send data" << ENDL();
                // DPRINT << "Wait for signal to send data" << ENDL();
                // DPRINT << "Current in1_start_addr=" << in1_start_addr << "; l1_write_addr_in1=" << l1_write_addr_in1 << "; The difference=" << l1_write_addr_in1 - in1_start_addr << ENDL();
                // DPRINT << "Signal next core (" << src1_next_core_physical_x << ", " << src1_next_core_physical_y << ") to receive data" << ENDL();
                // DPRINT << "Wait for signal to receive data" << ENDL();
                // DPRINT << "Receive data" << ENDL();
                // DPRINT << TSLICE(tt::CB::c_in1, (shift_num+1)*(per_core_K*per_core_N), SliceRange::hw0_32_16(), TSLICE_INPUT_CB, TSLICE_RD_PTR, true, false) << ENDL();
            }
            
        }
        
        if (bcast_B == 0) {
            src1_start_tile_id += Kt * Nt;
        }
        src0_start_tile_id += Mt * Kt;
    }
}
