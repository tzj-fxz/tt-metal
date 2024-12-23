// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"
#include "hostdevcommon/common_values.hpp"
#include "tools/profiler/kernel_profiler.hpp"
#include "debug/dprint.h"

void kernel_main() {
    DeviceZoneScopedN("TEST-reader_bmm_cannon_all");

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

    // first read: to dataflow CB
    // uint32_t l1_write_addr_dataflow0;
    // uint32_t l1_write_addr_dataflow1;
    uint32_t l1_write_addr_in0;
    uint32_t l1_write_addr_in1;
    // TODO currenty batch = 1
    for (uint32_t nbatch = 0; nbatch < batch; ++nbatch) {
        cb_reserve_back(tt::CB::c_in0, src0_block_tiles);
        cb_reserve_back(tt::CB::c_in1, src1_block_tiles);
        l1_write_addr_in0 = get_write_ptr(tt::CB::c_in0);
        l1_write_addr_in1 = get_write_ptr(tt::CB::c_in1);
        uint32_t in0_start_addr = l1_write_addr_in0;
        uint32_t in1_start_addr = l1_write_addr_in1;
        // currently only support square, means per_core_M = per_core_N (= per_core_K)
        // now, we manually skew src_start_tile_id to let reader kernel read the correct initial block.
        src0_start_tile_id = core_x * Kt * per_core_M + ((core_y + core_x) % num_block_y) * per_core_K;
        for (uint32_t h = 0; h < per_core_M; ++h) {
            for (uint32_t w = 0; w < per_core_K; ++w) {
                noc_async_read_tile(src0_start_tile_id + h * Kt + w, s0, l1_write_addr_in0);
                l1_write_addr_in0 += src0_tile_bytes;
            }
        }
        src1_start_tile_id = ((core_x + core_y) % num_block_x) * Nt * per_core_K + core_y * per_core_N;
        for (uint32_t h = 0; h < per_core_K; ++h) {
            for (uint32_t w = 0; w < per_core_N; ++w) {
                noc_async_read_tile(src1_start_tile_id + h * Nt + w, s1, l1_write_addr_in1);
                l1_write_addr_in1 += src1_tile_bytes;
            }
        }
        noc_async_read_barrier();
        cb_push_back(tt::CB::c_in0, src0_block_tiles);
        cb_push_back(tt::CB::c_in1, src1_block_tiles);
        DPRINT << TSLICE(tt::CB::c_in0, 0, SliceRange::hw0_32_16(), TSLICE_INPUT_CB, TSLICE_RD_PTR, true, false) << ENDL();
        DPRINT << TSLICE(tt::CB::c_in1, 0, SliceRange::hw0_32_16(), TSLICE_INPUT_CB, TSLICE_RD_PTR, true, false) << ENDL();

        // the address of truly skewed tiles in "in" CB
        // cb_reserve_back(tt::CB::c_in0, src0_block_tiles);
        // cb_reserve_back(tt::CB::c_in1, src1_block_tiles);
        // l1_write_addr_in0 = get_write_ptr(tt::CB::c_in0);
        // l1_write_addr_in1 = get_write_ptr(tt::CB::c_in1);
        // uint32_t in0_start_addr = l1_write_addr_in0;
        // uint32_t in1_start_addr = l1_write_addr_in1;


        // // initially skew
        // // Consider src0 and src1 separately, be careful of deadlock and L1 memory conflict
        // uint32_t src0_skew_to_x = core_x;
        // uint32_t src0_skew_to_y = (num_block_y + core_y - core_x) % num_block_y;
        // uint64_t src0_skew_to_addr_in0 = get_noc_addr(src0_skew_to_x, src0_skew_to_y, l1_write_addr_in0);
        // uint32_t src0_skew_from_y = (num_block_y + core_y + core_x) % num_block_y;
        // // First data in "dataflow" CB, the skewed data to send is in "in" CB
        // // May do not need to check if src0_skew_to_x != src0_skew_from_y, for the src CB != dst CB
        // noc_async_write(in0_start_addr, src0_skew_to_addr_in0, src0_tile_bytes * per_core_M * per_core_K);
        // l1_write_addr_in0 += src0_tile_bytes * per_core_M * per_core_K;
        // noc_async_write_barrier();

        // uint32_t src1_skew_to_x = (num_block_x + core_x - core_y) % num_block_x;
        // uint32_t src1_skew_to_y = core_y;
        // uint64_t src1_skew_to_addr_in1 = get_noc_addr(src1_skew_to_x, src1_skew_to_y, l1_write_addr_in1);
        // uint32_t src1_skew_from_x = (num_block_x + core_x + core_y) % num_block_x;
        // noc_async_write(in1_start_addr, src1_skew_to_addr_in1, src1_tile_bytes * per_core_K * per_core_N);
        // l1_write_addr_in1 += src1_tile_bytes * per_core_K * per_core_N;
        // noc_async_write_barrier();

        // // Then push original block and skewed block into CB for compute kernel
        // // for compute kernel, should get skewed block
        // cb_push_back(tt::CB::c_in0, src0_block_tiles);
        // cb_push_back(tt::CB::c_in1, src1_block_tiles);
        // cb_pop_front(tt::CB::c_in0, src0_block_tiles);
        // cb_pop_front(tt::CB::c_in1, src1_block_tiles);
        if (bcast_B == 0) {
            src1_start_tile_id += Kt * Nt;
        }
        src0_start_tile_id += Mt * Kt;
        // in cannon process, do not need to read one block from neighbor compute kernel
        // instead, directly write tile in current core into next core (shifted)
        // pay attention to the CB capacity: we first write into CB, then pop from CB, and finally push into CB for compute kernel to process
        // DPRINT << "src0 to core: " << src0_next_core_physical_x << ", " << src0_next_core_physical_y << ENDL();
        // DPRINT << "src1 to core: " << src1_next_core_physical_x << ", " << src1_next_core_physical_y << ENDL();

        for (uint32_t shift_num = 0; shift_num < Mt / per_core_M - 1; ++shift_num) {
            DeviceZoneScopedN("TEST-reader_bmm_cannon_shift");
            cb_reserve_back(tt::CB::c_in0, src0_block_tiles);
            cb_reserve_back(tt::CB::c_in1, src1_block_tiles);
            l1_write_addr_in0 = get_write_ptr(tt::CB::c_in0);
            l1_write_addr_in1 = get_write_ptr(tt::CB::c_in1);
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
            noc_async_write(in0_start_addr, src0_shift_to_addr_in0, src0_tile_bytes * per_core_M * per_core_K);
            in0_start_addr = l1_write_addr_in0;
            l1_write_addr_in0 += src0_tile_bytes * per_core_M * per_core_K;
            // DPRINT << "Send data" << ENDL();
            // DPRINT << "Current in0_start_addr=" << in0_start_addr << "; l1_write_addr_in0=" << l1_write_addr_in0 << "; The difference=" << l1_write_addr_in0 - in0_start_addr << ENDL();
            noc_async_write(in1_start_addr, src1_shift_to_addr_in1, src1_tile_bytes * per_core_K * per_core_N);
            in1_start_addr = l1_write_addr_in1;
            l1_write_addr_in1 += src1_tile_bytes * per_core_K * per_core_N;
            noc_async_write_barrier();
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
            cb_push_back(tt::CB::c_in0, src0_block_tiles);
            cb_push_back(tt::CB::c_in1, src1_block_tiles);

            // DPRINT << "Signal prev core (" << src1_prev_core_physical_x << ", " << src1_prev_core_physical_y << ") to send data" << ENDL();
            // DPRINT << "Wait for signal to send data" << ENDL();
            // DPRINT << "Current in1_start_addr=" << in1_start_addr << "; l1_write_addr_in1=" << l1_write_addr_in1 << "; The difference=" << l1_write_addr_in1 - in1_start_addr << ENDL();
            // DPRINT << "Signal next core (" << src1_next_core_physical_x << ", " << src1_next_core_physical_y << ") to receive data" << ENDL();
            // DPRINT << "Wait for signal to receive data" << ENDL();
            // DPRINT << "Receive data" << ENDL();
            // DPRINT << TSLICE(tt::CB::c_in1, (shift_num+1)*(per_core_K*per_core_N), SliceRange::hw0_32_16(), TSLICE_INPUT_CB, TSLICE_RD_PTR, true, false) << ENDL();


            // uint64_t in0_receiver_semaphore_noc_addr = get_noc_addr(src0_next_core_physical_x, src0_next_core_physical_y, in0_receiver_semaphore_addr);
            // volatile tt_l1_ptr uint32_t* in0_receiver_semaphore_noc_addr_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(in0_receiver_semaphore_noc_addr);
            // noc_semaphore_wait(in0_receiver_semaphore_noc_addr_ptr, shift_num + 1);
            // noc_async_write(in0_start_addr, src0_shift_to_addr_in0, src0_tile_bytes * per_core_M * per_core_K);
            // in0_start_addr += src0_tile_bytes * per_core_M * per_core_K;
            // l1_write_addr_in0 += src0_tile_bytes * per_core_M * per_core_K;
            // noc_async_write_barrier();
            // noc_semaphore_inc(in0_receiver_semaphore_noc_addr, 1);
            // noc_semaphore_set(in0_sender_semaphore_addr_ptr, 0);
            // DPRINT << "in0 receiver src: " << src0_next_core_physical_x << ", " << src0_next_core_physical_y << " semaphore noc: " << *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(in0_receiver_semaphore_noc_addr) << ENDL();

            // uint64_t in1_receiver_semaphore_noc_addr = get_noc_addr(src1_next_core_physical_x, src1_next_core_physical_y, in1_receiver_semaphore_addr);
            // volatile tt_l1_ptr uint32_t* in1_receiver_semaphore_noc_addr_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(in1_receiver_semaphore_noc_addr);
            // noc_semaphore_wait(in1_receiver_semaphore_noc_addr_ptr, shift_num + 1);
            // noc_async_write(in1_start_addr, src1_shift_to_addr_in1, src1_tile_bytes * per_core_K * per_core_N);
            // in1_start_addr += src1_tile_bytes * per_core_K * per_core_N;
            // l1_write_addr_in1 += src1_tile_bytes * per_core_K * per_core_N;
            // noc_async_write_barrier();
            // noc_semaphore_inc(in1_receiver_semaphore_noc_addr, 1);
            // noc_semaphore_set(in1_sender_semaphore_addr_ptr, 0);
            // DPRINT << "in1 receiver src: " << src1_next_core_physical_x << ", " << src1_next_core_physical_y << " semaphore noc: " << *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(in1_receiver_semaphore_noc_addr) << ENDL();
 
            // // Then, wait for previous core to finish writing data to our CB
            // DPRINT << "Wait for previous core to finish writing data to our CB" << ENDL();
            // noc_semaphore_wait(in0_receiver_semaphore_addr_ptr, shift_num + 2);
            // DPRINT << "in0 receiver src: " << core_x << ", " << core_y << " semaphore: " << *in0_receiver_semaphore_addr_ptr << ENDL();
            // // noc_semaphore_set(in0_receiver_semaphore_addr_ptr, 0);
            // noc_semaphore_wait(in1_receiver_semaphore_addr_ptr, shift_num + 2);
            // DPRINT << "in1 receiver src: " << core_x << ", " << core_y << " semaphore: " << *in1_receiver_semaphore_addr_ptr << ENDL();
            // // noc_semaphore_set(in1_receiver_semaphore_addr_ptr, 0);

            // // pop and push for compute kernel
            // DPRINT << "Push to compute kernel" << ENDL();
            // cb_push_back(tt::CB::c_in0, src0_block_tiles);
            // cb_push_back(tt::CB::c_in1, src1_block_tiles);

            // // the following semaphore should be set from compute kernel when finishing computing
            // DPRINT << "After push to CB" << ENDL();
            // noc_semaphore_set(in0_sender_semaphore_addr_ptr, 1);
            // noc_semaphore_set(in1_sender_semaphore_addr_ptr, 1);
        }
    }
}
