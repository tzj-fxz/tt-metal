// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"

#include "debug/dprint.h"
#include "tools/profiler/kernel_profiler.hpp"

void kernel_main() {
    // DeviceZoneScopedN("TEST-writer-bmm-8bank");
    // same arg indices as in reader_bmm_8bank for reuse
    uint32_t dst_addr   = get_arg_val<uint32_t>(0);
    uint32_t Mt         = get_arg_val<uint32_t>(2);
    uint32_t Nt         = get_arg_val<uint32_t>(4);
    uint32_t batch      = get_arg_val<uint32_t>(7);

    constexpr bool dst_is_dram = get_compile_time_arg_val(0) == 1;

    constexpr uint32_t cb_id_out0 = tt::CB::c_out0;
    const uint32_t tile_bytes = get_tile_size(cb_id_out0);
    uint32_t itileC = 0;
    const DataFormat data_format = get_dataformat(cb_id_out0);

    const InterleavedAddrGenFast<dst_is_dram> s = {
        .bank_base_address = dst_addr,
        .page_size = tile_bytes,
        .data_format = data_format
    };

    // DPRINT << "test" << ENDL();
    cb_wait_front(cb_id_out0, Mt * Nt);
    // DPRINT << TSLICE(tt::CB::c_out0, 0, SliceRange::hw0_32_4(), TSLICE_INPUT_CB, TSLICE_RD_PTR, true, false) << ENDL();

    // C is MN so we iterate in tile RM order
    for (uint32_t mt_C = 0; mt_C < Mt; ++mt_C)   // output tile of C
    for (uint32_t nt_C = 0; nt_C < Nt; ++nt_C) { // output tile index of C
        // bmm will generate C's tiles C=A*B, MN=MK*KN, in row major order, we just read them from CB and write out to DRAM
        uint32_t l1_read_addr = get_read_ptr(cb_id_out0);
        noc_async_write_tile(itileC, s, l1_read_addr);
        noc_async_write_barrier(); // This will wait until the write is done. As an alternative,
                                   // noc_async_write_flushed() can be faster because it waits
                                   // until the write request is sent. In that case, you have to
                                   // use noc_async_write_barrier() at least once at the end of
                                   // data movement kernel to make sure all writes are done.
        //DPRINT << 'W' << 'C' << itileC << ' ' << 'a' << dst_addr << ENDL();
        //DPRINT << itileC << ' ' << uint32_t(dst_noc_addr) << ENDL();
        itileC ++;
    }

    cb_pop_front(cb_id_out0, Mt * Nt);
}
