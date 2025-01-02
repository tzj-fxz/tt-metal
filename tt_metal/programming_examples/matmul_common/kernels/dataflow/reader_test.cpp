// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"

#include "debug/dprint.h"
#include "tools/profiler/kernel_profiler.hpp"

void kernel_main() {

    // same arg indices as in reader_binary_diff_lenghts for compat
    uint32_t src0_addr  = get_arg_val<uint32_t>(0);
    uint32_t src1_addr  = get_arg_val<uint32_t>(1);
    uint32_t Mt         = get_arg_val<uint32_t>(2);
    uint32_t Kt         = get_arg_val<uint32_t>(3);
    uint32_t Nt         = get_arg_val<uint32_t>(4);
    uint32_t MtKt       = get_arg_val<uint32_t>(5); // if 0
    uint32_t KtNt       = get_arg_val<uint32_t>(6);
    uint32_t batch      = get_arg_val<uint32_t>(7);
    uint32_t bcast_B    = get_arg_val<uint32_t>(8); // if 1 we broadcast B to batch

    constexpr bool src0_is_dram = get_compile_time_arg_val(0) == 1;
    constexpr bool src1_is_dram = get_compile_time_arg_val(1) == 1;

    //DPRINT << "Mt=" << Mt << " Kt=" << Kt << " Nt=" << Nt << " MtKt=" << MtKt << "KtNt=" << KtNt << ENDL();
    //DPRINT << "src0=" << src0_addr << " src1=" << src1_addr << ENDL();
    //DPRINT << "batch=" << batch << ENDL();

    constexpr uint32_t cb_id_in0 = tt::CB::c_in0;
    constexpr uint32_t cb_id_in1 = tt::CB::c_in1;

    constexpr uint32_t onetile = 1;
    const uint32_t src0_tile_bytes = get_tile_size(cb_id_in0);
    const DataFormat src0_data_format = get_dataformat(cb_id_in0);
    const uint32_t src1_tile_bytes = get_tile_size(cb_id_in1);
    const DataFormat src1_data_format = get_dataformat(cb_id_in1);

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

    cb_reserve_back(cb_id_in0, Mt * Kt);
    uint32_t l1_write_addr_in0 = get_write_ptr(cb_id_in0);
    for (uint32_t m = 0; m < Mt; ++m) {
        for (uint32_t k = 0; k < Kt; ++k) {
            noc_async_read_tile(m * Kt + k, s0, l1_write_addr_in0);
            l1_write_addr_in0 += src0_tile_bytes;
        }
    }
    noc_async_read_barrier();
    // DPRINT << TSLICE(tt::CB::c_in0, 0, SliceRange::hw0_32_4(), TSLICE_INPUT_CB, TSLICE_RD_PTR, true, false) << ENDL();

    cb_reserve_back(cb_id_in1, Kt * Nt);
    uint32_t l1_write_addr_in1 = get_write_ptr(cb_id_in1);
    for (uint32_t k = 0; k < Kt; ++k) {
        for (uint32_t n = 0; n < Nt; ++n) {
            noc_async_read_tile(k * Nt + n, s1, l1_write_addr_in1);
            l1_write_addr_in1 += src1_tile_bytes;
        }
    }
    noc_async_read_barrier();

    cb_push_back(cb_id_in0, Mt * Kt);
    cb_push_back(cb_id_in1, Kt * Nt);

}
