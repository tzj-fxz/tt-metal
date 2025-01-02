// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "compute_kernel_api/matmul.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "debug/dprint.h"
#include "tools/profiler/kernel_profiler.hpp"

namespace NAMESPACE {
void MAIN {
    uint32_t batch = get_compile_time_arg_val(0);
    uint32_t Mt = get_compile_time_arg_val(1);
    uint32_t Kt = get_compile_time_arg_val(2);
    uint32_t Nt = get_compile_time_arg_val(3);
    uint32_t round = get_compile_time_arg_val(4);

    mm_block_init(tt::CB::c_in0, tt::CB::c_in1, tt::CB::c_out0, false, Nt, Mt, Kt);

    cb_wait_front(tt::CB::c_in0, Mt * Kt);
    cb_wait_front(tt::CB::c_in1, Kt * Nt);
    tile_regs_acquire();

    // warmup
    for (uint32_t b = 0; b < 10; b++) {
        matmul_block(tt::CB::c_in0, tt::CB::c_in1, 0, 0, 0, false, Nt, Mt, Kt);
    }

    // test
    {
        DeviceZoneScopedN("TEST-bmm-block");

        for (uint32_t b = 0; b < round; b++) {
            matmul_block(tt::CB::c_in0, tt::CB::c_in1, 0, 0, 0, false, Nt, Mt, Kt);
        }
    }

    tile_regs_commit();
    cb_reserve_back(tt::CB::c_out0, Mt * Nt);
    tile_regs_wait();
    matmul_pack_tile(0, tt::CB::c_out0, Mt * Nt);
    tile_regs_release();
    cb_push_back(tt::CB::c_out0, Mt * Nt);
    // DPRINT << Mt << " " << Nt << ENDL();
    // DPRINT << "test" << ENDL();

    cb_pop_front(tt::CB::c_in0, Mt * Kt);
    cb_pop_front(tt::CB::c_in1, Kt * Nt);
}
}  // namespace NAMESPACE
