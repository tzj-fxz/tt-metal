// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/matmul.h"
#include "tools/profiler/kernel_profiler.hpp"
#include "debug/dprint.h"

using std::uint32_t;

// matmul C=A*B using dims MK*KN = MN (row major order)
//
namespace NAMESPACE {
void MAIN {

    uint32_t single_tile_size = get_compile_time_arg_val(0);
    uint32_t cb_tiles = get_compile_time_arg_val(1);
    uint32_t repeat = get_compile_time_arg_val(2);
    DeviceZoneScopedN("TEST-NoC-compute");

    for (uint32_t i = 0; i < repeat; ++i) {
        cb_wait_front(tt::CB::c_in0, cb_tiles);
    }
    // for (uint32_t tile = 0; tile < cb_tiles; ++tile) {
    //     copy_tile_init();
    //     tile_regs_acquire();
    //     copy_tile(tt::CB::c_in0, 0, 0);
    //     tile_regs_commit();
    //     cb_reserve_back(tt::CB::c_out0, 1);
    //     tile_regs_wait();
    //     pack_tile(0, tt::CB::c_out0);
    //     tile_regs_release();
    //     cb_push_back(tt::CB::c_out0, 1);
    // }
    // cb_pop_front(tt::CB::c_in0, cb_tiles);
}
} // NAMESPACE
