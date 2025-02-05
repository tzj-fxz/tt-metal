// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "compute_kernel_api/eltwise_binary.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "ttnn/cpp/ttnn/deprecated/tt_dnn/kernels/compute/moreh_common.hpp"

namespace NAMESPACE {
void MAIN {
    uint32_t num_tiles = get_arg_val<uint32_t>(0);
    constexpr uint32_t old_running_mean_has_value = get_compile_time_arg_val(0) == 1;
    constexpr uint32_t old_running_var_has_value = get_compile_time_arg_val(1) == 1;

    constexpr auto cb_batch_mean = tt::CBIndex::c_0;  // batch mean
    constexpr auto cb_batch_var = tt::CBIndex::c_1;   // batch var
    constexpr auto cb_out0 = tt::CBIndex::c_2;
    constexpr auto cb_old_running_mean = tt::CBIndex::c_3;       // old running mean tensor
    constexpr auto cb_old_running_var = tt::CBIndex::c_4;        // old running var tensor
    constexpr auto cb_updated_running_mean = tt::CBIndex::c_27;  // updated running mean tensor
    constexpr auto cb_updated_running_var = tt::CBIndex::c_28;   // updated running var tensor
    constexpr auto cb_momentum = tt::CBIndex::c_5;               // momentum
    constexpr auto cb_one = tt::CBIndex::c_6;                    // stores 1
    constexpr auto cb_tmp1 = tt::CBIndex::c_21;                  // tmp 1
    constexpr auto cb_tmp2 = tt::CBIndex::c_22;                  // tmp 2
    constexpr auto cb_tmp3 = tt::CBIndex::c_23;                  // tmp 3

    binary_op_init_common(cb_batch_mean, cb_batch_var, cb_out0);
    constexpr uint32_t onetile = 1;

    for (uint32_t tile_id = 0; tile_id < num_tiles; ++tile_id) {
        tile_regs_acquire();
        // updated_running_stat = (1 − momentum) × running_stat + momentum × batch_stat
        cb_wait_front(cb_one, 1);
        cb_wait_front(cb_momentum, 1);

        if constexpr (old_running_mean_has_value) {
            sub_tiles_to_cb(cb_one, cb_momentum, cb_tmp1, 0, 0, 0, 0);               // 1 - momentum
            mul_tiles_to_cb(cb_momentum, cb_batch_mean, cb_tmp2, 0, 0, 0, 1);        // momentum * batch stat
            mul_tiles_to_cb(cb_tmp1, cb_old_running_mean, cb_tmp3, 0, 0, 1, 1);      // cb_tmp1 * running stats
            add_tiles_to_cb(cb_tmp2, cb_tmp3, cb_updated_running_mean, 0, 0, 1, 1);  // cb_tmp2 + cb_tmp3
        }
        if constexpr (old_running_var_has_value) {
            sub_tiles_to_cb(cb_one, cb_momentum, cb_tmp1, 0, 0, 0, 0);              // 1 - momentum
            mul_tiles_to_cb(cb_momentum, cb_batch_var, cb_tmp2, 0, 0, 0, 1);        // momentum * batch stat
            mul_tiles_to_cb(cb_tmp1, cb_old_running_var, cb_tmp3, 0, 0, 1, 1);      // cb_tmp1 * running stats
            add_tiles_to_cb(cb_tmp2, cb_tmp3, cb_updated_running_var, 0, 0, 1, 1);  // cb_tmp2 + cb_tmp3
        }
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, cb_out0);
        tile_regs_release();
        cb_pop_front(cb_one, 1);
        cb_pop_front(cb_momentum, 1);
        cb_push_back(cb_out0, 1);
    }
}
}  // namespace NAMESPACE
