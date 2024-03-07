// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#define REDUCE_OP PoolType::SUM
#define REDUCE_DIM ReduceDim::REDUCE_ROW

#include "tt_eager/tt_dnn/kernels/compute/moreh_common.hpp"

namespace NAMESPACE {
void MAIN {
    constexpr uint32_t onetile = 1;

    constexpr auto cb_y = tt::CB::c_in0;
    constexpr auto cb_dy = tt::CB::c_in1;
    constexpr auto cb_bcast_scaler = tt::CB::c_in2;
    constexpr auto cb_mask = tt::CB::c_in3;
    constexpr auto cb_dx = tt::CB::c_out0;

    constexpr auto cb_ydy = tt::CB::c_intermed0;  // y * dy
    constexpr auto cb_sum = tt::CB::c_intermed1;
    constexpr auto cb_inter2 = tt::CB::c_intermed2;

    binary_op_init_common(cb_y, cb_bcast_scaler);

    uint32_t N = get_compile_time_arg_val(0);
    uint32_t Wt = get_compile_time_arg_val(1);

    for (uint32_t n = 0; n < N; ++n) {

        #ifdef LOG
            // sum(dy)
            if (Wt == 1) {
                // apply mask
                ACQ();
                mask_tile_to_cb(cb_dy, cb_mask, cb_inter2, /*itile=*/0, /*mtile=*/0, /*pop=*/0, /*popm=*/0);
                REL();

                ACQ();
                reduce_tile_to_cb(REDUCE_OP, REDUCE_DIM, cb_inter2, cb_bcast_scaler, cb_sum, 1, /*pop0=*/1, /*pop=1*/0);
                REL();
            } else {
                constexpr auto cb_inter0 = tt::CB::c_intermed0;
                ACQ();
                reduce_tile_to_cb(REDUCE_OP, REDUCE_DIM, cb_dy, cb_bcast_scaler, cb_inter0, Wt - 1, /*pop0=*/0, /*pop=1*/0);
                REL();

                constexpr auto cb_inter1 = tt::CB::c_intermed1;
                ACQ();
                mask_tile_to_cb(cb_dy, cb_mask, cb_inter1, /*itile=*/Wt - 1, /*mtile=*/0, /*pop=*/0, /*popm=*/0);
                REL();

                constexpr auto cb_inter2 = tt::CB::c_intermed2;
                ACQ();
                reduce_tile_to_cb(REDUCE_OP, REDUCE_DIM, cb_inter1, cb_bcast_scaler, cb_inter2, 1, /*pop0=*/1, /*pop=1*/0);
                REL();

                ACQ();
                add_tiles_to_cb(cb_inter0, cb_inter2, cb_sum);
                REL();
            }


            // dy - sum * exp(y)
            constexpr auto cb_exp = tt::CB::c_intermed0;  // y * dy

            for (uint32_t w = 0; w < Wt; w += onetile) {
                // exp(y)
                ACQ();
                exp_tile_to_cb(cb_y, cb_exp, w, /*dst=*/0, /*pop=*/0);
                REL();

                // sum * exp(y)
                ACQ();
                mul_tiles_bcast_cols_to_cb(cb_exp, cb_sum, cb_inter2, 0, 0, /*pop0=*/1, /*pop1=*/0);
                REL();

                // dy - sum * exp(y)
                ACQ();
                sub_tiles_to_cb(cb_dy, cb_inter2, cb_dx, w, 0, /*pop0=*/0, /*pop1=*/1);
                REL();
            }

            cb_pop_front(cb_sum, onetile);
            cb_pop_front(cb_y, Wt);
            cb_pop_front(cb_dy, Wt);
        #else
            // step 1, compute y * dy
            for (uint32_t w = 0; w < Wt; ++w) {
                ACQ();
                if (w == Wt - 1) {
                    mul_tiles_and_mask_tile_to_cb(
                        cb_y, cb_dy, cb_mask, cb_ydy, w, w, 0, /*pop0=*/0, /*pop1=*/0, /*popm=*/0);
                } else {
                    mul_tiles_to_cb(cb_y, cb_dy, cb_ydy, w, w, /*pop0=*/0, /*pop1=*/0);
                }
                REL();
            }

            // step 2, compute sum(y * dy)
            ACQ();
            reduce_tile_to_cb(REDUCE_OP, REDUCE_DIM, cb_ydy, cb_bcast_scaler, cb_sum, Wt, /*pop0=*/Wt, /*pop=1*/0);
            REL();

            // step 3, compute final result
            for (uint32_t w = 0; w < Wt; w += onetile) {
                // dy - sum
                ACQ();
                sub_tiles_bcast_cols_to_cb(cb_dy, cb_sum, cb_inter2, w, 0, /*pop0=*/0, /*pop1=*/0);
                REL();

                ACQ();
                #ifdef SOFTMAX
                    // (dy - sum) * y
                    mul_tiles_to_cb(cb_y, cb_inter2, cb_dx, w, 0, /*pop0=*/0, /*pop1=*/1);
                #else
                    // -(dy - sum) * y
                    mul_tiles_and_negative_to_cb(cb_y, cb_inter2, cb_dx, w, 0, /*pop0=*/0, /*pop1=*/1);
                #endif
                REL();
            }

            cb_pop_front(cb_sum, onetile);
            cb_pop_front(cb_dy, Wt);
            cb_pop_front(cb_y, Wt);
        #endif
    }
}
}  // namespace NAMESPACE
