#include <cstdint>

#include "llk_3c.h"

namespace NAMESPACE {
void MAIN {
    uint32_t w = 0;
    constexpr uint32_t onetile = 1;
    uint32_t B = get_compile_time_arg_val(0);
    uint32_t Ht = get_compile_time_arg_val(1);
    uint32_t Wt = get_compile_time_arg_val(2);

    init_bcast(CB::c_in0, CB::c_in1);

    for (uint32_t b = 0; b < B; b++) {
    for (uint32_t h = 0; h < Ht; h++) {
        cb_wait_front(CB::c_in1, onetile);
        for (uint32_t w = 0; w < Wt; w++) {

            cb_reserve_back(CB::c_out0, onetile);

            acquire_dst(DstMode::Half);

            cb_wait_front(CB::c_in0, onetile);
            BCAST_OP(tt::Dim::C, CB::c_in0, CB::c_in1, 0, 0, 0);
            pack_tile(0, CB::c_out0);
            cb_pop_front(CB::c_in0, onetile);

            release_dst(DstMode::Half);

            cb_push_back(CB::c_out0, onetile);

        }
        cb_pop_front(CB::c_in1, onetile);
    }}
}
} // NAMESPACE
