// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "debug/dprint.h"  // required in all kernels using DPRINT
#include "dataflow_api.h"

void kernel_main() {

        // Nothing to move. Print respond message.
        // Make sure to export TT_METAL_DPRINT_CORES=0,0 before runtime.

    for (int i = 0; i < (1 << 6); i ++)
    {
        DeviceZoneScopedN("TEST-FULL");
//Max unroll size
#pragma GCC unroll 65534
        for (int j = 0 ; j < (1 << 5); j++)
        {
            asm("nop");
        }
    }

        DPRINT_DATA0(DPRINT << "Hello, Master, I am running a void data movement kernel on NOC 0." << ENDL(););
        DPRINT_DATA1(DPRINT << "Hello, Master, I am running a void data movement kernel on NOC 1." << ENDL(););

}
