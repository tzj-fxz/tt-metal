#include <stdint.h>
#include "dataflow_kernel_api.h"

void kernel_main() {
    // READER RUNTIME ARGS
    uint32_t in0_tensor_addr                     = get_arg_val<uint32_t>(0);
    uint32_t in0_tensor_tile_id                  = get_arg_val<uint32_t>(1);

    // COMPILE TIME ARGS
    // dataflow::Interleaved accessor args
    constexpr uint32_t in0_is_dram               = get_compile_time_arg_val(1);
    // READER COMPILE TIME ARGS
    constexpr uint32_t out_w_tiles               = get_compile_time_arg_val(2);
    constexpr uint32_t out_c                     = get_compile_time_arg_val(3);


    constexpr uint32_t cb_id_in0 = 0;
    uint32_t single_tile_size_bytes = get_tile_size(cb_id_in0);

    constexpr bool in0_is_dram_bool = in0_is_dram == 1;
    #define tile_dtype_is_bfloat16 get_compile_time_arg_val(0) == 1
    #if (tile_dtype_is_bfloat16)
    const dataflow::InterleavedAddrGenFast<in0_is_dram_bool> s0 = {
        .bank_base_address = in0_tensor_addr,
        .page_size = single_tile_size_bytes,
        .data_format = DataFormat::Float16
    };
    #else
    const dataflow::InterleavedAddrGenFast<in0_is_dram_bool> s0 = {
        .bank_base_address = in0_tensor_addr,
        .page_size = single_tile_size_bytes,
        .data_format = DataFormat::Bfp8_b
    };
    #endif

    uint32_t l1_write_addr_in0 = dataflow::get_write_ptr(cb_id_in0);
    for (uint32_t c_dim = 0; c_dim < out_c; c_dim++) {
        dataflow::cb_reserve_back(cb_id_in0, out_w_tiles);
        for (uint32_t w_dim = 0; w_dim < out_w_tiles; w_dim++) {
            dataflow::noc_async_read_tile(in0_tensor_tile_id, s0, l1_write_addr_in0);
            l1_write_addr_in0 += single_tile_size_bytes;
            in0_tensor_tile_id++;
        }
        dataflow::noc_async_read_barrier();
        dataflow::cb_push_back(cb_id_in0, out_w_tiles);
    }
}
