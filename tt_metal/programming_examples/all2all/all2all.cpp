// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/host_api.hpp"
#include "tt_metal/common/constants.hpp"
#include "tt_metal/detail/util.hpp"
#include "tt_metal/common/bfloat16.hpp"
#include "tt_metal/impl/dispatch/command_queue.hpp"
#include "tt_metal/detail/tt_metal.hpp"
#include "tt_metal/impl/device/device.hpp"

using namespace tt;
using namespace tt::tt_metal;

int main(int argc, char **argv) {
    TT_FATAL(argc == 5, "Expected 5 arguments (M, N, core_x, core_y) but got {}", argc);
    uint32_t M = atoi(argv[1]);
    uint32_t N = atoi(argv[2]);
    uint32_t core_x = atoi(argv[3]);
    uint32_t core_y = atoi(argv[4]);
    // get program and device 
    int device_id = 0;
    Device *device = CreateDevice(device_id);
    CommandQueue& cq = device->command_queue();
    Program program = CreateProgram();
    // get cores
    CoreCoord start_core = {0, 0};
    CoreCoord end_core = {core_x - 1, core_y - 1};
    CoreRange cores(start_core, end_core);
    // get sharded arguments
    uint32_t num_values = M * N;
    uint32_t Mt = M / TILE_HEIGHT;
    uint32_t Nt = N / TILE_WIDTH;
    TT_FATAL(Mt % core_x == 0, "M must be divisible by core_x");
    TT_FATAL(Nt % core_y == 0, "N must be divisible by core_y");
    uint32_t sharded_height = Mt / core_x;
    uint32_t sharded_width = Nt / core_y;
    TT_FATAL(sharded_height % core_x == 0, "sharded_height must be divisible by core_x");
    TT_FATAL(sharded_width % core_y == 0, "sharded_width must be divisible by core_y");
    // get data
    tt::DataFormat data_format = tt::DataFormat::Float16_b;
    uint32_t single_tile_size = sizeof(bfloat16) * TILE_HEIGHT * TILE_WIDTH;
    uint32_t dram_buffer_size = single_tile_size * Mt * Nt;
    std::vector<bfloat16> src_vec = create_random_vector_of_bfloat16_native(dram_buffer_size, 1, 123);
    std::vector<bfloat16> result_vec(dram_buffer_size/sizeof(bfloat16));
    // create dram buffer
    tt_metal::InterleavedBufferConfig dram_config {
        .device = device,
        .size = dram_buffer_size,
        .page_size = single_tile_size,
        .buffer_type = tt_metal::BufferType::DRAM
    };
    std::shared_ptr<tt::tt_metal::Buffer> src_dram_buffer = CreateBuffer(dram_config);
    std::shared_ptr<tt::tt_metal::Buffer> dst_dram_buffer = CreateBuffer(dram_config);
    uint32_t src_dram_addr = src_dram_buffer->address();
    uint32_t dst_dram_addr = dst_dram_buffer->address();
    // create circular buffers
    CircularBufferConfig cb_config_in0 = CircularBufferConfig(single_tile_size, {{tt::CB::c_in0, data_format}})
		.set_page_size(tt::CB::c_in0, single_tile_size);
    auto cb_input_0 = tt_metal::CreateCircularBuffer(program, cores, cb_config_in0);
    CircularBufferConfig cb_config_out0 = CircularBufferConfig(single_tile_size, {{tt::CB::c_out0, data_format}})
		.set_page_size(tt::CB::c_out0, single_tile_size);
    auto cb_output = tt_metal::CreateCircularBuffer(program, cores, cb_config_out0);

    // create reader kernel
    std::vector<uint32_t> reader_compile_time_args = {
        (uint32_t)dram_addr,
        (uint32_t)single_tile_size,
        (uint32_t)sharded_height,
        (uint32_t)sharded_width,
        (uint32_t)Mt,
        (uint32_t)Nt,
        (uint32_t)core_x,
        (uint32_t)core_y
    };
    auto reader_id = tt_metal::CreateKernel(
        program,
        "tt_metal/programming_examples/all2all/kernels/reader.cpp",
        cores,
        tt_metal::DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default, .compile_args = reader_compile_time_args});
    // create writer kernel
    std::vector<uint32_t> writer_compile_time_args = {
        (uint32_t)dram_addr,
        (uint32_t)single_tile_size,
        (uint32_t)sharded_height,
        (uint32_t)sharded_width,
        (uint32_t)Mt,
        (uint32_t)Nt
    };
    auto writer_id = tt_metal::CreateKernel(
        program,
        "tt_metal/programming_examples/all2all/kernels/writer.cpp",
        cores,
        tt_metal::DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default, .compile_args = writer_compile_time_args});

    // set runtime arguments
    for (uint32_t i = 0; i < core_x; i++) {
        for (uint32_t j = 0; j < core_y; j++) {
            CoreCoord core = {i, j};
            uint32_t curr_idx_h = i * sharded_height;
            uint32_t curr_idx_w = j * sharded_width;
            tt_metal::SetRuntimeArgs(program, reader_id, core, {curr_idx_h, curr_idx_w, i, j});
            tt_metal::SetRuntimeArgs(program, writer_id, core, {i, j});
        }
    }

    EnqueueWriteBuffer(cq, src_dram_buffer, src_vec.data(), false);
    EnqueueProgram(cq, program, false);
    EnqueueReadBuffer(cq, dst_dram_buffer, result_vec.data(), false);

    // Compare src and dst vectors
    bool vectors_match = true;
    for (size_t i = 0; i < src_vec.size(); i++) {
        if (src_vec[i] != result_vec[i]) {
            vectors_match = false;
            std::cout << "Mismatch at index " << i << ": src=" << src_vec[i] << " dst=" << result_vec[i] << std::endl;
            break;
        }
    }

    if (vectors_match) {
        std::cout << "All2All test PASSED - src and dst vectors match" << std::endl;
    } else {
        std::cout << "All2All test FAILED - src and dst vectors differ" << std::endl;
    }
    float pearson = check_bfloat16_vector_pcc(golden_vec, result_vec);
    log_info(tt::LogVerif, "Metalium vs Golden -- PCC = {}", pearson);
    TT_FATAL(pearson > 0.98, "PCC not high enough. Result PCC: {}, Expected PCC: 0.98", pearson);

    Finish(cq);
    CloseDevice(device);
}
