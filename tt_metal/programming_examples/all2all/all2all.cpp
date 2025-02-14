// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/host_api.hpp"
#include "tt_metal/common/constants.hpp"
#include "tt_metal/detail/util.hpp"
#include "tt_metal/common/bfloat16.hpp"
#include "tt_metal/common/test_tiles.hpp"
#include "tt_metal/impl/dispatch/command_queue.hpp"
#include "tt_metal/common/tilize_untilize.hpp"
#include "impl/device/device.hpp"
#include "tt_metal/detail/tt_metal.hpp"
#include <chrono>

using namespace tt;
using namespace tt::tt_metal;

int main(int argc, char **argv) {
    TT_FATAL(argc == 5, "Expected 5 arguments (M, N, core_x, core_y) but got {}", argc);
    // M N are parameter of matrix that each core will hold
    uint32_t M = atoi(argv[1]);
    uint32_t N = atoi(argv[2]);
    uint32_t core_x = atoi(argv[3]);
    uint32_t core_y = atoi(argv[4]);
    TT_FATAL(core_x >= 1 && core_x <= 8, "core_x out of range");
    TT_FATAL(core_y >= 1 && core_y <= 8, "core_y out of range");
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
    uint32_t Mt = M / tt::constants::TILE_HEIGHT;
    uint32_t Nt = N / tt::constants::TILE_WIDTH;
    TT_FATAL(M % tt::constants::TILE_HEIGHT == 0, "M must be divisible by tile height");
    TT_FATAL(N % tt::constants::TILE_WIDTH == 0, "N must be divisible by tile width");
    TT_FATAL(Mt % core_x == 0, "Mt must be divisible by core_x");
    TT_FATAL(Nt % core_y == 0, "Nt must be divisible by core_y");
    // get data
    tt::DataFormat data_format = tt::DataFormat::Float16_b;
    uint32_t single_tile_elem = tt::constants::TILE_HEIGHT * tt::constants::TILE_WIDTH;
    uint32_t single_tile_size = sizeof(bfloat16) * tt::constants::TILE_HEIGHT * tt::constants::TILE_WIDTH; // 2048B
    // DRAM will hold data of all cores
    uint32_t dram_buffer_size = single_tile_size * Mt * Nt * core_x * core_y;
    std::vector<bfloat16> src_vec = create_random_vector_of_bfloat16_native(dram_buffer_size, 1, 1235);
    std::vector<bfloat16> result_vec(dram_buffer_size/sizeof(bfloat16));

    // Tilize input data before initialize device config
    // tilize(src_vec, (M * core_x * core_y), (N));

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
    uint32_t cb_size = 2 * Mt * Nt * single_tile_size;
    CircularBufferConfig cb_config_in0 = CircularBufferConfig(cb_size, {{tt::CB::c_in0, data_format}})
		.set_page_size(tt::CB::c_in0, single_tile_size);
    auto cb_input_0 = tt_metal::CreateCircularBuffer(program, cores, cb_config_in0);
    CircularBufferConfig cb_config_out0 = CircularBufferConfig(cb_size, {{tt::CB::c_out0, data_format}})
		.set_page_size(tt::CB::c_out0, single_tile_size);
    auto cb_output_0 = tt_metal::CreateCircularBuffer(program, cores, cb_config_out0);

    // create reader kernel
    auto reader_id = tt_metal::CreateKernel(
        program,
        "tt_metal/programming_examples/all2all/kernel/reader.cpp",
        cores,
        tt_metal::DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});
    // create writer kernel
    auto writer_id = tt_metal::CreateKernel(
        program,
        "tt_metal/programming_examples/all2all/kernel/writer.cpp",
        cores,
        tt_metal::DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default});

    auto sender_semaphore = tt::tt_metal::CreateSemaphore(program, cores, 0);

    // set runtime arguments
    for (uint32_t i = 0; i < core_x; i++) {
        for (uint32_t j = 0; j < core_y; j++) {
            CoreCoord core = {i, j};
            std::vector<uint32_t> reader_args = {
                (std::uint32_t)src_dram_addr,
                (std::uint32_t)single_tile_size,
                (std::uint32_t)Mt,
                (std::uint32_t)Nt,
                (std::uint32_t)core_x,
                (std::uint32_t)core_y,
                (std::uint32_t)i,
                (std::uint32_t)j,
                (std::uint32_t)sender_semaphore
            };
            std::vector<uint32_t> writer_args = {
                (std::uint32_t)dst_dram_addr,
                (std::uint32_t)single_tile_size,
                (std::uint32_t)Mt,
                (std::uint32_t)Nt,
                (std::uint32_t)core_x,
                (std::uint32_t)core_y,
                (std::uint32_t)i,
                (std::uint32_t)j
            };
            tt::tt_metal::SetRuntimeArgs(program, reader_id, core, reader_args);
            tt::tt_metal::SetRuntimeArgs(program, writer_id, core, writer_args);
        }
    }

    // pay attention to block host until device output data
    EnqueueWriteBuffer(cq, src_dram_buffer, src_vec.data(), false);
    EnqueueProgram(cq, program, false);
    EnqueueReadBuffer(cq, dst_dram_buffer, result_vec.data(), true);
    tt_metal::detail::DumpDeviceProfileResults(device);

    // Untilize result before check correctness
    // untilize(result_vec, (M * core_x * core_y), (N));

    // Compare src and dst vectors
    bool vectors_match = true;
    uint32_t row_stride = Mt / core_x;
    uint32_t col_stride = Nt / core_y;
    for (size_t i = 0; i < core_x; ++i) {
        for (size_t j = 0; j < core_y; ++j) {
            uint32_t core_num = i * core_y + j;
            // src base is the original location of this core's data; dst base is the dst location of the first core's data
            uint32_t src_base = core_num * (Mt * Nt) * single_tile_elem;
            uint32_t dst_base = (i * row_stride * Nt + j * col_stride) * single_tile_elem;
            for (size_t x = 0; x < core_x; ++x) {
                for (size_t y = 0; y < core_y; ++y) {
                    // src start is the original start location of data which has been sent to core(x,y)
                    // dst start is the dst location of corresponding data in core(x,y)
                    uint32_t src_start = src_base + (x * row_stride * Nt + y * col_stride) * single_tile_elem;
                    uint32_t dst_start = dst_base + (x * core_y + y) * (Mt * Nt) * single_tile_elem;
                    for (size_t row = 0; row < row_stride; ++row) {
                        for (size_t col = 0; col < col_stride; ++col) {
                            // focus on the tile addr
                            uint32_t src_tile_addr = src_start + (row * Nt + col) * single_tile_elem;
                            uint32_t dst_tile_addr = dst_start + (row * Nt + col) * single_tile_elem;
                            // std::cout << "src_tile_addr=" << src_tile_addr << "; dst_tile_addr=" << dst_tile_addr << ";" << std::endl;
                            for (uint32_t elem = 0; elem < single_tile_elem; ++elem) {
                                // element-wise comparison
                                if (src_vec[src_tile_addr + elem] != result_vec[dst_tile_addr + elem]) {
                                    vectors_match = false;
                                    std::stringstream ss;
                                    std::cout << "Mismatch at src_tile_addr=" << src_tile_addr << "; dst_tile_addr=" << dst_tile_addr << "; elem=" << elem << std::endl;
                                    std::cout << "Elem: src=" << src_vec[src_tile_addr + elem] << "; dst=" << result_vec[dst_tile_addr + elem] << std::endl;
                                    // TT_FATAL(false, "Mismatch");
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    if (vectors_match) {
        std::cout << "All2All test PASSED - src and dst vectors match" << std::endl;
    } else {
        std::cout << "All2All test FAILED - src and dst vectors differ" << std::endl;
    }
    // float pearson = check_bfloat16_vector_pcc(golden_vec, result_vec);
    // log_info(tt::LogVerif, "Metalium vs Golden -- PCC = {}", pearson);
    // TT_FATAL(pearson > 0.98, "PCC not high enough. Result PCC: {}, Expected PCC: 0.98", pearson);

    Finish(cq);
    CloseDevice(device);
}
