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


inline std::vector<float> create_random_vector_of_fp32(uint32_t num_bytes, float rand_max_float, int seed, float offset = 0.0f) {
    auto rand_float = std::bind(std::uniform_real_distribution<float>(0, rand_max_float), std::mt19937(seed));

    std::vector<float> vec(num_bytes/sizeof(float), 0);
    for (int i = 0; i < vec.size(); i++) {
        float num_1_float = rand_float() + offset;
        vec[i] = num_1_float;
    }
    return vec;
}

inline std::vector<uint32_t> create_vector_of_uint32(uint32_t num_bytes, uint32_t core_size, int seed) {
    std::vector<uint32_t> vec(num_bytes/sizeof(uint32_t), 0);
    uint32_t elem_per_core = num_bytes / sizeof(uint32_t) / core_size;
    for (uint32_t i = 0; i < core_size; ++i) {
        for (uint32_t j = 0; j < elem_per_core; ++j) {
            vec[i * elem_per_core + j] = i;
        }
    }
    return vec;
}


int main(int argc, char **argv) {
    TT_FATAL(argc == 5, "Expected 5 arguments (core_x, core_y, M, K) but got {}", argc);
    // Core range: (core_x, core_y), Tokens: M (tiles), Core Dests: K
    uint32_t core_x = atoi(argv[1]);
    uint32_t core_y = atoi(argv[2]);
    uint32_t M = atoi(argv[3]);
    uint32_t K = atoi(argv[4]);
    TT_FATAL(core_x >= 1 && core_x <= 8, "core_x out of range");
    TT_FATAL(core_y >= 1 && core_y <= 8, "core_y out of range");
    TT_FATAL(K >= 1 && K <= core_x * core_y, "K out of range");
    TT_FATAL(M * core_x * core_y < 360, "CB out of range");
    // get program and device
    int device_id = 0;
    Device *device = CreateDevice(device_id);
    CommandQueue& cq = device->command_queue();
    Program program = CreateProgram();
    // get cores
    CoreCoord start_core = {0, 0};
    CoreCoord end_core = {core_x - 1, core_y - 1};
    CoreRange cores(start_core, end_core);
    // get data
    tt::DataFormat data_format = tt::DataFormat::Float32;
    // tt::DataFormat data_format = tt::DataFormat::UInt32;
    uint32_t single_tile_elem = tt::constants::TILE_HEIGHT * tt::constants::TILE_WIDTH;
    uint32_t single_tile_size = sizeof(float) * single_tile_elem;
    // DRAM will hold data of all cores
    uint32_t dram_buffer_size_input = single_tile_size * M * core_x * core_y;
    uint32_t dram_buffer_size_output = single_tile_size * M * (core_x * core_y) * (core_x * core_y);
    std::vector<float> src_vec = create_random_vector_of_fp32(dram_buffer_size_input, 1, 1235);
    std::vector<float> result_vec(dram_buffer_size_output / sizeof(float));
    // std::vector<uint32_t> src_vec = create_vector_of_uint32(dram_buffer_size_input, core_x * core_y, 1235);
    // std::vector<uint32_t> result_vec(dram_buffer_size_output / sizeof(uint32_t));

    // create dram buffer
    tt_metal::InterleavedBufferConfig dram_config_input {
        .device = device,
        .size = dram_buffer_size_input,
        .page_size = single_tile_size,
        .buffer_type = tt_metal::BufferType::DRAM
    };
    tt_metal::InterleavedBufferConfig dram_config_output {
        .device = device,
        .size = dram_buffer_size_output,
        .page_size = single_tile_size,
        .buffer_type = tt_metal::BufferType::DRAM
    };
    std::shared_ptr<tt::tt_metal::Buffer> src_dram_buffer = CreateBuffer(dram_config_input);
    std::shared_ptr<tt::tt_metal::Buffer> dst_dram_buffer = CreateBuffer(dram_config_output);
    uint32_t src_dram_addr = src_dram_buffer->address();
    uint32_t dst_dram_addr = dst_dram_buffer->address();

    // create circular buffers
    uint32_t cb_size_in0 = 2 * M * single_tile_size;
    uint32_t cb_size_out0 = 2 * M * core_x * core_y * single_tile_size;
    uint32_t cb_size_out1 = 2 * core_x * core_y * sizeof(uint32_t);
    CircularBufferConfig cb_config_in0 = CircularBufferConfig(cb_size_in0, {{tt::CB::c_in0, data_format}})
		.set_page_size(tt::CB::c_in0, single_tile_size);
    auto cb_input_0 = tt_metal::CreateCircularBuffer(program, cores, cb_config_in0);
    CircularBufferConfig cb_config_out0 = CircularBufferConfig(cb_size_out0, {{tt::CB::c_out0, data_format}})
		.set_page_size(tt::CB::c_out0, single_tile_size);
    auto cb_output_0 = tt_metal::CreateCircularBuffer(program, cores, cb_config_out0);

    // create reader kernel
    auto reader_id = tt_metal::CreateKernel(
        program,
        "tt_metal/programming_examples/moe_random/kernel/reader.cpp",
        cores,
        tt_metal::DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});
    // create writer kernel
    auto writer_id = tt_metal::CreateKernel(
        program,
        "tt_metal/programming_examples/moe_random/kernel/writer.cpp",
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
                (std::uint32_t)M,
                (std::uint32_t)core_x,
                (std::uint32_t)core_y,
                (std::uint32_t)i,
                (std::uint32_t)j,
                (std::uint32_t)sender_semaphore,
                (std::uint32_t)K
            };
            std::vector<uint32_t> writer_args = {
                (std::uint32_t)dst_dram_addr,
                (std::uint32_t)single_tile_size,
                (std::uint32_t)M,
                (std::uint32_t)core_x,
                (std::uint32_t)core_y,
                (std::uint32_t)i,
                (std::uint32_t)j,
                (std::uint32_t)K
            };
            tt::tt_metal::SetRuntimeArgs(program, reader_id, core, reader_args);
            tt::tt_metal::SetRuntimeArgs(program, writer_id, core, writer_args);
        }
    }

    // pay attention to block host until device output data
    EnqueueWriteBuffer(cq, src_dram_buffer, src_vec.data(), false);
    EnqueueProgram(cq, program, false);
    Finish(cq);
    EnqueueReadBuffer(cq, dst_dram_buffer, result_vec.data(), true);
    tt_metal::detail::DumpDeviceProfileResults(device);

    // Untilize result before check correctness
    // untilize(result_vec, (M * core_x * core_y), (N));

    // Compare src and dst vectors
    bool vectors_match = true;
    for (uint32_t i = 0; i < (core_x * core_y); ++i) {
        uint32_t curr_core_x = i / core_y;
        uint32_t curr_core_y = i % core_y;
        uint32_t index_base = i * (core_x * core_y);
        uint32_t data_base = i * M * single_tile_elem * (core_x * core_y);
        for (uint32_t j = 0; j < (core_x * core_y); ++j) {
            // if (result_index_vec[index_base + j] == 1) {
                for (uint32_t k = 0; k < M * single_tile_elem; ++k) {
                    uint32_t result_index = data_base + j * M * single_tile_elem + k;
                    uint32_t src_index = j * M * single_tile_elem + k;
                    if (result_vec[result_index] != src_vec[src_index]) {
                        vectors_match = false;
                        std::cout << "Mismatch at core (" << curr_core_x << ", " << curr_core_y << ") receive from core (" << j / core_y << ", " << j % core_y << ")" << std::endl;
                        std::cout << "Mismatch position: " << k << " " << result_vec[result_index] << " " << src_vec[src_index] << std::endl;
                        break;
                    }
                }
                std::cout << "Match at core (" << curr_core_x << ", " << curr_core_y << ") receive from core (" << j / core_y << ", " << j % core_y << ")" << std::endl;
            // }
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }

    if (vectors_match) {
        std::cout << "MoE Random-K test PASSED - src and dst vectors match" << std::endl;
    } else {
        std::cout << "MoE Random-K test FAILED - src and dst vectors differ" << std::endl;
    }
    // float pearson = check_bfloat16_vector_pcc(golden_vec, result_vec);
    // log_info(tt::LogVerif, "Metalium vs Golden -- PCC = {}", pearson);
    // TT_FATAL(pearson > 0.98, "PCC not high enough. Result PCC: {}, Expected PCC: 0.98", pearson);

    CloseDevice(device);
}
