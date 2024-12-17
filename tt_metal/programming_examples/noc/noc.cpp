// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/host_api.hpp"
#include "tt_metal/impl/device/device.hpp"
#include "common/bfloat16.hpp"
#include "tt_metal/detail/tt_metal.hpp"

/*
* 1. Host writes data to buffer in DRAM
* 2. dram_copy kernel on logical core {0, 0} BRISC copies data from buffer
*      in step 1. to buffer in L1 and back to another buffer in DRAM
* 3. Host reads from buffer written to in step 2.
*/

using namespace tt;
using namespace tt::tt_metal;

int main(int argc, char **argv) {

    if (getenv("TT_METAL_SLOW_DISPATCH_MODE") != nullptr) {
        TT_THROW("Test not supported w/ slow dispatch, exiting");
    }

    bool pass = true;

    try {
        /*
        * Silicon accelerator setup
        */
        constexpr int device_id = 0;
        Device *device =
            CreateDevice(device_id);

        /*
        * Setup program and command queue to execute along with its buffers and kernels to use
        */
        CommandQueue& cq = device->command_queue();
        Program program = CreateProgram();

        constexpr CoreCoord core1 = {0, 0};
        constexpr CoreCoord core2 = {0, 1};

        KernelHandle dram_sender_kernel_id = CreateKernel(
            program,
            "tt_metal/programming_examples/noc/kernels/noc_sender_notile.cpp",
            core1,
            DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default}
        );

        KernelHandle dram_receiver_kernel_id = CreateKernel(
            program,
            "tt_metal/programming_examples/noc/kernels/noc_receiver.cpp",
            core2,
            DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default}
        );

        tt::DataFormat cb_data_format = tt::DataFormat::Float16_b;
        MathFidelity math_fidelity = MathFidelity::HiFi4;

        constexpr uint32_t single_tile_size = 2 * (32 * 32);
        constexpr uint32_t dram_tiles = 1 << 15;
        constexpr uint32_t cb_tiles = 1;
        constexpr uint32_t dram_buffer_size = single_tile_size * dram_tiles;
        constexpr uint32_t cb_buffer_size = single_tile_size * cb_tiles;
        constexpr uint32_t repeat = 1 << 8;
        constexpr uint32_t bandwidth_size = (1 << 10) - (1 << 6) - (1 << 5);
        tt::tt_metal::InterleavedBufferConfig dram_config{
                    .device = device,
                    .size = dram_buffer_size,
                    .page_size = dram_buffer_size,
                    .buffer_type = tt::tt_metal::BufferType::DRAM
        };

        auto input_dram_buffer = CreateBuffer(dram_config);
        const uint32_t input_dram_buffer_addr = input_dram_buffer->address();

        auto output_dram_buffer = CreateBuffer(dram_config);
        const uint32_t output_dram_buffer_addr = output_dram_buffer->address();

        uint32_t src_cb_index = tt::CB::c_in0; //0
        CircularBufferConfig cb_src_config = CircularBufferConfig(cb_buffer_size, {{src_cb_index, cb_data_format}})
            .set_page_size(src_cb_index, single_tile_size);
        auto cb_src = tt_metal::CreateCircularBuffer(program, core1, cb_src_config);
 
        uint32_t dst_cb_index = tt::CB::c_out0; //0
        CircularBufferConfig cb_dst_config = CircularBufferConfig(cb_buffer_size, {{dst_cb_index, cb_data_format}})
            .set_page_size(dst_cb_index, single_tile_size);
        auto cb_dst = tt_metal::CreateCircularBuffer(program, core2, cb_dst_config);
 
        /*
        * Create input data and runtime arguments, then execute
        */
        std::vector<uint32_t> input_vec = create_random_vector_of_bfloat16(
            dram_buffer_size, 100, std::chrono::system_clock::now().time_since_epoch().count());
        EnqueueWriteBuffer(cq, input_dram_buffer, input_vec, false);

        auto core1_physical = device->worker_core_from_logical_core(core1);
        auto core2_physical = device->worker_core_from_logical_core(core2);

        const std::vector<uint32_t> sender_args = {
            input_dram_buffer->address(),
            core2_physical.x,
            core2_physical.y,
            dram_tiles,
            cb_tiles,
            repeat,
            static_cast<uint32_t>(input_dram_buffer->noc_coordinates().x),
            static_cast<uint32_t>(input_dram_buffer->noc_coordinates().y),
            bandwidth_size
        };
        const std::vector<uint32_t> receiver_args = {
            output_dram_buffer->address(),
            core1_physical.x,
            core1_physical.y,
            dram_tiles,
            cb_tiles,
            repeat
        };

        SetRuntimeArgs(
            program,
            dram_sender_kernel_id,
            core1,
            sender_args
        );
        SetRuntimeArgs(
            program,
            dram_receiver_kernel_id,
            core2,
            receiver_args
        );

        EnqueueProgram(cq, program, false);
        Finish(cq);
        tt_metal::detail::DumpDeviceProfileResults(device);

        /*
        * Validation & Teardown
        */
        std::vector<uint32_t> result_vec;
        EnqueueReadBuffer(cq, output_dram_buffer, result_vec, true);

        pass &= input_vec == result_vec;

        pass &= CloseDevice(device);


    } catch (const std::exception &e) {
        tt::log_error(tt::LogTest, "Test failed with exception!");
        tt::log_error(tt::LogTest, "{}", e.what());

        throw;
    }

    if (pass) {
        tt::log_info(tt::LogTest, "Test Passed");
    } else {
        tt::log_info(tt::LogTest, "Test Failed");
    }

    return 0;
}
