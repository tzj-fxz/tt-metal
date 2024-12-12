// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/common/constants.hpp"
#include "tt_metal/detail/util.hpp"
#include "tt_metal/host_api.hpp"
#include "tt_metal/common/work_split.hpp"
#include "ttnn/operation.hpp"
#include "ttnn/operations/matmul/device/matmul_op.hpp"

using namespace tt;
using namespace tt::constants;
using namespace tt::tt_metal;

namespace ttnn {

namespace operations {

namespace matmul {

operation::ProgramWithCallbacks create_program_cannon(
    tt_metal::Device *device,
    tt::DataFormat in0_data_format,
    tt::DataFormat in1_data_format,
    tt::DataFormat output_data_format,
    MathFidelity math_fidelity,
    uint32_t num_cores_x,
    uint32_t num_cores_y,
    uint32_t B,
    uint32_t Mt,
    uint32_t Nt,
    uint32_t Kt,
    bool bcast_batch,
    uint32_t out_subblock_h,
    uint32_t out_subblock_w,
    uint32_t per_core_M,
    uint32_t per_core_N,
    uint32_t per_core_K,
    tt_metal::Buffer *in0_buffer,
    tt_metal::Buffer *in1_buffer,
    tt_metal::Buffer *output_buffer
) {
    CommandQueue &cq = device->command_queue();
    tt_metal::Program program{};

    tt::DataFormat cb_data_format = tt::DataFormat::Float16_b;
    uint32_t single_tile_size = tt_metal::detail::TileSize(cb_data_format); // bytes

    uint32_t in0_CB_tiles = per_core_M * per_core_K;
    uint32_t in1_CB_tiles = per_core_N * per_core_K;
    uint32_t out_CB_tiles = per_core_M * per_core_N;
    uint32_t in0_CB_size = in0_CB_tiles * single_tile_size * 2; // double buffer
    uint32_t in1_CB_size = in1_CB_tiles * single_tile_size * 2; // double buffer
    uint32_t out_CB_size = out_CB_tiles * single_tile_size * 2; // double buffer

    // dispatch to cores
    uint32_t start_core_x = 0;
    uint32_t start_core_y = 0;
    CoreRange all_cores(
        {(std::size_t)start_core_x, (std::size_t)start_core_y},
        {(std::size_t)start_core_x + num_cores_x - 1, (std::size_t)start_core_y + num_cores_y - 1});

    // allocate circular buffers
    uint32_t src0_cb_index = CB::c_in0; //0
    CircularBufferConfig cb_src0_config = CircularBufferConfig(in0_CB_size, {{src0_cb_index, cb_data_format}})
		.set_page_size(src0_cb_index, single_tile_size);
    auto cb_src0 = tt_metal::CreateCircularBuffer(program, all_cores, cb_src0_config);

    uint32_t src1_cb_index = CB::c_in1; // 1
    CircularBufferConfig cb_src1_config = CircularBufferConfig(in1_CB_size, {{src1_cb_index, cb_data_format}})
		.set_page_size(src1_cb_index, single_tile_size);
    auto cb_src1 = tt_metal::CreateCircularBuffer(program, all_cores, cb_src1_config);

    uint32_t output_cb_index = CB::c_out0; // output operands start at index 16
    uint32_t interm0_cb_index = CB::c_intermed0;
    std::map<uint8_t, tt::DataFormat> output_cb_data_format_spec {
        {output_cb_index, cb_data_format},
        {interm0_cb_index, cb_data_format}
    };
    CircularBufferConfig cb_output_config = CircularBufferConfig(out_CB_size, output_cb_data_format_spec)
		.set_page_size(output_cb_index, single_tile_size)
        .set_page_size(interm0_cb_index, single_tile_size);
    auto cb_output = tt_metal::CreateCircularBuffer(program, CoreRangeSet({all_cores}), cb_output_config);

    ////////////////////////////
    /*
    * Compile time arguments
    */
    bool src0_is_dram = in0_buffer->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;
    bool src1_is_dram = in1_buffer->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;
    std::vector<uint32_t> reader_compile_time_args = {(uint32_t)src0_is_dram, (uint32_t)src1_is_dram};

    bool dst_is_dram = output_buffer->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;
    std::vector<uint32_t> writer_compile_time_args = {(uint32_t)dst_is_dram};

    auto reader_kernel_cannon = tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/matmul/device/kernels/dataflow/reader_bmm_cannon_semaphore.cpp",
        all_cores,
        tt_metal::DataMovementConfig{.processor = tt_metal::DataMovementProcessor::RISCV_1, .noc = tt_metal::NOC::RISCV_1_default, .compile_args = reader_compile_time_args}
    );
    auto writer_kernel_cannon = tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/matmul/device/kernels/dataflow/writer_bmm_cannon.cpp",
        all_cores,
        tt_metal::DataMovementConfig{.processor = tt_metal::DataMovementProcessor::RISCV_0, .noc = tt_metal::NOC::RISCV_0_default, .compile_args = writer_compile_time_args}
    );
    std::vector<uint32_t> compute_compile_time_args = {
        (std::uint32_t) Mt,
        (std::uint32_t) Nt,
        (std::uint32_t) Kt,
        (std::uint32_t) bcast_batch,
        (std::uint32_t) per_core_M,
        (std::uint32_t) per_core_N,
        (std::uint32_t) per_core_K,
        (std::uint32_t) out_subblock_h,
        (std::uint32_t) out_subblock_w
    };
    auto compute_kernel_cannon = tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/matmul/device/kernels/compute/bmm_cannon_v2.cpp",
        all_cores,
        tt_metal::ComputeConfig{.math_fidelity = math_fidelity, .compile_args = compute_compile_time_args}
    );

    auto in0_sender_semaphore_id = tt_metal::CreateSemaphore(program, all_cores, 0);
    auto in0_receiver_semaphore_id = tt_metal::CreateSemaphore(program, all_cores, 0);
    auto in1_sender_semaphore_id = tt_metal::CreateSemaphore(program, all_cores, 0);
    auto in1_receiver_semaphore_id = tt_metal::CreateSemaphore(program, all_cores, 0);

    for (uint32_t core_x = start_core_x; core_x < start_core_x + num_cores_x; ++core_x) {
        for (uint32_t core_y = start_core_y; core_y < start_core_y + num_cores_y; ++core_y) {
            CoreCoord core = {(std::size_t) core_x, (std::size_t) core_y};
            std::vector<uint32_t> reader_args = {
                (std::uint32_t) Mt,
                (std::uint32_t) Nt,
                (std::uint32_t) Kt,
                (std::uint32_t) B,
                (std::uint32_t) bcast_batch,
                (std::uint32_t) core_x,
                (std::uint32_t) core_y,
                (std::uint32_t) in0_buffer->address(),
                (std::uint32_t) core_x * Kt * per_core_M + core_y * per_core_K,
                (std::uint32_t) per_core_M * per_core_K,
                (std::uint32_t) in1_buffer->address(),
                (std::uint32_t) core_x * Nt * per_core_K + core_y * per_core_N,
                (std::uint32_t) per_core_K * per_core_N,
                (std::uint32_t) per_core_M,
                (std::uint32_t) per_core_N,
                (std::uint32_t) per_core_K,
                (std::uint32_t) in0_sender_semaphore_id,
                (std::uint32_t) in0_receiver_semaphore_id,
                (std::uint32_t) in1_sender_semaphore_id,
                (std::uint32_t) in1_receiver_semaphore_id
            };
            tt_metal::SetRuntimeArgs(program, reader_kernel_cannon, core, reader_args);

            std::vector<uint32_t> writer_args = {
                (std::uint32_t) Mt,
                (std::uint32_t) Nt,
                (std::uint32_t) Kt,
                (std::uint32_t) B,
                (std::uint32_t) core_x,
                (std::uint32_t) core_y,
                (std::uint32_t) per_core_M,
                (std::uint32_t) per_core_N,
                (std::uint32_t) per_core_K,
                (std::uint32_t) out_subblock_h,
                (std::uint32_t) out_subblock_w,
                (std::uint32_t) output_buffer->address()
            };
            tt::tt_metal::SetRuntimeArgs(program, writer_kernel_cannon, core, writer_args);
            
            std::vector<uint32_t> compute_args = {
                (std::uint32_t) core_x,
                (std::uint32_t) core_y,
                (std::uint32_t) in0_sender_semaphore_id,
                (std::uint32_t) in1_sender_semaphore_id
            };
            tt::tt_metal::SetRuntimeArgs(program, compute_kernel_cannon, core, compute_args);
        }
    }

    auto override_runtime_args_callback = [reader_kernel_id = reader_kernel_cannon,
                                            writer_kernel_id = writer_kernel_cannon,
                                            num_cores_x,
                                            num_cores_y](
                                              const tt_metal::Program &program,
                                              const std::vector<Buffer *> &input_buffers,
                                              const std::vector<Buffer *> &output_buffers) {

        auto src0_buffer = input_buffers.at(0);
        auto src1_buffer = input_buffers.at(1);
        auto dst_buffer = output_buffers.at(0);

        for (uint32_t core_x = 0; core_x < num_cores_x; ++core_x) {
            for (uint32_t core_y = 0; core_y < num_cores_y; ++core_y) {
                CoreCoord core = {(std::size_t) core_x, (std::size_t) core_y};
                {
                    auto &runtime_args = GetRuntimeArgs(program, reader_kernel_id, core);
                    runtime_args[7] = src0_buffer->address();
                    runtime_args[10] = src1_buffer->address();
                }
                {
                    auto &runtime_args = GetRuntimeArgs(program, writer_kernel_id, core);
                    runtime_args[11] = dst_buffer->address();
                }
            }
        }
    };
    
    return {std::move(program), override_runtime_args_callback};
}

operation::ProgramWithCallbacks matmul_multi_core_cannon(
    const Tensor &input_tensor_a,
    const Tensor &input_tensor_b,
    Tensor &output_tensor,
    bool bcast_batch,
    CoreCoord compute_with_storage_grid_size,
    // tt::tt_metal::DataType output_dtype,
    DeviceComputeKernelConfig compute_kernel_config,
    uint32_t per_core_M,
    uint32_t per_core_N,
    uint32_t per_core_K,
    uint32_t out_subblock_h,
    uint32_t out_subblock_w) {
    
    const auto &ashape = input_tensor_a.get_legacy_shape(), bshape = input_tensor_b.get_legacy_shape();
    tt::DataFormat in0_data_format = tt_metal::datatype_to_dataformat_converter(input_tensor_a.get_dtype());
    tt::DataFormat in1_data_format = tt_metal::datatype_to_dataformat_converter(input_tensor_b.get_dtype());
    tt::DataFormat dataflow0_data_format = tt_metal::datatype_to_dataformat_converter(input_tensor_a.get_dtype());
    tt::DataFormat dataflow1_data_format = tt_metal::datatype_to_dataformat_converter(input_tensor_b.get_dtype());
    tt::DataFormat output_data_format = tt_metal::datatype_to_dataformat_converter(output_tensor.get_dtype());

    MathFidelity math_fidelity = MathFidelity::HiFi4;

    tt_metal::Buffer *in0_buffer = input_tensor_a.buffer();
    tt_metal::Buffer *in1_buffer = input_tensor_b.buffer();
    tt_metal::Buffer *output_buffer = output_tensor.buffer();

    uint32_t B = get_batch_size(ashape);
    uint32_t M = ashape[-2];
    uint32_t N = bshape[-1];
    uint32_t K = ashape[-1];
    uint32_t Mt = ashape[-2] / TILE_HEIGHT;
    uint32_t Nt = bshape[-1] / TILE_WIDTH;
    uint32_t Kt = ashape[-1] / TILE_WIDTH;

    TT_FATAL(Mt % per_core_M == 0, "Error");
    TT_FATAL(Nt % per_core_N == 0, "Error");
    TT_FATAL(Mt == Nt, "Currently only supports square matrices");

    tt_metal::Device *device = input_tensor_a.device();
    // auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    uint32_t num_cores_x = compute_with_storage_grid_size.x < compute_with_storage_grid_size.y ? compute_with_storage_grid_size.x : compute_with_storage_grid_size.y;
    uint32_t num_cores_y = compute_with_storage_grid_size.y < compute_with_storage_grid_size.x ? compute_with_storage_grid_size.y : compute_with_storage_grid_size.x;

    uint32_t num_blocks_total = (Mt / per_core_M) * (Nt / per_core_N);
    TT_FATAL(num_blocks_total <= num_cores_x * num_cores_y, "Error");

    // CoreRangeSet all_cores(num_cores_to_corerangeset(
    //     num_blocks_x * num_blocks_y, device->compute_with_storage_grid_size(), true));

    return create_program_cannon(
        device,
        in0_data_format,
        in1_data_format,
        output_data_format,
        math_fidelity,
        num_cores_x,
        num_cores_y,
        B,
        Mt,
        Nt,
        Kt,
        bcast_batch,
        out_subblock_h,
        out_subblock_w,
        per_core_M,
        per_core_N,
        per_core_K,
        in0_buffer,
        in1_buffer,
        output_buffer
    );

}

} // namespace matmul

} // namespace operations

} // namespace ttnn
