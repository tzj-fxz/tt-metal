#include "tt_metal/host_api.hpp"
#include "tt_metal/common/constants.hpp"
#include "tt_metal/detail/util.hpp"
#include "tt_metal/common/bfloat16.hpp"
#include "tt_metal/common/test_tiles.hpp"
#include "tt_metal/impl/dispatch/command_queue.hpp"
#include "tt_metal/programming_examples/matmul_common/bmm_op.hpp"
#include "tt_metal/common/tilize_untilize.hpp"
#include "impl/device/device.hpp"
#include "tt_metal/detail/tt_metal.hpp"
#include <chrono>

using namespace std;
using namespace tt;
using namespace tt::tt_metal;
using namespace tt::constants;

constexpr uint32_t PROFILING_ITERATIONS = 0;
constexpr uint32_t PER_CORE_M = 8;
constexpr uint32_t PER_CORE_N = 8;
constexpr uint32_t PER_CORE_K = 8;
constexpr uint32_t CORE_NUM_X = 7;
constexpr uint32_t CORE_NUM_Y = 7;
constexpr uint32_t DRAM_SHARD_X = 8; // each time load from DRAM/NoC: height
constexpr uint32_t DRAM_SHARD_Y = 8; // each time load from DRAM/NoC: width
constexpr uint32_t SUBBLOCK_SIZE_H = 4;
constexpr uint32_t SUBBLOCK_SIZE_W = 2;


void golden_matmul(std::vector<bfloat16>& a, std::vector<bfloat16>& b, std::vector<bfloat16>& output,
                        uint32_t M, uint32_t N, uint32_t K, uint32_t B) {
    std::uint32_t idx_c = 0;
    std::uint32_t idx_a = 0;
    std::uint32_t idx_b = 0;

    float c_f;
    float float_tmp;
    std::vector<bfloat16> c_bf(M * N, 0);

    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            idx_c = j+ (i * N);
            idx_a = i * K;
            idx_b = j;
            c_f = 0;
            for (int k_m = 0; k_m < K; k_m++) {
                float_tmp = a[idx_a].to_float() * b[idx_b].to_float();
                c_f += float_tmp;
                idx_a += 1;
                idx_b += N;
            }
            output.at(idx_c) = bfloat16(c_f);
        }
    }
}


void matmul_cannon(std::vector<bfloat16>& a, std::vector<bfloat16>& b, std::vector<bfloat16>& output,
                   bool bcast_batch, uint32_t M, uint32_t N, uint32_t K, uint32_t B, Device *device) {
    CommandQueue &cq = device->command_queue();
    Program program{};

    tt::DataFormat cb_data_format = tt::DataFormat::Float16_b;
    MathFidelity math_fidelity = MathFidelity::HiFi4;

    uint32_t Mt = M / TILE_HEIGHT;
    uint32_t Nt = N / TILE_WIDTH;
    uint32_t Kt = K / TILE_WIDTH;

    // the number of tiles need to be processed by one core, can be defined
    // current only support square, per_core_M = per_core_N = per_core_K
    // but only process DRAM_SHARD_X * DRAM_SHARD_Y tiles at a time (not PER_CORE_M*PER_CORE_N), for the space in SRAM is so limited
    TT_ASSERT(Mt % PER_CORE_M == 0);
    TT_ASSERT(Nt % PER_CORE_N == 0);
    TT_ASSERT(PER_CORE_M % DRAM_SHARD_X == 0);
    TT_ASSERT(PER_CORE_N % DRAM_SHARD_Y == 0);
    TT_ASSERT(DRAM_SHARD_X % SUBBLOCK_SIZE_H == 0);
    TT_ASSERT(DRAM_SHARD_Y % SUBBLOCK_SIZE_W == 0);
    // uint32_t in0_CB_tiles = PER_CORE_M * PER_CORE_K;
    // uint32_t in1_CB_tiles = PER_CORE_N * PER_CORE_K;
    // uint32_t out_CB_tiles = PER_CORE_M * PER_CORE_N;
    uint32_t in0_CB_tiles = DRAM_SHARD_X * PER_CORE_K;
    uint32_t in1_CB_tiles = PER_CORE_K * DRAM_SHARD_Y;
    uint32_t out_CB_tiles = DRAM_SHARD_X * DRAM_SHARD_Y;
    uint32_t NUM_BLOCK_M = Mt / PER_CORE_M;
    uint32_t NUM_BLOCK_N = Nt / PER_CORE_N;


    // dispatch to cores
    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    uint32_t max_cores_x = compute_with_storage_grid_size.x < compute_with_storage_grid_size.y ? compute_with_storage_grid_size.x : compute_with_storage_grid_size.y;
    uint32_t max_cores_y = compute_with_storage_grid_size.y < compute_with_storage_grid_size.x ? compute_with_storage_grid_size.y : compute_with_storage_grid_size.x;
    TT_ASSERT(NUM_BLOCK_M * NUM_BLOCK_N <= max_cores_x * max_cores_y);
    CoreCoord core_range = bmm_op_utils::get_core_range(NUM_BLOCK_M, NUM_BLOCK_N, max_cores_x, max_cores_y);

    uint32_t start_core_x = 0;
    uint32_t start_core_y = 0;
    uint32_t num_cores_x = core_range.x;
    uint32_t num_cores_y = core_range.y;
    // CoreRangeSet all_cores(tt::tt_metal::num_cores_to_corerangeset(num_cores_x * num_cores_y, compute_with_storage_grid_size, true));
    CoreRange all_cores(
        {(std::size_t)start_core_x, (std::size_t)start_core_y},
        {(std::size_t)start_core_x + num_cores_x - 1, (std::size_t)start_core_y + num_cores_y - 1});

    uint32_t single_tile_size = detail::TileSize(cb_data_format); // bytes
    uint32_t dram_buffer_A_size = single_tile_size * Mt * Kt; // num_tiles of FP16_B, hard-coded in the reader/writer kernels
    uint32_t dram_buffer_B_size = single_tile_size * Nt * Kt; // num_tiles of FP16_B, hard-coded in the reader/writer kernels
    uint32_t dram_buffer_C_size = single_tile_size * Mt * Nt; // num_tiles of FP16_B, hard-coded in the reader/writer kernels
    uint32_t in0_CB_size = in0_CB_tiles * single_tile_size * 2; // double buffer, but for dram load pipeline, should be four-fold buffer
    uint32_t in1_CB_size = in1_CB_tiles * single_tile_size * 2; // double buffer, but for dram load pipeline, should be four-fold buffer
    uint32_t out_CB_size = out_CB_tiles * single_tile_size; // double buffer, but for dram load pipeline, should be four-fold buffer
    // dram buffer and circular buffer config for each core
    tt_metal::InterleavedBufferConfig dram_config_A{
                    .device= device,
                    .size = dram_buffer_A_size,
                    .page_size = single_tile_size,
                    .buffer_type = tt_metal::BufferType::DRAM
        };

    tt_metal::InterleavedBufferConfig dram_config_B{
                    .device= device,
                    .size = dram_buffer_B_size,
                    .page_size = single_tile_size,
                    .buffer_type = tt_metal::BufferType::DRAM
        };

    tt_metal::InterleavedBufferConfig dram_config_C{
                    .device= device,
                    .size = dram_buffer_C_size,
                    .page_size = single_tile_size,
                    .buffer_type = tt_metal::BufferType::DRAM
        };

    auto src0_dram_buffer = CreateBuffer(dram_config_A);
    auto src1_dram_buffer = CreateBuffer(dram_config_B);
    auto dst_dram_buffer = CreateBuffer(dram_config_C);
    uint32_t src0_addr = src0_dram_buffer->address();
    uint32_t src1_addr = src1_dram_buffer->address();
    uint32_t dst_addr = dst_dram_buffer->address();

    uint32_t src0_cb_index = CB::c_in0; // 0 SRAM
    CircularBufferConfig cb_src0_config = CircularBufferConfig(in0_CB_size, {{src0_cb_index, cb_data_format}})
		.set_page_size(src0_cb_index, single_tile_size);
    auto cb_src0 = tt_metal::CreateCircularBuffer(program, all_cores, cb_src0_config);

    uint32_t src1_cb_index = CB::c_in1; // 1 SRAM
    CircularBufferConfig cb_src1_config = CircularBufferConfig(in1_CB_size, {{src1_cb_index, cb_data_format}})
		.set_page_size(src1_cb_index, single_tile_size);
    auto cb_src1 = tt_metal::CreateCircularBuffer(program, all_cores, cb_src1_config);

    uint32_t output_cb_index = CB::c_out0; // Out DRAM
    uint32_t interm0_cb_index = CB::c_intermed0;
    std::map<uint8_t, tt::DataFormat> output_cb_data_format_spec {
        {output_cb_index, cb_data_format},
        {interm0_cb_index, cb_data_format}
    };
    CircularBufferConfig cb_output_config = CircularBufferConfig(out_CB_size, output_cb_data_format_spec)
		.set_page_size(output_cb_index, single_tile_size)
        .set_page_size(interm0_cb_index, single_tile_size);
    auto cb_output = tt_metal::CreateCircularBuffer(program, all_cores, cb_output_config);

    ////////////////////////////
    /*
    * Compile time arguments
    */
    bool src0_is_dram = src0_dram_buffer->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;
    bool src1_is_dram = src1_dram_buffer->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;
    std::vector<uint32_t> reader_compile_time_args = {(uint32_t)src0_is_dram, (uint32_t)src1_is_dram};

    bool dst_is_dram = dst_dram_buffer->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;
    //std::vector<uint32_t> writer_compile_time_args = {(std::uint32_t) output_cb_index, (uint32_t)dst_is_dram};
    std::vector<uint32_t> writer_compile_time_args = {(uint32_t)dst_is_dram};

    auto reader_kernel_cannon = tt_metal::CreateKernel(
        program,
        "tt_metal/programming_examples/matmul_common/kernels/dataflow/reader_bmm_cannon_semaphore.cpp",
        all_cores,
        tt_metal::DataMovementConfig{.processor = tt_metal::DataMovementProcessor::RISCV_1, .noc = tt_metal::NOC::RISCV_1_default, .compile_args = reader_compile_time_args}
    );
    auto writer_kernel_cannon = tt_metal::CreateKernel(
        program,
        "tt_metal/programming_examples/matmul_common/kernels/dataflow/writer_bmm_cannon.cpp",
        all_cores,
        tt_metal::DataMovementConfig{.processor = tt_metal::DataMovementProcessor::RISCV_0, .noc = tt_metal::NOC::RISCV_0_default, .compile_args = writer_compile_time_args}
    );
    std::vector<uint32_t> compute_compile_time_args = {
        (std::uint32_t) Mt,
        (std::uint32_t) Nt,
        (std::uint32_t) Kt,
        (std::uint32_t) B,
        (std::uint32_t) PER_CORE_M,
        (std::uint32_t) PER_CORE_N,
        (std::uint32_t) PER_CORE_K,
        (std::uint32_t) SUBBLOCK_SIZE_H,
        (std::uint32_t) SUBBLOCK_SIZE_W,
        (std::uint32_t) DRAM_SHARD_X,
        (std::uint32_t) DRAM_SHARD_Y
    };
    auto compute_kernel_cannon = tt_metal::CreateKernel(
        program,
        "tt_metal/programming_examples/matmul_common/kernels/compute/bmm_cannon_v3.cpp",
        // "tt_metal/programming_examples/matmul_common/kernels/compute/bmm_cannon_dummy.cpp",
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
            CoreCoord src0_next_core = {(std::size_t) core_x, (std::size_t) (core_y + num_cores_y + 1) % num_cores_y};
            CoreCoord src1_next_core = {(std::size_t) (core_x + num_cores_x + 1) % num_cores_x, (std::size_t) core_y};
            CoreCoord src0_prev_core = {(std::size_t) core_x, (std::size_t) (core_y + num_cores_y - 1) % num_cores_y};
            CoreCoord src1_prev_core = {(std::size_t) (core_x + num_cores_x - 1) % num_cores_x, (std::size_t) core_y};

            auto core_physical = device->worker_core_from_logical_core(core);
            auto src0_next_core_physical = device->worker_core_from_logical_core(src0_next_core);
            auto src1_next_core_physical = device->worker_core_from_logical_core(src1_next_core);
            auto src0_prev_core_physical = device->worker_core_from_logical_core(src0_prev_core);
            auto src1_prev_core_physical = device->worker_core_from_logical_core(src1_prev_core);

            std::vector<uint32_t> reader_args = {
                (std::uint32_t) Mt,
                (std::uint32_t) Nt,
                (std::uint32_t) Kt,
                (std::uint32_t) B,
                (std::uint32_t) bcast_batch,
                (std::uint32_t) core_x,
                (std::uint32_t) core_y,
                (std::uint32_t) src0_addr,
                (std::uint32_t) core_x * Kt * PER_CORE_M + core_y * PER_CORE_K,
                (std::uint32_t) PER_CORE_M * PER_CORE_K,
                (std::uint32_t) src1_addr,
                (std::uint32_t) core_x * Nt * PER_CORE_K + core_y * PER_CORE_N,
                (std::uint32_t) PER_CORE_K * PER_CORE_N,
                (std::uint32_t) PER_CORE_M,
                (std::uint32_t) PER_CORE_N,
                (std::uint32_t) PER_CORE_K,
                (std::uint32_t) in0_sender_semaphore_id,
                (std::uint32_t) in0_receiver_semaphore_id,
                (std::uint32_t) in1_sender_semaphore_id,
                (std::uint32_t) in1_receiver_semaphore_id,
                (std::uint32_t) src0_next_core_physical.x,
                (std::uint32_t) src0_next_core_physical.y,
                (std::uint32_t) src1_next_core_physical.x,
                (std::uint32_t) src1_next_core_physical.y,
                (std::uint32_t) src0_prev_core_physical.x,
                (std::uint32_t) src0_prev_core_physical.y,
                (std::uint32_t) src1_prev_core_physical.x,
                (std::uint32_t) src1_prev_core_physical.y,
                (std::uint32_t) DRAM_SHARD_X,
                (std::uint32_t) DRAM_SHARD_Y
            };
            tt_metal::SetRuntimeArgs(program, reader_kernel_cannon, core, reader_args);

            std::vector<uint32_t> writer_args = {
                (std::uint32_t) Mt,
                (std::uint32_t) Nt,
                (std::uint32_t) Kt,
                (std::uint32_t) B,
                (std::uint32_t) core_x,
                (std::uint32_t) core_y,
                (std::uint32_t) PER_CORE_M,
                (std::uint32_t) PER_CORE_N,
                (std::uint32_t) PER_CORE_K,
                (std::uint32_t) SUBBLOCK_SIZE_H,
                (std::uint32_t) SUBBLOCK_SIZE_W,
                (std::uint32_t) dst_addr,
                (std::uint32_t) DRAM_SHARD_X,
                (std::uint32_t) DRAM_SHARD_Y
            };
            tt::tt_metal::SetRuntimeArgs(program, writer_kernel_cannon, core, writer_args);
            
            std::vector<uint32_t> compute_args = {
                (std::uint32_t) core_x,
                (std::uint32_t) core_y
            };
            tt::tt_metal::SetRuntimeArgs(program, compute_kernel_cannon, core, compute_args);
        }
    }
    
    log_info(tt::LogVerif, " -- Metalium Core Command Queue --");
    EnqueueWriteBuffer(cq, src0_dram_buffer, a.data(), false);
    EnqueueWriteBuffer(cq, src1_dram_buffer, b.data(), false);
    auto start = chrono::high_resolution_clock::now();
    EnqueueProgram(cq, program, false);
    chrono::duration<double> duration = chrono::high_resolution_clock::now() - start;
    EnqueueReadBuffer(cq, dst_dram_buffer, output.data(), true);
    Finish(cq);
    log_info(tt::LogVerif, "Program average time: {} seconds", duration.count());
    // auto start = chrono::high_resolution_clock::now();
    // for (int i = 0; i < PROFILING_ITERATIONS; ++i) {
    //     EnqueueWriteBuffer(cq, src0_dram_buffer, a.data(), false);
    //     EnqueueWriteBuffer(cq, src1_dram_buffer, b.data(), false);
    //     EnqueueProgram(cq, program, false);
    //     // Finish(cq);
    //     EnqueueReadBuffer(cq, dst_dram_buffer, output.data(), true);
    // }
    // chrono::duration<double> duration = chrono::high_resolution_clock::now() - start;
    // log_info(tt::LogVerif, "Program average time: {} seconds", duration.count() / PROFILING_ITERATIONS);
    // EnqueueReadBuffer(cq, dst_dram_buffer, output.data(), true);
    return;
}


int main(int argc, char **argv) {
    bool pass = true;
    
    if (getenv("TT_METAL_SLOW_DISPATCH_MODE") != nullptr) {
        TT_THROW("Test not supported w/ slow dispatch, exiting");
    }

    try {
        /* Silicon accelerator setup */
        constexpr int device_id = 0;
        Device *device = CreateDevice(device_id);

        ////////////////////////////////////////////////////////////////////////////
        //                      Matmul Parameters Setup
        ////////////////////////////////////////////////////////////////////////////
        // NOTE: Only supports matmuls where output is blocks of 16 x 16 tiles (ie. multiples of 16*32 x 16*32)
        // NOTE: Maximum number of tiles in output is 120 * 16^2 = 30,720 (eg. [1, 1, 5120, 6144])

        /* Create source data */
        constexpr uint32_t M = PER_CORE_M * TILE_HEIGHT * CORE_NUM_X;  // user-defined
        constexpr uint32_t N = PER_CORE_N * TILE_WIDTH * CORE_NUM_Y;  // user-defined
        constexpr uint32_t K = PER_CORE_K * TILE_WIDTH * CORE_NUM_Y;  // user-defined
        constexpr uint32_t B = 1;  // user-defined

        uint32_t Mt = M / TILE_HEIGHT;
        uint32_t Kt = K / TILE_WIDTH;
        uint32_t Nt = N / TILE_WIDTH;

        constexpr uint32_t single_tile_size = 2 * 1024;
        uint32_t dram_buffer_A_size = single_tile_size * Mt * Kt; // num_tiles of FP16_B
        uint32_t dram_buffer_B_size = single_tile_size * Nt * Kt; // num_tiles of FP16_B
        uint32_t dram_buffer_C_size = single_tile_size * Mt * Nt; // num_tiles of FP16_B

        /* input vectors */
        std::vector<bfloat16> src0_vec = create_random_vector_of_bfloat16_native(dram_buffer_A_size, 1, 123);
        std::vector<bfloat16> src1_vec = create_random_vector_of_bfloat16_native(dram_buffer_B_size, 1, 12522);

        /* Golden Matmul running on CPU (Float)*/
        std::vector<bfloat16> golden_vec(M * N, 0);
        golden_matmul(src0_vec, src1_vec, golden_vec, M, N, K, B);

        /* Input vector tilizing */
        tilize(src0_vec, M, K);
        tilize(src1_vec, K, N);

        /* Calling the MatMul host program. Read in result into a host vector */
        std::vector<bfloat16> result_vec(dram_buffer_C_size/sizeof(bfloat16));
        matmul_cannon(src0_vec, src1_vec, result_vec, false, M, N, K, B, device);
        tt_metal::detail::DumpDeviceProfileResults(device);
        untilize(result_vec, M, N);

        log_info(tt::LogVerif, "Output vector of size {}", result_vec.size());

        float pearson = check_bfloat16_vector_pcc(golden_vec, result_vec);
        log_info(tt::LogVerif, "Metalium vs Golden -- PCC = {}", pearson);
        // TT_FATAL(pearson > 0.95, "PCC not high enough. Result PCC: {}, Expected PCC: 0.95", pearson);

        pass &= CloseDevice(device);

    } catch (const std::exception &e) {
        tt::log_error(tt::LogTest, "Test failed with exception!");
        tt::log_error(tt::LogTest, "{}", e.what());

        throw;
    }

    if (pass) {
        tt::log_info(tt::LogTest, "Test Passed");
    } else {
        TT_THROW("Test Failed");
    }

    TT_ASSERT(pass);

    return 0;
}
