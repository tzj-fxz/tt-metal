// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <random>

#include <tt-metalium/distributed.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/bfloat16.hpp>

#include "tests/tt_metal/tt_metal/common/multi_device_fixture.hpp"
#include "tests/tt_metal/distributed/utils.hpp"

namespace tt::tt_metal::distributed::test {
namespace {

struct CBConfig {
    uint32_t cb_id = 0;
    uint32_t num_pages = 0;
    uint32_t page_size = 0;
    tt::DataFormat data_format;
};

std::vector<CBHandle> initialize_dummy_circular_buffers(
    Program& program, const CoreRangeSet& cr_set, const std::vector<CBConfig>& cb_configs) {
    std::vector<CBHandle> cb_handles;
    for (uint32_t i = 0; i < cb_configs.size(); i++) {
        const CBConfig& cb_config = cb_configs[i];
        const uint32_t cb_id = cb_config.cb_id;
        const uint32_t cb_num_pages = cb_config.num_pages;
        const uint32_t page_size = cb_config.page_size;
        const uint32_t cb_size = cb_num_pages * page_size;
        const tt::DataFormat data_format = cb_config.data_format;
        const CircularBufferConfig circular_buffer_config =
            CircularBufferConfig(cb_size, {{cb_id, data_format}}).set_page_size(cb_id, page_size);
        const CBHandle cb_handle = CreateCircularBuffer(program, cr_set, circular_buffer_config);
        cb_handles.push_back(cb_handle);
    }
    return cb_handles;
}

void initialize_dummy_kernels(Program& program, const CoreRangeSet& cr_set) {
    auto dummy_reader_kernel = CreateKernel(
        program,
        "tt_metal/kernels/dataflow/blank.cpp",
        cr_set,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default});

    auto dummy_writer_kernel = CreateKernel(
        program,
        "tt_metal/kernels/dataflow/blank.cpp",
        cr_set,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});

    auto dummy_compute_kernel = CreateKernel(program, "tt_metal/kernels/compute/blank.cpp", cr_set, ComputeConfig{});
}

std::shared_ptr<Program> initialize_dummy_program(CoreCoord worker_grid_size) {
    std::shared_ptr<Program> program = std::make_shared<Program>();
    CoreRange cr = CoreRange({0, 0}, {worker_grid_size.x - 1, worker_grid_size.y - 1});
    CoreRangeSet cr_set({cr});

    CBConfig cb_config_0 = {.cb_id = 0, .num_pages = 1, .page_size = 2048, .data_format = tt::DataFormat::Float16_b};
    CBConfig cb_config_1 = {.cb_id = 1, .num_pages = 2, .page_size = 4096, .data_format = tt::DataFormat::Float16_b};
    CBConfig cb_config_2 = {.cb_id = 2, .num_pages = 2, .page_size = 2048, .data_format = tt::DataFormat::Float16_b};
    CBConfig cb_config_3 = {.cb_id = 3, .num_pages = 4, .page_size = 2048, .data_format = tt::DataFormat::Float16_b};
    std::vector<CBConfig> cb_config_vector = {cb_config_0, cb_config_1, cb_config_2, cb_config_3};

    initialize_dummy_kernels(*program, cr_set);
    initialize_dummy_circular_buffers(*program, cr_set, cb_config_vector);
    return program;
}

void verify_cb_config(
    std::shared_ptr<MeshDevice>& mesh_device,
    MeshWorkload& workload,
    std::vector<CBConfig>& golden_cb_config,
    CoreRangeSet& crs) {
    std::vector<uint32_t> cb_config_vector;
    uint32_t cb_config_buffer_size =
        NUM_CIRCULAR_BUFFERS * UINT32_WORDS_PER_LOCAL_CIRCULAR_BUFFER_CONFIG * sizeof(uint32_t);

    for (const auto& device_range : workload.get_logical_device_ranges()) {
        for (std::size_t logical_x = device_range.start_coord.x; logical_x < device_range.end_coord.x; logical_x++) {
            for (std::size_t logical_y = device_range.start_coord.y; logical_y < device_range.end_coord.y;
                 logical_y++) {
                auto device = mesh_device->get_device(logical_y, logical_x);
                uint32_t l1_unreserved_base = device->allocator()->get_base_allocator_addr(HalMemType::L1);
                for (const auto& core_range : crs.ranges()) {
                    for (const auto& core_coord : core_range) {
                        ::tt::tt_metal::detail::ReadFromDeviceL1(
                            device,
                            core_coord,
                            workload.get_cb_base_addr(mesh_device, core_coord, CoreType::WORKER),
                            cb_config_buffer_size,
                            cb_config_vector);

                        uint32_t cb_addr = l1_unreserved_base;
                        for (uint32_t i = 0; i < golden_cb_config.size(); i++) {
                            const uint32_t index = golden_cb_config[i].cb_id * sizeof(uint32_t);
                            const uint32_t cb_num_pages = golden_cb_config[i].num_pages;
                            const uint32_t cb_size = cb_num_pages * golden_cb_config[i].page_size;
                            const bool addr_match = cb_config_vector.at(index) == cb_addr;
                            const bool size_match = cb_config_vector.at(index + 1) == cb_size;
                            const bool num_pages_match = cb_config_vector.at(index + 2) == cb_num_pages;
                            EXPECT_TRUE(addr_match);
                            EXPECT_TRUE(size_match);
                            EXPECT_TRUE(num_pages_match);

                            cb_addr += cb_size;
                        }
                    }
                }
            }
        }
    }
}

void validate_sems(
    std::shared_ptr<MeshDevice>& mesh_device,
    IDevice* device,
    CoreRange& crs,
    MeshWorkload& mesh_workload,
    std::vector<uint32_t>& expected_semaphore_values) {
    for (const auto& core : crs) {
        const uint32_t sem_buffer_size = mesh_workload.get_sem_size(mesh_device, core, CoreType::WORKER);
        const uint32_t sem_buffer_base = mesh_workload.get_sem_base_addr(mesh_device, core, CoreType::WORKER);
        std::vector<uint32_t> readback_sem_vals;
        ::tt::tt_metal::detail::ReadFromDeviceL1(device, core, sem_buffer_base, sem_buffer_size, readback_sem_vals);
        uint32_t sem_idx = 0;
        for (uint32_t i = 0; i < readback_sem_vals.size();
             i += (hal.get_alignment(HalMemType::L1) / sizeof(uint32_t))) {
            EXPECT_EQ(readback_sem_vals[i], expected_semaphore_values[sem_idx]);
            sem_idx++;
        }
    }
}

using MeshWorkloadTestT3000 = T3000MeshDeviceFixture;
using MeshWorkloadTestSuite = GenericMeshDeviceFixture;

TEST_F(MeshWorkloadTestT3000, MeshWorkloadOnActiveEthAsserts) {
    // A MeshWorkload cannot be run on ethernet core - Runtime should assert if the
    // user tries this. Verify this functionality here.
    std::shared_ptr<MeshWorkload> workload = std::make_shared<MeshWorkload>();
    uint32_t x_end = mesh_device_->num_cols();
    uint32_t y_end = mesh_device_->num_rows();
    uint32_t seed = 0;
    for (std::size_t logical_x = 0; logical_x < x_end; logical_x++) {
        for (std::size_t logical_y = 0; logical_y < y_end; logical_y++) {
            IDevice* device = mesh_device_->get_device(logical_y, logical_x);
            auto programs = tt::tt_metal::distributed::test::utils::create_random_programs(
                1, mesh_device_->compute_with_storage_grid_size(), seed, device->get_active_ethernet_cores(true));
            LogicalDeviceRange devices = {{logical_x, logical_y}, {logical_x, logical_y}};
            AddProgramToMeshWorkload(*workload, *programs[0], devices);
        }
    }
    EXPECT_THROW(EnqueueMeshWorkload(mesh_device_->mesh_command_queue(), *workload, false), std::exception);
}

TEST_F(MeshWorkloadTestT3000, SimultaneousMeshWorkloads) {
    uint32_t num_programs = 100;
    uint32_t num_heterogeneous_programs = 64;
    uint32_t num_iterations = 1000;
    auto random_seed = 0;
    uint32_t seed = tt::parse_env("TT_METAL_SEED", random_seed);
    log_info(tt::LogTest, "Using Test Seed: {}", seed);
    srand(seed);

    log_info("Create MeshWorkloads with multiple programs each");

    auto programs = tt::tt_metal::distributed::test::utils::create_random_programs(
        num_programs, mesh_device_->compute_with_storage_grid_size(), seed);
    std::vector<std::shared_ptr<MeshWorkload>> mesh_workloads = {};

    log_info(tt::LogTest, "Compile and load {} MeshWorkloads", num_programs);
    for (int i = 0; i < num_programs; i += 2) {
        std::shared_ptr<MeshWorkload> random_workload = std::make_shared<MeshWorkload>();
        if (i % 2) {
            LogicalDeviceRange devices_0 = LogicalDeviceRange({0, 0}, {3, 0});
            LogicalDeviceRange devices_1 = LogicalDeviceRange({0, 1}, {3, 1});
            AddProgramToMeshWorkload(*random_workload, *programs[i], devices_0);
            AddProgramToMeshWorkload(*random_workload, *programs[i + 1], devices_1);
        } else {
            LogicalDeviceRange devices_0 = LogicalDeviceRange({0, 0}, {1, 1});
            LogicalDeviceRange devices_1 = LogicalDeviceRange({2, 0}, {3, 1});
            AddProgramToMeshWorkload(*random_workload, *programs[i], devices_0);
            AddProgramToMeshWorkload(*random_workload, *programs[i + 1], devices_1);
        }
        EnqueueMeshWorkload(mesh_device_->mesh_command_queue(), *random_workload, false);
        mesh_workloads.push_back(random_workload);
    }
    programs = tt::tt_metal::distributed::test::utils::create_random_programs(
        num_programs, mesh_device_->compute_with_storage_grid_size(), seed);
    for (int i = 0; i < num_programs; i += 4) {
        std::shared_ptr<MeshWorkload> random_workload = std::make_shared<MeshWorkload>();
        LogicalDeviceRange devices_0 = LogicalDeviceRange({0, 0}, {0, 1});
        LogicalDeviceRange devices_1 = LogicalDeviceRange({1, 0}, {1, 1});
        LogicalDeviceRange devices_2 = LogicalDeviceRange({2, 0}, {2, 1});
        LogicalDeviceRange devices_3 = LogicalDeviceRange({3, 0}, {3, 1});
        AddProgramToMeshWorkload(*random_workload, *programs[i], devices_0);
        AddProgramToMeshWorkload(*random_workload, *programs[i + 1], devices_1);
        AddProgramToMeshWorkload(*random_workload, *programs[i + 2], devices_2);
        AddProgramToMeshWorkload(*random_workload, *programs[i + 3], devices_3);
        EnqueueMeshWorkload(mesh_device_->mesh_command_queue(), *random_workload, false);
        mesh_workloads.push_back(random_workload);
    }
    programs = tt::tt_metal::distributed::test::utils::create_random_programs(
        num_heterogeneous_programs, mesh_device_->compute_with_storage_grid_size(), seed);
    for (int i = 0; i < num_heterogeneous_programs; i += 8) {
        std::shared_ptr<MeshWorkload> random_workload = std::make_shared<MeshWorkload>();
        LogicalDeviceRange devices_0 = LogicalDeviceRange({0, 0}, {0, 0});
        LogicalDeviceRange devices_1 = LogicalDeviceRange({0, 1}, {0, 1});
        LogicalDeviceRange devices_2 = LogicalDeviceRange({1, 0}, {1, 0});
        LogicalDeviceRange devices_3 = LogicalDeviceRange({1, 1}, {1, 1});
        LogicalDeviceRange devices_4 = LogicalDeviceRange({2, 0}, {2, 0});
        LogicalDeviceRange devices_5 = LogicalDeviceRange({2, 1}, {2, 1});
        LogicalDeviceRange devices_6 = LogicalDeviceRange({3, 0}, {3, 0});
        LogicalDeviceRange devices_7 = LogicalDeviceRange({3, 1}, {3, 1});

        AddProgramToMeshWorkload(*random_workload, *programs[i], devices_0);
        AddProgramToMeshWorkload(*random_workload, *programs[i + 1], devices_1);
        AddProgramToMeshWorkload(*random_workload, *programs[i + 2], devices_2);
        AddProgramToMeshWorkload(*random_workload, *programs[i + 3], devices_3);
        AddProgramToMeshWorkload(*random_workload, *programs[i + 4], devices_4);
        AddProgramToMeshWorkload(*random_workload, *programs[i + 5], devices_5);
        AddProgramToMeshWorkload(*random_workload, *programs[i + 6], devices_6);
        AddProgramToMeshWorkload(*random_workload, *programs[i + 7], devices_7);
        EnqueueMeshWorkload(mesh_device_->mesh_command_queue(), *random_workload, false);
        mesh_workloads.push_back(random_workload);
    }

    for (int i = 0; i < num_iterations; i++) {
        if (i % 100 == 0) {
            log_info(tt::LogTest, "Run MeshWorkloads for iteration {}", i);
        }
        for (auto& workload : mesh_workloads) {
            EnqueueMeshWorkload(mesh_device_->mesh_command_queue(), *workload, false);
        }
    }
    Finish(mesh_device_->mesh_command_queue());
}

// MeshWorkload tests on N300 and T3000
TEST_F(MeshWorkloadTestSuite, RandomizedMeshWorkload) {
    uint32_t num_programs = 60;
    uint32_t num_iterations = 1500;
    auto random_seed = 10;
    uint32_t seed = tt::parse_env("TT_METAL_SEED", random_seed);
    log_info(tt::LogTest, "Using Test Seed: {}", seed);
    srand(seed);
    log_info("Create {} MeshWorkloads", num_programs);
    auto programs = tt::tt_metal::distributed::test::utils::create_random_programs(
        num_programs, mesh_device_->compute_with_storage_grid_size(), seed);
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> gen_x(1, mesh_device_->num_cols());
    std::uniform_int_distribution<int> gen_y(1, mesh_device_->num_rows());
    std::vector<std::shared_ptr<MeshWorkload>> mesh_workloads = {};

    // Create multiple mesh workloads on grids of random sizes.
    // Compile the workload (lower + send binaries to mesh device here as well)
    log_info(tt::LogTest, "Compile and load {} MeshWorkloads", num_programs);
    for (int i = 0; i < num_programs; i += 1) {
        // Choose a grid of random dimensions and run a MeshWorkload on it
        LogicalDeviceRange device_range = LogicalDeviceRange({0, 0}, {gen_x(rng) - 1, gen_y(rng) - 1});
        auto random_workload = std::make_shared<MeshWorkload>();
        AddProgramToMeshWorkload(*random_workload, *programs[i], device_range);
        EnqueueMeshWorkload(mesh_device_->mesh_command_queue(), *random_workload, false);
        mesh_workloads.push_back(random_workload);
    }
    for (int i = 0; i < num_iterations; i++) {
        if (i % 100 == 0) {
            log_info(tt::LogTest, "Run MeshWorkloads for iteration {}", i);
        }
        for (auto& workload : mesh_workloads) {
            EnqueueMeshWorkload(mesh_device_->mesh_command_queue(), *workload, false);
        }
    }
    log_info(tt::LogTest, "Calling Finish");
    Finish(mesh_device_->mesh_command_queue());
}

TEST_F(MeshWorkloadTestSuite, EltwiseBinaryMeshWorkload) {
    std::vector<std::shared_ptr<MeshBuffer>> src0_bufs = {};
    std::vector<std::shared_ptr<MeshBuffer>> src1_bufs = {};
    std::vector<std::shared_ptr<MeshBuffer>> output_bufs = {};

    CoreCoord worker_grid_size = mesh_device_->compute_with_storage_grid_size();

    auto programs = tt::tt_metal::distributed::test::utils::create_eltwise_bin_programs(
        mesh_device_, src0_bufs, src1_bufs, output_bufs);
    auto mesh_workload = CreateMeshWorkload();
    LogicalDeviceRange devices_0 = LogicalDeviceRange({0, 0}, {mesh_device_->num_cols() - 1, 0});
    LogicalDeviceRange devices_1 = LogicalDeviceRange(
        {0, mesh_device_->num_rows() - 1}, {mesh_device_->num_cols() - 1, mesh_device_->num_rows() - 1});
    AddProgramToMeshWorkload(mesh_workload, *programs[0], devices_0);
    AddProgramToMeshWorkload(mesh_workload, *programs[1], devices_1);
    std::vector<uint32_t> src0_vec = create_constant_vector_of_bfloat16(src0_bufs[0]->size(), 2);
    std::vector<uint32_t> src1_vec = create_constant_vector_of_bfloat16(src1_bufs[0]->size(), 3);

    for (std::size_t col_idx = 0; col_idx < worker_grid_size.x; col_idx++) {
        for (std::size_t row_idx = 0; row_idx < worker_grid_size.y; row_idx++) {
            EnqueueWriteMeshBuffer(
                mesh_device_->mesh_command_queue(), src0_bufs[col_idx * worker_grid_size.y + row_idx], src0_vec);
            EnqueueWriteMeshBuffer(
                mesh_device_->mesh_command_queue(), src1_bufs[col_idx * worker_grid_size.y + row_idx], src1_vec);
        }
    }

    // Run workload multiple times
    for (int i = 0; i < 1000; i++) {
        EnqueueMeshWorkload(mesh_device_->mesh_command_queue(), mesh_workload, false);
    }

    for (std::size_t logical_y = 0; logical_y < mesh_device_->num_rows(); logical_y++) {
        for (std::size_t logical_x = 0; logical_x < mesh_device_->num_cols(); logical_x++) {
            for (std::size_t col_idx = 0; col_idx < worker_grid_size.x; col_idx++) {
                for (std::size_t row_idx = 0; row_idx < worker_grid_size.y; row_idx++) {
                    std::vector<bfloat16> dst_vec = {};
                    ReadShard(
                        mesh_device_->mesh_command_queue(),
                        dst_vec,
                        output_bufs[col_idx * worker_grid_size.y + row_idx],
                        MeshCoordinate(logical_y, logical_x));
                    if (logical_y == 0) {
                        for (int i = 0; i < dst_vec.size(); i++) {
                            EXPECT_EQ(dst_vec[i].to_float(), 5);
                        }
                    } else {
                        for (int i = 0; i < dst_vec.size(); i++) {
                            EXPECT_EQ(dst_vec[i].to_float(), 6);
                        }
                    }
                }
            }
        }
    }
}

TEST_F(MeshWorkloadTestSuite, MeshWorkloadSanity) {
    CoreCoord worker_grid_size = mesh_device_->compute_with_storage_grid_size();
    uint32_t single_tile_size = ::tt::tt_metal::detail::TileSize(DataFormat::Float16_b);

    uint32_t num_tiles = 1;
    uint32_t dram_buffer_size = single_tile_size * num_tiles;
    // Create buffers
    std::vector<std::shared_ptr<MeshBuffer>> input_buffers = {};
    std::vector<std::shared_ptr<MeshBuffer>> output_buffers = {};

    ReplicatedBufferConfig global_buffer_config{.size = dram_buffer_size};

    DeviceLocalBufferConfig per_device_buffer_config{
        .page_size = dram_buffer_size,
        .buffer_type = tt_metal::BufferType::DRAM,
        .buffer_layout = TensorMemoryLayout::INTERLEAVED,
        .bottom_up = true};

    for (std::size_t col_idx = 0; col_idx < worker_grid_size.x; col_idx++) {
        for (std::size_t row_idx = 0; row_idx < worker_grid_size.y; row_idx++) {
            input_buffers.push_back(
                MeshBuffer::create(global_buffer_config, per_device_buffer_config, mesh_device_.get()));
            output_buffers.push_back(
                MeshBuffer::create(global_buffer_config, per_device_buffer_config, mesh_device_.get()));
        }
    }

    // Create MeshWorkload
    Program program = CreateProgram();
    auto full_grid = CoreRange({0, 0}, {worker_grid_size.x - 1, worker_grid_size.y - 1});
    auto reader_writer_kernel = CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/misc/full_grid_eltwise_device_reuse.cpp",
        full_grid,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});

    auto sem_scaling_factor = 2;
    auto scaling_sem_idx = CreateSemaphore(program, full_grid, sem_scaling_factor);
    uint32_t scaling_height_toggle = 16;
    constexpr uint32_t src0_cb_index = CBIndex::c_0;
    CircularBufferConfig cb_src0_config =
        CircularBufferConfig(dram_buffer_size, {{src0_cb_index, DataFormat::Float16_b}})
            .set_page_size(src0_cb_index, single_tile_size);
    uint32_t add_factor = 64;
    for (std::size_t col_idx = 0; col_idx < worker_grid_size.x; col_idx++) {
        for (std::size_t row_idx = 0; row_idx < worker_grid_size.y; row_idx++) {
            CoreCoord curr_core = {col_idx, row_idx};
            SetRuntimeArgs(
                program,
                reader_writer_kernel,
                curr_core,
                {input_buffers.at(col_idx * worker_grid_size.y + row_idx)->address(),
                 output_buffers.at(col_idx * worker_grid_size.y + row_idx)->address(),
                 0, /* src_bank_id */
                 0, /* dst_bank_id */
                 add_factor,
                 constants::TILE_HEIGHT,
                 constants::TILE_WIDTH,
                 scaling_sem_idx,
                 scaling_height_toggle});
            CBHandle cb_src0 = CreateCircularBuffer(program, curr_core, cb_src0_config);
        }
    }
    auto program_1 = initialize_dummy_program(worker_grid_size);
    auto mesh_workload = MeshWorkload();
    LogicalDeviceRange devices_0 = LogicalDeviceRange({0, 0}, {mesh_device_->num_cols() - 1, 0});
    LogicalDeviceRange devices_1 = LogicalDeviceRange(
        {0, mesh_device_->num_rows() - 1}, {mesh_device_->num_cols() - 1, mesh_device_->num_rows() - 1});
    AddProgramToMeshWorkload(mesh_workload, program, devices_0);
    AddProgramToMeshWorkload(mesh_workload, *program_1, devices_1);

    std::size_t buffer_idx = 0;
    std::vector<uint32_t> src_vec = create_constant_vector_of_bfloat16(dram_buffer_size, 1);

    for (std::size_t col_idx = 0; col_idx < worker_grid_size.x; col_idx++) {
        for (std::size_t row_idx = 0; row_idx < worker_grid_size.y; row_idx++) {
            EnqueueWriteMeshBuffer(
                mesh_device_->mesh_command_queue(), input_buffers[col_idx * worker_grid_size.y + row_idx], src_vec);
        }
    }

    for (int iter = 0; iter < 100; iter++) {
        log_info(LogTest, "Run iter {}", iter);
        if (iter) {
            auto& program = mesh_workload.get_program_on_device_range(devices_0);
            auto& rtas = GetRuntimeArgs(program, reader_writer_kernel);
            for (auto core : full_grid) {
                rtas[core.x][core.y].at(4) = ((iter % 2) + 1) * add_factor;
            }
        }
        EnqueueMeshWorkload(mesh_device_->mesh_command_queue(), mesh_workload, false);
        buffer_idx = 0;
        for (std::size_t logical_x = devices_0.start_coord.x; logical_x < devices_0.end_coord.x; logical_x++) {
            for (std::size_t logical_y = devices_0.start_coord.y; logical_y < devices_0.end_coord.y; logical_y++) {
                for (std::size_t col_idx = 0; col_idx < worker_grid_size.x; col_idx++) {
                    for (std::size_t row_idx = 0; row_idx < worker_grid_size.y; row_idx++) {
                        std::vector<bfloat16> dst_vec = {};
                        ReadShard(
                            mesh_device_->mesh_command_queue(),
                            dst_vec,
                            output_buffers[col_idx * worker_grid_size.y + row_idx],
                            MeshCoordinate(logical_y, logical_x));
                        for (int i = 0; i < dst_vec.size(); i++) {
                            float ref_val = std::pow(2, (iter % 2) + 1);
                            if (i >= 512) {
                                ref_val = std::pow(2, 2 * ((iter % 2) + 1));
                            }
                            EXPECT_EQ(dst_vec[i].to_float(), ref_val);
                        }
                    }
                }
            }
        }
    }
}

TEST_F(MeshWorkloadTestSuite, MeshWorkloadCBUpdate) {
    std::shared_ptr<Program> program = std::make_shared<Program>();
    CoreCoord worker_grid_size = mesh_device_->compute_with_storage_grid_size();
    CoreRange cr = CoreRange({0, 0}, {worker_grid_size.x - 1, worker_grid_size.y - 1});
    CoreRangeSet cr_set({cr});

    CBConfig cb_config_0 = {.cb_id = 0, .num_pages = 1, .page_size = 2048, .data_format = tt::DataFormat::Float16_b};
    CBConfig cb_config_1 = {.cb_id = 1, .num_pages = 2, .page_size = 4096, .data_format = tt::DataFormat::Float16_b};
    CBConfig cb_config_2 = {.cb_id = 2, .num_pages = 2, .page_size = 2048, .data_format = tt::DataFormat::Float16_b};
    CBConfig cb_config_3 = {.cb_id = 3, .num_pages = 4, .page_size = 2048, .data_format = tt::DataFormat::Float16_b};
    std::vector<CBConfig> cb_config_vector = {cb_config_0, cb_config_1, cb_config_2, cb_config_3};

    const std::vector<CBHandle>& cb_handles = initialize_dummy_circular_buffers(*program, cr_set, cb_config_vector);
    initialize_dummy_kernels(*program, cr_set);

    auto mesh_workload = CreateMeshWorkload();
    LogicalDeviceRange devices =
        LogicalDeviceRange({0, 0}, {mesh_device_->num_cols() - 1, mesh_device_->num_rows() - 1});

    AddProgramToMeshWorkload(mesh_workload, *program, devices);
    EnqueueMeshWorkload(mesh_device_->mesh_command_queue(), mesh_workload, false);
    Finish(mesh_device_->mesh_command_queue());
    verify_cb_config(mesh_device_, mesh_workload, cb_config_vector, cr_set);

    std::vector<CBConfig> updated_cb_config_vector = cb_config_vector;
    for (uint32_t cb_id = 0; cb_id < cb_config_vector.size(); cb_id++) {
        CBConfig& cb_config = updated_cb_config_vector[cb_id];
        cb_config.num_pages *= 2;
        const uint32_t cb_size = cb_config.num_pages * cb_config.page_size;
        UpdateCircularBufferTotalSize(mesh_workload.get_program_on_device_range(devices), cb_handles[cb_id], cb_size);
    }
    EnqueueMeshWorkload(mesh_device_->mesh_command_queue(), mesh_workload, false);
    Finish(mesh_device_->mesh_command_queue());
    verify_cb_config(mesh_device_, mesh_workload, updated_cb_config_vector, cr_set);
}

TEST_F(MeshWorkloadTestSuite, MeshWorkloadSemaphoreSanity) {
    auto worker_grid_size = mesh_device_->compute_with_storage_grid_size();
    auto full_grid = CoreRange({0, 0}, {worker_grid_size.x - 1, worker_grid_size.y - 1});
    Program program;
    std::vector<uint32_t> expected_semaphore_values;

    for (uint32_t sem = 0; sem < NUM_SEMAPHORES; sem++) {
        CreateSemaphore(program, full_grid, sem);
        expected_semaphore_values.push_back(sem);
    }
    auto mesh_workload = CreateMeshWorkload();
    LogicalDeviceRange devices =
        LogicalDeviceRange({0, 0}, {mesh_device_->num_cols() - 1, mesh_device_->num_rows() - 1});
    AddProgramToMeshWorkload(mesh_workload, program, devices);
    EnqueueMeshWorkload(mesh_device_->mesh_command_queue(), mesh_workload, false);
    Finish(mesh_device_->mesh_command_queue());

    for (const auto device : mesh_device_->get_devices()) {
        validate_sems(mesh_device_, device, full_grid, mesh_workload, expected_semaphore_values);
    }
}

TEST_F(MeshWorkloadTestSuite, MeshWorkloadSemaphoreDifferentPrograms) {
    auto worker_grid_size = mesh_device_->compute_with_storage_grid_size();
    auto full_grid = CoreRange({0, 0}, {worker_grid_size.x - 1, worker_grid_size.y - 1});
    Program program0;
    Program program1;
    std::vector<uint32_t> expected_semaphore_values_0;
    std::vector<uint32_t> expected_semaphore_values_1;

    for (uint32_t sem = 0; sem < NUM_SEMAPHORES; sem++) {
        CreateSemaphore(program0, full_grid, sem);
        expected_semaphore_values_0.push_back(sem);

        CreateSemaphore(program1, full_grid, sem + 1);
        expected_semaphore_values_1.push_back(sem + 1);
    }
    auto mesh_workload = CreateMeshWorkload();
    LogicalDeviceRange devices_0 = LogicalDeviceRange({0, 0}, {mesh_device_->num_cols() - 1, 0});
    LogicalDeviceRange devices_1 = LogicalDeviceRange(
        {0, mesh_device_->num_rows() - 1}, {mesh_device_->num_cols() - 1, mesh_device_->num_rows() - 1});

    AddProgramToMeshWorkload(mesh_workload, program0, devices_0);
    AddProgramToMeshWorkload(mesh_workload, program1, devices_1);
    EnqueueMeshWorkload(mesh_device_->mesh_command_queue(), mesh_workload, false);
    Finish(mesh_device_->mesh_command_queue());

    for (std::size_t logical_x = devices_0.start_coord.x; logical_x < devices_0.end_coord.x; logical_x++) {
        for (std::size_t logical_y = devices_0.start_coord.y; logical_y < devices_0.end_coord.y; logical_y++) {
            auto device = mesh_device_->get_device(logical_y, logical_x);
            validate_sems(mesh_device_, device, full_grid, mesh_workload, expected_semaphore_values_0);
        }
    }

    for (std::size_t logical_x = devices_1.start_coord.x; logical_x < devices_1.end_coord.x; logical_x++) {
        for (std::size_t logical_y = devices_1.start_coord.y; logical_y < devices_1.end_coord.y; logical_y++) {
            auto device = mesh_device_->get_device(logical_y, logical_x);
            validate_sems(mesh_device_, device, full_grid, mesh_workload, expected_semaphore_values_1);
        }
    }
}

}  // namespace
}  // namespace tt::tt_metal::distributed::test
