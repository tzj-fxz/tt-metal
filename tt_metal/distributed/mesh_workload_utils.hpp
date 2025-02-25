// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <host_api.hpp>
#include <mesh_common.hpp>

// Utility functions for dispatch MeshWorkloads
// Used by MeshCommandQueue
namespace tt::tt_metal::distributed {

void write_go_signal(
    uint8_t cq_id,
    IDevice* device,
    SubDeviceId sub_device_id,
    SystemMemoryManager& sysmem_manager,
    uint32_t expected_num_workers_completed,
    CoreCoord dispatch_core,
    bool send_mcast,
    bool send_unicasts,
    int num_unicast_txns = -1);

}  // namespace tt::tt_metal::distributed
