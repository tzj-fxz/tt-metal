#!/bin/bash
set -eo pipefail

# Exit immediately if ARCH_NAME is not set or empty
if [ -z "${ARCH_NAME}" ]; then
  echo "Error: ARCH_NAME is not set. Exiting." >&2
  exit 1
fi

run_t3000_ttmetal_tests() {
  # Record the start time
  fail=0
  start_time=$(date +%s)

  echo "LOG_METAL: Running run_t3000_ttmetal_tests"
  ./build/test/tt_metal/distributed/distributed_unit_tests_${ARCH_NAME}
  TT_METAL_SLOW_DISPATCH_MODE=1 ./build/test/tt_metal/unit_tests_eth_${ARCH_NAME} --gtest_filter="DeviceFixture.ActiveEthKernelsDirectSendAllConnectedChips" ; fail+=$?
  TT_METAL_SLOW_DISPATCH_MODE=1 ./build/test/tt_metal/unit_tests_eth_${ARCH_NAME} --gtest_filter="DeviceFixture.ActiveEthKernelsSendInterleavedBufferAllConnectedChips" ; fail+=$?
  TT_METAL_SLOW_DISPATCH_MODE=1 ./build/test/tt_metal/unit_tests_eth_${ARCH_NAME} --gtest_filter="DeviceFixture.ActiveEthKernelsDirectRingGatherAllChips" ; fail+=$?
  TT_METAL_SLOW_DISPATCH_MODE=1 ./build/test/tt_metal/unit_tests_eth_${ARCH_NAME} --gtest_filter="DeviceFixture.ActiveEthKernelsInterleavedRingGatherAllChips" ; fail+=$?
  TT_METAL_ENABLE_REMOTE_CHIP=1 ./build/test/tt_metal/unit_tests_dispatch --gtest_filter="CommandQueueSingleCard*Fixture.*" ; fail+=$?
  ./build/test/tt_metal/unit_tests_dispatch --gtest_filter="CommandQueueMultiDevice*Fixture.*" ; fail+=$?
  ./build/test/tt_metal/unit_tests_debug_tools_${ARCH_NAME} --gtest_filter="DPrintFixture.*:WatcherFixture.*" ; fail+=$?

  # Programming examples
  ./build/programming_examples/distributed/distributed_program_dispatch
  ./build/programming_examples/distributed/distributed_buffer_rw
  ./build/programming_examples/distributed/distributed_eltwise_add
  ./build/programming_examples/distributed/distributed_trace_and_events

  # Record the end time
  end_time=$(date +%s)
  duration=$((end_time - start_time))
  echo "LOG_METAL: run_t3000_ttmetal_tests $duration seconds to complete"
  if [[ $fail -ne 0 ]]; then
    exit 1
  fi
}

run_t3000_ttfabric_tests() {
  # Record the start time
  fail=0
  start_time=$(date +%s)

  echo "LOG_METAL: Running run_t3000_ttfabric_tests"
  TT_METAL_SLOW_DISPATCH_MODE=1 ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter=ControlPlaneFixture.*T3k*
  TT_METAL_SLOW_DISPATCH_MODE=1 ./build/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="FabricFixture.*"
  # Unicast tests
  TT_METAL_SLOW_DISPATCH_MODE=1 ./build/test/tt_metal/perf_microbenchmark/routing/test_tt_fabric_sanity_wormhole_b0 --fabric_command 1 --board_type t3k --data_kb_per_tx 10 --num_src_endpoints 20 --num_dest_endpoints 8 --num_links 16
  TT_METAL_SLOW_DISPATCH_MODE=1 ./build/test/tt_metal/perf_microbenchmark/routing/test_tt_fabric_sanity_wormhole_b0 --fabric_command 64 --board_type t3k --data_kb_per_tx 10 --num_src_endpoints 20 --num_dest_endpoints 8 --num_links 16
  TT_METAL_SLOW_DISPATCH_MODE=1 ./build/test/tt_metal/perf_microbenchmark/routing/test_tt_fabric_sanity_wormhole_b0 --fabric_command 65 --board_type t3k --data_kb_per_tx 10 --num_src_endpoints 20 --num_dest_endpoints 8 --num_links 16
  TT_METAL_SLOW_DISPATCH_MODE=1 ./build/test/tt_metal/perf_microbenchmark/routing/test_tt_fabric_sanity_wormhole_b0 --fabric_command 1 --board_type t3k --data_kb_per_tx 10 --num_src_endpoints 20 --num_dest_endpoints 8 --num_links 16 --metal_fabric_init_level 1
  # Line Mcast tests
  TT_METAL_SLOW_DISPATCH_MODE=1 ./build/test/tt_metal/perf_microbenchmark/routing/test_tt_fabric_sanity_wormhole_b0 --fabric_command 1 --board_type t3k --data_kb_per_tx 10 --num_src_endpoints 20 --num_dest_endpoints 8 --num_links 16 --e_depth 3
  TT_METAL_SLOW_DISPATCH_MODE=1 ./build/test/tt_metal/perf_microbenchmark/routing/test_tt_fabric_sanity_wormhole_b0 --fabric_command 1 --board_type t3k --data_kb_per_tx 10 --num_src_endpoints 20 --num_dest_endpoints 8 --num_links 16 --w_depth 3
  TT_METAL_SLOW_DISPATCH_MODE=1 ./build/test/tt_metal/perf_microbenchmark/routing/test_tt_fabric_sanity_wormhole_b0 --fabric_command 1 --board_type t3k --data_kb_per_tx 10 --num_src_endpoints 20 --num_dest_endpoints 8 --num_links 16 --n_depth 1
  TT_METAL_SLOW_DISPATCH_MODE=1 ./build/test/tt_metal/perf_microbenchmark/routing/test_tt_fabric_sanity_wormhole_b0 --fabric_command 1 --board_type t3k --data_kb_per_tx 10 --num_src_endpoints 20 --num_dest_endpoints 8 --num_links 16 --s_depth 1
  TT_METAL_SLOW_DISPATCH_MODE=1 ./build/test/tt_metal/perf_microbenchmark/routing/test_tt_fabric_sanity_wormhole_b0 --fabric_command 1 --board_type t3k --data_kb_per_tx 10 --num_src_endpoints 20 --num_dest_endpoints 8 --num_links 16 --e_depth 3 --metal_fabric_init_level 1

  # Record the end time
  end_time=$(date +%s)
  duration=$((end_time - start_time))
  echo "LOG_METAL: run_t3000_ttmetal_tests $duration seconds to complete"
  if [[ $fail -ne 0 ]]; then
    exit 1
  fi
}

run_t3000_ttnn_tests() {
  # Record the start time
  fail=0
  start_time=$(date +%s)

  echo "LOG_METAL: Running run_t3000_ttnn_tests"
  WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml ./build/test/ttnn/test_multi_device
  WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml ./build/test/ttnn/unit_tests_ttnn
  ./build/test/ttnn/unit_tests_ttnn_ccl
  WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml pytest tests/ttnn/unit_tests/test_multi_device_trace.py ; fail+=$?
  WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml pytest tests/ttnn/unit_tests/test_multi_device_events.py ; fail+=$?
  pytest -n auto tests/ttnn/unit_tests/test_multi_device.py ; fail+=$?
  pytest -n auto tests/ttnn/unit_tests/test_multi_device_async.py ; fail+=$?
  pytest tests/ttnn/distributed/test_tensor_parallel_example_T3000.py ; fail+=$?
  # Record the end time
  end_time=$(date +%s)
  duration=$((end_time - start_time))
  echo "LOG_METAL: run_t3000_ttnn_tests $duration seconds to complete"
  if [[ $fail -ne 0 ]]; then
    exit 1
  fi
}

run_t3000_falcon7b_tests() {
  # Record the start time
  fail=0
  start_time=$(date +%s)

  echo "LOG_METAL: Running run_t3000_falcon7b_tests"

  pytest -n auto models/demos/ttnn_falcon7b/tests/multi_chip/test_falcon_mlp.py ; fail+=$?
  pytest -n auto models/demos/ttnn_falcon7b/tests/multi_chip/test_falcon_attention.py ; fail+=$?
  pytest -n auto models/demos/ttnn_falcon7b/tests/multi_chip/test_falcon_decoder.py ; fail+=$?
  #pytest models/demos/ttnn_falcon7b/tests/multi_chip/test_falcon_causallm.py

  # Record the end time
  end_time=$(date +%s)
  duration=$((end_time - start_time))
  echo "LOG_METAL: run_t3000_falcon7b_tests $duration seconds to complete"
  if [[ $fail -ne 0 ]]; then
    exit 1
  fi
}

run_t3000_falcon40b_tests() {
  # Record the start time
  fail=0
  start_time=$(date +%s)

  echo "LOG_METAL: Running run_t3000_falcon40b_tests"

  WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml pytest -n auto models/demos/t3000/falcon40b/tests/ci/test_falcon_end_to_end_1_layer_t3000.py ; fail+=$?


  # Record the end time
  end_time=$(date +%s)
  duration=$((end_time - start_time))
  echo "LOG_METAL: run_t3000_falcon40b_tests $duration seconds to complete"
  if [[ $fail -ne 0 ]]; then
    exit 1
  fi
}

run_t3000_llama3-small_tests() {
  # Record the start time
  fail=0
  start_time=$(date +%s)

  echo "LOG_METAL: Running run_t3000_llama3-small_tests"

  wh_arch_yaml=wormhole_b0_80_arch_eth_dispatch.yaml
  # Llama3.2-1B
  llama1b=/mnt/MLPerf/tt_dnn-models/llama/Llama3.2-1B-Instruct/
  # Llama3.2-3B
  llama3b=/mnt/MLPerf/tt_dnn-models/llama/Llama3.2-3B-Instruct/
  # Llama3.1-8B
  llama8b=/mnt/MLPerf/tt_dnn-models/llama/Meta-Llama-3.1-8B-Instruct/

  # Run all Llama3 tests for 1B, 3B and 8B weights
  for llama_dir in "$llama1b" "$llama3b" "$llama8b"; do
    LLAMA_DIR=$llama_dir WH_ARCH_YAML=$wh_arch_yaml pytest -n auto models/demos/llama3/tests/test_llama_attention.py ; fail+=$?
    LLAMA_DIR=$llama_dir WH_ARCH_YAML=$wh_arch_yaml pytest -n auto models/demos/llama3/tests/test_llama_attention_prefill.py ; fail+=$?
    LLAMA_DIR=$llama_dir WH_ARCH_YAML=$wh_arch_yaml pytest -n auto models/demos/llama3/tests/test_llama_embedding.py ; fail+=$?
    LLAMA_DIR=$llama_dir WH_ARCH_YAML=$wh_arch_yaml pytest -n auto models/demos/llama3/tests/test_llama_mlp.py ; fail+=$?
    LLAMA_DIR=$llama_dir WH_ARCH_YAML=$wh_arch_yaml pytest -n auto models/demos/llama3/tests/test_llama_rms_norm.py ; fail+=$?
    LLAMA_DIR=$llama_dir WH_ARCH_YAML=$wh_arch_yaml pytest -n auto models/demos/llama3/tests/test_llama_decoder.py ; fail+=$?
    LLAMA_DIR=$llama_dir WH_ARCH_YAML=$wh_arch_yaml pytest -n auto models/demos/llama3/tests/test_llama_decoder_prefill.py ; fail+=$?
    echo "LOG_METAL: Llama3 tests for $llama_dir completed"
  done

  # Record the end time
  end_time=$(date +%s)
  duration=$((end_time - start_time))
  echo "LOG_METAL: run_t3000_llama3-small_tests $duration seconds to complete"
  if [[ $fail -ne 0 ]]; then
    exit 1
  fi
}

run_t3000_llama3.2-11b_tests() {
  # Record the start time
  fail=0
  start_time=$(date +%s)

  echo "LOG_METAL: Running run_t3000_llama3.2-11b_tests"

  wh_arch_yaml=wormhole_b0_80_arch_eth_dispatch.yaml
  # Llama3.2-11B weights
  llama11b=/mnt/MLPerf/tt_dnn-models/llama/Llama3.2-11B-Vision-Instruct/

  LLAMA_DIR=$llama11b WH_ARCH_YAML=$wh_arch_yaml pytest -n auto models/demos/llama3/tests/test_llama_attention.py ; fail+=$?
  LLAMA_DIR=$llama11b WH_ARCH_YAML=$wh_arch_yaml pytest -n auto models/demos/llama3/tests/test_llama_attention_prefill.py ; fail+=$?
  LLAMA_DIR=$llama11b WH_ARCH_YAML=$wh_arch_yaml pytest -n auto models/demos/llama3/tests/test_llama_embedding.py ; fail+=$?
  LLAMA_DIR=$llama11b WH_ARCH_YAML=$wh_arch_yaml pytest -n auto models/demos/llama3/tests/test_llama_mlp.py ; fail+=$?
  LLAMA_DIR=$llama11b WH_ARCH_YAML=$wh_arch_yaml pytest -n auto models/demos/llama3/tests/test_llama_rms_norm.py ; fail+=$?
  LLAMA_DIR=$llama11b WH_ARCH_YAML=$wh_arch_yaml pytest -n auto models/demos/llama3/tests/test_llama_decoder.py ; fail+=$?
  LLAMA_DIR=$llama11b WH_ARCH_YAML=$wh_arch_yaml pytest -n auto models/demos/llama3/tests/test_llama_decoder_prefill.py ; fail+=$?

  # Record the end time
  end_time=$(date +%s)
  duration=$((end_time - start_time))
  echo "LOG_METAL: run_t3000_llama3.2-11b_tests $duration seconds to complete"
  if [[ $fail -ne 0 ]]; then
    exit 1
  fi
}

run_t3000_llama3.1-70b_tests() {
  # Record the start time
  fail=0
  start_time=$(date +%s)

  echo "LOG_METAL: Running run_t3000_llama3.1-70b_tests"

  wh_arch_yaml=wormhole_b0_80_arch_eth_dispatch.yaml
  # Llama3.1-70B weights
  llama70b=/mnt/MLPerf/tt_dnn-models/llama/Llama3.1-70B-Instruct/

  LLAMA_DIR=$llama70b WH_ARCH_YAML=$wh_arch_yaml pytest -n auto models/demos/llama3/tests/test_llama_attention.py ; fail+=$?
  LLAMA_DIR=$llama70b WH_ARCH_YAML=$wh_arch_yaml pytest -n auto models/demos/llama3/tests/test_llama_attention_prefill.py ; fail+=$?
  LLAMA_DIR=$llama70b WH_ARCH_YAML=$wh_arch_yaml pytest -n auto models/demos/llama3/tests/test_llama_embedding.py ; fail+=$?
  LLAMA_DIR=$llama70b WH_ARCH_YAML=$wh_arch_yaml pytest -n auto models/demos/llama3/tests/test_llama_mlp.py ; fail+=$?
  LLAMA_DIR=$llama70b WH_ARCH_YAML=$wh_arch_yaml pytest -n auto models/demos/llama3/tests/test_llama_rms_norm.py ; fail+=$?
  LLAMA_DIR=$llama70b WH_ARCH_YAML=$wh_arch_yaml pytest -n auto models/demos/llama3/tests/test_llama_decoder.py ; fail+=$?
  LLAMA_DIR=$llama70b WH_ARCH_YAML=$wh_arch_yaml pytest -n auto models/demos/llama3/tests/test_llama_decoder_prefill.py ; fail+=$?

  # Record the end time
  end_time=$(date +%s)
  duration=$((end_time - start_time))
  echo "LOG_METAL: run_t3000_llama3.1-70b_tests $duration seconds to complete"
  if [[ $fail -ne 0 ]]; then
    exit 1
  fi
}

run_t3000_llama3.2-11b-vision_unit_tests() {
  # Record the start time
  fail=0
  start_time=$(date +%s)

  echo "LOG_METAL: Running run_t3000_llama3.2-11b-vision_unit_tests"

  wh_arch_yaml=wormhole_b0_80_arch_eth_dispatch.yaml
  # Llama3.2-11B
  llama11b=/mnt/MLPerf/tt_dnn-models/llama/Llama3.2-11B-Vision-Instruct/

  LLAMA_DIR=$llama11b WH_ARCH_YAML=$wh_arch_yaml pytest -n auto models/demos/llama3/tests/multimodal/test_llama_image_mlp.py ; fail+=$?
  LLAMA_DIR=$llama11b WH_ARCH_YAML=$wh_arch_yaml pytest -n auto models/demos/llama3/tests/multimodal/test_llama_image_attention.py ; fail+=$?
  LLAMA_DIR=$llama11b WH_ARCH_YAML=$wh_arch_yaml pytest -n auto models/demos/llama3/tests/multimodal/test_llama_image_block.py ; fail+=$?
  LLAMA_DIR=$llama11b WH_ARCH_YAML=$wh_arch_yaml pytest -n auto models/demos/llama3/tests/multimodal/test_llama_cross_attention.py -k "batch_1" ; fail+=$?
  LLAMA_DIR=$llama11b WH_ARCH_YAML=$wh_arch_yaml pytest -n auto models/demos/llama3/tests/multimodal/test_llama_cross_block.py -k "batch_1" ; fail+=$?
  LLAMA_DIR=$llama11b WH_ARCH_YAML=$wh_arch_yaml pytest -n auto models/demos/llama3/tests/multimodal/test_llama_conv2d_patch.py ; fail+=$?
  LLAMA_DIR=$llama11b WH_ARCH_YAML=$wh_arch_yaml pytest -n auto models/demos/llama3/tests/multimodal/test_llama_class_embedding.py ; fail+=$?
  LLAMA_DIR=$llama11b WH_ARCH_YAML=$wh_arch_yaml pytest -n auto models/demos/llama3/tests/multimodal/test_llama_tile_position_embedding.py ; fail+=$?
  LLAMA_DIR=$llama11b WH_ARCH_YAML=$wh_arch_yaml pytest -n auto models/demos/llama3/tests/multimodal/test_llama_positional_embedding.py ; fail+=$?

  # Record the end time
  end_time=$(date +%s)
  duration=$((end_time - start_time))
  echo "LOG_METAL: run_t3000_llama3.2-11b-vision_unit_tests $duration seconds to complete"
  if [[ $fail -ne 0 ]]; then
    exit 1
  fi
}

run_t3000_spoof_n300_llama3.2-11b-vision_unit_tests() {
  # Record the start time
  fail=0
  start_time=$(date +%s)

  echo "LOG_METAL: Running run_t3000_spoof_n300_llama3.2-11b-vision_unit_tests"

  wh_arch_yaml=wormhole_b0_80_arch_eth_dispatch.yaml
  # Llama3.2-11B
  llama11b=/mnt/MLPerf/tt_dnn-models/llama/Llama3.2-11B-Vision-Instruct/
  # Use FAKE_DEVICE env variable to run on an N300 mesh
  fake_device=N300

  FAKE_DEVICE=$fake_device LLAMA_DIR=$llama11b WH_ARCH_YAML=$wh_arch_yaml pytest -n auto models/demos/llama3/tests/multimodal/test_llama_image_mlp.py ; fail+=$?
  FAKE_DEVICE=$fake_device LLAMA_DIR=$llama11b WH_ARCH_YAML=$wh_arch_yaml pytest -n auto models/demos/llama3/tests/multimodal/test_llama_image_attention.py ; fail+=$?
  FAKE_DEVICE=$fake_device LLAMA_DIR=$llama11b WH_ARCH_YAML=$wh_arch_yaml pytest -n auto models/demos/llama3/tests/multimodal/test_llama_image_block.py ; fail+=$?
  FAKE_DEVICE=$fake_device LLAMA_DIR=$llama11b WH_ARCH_YAML=$wh_arch_yaml pytest -n auto models/demos/llama3/tests/multimodal/test_llama_cross_attention.py -k "batch_1" ; fail+=$?
  FAKE_DEVICE=$fake_device LLAMA_DIR=$llama11b WH_ARCH_YAML=$wh_arch_yaml pytest -n auto models/demos/llama3/tests/multimodal/test_llama_cross_block.py -k "batch_1" ; fail+=$?
  FAKE_DEVICE=$fake_device LLAMA_DIR=$llama11b WH_ARCH_YAML=$wh_arch_yaml pytest -n auto models/demos/llama3/tests/multimodal/test_llama_conv2d_patch.py ; fail+=$?
  FAKE_DEVICE=$fake_device LLAMA_DIR=$llama11b WH_ARCH_YAML=$wh_arch_yaml pytest -n auto models/demos/llama3/tests/multimodal/test_llama_class_embedding.py ; fail+=$?
  FAKE_DEVICE=$fake_device LLAMA_DIR=$llama11b WH_ARCH_YAML=$wh_arch_yaml pytest -n auto models/demos/llama3/tests/multimodal/test_llama_tile_position_embedding.py ; fail+=$?
  FAKE_DEVICE=$fake_device LLAMA_DIR=$llama11b WH_ARCH_YAML=$wh_arch_yaml pytest -n auto models/demos/llama3/tests/multimodal/test_llama_positional_embedding.py ; fail+=$?

  # Record the end time
  end_time=$(date +%s)
  duration=$((end_time - start_time))
  echo "LOG_METAL: run_t3000_spoof_n300_llama3.2-11b-vision_unit_tests $duration seconds to complete"
  if [[ $fail -ne 0 ]]; then
    exit 1
  fi
}

run_t3000_mixtral_tests() {
  # Record the start time
  fail=0
  start_time=$(date +%s)

  echo "LOG_METAL: Running run_t3000_mixtral_tests"

  WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml pytest -n auto models/demos/t3000/mixtral8x7b/tests/test_mixtral_attention.py ; fail+=$?
  WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml pytest -n auto models/demos/t3000/mixtral8x7b/tests/test_mixtral_mlp.py ; fail+=$?
  WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml pytest -n auto models/demos/t3000/mixtral8x7b/tests/test_mixtral_rms_norm.py ; fail+=$?
  WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml pytest -n auto models/demos/t3000/mixtral8x7b/tests/test_mixtral_embedding.py ; fail+=$?
  WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml pytest -n auto models/demos/t3000/mixtral8x7b/tests/test_mixtral_moe.py ; fail+=$?
  WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml pytest -n auto models/demos/t3000/mixtral8x7b/tests/test_mixtral_decoder.py ; fail+=$?
  # Mixtral prefill tests
  WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml pytest -n auto models/demos/t3000/mixtral8x7b/tests/test_mixtral_mlp_prefill.py ; fail+=$?
  WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml pytest -n auto models/demos/t3000/mixtral8x7b/tests/test_mixtral_moe_prefill.py ; fail+=$?
  # Record the end time
  end_time=$(date +%s)
  duration=$((end_time - start_time))
  echo "LOG_METAL: run_t3000_mixtral_tests $duration seconds to complete"
  if [[ $fail -ne 0 ]]; then
    exit 1
  fi
}

run_t3000_grok_tests() {
  # Record the start time
  fail=0
  start_time=$(date +%s)

  echo "LOG_METAL: Running run_t3000_grok_tests"

  WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml pytest -n auto models/experimental/grok/tests/test_grok_rms_norm.py ; fail+=$?
  WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml pytest -n auto models/experimental/grok/tests/test_grok_attention.py ; fail+=$?
  WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml pytest -n auto models/experimental/grok/tests/test_grok_mlp.py --timeout=500; fail+=$?
  WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml pytest -n auto models/experimental/grok/tests/test_grok_moe.py --timeout=600; fail+=$?

  # Record the end time
  end_time=$(date +%s)
  duration=$((end_time - start_time))
  echo "LOG_METAL: run_t3000_grok_tests $duration seconds to complete"
  if [[ $fail -ne 0 ]]; then
    exit 1
  fi
}

run_t3000_unet_shallow_tests() {
  # Record the start time
  fail=0
  start_time=$(date +%s)

  echo "LOG_METAL: Running run_t3000_unet_shallow_tests"

  WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml pytest -n auto models/experimental/functional_unet/tests/test_unet_multi_device.py; fail+=$?

  # Record the end time
  end_time=$(date +%s)
  duration=$((end_time - start_time))
  echo "LOG_METAL: run_t3000_unet_shallow_tests took $duration seconds to complete"
  if [[ $fail -ne 0 ]]; then
    exit 1
  fi
}

run_t3000_tests() {
  # Run ttmetal tests
  run_t3000_ttmetal_tests

  # Run ttnn tests
  run_t3000_ttnn_tests

  # Run falcon7b tests
  run_t3000_falcon7b_tests

  # Run falcon40b tests
  run_t3000_falcon40b_tests

  # Run llama3-small (1B, 3B, 8B) tests
  run_t3000_llama3-small_tests

  # Run llama3.2-11B tests
  run_t3000_llama3.2-11b_tests

  # Run llama3.1-70B tests
  run_t3000_llama3.1-70b_tests

  # Run llama3.2-11B-vision tests
  run_t3000_llama3.2-11b-vision_unit_tests

  # Run llama3.2-11B-vision tests on spoofed N300 mesh
  run_t3000_spoof_n300_llama3.2-11b-vision_unit_tests

  # Run mixtral tests
  run_t3000_mixtral_tests

  # Run grok tests
  run_t3000_grok_tests

  # Run unet shallow tests
  run_t3000_unet_shallow_tests
}

fail=0
main() {
  # For CI pipeline - source func commands but don't execute tests if not invoked directly
  if [[ "${BASH_SOURCE[0]}" != "${0}" ]]; then
    echo "Script is being sourced, not executing main function"
    return 0
  fi

  if [[ -z "$TT_METAL_HOME" ]]; then
    echo "Must provide TT_METAL_HOME in environment" 1>&2
    exit 1
  fi

  if [[ -z "$ARCH_NAME" ]]; then
    echo "Must provide ARCH_NAME in environment" 1>&2
    exit 1
  fi

  # Run all tests
  cd $TT_METAL_HOME
  export PYTHONPATH=$TT_METAL_HOME

  run_t3000_tests

  if [[ $fail -ne 0 ]]; then
    exit 1
  fi
}

main "$@"
