# Advanced Performance Optimizations for Models

1. [Metal Trace](#metal-trace)
2. [Multiple Command Queues](#multiple-command-queues)
3. [Putting Trace and Multiple Command Queues Together](#putting-trace-and-multiple-command-queues-together)

## Metal Trace

Metal Trace is a performance optimization feature that aims to remove the host overhead of constructing and dispatching operations for models, and will show benefits when the host time to execute/dispatch a model exceeds the device time needed to run the operations.

This feature works by recording the commands for dispatching operations into a DRAM buffer, and replaying these commands when executing a trace. This means that all of the operation’s parameters are statically saved, the input/output tensor shapes, addresses, etc. do not change. This makes using trace for entire generative models more difficult since these models have a changing sequence length, but statically sized inputs/outputs such as for image or sequence classification work well with metal trace. Tracing generative models can still be done using a variety of different techniques, such as using multiple traces, but is not currently covered in this document.

The following figure shows the runtime of a model execution that is host-bound. We see that the host is not able to run and dispatch commands to the device ahead of time, and the device is stuck stalling waiting for the host in order to receive commands to run operations, which is why there are large gaps between operations on the device.
<!-- ![image1](images/image1.png){width=15 height=15} -->
<img src="images/image1.png" style="width:1000px;"/>

With trace, we can eliminate a large portion of these gaps. In the figure below we now execute the model using trace. We see that the host finishes dispatching the model almost immediately and is just waiting for the device to finish for most of the time. On the device, we see that the gaps between ops is much smaller
<!-- ![image2](images/image2.png){width=15 height=15} -->
<img src="images/image2.png" style="width:1000px;"/>

In order to use trace, we need to use the following trace apis:

* `trace_region_size`

  This is a parameter to the device creation api, and this determines the size of the memory region we remove from the regular DRAM buffer space and where we can allocate our trace buffers on device. Since this is preallocating the region, there is currently no automated infrastructure in order to determine the size needed. To determine what size should be used, we can try capturing the operations we want to trace, and when this fails this will report the required size to use.

  `Always | FATAL    | Creating trace buffers of size 751616B on device 0, but only 0B is allocated for trace region.`

  In pytest, we can pass the `trace_region_size` using the `device_params` fixture:

  `@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576, "trace_region_size": 800768}], indirect=True)`
* `tid = ttnn.begin_trace_capture(device, cq_id=0)`

  This marks the beginning of a trace. The commands for dispatching operations after this is called will be recorded and associated with the returned trace id.
* `ttnn.end_trace_capture(device, tid, cq_id=0)`

  This marks the end of the trace. All operations that were run between beginning and end will be recorded and associated with the trace id.
* `ttnn.execute_trace(device, tid, cq_id=0, blocking=False)`

  This will execute the captured trace with the specified trace id, and is equivalent to running all the operations that were captured between begin and end

In addition, since trace requires the addresses of the used tensors to be the same, we need to statically preallocate our input tensor, and reuse this tensor instead of recreating our input tensor each iteration using the following apis:

* `device_tensor = ttnn.allocate_tensor_on_device(shape, dtype, layout, device, input_mem_config)`

  This will allocate a tensor with the specified parameters on the device. The tensor data will be uninitialized
* `ttnn.copy_host_to_device_tensor(host_tensor, device_tensor, cq_id=0)`

  This will copy data from the input host tensor to the allocated on device tensor

Normally for performance we try to allocate tensors in L1, but many models are not able to fit in L1 if we keep the input tensor in L1 memory. To work around this, we can allocate our input in DRAM so that we can keep the tensor persistent in memory, then run an operation to move it to L1. For performance, we’d expect to allocate the input as DRAM sharded and move it to L1 sharded using the reshard operation.

Trace only supports capturing and executing operations, and not other commands such as reading/writing tensors. This also means that to capture the trace of a sequence of operations, we must run with program cache and have already compiled the target operations before we capture them.

Putting this together, we can capture and execute the trace of a model using the following basic structure:

```py
# Allocate our persistent input tensor
input_dram_tensor = ttnn.allocate_tensor_on_device(shape, dtype, layout, device, sharded_dram_mem_config)

# First run to compile the model
ttnn.copy_host_to_device_tensor(host_tensor, device_dram_tensor, cq_id=0)
input_l1_tensor = ttnn.to_memory_config(input_dram_tensor, sharded_l1_mem_config)
output_tensor = run_model(input_l1_tensor)

# Capture the trace of the model
ttnn.copy_host_to_device_tensor(host_tensor, device_dram_tensor, cq_id=0)
tid = ttnn.begin_trace_capture(device, cq_id=0)
input_l1_tensor = ttnn.to_memory_config(input_dram_tensor, sharded_l1_mem_config)
# It is important that we keep the output tensor on device returned here, so that we have the output tensor and associated address to read from after executing trace
output_tensor = run_model(input_l1_tensor)
ttnn.end_trace_capture(device, tid, cq_id=0)

# Execute the trace
ttnn.copy_host_to_device_tensor(host_tensor, device_dram_tensor, cq_id=0)
ttnn.execute_trace(device, tid, cq_id=0, blocking=False)
host_output_tensor = output_tensor.cpu(blocking=False)

```

## Multiple Command Queues

Metal supports multiple command queues for fast dispatch, up to two queues. These command queues are independent of each other and allow for us to dispatch commands on a device in parallel. Both command queues support any dispatch command, so we use either for I/O data transfers or launching programs on either command queue. As these queues are independent of each other, to coordinate and guarantee command order such as having one queue be used to write the input and the other queue be used to run operations on that input, we need to use events to synchronize between the queues. A common setup for multiple command queues is to have one only responsible for writing inputs, while the other command queue is used for dispatching programs and reading back the output, which is what will be described in this document. This is useful where we are device bound and our input tensor takes a long time to write, and allows us to overlap dispatching of the next input tensor with the execution of the previous model run. Other setups are also possible, such as having one command queue for both writes and reads, while the other is used for only dispatching programs, or potentially having both command queues running different programs concurrently.

The figure below shows an example of where we can see the benefits of using an independent queue for writing inputs. We see a large gap between each run of the model, as well as the host being stalled waiting to be able to finish writing the next tensor.
<!-- ![image3](images/image4.png){width=15 height=15} -->
<img src="images/image3.png" style="width:1000px;"/>

Using a second command queue only for writes enables us to eliminate the gap between model executions, and allows the host to go ahead of the device and enqueue commands for subsequent models runs before we have finished executing our current run.
<!-- ![image4](images/image4.png){width=15 height=15} -->
<img src="images/image4.png" style="width:1000px;"/>

In order to use multiple command queues, we need to be familiar with the following apis:

* `num_hw_cqs`/`num_command_queues` (Currently dependent on whether we are running with single or multi device fixture)

  This is a parameter to the device creation api, and sets how many command queues to create the device with. The default is one, and the max is two. In pytest, we can pass this using the `device_params` fixture:

  `@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576, "trace_region_size": 800768, "num_command_queues": 2}], indirect=True)`
* `event = ttnn.create_event(device)`

  This will create an event object for the specified device
* `ttnn.record_event(cq_id = 0, event = event)`

  This will record the event on the device after all current commands on the specified command queue are finished. This event will be visible to all command queue
* `ttnn.wait_for_event(cq_id = 0, event = event)`

  This will enqueue a command to the specified command queue to stall until the specified event has been recorded on the device. No commands sent to the command queue after the wait for event command will be executed until the event occurs

In addition, for our example of using one command queue only for writing inputs, we need to preallocate our input tensor and keep it in memory. This is so that we have a static location we will write our inputs to as this will happen in parallel with op execution that produces intermediate tensors, so we can’t have them use overlapping data regions. We can use the following apis to achieve this:

* `device_tensor = ttnn.allocate_tensor_on_device(shape, dtype, layout, device, input_mem_config)`

  This will allocate a tensor with the specified parameters on the device. The tensor data will be uninitialized
* `ttnn.copy_host_to_device_tensor(host_tensor, device_tensor, cq_id=0)`

  This will copy data from the input host tensor to the allocated on device tensor

Normally for performance we try to allocate tensors in L1, but many models are not able to fit in L1 if we keep the input tensor in L1 memory. To work around this, we can allocate our input in DRAM so that we can keep the tensor persistent in memory, then run an operation to move it to L1. For performance, we’d expect to allocate the input as DRAM sharded and move it to L1 sharded using the reshard operation.

For using 2 command queues where one is just for writes, and one for running programs and reading, we will need to create and use 2 events.
The first event we use is an event to signal that the write has completed on command queue 1\. This event will be waited on by command queue 0 so that it only executes operations after the write has completed. The second event we have is for signaling that command queue 0 has consumed the input tensor, and that it is okay for command queue 1 to overwrite it with new data. This is waited on by command queue 1 before it writes the next input.

```py
# This example uses 1 CQ for only writing inputs (CQ 1), and one CQ for executing programs/reading back the output (CQ 0)

# Create the event for signalling when the first operation is completed. This is the consumer of the input tensor so once this is completed, we can issue the next write
op_event = ttnn.create_event(device)
# Create the event for when input write is completed. This is used to signal that the input tensor can be read/consumed
write_event = ttnn.create_event(device)

# Allocate our persistent input tensor
input_dram_tensor = ttnn.allocate_tensor_on_device(shape, dtype, layout, device, sharded_dram_mem_config)

# Dummy record an op event on CQ 0 since we wait on this first in the loop
ttnn.record_event(0, op_event)

outputs = []

for iter in range(0, 2):
    # Stall CQ 1 for the input tensor consumer (CQ 0) to signal it has finished so we can start overwriting the previous input tensor with the new one
    ttnn.wait_for_event(1, op_event)
    # Write the next input tensor on CQ 1
    ttnn.copy_host_to_device_tensor(host_tensor, input_dram_tensor, cq_id=1)
    # Signal that the write has finished on CQ 1
    ttnn.record_event(1, write_event)
    # Make CQ 0 stall until CQ 1 has signalled that the write has finished
    ttnn.wait_for_event(0, write_event)
    # Run the first operation of the model on CQ 0
    input_l1_tensor = ttnn.to_memory_config(input_dram_tensor, sharded_l1_mem_config)
    # Signal to the producer (CQ 1) that CQ 0 is finished with the input and it can be overwritten
    ttnn.record_event(0, op_event)
    # Run the rest of the model and issue output readback on the default CQ (0)
    output_tensor = run_model(input_l1_tensor)
    outputs.append(output_tensor.cpu(blocking=False))

```

## Putting Trace and Multiple Command Queues Together

This section assumes that you are familiar with the contents and apis described in [Metal Trace](#metal-trace) and [Multiple Command Queues](#multiple-command-queues).

By combining these two optimizations, we can achieve higher end-to-end performance where host is running well ahead of device and enqueuing for many subsequent iterations ahead, and device is continuously executing operations with little to no latency between them. This can be seen in the following figure where host has enqueued 10 iterations before device has finished 1 iteration, and there is little to no gap between the device operations and model execution iterations.
<!-- ![image5](images/image5.png){width=15 height=15} -->
<img src="images/image5.png" style="width:1000px;"/>

When combining these two optimizations, there are a few things we need to be aware of / change:

* Trace cannot capture events, so we do not capture the events or the consumer ops of the input tensor since we need to enqueue event commands right after
* Because the input to trace is from the output of an operation, and we want the output to be in L1 we need to be ensure and assert that the location this output gets written to is a constant address when executing the trace

```py
# This example uses 1 CQ for only writing inputs (CQ 1), and one CQ for executing programs/reading back the output (CQ 0)

# Create the event for signalling when the first operation is completed. This is the consumer of the input tensor so once this is completed, we can issue the next write
op_event = ttnn.create_event(device)
# Create the event for when input write is completed. This is used to signal that the input tensor can be read/consumed
write_event = ttnn.create_event(device)

# Allocate our persistent input tensor
input_dram_tensor = ttnn.allocate_tensor_on_device(shape, dtype, layout, device, sharded_dram_mem_config)

# Dummy record an op event on CQ 0 since we wait on this first in the loop
ttnn.record_event(0, op_event)

# First run to compile the model
# Stall CQ 1 for the input tensor consumer (CQ 0) to signal it has finished so we can start overwriting the previous input tensor with the new one
ttnn.wait_for_event(1, op_event)
# Write the next input tensor on CQ 1
ttnn.copy_host_to_device_tensor(host_tensor, input_dram_tensor, cq_id=1)
# Signal that the write has finished on CQ 1
ttnn.record_event(1, write_event)
# Make CQ 0 stall until CQ 1 has signalled that the write has finished
ttnn.wait_for_event(0, write_event)
# Run the first operation of the model on CQ 0
input_l1_tensor = ttnn.to_memory_config(input_dram_tensor, sharded_l1_mem_config)
# Signal to the producer (CQ 1) that CQ 0 is finished with the input and it can be overwritten
ttnn.record_event(0, op_event)
# Run the rest of the model and issue output readback on the default CQ (0)
output_tensor = run_model(input_l1_tensor)

# Capture the trace of the model
ttnn.wait_for_event(1, op_event)
ttnn.copy_host_to_device_tensor(host_tensor, input_dram_tensor, cq_id=1)
ttnn.record_event(1, write_event)
ttnn.wait_for_event(0, write_event)
input_l1_tensor = ttnn.to_memory_config(input_dram_tensor, sharded_l1_mem_config)
ttnn.record_event(0, op_event)
# Record the address of the input tensor to trace so that we can validate we allocated our input tensor at the right address
first_out_addr = input_l1_tensor.buffer_address()
shape = input_l1_tensor.shape
dtype = input_l1_tensor.dtype
layout = input_l1_tensor.layout
# Deallocate the previous output tensor here so that we will allocate our input tensor at the right address afterwards
output_tensor.deallocate(force=True)
tid = ttnn.begin_trace_capture(device, cq_id=0)
# It is important that we keep the output tensor on device returned here, so that we have the output tensor and associated address to read from after executing trace
output_tensor = run_model(input_l1_tensor)

# Try allocating our persistent input tensor here and verifying it matches the address that trace captured
input_l1_tensor = ttnn.allocate_tensor_on_device(shape, dtype, layout, device, sharded_l1_mem_config)
assert first_out_addr == input_l1_tensor.buffer_address()

ttnn.end_trace_capture(device, tid, cq_id=0)

outputs = []

for iter in range(0, 2):
    # Stall CQ 1 for the input tensor consumer (CQ 0) to signal it has finished so we can start overwriting the previous input tensor with the new one
    ttnn.wait_for_event(1, op_event)
    # Write the next input tensor on CQ 1
    ttnn.copy_host_to_device_tensor(host_tensor, input_dram_tensor, cq_id=1)
    # Signal that the write has finished on CQ 1
    ttnn.record_event(1, write_event)
    # Make CQ 0 stall until CQ 1 has signalled that the write has finished
    ttnn.wait_for_event(0, write_event)
    # Run the first operation of the model on CQ 0
    # Note here that we are writing to our persisent input tensor in place to reuse the address
    input_l1_tensor = ttnn.to_memory_config(input_dram_tensor, sharded_l1_mem_config, input_l1_tensor)
    # Signal to the producer (CQ 1) that CQ 0 is finished with the input and it can be overwritten
    ttnn.record_event(0, op_event)
    # Run the rest of the model and issue output readback on the default CQ (0)
    ttnn.execute_trace(device, tid, cq_id=0, blocking=False)
    outputs.append(output_tensor.cpu(blocking=False))
```