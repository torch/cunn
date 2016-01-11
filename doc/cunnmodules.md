<a name="nn.cunnmodules.dok"/>
# Additional Modules #

The following nn modules are also made available by the cunn package:
 * [DataParallelTable](#nn.DataParallelTable) : A module to parallelize FPROP and BPROP across multiple-GPUs.

<a name="nn.DataParallelTable"/>
## DataParallelTable ##

```lua
module = nn.DataParallelTable(dim, [flattenParams], [useNCCL])
module:add(net, {gpu1, [gpu2, ...]})
```

DataParallelTable implements data parallelism for Torch modules. The same model
is replicated on multiple GPUs. The input is split, typically into smaller mini-batches.
Each replicated model handles only its portion of the input. The weight updates for 
each replica are summed together on the first replica in accGradParameters.

### DataParallelTable(dim, [flattenParams], [useNCCL]) ###

Creates a `DataParallelTable` that splits the input on the dimension `dim`. If `flattenParams` is `true`, [`getParameters()`](https://github.com/torch/nn/blob/master/doc/module.md#nn.Module.getParameters) will be called on the replicated module. If `useNCCL` is `true` and both [NCCL](https://github.com/NVIDIA/nccl) and the [NCCL torch bindings](https://github.com/ngimel/nccl.torch) are installed, NCCL will be used for inter-GPU communication.

For best performance, use `flattenParams` and `NCCL`.

### DataParallelTable:add(module, gpus) ###

Replicates `module` on the table of `gpus`. For example:

```lua
nn.DataParallelTable(1):add(module, {1, 2, 3, 4})
```

### DataParallelTable:threads(initFunc) ###

Switches the internal implementation to  use a seperate thread for each replica. This may hide the cost of kernel launches by dispatching them in parallel. The `initFunc` is executed in each thread.

```lua
nn.DataParallelTable(1):threads(function()
  require 'cudnn'
end)
```

### DataParallelTable:syncParameters() ###

Copies the model parameters from the first replica to all other replicas. This is automatically called from `updateOutput`, if it has not been called since the last `accGradParameters`.

### Example of training using DataParallelTable ###

```lua
-- CONSTRUCT MODEL:
conv_net = makeConvNet()  -- i.e. create nn.Sequential() and fill it
net = nn.DataParallelTable(1)  -- Split along first (batch) dimension
net:add(conv_net, {1, 2}) -- Use GPUs 1 and 2
-- TRAINING:
for i = 1, num_epochs do
  local output = net:forward(input)
  local err = criterion:forward(output, target)
  net:zeroGradParameters()
  local gradOutput = criterion:backward(output, target)
  local gradInput = net:backward(input, gradOutput)
  net:updateParameters(lr)
end
```

