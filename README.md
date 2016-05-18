<a name="cunn.dok"/>
# CUDA backend for the Neural Network Package #

This package provides a CUDA implementation for many of the modules in the base nn package: [nn](https://github.com/torch/nn/blob/master/README.md)
 * [Modules](doc/cunnmodules.md#nn.cunnmodules.dok): There are also additional GPU-related modules not found in the nn package.

## To use

Simply convert your network model to CUDA by calling `:cuda()`:
```lua
local model = nn.Sequential()
model:add(nn.Linear(2,2))
model:add(nn.LogSoftMax())

model:cuda()  -- convert model to CUDA
```

... and similarly for your tensors:
```lua
local input = torch.Tensor(32,2):uniform()
input = input:cuda()
local output = model:forward(input)
```
... or create them directly as `CudaTensor`s:
```lua
local input = torch.CudaTensor(32,2):uniform()
local output = model:forward(input)
```

## To run unit-tests

```lua
luajit -l cunn -e 'cunn.test()'
```

## GPU Training Concepts

__Performance__

* data should be transferred between main memory and gpu in batches, otherwise the transfer time will be dominated
by latency associated with speed of light, and execution overheads, rather than by bandwidth
* therefore, train and predict using mini-batches
* allocating GPU memory causes a sync-point, which will noticeably affect performance
  * therefore try to allocate any `CudaTensor`s once, at the start of the program,
  and then simply copy data backwards and forwards
  between main memory and existing `CudaTensor`s
* similarly, try to avoid any operations that implicitly allocate new tensors.  For example, if you write:
```lua
require 'cutorch'

local a = torch.CudaTensor(1000):uniform()
for it=1,1000 do
  local b = torch.add(a, 1)
end
```
... this will allocate one thousand new `CudaTensor`s, one for each call to `torch.add(a, 1)`.

Use instead this form:
```lua
require 'cutorch'

local a = torch.CudaTensor(1000):uniform()
local b = torch.CudaTensor(1000):uniform()
for it=1,1000 do
  b:add(a, 1)
end
```
In this form, `b` is allocated only once, before the loop.  Then the `b:add(a,1)` operation will perform
the add inside the GPU kernel, and store the result into the original `b` `CudaTensor`.  This
will run noticeably faster, in general.  It's also a lot less likely to eat up arbitrary amounts of memory,
and less likely to need frequent calls to `collectgarbage(); collectgarbage()`.

__Benchmarking__

* GPU operations will typically continue after an instruction has been issued
* eg, if you do:
```lua
require 'cutorch'
local a = torch.CudaTensor(1000,1000):uniform()
a:add(1)
```
... the GPU kernel to add 1 will only be scheduled for launch by `a:add(1)`.  It might not have completed yet, or
even have reached the GPU, at the time that the `a:add(1)` instructions has completed
* therefore for running wall-clock timings, you should call `cutorch.synchronize()` before each timecheck
point:
```lua
require 'cutorch'
require 'sys'

local a = torch.CudaTensor(1000,1000):uniform()
cutorch.synchronize()
start = sys.tic()
a:add(1)
cutorch.synchronize()
print(sys.toc())
```

