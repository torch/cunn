<a name="nn.cunnmodules.dok"/>
# Additional Modules #

The following nn modules are also made available by the cunn package:
 * [DataParallelTable](#nn.DataParallelTable) : A module to parallelize FPROP and BPROP across multiple-GPUs.

<a name="nn.DataParallelTable"/>
## DataParallelTable ##

This module makes splitting data across multiple GPUs easy and seamless. The API is based loosely on the fbcunn ```DataParallel``` module (hence the name), but support for nested table input and outputs has been added.

Note that some of the code in this module borrows heavily from fbcunn (particularly ```asyncGPUCopy``` and ```getBuffer```), so credit also should go to Facebook for supplying the original starter code.  Usage:

```lua
-- CONSTRUCT MODEL:
conv_net = makeConvNet()  -- i.e. create nn.Sequential() and fill it
net = nn.DataParallelTable(1)  -- Split along first (batch) dimension
for i = 1, 2 do
  cutorch.setDevice(i)
  net:add(conv_net:clone():cuda(), i)  -- Use the ith GPU
end
cutorch.setDevice(1)  -- This is the 'primary' GPU
parameters, gradParameters = net:getParameters()
-- TRAINING:
for i = 1, num_epochs do
  feval = function(x)
    net:zeroGradParameters()
    local output = net:forward(input)
    local err = criterion:forward(output, target)
    local gradOutput = criterion:backward(output, target)
    local gradInput = net:backward(input, gradOutput)
    return err, gradParameters
  end
  optim.sgd(feval, parameters, optimState)
  net:syncParameters()  -- **** NEW ADDITIONAL CALL ****
end
```

To the outside world we make this module look like it just includes the parameters for the primary GPU (which is defined as the first module added by the ```:add()``` call), so that the optimizer can pretend it's one model on one GPU. Unfortunately we break the abstraction in one annoying way: every time you update the parameters you need to call ```:syncParameters()```, to distribute the new parameters to the other GPUs (this additional call is highlighted in the above usage example).

There are still some limitations with this module:
 * weight sharing is not implemented and if you use models with shared weights it will likely break across GPUs.
 * accGradParameters and updateParameters functions are not implemented because optim doesn't use them.

The differences between this module and fbcunn's ```DataParallel``` are:
 * ```DataParallel``` requires the use of nn.Optim (since it includes the ```_mixGrads``` call and because ```getParameters()``` is not implemented), whereas ```DataParallelTable``` allows the use of the optim package with closures.
 * The ```DataParallel``` instance must be the outer container, whereas ```DataParallelTable``` supports nesting within a larger network.
 * ```DataParallel``` requires tensor inputs and outputs, whereas ```DataParallelTable``` supports arbitrarily nested tables of tensors.
 * At the time of writing ```DataParallelTable``` is a little faster.
