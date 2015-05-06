require "cutorch"
require "nn"
require "libcunn"

include('test.lua')
include('utils.lua')

include('DataParallelTable.lua')
include('TransferGPU.lua')

function nn.Module:cudaOn(device)
  return nn.utils.recursiveCudaOn(self, device)
end

function nn.Criterion:cudaOn(device)
  return nn.utils.recursiveCudaOn(self, device)
end
