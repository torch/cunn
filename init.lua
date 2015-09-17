require "cutorch"
require "nn"
require "libcunn"

include('test.lua')

include('DataParallelTable.lua')

nn.Module._flattenTensorBuffer['torch.CudaTensor'] = torch.FloatTensor.new
