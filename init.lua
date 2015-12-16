require "cutorch"
require "nn"
require "libcunn"
require('cunn.THCUNN')

include('test.lua')

include('DataParallelTable.lua')

nn.Module._flattenTensorBuffer['torch.CudaTensor'] = torch.FloatTensor.new
