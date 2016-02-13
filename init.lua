require "cutorch"
require "nn"
require "libcunn"
require "cunn.THCUNN"

require('cunn.test')
require('cunn.DataParallelTable')

nn.Module._flattenTensorBuffer['torch.CudaTensor'] = torch.FloatTensor.new
