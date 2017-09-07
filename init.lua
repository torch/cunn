cunn = nil

require "cutorch"
require "nn"
require "cunn.THCUNN"

require('cunn.test')
require('cunn.DataParallelTable')

nn.Module._flattenTensorBuffer['torch.CudaTensor'] = torch.FloatTensor.new
nn.Module._flattenTensorBuffer['torch.CudaDoubleTensor'] = torch.DoubleTensor.new
nn.Module._flattenTensorBuffer['torch.CudaHalfTensor'] = torch.HalfTensor.new
