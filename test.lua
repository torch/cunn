local cunntest = {}
local precision_forward = 1e-4
local precision_backward = 1e-2
local nloop = 1
local times = {}

--e.g.: th -lcunn -e "nn.testcuda{'copies'}"

function cunntest.copies()
   -- test vector
   local t = torch.CudaTensor(100,10)

   -- simple copy
   t:normal()
   local t2 = t:clone()
   mytester:asserteq( t:add(-1,t2):abs():max(), 0, 'simple copy')

   -- transpose copy
   t:normal()
   local t3 = t:transpose(1,2)
   local t4 = t3:clone()
   mytester:asserteq( t3:add(-1,t4):abs():max(), 0, 'transpose copy')

   -- unfold copy
   t:normal()
   local t5 = t:unfold(2,5,1)
   local t6 = t5:clone()
   mytester:asserteq( t5:add(-1,t6):abs():max(), 0, 'transpose copy')

   -- host copy
   t = torch.FloatTensor(100,10)
   t:normal()
   local tc = t:cuda()
   tc = tc:transpose(1,2)
   local t2 = tc:float()
   mytester:asserteq(t:transpose(1,2):add(-1,t2):abs():max(), 0, 'host copy, plus transpoe')
end

local function pointwise_forward(proto_module, name, max_error)
   local size = math.random(1,100)

   local tm = {}
   local title = string.format(name..'.forward %d -> %d', size, size)
   times[title] = tm

   local input = torch.randn(size)
   if name == 'Sqrt' then input:abs() end
   local sconv = proto_module
   local groundtruth = sconv:forward(input)
   local a = torch.Timer()
   for i = 1,nloop do
      groundtruth = sconv:forward(input)
   end
   tm.cpu = a:time().real

   input = input:cuda()
   local gconv = proto_module:clone():cuda()
   local rescuda = gconv:forward(input)
   a:reset()
   for i = 1,nloop do
      rescuda = gconv:forward(input)
   end
   cutorch.synchronize()
   tm.gpu = a:time().real

   local error = rescuda:float() - groundtruth
   mytester:assertlt(error:abs():max(), max_error, 'error on state (forward) ')
end

local function pointwise_backward(proto_module, name, max_error)
   local size = math.random(1,100)

   local tm = {}
   local title = string.format(name..'.backward %d -> %d', size, size)
   times[title] = tm

   local input = torch.randn(size)
   if name == 'Sqrt' then input:abs() end
   local gradOutput = torch.randn(size)
   local sconv = proto_module
   sconv:forward(input)
   local groundgrad = sconv:backward(input, gradOutput)
   local a = torch.Timer()
   for i = 1,nloop do
      groundgrad = sconv:backward(input, gradOutput)
   end
   tm.cpu = a:time().real

   input = input:cuda()
   gradOutput = gradOutput:cuda()
   local gconv = proto_module:clone():cuda()
   gconv:forward(input)
   local rescuda = gconv:backward(input, gradOutput)
   a:reset()
   for i = 1,nloop do
      rescuda = gconv:backward(input, gradOutput)
   end
   cutorch.synchronize()
   tm.gpu = a:time().real

   local error = rescuda:float() - groundgrad

   mytester:assertlt(error:abs():max(), max_error, 'error on state (backward) ')
end

local function pointwise_transposed(proto_module, name, max_error)
   max_error = max_error or 1e-7
   local tm = {}
   local title = name .. '.transposed'
   times[title] = tm

   local input = torch.Tensor(11, 19):uniform(-1, 1)
   if name == 'Sqrt' then
      input:uniform(0.1, 1)
   end
   local inputCUDA = input:clone():cuda()

   local cuda_module = proto_module:clone():cuda()

   -- transpose the inputs and DON'T make contiguous
   input = input:transpose(1, 2)
   inputCUDA = inputCUDA:transpose(1, 2)

   local output = proto_module:forward(input)
   local outputCUDA = cuda_module:forward(inputCUDA)

   local error = outputCUDA:float() - output
   mytester:assertlt(error:abs():max(), max_error, 'error on state (forward) ')

   local gradOutput = torch.Tensor(11, 19):uniform(-1, 1)
   local gradOutputCUDA = gradOutput:clone():cuda()

   gradOutput = gradOutput:transpose(1, 2)
   gradOutputCUDA = gradOutputCUDA:transpose(1, 2)

   local gradInput = proto_module:backward(input, gradOutput)
   local gradInputCUDA  = cuda_module:backward(inputCUDA, gradOutputCUDA)

   local error = gradInputCUDA:float() - gradInput
   mytester:assertlt(error:abs():max(), max_error,  'error on state (backward) ')
end

function cunntest.Tanh_forward()
   pointwise_forward(nn.Tanh(), 'Tanh', precision_forward)
end

function cunntest.Tanh_backward()
   pointwise_backward(nn.Tanh(), 'Tanh', precision_backward)
end

function cunntest.Tanh_transposed()
   pointwise_transposed(nn.Tanh(), 'Tanh', 1.5e-7)
end

function cunntest.HardTanh_forward()
   pointwise_forward(nn.HardTanh(), 'HardTanh', precision_forward)
end

function cunntest.HardTanh_backward()
   pointwise_backward(nn.HardTanh(), 'HardTanh', precision_backward)
end

function cunntest.HardTanh_transposed()
   pointwise_transposed(nn.HardTanh(), 'HardTanh', 1.5e-7)
end

function cunntest.Abs_forward()
   pointwise_forward(nn.Abs(), 'Abs', precision_forward)
end

function cunntest.Abs_backward()
   pointwise_backward(nn.Abs(), 'Abs', precision_backward)
end

function cunntest.Abs_transposed()
   pointwise_transposed(nn.Abs(), 'Abs')
end

function cunntest.Sigmoid_forward()
   pointwise_forward(nn.Sigmoid(), 'Sigmoid', precision_forward)
end

function cunntest.Sigmoid_backward()
   pointwise_backward(nn.Sigmoid(), 'Sigmoid', precision_backward)
end

function cunntest.Sigmoid_transposed()
   pointwise_transposed(nn.Sigmoid(), 'Sigmoid')
end

function cunntest.LogSigmoid_forward()
   pointwise_forward(nn.LogSigmoid(), 'LogSigmoid', precision_forward)
end

function cunntest.LogSigmoid_backward()
   pointwise_backward(nn.LogSigmoid(), 'LogSigmoid', precision_backward)
end

function cunntest.LogSigmoid_transposed()
   pointwise_transposed(nn.LogSigmoid(), 'LogSigmoid', 1e-6)
end

function cunntest.Threshold_forward()
  pointwise_forward(nn.Threshold(), 'Threshold', precision_forward)
  pointwise_forward(nn.Threshold(nil, nil, true), 'Threshold_inplace', precision_forward)
end

function cunntest.Threshold_backward()
  pointwise_backward(nn.Threshold(), 'Threshold', precision_backward)
  pointwise_backward(nn.Threshold(nil, nil, true), 'Threshold_inplace', precision_backward)
end

function cunntest.Sqrt_forward()
   pointwise_forward(nn.Sqrt(), 'Sqrt', precision_forward)
end

function cunntest.Sqrt_backward()
   pointwise_backward(nn.Sqrt(), 'Sqrt', precision_backward)
end

function cunntest.Sqrt_zero()
   local size = math.random(1, 100)

   -- Test zero inputs; we will avoid a div-by-zero by setting to zero
   local module_gpu = nn.Sqrt():cuda()
   local input_gpu = torch.CudaTensor(size, size):zero()
   module_gpu:forward(input_gpu)

   local gradOutput_gpu = torch.CudaTensor(size, size):fill(1)
   local gradInput_gpu = module_gpu:backward(input_gpu, gradOutput_gpu)

   mytester:assertTensorEq(gradInput_gpu:float(),
                           torch.FloatTensor(size, size):zero(),
                           0.000001, "error in sqrt backward singularity")

   -- Verify CPU and GPU zero behavior equivalency
   local module_cpu = nn.Sqrt()
   local input_cpu = input_gpu:float()
   module_cpu:forward(input_cpu)

   local gradOutput_cpu = gradOutput_gpu:float()
   local gradInput_cpu = module_cpu:backward(input_cpu, gradOutput_cpu)

   mytester:assertTensorEq(gradInput_gpu:float(),
                           gradInput_cpu:float(),
                           0.000001, "Sqrt_zero CPU and GPU not equivalent")
end

function cunntest.Sqrt_transposed()
   pointwise_transposed(nn.Sqrt(), 'Sqrt')
end

function cunntest.Square_forward()
   pointwise_forward(nn.Square(), 'Square', precision_forward)
end

function cunntest.Square_backward()
   pointwise_backward(nn.Square(), 'Square', precision_backward)
end

function cunntest.Square_transposed()
   pointwise_transposed(nn.Square(), 'Square')
end

function cunntest.SoftMax_forward()
   pointwise_forward(nn.SoftMax(), 'SoftMax', precision_forward)
end

function cunntest.SoftMax_backward()
   pointwise_backward(nn.SoftMax(), 'SoftMax', precision_backward)
end

function cunntest.LogSoftMax_forward()
   pointwise_forward(nn.LogSoftMax(), 'LogSoftMax', precision_forward*10)
end

function cunntest.LogSoftMax_backward()
   pointwise_backward(nn.LogSoftMax(), 'LogSoftMax', precision_backward)
end

function cunntest.LogSoftMax_forward_batch()
   local size = math.random(1,256)
   local bs = math.random(32,256)

   local tm = {}
   local title = string.format('LogSoftMax forward batch %d x %d -> %d x %d', bs, size, bs, size)
   times[title] = tm

   local input = torch.randn(bs, size)
   local sconv = nn.LogSoftMax()
   local groundtruth = sconv:forward(input)
   local a = torch.Timer()
   for i = 1,nloop do
      groundtruth = sconv:forward(input)
   end
   tm.cpu = a:time().real

   input = input:cuda()
   local gconv = nn.LogSoftMax():cuda()
   local rescuda = gconv:forward(input)
   a:reset()
   for i = 1,nloop do
      rescuda = gconv:forward(input)
   end
   cutorch.synchronize()
   tm.gpu = a:time().real

   local error = rescuda:float() - groundtruth
   mytester:assertlt(error:abs():max(), precision_forward*10, 'error on state (forward) ')
end

function cunntest.LogSoftMax_backward_batch()
   local size = math.random(1,256)
   local bs = math.random(32,256)

   local tm = {}
   local title = string.format('LogSoftMax.backward batch %d x %d -> %d x %d', bs, size, bs, size)
   times[title] = tm

   local input = torch.randn(bs, size)
   local gradOutput = torch.randn(bs, size)
   local sconv = nn.LogSoftMax()
   sconv:forward(input)
   local groundgrad = sconv:backward(input, gradOutput)
   local a = torch.Timer()
   for i = 1,nloop do
      groundgrad = sconv:backward(input, gradOutput)
   end
   tm.cpu = a:time().real

   input = input:cuda()
   gradOutput = gradOutput:cuda()
   local gconv = sconv:clone():cuda()
   gconv:forward(input)
   local rescuda = gconv:backward(input, gradOutput)
   a:reset()
   for i = 1,nloop do
      rescuda = gconv:backward(input, gradOutput)
   end
   cutorch.synchronize()
   tm.gpu = a:time().real

   local error = rescuda:float() - groundgrad

   mytester:assertlt(error:abs():max(), precision_backward, 'error on state (backward) ')
end

function cunntest.Euclidean_forward_batch()
   local bs = math.random(8,32)
   local nin = math.random(1,100)
   local nout = math.random(1,100)

   local tm = {}
   local title = string.format('Euclidean forward %d %d -> %d %d', bs, nin, bs, nout)
   times[title] = tm

   local input = torch.randn(bs, nin)
   local sconv = nn.Euclidean(nin, nout)
   local groundtruth = sconv:forward(input)
   local a = torch.Timer()
   for i = 1,nloop do
      groundtruth = sconv:forward(input)
   end
   tm.cpu = a:time().real

   input = input:cuda()
   local gconv = sconv:clone():cuda()
   local rescuda = gconv:forward(input)
   a:reset()
   for i = 1,nloop do
      rescuda = gconv:forward(input)
   end
   cutorch.synchronize()
   tm.gpu = a:time().real

   local error = rescuda:float() - groundtruth
   mytester:assertlt(error:abs():max(), precision_forward, 'error on state (forward) batch ')
end

function cunntest.Euclidean_backward_batch()
   local bs = math.random(8,32)
   local nin = math.random(1,100)
   local nout = math.random(1,100)

   local tm = {}
   local title = string.format('Euclidean backward %d %d <- %d %d', bs, nin, bs, nout)
   times[title] = tm

   local input = torch.randn(bs, nin)
   local gradOutput = torch.randn(bs, nout)
   local sconv = nn.Euclidean(nin, nout)
   sconv:forward(input)
   sconv:zeroGradParameters()
   local groundgrad = sconv:backward(input, gradOutput)
   local a = torch.Timer()
   for i = 1,nloop do
      sconv:zeroGradParameters()
      groundgrad = sconv:backward(input, gradOutput)
   end
   local groundweight = sconv.gradWeight
   tm.cpu = a:time().real

   input = input:cuda()
   gradOutput = gradOutput:cuda()
   local gconv = sconv:clone():cuda()
   gconv:forward(input)
   gconv:zeroGradParameters()
   local rescuda = gconv:backward(input, gradOutput)
   a:reset()
   for i = 1,nloop do
      gconv:zeroGradParameters()
      rescuda = gconv:backward(input, gradOutput)
   end
   cutorch.synchronize()
   tm.gpu = a:time().real

   local weightcuda = gconv.gradWeight

   local error = rescuda:float() - groundgrad
   local werror = weightcuda:float() - groundweight

   mytester:assertlt(error:abs():max(), precision_backward, 'error on state (backward) ')
   mytester:assertlt(werror:abs():max(), precision_backward, 'error on weight (backward) ')
end

function cunntest.WeightedEuclidean_forward_batch()
   local bs = math.random(8,32)
   local nin = math.random(1,100)
   local nout = math.random(1,100)

   local tm = {}
   local title = string.format('WeightedEuclidean forward %d %d -> %d %d', bs, nin, bs, nout)
   times[title] = tm

   local input = torch.randn(bs, nin)
   local sconv = nn.WeightedEuclidean(nin, nout)
   local groundtruth = sconv:forward(input)
   local a = torch.Timer()
   for i = 1,nloop do
      groundtruth = sconv:forward(input)
   end
   tm.cpu = a:time().real

   input = input:cuda()
   local gconv = sconv:clone():cuda()
   local rescuda = gconv:forward(input)
   a:reset()
   for i = 1,nloop do
      rescuda = gconv:forward(input)
   end
   cutorch.synchronize()
   tm.gpu = a:time().real

   local error = rescuda:float() - groundtruth
   mytester:assertlt(error:abs():max(), precision_forward, 'error on state (forward) batch ')
end

function cunntest.WeightedEuclidean_backward_batch()
   local bs = math.random(8,32)
   local nin = math.random(1,100)
   local nout = math.random(1,100)

   local tm = {}
   local title = string.format('WeightedEuclidean backward %d %d <- %d %d', bs, nin, bs, nout)
   times[title] = tm

   local input = torch.randn(bs, nin)
   local gradOutput = torch.randn(bs, nout)
   local sconv = nn.WeightedEuclidean(nin, nout)
   sconv:forward(input)
   sconv:zeroGradParameters()
   local groundgrad = sconv:backward(input, gradOutput)
   local a = torch.Timer()
   for i = 1,nloop do
      sconv:zeroGradParameters()
      groundgrad = sconv:backward(input, gradOutput)
   end
   local groundweight = sconv.gradWeight
   local grounddiagCov = sconv.gradDiagCov
   tm.cpu = a:time().real

   input = input:cuda()
   gradOutput = gradOutput:cuda()
   local gconv = sconv:clone():cuda()
   gconv:forward(input)
   gconv:zeroGradParameters()
   local rescuda = gconv:backward(input, gradOutput)
   a:reset()
   for i = 1,nloop do
      gconv:zeroGradParameters()
      rescuda = gconv:backward(input, gradOutput)
   end
   cutorch.synchronize()
   tm.gpu = a:time().real

   local weightcuda = gconv.gradWeight
   local diagCovcuda = gconv.gradDiagCov

   local error = rescuda:float() - groundgrad
   local werror = weightcuda:float() - groundweight
   local derror = diagCovcuda:float() - grounddiagCov

   mytester:assertlt(error:abs():max(), precision_backward, 'error on state (backward) ')
   mytester:assertlt(werror:abs():max(), precision_backward, 'error on weight (backward) ')
   mytester:assertlt(derror:abs():max(), precision_backward, 'error on diagCov (backward) ')
end


function cunntest.Max_forward()
   local size1 = math.random(1,1000)
   local size2 = math.random(2,100)

   local tm = {}
   local title = string.format('Max forward %dx%d', size1, size2)
   times[title] = tm

   local input = torch.randn(size1,size2)
   local sconv = nn.Max(2)
   local groundtruth = sconv:forward(input)
   local a = torch.Timer()
   for i = 1,nloop do
      groundtruth = sconv:forward(input)
   end
   tm.cpu = a:time().real

   input = input:cuda()
   local gconv = nn.Max(2):cuda()
   local rescuda = gconv:forward(input)
   a:reset()
   for i = 1,nloop do
      rescuda = gconv:forward(input)
   end
   cutorch.synchronize()
   tm.gpu = a:time().real

   local error = rescuda:float() - groundtruth
   mytester:assertlt(error:abs():max(), precision_forward, 'error on state (forward) ')
end

function cunntest.Max_backward()
   local size1 = math.random(1,1000)
   local size2 = math.random(2,100)

   local tm = {}
   local title = string.format('Max.backward %dx%d', size1, size2)
   times[title] = tm

   local input = torch.randn(size1,size2)
   local gradOutput = torch.randn(size1)
   local sconv = nn.Max(2)
   sconv:forward(input)
   local groundgrad = sconv:backward(input, gradOutput)
   local a = torch.Timer()
   for i = 1,nloop do
      groundgrad = sconv:backward(input, gradOutput)
   end
   tm.cpu = a:time().real

   input = input:cuda()
   gradOutput = gradOutput:cuda()
   local gconv = sconv:clone():cuda()
   gconv:forward(input)
   local rescuda = gconv:backward(input, gradOutput)
   a:reset()
   for i = 1,nloop do
      rescuda = gconv:backward(input, gradOutput)
   end
   cutorch.synchronize()
   tm.gpu = a:time().real

   local error = rescuda:float() - groundgrad

   mytester:assertlt(error:abs():max(), precision_backward, 'error on state (backward) ')
end

function cunntest.Min_forward()
   local size1 = math.random(1,1000)
   local size2 = math.random(2,100)

   local tm = {}
   local title = string.format('Min forward %dx%d', size1, size2)
   times[title] = tm

   local input = torch.randn(size1,size2)
   local sconv = nn.Min(2)
   local groundtruth = sconv:forward(input)
   local a = torch.Timer()
   for i = 1,nloop do
      groundtruth = sconv:forward(input)
   end
   tm.cpu = a:time().real

   input = input:cuda()
   local gconv = nn.Min(2):cuda()
   local rescuda = gconv:forward(input)
   a:reset()
   for i = 1,nloop do
      rescuda = gconv:forward(input)
   end
   cutorch.synchronize()
   tm.gpu = a:time().real

   local error = rescuda:float() - groundtruth
   mytester:assertlt(error:abs():max(), precision_forward, 'error on state (forward) ')
end

function cunntest.Min_backward()
   local size1 = math.random(1,1000)
   local size2 = math.random(2,100)

   local tm = {}
   local title = string.format('Min.backward %dx%d', size1, size2)
   times[title] = tm

   local input = torch.randn(size1,size2)
   local gradOutput = torch.randn(size1)
   local sconv = nn.Min(2)
   sconv:forward(input)
   local groundgrad = sconv:backward(input, gradOutput)
   local a = torch.Timer()
   for i = 1,nloop do
      groundgrad = sconv:backward(input, gradOutput)
   end
   tm.cpu = a:time().real

   input = input:cuda()
   gradOutput = gradOutput:cuda()
   local gconv = sconv:clone():cuda()
   gconv:forward(input)
   local rescuda = gconv:backward(input, gradOutput)
   a:reset()
   for i = 1,nloop do
      rescuda = gconv:backward(input, gradOutput)
   end
   cutorch.synchronize()
   tm.gpu = a:time().real

   local error = rescuda:float() - groundgrad

   mytester:assertlt(error:abs():max(), precision_backward, 'error on state (backward) ')
end

function cunntest.Sum_forward()
   local size1 = math.random(1,1000)
   local size2 = math.random(2,100)

   local tm = {}
   local title = string.format('Sum forward %dx%d', size1, size2)
   times[title] = tm

   local input = torch.randn(size1,size2)
   local sconv = nn.Sum(2)
   local groundtruth = sconv:forward(input)
   local a = torch.Timer()
   for i = 1,nloop do
      groundtruth = sconv:forward(input)
   end
   tm.cpu = a:time().real

   input = input:cuda()
   local gconv = nn.Sum(2):cuda()
   local rescuda = gconv:forward(input)
   a:reset()
   for i = 1,nloop do
      rescuda = gconv:forward(input)
   end
   cutorch.synchronize()
   tm.gpu = a:time().real

   local error = rescuda:float() - groundtruth
   mytester:assertlt(error:abs():max(), precision_forward, 'error on state (forward) ')
end

function cunntest.Sum_backward()
   local size1 = math.random(1,1000)
   local size2 = math.random(2,100)

   local tm = {}
   local title = string.format('Sum.backward %dx%d', size1, size2)
   times[title] = tm

   local input = torch.randn(size1,size2)
   local gradOutput = torch.randn(size1)
   local sconv = nn.Sum(2)
   sconv:forward(input)
   local groundgrad = sconv:backward(input, gradOutput)
   local a = torch.Timer()
   for i = 1,nloop do
      groundgrad = sconv:backward(input, gradOutput)
   end
   tm.cpu = a:time().real

   input = input:cuda()
   gradOutput = gradOutput:cuda()
   local gconv = sconv:clone():cuda()
   gconv:forward(input)
   local rescuda = gconv:backward(input, gradOutput)
   a:reset()
   for i = 1,nloop do
      rescuda = gconv:backward(input, gradOutput)
   end
   cutorch.synchronize()
   tm.gpu = a:time().real

   local error = rescuda:float() - groundgrad

   mytester:assertlt(error:abs():max(), precision_backward, 'error on state (backward) ')
end

function cunntest.Mean_forward()
   local size1 = math.random(1,1000)
   local size2 = math.random(2,100)

   local tm = {}
   local title = string.format('Mean forward %dx%d', size1, size2)
   times[title] = tm

   local input = torch.randn(size1,size2)
   local sconv = nn.Mean(2)
   local groundtruth = sconv:forward(input)
   local a = torch.Timer()
   for i = 1,nloop do
      groundtruth = sconv:forward(input)
   end
   tm.cpu = a:time().real

   input = input:cuda()
   local gconv = nn.Mean(2):cuda()
   local rescuda = gconv:forward(input)
   a:reset()
   for i = 1,nloop do
      rescuda = gconv:forward(input)
   end
   cutorch.synchronize()
   tm.gpu = a:time().real

   local error = rescuda:float() - groundtruth
   mytester:assertlt(error:abs():max(), precision_forward, 'error on state (forward) ')
end

function cunntest.Mean_backward()
   local size1 = math.random(1,1000)
   local size2 = math.random(2,100)

   local tm = {}
   local title = string.format('Mean.backward %dx%d', size1, size2)
   times[title] = tm

   local input = torch.randn(size1,size2)
   local gradOutput = torch.randn(size1)
   local sconv = nn.Mean(2)
   sconv:forward(input)
   local groundgrad = sconv:backward(input, gradOutput)
   local a = torch.Timer()
   for i = 1,nloop do
      groundgrad = sconv:backward(input, gradOutput)
   end
   tm.cpu = a:time().real

   input = input:cuda()
   gradOutput = gradOutput:cuda()
   local gconv = sconv:clone():cuda()
   gconv:forward(input)
   local rescuda = gconv:backward(input, gradOutput)
   a:reset()
   for i = 1,nloop do
      rescuda = gconv:backward(input, gradOutput)
   end
   cutorch.synchronize()
   tm.gpu = a:time().real

   local error = rescuda:float() - groundgrad

   mytester:assertlt(error:abs():max(), precision_backward, 'error on state (backward) ')
end

function cunntest.SpatialConvolutionMM_forward_single()
   local from = math.random(1,32)
   local to = math.random(1,8) * 8
   local ki = math.random(3,15)
   local kj = math.random(3,15)
   local si = math.random(1,3)
   local sj = math.random(1,3)
   local outi = math.random(1,64)
   local outj = math.random(1,64)
   local padW = math.random(0,1)
   local padH = math.random(0,1)
   local ini = (outi-1)*si+ki-padW*2
   local inj = (outj-1)*sj+kj-padH*2

   local tm = {}
   local title = string.format('SpatialConvolutionMM.forward %dx%dx%d o %dx%d -> %dx%dx%d [s: %dx%d] [p: %dx%d]',
                               from, inj, ini, kj, ki, to, outj, outi, sj, si, padH, padW)
   times[title] = tm

   local input = torch.randn(from,inj,ini)
   local sconv = nn.SpatialConvolutionMM(from,to,ki,kj,si,sj,padW,padH)
   local groundtruth = sconv:forward(input)
   local a = torch.Timer()
   for i = 1,nloop do
      groundtruth = sconv:forward(input)
   end
   tm.cpu = a:time().real

   input = input:cuda()
   local gconv = nn.SpatialConvolutionMM(from,to,ki,kj,si,sj,padW,padH):cuda()
   gconv.weight = sconv.weight:cuda()
   gconv.bias = sconv.bias:cuda()
   local rescuda = gconv:forward(input)
   a:reset()
   for i = 1,nloop do
      rescuda = gconv:forward(input)
   end
   cutorch.synchronize()
   tm.gpu = a:time().real

   local error = rescuda:float() - groundtruth
   mytester:assertlt(error:abs():max(), precision_forward, 'error on state (forward) ')
end

function cunntest.SpatialConvolutionMM_forward_batch()
   local bs = math.random(1,4) * 4
   local from = math.random(1,32)
   local to = math.random(1,8) * 8
   local ki = math.random(3,15)
   local kj = math.random(3,15)
   local si = math.random(1,3)
   local sj = math.random(1,3)
   local outi = math.random(1,64)
   local outj = math.random(1,64)
   local padW = math.random(0,1)
   local padH = math.random(0,1)
   local ini = (outi-1)*si+ki-padW*2
   local inj = (outj-1)*sj+kj-padH*2

   local tm = {}
   local title = string.format('SpatialConvolutionMM.forward %dx%dx%dx%d o %dx%d -> %dx%dx%dx%d [s: %dx%d] [p: %dx%d]',
                               bs, from, inj, ini, kj, ki, bs, to, outj, outi, sj, si, padH, padW)
   times[title] = tm

   local input = torch.randn(bs,from,inj,ini)
   local sconv = nn.SpatialConvolutionMM(from,to,ki,kj,si,sj,padW,padH)
   local groundtruth = sconv:forward(input)
   local a = torch.Timer()
   for i = 1,nloop do
      groundtruth = sconv:forward(input)
   end
   tm.cpu = a:time().real

   input = input:cuda()
   local gconv = nn.SpatialConvolutionMM(from,to,ki,kj,si,sj,padW,padH):cuda()
   gconv.weight = sconv.weight:cuda()
   gconv.bias = sconv.bias:cuda()
   local rescuda = gconv:forward(input)
   a:reset()
   for i = 1,nloop do
      rescuda = gconv:forward(input)
   end
   cutorch.synchronize()
   tm.gpu = a:time().real

   local error = rescuda:float() - groundtruth
   mytester:assertlt(error:abs():max(), precision_forward, 'error on state (forward) ')
end

function cunntest.SpatialConvolutionMM_backward_single()
   local from = math.random(1,32)
   local to = math.random(1,8) * 8
   local ki = math.random(3,15)
   local kj = math.random(3,15)
   local si = math.random(1,3)
   local sj = math.random(1,3)
   local outi = math.random(1,64)
   local outj = math.random(1,64)
   local padW = math.random(0,1)
   local padH = math.random(0,1)
   local ini = (outi-1)*si+ki-padW*2
   local inj = (outj-1)*sj+kj-padH*2

   local tm = {}
   local title = string.format('SpatialConvolutionMM.backward %dx%dx%d o %dx%d -> %dx%dx%d [s: %dx%d] [p: %dx%d]',
                               from, inj, ini, kj, ki, to, outj, outi, sj, si, padH, padW)
   times[title] = tm

   local input = torch.randn(from,inj,ini)
   local gradOutput = torch.randn(to,outj,outi)
   local sconv = nn.SpatialConvolutionMM(from,to,ki,kj,si,sj,padW,padH)
   sconv:forward(input)
   sconv:zeroGradParameters()
   local groundgrad = sconv:backward(input, gradOutput)
   local a = torch.Timer()
   for i = 1,nloop do
      sconv:zeroGradParameters()
      groundgrad = sconv:backward(input, gradOutput)
   end
   local groundweight = sconv.gradWeight
   local groundbias = sconv.gradBias
   tm.cpu = a:time().real

   input = input:cuda()
   gradOutput = gradOutput:cuda()
   local gconv = nn.SpatialConvolutionMM(from,to,ki,kj,si,sj,padW,padH):cuda()
   gconv.weight = sconv.weight:cuda()
   gconv.bias = sconv.bias:cuda()
   gconv:forward(input)
   gconv:zeroGradParameters()
   local rescuda = gconv:backward(input, gradOutput)
   a:reset()
   for i = 1,nloop do
      gconv:zeroGradParameters()
      rescuda = gconv:backward(input, gradOutput)
   end
   local weightcuda = gconv.gradWeight
   local biascuda = gconv.gradBias
   cutorch.synchronize()
   tm.gpu = a:time().real

   local error = rescuda:float() - groundgrad
   local werror = weightcuda:float() - groundweight
   local berror = biascuda:float() - groundbias

   mytester:assertlt(error:abs():max(), precision_backward, 'error on state (backward) ')
   mytester:assertlt(werror:abs():max(), precision_backward, 'error on weight (backward) ')
   mytester:assertlt(berror:abs():max(), precision_backward, 'error on bias (backward) ')
end

function cunntest.SpatialConvolutionMM_backward_batch()
   local bs = math.random(1,4) * 4
   local from = math.random(1,32)
   local to = math.random(1,8) * 8
   local ki = math.random(3,15)
   local kj = math.random(3,15)
   local si = math.random(1,3)
   local sj = math.random(1,3)
   local outi = math.random(1,64)
   local outj = math.random(1,64)
   local padW = math.random(0,1)
   local padH = math.random(0,1)
   local ini = (outi-1)*si+ki-padW*2
   local inj = (outj-1)*sj+kj-padH*2

   local tm = {}
   local title = string.format('SpatialConvolutionMM.backward %dx%dx%dx%d o %dx%d -> %dx%dx%dx%d [s: %dx%d] [p: %dx%d]',
                               bs, from, inj, ini, kj, ki, bs, to, outj, outi, sj, si, padH, padW)
   times[title] = tm

   local input = torch.randn(bs,from,inj,ini)
   local gradOutput = torch.randn(bs,to,outj,outi)
   local sconv = nn.SpatialConvolutionMM(from,to,ki,kj,si,sj,padW,padH)
   sconv:forward(input)
   sconv:zeroGradParameters()
   local groundgrad = sconv:backward(input, gradOutput)
   local a = torch.Timer()
   for i = 1,nloop do
      sconv:zeroGradParameters()
      groundgrad = sconv:backward(input, gradOutput)
   end
   local groundweight = sconv.gradWeight
   local groundbias = sconv.gradBias
   tm.cpu = a:time().real

   input = input:cuda()
   gradOutput = gradOutput:cuda()
   local gconv = nn.SpatialConvolutionMM(from,to,ki,kj,si,sj,padW,padH):cuda()
   gconv.weight = sconv.weight:cuda()
   gconv.bias = sconv.bias:cuda()
   gconv:forward(input)
   gconv:zeroGradParameters()
   local rescuda = gconv:backward(input, gradOutput)
   a:reset()
   for i = 1,nloop do
      gconv:zeroGradParameters()
      rescuda = gconv:backward(input, gradOutput)
   end
   local weightcuda = gconv.gradWeight
   local biascuda = gconv.gradBias
   cutorch.synchronize()
   tm.gpu = a:time().real

   local error = rescuda:float() - groundgrad
   local werror = weightcuda:float() - groundweight
   local berror = biascuda:float() - groundbias

   mytester:assertlt(error:abs():max(), precision_backward, 'error on state (backward) ')
   mytester:assertlt(werror:abs():max(), precision_backward, 'error on weight (backward) ')
   mytester:assertlt(berror:abs():max(), precision_backward, 'error on bias (backward) ')
end

function cunntest.SpatialSubSampling_forward()
   local from = math.random(1,64)
   local to = from
   local ki = math.random(2,4)
   local kj = math.random(2,4)
   local si = math.random(2,4)
   local sj = math.random(2,4)
   local outi = math.random(32,256)
   local outj = math.random(32,256)
   local ini = (outi-1)*si+ki
   local inj = (outj-1)*sj+kj

   local tm = {}
   local title = string.format('SpatialSubSampling.forward %dx%dx%d o %dx%d -> %dx%dx%d',
                               from, inj, ini, kj, ki, to, outj, outi)
   times[title] = tm

   local input = torch.randn(from,inj,ini)
   local sconv = nn.SpatialSubSampling(from,ki,kj,si,sj)
   local groundtruth = sconv:forward(input)
   local a = torch.Timer()
   for i = 1,nloop do
      groundtruth = sconv:forward(input)
   end
   tm.cpu = a:time().real

   input = input:cuda()
   local gconv = nn.SpatialSubSampling(from,ki,kj,si,sj):cuda()
   gconv.weight = sconv.weight:cuda()
   gconv.bias = sconv.bias:cuda()
   local rescuda = gconv:forward(input)
   a:reset()
   for i = 1,nloop do
      rescuda = gconv:forward(input)
   end
   cutorch.synchronize()
   tm.gpu = a:time().real

   local error = rescuda:float() - groundtruth
   mytester:assertlt(error:abs():max(), precision_forward, 'error on state (forward) ')
end

function cunntest.SpatialSubSampling_forward_batch()
   local bs = math.random(4,10)
   local from = math.random(1,64)
   local to = from
   local ki = math.random(2,4)
   local kj = math.random(2,4)
   local si = math.random(2,4)
   local sj = math.random(2,4)
   local outi = math.random(32,256)
   local outj = math.random(32,256)
   local ini = (outi-1)*si+ki
   local inj = (outj-1)*sj+kj

   local tm = {}
   local title = string.format('SpatialSubSampling.forward %dx%dx%dx%d o %dx%d -> %dx%dx%dx%d',
                               bs, from, inj, ini, kj, ki, bs, to, outj, outi)
   times[title] = tm

   local input = torch.randn(bs,from,inj,ini)
   local sconv = nn.SpatialSubSampling(from,ki,kj,si,sj)
   local groundtruth = sconv:forward(input)
   local a = torch.Timer()
   for i = 1,nloop do
      groundtruth = sconv:forward(input)
   end
   tm.cpu = a:time().real

   input = input:cuda()
   local gconv = nn.SpatialSubSampling(from,ki,kj,si,sj):cuda()
   gconv.weight = sconv.weight:cuda()
   gconv.bias = sconv.bias:cuda()
   local rescuda = gconv:forward(input)
   a:reset()
   for i = 1,nloop do
      rescuda = gconv:forward(input)
   end
   cutorch.synchronize()
   tm.gpu = a:time().real

   local error = rescuda:float() - groundtruth
   mytester:assertlt(error:abs():max(), precision_forward, 'error on state (forward) ')
end

function cunntest.SpatialSubSampling_backward()
   local from = math.random(1,64)
   local to = from
   local ki = math.random(2,4)
   local kj = math.random(2,4)
   local si = math.random(2,4)
   local sj = math.random(2,4)
   local outi = math.random(32,64)
   local outj = math.random(32,64)
   local ini = (outi-1)*si+ki
   local inj = (outj-1)*sj+kj

   local tm = {}
   local title = string.format('SpatialSubSampling.backward %dx%dx%d o %dx%d -> %dx%dx%d',
                               from, inj, ini, kj, ki, to, outj, outi)
   times[title] = tm

   local input = torch.randn(from,inj,ini)
   local gradOutput = torch.randn(to,outj,outi)
   local sconv = nn.SpatialSubSampling(from,ki,kj,si,sj)
   sconv:forward(input)
   sconv:zeroGradParameters()
   local groundgrad = sconv:backward(input, gradOutput)
   local a = torch.Timer()
   for i = 1,nloop do
      sconv:zeroGradParameters()
      groundgrad = sconv:backward(input, gradOutput)
   end
   local groundweight = sconv.gradWeight
   local groundbias = sconv.gradBias
   tm.cpu = a:time().real

   input = input:cuda()
   gradOutput = gradOutput:cuda()
   local gconv = nn.SpatialSubSampling(from,ki,kj,si,sj):cuda()
   gconv.weight = sconv.weight:cuda()
   gconv.bias = sconv.bias:cuda()
   gconv:forward(input)
   gconv:zeroGradParameters()
   local rescuda = gconv:backward(input, gradOutput)
   a:reset()
   for i = 1,nloop do
      gconv:zeroGradParameters()
      rescuda = gconv:backward(input, gradOutput)
   end
   local weightcuda = gconv.gradWeight
   local biascuda = gconv.gradBias
   cutorch.synchronize()
   tm.gpu = a:time().real

   local error = rescuda:float() - groundgrad
   local werror = weightcuda:float() - groundweight
   local berror = biascuda:float() - groundbias

   mytester:assertlt(error:abs():max(), precision_backward, 'error on state (backward) ')
   mytester:assertlt(werror:abs():max(), precision_backward, 'error on weight (backward) ')
   mytester:assertlt(berror:abs():max(), precision_backward, 'error on bias (backward) ')
end

function cunntest.SpatialSubSampling_backward_batch()
   local bs = math.random(4,10)
   local from = math.random(1,64)
   local to = from
   local ki = math.random(2,4)
   local kj = math.random(2,4)
   local si = math.random(2,4)
   local sj = math.random(2,4)
   local outi = math.random(32,64)
   local outj = math.random(32,64)
   local ini = (outi-1)*si+ki
   local inj = (outj-1)*sj+kj

   local tm = {}
   local title = string.format('SpatialSubSampling.backward %dx%dx%dx%d o %dx%d -> %dx%dx%dx%d',
                               bs, from, inj, ini, kj, ki, bs, to, outj, outi)
   times[title] = tm

   local input = torch.randn(bs,from,inj,ini)
   local gradOutput = torch.randn(bs,to,outj,outi)
   local sconv = nn.SpatialSubSampling(from,ki,kj,si,sj)
   sconv:forward(input)
   sconv:zeroGradParameters()
   local groundgrad = sconv:backward(input, gradOutput)
   local a = torch.Timer()
   for i = 1,nloop do
      sconv:zeroGradParameters()
      groundgrad = sconv:backward(input, gradOutput)
   end
   local groundweight = sconv.gradWeight
   local groundbias = sconv.gradBias
   tm.cpu = a:time().real

   input = input:cuda()
   gradOutput = gradOutput:cuda()
   local gconv = nn.SpatialSubSampling(from,ki,kj,si,sj):cuda()
   gconv.weight = sconv.weight:cuda()
   gconv.bias = sconv.bias:cuda()
   gconv:forward(input)
   gconv:zeroGradParameters()
   local rescuda = gconv:backward(input, gradOutput)
   a:reset()
   for i = 1,nloop do
      gconv:zeroGradParameters()
      rescuda = gconv:backward(input, gradOutput)
   end
   local weightcuda = gconv.gradWeight
   local biascuda = gconv.gradBias
   cutorch.synchronize()
   tm.gpu = a:time().real

   local error = rescuda:float() - groundgrad
   local werror = weightcuda:float() - groundweight
   local berror = biascuda:float() - groundbias

   mytester:assertlt(error:abs():max(), precision_backward, 'error on state (backward) ')
   mytester:assertlt(werror:abs():max(), precision_backward, 'error on weight (backward) ')
   mytester:assertlt(berror:abs():max(), precision_backward, 'error on bias (backward) ')
end

function cunntest.SpatialMaxPooling_forward()
   local from = math.random(1,64)
   local to = from
   local ki = math.random(2,4)
   local kj = math.random(2,4)
   local si = math.random(1,4)
   local sj = math.random(1,4)
   local outi = math.random(32,256)
   local outj = math.random(32,256)
   local padi = math.random(0,ki/2-1)
   local padj = math.random(0,kj/2-1)
   local ini = (outi-1)*si+ki - padi*2
   local inj = (outj-1)*sj+kj - padj*2
   local ceil_mode = math.random(0,1) == 1

   local tm = {}
   local title = string.format('SpatialMaxPooling.forward %dx%dx%d o %dx%d -> %dx%dx%d',
                               from, inj, ini, kj, ki, to, outj, outi)
   times[title] = tm

   local input = torch.randn(from,inj,ini)
   local sconv = nn.SpatialMaxPooling(ki,kj,si,sj,padi,padj)
   if ceil_mode then sconv:ceil() end
   local groundtruth = sconv:forward(input)
   local a = torch.Timer()
   for i = 1,nloop do
      groundtruth = sconv:forward(input)
   end
   tm.cpu = a:time().real

   input = input:cuda()
   local gconv = nn.SpatialMaxPooling(ki,kj,si,sj,padi,padj):cuda()
   if ceil_mode then gconv:ceil() end
   local rescuda = gconv:forward(input)
   a:reset()
   for i = 1,nloop do
      rescuda = gconv:forward(input)
   end
   cutorch.synchronize()
   tm.gpu = a:time().real

   local error = rescuda:float() - groundtruth
   mytester:assertlt(error:abs():max(), precision_forward, 'error on state (forward) ')
   local error_ind = gconv.indices:float() - sconv.indices
   mytester:asserteq(error_ind:max(), 0, 'error on indices (forward) ')
end

function cunntest.SpatialMaxPooling_forward_batch()
   local bs = math.random(4,10)
   local from = math.random(1,64)
   local to = from
   local ki = math.random(2,4)
   local kj = math.random(2,4)
   local si = math.random(2,4)
   local sj = math.random(2,4)
   local outi = math.random(32,256)
   local outj = math.random(32,256)
   local padi = math.random(0,ki/2-1)
   local padj = math.random(0,kj/2-1)
   local ini = (outi-1)*si+ki - padi*2
   local inj = (outj-1)*sj+kj - padj*2
   local ceil_mode = math.random(0,1) == 1

   local tm = {}
   local title = string.format('SpatialMaxPooling.forward %dx%dx%dx%d o %dx%d -> %dx%dx%dx%d',
                               bs, from, inj, ini, kj, ki, bs, to, outj, outi)
   times[title] = tm

   local input = torch.randn(bs,from,inj,ini)
   local sconv = nn.SpatialMaxPooling(ki,kj,si,sj,padi,padj)
   if ceil_mode then sconv:ceil() end
   local groundtruth = sconv:forward(input)
   local a = torch.Timer()
   for i = 1,nloop do
      groundtruth = sconv:forward(input)
   end
   tm.cpu = a:time().real

   input = input:cuda()
   local gconv = nn.SpatialMaxPooling(ki,kj,si,sj,padi,padj):cuda()
   if ceil_mode then gconv:ceil() end
   local rescuda = gconv:forward(input)
   a:reset()
   for i = 1,nloop do
      rescuda = gconv:forward(input)
   end
   cutorch.synchronize()
   tm.gpu = a:time().real

   local error = rescuda:float() - groundtruth
   mytester:assertlt(error:abs():max(), precision_forward, 'error on state (forward) ')
end

function cunntest.SpatialMaxPooling_backward()
   local from = math.random(1,64)
   local to = from
   local ki = math.random(2,4)
   local kj = math.random(2,4)
   local si = math.random(1,4)
   local sj = math.random(1,4)
   local outi = math.random(32,64)
   local outj = math.random(32,64)
   local padi = math.random(0,ki/2-1)
   local padj = math.random(0,kj/2-1)
   local ini = (outi-1)*si+ki - padi*2
   local inj = (outj-1)*sj+kj - padj*2
   local ceil_mode = math.random(0,1) == 1

   local tm = {}
   local title = string.format('SpatialMaxPooling.backward %dx%dx%d o %dx%d -> %dx%dx%d',
                               from, inj, ini, kj, ki, to, outj, outi)
   times[title] = tm

   local input = torch.randn(from,inj,ini)
   local gradOutput = torch.randn(to,outj,outi)
   local sconv = nn.SpatialMaxPooling(ki,kj,si,sj,padi,padj)
   if ceil_mode then sconv:ceil() end
   sconv:forward(input)
   sconv:zeroGradParameters()
   local groundgrad = sconv:backward(input, gradOutput)
   local a = torch.Timer()
   for i = 1,nloop do
      sconv:zeroGradParameters()
      groundgrad = sconv:backward(input, gradOutput)
   end
   tm.cpu = a:time().real

   input = input:cuda()
   gradOutput = gradOutput:cuda()
   local gconv = nn.SpatialMaxPooling(ki,kj,si,sj,padi,padj):cuda()
   if ceil_mode then gconv:ceil() end
   gconv:forward(input)
   gconv:zeroGradParameters()
   local rescuda = gconv:backward(input, gradOutput)
   a:reset()
   for i = 1,nloop do
      gconv:zeroGradParameters()
      rescuda = gconv:backward(input, gradOutput)
   end
   cutorch.synchronize()
   tm.gpu = a:time().real

   local error = rescuda:float() - groundgrad

   mytester:assertlt(error:abs():max(), precision_backward, 'error on state (backward) ')
end

function cunntest.SpatialMaxPooling_backward_batch()
   local bs = math.random(4,10)
   local from = math.random(1,64)
   local to = from
   local ki = math.random(2,4)
   local kj = math.random(2,4)
   local si = math.random(2,4)
   local sj = math.random(2,4)
   local outi = math.random(32,64)
   local outj = math.random(32,64)
   local padi = math.random(0,ki/2-1)
   local padj = math.random(0,kj/2-1)
   local ini = (outi-1)*si+ki - padi*2
   local inj = (outj-1)*sj+kj - padj*2
   local ceil_mode = math.random(0,1) == 1

   local tm = {}
   local title = string.format('SpatialMaxPooling.backward %dx%dx%dx%d o %dx%d -> %dx%dx%dx%d',
                               bs, from, inj, ini, kj, ki, bs, to, outj, outi)
   times[title] = tm

   local input = torch.randn(bs,from,inj,ini)
   local gradOutput = torch.randn(bs,to,outj,outi)
   local sconv = nn.SpatialMaxPooling(ki,kj,si,sj,padi,padj)
   if ceil_mode then sconv:ceil() end
   sconv:forward(input)
   sconv:zeroGradParameters()
   local groundgrad = sconv:backward(input, gradOutput)
   local a = torch.Timer()
   for i = 1,nloop do
      sconv:zeroGradParameters()
      groundgrad = sconv:backward(input, gradOutput)
   end
   tm.cpu = a:time().real

   input = input:cuda()
   gradOutput = gradOutput:cuda()
   local gconv = nn.SpatialMaxPooling(ki,kj,si,sj,padi,padj):cuda()
   if ceil_mode then gconv:ceil() end
   gconv:forward(input)
   gconv:zeroGradParameters()
   local rescuda = gconv:backward(input, gradOutput)
   a:reset()
   for i = 1,nloop do
      gconv:zeroGradParameters()
      rescuda = gconv:backward(input, gradOutput)
   end
   cutorch.synchronize()
   tm.gpu = a:time().real

   local error = rescuda:float() - groundgrad

   mytester:assertlt(error:abs():max(), precision_backward, 'error on state (backward) ')
end


function cunntest.SpatialFractionalMaxPooling_forward()
    local batch = math.random(1, 3)
    local plane = math.random(1, 3)
    local outW = math.random(1, 7)
    local outH = math.random(1, 7)
    local poolSizeW = math.random(2, 4)
    local poolSizeH = math.random(2, 4)

    local minInW = outW + poolSizeW
    local minInH = outH + poolSizeH

    local inW = math.random(minInW, minInW + 6)
    local inH = math.random(minInH, minInH + 6)

    local useRatio = (math.random(1, 2) == 1)
    local ratioW = outW / inW
    local ratioH = outH / inH

    local tm = {}
    local title =
        string.format('SpatialFractionalMaxPooling.forward %dx%dx%dx%d o %dx%d -> %dx%dx%dx%d',
                      batch, plane, inH, inW, poolSizeH, poolSizeW, batch, plane, outH, outW)
    times[title] = tm

    local input = nil
    if batch == 1 then
        input = torch.Tensor(plane, inH, inW):uniform()
    else
        input = torch.Tensor(batch, plane, inH, inW):uniform()
    end

    local module = nil
    if useRatio then
        module =
            nn.SpatialFractionalMaxPooling(poolSizeW, poolSizeH, ratioW, ratioH)
    else
        module =
            nn.SpatialFractionalMaxPooling(poolSizeW, poolSizeH, outW, outH)
    end

    module:fixPoolingRegions()

    local groundtruth = module:forward(input)
    local a = torch.Timer()
    for i = 1,nloop do
        groundtruth = module:forward(input)
    end
    tm.cpu = a:time().real

    input = input:cuda()

    local gmodule = nil
    if useRatio then
        gmodule =
            nn.SpatialFractionalMaxPooling(poolSizeW, poolSizeH, ratioW, ratioH)
    else
        gmodule =
            nn.SpatialFractionalMaxPooling(poolSizeW, poolSizeH, outW, outH)
    end

    gmodule = gmodule:fixPoolingRegions():cuda()

    -- For comparison purposes, make sure we are using the same random pooling regions
    -- as the CPU
    gmodule.randomSamples = module.randomSamples:cuda()

    local rescuda = gmodule:forward(input)
    a:reset()
    for i = 1,nloop do
        rescuda = gmodule:forward(input)
    end
    cutorch.synchronize()
    tm.gpu = a:time().real

    local error = rescuda:float() - groundtruth
    mytester:assertlt(error:abs():max(), precision_forward, 'error on state (forward) ')
    local error_ind = gmodule.indices:float() - module.indices
    mytester:asserteq(error_ind:abs():max(), 0, 'error on indices (forward) ')
end

function cunntest.SpatialFractionalMaxPooling_backward()
    local batch = math.random(1, 3)
    local plane = math.random(1, 3)
    local outW = math.random(1, 7)
    local outH = math.random(1, 7)
    local poolSizeW = math.random(2, 4)
    local poolSizeH = math.random(2, 4)

    local minInW = outW + poolSizeW
    local minInH = outH + poolSizeH

    local inW = math.random(minInW, minInW + 6)
    local inH = math.random(minInH, minInH + 6)

    local tm = {}
    local title =
        string.format('SpatialFractionalMaxPooling.backward %dx%dx%dx%d o %dx%d -> %dx%dx%dx%d',
                      batch, plane, inH, inW, poolSizeH, poolSizeW, batch, plane, outH, outW)
    times[title] = tm

    local input = nil
    local gradOutput = nil
    if batch == 1 then
        input = torch.Tensor(plane, inH, inW):uniform()
        gradOutput = torch.Tensor(plane, outH, outW):uniform()
    else
        input = torch.Tensor(batch, plane, inH, inW):uniform()
        gradOutput = torch.Tensor(batch, plane, outH, outW):uniform()
    end

    local module =
        nn.SpatialFractionalMaxPooling(poolSizeW, poolSizeH, outW, outH)
        :fixPoolingRegions()

    module:forward(input)
    module:zeroGradParameters()
    local groundgrad = module:backward(input, gradOutput)
    local a = torch.Timer()
    for i = 1,nloop do
        module:zeroGradParameters()
        groundgrad = module:backward(input, gradOutput)
    end
    tm.cpu = a:time().real

    input = input:cuda()
    gradOutput = gradOutput:cuda()

    local gmodule =
        nn.SpatialFractionalMaxPooling(poolSizeW, poolSizeH, outW, outH)
        :fixPoolingRegions():cuda()
    -- For comparison purposes, make sure we are using the same random pooling regions
    -- as the CPU
    gmodule.randomSamples = module.randomSamples:cuda()

    gmodule:forward(input)
    gmodule:zeroGradParameters()
    local rescuda = gmodule:backward(input, gradOutput)
    a:reset()
    for i = 1,nloop do
        gmodule:zeroGradParameters()
        rescuda = gmodule:backward(input, gradOutput)
    end
    cutorch.synchronize()
    tm.gpu = a:time().real

    local error = rescuda:float() - groundgrad
    mytester:assertlt(error:abs():max(), precision_backward, 'error on state (backward) ')
end

function cunntest.SpatialAveragePooling_forward()
   local from = math.random(1,64)
   local to = from
   local ki = math.random(2,4)
   local kj = math.random(2,4)
   local si = math.random(1,ki)
   local sj = math.random(1,kj)
   local outi = math.random(32,256)
   local outj = math.random(32,256)
   local ini = (outi-1)*si+ki
   local inj = (outj-1)*sj+kj

   local tm = {}
   local title = string.format('SpatialAveragePooling.forward %dx%dx%d o %dx%d -> %dx%dx%d',
                               from, inj, ini, kj, ki, to, outj, outi)
   times[title] = tm

   local input = torch.randn(from,inj,ini)
   local sconv = nn.SpatialAveragePooling(ki,kj,si,sj)
   local groundtruth = sconv:forward(input)
   local a = torch.Timer()
   for i = 1,nloop do
      groundtruth = sconv:forward(input)
   end
   tm.cpu = a:time().real

   input = input:cuda()
   local gconv = nn.SpatialAveragePooling(ki,kj,si,sj):cuda()
   local rescuda = gconv:forward(input)
   a:reset()
   for i = 1,nloop do
      rescuda = gconv:forward(input)
   end
   cutorch.synchronize()
   tm.gpu = a:time().real

   local error = rescuda:float() - groundtruth
   mytester:assertlt(error:abs():max(), precision_forward, 'error on state (forward) ')
end

function cunntest.SpatialAveragePooling_forward_batch()
   local bs = math.random(4,10)
   local from = math.random(1,64)
   local to = from
   local ki = math.random(2,4)
   local kj = math.random(2,4)
   local si = math.random(1,ki)
   local sj = math.random(1,kj)
   local outi = math.random(32,256)
   local outj = math.random(32,256)
   local ini = (outi-1)*si+ki
   local inj = (outj-1)*sj+kj

   local tm = {}
   local title = string.format('SpatialAveragePooling.forward %dx%dx%dx%d o %dx%d -> %dx%dx%dx%d',
                               bs, from, inj, ini, kj, ki, bs, to, outj, outi)
   times[title] = tm

   local input = torch.randn(bs,from,inj,ini)
   local sconv = nn.SpatialAveragePooling(ki,kj,si,sj)
   local groundtruth = sconv:forward(input)
   local a = torch.Timer()
   for i = 1,nloop do
      groundtruth = sconv:forward(input)
   end
   tm.cpu = a:time().real

   input = input:cuda()
   local gconv = nn.SpatialAveragePooling(ki,kj,si,sj):cuda()
   local rescuda = gconv:forward(input)
   a:reset()
   for i = 1,nloop do
      rescuda = gconv:forward(input)
   end
   cutorch.synchronize()
   tm.gpu = a:time().real

   local error = rescuda:float() - groundtruth
   mytester:assertlt(error:abs():max(), precision_forward, 'error on state (forward) ')
end

function cunntest.SpatialAveragePooling_backward()
   local from = math.random(1,64)
   local to = from
   local ki = math.random(2,4)
   local kj = math.random(2,4)
   local si = math.random(1,ki)
   local sj = math.random(1,kj)
   local outi = math.random(32,64)
   local outj = math.random(32,64)
   local ini = (outi-1)*si+ki
   local inj = (outj-1)*sj+kj

   local tm = {}
   local title = string.format('SpatialMaxPooling.backward %dx%dx%d o %dx%d -> %dx%dx%d',
                               from, inj, ini, kj, ki, to, outj, outi)
   times[title] = tm

   local input = torch.randn(from,inj,ini)
   local gradOutput = torch.randn(to,outj,outi)
   local sconv = nn.SpatialAveragePooling(ki,kj,si,sj)
   sconv:forward(input)
   sconv:zeroGradParameters()
   local groundgrad = sconv:backward(input, gradOutput)
   local a = torch.Timer()
   for i = 1,nloop do
      sconv:zeroGradParameters()
      groundgrad = sconv:backward(input, gradOutput)
   end
   tm.cpu = a:time().real

   input = input:cuda()
   gradOutput = gradOutput:cuda()
   local gconv = nn.SpatialAveragePooling(ki,kj,si,sj):cuda()
   gconv:forward(input)
   gconv:zeroGradParameters()
   local rescuda = gconv:backward(input, gradOutput)
   a:reset()
   for i = 1,nloop do
      gconv:zeroGradParameters()
      rescuda = gconv:backward(input, gradOutput)
   end
   cutorch.synchronize()
   tm.gpu = a:time().real

   local error = rescuda:float() - groundgrad

   mytester:assertlt(error:abs():max(), precision_backward, 'error on state (backward) ')
end

function cunntest.SpatialAveragePooling_backward_batch()
   local bs = math.random(4,10)
   local from = math.random(1,64)
   local to = from
   local ki = math.random(2,4)
   local kj = math.random(2,4)
   local si = math.random(1,ki)
   local sj = math.random(1,kj)
   local outi = math.random(32,64)
   local outj = math.random(32,64)
   local ini = (outi-1)*si+ki
   local inj = (outj-1)*sj+kj

   local tm = {}
   local title = string.format('SpatialAveragePooling.backward %dx%dx%dx%d o %dx%d -> %dx%dx%dx%d',
                               bs, from, inj, ini, kj, ki, bs, to, outj, outi)
   times[title] = tm

   local input = torch.randn(bs,from,inj,ini)
   local gradOutput = torch.randn(bs,to,outj,outi)
   local sconv = nn.SpatialAveragePooling(ki,kj,si,sj)
   sconv:forward(input)
   sconv:zeroGradParameters()
   local groundgrad = sconv:backward(input, gradOutput)
   local a = torch.Timer()
   for i = 1,nloop do
      sconv:zeroGradParameters()
      groundgrad = sconv:backward(input, gradOutput)
   end
   tm.cpu = a:time().real

   input = input:cuda()
   gradOutput = gradOutput:cuda()
   local gconv = nn.SpatialAveragePooling(ki,kj,si,sj):cuda()
   gconv:forward(input)
   gconv:zeroGradParameters()
   local rescuda = gconv:backward(input, gradOutput)
   a:reset()
   for i = 1,nloop do
      gconv:zeroGradParameters()
      rescuda = gconv:backward(input, gradOutput)
   end
   cutorch.synchronize()
   tm.gpu = a:time().real

   local error = rescuda:float() - groundgrad

   mytester:assertlt(error:abs():max(), precision_backward, 'error on state (backward) ')
end

function cunntest.SpatialAdaptiveMaxPooling_forward()
   local from = math.random(1,64)
   local to = from
   local outi = math.random(2,64)
   local outj = math.random(2,64)
   local ini = math.random(10,256)
   local inj = math.random(10,256)

   local tm = {}
   local title = string.format('SpatialAdaptiveMaxPooling.forward %dx%dx%d -> %dx%dx%d',
                               from, inj, ini, to, outj, outi)
   times[title] = tm

   local input = torch.randn(from,inj,ini)
   local sconv = nn.SpatialAdaptiveMaxPooling(outi,outj)
   local groundtruth = sconv:forward(input)
   local a = torch.Timer()
   for i = 1,nloop do
      groundtruth = sconv:forward(input)
   end
   tm.cpu = a:time().real

   input = input:cuda()
   local gconv = nn.SpatialAdaptiveMaxPooling(outi,outj):cuda()
   local rescuda = gconv:forward(input)
   a:reset()
   for i = 1,nloop do
      rescuda = gconv:forward(input)
   end
   cutorch.synchronize()
   tm.gpu = a:time().real

   local error = rescuda:float() - groundtruth
   mytester:assertlt(error:abs():max(), precision_forward, 'error on state (forward) ')
   local error_ind = gconv.indices:float() - sconv.indices
   mytester:asserteq(error_ind:max(), 0, 'error on indices (forward) ')
end

function cunntest.SpatialAdaptiveMaxPooling_forward_noncontig()
   local from = math.random(1,64)
   local to = from
   local outi = math.random(2,64)
   local outj = math.random(2,64)
   local ini = math.random(10,256)
   local inj = math.random(10,256)

   local tm = {}
   local title = string.format('SpatialAdaptiveMaxPooling.forward %s %dx%dx%d -> %dx%dx%d',
                               'non-contiguous',from, inj, ini, to, outj, outi)
   times[title] = tm

   local input0 = torch.randn(from,ini,inj)
   local input = input0:transpose(2,3)
   local sconv = nn.SpatialAdaptiveMaxPooling(outi,outj)
   local groundtruth = sconv:forward(input)
   local a = torch.Timer()
   for i = 1,nloop do
      groundtruth = sconv:forward(input)
   end
   tm.cpu = a:time().real

   input = input0:cuda():transpose(2,3)
   local gconv = nn.SpatialAdaptiveMaxPooling(outi,outj):cuda()
   local rescuda = gconv:forward(input)
   a:reset()
   for i = 1,nloop do
      rescuda = gconv:forward(input)
   end
   cutorch.synchronize()
   tm.gpu = a:time().real

   local error = rescuda:float() - groundtruth
   mytester:assertlt(error:abs():max(), precision_forward, 'error on state (forward) ')
   local error_ind = gconv.indices:float() - sconv.indices
   mytester:asserteq(error_ind:max(), 0, 'error on indices (forward) ')
end

function cunntest.SpatialAdaptiveMaxPooling_forward_batch()
   local bs = math.random(4,10)
   local from = math.random(1,64)
   local to = from
   local outi = math.random(2,64)
   local outj = math.random(2,64)
   local ini = math.random(10,256)
   local inj = math.random(10,256)

   local tm = {}
   local title = string.format('SpatialAdaptiveMaxPooling.forward %dx%dx%dx%d -> %dx%dx%dx%d',
                               bs, from, inj, ini, bs, to, outj, outi)
   times[title] = tm

   local input = torch.randn(bs,from,inj,ini)
   local sconv = nn.SpatialAdaptiveMaxPooling(outi,outj)
   local groundtruth = sconv:forward(input)
   local a = torch.Timer()
   for i = 1,nloop do
      groundtruth = sconv:forward(input)
   end
   tm.cpu = a:time().real

   input = input:cuda()
   local gconv = nn.SpatialAdaptiveMaxPooling(outi,outj):cuda()
   local rescuda = gconv:forward(input)
   a:reset()
   for i = 1,nloop do
      rescuda = gconv:forward(input)
   end
   cutorch.synchronize()
   tm.gpu = a:time().real

   local error = rescuda:float() - groundtruth
   mytester:assertlt(error:abs():max(), precision_forward, 'error on state (forward) ')
end

function cunntest.SpatialAdaptiveMaxPooling_backward()
   local from = math.random(1,64)
   local to = from
   local outi = math.random(2,64)
   local outj = math.random(2,64)
   local ini = math.random(10,256)
   local inj = math.random(10,256)

   local tm = {}
   local title = string.format('SpatialAdaptiveMaxPooling.backward %dx%dx%d -> %dx%dx%d',
                               from, inj, ini, to, outj, outi)
   times[title] = tm

   local input = torch.randn(from,inj,ini)
   local gradOutput = torch.randn(to,outj,outi)
   local sconv = nn.SpatialAdaptiveMaxPooling(outi,outj)
   sconv:forward(input)
   sconv:zeroGradParameters()
   local groundgrad = sconv:backward(input, gradOutput)
   local a = torch.Timer()
   for i = 1,nloop do
      sconv:zeroGradParameters()
      groundgrad = sconv:backward(input, gradOutput)
   end
   tm.cpu = a:time().real

   input = input:cuda()
   gradOutput = gradOutput:cuda()
   local gconv = nn.SpatialAdaptiveMaxPooling(outi,outj):cuda()
   gconv:forward(input)
   gconv:zeroGradParameters()
   local rescuda = gconv:backward(input, gradOutput)
   a:reset()
   for i = 1,nloop do
      gconv:zeroGradParameters()
      rescuda = gconv:backward(input, gradOutput)
   end
   cutorch.synchronize()
   tm.gpu = a:time().real

   local error = rescuda:float() - groundgrad

   mytester:assertlt(error:abs():max(), precision_backward, 'error on state (backward) ')
end

function cunntest.SpatialAdaptiveMaxPooling_backward_noncontig()
   local from = math.random(1,64)
   local to = from
   local outi = math.random(2,64)
   local outj = math.random(2,64)
   local ini = math.random(10,256)
   local inj = math.random(10,256)

   local tm = {}
   local title = string.format('SpatialAdaptiveMaxPooling.backward %s %dx%dx%d -> %dx%dx%d',
                               'non-contiguous', from, inj, ini, to, outj, outi)
   times[title] = tm

   local input0 = torch.randn(from,ini,inj)
   local input = input0:transpose(2,3)
   local gradOutput = torch.randn(to,outj,outi)
   local sconv = nn.SpatialAdaptiveMaxPooling(outi,outj)
   sconv:forward(input)
   sconv:zeroGradParameters()
   local groundgrad = sconv:backward(input, gradOutput)
   local a = torch.Timer()
   for i = 1,nloop do
      sconv:zeroGradParameters()
      groundgrad = sconv:backward(input, gradOutput)
   end
   tm.cpu = a:time().real

   input = input0:cuda():transpose(2,3)
   gradOutput = gradOutput:cuda()
   local gconv = nn.SpatialAdaptiveMaxPooling(outi,outj):cuda()
   gconv:forward(input)
   gconv:zeroGradParameters()
   local rescuda = gconv:backward(input, gradOutput)
   a:reset()
   for i = 1,nloop do
      gconv:zeroGradParameters()
      rescuda = gconv:backward(input, gradOutput)
   end
   cutorch.synchronize()
   tm.gpu = a:time().real

   local error = rescuda:float() - groundgrad

   mytester:assertlt(error:abs():max(), precision_backward, 'error on state (backward) ')
end

function cunntest.SpatialAdaptiveMaxPooling_backward_batch()
   local bs = math.random(4,10)
   local from = math.random(1,64)
   local to = from
   local outi = math.random(2,64)
   local outj = math.random(2,64)
   local ini = math.random(10,256)
   local inj = math.random(10,256)

   local tm = {}
   local title = string.format('SpatialAdaptiveMaxPooling.backward %dx%dx%dx%d -> %dx%dx%dx%d',
                               bs, from, inj, ini, bs, to, outj, outi)
   times[title] = tm

   local input = torch.randn(bs,from,inj,ini)
   local gradOutput = torch.randn(bs,to,outj,outi)
   local sconv = nn.SpatialAdaptiveMaxPooling(outi,outj)
   sconv:forward(input)
   sconv:zeroGradParameters()
   local groundgrad = sconv:backward(input, gradOutput)
   local a = torch.Timer()
   for i = 1,nloop do
      sconv:zeroGradParameters()
      groundgrad = sconv:backward(input, gradOutput)
   end
   tm.cpu = a:time().real

   input = input:cuda()
   gradOutput = gradOutput:cuda()
   local gconv = nn.SpatialAdaptiveMaxPooling(outi,outj):cuda()
   gconv:forward(input)
   gconv:zeroGradParameters()
   local rescuda = gconv:backward(input, gradOutput)
   a:reset()
   for i = 1,nloop do
      gconv:zeroGradParameters()
      rescuda = gconv:backward(input, gradOutput)
   end
   cutorch.synchronize()
   tm.gpu = a:time().real

   local error = rescuda:float() - groundgrad

   mytester:assertlt(error:abs():max(), precision_backward, 'error on state (backward) ')
end

function cunntest.SpatialLPPooling_forward()
   local from = math.random(1,64)
   local to = from
   local pnorm = 2
   local ki = math.random(2,4)
   local kj = math.random(2,4)
   local si = ki
   local sj = kj
   local outi = math.random(32,256)
   local outj = math.random(32,256)
   local ini = (outi-1)*si+ki
   local inj = (outj-1)*sj+kj

   local tm = {}
   local title = string.format('SpatialLPPooling.forward (P=2 only) %dx%dx%d o %dx%d -> %dx%dx%d',
                               from, inj, ini, kj, ki, to, outj, outi)
   times[title] = tm

   local input = torch.randn(from,inj,ini)
   local sconv = nn.SpatialLPPooling(from,pnorm,ki,kj,si,sj)
   local groundtruth = sconv:forward(input)
   local a = torch.Timer()
   for i = 1,nloop do
      groundtruth = sconv:forward(input)
   end
   tm.cpu = a:time().real

   input = input:cuda()
   local gconv = nn.SpatialLPPooling(from,pnorm,ki,kj,si,sj):cuda()
   local rescuda = gconv:forward(input)
   a:reset()
   for i = 1,nloop do
      rescuda = gconv:forward(input)
   end
   cutorch.synchronize()
   tm.gpu = a:time().real

   local error = rescuda:float() - groundtruth
   mytester:assertlt(error:abs():max(), precision_forward, 'error on state (forward) ')
end

function cunntest.SpatialLPPooling_backward()
   local from = math.random(1,64)
   local to = from
   local pnorm = 2
   local ki = math.random(2,4)
   local kj = math.random(2,4)
   local si = ki
   local sj = kj
   local outi = math.random(32,64)
   local outj = math.random(32,64)
   local ini = (outi-1)*si+ki
   local inj = (outj-1)*sj+kj

   local tm = {}
   local title = string.format('SpatialLPPooling.backward (P=2 only) %dx%dx%d o %dx%d -> %dx%dx%d',
                               from, inj, ini, kj, ki, to, outj, outi)
   times[title] = tm

   local input = torch.randn(from,inj,ini)
   local gradOutput = torch.randn(to,outj,outi)
   local sconv = nn.SpatialLPPooling(from,pnorm,ki,kj,si,sj)
   sconv:forward(input)
   sconv:zeroGradParameters()
   local groundgrad = sconv:backward(input, gradOutput)
   local a = torch.Timer()
   for i = 1,nloop do
      sconv:zeroGradParameters()
      groundgrad = sconv:backward(input, gradOutput)
   end
   tm.cpu = a:time().real

   input = input:cuda()
   gradOutput = gradOutput:cuda()
   local gconv = sconv:clone():cuda()
   gconv:forward(input)
   gconv:zeroGradParameters()
   local rescuda = gconv:backward(input, gradOutput)
   a:reset()
   for i = 1,nloop do
      gconv:zeroGradParameters()
      rescuda = gconv:backward(input, gradOutput)
   end
   cutorch.synchronize()
   tm.gpu = a:time().real

   local error = rescuda:float() - groundgrad

   mytester:assertlt(error:abs():max(), precision_backward, 'error on state (backward) ')
end


-- Criterion tests

function cunntest.MarginCriterion_forward()
  local size = math.random(1,100)
  local input = (torch.rand(size)-0.5) * 2 -- data spread from -1 to 1
  local target = (torch.round(torch.rand(size))*2)-1 -- generate random labels -1, 1

  local tm = {}
  local title = string.format('MarginCriterion.forward, Size: %d', size)
  times[title] = tm

  local crit = nn.MarginCriterion()
  local groundtruth= crit:forward(input, target)
  local a = torch.Timer()
  for i = 1,nloop do
     groundtruth = crit:forward(input, target)
  end
  tm.cpu = a:time().real

  input = input:cuda()
  target = target:cuda()
  local g_crit = nn.MarginCriterion():cuda()
  local rescuda = g_crit:forward(input, target)
  a:reset()
  for i = 1,nloop do
     rescuda = g_crit:forward(input, target)
  end
  cutorch.synchronize()
  tm.gpu = a:time().real
  local errorVal = rescuda - groundtruth
  mytester:assertlt(errorVal, precision_forward, 'error on state (forward) ')
end

function cunntest.MarginCriterion_backward()
   local size = math.random(1,100)

   local tm = {}
   local title = string.format('MarginCriterion.backward, Size %d', size)
   times[title] = tm

   local input = (torch.rand(size)-0.5) * 2 -- data spread from -1 to 1
   local target = (torch.round(torch.rand(size))*2)-1 -- generate random labels -1, 1
   
   local crit = nn.MarginCriterion()
   crit:forward(input, target)
   local groundgrad = crit:backward(input, target)
   local a = torch.Timer()
   for i = 1,nloop do
      groundgrad = crit:backward(input, target)
   end
   tm.cpu = a:time().real

   input = input:cuda()
   target = target:cuda()
   local g_crit = nn.MarginCriterion():cuda()
   g_crit:forward(input, target)
   local rescuda = g_crit:backward(input, target)
   a:reset()
   for i = 1,nloop do
      rescuda = g_crit:backward(input, target)
   end
   cutorch.synchronize()
   tm.gpu = a:time().real

   local error = rescuda:float() - groundgrad

   mytester:assertlt(error:abs():max(), precision_backward, 'error on state (backward) ')
end

function cunntest.mse()
   for sizeAverage = 0, 1 do
      local size = math.random(3000,5000)
      local input = torch.randn(size,1,1)
      local target = torch.randn(size)
      local mod = nn.MSECriterion(sizeAverage == 1)

      local tm = {}
      local title = string.format('MSECriterion sizeAverage %d, %d ', sizeAverage, size)
      times[title] = tm

      local a = torch.Timer()
      local fout = mod:forward(input,target)
      local fgin = mod:backward(input,target):clone()
      tm.cpu = a:time().real

      local cinput = input:cuda()
      local ctarget = target:cuda()
      local cmod = nn.MSECriterion(sizeAverage == 1):cuda()
      a:reset()
      local cout = cmod:forward(cinput,ctarget)
      local cgin = cmod:backward(cinput,ctarget)
      cutorch.synchronize()
      tm.gpu = a:time().real

      local tm2 = {}
      local title = string.format('MSECriterion2 sizeAverage %d, %d ',sizeAverage, size)
      times[title] = tm2
      tm2.cpu = tm.cpu
      local cinput2 = input:cuda()
      local ctarget2 = target:cuda()
      local cmod2 = nn.MSECriterion(sizeAverage == 1):cuda()
      a:reset()
      local cout2 = cinput2.nn.MSECriterion_updateOutput2(cmod,cinput2,ctarget2)
      local cgin2 = cinput2.nn.MSECriterion_updateGradInput2(cmod,cinput2,ctarget2)
      cutorch.synchronize()
      tm2.gpu = a:time().real

      mytester:assertlt(math.abs(fout-cout), precision_forward, 'error  on output')
      local gerr = cgin:float() - fgin
      mytester:assertlt(gerr:abs():max(), precision_forward, 'error  on gradInput')

      mytester:assertlt(math.abs(fout-cout2), precision_forward, 'error  on output - 2')
      local gerr2 = cgin2:float() - fgin
      mytester:assertlt(gerr2:abs():max(), precision_forward, 'error  on gradInput -2')
   end
end

function cunntest.SmoothL1()
   for sizeAverage = 0, 1 do
      local size = math.random(3000,5000)
      local input = torch.randn(size,1,1)
      local target = torch.randn(size)
      local mod = nn.SmoothL1Criterion(sizeAverage == 1)

      local tm = {}
      local title = string.format('SmoothL1Criterion sizeAverage %d, %d ', sizeAverage, size)
      times[title] = tm

      local a = torch.Timer()
      local fout = mod:forward(input,target)
      local fgin = mod:backward(input,target):clone()
      tm.cpu = a:time().real

      local cinput = input:cuda()
      local ctarget = target:cuda()
      local cmod = nn.SmoothL1Criterion(sizeAverage == 1):cuda()
      a:reset()
      local cout = cmod:forward(cinput,ctarget)
      local cgin = cmod:backward(cinput,ctarget)
      cutorch.synchronize()
      tm.gpu = a:time().real

      mytester:assertlt(math.abs(fout-cout), precision_forward, 'error  on output')
      local gerr = cgin:float() - fgin
      mytester:assertlt(gerr:abs():max(), precision_forward, 'error  on gradInput')
   end
end

function cunntest.distkldiv()
   for sizeAverage = 0, 1 do
      local size = math.random(3000,5000)
      local input = torch.randn(size,1,1)
      local target = torch.randn(size)
      local mod = nn.DistKLDivCriterion(sizeAverage == 1)

      local tm = {}
      local title = string.format('DistKLDivCriterion sizeAverage %d, %d ',sizeAverage,size)
      times[title] = tm

      local a = torch.Timer()
      local fout = mod:forward(input,target)
      local fgin = mod:backward(input,target):clone()
      tm.cpu = a:time().real

      local cinput = input:cuda()
      local ctarget = target:cuda()
      local cmod = nn.DistKLDivCriterion(sizeAverage == 1):cuda()
      a:reset()
      local cout = cmod:forward(cinput,ctarget)
      local cgin = cmod:backward(cinput,ctarget)
      cutorch.synchronize()
      tm.gpu = a:time().real

      mytester:assertlt(math.abs(fout-cout), precision_forward, 'error  on output')
      local gerr = cgin:float() - fgin
      mytester:assertlt(gerr:abs():max(), precision_backward, 'error  on gradInput')
   end
end

function cunntest.TemporalConvolution_forward()
   local from = math.random(1,64) -- inputFrameSize
   local to = math.random(1,64) -- outputFrameSize
   local ki = math.random(3,15) -- kernelWidth (kW)
   local si = math.random(1,2) -- stepSize (dW)
   local outi = math.random(1,256) -- nOutputFrame
   local ini = (outi-1)*si+ki -- nInputFrame

   local tm = {}
   local title = string.format('TemporalConvolution.forward %dx%d o %d -> %dx%d [s: %d]',
                               from, ini, ki, to, outi, si)
   times[title] = tm

   local input = torch.randn(ini,from)
   local sconv = nn.TemporalConvolution(from,to,ki,si)
   local groundtruth = sconv:forward(input)
   local a = torch.Timer()
   for i = 1,nloop do
      groundtruth = sconv:forward(input)
   end
   tm.cpu = a:time().real

   input = input:cuda()
   local gconv = nn.TemporalConvolution(from,to,ki,si):cuda()
   gconv.weight = sconv.weight:cuda()
   gconv.bias = sconv.bias:cuda()
   local rescuda = gconv:forward(input)
   a:reset()
   for i = 1,nloop do
      rescuda = gconv:forward(input)
   end
   cutorch.synchronize()
   tm.gpu = a:time().real

   local error = rescuda:float() - groundtruth
   mytester:assertlt(error:abs():max(), precision_forward, 'error on state (forward) ')
end

function cunntest.TemporalConvolution_forward_batch()
   local bs = math.random(4,16)
   local from = math.random(1,64)
   local to = math.random(1,64)
   local ki = math.random(3,15)
   local si = math.random(1,2)
   local outi = math.random(1,256)
   local ini = (outi-1)*si+ki

   local tm = {}
   local title = string.format('TemporalConvolution.forward %dx%dx%d o %d -> %dx%dx%d [s: %d]',
                               bs, from, ini, ki, bs, to, outi, si)
   times[title] = tm

   local input = torch.randn(bs,ini,from)
   local sconv = nn.TemporalConvolution(from,to,ki,si)
   local groundtruth = sconv:forward(input)
   local a = torch.Timer()
   for i = 1,nloop do
      groundtruth = sconv:forward(input)
   end
   tm.cpu = a:time().real

   input = input:cuda()
   local gconv = nn.TemporalConvolution(from,to,ki,si):cuda()
   gconv.weight = sconv.weight:cuda()
   gconv.bias = sconv.bias:cuda()
   local rescuda = gconv:forward(input)
   a:reset()
   for i = 1,nloop do
      rescuda = gconv:forward(input)
   end
   cutorch.synchronize()
   tm.gpu = a:time().real

   local error = rescuda:float() - groundtruth
   mytester:assertlt(error:abs():max(), precision_forward, 'error on state (forward) ')
end

function cunntest.TemporalConvolution_backward()
  local from = math.random(1,64)
   local to = math.random(1,64)
   local ki = math.random(3,15)
   local si = math.random(1,2)
   local outi = math.random(1,256)
   local ini = (outi-1)*si+ki

   local tm = {}
   local title = string.format('TemporalConvolution.backward %dx%d o %d -> %dx%d',
                               from, ini, ki, to, outi)

   times[title] = tm

   local input = torch.randn(ini,from)
   local gradOutput = torch.randn(outi,to)
   local sconv = nn.TemporalConvolution(from,to,ki,si)
   sconv:forward(input)
   sconv:zeroGradParameters()
   local groundgrad = sconv:backward(input, gradOutput)
   local a = torch.Timer()
   for i = 1,nloop do
      sconv:zeroGradParameters()
      groundgrad = sconv:backward(input, gradOutput)
   end
   local groundweight = sconv.gradWeight
   local groundbias = sconv.gradBias
   tm.cpu = a:time().real

   input = input:cuda()
   gradOutput = gradOutput:cuda()
   local gconv = nn.TemporalConvolution(from,to,ki,si):cuda()
   gconv.weight = sconv.weight:cuda()
   gconv.bias = sconv.bias:cuda()
   gconv:forward(input)
   gconv:zeroGradParameters()
   local rescuda = gconv:backward(input, gradOutput)
   a:reset()
   for i = 1,nloop do
      gconv:zeroGradParameters()
      rescuda = gconv:backward(input, gradOutput)
   end
   local weightcuda = gconv.gradWeight
   local biascuda = gconv.gradBias
   cutorch.synchronize()
   tm.gpu = a:time().real

   local error = rescuda:float() - groundgrad
   local werror = weightcuda:float() - groundweight
   local berror = biascuda:float() - groundbias

   mytester:assertlt(error:abs():max(), precision_backward, 'error on state (backward) ')
   mytester:assertlt(werror:abs():max(), precision_backward, 'error on weight (backward) ')
   mytester:assertlt(berror:abs():max(), precision_backward, 'error on bias (backward) ')
end

function cunntest.TemporalConvolution_backward_batch()
   local bs = math.random(4,16)
   local from = math.random(1,64)
   local to = math.random(1,64)
   local ki = math.random(3,15)
   local si = math.random(1,2)
   local outi = math.random(1,256)
   local ini = (outi-1)*si+ki

   local tm = {}
   local title = string.format('TemporalConvolution.backward %dx%dx%d o %d -> %dx%dx%d',
                               bs, from, ini, ki, bs, to, outi)
   times[title] = tm

   local input = torch.randn(bs,ini,from)
   local gradOutput = torch.randn(bs,outi,to)
   local sconv = nn.TemporalConvolution(from,to,ki,si)
   sconv:forward(input)
   sconv:zeroGradParameters()
   local groundgrad = sconv:backward(input, gradOutput)
   local a = torch.Timer()
   for i = 1,nloop do
      sconv:zeroGradParameters()
      groundgrad = sconv:backward(input, gradOutput)
   end
   local groundweight = sconv.gradWeight
   local groundbias = sconv.gradBias
   tm.cpu = a:time().real

   input = input:cuda()
   gradOutput = gradOutput:cuda()
   local gconv = nn.TemporalConvolution(from,to,ki,si):cuda()
   gconv.weight = sconv.weight:cuda()
   gconv.bias = sconv.bias:cuda()
   gconv:forward(input)
   gconv:zeroGradParameters()
   local rescuda = gconv:backward(input, gradOutput)
   a:reset()
   for i = 1,nloop do
      gconv:zeroGradParameters()
      rescuda = gconv:backward(input, gradOutput)
   end
   local weightcuda = gconv.gradWeight
   local biascuda = gconv.gradBias
   cutorch.synchronize()
   tm.gpu = a:time().real

   local error = rescuda:float() - groundgrad
   local werror = weightcuda:float() - groundweight
   local berror = biascuda:float() - groundbias

   mytester:assertlt(error:abs():max(), precision_backward, 'error on state (backward) ')
   mytester:assertlt(werror:abs():max(), precision_backward, 'error on weight (backward) ')
   mytester:assertlt(berror:abs():max(), precision_backward, 'error on bias (backward) ')
end

function cunntest.Dropout()
   local p = 0.2 --prob of droping out a neuron
   local input = torch.CudaTensor(1000):fill((1-p))
   local module = nn.Dropout(p)
   module:cuda()
   -- version 2
   local output = module:forward(input)
   mytester:assert(math.abs(output:mean() - (1-p)) < 0.05, 'dropout output')
   local gradInput = module:backward(input, input)
   mytester:assert(math.abs(gradInput:mean() - (1-p)) < 0.05, 'dropout gradInput')
   -- version 1 (old nnx version)
   local input = input:fill(1)
   local module = nn.Dropout(p,true)
   module:cuda()
   local output = module:forward(input)
   mytester:assert(math.abs(output:mean() - (1-p)) < 0.05, 'dropout output')
   local gradInput = module:backward(input, input)
   mytester:assert(math.abs(gradInput:mean() - (1-p)) < 0.05, 'dropout gradInput')
end

function cunntest.Dropout_forward()
   local size = math.random(1,200)

   local tm = {}
   local title = string.format('Dropout forward %d -> %d', size, size)
   times[title] = tm

   local input = torch.randn(size)
   local sconv = nn.Dropout()
   local groundtruth = sconv:forward(input)
   local a = torch.Timer()
   for i = 1,nloop do
      groundtruth = sconv:forward(input)
   end
   tm.cpu = a:time().real

   input = input:cuda()
   local gconv = nn.Dropout():cuda()
   local rescuda = gconv:forward(input)
   a:reset()
   for i = 1,nloop do
      rescuda = gconv:forward(input)
   end
   cutorch.synchronize()
   tm.gpu = a:time().real

end

function cunntest.SoftPlus_forward()
   local size = math.random(1,100)

   local tm = {}
   local title = string.format('SoftPlus forward %d -> %d', size, size)
   times[title] = tm

   local input = torch.randn(size)
   local sconv = nn.SoftPlus()
   local groundtruth = sconv:forward(input)
   local a = torch.Timer()
   for i = 1,nloop do
      groundtruth = sconv:forward(input)
   end
   tm.cpu = a:time().real

   input = input:cuda()
   local gconv = nn.SoftPlus():cuda()
   local rescuda = gconv:forward(input)
   a:reset()
   for i = 1,nloop do
      rescuda = gconv:forward(input)
   end
   cutorch.synchronize()
   tm.gpu = a:time().real

   local error = rescuda:float() - groundtruth
   mytester:assertlt(error:abs():max(), precision_forward, 'error on state (forward) ')
end

function cunntest.SoftPlus_backward()
   local size = math.random(1,100)

   local tm = {}
   local title = string.format('SoftPlus.backward %d -> %d', size, size)
   times[title] = tm

   local input = torch.randn(size)
   local gradOutput = torch.randn(size)
   local sconv = nn.SoftPlus()
   sconv:forward(input)
   local groundgrad = sconv:backward(input, gradOutput)
   local a = torch.Timer()
   for i = 1,nloop do
      groundgrad = sconv:backward(input, gradOutput)
   end
   tm.cpu = a:time().real

   input = input:cuda()
   gradOutput = gradOutput:cuda()
   local gconv = sconv:clone():cuda()
   gconv:forward(input)
   local rescuda = gconv:backward(input, gradOutput)
   a:reset()
   for i = 1,nloop do
      rescuda = gconv:backward(input, gradOutput)
   end
   cutorch.synchronize()
   tm.gpu = a:time().real

   local error = rescuda:float() - groundgrad

   mytester:assertlt(error:abs():max(), precision_backward, 'error on state (backward) ')
end

function cunntest.SpatialUpSamplingNearest_forward()
   local f = torch.random(3, 15)
   local h = torch.random(3, 15)
   local w = torch.random(3, 15)
   local scale = torch.random(2,5)

   local tm = {}
   local title = string.format('SpatialUpSamplingNearest.forward %dx%dx%d -> %dx%dx%d',
                               f, h, w, f, h*scale, w*scale)
   times[title] = tm

   local input = torch.randn(f, h, w)
   local sconv = nn.SpatialUpSamplingNearest(scale)
   local groundtruth = sconv:forward(input)
   local a = torch.Timer()
   for i = 1,nloop do
      groundtruth = sconv:forward(input)
   end
   tm.cpu = a:time().real

   input = input:cuda()
   local gconv = sconv:clone():cuda()
   local rescuda = gconv:forward(input)
   a:reset()
   for i = 1,nloop do
      rescuda = gconv:forward(input)
   end
   cutorch.synchronize()
   tm.gpu = a:time().real

   local error = rescuda:float() - groundtruth
   mytester:assertlt(error:abs():max(), precision_forward, 'error on state (forward) ')
end

function cunntest.SpatialUpSamplingNearest_forward_batch()
   local nbatch = torch.random(3, 15)
   local f = torch.random(3, 15)
   local h = torch.random(3, 15)
   local w = torch.random(3, 15)
   local scale = torch.random(2,5)

   local tm = {}
   local title = string.format('SpatialUpSamplingNearest.forward %dx%dx%dx%d -> %dx%dx%dx%d',
                               nbatch, f, h, w, nbatch, f, h*scale, w*scale)
   times[title] = tm

   local input = torch.randn(nbatch, f, h, w)
   local sconv = nn.SpatialUpSamplingNearest(scale)
   local groundtruth = sconv:forward(input)
   local a = torch.Timer()
   for i = 1,nloop do
      groundtruth = sconv:forward(input)
   end
   tm.cpu = a:time().real

   input = input:cuda()
   local gconv = sconv:clone():cuda()
   local rescuda = gconv:forward(input)
   a:reset()
   for i = 1,nloop do
      rescuda = gconv:forward(input)
   end
   cutorch.synchronize()
   tm.gpu = a:time().real

   local error = rescuda:float() - groundtruth
   mytester:assertlt(error:abs():max(), precision_forward, 'error on state (forward) ')

end

function cunntest.SpatialUpSamplingNearest_backward()
   local f = torch.random(3, 15)
   local h = torch.random(3, 15)
   local w = torch.random(3, 15)
   local scale = torch.random(2,5)

   local tm = {}
   local title = string.format('SpatialUpSamplingNearest.backward %dx%dx%d -> %dx%dx%d',
                               f, h, w, f, h*scale, w*scale)
   times[title] = tm

   local input = torch.randn(f, h, w)
   local gradOutput = torch.randn(f, h*scale, w*scale)
   local sconv = nn.SpatialUpSamplingNearest(scale)
   sconv:forward(input)
   sconv:zeroGradParameters()
   local groundgrad = sconv:backward(input, gradOutput)
   local a = torch.Timer()
   for i = 1,nloop do
      sconv:zeroGradParameters()
      groundgrad = sconv:backward(input, gradOutput)
   end
   tm.cpu = a:time().real

   input = input:cuda()
   gradOutput = gradOutput:cuda()
   local gconv = sconv:clone():cuda()
   gconv:forward(input)
   gconv:zeroGradParameters()
   local rescuda = gconv:backward(input, gradOutput)
   a:reset()
   for i = 1,nloop do
      gconv:zeroGradParameters()
      rescuda = gconv:backward(input, gradOutput)
   end
   cutorch.synchronize()
   tm.gpu = a:time().real

   local error = rescuda:float() - groundgrad

   mytester:assertlt(error:abs():max(), precision_backward, 'error on state (backward) ')
end

function cunntest.SpatialUpSamplingNearest_backward_batch()
   local nbatch = torch.random(3, 15)
   local f = torch.random(3, 15)
   local h = torch.random(3, 15)
   local w = torch.random(3, 15)
   local scale = torch.random(2,5)

   local tm = {}
   local title = string.format('SpatialUpSamplingNearest.backward %dx%dx%dx%d -> %dx%dx%dx%d',
                               nbatch, f, h, w, nbatch, f, h*scale, w*scale)
   times[title] = tm

   local input = torch.randn(nbatch, f, h, w)
   local gradOutput = torch.randn(nbatch, f, h*scale, w*scale)
   local sconv = nn.SpatialUpSamplingNearest(scale)
   sconv:forward(input)
   sconv:zeroGradParameters()
   local groundgrad = sconv:backward(input, gradOutput)
   local a = torch.Timer()
   for i = 1,nloop do
      sconv:zeroGradParameters()
      groundgrad = sconv:backward(input, gradOutput)
   end
   tm.cpu = a:time().real

   input = input:cuda()
   gradOutput = gradOutput:cuda()
   local gconv = sconv:clone():cuda()
   gconv:forward(input)
   gconv:zeroGradParameters()
   local rescuda = gconv:backward(input, gradOutput)
   a:reset()
   for i = 1,nloop do
      gconv:zeroGradParameters()
      rescuda = gconv:backward(input, gradOutput)
   end
   cutorch.synchronize()
   tm.gpu = a:time().real

   local error = rescuda:float() - groundgrad

   mytester:assertlt(error:abs():max(), precision_backward, 'error on state (backward) ')
end

function cunntest.l1cost()
   local size = math.random(300,500)
   local input = torch.randn(size)
   local mod = nn.L1Cost()

   local tm = {}
   local title = string.format('L1Cost %d ',size)
   times[title] = tm

   local a = torch.Timer()
   local fout = mod:forward(input)
   local fgin = mod:backward(input):clone()
   tm.cpu = a:time().real

   local cinput = input:cuda()
   local cmod = nn.L1Cost():cuda()
   a:reset()
   local cout = cmod:forward(cinput)
   local cgin = cmod:backward(cinput)
   cutorch.synchronize()
   tm.gpu = a:time().real

   mytester:assertlt(math.abs(fout-cout), precision_forward, 'error  on output')
   local gerr = cgin:float() - fgin
   mytester:assertlt(gerr:abs():max(), precision_forward, 'error  on gradInput')
end


function cunntest.ClassNLLCriterionSingleTarget()
   local size = math.random(3000,5000)
   local input = torch.randn(size)
   local target = 1
   local mod = nn.ClassNLLCriterion()

   local tm = {}
   local title = string.format('ClassNLLCriterionSingleTarget %d ',size)
   times[title] = tm

   local a = torch.Timer()
   local fout = mod:forward(input, target)
   local fgin = mod:backward(input, target):clone()
   tm.cpu = a:time().real

   local cinput = input:cuda()
   local ctarget = torch.CudaTensor(1):fill(target)
   local cmod = nn.ClassNLLCriterion():cuda()
   a:reset()
   local cout = cmod:forward(cinput,ctarget)
   local cgin = cmod:backward(cinput,ctarget)
   cutorch.synchronize()
   tm.gpu = a:time().real

   mytester:assertlt(
       math.abs(fout-cout), precision_forward, 'error  on output')
   local gerr = cgin:float() - fgin
   mytester:assertlt(gerr:abs():max(), precision_forward, 'error  on gradInput')
end

function cunntest.ClassNLLCriterionSingleTargetWeights()
   local size = math.random(3000,5000)
   local input = torch.randn(size)
   local target = 1
   local weights = torch.rand(size)
   local mod = nn.ClassNLLCriterion(weights)

   local tm = {}
   local title = string.format('ClassNLLCriterionSingleTargetWeights %d ',size)
   times[title] = tm

   local a = torch.Timer()
   local fout = mod:forward(input, target)
   local fgin = mod:backward(input, target):clone()
   tm.cpu = a:time().real

   local cinput = input:cuda()
   local cweights = weights:cuda()
   local ctarget = torch.CudaTensor(1):fill(target)
   local cmod = nn.ClassNLLCriterion(cweights):cuda()
   a:reset()
   local cout = cmod:forward(cinput,ctarget)
   local cgin = cmod:backward(cinput,ctarget)
   cutorch.synchronize()
   tm.gpu = a:time().real

   mytester:assertlt(
       math.abs(fout-cout), precision_forward, 'error  on output')
   local gerr = cgin:float() - fgin
   mytester:assertlt(gerr:abs():max(), precision_forward, 'error  on gradInput')
end

function cunntest.ClassNLLCriterionMultipleTarget()
   local size = math.random(3000,5000)
   local input = torch.randn(size, size)
   local target = torch.randperm(size)
   local mod = nn.ClassNLLCriterion()

   local tm = {}
   local title = string.format('ClassNLLCriterionMultiTarget %d ',size)
   times[title] = tm

   local a = torch.Timer()
   local fout = mod:forward(input, target)
   local fgin = mod:backward(input, target):clone()
   tm.cpu = a:time().real

   local cinput = input:cuda()
   local ctarget = target:cuda()

   local cmod = nn.ClassNLLCriterion():cuda()
   a:reset()
   local cout = cmod:forward(cinput,ctarget)
   local cgin = cmod:backward(cinput,ctarget)
   cutorch.synchronize()
   tm.gpu = a:time().real

   mytester:assertlt(
       math.abs(fout-cout), precision_forward, 'error on output')

   local gerr = cgin:float() - fgin
   mytester:assertlt(gerr:abs():max(), precision_forward, 'error  on gradInput')
end


function cunntest.ClassNLLCriterionMultipleTargetWeights()
   local size = math.random(3000,5000)
   local input = torch.randn(size, size)
   local target = torch.randperm(size)
   local weights = torch.rand(size)
   local mod = nn.ClassNLLCriterion(weights)

   local tm = {}
   local title = string.format('ClassNLLCriterionMultiTargetWeights %d ',size)
   times[title] = tm

   local a = torch.Timer()
   local fout = mod:forward(input, target)
   local fgin = mod:backward(input, target):clone()
   tm.cpu = a:time().real

   local cinput = input:cuda()
   local ctarget = target:cuda()
   local cweights = weights:cuda()

   local cmod = nn.ClassNLLCriterion(cweights):cuda()
   a:reset()
   local cout = cmod:forward(cinput,ctarget)
   local cgin = cmod:backward(cinput,ctarget)
   cutorch.synchronize()
   tm.gpu = a:time().real

   mytester:assertlt(
       math.abs(fout-cout), precision_forward, 'error on output')

   local gerr = cgin:float() - fgin
   mytester:assertlt(gerr:abs():max(), precision_forward, 'error  on gradInput')
end

function cunntest.TemporalMaxPooling()
   local input = torch.rand(16, 18, 3)
   local settings = {{2, 2}, {3, 3}, {4, 2}, {2, 4}, {3, 5}}

   for i, setting in ipairs(settings) do
      local mod = nn.TemporalMaxPooling(setting[1], setting[2])

      local tm = {}
      local title = 'TemporalMaxPooling '..setting[1]..' '..setting[2]
      times[title] = tm

      local a = torch.Timer()
      local fout = mod:forward(input)
      local fgout = torch.rand(fout:size())
      local fgin = mod:backward(input, fgout):clone()
      tm.cpu = a:time().real

      local cinput = input:cuda()
      local cgout = fgout:cuda()
      local cmod = nn.TemporalMaxPooling(setting[1], setting[2]):cuda()
      a:reset()
      local cout = cmod:forward(cinput)
      local cgin = cmod:backward(cinput, cgout)
      cutorch.synchronize()
      tm.gpu = a:time().real

      local outerror = cout:float() - fout
      mytester:assertlt(outerror:abs():max(), precision_forward, 'error on output')

      local ginerror = cgin:float() - fgin
      mytester:assertlt(ginerror:abs():max(), precision_backward, 'error on gradInput')
   end
end

function cunntest.VolumetricConvolution_forward_single()
   local from = math.random(1,32)
   local to = math.random(1,8) * 8
   local ki = math.random(3,15)
   local kj = math.random(3,15)
   local kk = math.random(3,15)
   local si = math.random(1,ki)
   local sj = math.random(1,kj)
   local sk = math.random(1,kk)
   local outi = math.random(1,64)
   local outj = math.random(1,64)
   local outk = math.random(1,64)
   local ini = (outi-1)*si+ki
   local inj = (outj-1)*sj+kj
   local ink = (outk-1)*sk+kk

   local tm = {}
   local title = string.format('VolumetricConvolution.forward %dx%dx%dx%d o %dx%dx%d -> %dx%dx%dx%d',
                           from, ink, inj, ini, kk, kj, ki, to, outk, outj, outi)
   times[title] = tm

   local input = torch.randn(from,ini,inj,ink)
   local sconv = nn.VolumetricConvolution(from,to,ki,kk,kj,si,sk,sj)
   local groundtruth = sconv:forward(input)
   local a = torch.Timer()
   for i = 1,nloop do
      groundtruth = sconv:forward(input)
   end
   tm.cpu = a:time().real

   input = input:cuda()
   local gconv = nn.VolumetricConvolution(from,to,ki,kk,kj,si,sk,sj):cuda()
   gconv.weight = sconv.weight:cuda()
   gconv.bias = sconv.bias:cuda()
   local rescuda = gconv:forward(input)
   a:reset()
   for i = 1,nloop do
      rescuda = gconv:forward(input)
   end
   cutorch.synchronize()
   tm.gpu = a:time().real

   local error = rescuda:float() - groundtruth
   mytester:assertlt(error:abs():max(), precision_forward, 'error on state (forward) ')
   mytester:assert(groundtruth:isSize(rescuda:size()), 'size mismatch on state (forward)')
end

function cunntest.VolumetricConvolution_forward_batch()
   local bs = math.random(1,4) * 4
   local from = math.random(1,8)
   local to = math.random(1,4) * 4
   local ki = math.random(3,8)
   local kj = math.random(3,8)
   local kk = math.random(3,8)
   local si = math.random(1,ki)
   local sj = math.random(1,kj)
   local sk = math.random(1,kk)
   local outi = math.random(1,16)
   local outj = math.random(1,16)
   local outk = math.random(1,16)
   local ini = (outi-1)*si+ki
   local inj = (outj-1)*sj+kj
   local ink = (outk-1)*sk+kk

   local tm = {}
   local title = string.format('VolumetricConvolution.forward %dx%dx%dx%dx%d o %dx%dx%d -> %dx%dx%dx%dx%d',
                           bs, from, ink, inj, ini, kk, kj, ki, bs, to, outk, outj, outi)
   times[title] = tm

   local input = torch.randn(bs,from,ini,inj, ink)
   local sconv = nn.VolumetricConvolution(from,to,ki,kk,kj,si,sj,sk)
   local groundtruth = sconv:forward(input)
   local a = torch.Timer()
   for i = 1,nloop do
      groundtruth = sconv:forward(input, sconv)
   end
   tm.cpu = a:time().real

   input = input:cuda()
   local gconv = nn.VolumetricConvolution(from,to,ki,kk,kj,si,sj,sk):cuda()
   gconv.weight = sconv.weight:cuda()
   gconv.bias = sconv.bias:cuda()
   local rescuda = gconv:forward(input)
   a:reset()
   for i = 1,nloop do
      rescuda = gconv:forward(input)
   end
   cutorch.synchronize()
   tm.gpu = a:time().real

   local error = rescuda:float() - groundtruth
   mytester:assertlt(error:abs():max(), precision_forward, 'error on state (forward) ')
   mytester:assert(groundtruth:isSize(rescuda:size()), 'size mismatch on state (forward)')
end

function cunntest.VolumetricConvolution_backward_single()
   local from = math.random(1,4)
   local to = math.random(1,3) * 8
   local ki = math.random(3,8)
   local kj = math.random(3,8)
   local kk = math.random(3,8)
   local si = math.random(1,ki)
   local sj = math.random(1,kj)
   local sk = math.random(1,kk)
   local outi = math.random(1,16)
   local outj = math.random(1,16)
   local outk = math.random(1,16)
   local ini = (outi-1)*si+ki
   local inj = (outj-1)*sj+kj
   local ink = (outk-1)*sk+kk

   local tm = {}
   local title = string.format('VolumetricConvolution.backward %dx%dx%dx%d o %dx%dx%d -> %dx%dx%dx%d',
                               from, ink, inj, ini, kk, kj, ki, to, outk, outj, outi)
   times[title] = tm

   local input = torch.randn(from, ini, inj, ink)
   local gradOutput = torch.randn(to, outi, outj, outk)
   local sconv = nn.VolumetricConvolution(from,to,ki,kk,kj,si,sk,sj)
   sconv:forward(input)
   sconv:zeroGradParameters()
   local groundgrad = sconv:backward(input, gradOutput)
   local a = torch.Timer()
   for i = 1,nloop do
      sconv:zeroGradParameters()
      groundgrad = sconv:backward(input, gradOutput)
   end
   local groundweight = sconv.gradWeight
   local groundbias = sconv.gradBias
   tm.cpu = a:time().real

   input = input:cuda()
   gradOutput = gradOutput:cuda()
   local gconv = nn.VolumetricConvolution(from,to,ki,kk,kj,si,sk,sj):cuda()
   gconv.weight = sconv.weight:cuda()
   gconv.bias = sconv.bias:cuda()
   gconv:forward(input)
   gconv:zeroGradParameters()
   local rescuda = gconv:backward(input, gradOutput)
   a:reset()
   for i = 1,nloop do
      gconv:zeroGradParameters()
      rescuda = gconv:backward(input, gradOutput)
   end
   local weightcuda = gconv.gradWeight
   local biascuda = gconv.gradBias
   cutorch.synchronize()
   tm.gpu = a:time().real
   local error = rescuda:float() - groundgrad
   local werror = weightcuda:float() - groundweight
   local berror = biascuda:float() - groundbias
   mytester:assert(groundgrad:isSize(rescuda:size()), 'size mismatch on state (forward)')
   mytester:assertlt(error:abs():max(), precision_backward, 'error on state (backward) ')
   mytester:assertlt(werror:abs():max(), precision_backward, 'error on weight (backward) ')
   mytester:assertlt(berror:abs():max(), precision_backward, 'error on bias (backward) ')
end

function cunntest.VolumetricConvolution_backward_batch()
   local bs = math.random(1,4) * 4
   local from = math.random(1,4)
   local to = math.random(1,3) * 8
   local ki = math.random(3,8)
   local kj = math.random(3,8)
   local kk = math.random(3,8)
   local si = math.random(1,ki)
   local sj = math.random(1,kj)
   local sk = math.random(1,kk)
   local outi = math.random(1,16)
   local outj = math.random(1,16)
   local outk = math.random(1,16)
   local ini = (outi-1)*si+ki
   local inj = (outj-1)*sj+kj
   local ink = (outk-1)*sk+kk

   local tm = {}
   local title = string.format('VolumetricConvolution.backward %dx%dx%dx%dx%d o %dx%dx%d -> %dx%dx%dx%dx%d',
                           bs, from, ink, inj, ini, kk, kj, ki, bs, to, outk, outj, outi)
   times[title] = tm

   local input = torch.randn(bs, from, ini, inj, ink)
   local gradOutput = torch.randn(bs, to, outi, outj, outk)
   local sconv = nn.VolumetricConvolution(from,to,ki,kk,kj,si,sk,sj)
   sconv:forward(input)
   sconv:zeroGradParameters()
   local groundgrad = sconv:backward(input, gradOutput)
   local a = torch.Timer()
   for i = 1,nloop do
      sconv:zeroGradParameters()
      groundgrad = sconv:backward(input, gradOutput)
   end
   local groundweight = sconv.gradWeight
   local groundbias = sconv.gradBias
   tm.cpu = a:time().real

   input = input:cuda()
   gradOutput = gradOutput:cuda()
   local gconv = nn.VolumetricConvolution(from,to,ki,kk,kj,si,sk,sj):cuda()
   gconv.weight = sconv.weight:cuda()
   gconv.bias = sconv.bias:cuda()
   gconv:forward(input)
   gconv:zeroGradParameters()
   local rescuda = gconv:backward(input, gradOutput)
   a:reset()
   for i = 1,nloop do
      gconv:zeroGradParameters()
      rescuda = gconv:backward(input, gradOutput)
   end
   local weightcuda = gconv.gradWeight
   local biascuda = gconv.gradBias
   cutorch.synchronize()
   tm.gpu = a:time().real
   local error = rescuda:float() - groundgrad
   local werror = weightcuda:float() - groundweight
   local berror = biascuda:float() - groundbias
   mytester:assert(groundgrad:isSize(rescuda:size()), 'size mismatch on state (forward)')
   mytester:assertlt(error:abs():max(), precision_backward, 'error on state (backward) ')
   mytester:assertlt(werror:abs():max(), precision_backward, 'error on weight (backward) ')
   mytester:assertlt(berror:abs():max(), precision_backward, 'error on bias (backward) ')
end

function cunntest.VolumetricMaxPooling_forward()
   local kT = math.random(3, 7)
   local kH = math.random(3, 7)
   local kW = math.random(3, 7)
   local dT = math.random(1, 13)
   local dH = math.random(1, 13)
   local dW = math.random(1, 13)
   local oT = math.random(1, 64)
   local oH = math.random(1, 64)
   local oW = math.random(1, 64)
   local iF = math.random(1, 16) -- features
   local iT = (oT - 1) * dT + kT
   local iH = (oH - 1) * dH + kH
   local iW = (oW - 1) * dW + kW

   local tm = {}
   local title = string.format('VolumetricMaxPooling.forward %dx%dx%dx%d o %dx%dx%d -> %dx%dx%dx%d',
                           iF, iT, iH, iW, kT, kH, kW, iF, oT, oH, oW)
   times[title] = tm

   local input = torch.Tensor(iF, iT, iH, iW):float():uniform(-1, 1)
   local layer = nn.VolumetricMaxPooling(kT, kH, kW, dT, dH, dW):float()
   local output = layer:forward(input)
   local timer = torch.Timer()
   for i = 1,nloop do
      output = layer:forward(input)
   end
   tm.cpu = timer:time().real

   local inputCUDA = input:cuda()
   local layerCUDA = layer:clone():cuda()
   local outputCUDA = layerCUDA:forward(inputCUDA)
   timer:reset()
   for i = 1,nloop do
      outputCUDA = layerCUDA:forward(inputCUDA)
   end
   cutorch.synchronize()
   tm.gpu = timer:time().real

   local error = outputCUDA:float() - output
   mytester:assertlt(error:abs():max(), precision_forward, 'error on state (forward) ')
end

function cunntest.VolumetricMaxPooling_backward()
   local kT = math.random(3, 7)
   local kH = math.random(3, 7)
   local kW = math.random(3, 7)
   local dT = math.random(1, 13)
   local dH = math.random(1, 13)
   local dW = math.random(1, 13)
   local oT = math.random(1, 64)
   local oH = math.random(1, 64)
   local oW = math.random(1, 64)
   local iF = math.random(1, 16) -- features
   local iT = (oT - 1) * dT + kT
   local iH = (oH - 1) * dH + kH
   local iW = (oW - 1) * dW + kW

   local tm = {}
   local title = string.format('VolumetricMaxPooling.backward %dx%dx%dx%d o %dx%dx%d -> %dx%dx%dx%d',
                               iF, iT, iH, iW, kT, kH, kW, iF, oT, oH, oW)
   times[title] = tm

   local input = torch.Tensor(iF, iT, iH, iW):float():uniform(-1, 1)
   local layer = nn.VolumetricMaxPooling(kT, kH, kW, dT, dH, dW):float()
   local output = layer:forward(input)
   local gradOutput = output:clone():uniform(-1, 1)

   local gradInput = layer:backward(input, gradOutput)
   local timer = torch.Timer()
   for i = 1,nloop do
      gradInput = layer:backward(input, gradOutput)
   end
   tm.cpu = timer:time().real

   local inputCUDA = input:cuda()
   local layerCUDA = layer:clone():cuda()
   local outputCUDA = layerCUDA:forward(inputCUDA)
   local gradOutputCUDA = gradOutput:cuda()
   local gradInputCUDA = layerCUDA:backward(inputCUDA, gradOutputCUDA)

   timer:reset()
   for i = 1,nloop do
      gradInputCUDA = layerCUDA:backward(inputCUDA, gradOutputCUDA)
   end
   cutorch.synchronize()
   tm.gpu = timer:time().real

   local error = gradInputCUDA:float() - gradInput
   mytester:assertlt(error:abs():max(), precision_forward, 'error on state (backward) ')
end

function cunntest.VolumetricAveragePooling_forward()
   local kT = math.random(3, 7)
   local kH = math.random(3, 7)
   local kW = math.random(3, 7)
   local dT = math.random(1, 13)
   local dH = math.random(1, 13)
   local dW = math.random(1, 13)
   local oT = math.random(1, 64)
   local oH = math.random(1, 64)
   local oW = math.random(1, 64)
   local iF = math.random(1, 16) -- features
   local iT = (oT - 1) * dT + kT
   local iH = (oH - 1) * dH + kH
   local iW = (oW - 1) * dW + kW

   local tm = {}
   local title = string.format('VolumetricAveragePooling.forward %dx%dx%dx%d o %dx%dx%d -> %dx%dx%dx%d',
                               iF, iT, iH, iW, kT, kH, kW, iF, oT, oH, oW)
   times[title] = tm

   local input = torch.Tensor(iF, iT, iH, iW):float():uniform(-1, 1)
   local layer = nn.VolumetricAveragePooling(kT, kW, kH, dT, dW, dH):float()
   local output = layer:forward(input)
   local timer = torch.Timer()
   for i = 1,nloop do
      output = layer:forward(input)
   end
   tm.cpu = timer:time().real

   local inputCUDA = input:cuda()
   local layerCUDA = layer:clone():cuda()
   local outputCUDA = layerCUDA:forward(inputCUDA)
   timer:reset()
   for i = 1,nloop do
      outputCUDA = layerCUDA:forward(inputCUDA)
   end
   cutorch.synchronize()
   tm.gpu = timer:time().real

   local error = outputCUDA:float() - output
   mytester:assertlt(error:abs():max(), precision_forward, 'error on state (forward) ')
end

function cunntest.VolumetricAveragePooling_backward()
   local kT = math.random(3, 7)
   local kH = math.random(3, 7)
   local kW = math.random(3, 7)
   local dT = math.random(1, 13)
   local dH = math.random(1, 13)
   local dW = math.random(1, 13)
   local oT = math.random(1, 64)
   local oH = math.random(1, 64)
   local oW = math.random(1, 64)
   local iF = math.random(1, 16) -- features
   local iT = (oT - 1) * dT + kT
   local iH = (oH - 1) * dH + kH
   local iW = (oW - 1) * dW + kW

   local tm = {}
   local title = string.format('VolumetricAveragePooling.backward %dx%dx%dx%d o %dx%dx%d -> %dx%dx%dx%d',
                           iF, iT, iH, iW, kT, kH, kW, iF, oT, oH, oW)
   times[title] = tm

   local input = torch.Tensor(iF, iT, iH, iW):float():uniform(-1, 1)
   local layer = nn.VolumetricAveragePooling(kT, kW, kH, dT, dW, dH):float()
   local output = layer:forward(input)
   local gradOutput = output:clone():uniform(-1, 1)

   local gradInput = layer:backward(input, gradOutput)
   local timer = torch.Timer()
   for i = 1,nloop do
      gradInput = layer:backward(input, gradOutput)
   end
   tm.cpu = timer:time().real

   local inputCUDA = input:cuda()  local layerCUDA = layer:clone():cuda()
   local outputCUDA = layerCUDA:forward(inputCUDA)   local gradOutputCUDA = gradOutput:cuda()
   local gradInputCUDA = layerCUDA:backward(inputCUDA, gradOutputCUDA)

   timer:reset()
   for i = 1,nloop do
      gradInputCUDA = layerCUDA:backward(inputCUDA, gradOutputCUDA)
   end
   cutorch.synchronize()
   tm.gpu = timer:time().real

   local error = gradInputCUDA:float() - gradInput
   mytester:assertlt(error:abs():max(), precision_forward, 'error on state (backward) ')
end

function cunntest.CMul_forward_batch()
   local bs = math.random(8,32)
   local nini = math.random(1,100)
   local ninj = math.random(1,100)
   local nink = math.random(1,100)

   local tm = {}
   local title = string.format('CMul forward %d %d %d %d', bs, nini, ninj, nink)
   times[title] = tm

   local input = torch.randn(bs, nini, ninj, nink)
   local sconv = nn.CMul(nini, ninj, nink)
   local groundtruth = sconv:forward(input)
   local a = torch.Timer()
   for i = 1,nloop do
      groundtruth = sconv:forward(input)
   end
   tm.cpu = a:time().real

   input = input:cuda()
   local gconv = sconv:clone():cuda()
   local rescuda = gconv:forward(input)
   a:reset()
   for i = 1,nloop do
      rescuda = gconv:forward(input)
   end
   cutorch.synchronize()
   tm.gpu = a:time().real

   local error = rescuda:float() - groundtruth
   mytester:assertlt(error:abs():max(), precision_forward, 'error on state (forward) batch ')
end

function cunntest.CMul_backward_batch()
   local bs = math.random(8,32)
   local nini = math.random(1,100)
   local ninj = math.random(1,100)
   local nink = math.random(1,100)

   local tm = {}
   local title = string.format('CMul backward %d %d %d %d', bs, nini, ninj, nink)
   times[title] = tm

   local input = torch.randn(bs, nini, ninj, nink)
   local gradOutput = torch.randn(bs, nini, ninj, nink)
   local sconv = nn.CMul(nini, ninj, nink)
   sconv:forward(input)
   sconv:zeroGradParameters()
   local groundgrad = sconv:backward(input, gradOutput)
   local a = torch.Timer()
   for i = 1,nloop do
      sconv:zeroGradParameters()
      groundgrad = sconv:backward(input, gradOutput)
   end
   local groundweight = sconv.gradWeight
   tm.cpu = a:time().real

   input = input:cuda()
   gradOutput = gradOutput:cuda()
   local gconv = sconv:clone():cuda()
   gconv:forward(input)
   gconv:zeroGradParameters()
   local rescuda = gconv:backward(input, gradOutput)
   a:reset()
   for i = 1,nloop do
      gconv:zeroGradParameters()
      rescuda = gconv:backward(input, gradOutput)
   end
   cutorch.synchronize()
   tm.gpu = a:time().real

   local weightcuda = gconv.gradWeight

   local error = rescuda:float() - groundgrad
   local werror = weightcuda:float() - groundweight

   mytester:assertlt(error:abs():max(), precision_backward, 'error on state (backward) ')
   mytester:assertlt(werror:abs():max(), precision_backward, 'error on weight (backward) ')
end

function cunntest.PReLU_forward()
    local nOutputPlane = 8
    local w = math.random(1,100)
    local h = math.random(1,100)

    local tm = {}
    local title = string.format('PReLU forward %d x %d', w, h)
    times[title] = tm

    local input = torch.randn(nOutputPlane,h,w)
    local sconv = nn.PReLU(nOutputPlane)
    local groundtruth = sconv:forward(input)
    local a = torch.Timer()
    for i = 1,nloop do
        groundtruth = sconv:forward(input)
    end
    tm.cpu = a:time().real

    input = input:cuda()
    local gconv = sconv:cuda()
    local rescuda = gconv:forward(input)
    a:reset()
    for i = 1,nloop do
        rescuda = gconv:forward(input)
    end
    cutorch.synchronize()
    tm.gpu = a:time().real

    local error = rescuda:float() - groundtruth
    mytester:assertlt(error:abs():max(), precision_forward, 'error on state')
end

function cunntest.PReLU_backward()
    local nOutputPlane = 8
    local w = math.random(1,10)
    local h = math.random(1,10)

    local tm = {}
    local title = string.format('PReLU backward %d x %d', w, h)
    times[title] = tm

    local input = torch.randn(nOutputPlane, h, w)
    local gradOutput = torch.randn(#input)
    local sconv = nn.PReLU(nOutputPlane)
    local gconv = sconv:clone():cuda()

    sconv:forward(input)
    local groundgrad = sconv:backward(input, gradOutput)
    local a = torch.Timer()
    for i = 1,nloop do
        groundgrad = sconv:backward(input, gradOutput)
    end
    tm.cpu = a:time().real

    input = input:cuda()
    gradOutput = gradOutput:cuda()
    gconv:forward(input)
    local rescuda = gconv:backward(input, gradOutput)
    a:reset()
    for i = 1,nloop do
        rescuda = gconv:backward(input, gradOutput)
    end
    cutorch.synchronize()
    tm.gpu = a:time().real

    local err = rescuda:float() - groundgrad
    local weightGradError = gconv.gradWeight:float() - sconv.gradWeight

    mytester:assertlt(err:abs():max(), precision_backward, 'error on state')
    mytester:assertlt(weightGradError:abs():max(), precision_backward, 'error on weight')
end


function cunntest.RReLU_forward()
    local nOutputPlane = 8
    local w = math.random(1,100)
    local h = math.random(1,100)

    for _,train in ipairs({true,false}) do
       for _,inplace in ipairs({false,true}) do
          local tm = {}
          local title = string.format('RReLU forward %d x %d (inplace: %s, train: %s)', 
             w, h, tostring(inplace), tostring(train))
          times[title] = tm
          
          local input = torch.randn(nOutputPlane, h, w) - 0.5
          local sconv = nn.RReLU(1/8, 1/3, inplace)
          if not train then
             sconv:evaluate()
          end
          local groundtruth = sconv:forward(input:clone())
          local a = torch.Timer()
          for i = 1,nloop do
             groundtruth = sconv:forward(input:clone())
          end
          tm.cpu = a:time().real
          
          input = input:cuda()
          local gconv = sconv:cuda()
          local rescuda = gconv:forward(input:clone())
          a:reset()
          for i = 1,nloop do
             rescuda = gconv:forward(input:clone())
          end
          cutorch.synchronize()
          tm.gpu = a:time().real
          
          if not train then
             local error = rescuda:float() - groundtruth
             mytester:assertlt(error:abs():max(), precision_forward, 'error on state')
          end
       end
    end
end

function cunntest.RReLU_backward()
    local nOutputPlane = 8
    local w = math.random(1,10)
    local h = math.random(1,10)

    for _,train in ipairs({true,false}) do
       for _,inplace in ipairs({false,true}) do
          local tm = {}
          local title = string.format('RReLU backward %d x %d (inplace: %s, train: %s)', 
            w, h, tostring(inplace), tostring(train))
          times[title] = tm

          local input = torch.randn(nOutputPlane, h, w)
          local gradOutput = torch.randn(#input) - 0.5
          local sconv = nn.RReLU(1/8, 1/3, inplace)
          if not train then
             sconv:evaluate()
          end

          sconv:forward(input:clone())
          local groundgrad = sconv:backward(input, gradOutput:clone())
          local a = torch.Timer()
          for i = 1,nloop do
             groundgrad = sconv:backward(input, gradOutput:clone())
          end
          tm.cpu = a:time().real

          local gconv = sconv:clone():cuda()
          input = input:cuda()
          gradOutput = gradOutput:cuda()
          gconv:forward(input:clone())
          local rescuda = gconv:backward(input, gradOutput:clone())
          a:reset()
          for i = 1,nloop do
             rescuda = gconv:backward(input, gradOutput:clone())
          end
          cutorch.synchronize()
          tm.gpu = a:time().real

          if not train then
             local err = rescuda:float() - groundgrad
             mytester:assertlt(err:abs():max(), precision_backward, 'error on state')
          end
          
          input = -torch.rand(1000):cuda()
          gconv:forward(input) -- fill internal noise tensor
          local g = gconv:backward(input, torch.ones(1000):cuda())
          local err = math.abs(g[input:le(0)]:mean()-(gconv.lower+gconv.upper)/2)
          mytester:assertlt(err, 0.05, 'mean deviation of gradient for negative inputs')
       end
    end
end

function cunntest.VolumetricDeconvolution_pair_test()

    local kT = 2 * math.random(1,3) + 1  -- odd number
    local kH = 2 * math.random(1,3) + 1  -- odd number
    local kW = kH
    local dT = math.random(1,3)
    local dH = math.random(1,3)
    local dW = dH
    local pT = (kT-1)/2
    local pH = (kH-1)/2
    local pW = pH

    local inChan = math.random(1,32)
    local outChan = math.random(1,32)

    local module = nn.VolumetricDeconvolution(inChan, outChan, kT, kH, kW,
                                          dT, dH, dW, pT, pH, pW);
    module.weight:fill(1);
    module.bias:fill(0.1);

    local bs = math.random(8,32)
    local inD = math.random(8,32)
    local inH = math.random(8,32)
    local inW = math.random(8,32)
    local outD = (inD - 1) * dT - 2 * pT + kT
    local outH = (inH - 1) * dH - 2 * pH + kH
    local outW = (inW - 1) * dW - 2 * pW + kW
    local input = torch.Tensor(bs, inChan, inD, inH, inW):fill(1);
    local gradOut = torch.randn(bs, outChan, outD, outH, outW)

    local outcpu = module:forward(input)
    local gradcpu = module:backward(input, gradOut)
    module:cuda()
    local outgpu = module:forward(input:cuda())
    local gradgpu = module:backward(input:cuda(), gradOut:cuda())

    local error = outgpu:float() - outcpu
    mytester:assertlt(error:abs():max(), precision_forward,
                      'error on state (forward) ')

    local error = gradgpu:float() - gradcpu
    mytester:assertlt(error:abs():max(), precision_backward,
                      'error on state (backward) ')
end

function cunntest.VolumetricDeconvolution()
    local module = nn.VolumetricDeconvolution(3, 1, 3, 3, 3, 3, 3, 3);
    module.weight:fill(1);
    module.bias:fill(0.1);
    module:cuda();

    local input = torch.Tensor(1, 3, 2, 2, 2):zero();
    for c = 1,3 do
        input[1][c][1][1][1] = 1
    end
    local output = module:forward(input:cuda())
    for t = 1,6 do
        for h = 1,6 do
            for w = 1,6 do
                if t <= 3 and h <= 3 and w <= 3 then
                    mytester:assertlt(output[1][1][t][h][w] - 3.1, precision_forward, 'error on forward ')
                else
                    mytester:assertlt(output[1][1][t][h][w] - 0.1, precision_forward, 'error on forward ')
                end
            end
        end
    end

    local gradOut = torch.Tensor(1, 1, 6, 6, 6):fill(0.1);
    local gradIn = module:backward(input:cuda(), gradOut:cuda())
    for t = 1,2 do
        for h = 1,2 do
            for w = 1,2 do
                mytester:assertlt(gradIn[1][1][t][h][w] - 2.7, precision_backward,
                                  'error on backward input gradients ')
            end
        end
    end

    mytester:assertlt(module.gradBias[1] - 21.6, precision_backward,
                      'error on backward gradBias ')
    for c = 1,3 do
        for t = 1,3 do
            for h = 1,3 do
                for w = 1,3 do
                    mytester:assertlt(module.gradWeight[1][c][t][h][w] - 0.1, precision_backward,
                                      'error on backward weight gradients ')
                end
            end
        end
    end
end

function cunntest.getParameters()
   -- smoke test for getParameters
   local model = nn.Sequential()
   model:add( nn.SpatialConvolution(1,10,3,3) )
   model:add( nn.SpatialConvolution(10,20,3,3) )
   model:cuda()
   local weights, gradients = model:getParameters()
   local copy = model:clone():float():cuda()
   local weights, gradients = copy:getParameters()
end

function cunntest.LookupTable_forward()
   local nVocab = 10000
   local nDim = 100
   local nInput = 1000

   local tm = {}
   local title = string.format('LookupTable forward %d x %d', nVocab, nDim)
   times[title] = tm

   local input = torch.LongTensor(nInput):random(nVocab)
   local sconv = nn.LookupTable(nVocab, nDim)
   local groundtruth = sconv:forward(input)
   local a = torch.Timer()
   for i = 1,nloop do
       groundtruth = sconv:forward(input)
   end
   tm.cpu = a:time().real

   input = input:cuda()
   local gconv = sconv:cuda()
   local rescuda = gconv:forward(input)
   a:reset()
   for i = 1,nloop do
       rescuda = gconv:forward(input)
   end
   cutorch.synchronize()
   tm.gpu = a:time().real

   local error = rescuda:float() - groundtruth
   mytester:assertlt(error:abs():max(), precision_forward, 'error on state')
end

function cunntest.LookupTable_backward()
   local nVocab = 10000

   for _,nDim in ipairs{97,255} do
      for _,nInput in ipairs{10,101,1000,10007} do
         for _,scaleGradByFreq in ipairs{false,true} do
            for _,batch in ipairs{false, true} do
               local input, gradOutput
               if batch then
                  input = torch.LongTensor(nInput, 5):random(nVocab)
                  gradOutput = torch.randn(nInput, 5, nDim)
               else
                  input = torch.LongTensor(nInput):random(nVocab)
                  gradOutput = torch.randn(nInput, nDim)
               end

               local sconv = nn.LookupTable(nVocab, nDim)
               local gconv = sconv:clone():cuda()
               if scaleGradByFreq then
                  sconv = sconv:scaleGradByFreq()
                  gconv = gconv:scaleGradByFreq()
               end

               sconv:forward(input)
               sconv:backward(input, gradOutput)

               input = input:cuda()
               gradOutput = gradOutput:cuda()
               gconv:forward(input)
               gconv:backward(input, gradOutput)

               local weightGradError = gconv.gradWeight:float() - sconv.gradWeight
               mytester:assertlt(weightGradError:abs():max(), precision_backward,
                  'error on weight for size ' .. tostring(nInput) .. ' scaleGradByFreq: ' .. tostring(scaleGradByFreq)
                  .. ' nDim ' .. tostring(nDim))
            end
         end
      end
   end

   local nDim = 128
   local nInput = 1000
   local tm = {}
   local title = string.format('LookupTable backward %d x %d', nVocab, nDim, nInput)
   times[title] = tm

   local input = torch.LongTensor(nInput):random(nVocab)
   local gradOutput = torch.randn(nInput, nDim)
   local sconv = nn.LookupTable(nVocab, nDim)
   local gconv = sconv:clone():cuda()

   sconv:forward(input)
   sconv:backward(input, gradOutput)
   local a = torch.Timer()
   for i = 1,nloop do
       sconv:backward(input, gradOutput)
   end
   tm.cpu = a:time().real

   input = input:cuda()
   gradOutput = gradOutput:cuda()
   gconv:forward(input)
   gconv:backward(input, gradOutput)
   a:reset()
   for i = 1,nloop do
       gconv:backward(input, gradOutput)
   end
   cutorch.synchronize()
   tm.gpu = a:time().real

   local weightGradError = gconv.gradWeight:float() - sconv.gradWeight
   mytester:assertlt(weightGradError:abs():max(), precision_backward, 'error on weight')
end

local function setUp()
   cutorch.setDevice(1)
end

for k,v in pairs(cunntest) do
   cunntest[k] = function()
      setUp()
      v()
   end
end

function initSeed(seed)
   seed = seed or os.time()
   -- ensure that you can reproduce a failing test
   print('seed: ', seed)
   math.randomseed(seed)
   torch.manualSeed(seed)
   cutorch.manualSeedAll(seed)
end

function nn.testcuda(tests, print_timing, n_loop, seed)
   nloop = n_loop or nloop
   local oldtype = torch.getdefaulttensortype()
   torch.setdefaulttensortype('torch.FloatTensor')
   initSeed(seed)
   mytester = torch.Tester()
   mytester:add(cunntest)
   mytester:run(tests)
   torch.setdefaulttensortype(oldtype)
   if print_timing then
       print ''
       print ' ------------------------------------------------------------------------------------------------'
       print '|  Module                                                                          |  Speedup    |'
       print ' ------------------------------------------------------------------------------------------------'
       for module,tm in pairs(times) do
           local str = string.format('| %-80s | %4.2f        |', module, (tm.cpu / (tm.gpu or 1e6)))
           print(str)
       end
       print ' ------------------------------------------------------------------------------------------------'
   end
end
