local cunntest = torch.TestSuite()
local ffi = require 'ffi'
local precision_forward = 1e-4
local precision_backward = 1e-2
local nloop = 1
local times = {}

--e.g.: th -lcunn -e "nn.testcuda{'Sigmoid_forward'}"

local typenames = {
  'torch.CudaTensor',
  'torch.CudaDoubleTensor',
}

local t2cpu = {
  ['torch.CudaTensor'] = 'torch.FloatTensor',
  ['torch.CudaDoubleTensor'] = 'torch.DoubleTensor',

}

local function checkHalf()
   if cutorch.hasHalf then
       table.insert(typenames, 'torch.CudaHalfTensor')
       t2cpu['torch.CudaHalfTensor'] = 'torch.FloatTensor'
   end
end

-- workarounds for non-existant functions
function torch.CudaHalfTensor:mean()
   return self:cuda():mean()
end

function torch.CudaDoubleTensor:mean()
   return self:cuda():mean()
end

-- half has additional error on top of double/float
local function precision_forward_type(precision_f, tensor_type)
   if (tensor_type == 'torch.CudaHalfTensor') then
      return 1e-2 + precision_f
   else
      return precision_f
   end
end

local function precision_backward_type(precision_b, tensor_type)
   if (tensor_type == 'torch.CudaHalfTensor') then
      return 1e-1 + precision_b
   else
      return precision_b
   end
end

local function precision_backward_conv_weightbias(precision_b, tensor_type)
   if (tensor_type == 'torch.CudaHalfTensor') then
      return 2 + precision_b -- cudnn uses 8 here
   else
      return precision_b
   end
end

local function pointwise_forward(proto_module, name, max_error)
   local size = math.random(1,100)

   for k, typename in ipairs(typenames) do
      local input = torch.randn(size):type(typename)
      local ctype = t2cpu[typename]
      local input = input:type(ctype)
      if name == 'Sqrt' then input:abs() end
      local sconv = proto_module:type(ctype)
      local groundtruth = sconv:forward(input)

      input = input:type(typename)
      local gconv = proto_module:clone():type(typename)
      local rescuda = gconv:forward(input)

      local error = rescuda:double() - groundtruth:double()
      mytester:assertlt(error:abs():max(), precision_forward_type(max_error, typename),
        string.format('error on state (forward) with %s', typename))
    end
end

local function pointwise_backward(proto_module, name, max_error)
   local size = math.random(1,100)

   for k, typename in ipairs(typenames) do
      local input = torch.randn(size):type(typename)
      local gradOutput = torch.randn(size):type(typename)

      local ctype = t2cpu[typename]
      input = input:type(ctype)
      gradOutput = gradOutput:type(ctype)
      if name == 'Sqrt' then input:abs() end
      local sconv = proto_module:type(ctype)
      sconv:forward(input)
      local groundgrad = sconv:backward(input, gradOutput)

      input = input:type(typename)
      gradOutput = gradOutput:type(typename)
      local gconv = proto_module:clone():type(typename)
      gconv:forward(input)
      local rescuda = gconv:backward(input, gradOutput)

      local error = rescuda:double() - groundgrad:double()

      mytester:assertlt(error:abs():max(), precision_backward_type(max_error, typename),
        string.format('error on state (backward) with %s', typename))
    end
end

local function pointwise_backward_inplace(proto_module, name)
   local size = math.random(1,100)

   for k, typename in ipairs(typenames) do
      local input = torch.randn(size):type(typename)
      local ctype = t2cpu[typename]
      input = input:type(ctype)
      if name == 'Sqrt' then input:abs() end
      local gradOutput = torch.randn(size)
      gradOutput = gradOutput:type(ctype)
      local sconv = proto_module:type(ctype)
      local groundgrad = sconv:backward(input, gradOutput)
      mytester:assertTensorEq(groundgrad:double(),
                              gradOutput:double(),
                              0.000001,
                              string.format("inplace not respected for %s", ctype))

      input = torch.randn(size)
      input = input:type(typename)
      if name == 'Sqrt' then input:abs() end
      gradOutput = torch.randn(size)
      gradOutput = gradOutput:type(typename)
      local sconv = proto_module:clone():type(typename)
      local groundgrad = sconv:backward(input, gradOutput)
      mytester:assertTensorEq(groundgrad:double(),
                              gradOutput:double(),
                              0.000001,
                              string.format("cuda inplace not respected for %s", typename))
    end
end

local function pointwise_transposed(proto_module, name, max_error)
   max_error = max_error or 1e-7

   for k, typename in ipairs(typenames) do
      local ctype = t2cpu[typename]
      local input = torch.Tensor(11, 19):uniform(-1, 1):type(typename)
      input = input:type(ctype)
      local proto_module = proto_module:type(ctype)
      if name == 'Sqrt' then
        input:uniform(0.1, 1)
      end
      local inputCUDA = input:clone():type(typename)

      local cuda_module = proto_module:clone():type(typename)

      -- transpose the inputs and DON'T make contiguous
      input = input:transpose(1, 2)
      inputCUDA = inputCUDA:transpose(1, 2)

      local output = proto_module:forward(input)
      local outputCUDA = cuda_module:forward(inputCUDA)

      local error = outputCUDA:double() - output:double()
      mytester:assertlt(error:abs():max(), precision_forward_type(max_error, typename),
        string.format('error on state (forward) for %s', typename))

      local gradOutput = torch.Tensor(11, 19):uniform(-1, 1):type(ctype)
      local gradOutputCUDA = gradOutput:clone():type(typename)

      gradOutput = gradOutput:transpose(1, 2)
      gradOutputCUDA = gradOutputCUDA:transpose(1, 2)

      local gradInput = proto_module:backward(input, gradOutput)
      local gradInputCUDA  = cuda_module:backward(inputCUDA, gradOutputCUDA)

      local error = gradInputCUDA:double() - gradInput:double()
      mytester:assertlt(error:abs():max(), precision_backward_type(max_error, typename),
        string.format('error on state (backward) for %s', typename))
    end
end

function cunntest.Tanh_forward()
   pointwise_forward(nn.Tanh(), 'Tanh', precision_forward)
end

function cunntest.Tanh_backward()
   pointwise_backward(nn.Tanh(), 'Tanh', precision_backward)
end

function cunntest.Tanh_transposed()
   pointwise_transposed(nn.Tanh(), 'Tanh', 1.8e-7)
end

function cunntest.HardTanh_forward()
   pointwise_forward(nn.HardTanh(), 'HardTanh', precision_forward)
end

function cunntest.HardTanh_backward()
   pointwise_backward(nn.HardTanh(), 'HardTanh', precision_backward)
end

function cunntest.HardTanh_backward_inplace()
   pointwise_backward_inplace(nn.HardTanh(nil, nil, true), 'HardTanh')
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

function cunntest.ReLU6_forward()
  for inplace = 0, 1 do
    local net = nn.Sequential()
    -- pointwise_forward uses randn, so add a big constant to make sure some
    -- of the values saturate.
    net:add(nn.MulConstant(6))
    net:add(nn.ReLU6(inplace == 1))
    pointwise_forward(net, 'ReLU6 inplace ' .. inplace, precision_forward)
  end
end

function cunntest.ReLU6_backward()
  for inplace = 0, 1 do
    local net = nn.Sequential()
    net:add(nn.MulConstant(6))
    net:add(nn.ReLU6(inplace == 1))
    pointwise_backward(net, 'ReLU6 inplace ' .. inplace, precision_backward)
  end
end

function cunntest.LeakyReLU_forward()
   pointwise_forward(nn.LeakyReLU(), 'LeakyReLU', precision_forward)
end

function cunntest.LeakyReLU_backward()
   pointwise_backward(nn.LeakyReLU(), 'LeakyReLU', precision_backward)
end

function cunntest.LeakyReLU_transposed()
   pointwise_transposed(nn.LeakyReLU(), 'LeakyReLU', 1.5e-7)
end

function cunntest.Sqrt_forward()
   pointwise_forward(nn.Sqrt(), 'Sqrt', precision_forward)
end

function cunntest.Sqrt_backward()
   pointwise_backward(nn.Sqrt(), 'Sqrt', precision_backward)
end

function cunntest.Sqrt_zero()
   local size = math.random(1, 100)

   for k, typename in ipairs(typenames) do
      -- Test zero inputs; we will avoid a div-by-zero by setting to zero
      local module_gpu = nn.Sqrt():type(typename)
      local input_gpu = torch.CudaTensor(size, size):zero():type(typename)
      module_gpu:forward(input_gpu)

      local gradOutput_gpu = torch.CudaTensor(size, size):fill(1):type(typename)
      local gradInput_gpu = module_gpu:backward(input_gpu, gradOutput_gpu)

      mytester:assertTensorEq(gradInput_gpu:double(),
                              torch.DoubleTensor(size, size):zero(),
                              0.000001, "error in sqrt backward singularity")

      -- Verify CPU and GPU zero behavior equivalency
      local ctype = t2cpu[typename]
      local module_cpu = nn.Sqrt():type(ctype)
      local input_cpu = input_gpu:type(ctype)
      module_cpu:forward(input_cpu)

      local gradOutput_cpu = gradOutput_gpu:type(ctype)
      local gradInput_cpu = module_cpu:backward(input_cpu, gradOutput_cpu)

      mytester:assertTensorEq(gradInput_gpu:double(),
                            gradInput_cpu:double(),
                            0.000001, "Sqrt_zero CPU and GPU not equivalent")
    end
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

function cunntest.SoftShrink_forward()
  pointwise_forward(nn.SoftShrink(math.random()), 'SoftShrink', precision_forward)
end

function cunntest.SoftShrink_backward()
  pointwise_backward(nn.SoftShrink(math.random()), 'SoftShrink', precision_backward)
end

function cunntest.SoftShrink_transposed()
  pointwise_transposed(nn.SoftShrink(math.random()), 'SoftShrink', precision_backward)
end

function cunntest.ELU_forward()
   pointwise_forward(nn.ELU(), 'ELU', precision_forward)
end

function cunntest.ELU_backward()
   pointwise_backward(nn.ELU(), 'ELU', precision_backward)
end

function cunntest.ELU_transposed()
   pointwise_transposed(nn.ELU(), 'ELU', 1e-6)
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

function cunntest.SpatialSoftMax()
   local bs = math.random(32,256)
   local dim = torch.random(1, 50)
   local h = torch.random(1, 50)
   local w = torch.random(1, 50)

   local input = torch.randn(bs, dim, h, w)
   local sconv = nn.SpatialSoftMax()
   local groundtruth = sconv:forward(input)
   local gradOutput = groundtruth:clone():fill(0.5)
   local gradInput = sconv:backward(input, gradOutput)

   input = input:cuda()
   gradOutput = gradOutput:cuda()
   local gconv = nn.SpatialSoftMax():cuda()
   local rescuda = gconv:forward(input)
   local gradcuda = gconv:backward(input, gradOutput)

   local error = rescuda:float() - groundtruth
   mytester:assertlt(error:abs():max(), precision_forward*10, 'error on state (forward) ')

   local error = gradcuda:float() - gradInput
   mytester:assertlt(error:abs():max(), precision_backward*10, 'error on state (backward) ')
end

function cunntest.LogSoftMax_forward_batch()
   local size = math.random(1,256)
   local bs = math.random(32,256)

   for k, typename in ipairs(typenames) do
      local input = torch.randn(bs, size):type(typename)
      local ctype = t2cpu[typename]
      input = input:type(ctype)
      local sconv = nn.LogSoftMax():type(ctype)
      local groundtruth = sconv:forward(input)

      input = input:type(typename)
      local gconv = nn.LogSoftMax():type(typename)
      local rescuda = gconv:forward(input)

      local error = rescuda:double() - groundtruth:double()
      mytester:assertlt(error:abs():max(), precision_forward_type(precision_forward*10, typename),
          string.format('error on state (forward) with %s', typename))
    end
end

function cunntest.LogSoftMax_backward_batch()
   local size = math.random(1,256)
   local bs = math.random(32,256)

   for k, typename in ipairs(typenames) do
      local input = torch.randn(bs, size):type(typename)
      local gradOutput = torch.randn(bs, size):type(typename)
      local ctype = t2cpu[typename]
      input = input:type(ctype)
      gradOutput = gradOutput:type(ctype)
      local sconv = nn.LogSoftMax():type(ctype)
      sconv:forward(input)
      local groundgrad = sconv:backward(input, gradOutput)

      input = input:type(typename)
      gradOutput = gradOutput:type(typename)
      local gconv = sconv:clone():type(typename)
      gconv:forward(input)
      local rescuda = gconv:backward(input, gradOutput)

      local error = rescuda:double() - groundgrad:double()

      mytester:assertlt(error:abs():max(), precision_backward_type(precision_backward, typename),
          string.format('error on state (backward) with %s', typename))
    end
end

function cunntest.SpatialLogSoftMax_forward()
   local size = math.random(1,256)
   local ini = math.random(8,32)
   local inj = math.random(8,32)

   for k, typename in ipairs(typenames) do
      local input = torch.randn(size, inj, ini):type(typename)
      local ctype = t2cpu[typename]
      input = input:type(ctype)
      local sconv = nn.SpatialLogSoftMax():type(ctype)
      local groundtruth = sconv:forward(input):type(ctype)

      input = input:type(typename)
      local gconv = nn.SpatialLogSoftMax():type(typename)
      local rescuda = gconv:forward(input)

      local error = rescuda:double() - groundtruth:double()
      mytester:assertlt(error:abs():max(),
          precision_forward_type(precision_forward*25, typename),
          string.format('error on state (forward) with %s', typename))
    end
end

function cunntest.SpatialLogSoftMax_backward()
   local size = math.random(1,256)
   local ini = math.random(8,32)
   local inj = math.random(8,32)

   for k, typename in ipairs(typenames) do
      local input = torch.randn(size, inj, ini):type(typename)
      local gradOutput = torch.randn(size, inj, ini):type(typename)
      local ctype = t2cpu[typename]
      input = input:type(ctype)
      gradOutput = gradOutput:type(ctype)
      local sconv = nn.SpatialLogSoftMax():type(ctype)
      sconv:forward(input)
      local groundgrad = sconv:backward(input, gradOutput)

      input = input:type(typename)
      gradOutput = gradOutput:type(typename)
      local gconv = sconv:clone():type(typename)
      gconv:forward(input)
      local rescuda = gconv:backward(input, gradOutput)

      local error = rescuda:double() - groundgrad:double()

      mytester:assertlt(error:abs():max(), precision_backward_type(precision_backward, typename),
          string.format('error on state (backward) with %s', typename))
    end
end

function cunntest.SpatialLogSoftMax_forward_batch()
   local size = math.random(1,256)
   local bs = math.random(8,32)
   local ini = math.random(8,32)
   local inj = math.random(8,32)

   for k, typename in ipairs(typenames) do
      local input = torch.randn(bs, size, inj, ini):type(typename)
      local ctype = t2cpu[typename]
      input = input:type(ctype)
      local sconv = nn.SpatialLogSoftMax():type(ctype)
      local groundtruth = sconv:forward(input)

      input = input:type(typename)
      local gconv = nn.SpatialLogSoftMax():type(typename)
      local rescuda = gconv:forward(input)

      local error = rescuda:double() - groundtruth:double()
      mytester:assertlt(error:abs():max(),
          precision_forward_type(precision_forward*25, typename),
          string.format('error on state (forward) with %s', typename))
    end
end

function cunntest.SpatialLogSoftMax_backward_batch()
   local size = math.random(1,256)
   local bs = math.random(8,32)
   local ini = math.random(8,32)
   local inj = math.random(8,32)

   for k, typename in ipairs(typenames) do
      local input = torch.randn(bs, size, inj, ini):type(typename)
      local gradOutput = torch.randn(bs, size, inj, ini):type(typename)
      local ctype = t2cpu[typename]
      input = input:type(ctype)
      gradOutput = gradOutput:type(ctype)
      local sconv = nn.SpatialLogSoftMax():type(ctype)
      sconv:forward(input)
      local groundgrad = sconv:backward(input, gradOutput)

      input = input:type(typename)
      gradOutput = gradOutput:type(typename)
      local gconv = sconv:clone():type(typename)
      gconv:forward(input)
      local rescuda = gconv:backward(input, gradOutput)

      local error = rescuda:double() - groundgrad:double()

      mytester:assertlt(error:abs():max(), precision_backward_type(precision_backward, typename),
          string.format('error on state (backward) with %s', typename))
    end
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

function cunntest.SparseLinear_forward()
    local inb = math.random(5,10)
    local ini = math.random(50,100)
    local inj = math.random(5,10)

    local module = nn.SparseLinear(ini,inj)
    local sslin = module
    local gslin = module:clone():cuda()

    -- Create a random sparse vector
    local input = {}
    for i=1,inb do
        local nnz = math.random(5, 10)
        local inds = torch.randperm(ini)[{{1,nnz}}]
        input[i] = torch.Tensor(nnz, 2)
        input[i]:select(2,1):copy(inds)
        input[i]:select(2,2):copy(torch.rand(nnz))
    end

    local tm = {}
    local title = string.format('SparseLinear forward %d -> %d', ini, inj)
    times[title] = tm

    local groundtruth = sslin:forward(input)
    sslin:zeroGradParameters()
    local a = torch.Timer()
    for i = 1,nloop do
        groundtruth = sslin:forward(input)
    end
    tm.cpu = a:time().real

    for i,v in ipairs(input) do input[i] = input[i]:cuda() end
    local rescuda = gslin:forward(input)
    gslin:zeroGradParameters()
    a:reset()
    for i = 1,nloop do
        rescuda = gslin:forward(input)
    end
    cutorch.synchronize()
    tm.gpu = a:time().real

    local error = rescuda:float() - groundtruth
    mytester:assertlt(error:abs():max(), precision_forward, 'error on state (forward) ')
end

function cunntest.SparseLinear_backward()
    local inb = math.random(5,10)
    local ini = math.random(50,100)
    local inj = math.random(5,10)

    local gslin = nn.SparseLinear(ini,inj):cuda()
    local sslin = nn.Linear(ini,inj)
    gslin.weight = sslin.weight:clone():cuda()
    gslin.bias = sslin.bias:clone():cuda()

    -- Create a random sparse vector
    local input = {}
    local nonsparse = torch.zeros(inb, ini)
    for i=1,inb do
        local nnz = math.random(3, 5)
        local inds = torch.randperm(ini)[{{1,nnz}}]
        input[i] = torch.Tensor(nnz, 2)
        input[i]:select(2,1):copy(inds)
        input[i]:select(2,2):copy(torch.rand(nnz))
        nonsparse[i]:scatter(1, input[i]:select(2,1):long(), input[i]:select(2,2))
    end

    local tm = {}
    local title = string.format('SparseLinear backward %d <- %d', ini, inj)
    times[title] = tm

    local gradOutput = torch.randn(inb, inj)
    sslin:forward(nonsparse)
    local groundgrad = sslin:backward(nonsparse, gradOutput)
    sslin:zeroGradParameters()
    local a = torch.Timer()
    for i = 1,nloop do
        sslin:backward(nonsparse, gradOutput)
    end
    tm.cpu = a:time().real
    local groundweight = sslin.gradWeight
    local groundbias = sslin.gradBias

    for i,v in ipairs(input) do input[i] = input[i]:cuda() end
    gradOutput = gradOutput:cuda()
    gslin:forward(input)
    local rescuda = gslin:backward(input, gradOutput)
    gslin:zeroGradParameters()
    a:reset()
    for i = 1,nloop do
        gslin:backward(input, gradOutput)
    end
    local weightcuda = gslin.gradWeight
    local biascuda = gslin.gradBias
    cutorch.synchronize()
    tm.gpu = a:time().real

    local werror = weightcuda:float() - groundweight
    local berror = biascuda:float() - groundbias

    mytester:assertlt(werror:abs():max(), precision_backward, 'error on weight (backward) ')
    mytester:assertlt(berror:abs():max(), precision_backward, 'error on bias (backward) ')

    gslin:updateParameters(.1)
    sslin:updateParameters(.1)
    werror = gslin.weight:float() - sslin.weight
    berror = gslin.bias:float() - sslin.bias

    mytester:assertlt(werror:abs():max(), precision_backward, 'error on weight (update) ')
    mytester:assertlt(berror:abs():max(), precision_backward, 'error on bias (update) ')

    gslin:zeroGradParameters()
end

local function BatchNormalization_forward(moduleName, inputSize)
   local planes = inputSize[2]
   local tm = {}
   local title = moduleName .. '.forward ' .. table.concat(inputSize, 'x')
   times[title] = tm

   local input = torch.randn(table.unpack(inputSize))
   local sbnorm = nn[moduleName](planes)
   local groundtruth = sbnorm:forward(input)
   local a = torch.Timer()
   for i = 1,nloop do
      groundtruth = sbnorm:forward(input)
   end
   tm.cpu = a:time().real

   input = input:cuda()
   local gbnorm = nn[moduleName](planes):cuda()
   gbnorm.weight = sbnorm.weight:cuda()
   gbnorm.bias = sbnorm.bias:cuda()
   local rescuda = gbnorm:forward(input)

   a:reset()
   for i = 1,nloop do
      rescuda = gbnorm:forward(input)
   end
   cutorch.synchronize()
   tm.gpu = a:time().real

   local error = rescuda:float() - groundtruth
   mytester:assertlt(error:abs():max(), precision_forward, 'error on state (forward)')
   mytester:assertlt((gbnorm.running_mean:float() - sbnorm.running_mean):abs():max(),
      precision_forward, 'error on running_mean (forward)')
   mytester:assertlt((gbnorm.running_var:float() - sbnorm.running_var):abs():max(),
      precision_forward, 'error on running_var (forward)')
end

local function BatchNormalization_forward_inference(moduleName, inputSize)
   local planes = inputSize[2]
   local tm = {}
   local title = moduleName .. '.forward (evaluate) ' .. table.concat(inputSize, 'x')
   times[title] = tm

   local input = torch.randn(table.unpack(inputSize))
   local sbnorm = nn[moduleName](planes)
   sbnorm.running_mean:normal(1, 2)
   sbnorm.running_var:uniform(1e-3, 2)
   sbnorm:evaluate()
   local groundtruth = sbnorm:forward(input)
   local a = torch.Timer()
   for i = 1,nloop do
      groundtruth = sbnorm:forward(input)
   end
   tm.cpu = a:time().real

   input = input:cuda()
   local gbnorm = nn[moduleName](planes):cuda()
   gbnorm:evaluate()
   gbnorm.weight = sbnorm.weight:cuda()
   gbnorm.bias = sbnorm.bias:cuda()
   gbnorm.running_mean = sbnorm.running_mean:cuda()
   gbnorm.running_var = sbnorm.running_var:cuda()
   local rescuda = gbnorm:forward(input)
   a:reset()
   for i = 1,nloop do
      rescuda = gbnorm:forward(input)
   end
   cutorch.synchronize()
   tm.gpu = a:time().real

   local error = rescuda:float() - groundtruth
   mytester:assertlt(error:abs():max(), precision_forward, 'error on state (forward evaluate)')
end

local function BatchNormalization_backward(moduleName, mode, inputSize, backwardFn)
   assert(mode == 'training' or mode == 'evaluation', 'invalid mode')

   local planes = inputSize[2]
   local tm = {}
   local title = moduleName .. '.backward ' .. table.concat(inputSize, 'x')
   times[title] = tm

   local input = torch.randn(table.unpack(inputSize))
   local gradOutput = torch.randn(table.unpack(inputSize))
   local sbnorm = nn[moduleName](planes)
   if mode == 'training' then
     sbnorm:training()
   else
     sbnorm:evaluate()
   end
   sbnorm:forward(input)
   sbnorm:zeroGradParameters()
   local groundgrad = backwardFn(sbnorm, input, gradOutput)
   local a = torch.Timer()
   for i = 1,nloop do
      sbnorm:zeroGradParameters()
      groundgrad = backwardFn(sbnorm, input, gradOutput)
   end
   local groundweight = sbnorm.gradWeight
   local groundbias = sbnorm.gradBias
   tm.cpu = a:time().real

   input = input:cuda()
   gradOutput = gradOutput:cuda()
   local gbnorm = nn[moduleName](planes):cuda()
   if mode == 'training' then
     gbnorm:training()
   else
     gbnorm:evaluate()
   end
   gbnorm.weight = sbnorm.weight:cuda()
   gbnorm.bias = sbnorm.bias:cuda()
   gbnorm:forward(input)
   gbnorm:zeroGradParameters()
   local rescuda = backwardFn(gbnorm, input, gradOutput)
   a:reset()
   for i = 1,nloop do
      gbnorm:zeroGradParameters()
      rescuda = backwardFn(gbnorm, input, gradOutput)
   end
   local weightcuda = gbnorm.gradWeight
   local biascuda = gbnorm.gradBias
   cutorch.synchronize()
   tm.gpu = a:time().real

   local error = rescuda:float() - groundgrad
   local werror = weightcuda:float() - groundweight
   local berror = biascuda:float() - groundbias

   mytester:assertlt(error:abs():max(), precision_backward, 'error on state (backward) ')
   mytester:assertlt(werror:abs():max(), precision_backward, 'error on weight (backward) ')
   mytester:assertlt(berror:abs():max(), precision_backward, 'error on bias (backward) ')
end

local function testBatchNormalization(name, dim, k)
   local function inputSize()
      local inputSize = { torch.random(2,32), torch.random(1, k) }
      for i=1,dim do
         table.insert(inputSize, torch.random(1,k))
      end
      return inputSize
   end
   local function backward1(m, input, gradOutput)
      return m:backward(input, gradOutput)
   end
   local function backward2(m, input, gradOutput)
      local gradInput = m:updateGradInput(input, gradOutput)
      m:accGradParameters(input, gradOutput)
      return gradInput
   end

   BatchNormalization_forward(name, inputSize())
   BatchNormalization_forward_inference(name, inputSize())
   BatchNormalization_backward(name, 'training', inputSize(), backward1)
   BatchNormalization_backward(name, 'training', inputSize(), backward2)
   BatchNormalization_backward(name, 'evaluation', inputSize(), backward1)
   BatchNormalization_backward(name, 'evaluation', inputSize(), backward2)
end

function cunntest.BatchNormalization()
   testBatchNormalization('BatchNormalization', 0, 128)
end

function cunntest.SpatialBatchNormalization()
   testBatchNormalization('SpatialBatchNormalization', 2, 64)
   -- check with large image size (32*32 = 1024)
   BatchNormalization_forward('SpatialBatchNormalization', {2, 2, 32, 32})
end

function cunntest.VolumetricBatchNormalization()
   testBatchNormalization('VolumetricBatchNormalization', 3, 16)
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

   local function jacTests(noBias)
      noBias = noBias or false
      for k, typename in ipairs(typenames) do
         local input = torch.randn(from,inj,ini):type(typename)

         local ctype = t2cpu[typename]
         input = input:type(ctype)
         local sconv = nn.SpatialConvolutionMM(from,to,ki,kj,si,sj,padW,padH):type(ctype)
         if noBias then
            sconv:noBias()
         end
         local groundtruth = sconv:forward(input)

         input = input:type(typename)
         local gconv = nn.SpatialConvolutionMM(from,to,ki,kj,si,sj,padW,padH):type(typename)
         if noBias then
            gconv:noBias()
         end
         gconv.weight = sconv.weight:type(typename)
         if gconv.bias then
            gconv.bias = sconv.bias:type(typename)
         end
         local rescuda = gconv:forward(input)

         local error = rescuda:double() - groundtruth:double()
         mytester:assertlt(error:abs():max(), precision_forward_type(precision_forward, typename),
            string.format('error on state (forward) with %s', typename))
      end
   end

   jacTests(false)
   jacTests(true)
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

   local function jacTests(noBias)
      noBias = noBias or false
      for k, typename in ipairs(typenames) do
         local input = torch.randn(bs,from,inj,ini):type(typename)

         local ctype = t2cpu[typename]
         input = input:type(ctype)
         local sconv = nn.SpatialConvolutionMM(from,to,ki,kj,si,sj,padW,padH):type(ctype)
         if noBias then
            sconv:noBias()
         end
         local groundtruth = sconv:forward(input)

         input = input:type(typename)
         local gconv = nn.SpatialConvolutionMM(from,to,ki,kj,si,sj,padW,padH):type(typename)
         if noBias then
            gconv:noBias()
         end
         gconv.weight = sconv.weight:type(typename)
         if gconv.bias then
            gconv.bias = sconv.bias:type(typename)
         end
         local rescuda = gconv:forward(input)

         local error = rescuda:double() - groundtruth:double()
         mytester:assertlt(error:abs():max(), precision_forward_type(precision_forward, typename),
            string.format('error on state (forward) with %s', typename))
      end
   end


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

   local function jacTests(noBias)
      noBias = noBias or false

      for k, typename in ipairs(typenames) do
         local input = torch.randn(from,inj,ini):type(typename)
         local gradOutput = torch.randn(to,outj,outi):type(typename)

         local ctype = t2cpu[typename]
         input = input:type(ctype)
         gradOutput = gradOutput:type(ctype)
         local sconv = nn.SpatialConvolutionMM(from,to,ki,kj,si,sj,padW,padH):type(ctype)
         if noBias then
            sconv:noBias()
         end
         sconv:forward(input)
         sconv:zeroGradParameters()
         local groundgrad = sconv:backward(input, gradOutput)
         local groundweight = sconv.gradWeight
         local groundbias = sconv.gradBias

         input = input:type(typename)
         gradOutput = gradOutput:type(typename)
         local gconv = nn.SpatialConvolutionMM(from,to,ki,kj,si,sj,padW,padH):type(typename)
         if noBias then
            gconv:noBias()
         end
         gconv.weight = sconv.weight:type(typename)
         if gconv.bias then
            gconv.bias = sconv.bias:type(typename)
         end
         gconv:forward(input)
         gconv:zeroGradParameters()
         local rescuda = gconv:backward(input, gradOutput)
         local weightcuda = gconv.gradWeight

         local error = rescuda:double() - groundgrad:double()
         local werror = weightcuda:double() - groundweight:double()

         mytester:assertlt(error:abs():max(), precision_backward_type(precision_backward, typename),
            string.format('error on state (backward) with %s', typename))
         mytester:assertlt(werror:abs():max(), precision_backward_conv_weightbias(precision_backward, typename),
            string.format('error on weight (backward) with %s', typename))

         if gconv.bias then
            local berror = gconv.gradBias:double() - groundbias:double()
            mytester:assertlt(berror:abs():max(), precision_backward_conv_weightbias(precision_backward, typename),
                string.format('error on bias (backward) with %s', typename))
         end
      end
   end

   jacTests(false)
   jacTests(true)
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

   local function jacTests(noBias)
      noBias = noBias or false

      for k, typename in ipairs(typenames) do
         local input = torch.randn(bs,from,inj,ini)
         local gradOutput = torch.randn(bs,to,outj,outi)

         local ctype = t2cpu[typename]
         input = input:type(ctype)
         gradOutput = gradOutput:type(ctype)
         local sconv = nn.SpatialConvolutionMM(from,to,ki,kj,si,sj,padW,padH):type(ctype)
         if noBias then
            sconv:noBias()
         end
         sconv:forward(input)
         sconv:zeroGradParameters()
         local groundgrad = sconv:backward(input, gradOutput)
         local groundweight = sconv.gradWeight
         local groundbias = sconv.gradBias

         input = input:type(typename)
         gradOutput = gradOutput:type(typename)
         local gconv = nn.SpatialConvolutionMM(from,to,ki,kj,si,sj,padW,padH):type(typename)
         if noBias then
            gconv:noBias()
         end
         gconv.weight = sconv.weight:type(typename)
         if gconv.bias then
            gconv.bias = sconv.bias:type(typename)
         end
         gconv:forward(input)
         gconv:zeroGradParameters()
         local rescuda = gconv:backward(input, gradOutput)
         local weightcuda = gconv.gradWeight

         local error = rescuda:double() - groundgrad:double()
         local werror = weightcuda:double() - groundweight:double()

         mytester:assertlt(error:abs():max(), precision_backward_type(precision_backward, typename),
            string.format('error on state (backward) with %s', typename))
         mytester:assertlt(werror:abs():max(), precision_backward_conv_weightbias(precision_backward, typename),
            string.format('error on weight (backward) with %s', typename))
         if gconv.bias then
            local berror = gconv.gradBias:double() - groundbias:double()
            mytester:assertlt(berror:abs():max(), precision_backward_conv_weightbias(precision_backward, typename),
                string.format('error on bias (backward) with %s', typename))
         end
      end
   end

   jacTests(false)
   jacTests(true)
end

function cunntest.SpatialConvolutionLocal_forward_single()
   local from = math.random(1,32)
   local to = math.random(1,8) * 8
   local ki = math.random(3,15)
   local kj = math.random(3,15)
   local si = math.random(1,3)
   local sj = math.random(1,3)
   local outi = math.random(1,48)
   local outj = math.random(1,48)
   local padW = math.random(0,1)
   local padH = math.random(0,1)
   local ini = (outi-1)*si+ki-padW*2
   local inj = (outj-1)*sj+kj-padH*2

   for k, typename in ipairs(typenames) do
      local input = torch.randn(from,inj,ini):type(typename)

      local ctype = t2cpu[typename]
      input = input:type(ctype)
      local sconv = nn.SpatialConvolutionLocal(from,to,ini,inj,ki,kj,si,sj,padW,padH):type(ctype)
      local groundtruth = sconv:forward(input)

      input = input:type(typename)
      local gconv = nn.SpatialConvolutionLocal(from,to,ini,inj,ki,kj,si,sj,padW,padH):type(typename)
      gconv.weight = sconv.weight:type(typename)
      gconv.bias = sconv.bias:type(typename)
      local rescuda = gconv:forward(input)

      local error = rescuda:double() - groundtruth:double()
      mytester:assertlt(error:abs():max(), precision_forward_type(precision_forward, typename),
          string.format('error on state (forward) with %s', typename))
   end
end

function cunntest.SpatialConvolutionLocal_forward_batch()
   local bs = math.random(1,4) * 4
   local from = math.random(1,32)
   local to = math.random(1,8) * 8
   local ki = math.random(3,15)
   local kj = math.random(3,15)
   local si = math.random(1,3)
   local sj = math.random(1,3)
   local outi = math.random(1,48)
   local outj = math.random(1,48)
   local padW = math.random(0,1)
   local padH = math.random(0,1)
   local ini = (outi-1)*si+ki-padW*2
   local inj = (outj-1)*sj+kj-padH*2

   for k, typename in ipairs(typenames) do
      local input = torch.randn(bs,from,inj,ini):type(typename)

      local ctype = t2cpu[typename]
      input = input:type(ctype)
      local sconv = nn.SpatialConvolutionLocal(from,to,ini,inj,ki,kj,si,sj,padW,padH):type(ctype)
      local groundtruth = sconv:forward(input)

      input = input:type(typename)
      local gconv = nn.SpatialConvolutionLocal(from,to,ini,inj,ki,kj,si,sj,padW,padH):type(typename)
      gconv.weight = sconv.weight:type(typename)
      gconv.bias = sconv.bias:type(typename)
      local rescuda = gconv:forward(input)

      local error = rescuda:double() - groundtruth:double()
      mytester:assertlt(error:abs():max(), precision_forward_type(precision_forward, typename),
          string.format('error on state (forward) with %s', typename))
   end
end

function cunntest.SpatialConvolutionLocal_backward_single()
   local from = math.random(1,32)
   local to = math.random(1,8) * 8
   local ki = math.random(3,15)
   local kj = math.random(3,15)
   local si = math.random(1,3)
   local sj = math.random(1,3)
   local outi = math.random(1,48)
   local outj = math.random(1,48)
   local padW = math.random(0,1)
   local padH = math.random(0,1)
   local ini = (outi-1)*si+ki-padW*2
   local inj = (outj-1)*sj+kj-padH*2

   for k, typename in ipairs(typenames) do
      local input = torch.randn(from,inj,ini):type(typename)
      local gradOutput = torch.randn(to,outj,outi):type(typename)

      local ctype = t2cpu[typename]
      input = input:type(ctype)
      gradOutput = gradOutput:type(ctype)
      local sconv = nn.SpatialConvolutionLocal(from,to,ini,inj,ki,kj,si,sj,padW,padH):type(ctype)
      sconv:forward(input)
      sconv:zeroGradParameters()
      local groundgrad = sconv:backward(input, gradOutput)
      local groundweight = sconv.gradWeight
      local groundbias = sconv.gradBias

      input = input:type(typename)
      gradOutput = gradOutput:type(typename)
      local gconv = nn.SpatialConvolutionLocal(from,to,ini,inj,ki,kj,si,sj,padW,padH):type(typename)
      gconv.weight = sconv.weight:type(typename)
      gconv.bias = sconv.bias:type(typename)
      gconv:forward(input)
      gconv:zeroGradParameters()
      local rescuda = gconv:backward(input, gradOutput)
      local weightcuda = gconv.gradWeight
      local biascuda = gconv.gradBias

      local error = rescuda:double() - groundgrad:double()
      local werror = weightcuda:double() - groundweight:double()
      local berror = biascuda:double() - groundbias:double()

      mytester:assertlt(error:abs():max(), precision_backward_type(precision_backward, typename),
          string.format('error on state (backward) with %s', typename))
      mytester:assertlt(werror:abs():max(), precision_backward_conv_weightbias(precision_backward, typename),
          string.format('error on weight (backward) with %s', typename))
      mytester:assertlt(berror:abs():max(), precision_backward_conv_weightbias(precision_backward, typename),
          string.format('error on bias (backward) with %s', typename))
   end
end

function cunntest.SpatialConvolutionLocal_backward_batch()
   local bs = math.random(1,4) * 4
   local from = math.random(1,32)
   local to = math.random(1,8) * 8
   local ki = math.random(3,15)
   local kj = math.random(3,15)
   local si = math.random(1,3)
   local sj = math.random(1,3)
   local outi = math.random(1,48)
   local outj = math.random(1,48)
   local padW = math.random(0,1)
   local padH = math.random(0,1)
   local ini = (outi-1)*si+ki-padW*2
   local inj = (outj-1)*sj+kj-padH*2

   for k, typename in ipairs(typenames) do
      local input = torch.randn(bs,from,inj,ini):type(typename)
      local gradOutput = torch.randn(bs,to,outj,outi):type(typename)

      local ctype = t2cpu[typename]
      input = input:type(ctype)
      gradOutput = gradOutput:type(ctype)
      local sconv = nn.SpatialConvolutionLocal(from,to,ini,inj,ki,kj,si,sj,padW,padH):type(ctype)
      sconv:forward(input)
      sconv:zeroGradParameters()
      local groundgrad = sconv:backward(input, gradOutput)
      local groundweight = sconv.gradWeight
      local groundbias = sconv.gradBias

      input = input:type(typename)
      gradOutput = gradOutput:type(typename)
      local gconv = nn.SpatialConvolutionLocal(from,to,ini,inj,ki,kj,si,sj,padW,padH):type(typename)
      gconv.weight = sconv.weight:type(typename)
      gconv.bias = sconv.bias:type(typename)
      gconv:forward(input)
      gconv:zeroGradParameters()
      local rescuda = gconv:backward(input, gradOutput)
      local weightcuda = gconv.gradWeight
      local biascuda = gconv.gradBias

      local error = rescuda:double() - groundgrad:double()
      local werror = weightcuda:double() - groundweight:double()
      local berror = biascuda:double() - groundbias:double()

      mytester:assertlt(error:abs():max(), precision_backward_type(precision_backward, typename),
          string.format('error on state (backward) with %s', typename))
      mytester:assertlt(werror:abs():max(), precision_backward_conv_weightbias(precision_backward, typename),
          string.format('error on weight (backward) with %s', typename))
      mytester:assertlt(berror:abs():max(), precision_backward_conv_weightbias(precision_backward, typename),
          string.format('error on bias (backward) with %s', typename))
   end
end

function cunntest.SpatialFullConvolution_forward_single()
   local from = math.random(1,32)
   local to = math.random(1,8) * 8
   local ki = math.random(3,15)
   local kj = math.random(3,15)
   local si = math.random(1,3)
   local sj = math.random(1,3)
   local padW = math.random(0,1)
   local padH = math.random(0,1)
   local outi = math.random(ki, 64)
   local outj = math.random(kj, 64)
   local adjW = (outi + padW*2 - ki) % si
   local adjH = (outj + padH*2 - kj) % sj
   local ini = math.floor((outi + 2 * padW - ki) / si + 1)
   local inj = math.floor((outj + 2 * padH - kj) / sj + 1)

   local function jacTests(noBias)
      noBias = noBias or false
      for k, typename in ipairs(typenames) do
         local input = torch.randn(from,inj,ini):type(typename)

         local ctype = t2cpu[typename]
         input = input:type(ctype)
         local sconv = nn.SpatialFullConvolution(from,to,ki,kj,si,sj,padW,padH,adjW,adjH):type(ctype)
         if noBias then
            sconv:noBias()
         end
         local groundtruth = sconv:forward(input)

         input = input:type(typename)
         local gconv = nn.SpatialFullConvolution(from,to,ki,kj,si,sj,padW,padH,adjW,adjH):type(typename)
         if noBias then
            gconv:noBias()
         end
         gconv.weight = sconv.weight:type(typename)
         if gconv.bias then
            gconv.bias = sconv.bias:type(typename)
         end
         local rescuda = gconv:forward(input)

         local error = rescuda:double() - groundtruth:double()
         mytester:assertlt(error:abs():max(), precision_forward_type(precision_forward, typename),
            string.format('error on state (forward) with %s', typename))
      end
   end

   jacTests(false)
   jacTests(true)
end

function cunntest.SpatialFullConvolution_forward_batch()
   local bs = math.random(1,4) * 4
   local from = math.random(1,32)
   local to = math.random(1,8) * 8
   local ki = math.random(3,15)
   local kj = math.random(3,15)
   local si = math.random(1,3)
   local sj = math.random(1,3)
   local padW = math.random(0,1)
   local padH = math.random(0,1)
   local outi = math.random(ki, 64)
   local outj = math.random(kj, 64)
   local adjW = (outi + padW*2 - ki) % si
   local adjH = (outj + padH*2 - kj) % sj
   local ini = math.floor((outi + 2 * padW - ki) / si + 1)
   local inj = math.floor((outj + 2 * padH - kj) / sj + 1)

   local function jacTests(noBias)
      noBias = noBias or false
      for k, typename in ipairs(typenames) do
         local input = torch.randn(bs,from,inj,ini):type(typename)

         local ctype = t2cpu[typename]
         input = input:type(ctype)
         local sconv = nn.SpatialFullConvolution(from,to,ki,kj,si,sj,padW,padH,adjW,adjH):type(ctype)
         if noBias then
            sconv:noBias()
         end
         local groundtruth = sconv:forward(input)

         input = input:type(typename)
         local gconv = nn.SpatialFullConvolution(from,to,ki,kj,si,sj,padW,padH,adjW,adjH):type(typename)
         if noBias then
            gconv:noBias()
         end
         gconv.weight = sconv.weight:type(typename)
         if gconv.bias then
            gconv.bias = sconv.bias:type(typename)
         end
         local rescuda = gconv:forward(input)

         local error = rescuda:double() - groundtruth:double()
         mytester:assertlt(error:abs():max(), precision_forward_type(precision_forward, typename),
              string.format('error on state (forward) with %s', typename))
      end
   end

   jacTests(false)
   jacTests(true)
end

function cunntest.SpatialFullConvolution_backward_single()
   local from = math.random(1,32)
   local to = math.random(1,8) * 8
   local ki = math.random(3,15)
   local kj = math.random(3,15)
   local si = math.random(1,3)
   local sj = math.random(1,3)
   local padW = math.random(0,1)
   local padH = math.random(0,1)
   local outi = math.random(ki, 64)
   local outj = math.random(kj, 64)
   local adjW = (outi + padW*2 - ki) % si
   local adjH = (outj + padH*2 - kj) % sj
   local ini = math.floor((outi + 2 * padW - ki) / si + 1)
   local inj = math.floor((outj + 2 * padH - kj) / sj + 1)

   local function jacTests(noBias)
      noBias = noBias or false
      for k, typename in ipairs(typenames) do
         local input = torch.randn(from,inj,ini):type(typename)

         local ctype = t2cpu[typename]
         input = input:type(ctype)
         local sconv = nn.SpatialFullConvolution(from,to,ki,kj,si,sj,padW,padH,adjW,adjH):type(ctype)
         if noBias then
            sconv:noBias()
         end
         local output = sconv:forward(input)
         local gradOutput = output:clone():normal()
         sconv:zeroGradParameters()
         local groundgrad = sconv:backward(input, gradOutput)
         local groundweight = sconv.gradWeight
         local groundbias = sconv.gradBias

         input = input:type(typename)
         gradOutput = gradOutput:type(typename)
         local gconv = nn.SpatialFullConvolution(from,to,ki,kj,si,sj,padW,padH,adjW,adjH):type(typename)
         if noBias then
            gconv:noBias()
         end
         gconv.weight = sconv.weight:type(typename)
         if gconv.bias then
            gconv.bias = sconv.bias:type(typename)
         end
         gconv:forward(input)
         gconv:zeroGradParameters()
         local rescuda = gconv:backward(input, gradOutput)
         local weightcuda = gconv.gradWeight

         local error = rescuda:double() - groundgrad:double()
         local werror = weightcuda:double() - groundweight:double()

         mytester:assertlt(error:abs():max(), precision_backward_type(precision_backward, typename),
            string.format('error on state (backward) with %s', typename))
         mytester:assertlt(werror:abs():max(), precision_backward_conv_weightbias(precision_backward, typename),
            string.format('error on weight (backward) with %s', typename))

         if gconv.bias then
            local berror = gconv.gradBias:double() - groundbias:double()
            mytester:assertlt(berror:abs():max(), precision_backward_conv_weightbias(precision_backward, typename),
               string.format('error on bias (backward) with %s', typename))
         end
      end
   end

  jacTests(false)
  jacTests(true)
end

function cunntest.SpatialFullConvolution_backward_batch()
   local bs = math.random(1,4) * 4
   local from = math.random(1,32)
   local to = math.random(1,8) * 8
   local ki = math.random(3,15)
   local kj = math.random(3,15)
   local si = math.random(1,3)
   local sj = math.random(1,3)
   local padW = math.random(0,1)
   local padH = math.random(0,1)
   local outi = math.random(ki, 64)
   local outj = math.random(kj, 64)
   local adjW = (outi + padW*2 - ki) % si
   local adjH = (outj + padH*2 - kj) % sj
   local ini = math.floor((outi + 2 * padW - ki) / si + 1)
   local inj = math.floor((outj + 2 * padH - kj) / sj + 1)

   local function jacTests(noBias)
      noBias = noBias or false

      for k, typename in ipairs(typenames) do
         local input = torch.randn(bs,from,inj,ini):type(typename)

         local ctype = t2cpu[typename]
         input = input:type(ctype)
         local sconv = nn.SpatialFullConvolution(from,to,ki,kj,si,sj,padW,padH,adjW,adjH):type(ctype)
         if noBias then
            sconv:noBias()
         end
         local output = sconv:forward(input)
         local gradOutput = output:clone():normal()
         sconv:zeroGradParameters()
         local groundgrad = sconv:backward(input, gradOutput)
         local groundweight = sconv.gradWeight
         local groundbias = sconv.gradBias

         input = input:type(typename)
         gradOutput = gradOutput:type(typename)
         local gconv = nn.SpatialFullConvolution(from,to,ki,kj,si,sj,padW,padH,adjW,adjH):type(typename)
         if noBias then
            gconv:noBias()
         end
         gconv.weight = sconv.weight:type(typename)
         if gconv.bias then
            gconv.bias = sconv.bias:type(typename)
         end
         gconv:forward(input)
         gconv:zeroGradParameters()
         local rescuda = gconv:backward(input, gradOutput)
         local weightcuda = gconv.gradWeight

         local error = rescuda:double() - groundgrad:double()
         local werror = weightcuda:double() - groundweight:double()

         mytester:assertlt(error:abs():max(), precision_backward_type(precision_backward, typename),
            string.format('error on state (backward) with %s', typename))
         mytester:assertlt(werror:abs():max(), precision_backward_conv_weightbias(precision_backward, typename),
            string.format('error on weight (backward) with %s', typename))
         if gconv.bias then
            local berror = gconv.gradBias:double() - groundbias:double()
            mytester:assertlt(berror:abs():max(), precision_backward_conv_weightbias(precision_backward, typename),
               string.format('error on bias (backward) with %s', typename))
         end
      end
   end

   jacTests(false)
   jacTests(true)
end

function cunntest.SpatialDilatedConvolution_forward_single()
   local from = math.random(1,32)
   local to = math.random(1,8) * 8
   local ki = math.random(3,15)
   local kj = math.random(3,15)
   local si = math.random(1,3)
   local sj = math.random(1,3)
   local padW = math.random(0,1)
   local padH = math.random(0,1)
   local outi = math.random(ki, 64)
   local outj = math.random(kj, 64)
   local dilationW = math.random(1,10)
   local dilationH = math.random(1,10)
   local ini = (outi - 1) * si - 2 * padW + dilationW * (ki-1) + 1
   local inj = (outj - 1) * sj - 2 * padH + dilationH * (kj-1) + 1

   for k, typename in ipairs(typenames) do
      local input = torch.randn(from,inj,ini):type(typename)

      local ctype = t2cpu[typename]
      input = input:type(ctype)
      local sconv = nn.SpatialDilatedConvolution(from,to,ki,kj,si,sj,padW,padH,dilationW,dilationH):type(ctype)
      local groundtruth = sconv:forward(input)

      input = input:type(typename)
      local gconv = nn.SpatialDilatedConvolution(from,to,ki,kj,si,sj,padW,padH,dilationW,dilationH):type(typename)
      gconv.weight = sconv.weight:type(typename)
      gconv.bias = sconv.bias:type(typename)
      local rescuda = gconv:forward(input)

      local error = rescuda:double() - groundtruth:double()
      mytester:assertlt(error:abs():max(), precision_forward_type(precision_forward, typename),
         string.format('error on state (forward) with %s', typename))
   end
end

function cunntest.SpatialDilatedConvolution_forward_batch()
   local bs = math.random(1,4) * 4
   local from = math.random(1,32)
   local to = math.random(1,8) * 8
   local ki = math.random(3,15)
   local kj = math.random(3,15)
   local si = math.random(1,3)
   local sj = math.random(1,3)
   local padW = math.random(0,1)
   local padH = math.random(0,1)
   local outi = math.random(ki, 64)
   local outj = math.random(kj, 64)
   local dilationW = math.random(0,10)
   local dilationH = math.random(0,10)
   local ini = (outi - 1) * si - 2 * padW + dilationW * (ki-1) + 1
   local inj = (outj - 1) * sj - 2 * padH + dilationH * (kj-1) + 1

   for k, typename in ipairs(typenames) do
      local input = torch.randn(bs,from,inj,ini):type(typename)

      local ctype = t2cpu[typename]
      input = input:type(ctype)
      local sconv = nn.SpatialDilatedConvolution(from,to,ki,kj,si,sj,padW,padH,dilationW,dilationH):type(ctype)
      local groundtruth = sconv:forward(input)

      input = input:type(typename)
      local gconv = nn.SpatialDilatedConvolution(from,to,ki,kj,si,sj,padW,padH,dilationW,dilationH):type(typename)
      gconv.weight = sconv.weight:type(typename)
      gconv.bias = sconv.bias:type(typename)
      local rescuda = gconv:forward(input)

      local error = rescuda:double() - groundtruth:double()
      mytester:assertlt(error:abs():max(), precision_forward_type(precision_forward, typename),
         string.format('error on state (forward) with %s', typename))
   end
end

function cunntest.SpatialDilatedConvolution_backward_single()
   local from = math.random(1,32)
   local to = math.random(1,8) * 8
   local ki = math.random(3,15)
   local kj = math.random(3,15)
   local si = math.random(1,3)
   local sj = math.random(1,3)
   local padW = math.random(0,1)
   local padH = math.random(0,1)
   local outi = math.random(ki, 64)
   local outj = math.random(kj, 64)
   local dilationW = math.random(0,10)
   local dilationH = math.random(0,10)
   local ini = (outi - 1) * si - 2 * padW + dilationW * (ki-1) + 1
   local inj = (outj - 1) * sj - 2 * padH + dilationH * (kj-1) + 1

   for k, typename in ipairs(typenames) do
      local input = torch.randn(from,inj,ini):type(typename)

      local ctype = t2cpu[typename]
      input = input:type(ctype)
      local sconv = nn.SpatialDilatedConvolution(from,to,ki,kj,si,sj,padW,padH,dilationW,dilationH):type(ctype)
      local output = sconv:forward(input)
      local gradOutput = output:clone():normal()
      sconv:zeroGradParameters()
      local groundgrad = sconv:backward(input, gradOutput)
      local groundweight = sconv.gradWeight
      local groundbias = sconv.gradBias

      input = input:type(typename)
      gradOutput = gradOutput:type(typename)
      local gconv = nn.SpatialDilatedConvolution(from,to,ki,kj,si,sj,padW,padH,dilationW,dilationH):type(typename)
      gconv.weight = sconv.weight:type(typename)
      gconv.bias = sconv.bias:type(typename)
      gconv:forward(input)
      gconv:zeroGradParameters()
      local rescuda = gconv:backward(input, gradOutput)
      local weightcuda = gconv.gradWeight
      local biascuda = gconv.gradBias

      local error = rescuda:double() - groundgrad:double()
      local werror = weightcuda:double() - groundweight:double()
      local berror = biascuda:double() - groundbias:double()

      mytester:assertlt(error:abs():max(), precision_backward_type(precision_backward, typename),
         string.format('error on state (backward) with %s', typename))
      mytester:assertlt(werror:abs():max(), precision_backward_conv_weightbias(precision_backward, typename),
         string.format('error on weight (backward) with %s', typename))
      mytester:assertlt(berror:abs():max(), precision_backward_conv_weightbias(precision_backward, typename),
         string.format('error on bias (backward) with %s', typename))
   end
end

function cunntest.SpatialDilatedConvolution_backward_batch()
   local bs = math.random(1,4) * 4
   local from = math.random(1,32)
   local to = math.random(1,8) * 8
   local ki = math.random(3,15)
   local kj = math.random(3,15)
   local si = math.random(1,3)
   local sj = math.random(1,3)
   local padW = math.random(0,1)
   local padH = math.random(0,1)
   local outi = math.random(ki, 64)
   local outj = math.random(kj, 64)
   local dilationW = math.random(0,10)
   local dilationH = math.random(0,10)
   local ini = (outi - 1) * si - 2 * padW + dilationW * (ki-1) + 1
   local inj = (outj - 1) * sj - 2 * padH + dilationH * (kj-1) + 1

   for k, typename in ipairs(typenames) do
      local input = torch.randn(bs,from,inj,ini):type(typename)

      local ctype = t2cpu[typename]
      input = input:type(ctype)
      local sconv = nn.SpatialDilatedConvolution(from,to,ki,kj,si,sj,padW,padH,dilationW,dilationH):type(ctype)
      local output = sconv:forward(input)
      local gradOutput = output:clone():normal()
      sconv:zeroGradParameters()
      local groundgrad = sconv:backward(input, gradOutput)
      local groundweight = sconv.gradWeight
      local groundbias = sconv.gradBias

      input = input:type(typename)
      gradOutput = gradOutput:type(typename)
      local gconv = nn.SpatialDilatedConvolution(from,to,ki,kj,si,sj,padW,padH,dilationW,dilationH):type(typename)
      gconv.weight = sconv.weight:type(typename)
      gconv.bias = sconv.bias:type(typename)
      gconv:forward(input)
      gconv:zeroGradParameters()
      local rescuda = gconv:backward(input, gradOutput)
      local weightcuda = gconv.gradWeight
      local biascuda = gconv.gradBias

      local error = rescuda:double() - groundgrad:double()
      local werror = weightcuda:double() - groundweight:double()
      local berror = biascuda:double() - groundbias:double()

      mytester:assertlt(error:abs():max(), precision_backward_type(precision_backward, typename),
         string.format('error on state (backward) with %s', typename))
      mytester:assertlt(werror:abs():max(), precision_backward_conv_weightbias(precision_backward, typename),
         string.format('error on weight (backward) with %s', typename))
      mytester:assertlt(berror:abs():max(), precision_backward_conv_weightbias(precision_backward, typename),
         string.format('error on bias (backward) with %s', typename))
   end
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

   for k, typename in ipairs(typenames) do
      local input = torch.randn(from,inj,ini):type(typename)

      local ctype = t2cpu[typename]
      input = input:type(ctype)
      local sconv = nn.SpatialSubSampling(from,ki,kj,si,sj):type(ctype)
      local groundtruth = sconv:forward(input)

      input = input:type(typename)
      local gconv = nn.SpatialSubSampling(from,ki,kj,si,sj):type(typename)
      gconv.weight = sconv.weight:type(typename)
      gconv.bias = sconv.bias:type(typename)
      local rescuda = gconv:forward(input)

      local error = rescuda:double() - groundtruth:double()
      mytester:assertlt(error:abs():max(), precision_forward_type(precision_forward, typename),
          string.format('error on state (forward) with %s', typename))
   end
end

function cunntest.Sampling_forward_batch()
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

   for k, typename in ipairs(typenames) do
      local input = torch.randn(bs,from,inj,ini):type(typename)

      local ctype = t2cpu[typename]
      input = input:type(ctype)
      local sconv = nn.SpatialSubSampling(from,ki,kj,si,sj):type(ctype)
      local groundtruth = sconv:forward(input)

      input = input:type(typename)
      local gconv = nn.SpatialSubSampling(from,ki,kj,si,sj):type(typename)
      gconv.weight = sconv.weight:type(typename)
      gconv.bias = sconv.bias:type(typename)
      local rescuda = gconv:forward(input)

      local error = rescuda:double() - groundtruth:double()
      mytester:assertlt(error:abs():max(), precision_forward_type(precision_forward, typename),
          string.format('error on state (forward) with %s', typename))
   end
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

   for k, typename in ipairs(typenames) do
      -- FIXME: SpatialSubSampling accumulates directly to real, causes
      -- precision issues with half
      precision_backward_old = precision_backward
      if typename == 'torch.CudaHalfTensor' then
          precision_backward = 0.4
      end
      local input = torch.randn(from,inj,ini):type(typename)
      local gradOutput = torch.randn(to,outj,outi):type(typename)

      local ctype = t2cpu[typename]
      input = input:type(ctype)
      gradOutput = gradOutput:type(ctype)
      local sconv = nn.SpatialSubSampling(from,ki,kj,si,sj):type(ctype)
      sconv:forward(input)
      sconv:zeroGradParameters()
      local groundgrad = sconv:backward(input, gradOutput)
      local groundweight = sconv.gradWeight
      local groundbias = sconv.gradBias

      input = input:type(typename)
      gradOutput = gradOutput:type(typename)
      local gconv = nn.SpatialSubSampling(from,ki,kj,si,sj):type(typename)
      gconv.weight = sconv.weight:type(typename)
      gconv.bias = sconv.bias:type(typename)
      gconv:forward(input)
      gconv:zeroGradParameters()
      local rescuda = gconv:backward(input, gradOutput)
      local weightcuda = gconv.gradWeight
      local biascuda = gconv.gradBias

      local error = rescuda:double() - groundgrad:double()
      local werror = weightcuda:double() - groundweight:double()
      local berror = biascuda:double() - groundbias:double()

      mytester:assertlt(error:abs():max(), precision_backward_type(precision_backward, typename),
          string.format('error on state (backward) with %s', typename))
      mytester:assertlt(werror:abs():max(), precision_backward_type(precision_backward, typename),
          string.format('error on weight (backward) with %s', typename))
      mytester:assertlt(berror:abs():max(), precision_backward_type(precision_backward, typename),
          string.format('error on bias (backward) with %s', typename))

      precision_backward = precision_backward_old
   end
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

   for k, typename in ipairs(typenames) do
      -- FIXME: SpatialSubSampling accumulates directly to real, causes
      -- precision issues with half
      precision_backward_old = precision_backward
      if typename == 'torch.CudaHalfTensor' then
        precision_backward = 0.7
      end
      local input = torch.randn(bs,from,inj,ini):type(typename)
      local gradOutput = torch.randn(bs,to,outj,outi):type(typename)

      local ctype = t2cpu[typename]
      input = input:type(ctype)
      gradOutput = gradOutput:type(ctype)
      local sconv = nn.SpatialSubSampling(from,ki,kj,si,sj):type(ctype)
      sconv:forward(input)
      sconv:zeroGradParameters()
      local groundgrad = sconv:backward(input, gradOutput)
      local groundweight = sconv.gradWeight
      local groundbias = sconv.gradBias

      input = input:type(typename)
      gradOutput = gradOutput:type(typename)
      local gconv = nn.SpatialSubSampling(from,ki,kj,si,sj):type(typename)
      gconv.weight = sconv.weight:type(typename)
      gconv.bias = sconv.bias:type(typename)
      gconv:forward(input)
      gconv:zeroGradParameters()
      local rescuda = gconv:backward(input, gradOutput)
      local weightcuda = gconv.gradWeight
      local biascuda = gconv.gradBias

      local error = rescuda:double() - groundgrad:double()
      local werror = weightcuda:double() - groundweight:double()
      local berror = biascuda:double() - groundbias:double()

      mytester:assertlt(error:abs():max(), precision_backward_type(precision_backward, typename),
          string.format('error on state (backward) with %s', typename))
      mytester:assertlt(werror:abs():max(), precision_backward_type(precision_backward, typename),
          string.format('error on weight (backward) with %s', typename))
      mytester:assertlt(berror:abs():max(), precision_backward_type(precision_backward, typename),
          string.format('error on bias (backward) with %s', typename))

      precision_backward = precision_backward_old
   end
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

   for k, typename in ipairs(typenames) do
      local input = torch.randn(from,inj,ini):type(typename)
      local ctype = t2cpu[typename]
      input = input:type(ctype)
      local sconv = nn.SpatialMaxPooling(ki,kj,si,sj,padi,padj):type(ctype)
      if ceil_mode then sconv:ceil() end
      local groundtruth = sconv:forward(input)

      input = input:type(typename)
      local gconv = nn.SpatialMaxPooling(ki,kj,si,sj,padi,padj):type(typename)
      if ceil_mode then gconv:ceil() end
      local rescuda = gconv:forward(input)

      local error = rescuda:double() - groundtruth:double()
      mytester:assertlt(error:abs():max(), precision_forward_type(precision_forward, typename),
          string.format('error on state (forward) with %s', typename))
      local error_ind = gconv.indices:long() - sconv.indices
      mytester:asserteq(error_ind:max(), 0,
          string.format('error on indices (forward) with %s', typename))
    end
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

   for k, typename in ipairs(typenames) do
      local input = torch.randn(bs,from,inj,ini):type(typename)
      local ctype = t2cpu[typename]
      input = input:type(ctype)
      local sconv = nn.SpatialMaxPooling(ki,kj,si,sj,padi,padj):type(ctype)
      if ceil_mode then sconv:ceil() end
      local groundtruth = sconv:forward(input)

      input = input:type(typename)
      local gconv = nn.SpatialMaxPooling(ki,kj,si,sj,padi,padj):type(typename)
      if ceil_mode then gconv:ceil() end
      local rescuda = gconv:forward(input)

      local error = rescuda:double() - groundtruth:double()
      mytester:assertlt(error:abs():max(), precision_forward_type(precision_forward, typename),
          string.format('error on state (forward) with %s', typename))
    end
end

function cunntest.SpatialMaxUnpooling_forward_batch()
   local bs = math.random(4,10)
   local from = math.random(1,64)
   local to = from
   local ki = math.random(2,4)
   local kj = math.random(2,4)
   local si = ki
   local sj = kj
   local outi = math.random(32,256)
   local outj = math.random(32,256)
   local padi = math.random(0,ki/2-1)
   local padj = math.random(0,kj/2-1)
   local ceil_mode = math.random(0,1) == 1
   local fun = ceil_mode and torch.ceil or torch.floor
   local ini = fun((outi + padi*2 - ki)/si) +1
   local inj = fun((outj + padj*2 - kj)/sj) +1

   for k, typename in ipairs(typenames) do
      local ctype = t2cpu[typename]
      local pooler = nn.SpatialMaxPooling(ki,kj,si,sj,padi,padj):type(ctype)
      if ceil_mode then pooler:ceil() end
      local sunpool = nn.SpatialMaxUnpooling(pooler):type(ctype)

      local original = torch.randn(bs,from,outj,outi):type(typename)
      original = original:type(ctype)
      local input = pooler:forward(original)
      local groundtruth = sunpool:forward(input)

      original = original:type(typename)
      pooler = nn.SpatialMaxPooling(ki,kj,si,sj,padi,padj):type(typename)
      if ceil_mode then pooler:ceil() end
      local gunpool = nn.SpatialMaxUnpooling(pooler):type(typename)

      input = pooler:forward(original)
      local rescuda = gunpool:forward(input)

      local error = rescuda:double() - groundtruth:double()
      mytester:assertlt(error:abs():max(), precision_forward_type(precision_forward, typename),
          string.format('error on state (forward) with %s', typename))
    end
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
   local ceil_mode = true--math.random(0,1) == 1

   for k, typename in ipairs(typenames) do
      local input = torch.randn(from,inj,ini):type(typename)
      local gradOutput = torch.randn(to,outj,outi):type(typename)
      local ctype = t2cpu[typename]
      input = input:type(ctype)
      gradOutput = gradOutput:type(ctype)

      local sconv = nn.SpatialMaxPooling(ki,kj,si,sj,padi,padj):type(ctype)
      if ceil_mode then sconv:ceil() end
      sconv:forward(input)
      sconv:zeroGradParameters()
      local groundgrad = sconv:backward(input, gradOutput)

      input = input:type(typename)
      gradOutput = gradOutput:type(typename)
      local gconv = nn.SpatialMaxPooling(ki,kj,si,sj,padi,padj):type(typename)
      if ceil_mode then gconv:ceil() end
      gconv:forward(input)
      gconv:zeroGradParameters()
      local rescuda = gconv:backward(input, gradOutput)

      local error = rescuda:double() - groundgrad:double()

      mytester:assertlt(error:abs():max(), precision_backward_type(precision_backward, typename),
          string.format('error on state (backward) with %s', typename))
    end
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

   for k, typename in ipairs(typenames) do
      local input = torch.randn(bs,from,inj,ini):type(typename)
      local gradOutput = torch.randn(bs,to,outj,outi):type(typename)
      local ctype = t2cpu[typename]
      local input = input:type(ctype)
      local gradOutput = gradOutput:type(ctype)
      local sconv = nn.SpatialMaxPooling(ki,kj,si,sj,padi,padj):type(ctype)
      if ceil_mode then sconv:ceil() end
      sconv:forward(input)
      sconv:zeroGradParameters()
      local groundgrad = sconv:backward(input, gradOutput)

      input = input:type(typename)
      gradOutput = gradOutput:type(typename)
      local gconv = nn.SpatialMaxPooling(ki,kj,si,sj,padi,padj):type(typename)
      if ceil_mode then gconv:ceil() end
      gconv:forward(input)
      gconv:zeroGradParameters()
      local rescuda = gconv:backward(input, gradOutput)

      local error = rescuda:double() - groundgrad:double()

      mytester:assertlt(error:abs():max(), precision_backward_type(precision_backward, typename),
          string.format('error on state (backward) with %s', typename))
    end
end

function cunntest.SpatialMaxUnpooling_backward_batch()
   local bs = math.random(4,10)
   local from = math.random(1,64)
   local to = from
   local ki = math.random(2,4)
   local kj = math.random(2,4)
   local si = ki
   local sj = kj
   local outi = math.random(32,256)
   local outj = math.random(32,256)
   local padi = math.random(0,ki/2-1)
   local padj = math.random(0,kj/2-1)
   local ceil_mode = math.random(0,1) == 1
   local fun = ceil_mode and torch.ceil or torch.floor
   local ini = fun((outi + padi*2 - ki)/si) +1
   local inj = fun((outj + padj*2 - kj)/sj) +1

   for k, typename in ipairs(typenames) do
      local ctype = t2cpu[typename]
      local pooler = nn.SpatialMaxPooling(ki,kj,si,sj,padi,padj):type(ctype)
      if ceil_mode then pooler:ceil() end
      local sunpool = nn.SpatialMaxUnpooling(pooler):type(ctype)

      local original = torch.randn(bs,from,outj,outi):type(typename)
      original = original:type(ctype)
      local input = pooler:forward(original)
      local gradOutput = torch.randn(original:size()):type(typename)
      gradOutput = gradOutput:type(ctype)
      sunpool:forward(input)
      sunpool:zeroGradParameters()
      local groundgrad = sunpool:backward(input, gradOutput)

      pooler = nn.SpatialMaxPooling(ki,kj,si,sj,padi,padj):type(typename)
      if ceil_mode then pooler:ceil() end
      local gunpool = nn.SpatialMaxUnpooling(pooler):type(typename)

      original = original:type(typename)
      input = pooler:forward(original)
      gunpool:forward(input)

      gradOutput = gradOutput:type(typename)
      gunpool:zeroGradParameters()
      local rescuda = gunpool:backward(input, gradOutput)

      local error = rescuda:double() - groundgrad:double()

      mytester:assertlt(error:abs():max(), precision_backward_type(precision_backward, typename),
          string.format('error on state (backward) with %s', typename))
    end
end

function cunntest.SpatialDilatedMaxPooling_forward()
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
   local dilationi = math.random(1,10)
   local dilationj = math.random(1,10)
   local ini = (outi-1)*si+(dilationi*(ki-1)+1)-2*padi
   local inj = (outj-1)*sj+(dilationj*(kj-1)+1)-2*padj
   local ceil_mode = math.random(0,1) == 1

   for k, typename in ipairs(typenames) do
      local input = torch.randn(from,inj,ini):type(typename)
      local ctype = t2cpu[typename]
      input = input:type(ctype)
      local sconv = nn.SpatialDilatedMaxPooling(ki,kj,si,sj,padi,padj,dilationi,dilationj):type(ctype)
      if ceil_mode then sconv:ceil() end
      local groundtruth = sconv:forward(input)

      input = input:type(typename)
      local gconv = nn.SpatialDilatedMaxPooling(ki,kj,si,sj,padi,padj,dilationi,dilationj):type(typename)
      if ceil_mode then gconv:ceil() end
      local rescuda = gconv:forward(input)

      local error = rescuda:double() - groundtruth:double()
      mytester:assertlt(error:abs():max(), precision_forward_type(precision_forward, typename),
          string.format('error on state (forward) with %s', typename))
      local error_ind = gconv.indices:long() - sconv.indices
      mytester:asserteq(error_ind:max(), 0,
          string.format('error on indices (forward) with %s', typename))
    end
end

function cunntest.SpatialDilatedMaxPooling_forward_batch()
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
   local dilationi = math.random(1,10)
   local dilationj = math.random(1,10)
   local ini = (outi-1)*si+(dilationi*(ki-1)+1)-2*padi
   local inj = (outj-1)*sj+(dilationj*(kj-1)+1)-2*padj
   local ceil_mode = math.random(0,1) == 1

   for k, typename in ipairs(typenames) do
      local input = torch.randn(bs,from,inj,ini):type(typename)
      local ctype = t2cpu[typename]
      input = input:type(ctype)
      local sconv = nn.SpatialDilatedMaxPooling(ki,kj,si,sj,padi,padj,dilationi,dilationj):type(ctype)
      if ceil_mode then sconv:ceil() end
      local groundtruth = sconv:forward(input)

      input = input:type(typename)
      local gconv = nn.SpatialDilatedMaxPooling(ki,kj,si,sj,padi,padj,dilationi,dilationj):type(typename)
      if ceil_mode then gconv:ceil() end
      local rescuda = gconv:forward(input)

      local error = rescuda:double() - groundtruth:double()
      mytester:assertlt(error:abs():max(), precision_forward_type(precision_forward, typename),
          string.format('error on state (forward) with %s', typename))
    end
end

function cunntest.SpatialDilatedMaxPooling_backward()
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
   local dilationi = math.random(1,10)
   local dilationj = math.random(1,10)
   local ini = (outi-1)*si+(dilationi*(ki-1)+1)-2*padi
   local inj = (outj-1)*sj+(dilationj*(kj-1)+1)-2*padj
   local ceil_mode = math.random(0,1) == 1

   for k, typename in ipairs(typenames) do
      local input = torch.randn(from,inj,ini):type(typename)
      local gradOutput = torch.randn(to,outj,outi):type(typename)
      local ctype = t2cpu[typename]
      input = input:type(ctype)
      gradOutput = gradOutput:type(ctype)
      local sconv = nn.SpatialDilatedMaxPooling(ki,kj,si,sj,padi,padj,dilationi,dilationj):type(ctype)
      if ceil_mode then sconv:ceil() end
      sconv:forward(input)
      sconv:zeroGradParameters()
      local groundgrad = sconv:backward(input, gradOutput)

      input = input:type(typename)
      gradOutput = gradOutput:type(typename)
      local gconv = nn.SpatialDilatedMaxPooling(ki,kj,si,sj,padi,padj,dilationi,dilationj):type(typename)
      if ceil_mode then gconv:ceil() end
      gconv:forward(input)
      gconv:zeroGradParameters()
      local rescuda = gconv:backward(input, gradOutput)

      local error = rescuda:double() - groundgrad:double()

      mytester:assertlt(error:abs():max(), precision_backward_type(precision_backward, typename),
          string.format('error on state (backward) with %s', typename))
    end
end

function cunntest.SpatialDilatedMaxPooling_backward_batch()
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
   local dilationi = math.random(1,10)
   local dilationj = math.random(1,10)
   local ini = (outi-1)*si+(dilationi*(ki-1)+1)-2*padi
   local inj = (outj-1)*sj+(dilationj*(kj-1)+1)-2*padj
   local ceil_mode = math.random(0,1) == 1

   for k, typename in ipairs(typenames) do
      local input = torch.randn(bs,from,inj,ini):type(typename)
      local gradOutput = torch.randn(bs,to,outj,outi):type(typename)
      local ctype = t2cpu[typename]
      input = input:type(ctype)
      gradOutput = gradOutput:type(ctype)
      local sconv = nn.SpatialDilatedMaxPooling(ki,kj,si,sj,padi,padj,dilationi,dilationj):type(ctype)
      if ceil_mode then sconv:ceil() end
      sconv:forward(input)
      sconv:zeroGradParameters()
      local groundgrad = sconv:backward(input, gradOutput)

      input = input:type(typename)
      gradOutput = gradOutput:type(typename)
      local gconv = nn.SpatialDilatedMaxPooling(ki,kj,si,sj,padi,padj,dilationi,dilationj):type(typename)
      if ceil_mode then gconv:ceil() end
      gconv:forward(input)
      gconv:zeroGradParameters()
      local rescuda = gconv:backward(input, gradOutput)

      local error = rescuda:double() - groundgrad:double()

      mytester:assertlt(error:abs():max(), precision_backward_type(precision_backward, typename),
          string.format('error on state (backward) with %s', typename))
    end
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

    for k, typename in ipairs(typenames) do
        local input = nil
        if batch == 1 then
            input = torch.Tensor(plane, inH, inW):uniform():type(typename)
        else
            input = torch.Tensor(batch, plane, inH, inW):uniform():type(typename)
        end

        local ctype = t2cpu[typename]
        input = input:type(ctype)
        local module = nil
        if useRatio then
            module =
                nn.SpatialFractionalMaxPooling(poolSizeW, poolSizeH, ratioW, ratioH):type(ctype)
        else
            module =
                nn.SpatialFractionalMaxPooling(poolSizeW, poolSizeH, outW, outH):type(ctype)
        end

        module:fixPoolingRegions()

        local groundtruth = module:forward(input)

        input = input:type(typename)

        local gmodule = nil
        if useRatio then
            gmodule =
                nn.SpatialFractionalMaxPooling(poolSizeW, poolSizeH, ratioW, ratioH)
        else
            gmodule =
                nn.SpatialFractionalMaxPooling(poolSizeW, poolSizeH, outW, outH)
        end

        gmodule = gmodule:fixPoolingRegions():type(typename)

        -- For comparison purposes, make sure we are using the same random pooling regions
        -- as the CPU
        gmodule.randomSamples = module.randomSamples:type(typename)

        local rescuda = gmodule:forward(input)

        local error = rescuda:double() - groundtruth:double()
        mytester:assertlt(error:abs():max(), precision_forward_type(precision_forward, typename),
            string.format('error on state (forward) with %s', typename))
        local error_ind = gmodule.indices:long() - module.indices
        mytester:asserteq(error_ind:abs():max(), 0,
            string.format('error on indices (forward) with %s', typename))
    end
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

    for k, typename in ipairs(typenames) do
        local input = nil
        local gradOutput = nil
        if batch == 1 then
            input = torch.Tensor(plane, inH, inW):uniform():type(typename)
            gradOutput = torch.Tensor(plane, outH, outW):uniform():type(typename)
        else
            input = torch.Tensor(batch, plane, inH, inW):uniform():type(typename)
            gradOutput = torch.Tensor(batch, plane, outH, outW):uniform():type(typename)
        end

        local ctype = t2cpu[typename]
        input = input:type(ctype)
        gradOutput = gradOutput:type(ctype)
        local module =
            nn.SpatialFractionalMaxPooling(poolSizeW, poolSizeH, outW, outH)
            :fixPoolingRegions():type(ctype)

        module:forward(input)
        module:zeroGradParameters()
        local groundgrad = module:backward(input, gradOutput)

        input = input:type(typename)
        gradOutput = gradOutput:type(typename)

        local gmodule =
            nn.SpatialFractionalMaxPooling(poolSizeW, poolSizeH, outW, outH)
            :fixPoolingRegions():type(typename)
        -- For comparison purposes, make sure we are using the same random pooling regions
        -- as the CPU
        gmodule.randomSamples = module.randomSamples:type(typename)

        gmodule:forward(input)
        gmodule:zeroGradParameters()
        local rescuda = gmodule:backward(input, gradOutput)

        local error = rescuda:double() - groundgrad:double()
        mytester:assertlt(error:abs():max(), precision_backward_type(precision_backward, typename),
            string.format('error on state (backward) with %s', typename))
    end
end

function cunntest.SpatialAveragePooling_includepad()
   for k, typename in ipairs(typenames) do
      local net = nn.SpatialAveragePooling(2, 2, 1, 1, 1, 1):type(typename)
      local net_no_include_pad = net:clone()
      net_no_include_pad:setCountExcludePad()
      local net_include_pad = net:clone()
      net_include_pad:setCountIncludePad()

      local input = torch.FloatTensor(1, 1, 1, 1):type(typename)
      input[1][1][1][1] = 3
      local out_noinclude = net_no_include_pad:forward(input)
      local out_include = net_include_pad:forward(input)

      local noinc_out = out_noinclude[1][1][1][1]
      local inc_out = out_include[1][1][1][1]
      mytester:assertne(noinc_out, inc_out)
      mytester:asserteq(3, noinc_out)
      mytester:asserteq(3/4, inc_out)
   end
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
   local padi = math.random(0,ki/2-1)
   local padj = math.random(0,kj/2-1)
   local ini = (outi-1)*si+ki - padi*2
   local inj = (outj-1)*sj+kj - padj*2
   local ceil_mode = math.random(0,1) == 1
   local count_exclude_pad = math.random(0,1) == 1

   for k, typename in ipairs(typenames) do
      local input = torch.randn(from,inj,ini):type(typename)

      local ctype = t2cpu[typename]
      input = input:type(ctype)
      local sconv = nn.SpatialAveragePooling(ki,kj,si,sj,padi,padj):type(ctype)
      if ceil_mode then sconv:ceil() end
      if count_exclude_pad then sconv:setCountExcludePad() end
      local groundtruth = sconv:forward(input)

      input = input:type(typename)
      local gconv = nn.SpatialAveragePooling(ki,kj,si,sj,padi,padj):type(typename)
      if ceil_mode then gconv:ceil() end
      if count_exclude_pad then gconv:setCountExcludePad() end
      local rescuda = gconv:forward(input)

      local error = rescuda:double() - groundtruth:double()
      mytester:assertlt(error:abs():max(), precision_forward_type(precision_forward, typename),
          string.format('error on state (forward) with %s', typename))
   end
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
   local padi = math.random(0,ki/2-1)
   local padj = math.random(0,kj/2-1)
   local ini = (outi-1)*si+ki - padi*2
   local inj = (outj-1)*sj+kj - padj*2
   local ceil_mode = math.random(0,1) == 1
   local count_exclude_pad = math.random(0,1) == 1

   for k, typename in ipairs(typenames) do
      local input = torch.randn(bs,from,inj,ini):type(typename)
      local ctype = t2cpu[typename]

      input = input:type(ctype)
      local sconv = nn.SpatialAveragePooling(ki,kj,si,sj,padi,padj):type(ctype)
      if ceil_mode then sconv:ceil() end
      if count_exclude_pad then sconv:setCountExcludePad() end
      local groundtruth = sconv:forward(input)

      input = input:type(typename)
      local gconv = nn.SpatialAveragePooling(ki,kj,si,sj,padi,padj):type(typename)
      if ceil_mode then gconv:ceil() end
      if count_exclude_pad then gconv:setCountExcludePad() end
      local rescuda = gconv:forward(input)

      local error = rescuda:double() - groundtruth:double()
      mytester:assertlt(error:abs():max(), precision_forward_type(precision_forward, typename),
          string.format('error on state (forward) with %s', typename))
   end
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
   local padi = math.random(0,ki/2-1)
   local padj = math.random(0,kj/2-1)
   local ini = (outi-1)*si+ki - padi*2
   local inj = (outj-1)*sj+kj - padj*2
   local ceil_mode = math.random(0,1) == 1
   local count_exclude_pad = math.random(0,1) == 1

   for k, typename in ipairs(typenames) do
      local input = torch.randn(from,inj,ini):type(typename)
      local gradOutput = torch.randn(to,outj,outi):type(typename)

      local ctype = t2cpu[typename]
      input = input:type(ctype)
      gradOutput = gradOutput:type(ctype)
      local sconv = nn.SpatialAveragePooling(ki,kj,si,sj,padi,padj):type(ctype)
      if ceil_mode then sconv:ceil() end
      if count_exclude_pad then sconv:setCountExcludePad() end
      sconv:forward(input)
      sconv:zeroGradParameters()
      local groundgrad = sconv:backward(input, gradOutput)

      input = input:type(typename)
      gradOutput = gradOutput:type(typename)
      local gconv = nn.SpatialAveragePooling(ki,kj,si,sj,padi,padj):type(typename)
      if ceil_mode then gconv:ceil() end
      if count_exclude_pad then gconv:setCountExcludePad() end
      gconv:forward(input)
      gconv:zeroGradParameters()
      local rescuda = gconv:backward(input, gradOutput)

      local error = rescuda:double() - groundgrad:double()

      mytester:assertlt(error:abs():max(), precision_backward_type(precision_backward, typename),
          string.format('error on state (backward) with %s', typename))
   end
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
   local padi = math.random(0,ki/2-1)
   local padj = math.random(0,kj/2-1)
   local ini = (outi-1)*si+ki - padi*2
   local inj = (outj-1)*sj+kj - padj*2
   local ceil_mode = math.random(0,1) == 1
   local count_exclude_pad = math.random(0,1) == 1

   for k, typename in ipairs(typenames) do
      local input = torch.randn(bs,from,inj,ini):type(typename)
      local gradOutput = torch.randn(bs,to,outj,outi):type(typename)

      local ctype = t2cpu[typename]
      input = input:type(ctype)
      gradOutput = gradOutput:type(ctype)
      local sconv = nn.SpatialAveragePooling(ki,kj,si,sj,padi,padj):type(ctype)
      if ceil_mode then sconv:ceil() end
      if count_exclude_pad then sconv:setCountExcludePad() end
      sconv:forward(input)
      sconv:zeroGradParameters()
      local groundgrad = sconv:backward(input, gradOutput)

      input = input:type(typename)
      gradOutput = gradOutput:type(typename)
      local gconv = nn.SpatialAveragePooling(ki,kj,si,sj,padi,padj):type(typename)
      if ceil_mode then gconv:ceil() end
      if count_exclude_pad then gconv:setCountExcludePad() end
      gconv:forward(input)
      gconv:zeroGradParameters()
      local rescuda = gconv:backward(input, gradOutput)

      local error = rescuda:double() - groundgrad:double()

      mytester:assertlt(error:abs():max(), precision_backward_type(precision_backward, typename),
          string.format('error on state (backward) with %s', typename))
   end
end

function cunntest.SpatialAdaptiveMaxPooling_forward()
   local from = math.random(1,64)
   local to = from
   local outi = math.random(2,64)
   local outj = math.random(2,64)
   local ini = math.random(10,256)
   local inj = math.random(10,256)

   for k, typename in ipairs(typenames) do
      local input = torch.randn(from,inj,ini):type(typename)
      local ctype = t2cpu[typename]
      input = input:type(ctype)
      local sconv = nn.SpatialAdaptiveMaxPooling(outi,outj):type(ctype)
      local groundtruth = sconv:forward(input):type(ctype)

      input = input:type(typename)
      local gconv = nn.SpatialAdaptiveMaxPooling(outi,outj):type(typename)
      local rescuda = gconv:forward(input)

      local error = rescuda:double() - groundtruth:double()
      mytester:assertlt(error:abs():max(), precision_forward_type(precision_forward, typename),
          string.format('error on state (forward) with %s', typename))
      local error_ind = gconv.indices:long() - sconv.indices
      mytester:asserteq(error_ind:max(), 0,
          string.format('error on indices (forward) with %s', typename))
   end
end

function cunntest.SpatialAdaptiveMaxPooling_forward_noncontig()
   local from = math.random(1,64)
   local to = from
   local outi = math.random(2,64)
   local outj = math.random(2,64)
   local ini = math.random(10,256)
   local inj = math.random(10,256)

   for k, typename in ipairs(typenames) do
      local input0 = torch.randn(from,ini,inj):type(typename)
      local ctype = t2cpu[typename]
      local input = input0:type(ctype):transpose(2,3)
      local sconv = nn.SpatialAdaptiveMaxPooling(outi,outj):type(ctype)
      local groundtruth = sconv:forward(input)

      input = input0:type(typename):transpose(2,3)
      local gconv = nn.SpatialAdaptiveMaxPooling(outi,outj):type(typename)
      local rescuda = gconv:forward(input)

      local error = rescuda:double() - groundtruth:double()
      mytester:assertlt(error:abs():max(), precision_forward_type(precision_forward, typename),
          string.format('error on state (forward) with %s', typename))
      local error_ind = gconv.indices:long() - sconv.indices
      mytester:asserteq(error_ind:max(), 0,
          string.format('error on indices (forward) with %s', typename))
   end
end

function cunntest.SpatialAdaptiveMaxPooling_forward_batch()
   local bs = math.random(4,10)
   local from = math.random(1,48)
   local to = from
   local outi = math.random(2,48)
   local outj = math.random(2,48)
   local ini = math.random(10,256)
   local inj = math.random(10,256)

   for k, typename in ipairs(typenames) do
      local input = torch.randn(bs,from,inj,ini):type(typename)
      local ctype = t2cpu[typename]
      input = input:type(ctype)
      local sconv = nn.SpatialAdaptiveMaxPooling(outi,outj):type(ctype)
      local groundtruth = sconv:forward(input)

      input = input:type(typename)
      local gconv = nn.SpatialAdaptiveMaxPooling(outi,outj):type(typename)
      local rescuda = gconv:forward(input)

      local error = rescuda:double() - groundtruth:double()
      mytester:assertlt(error:abs():max(), precision_forward_type(precision_forward, typename),
          string.format('error on state (forward) with %s', typename))
   end
end

function cunntest.SpatialAdaptiveMaxPooling_backward()
   local from = math.random(1,64)
   local to = from
   local outi = math.random(2,64)
   local outj = math.random(2,64)
   local ini = math.random(10,256)
   local inj = math.random(10,256)

   for k, typename in ipairs(typenames) do
      local input = torch.randn(from,inj,ini):type(typename)
      local gradOutput = torch.randn(to,outj,outi):type(typename)
      local ctype = t2cpu[typename]
      input = input:type(ctype)
      gradOutput = gradOutput:type(ctype)
      local sconv = nn.SpatialAdaptiveMaxPooling(outi,outj):type(ctype)
      sconv:forward(input)
      sconv:zeroGradParameters()
      local groundgrad = sconv:backward(input, gradOutput)

      input = input:type(typename)
      gradOutput = gradOutput:type(typename)
      local gconv = nn.SpatialAdaptiveMaxPooling(outi,outj):type(typename)
      gconv:forward(input)
      gconv:zeroGradParameters()
      local rescuda = gconv:backward(input, gradOutput)

      local error = rescuda:double() - groundgrad:double()

      mytester:assertlt(error:abs():max(), precision_backward_type(precision_backward, typename),
          string.format('error on state (backward) with %s', typename))
   end
end

function cunntest.SpatialAdaptiveMaxPooling_backward_noncontig()
   local from = math.random(1,64)
   local to = from
   local outi = math.random(2,64)
   local outj = math.random(2,64)
   local ini = math.random(10,256)
   local inj = math.random(10,256)

   for k, typename in ipairs(typenames) do
      local input0 = torch.randn(from,ini,inj):type(typename)
      local gradOutput = torch.randn(to,outj,outi):type(typename)
      local ctype = t2cpu[typename]
      local input = input0:type(ctype):transpose(2,3)
      gradOutput = gradOutput:type(ctype)
      local sconv = nn.SpatialAdaptiveMaxPooling(outi,outj):type(ctype)
      sconv:forward(input)
      sconv:zeroGradParameters()
      local groundgrad = sconv:backward(input, gradOutput)

      input = input0:type(typename):transpose(2,3)
      gradOutput = gradOutput:type(typename)
      local gconv = nn.SpatialAdaptiveMaxPooling(outi,outj):type(typename)
      gconv:forward(input)
      gconv:zeroGradParameters()
      local rescuda = gconv:backward(input, gradOutput)

      local error = rescuda:double() - groundgrad:double()

      mytester:assertlt(error:abs():max(), precision_backward_type(precision_backward, typename),
          string.format('error on state (backward) with %s', typename))
   end
end

function cunntest.SpatialAdaptiveMaxPooling_backward_batch()
   local bs = math.random(4,10)
   local from = math.random(1,64)
   local to = from
   local outi = math.random(2,64)
   local outj = math.random(2,64)
   local ini = math.random(10,256)
   local inj = math.random(10,256)

   for k, typename in ipairs(typenames) do
      local input = torch.randn(bs,from,inj,ini):type(typename)
      local gradOutput = torch.randn(bs,to,outj,outi):type(typename)
      local ctype = t2cpu[typename]
      input = input:type(ctype)
      gradOutput = gradOutput:type(ctype)
      local sconv = nn.SpatialAdaptiveMaxPooling(outi,outj):type(ctype)
      sconv:forward(input)
      sconv:zeroGradParameters()
      local groundgrad = sconv:backward(input, gradOutput)

      input = input:type(typename)
      gradOutput = gradOutput:type(typename)
      local gconv = nn.SpatialAdaptiveMaxPooling(outi,outj):type(typename)
      gconv:forward(input)
      gconv:zeroGradParameters()
      local rescuda = gconv:backward(input, gradOutput)

      local error = rescuda:double() - groundgrad:double()

      mytester:assertlt(error:abs():max(), precision_backward_type(precision_backward, typename),
          string.format('error on state (backward) with %s', typename))
   end
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

local function BCECriterion_forward_truth(buffer, input, target, weights, sizeAverage)

  local eps = 1e-12
  local output

  buffer:resizeAs(input)

  if weights ~= nil and target:dim() ~= 1 then
    weights = weights:view(1, target:size(2)):expandAs(target)
  end

  -- log(input) * target
  buffer:add(input, eps):log()
  if weights ~= nil then buffer:cmul(weights) end

  output = torch.dot(target, buffer)

  -- log(1 - input) * (1 - target)
  buffer:mul(input, -1):add(1):add(eps):log()
  if weights ~= nil then buffer:cmul(weights) end

  output = output + torch.sum(buffer)
  output = output - torch.dot(target, buffer)

  if sizeAverage then
    output = output / input:nElement()
  end

  output = - output

  return output

end

function cunntest.BCECriterion_forward()
  local size = math.random(1,100)
  local input = torch.Tensor(size):uniform()
  local target = torch.Tensor(size):uniform():gt(0.5):type(torch.type(input))

  local tm = {}
  local title = string.format('BCECriterion.forward, Size: %d', size)
  times[title] = tm

  local crit = nn.BCECriterion()
  local rescpu = crit:forward(input, target)
  local a = torch.Timer()
  for i = 1,nloop do
     rescpu = crit:forward(input, target)
  end
  tm.cpu = a:time().real

  input = input:cuda()
  target = target:cuda()
  local g_crit = nn.BCECriterion():cuda()
  local rescuda = g_crit:forward(input, target)
  a:reset()
  for i = 1,nloop do
     rescuda = g_crit:forward(input, target)
  end
  cutorch.synchronize()
  tm.gpu = a:time().real
  local errorVal = rescuda - rescpu
  mytester:assertlt(errorVal, precision_forward, 'error on state (forward) ')

  -- test vs lua implementation
  buffer = input.new()
  local restruth = BCECriterion_forward_truth(buffer, input, target, nil, true)
  for i = 1,nloop do
    local restruth = BCECriterion_forward_truth(buffer, input, target, nil, true)
  end
  errorVal = rescpu - restruth
  mytester:assertlt(errorVal, precision_forward, 'error on state (forward) ')
  errorVal = rescuda - restruth
  mytester:assertlt(errorVal, precision_forward, 'error on state (forward) ')
end


function cunntest.BCECriterionWeights_forward()
  local size = math.random(1,100)
  local input = torch.Tensor(size):uniform()
  local target = torch.Tensor(size):uniform():gt(0.5):type(torch.type(input))
  local weights = torch.Tensor(size):uniform()

  local tm = {}
  local title = string.format('BCECriterionWeights.forward, Size: %d', size)
  times[title] = tm

  local crit = nn.BCECriterion(weights)
  local rescpu = crit:forward(input, target)
  local a = torch.Timer()
  for i = 1,nloop do
    rescpu = crit:forward(input, target)
  end
  tm.cpu = a:time().real

  input = input:cuda()
  target = target:cuda()
  weights = weights:cuda()
  local g_crit = nn.BCECriterion(weights):cuda()
  local rescuda = g_crit:forward(input, target)
  a:reset()
  for i = 1,nloop do
    rescuda = g_crit:forward(input, target)
  end
  cutorch.synchronize()
  tm.gpu = a:time().real
  local errorVal = rescuda - rescpu
  mytester:assertlt(errorVal, precision_forward, 'error on state (forward) ')

  -- test vs lua implementation
  buffer = input.new()
  local restruth = BCECriterion_forward_truth(buffer, input, target, weights, true)
  for i = 1,nloop do
    local restruth = BCECriterion_forward_truth(buffer, input, target, weights, true)
  end
  errorVal = rescpu - restruth
  mytester:assertlt(errorVal, precision_forward, 'error on state (forward) ')
  errorVal = rescuda - restruth
  mytester:assertlt(errorVal, precision_forward, 'error on state (forward) ')
end


function cunntest.MarginCriterion_forward()
  local size = math.random(1,100)

  for k, typename in ipairs(typenames) do
    local input = ((torch.rand(size)-0.5) * 2):type(typename) -- data spread from -1 to 1
    local target = ((torch.round(torch.rand(size))*2)-1):type(typename) -- generate random labels -1, 1

    local ctype = t2cpu[typename]
    input = input:type(ctype)
    target = input:type(ctype)
    local crit = nn.MarginCriterion():type(ctype)
    local groundtruth= crit:forward(input, target)

    input = input:type(typename)
    target = target:type(typename)
    local g_crit = nn.MarginCriterion():type(typename)
    local rescuda = g_crit:forward(input, target)
    local errorVal = rescuda - groundtruth
    mytester:assertlt(errorVal, precision_forward_type(precision_forward, typename),
        string.format('error on state (forward) with %s', typename))
  end
end

function cunntest.MultiLabelMarginCriterion_forward()
  local size = math.random(1,100)

  for k, typename in ipairs(typenames) do
     local input = ((torch.rand(size)-0.5) * 2):type(typename) -- data spread from -1 to 1
     local target = torch.round(torch.rand(size)*(size-1)):add(1) -- generate random labels > 0
     local zero = math.random(0,size) -- turn some labels into 0 targets
     if zero > 0 then
        target:sub(size-zero+1,size):zero()
     end

     local ctype = t2cpu[typename]
     input = input:type(ctype)
     local crit = nn.MultiLabelMarginCriterion():type(ctype)
     local groundtruth= crit:forward(input, target)
     input = input:type(typename)
     target = target:type(typename)
     local g_crit = nn.MultiLabelMarginCriterion():type(typename)
     local rescuda = g_crit:forward(input, target)
     local errorVal = rescuda - groundtruth
     mytester:assertlt(errorVal, precision_forward_type(precision_forward, typename),
        string.format('error on state (forward) with %s', typename))
  end
end

function cunntest.MultiLabelMarginCriterion_backward()
   local size = math.random(1,100)

   for k, typename in ipairs(typenames) do
      local input = ((torch.rand(size)-0.5) * 2):type(typename) -- data spread from -1 to 1
      local target = torch.round(torch.rand(size)*(size-1)):add(1) -- generate random labels > 0
      local zero = math.random(0,size) -- turn some labels into 0 targets
      if zero > 0 then
         target:sub(size-zero+1,size):zero()
      end

      local ctype = t2cpu[typename]
      input = input:type(ctype)
      local crit = nn.MultiLabelMarginCriterion():type(ctype)
      local pred = crit:forward(input, target)
      local groundgrad = crit:backward(input, target)

      input = input:type(typename)
      target = target:type(typename)
      local g_crit = nn.MultiLabelMarginCriterion():type(typename)
      g_crit:forward(input, target)
      local rescuda = g_crit:backward(input, target)

      local error = rescuda:double() - groundgrad:double()

      mytester:assertlt(error:abs():max(), precision_backward_type(precision_backward, typename),
         string.format('error on state (backward) with %s', typename))
   end
end

function cunntest.SpatialCrossMapLRN_forward_batch()
   local bs = math.random(4,10)
   local inputSize = math.random(6,9)
   local size = math.random(1,3)*2+1
   local nbfeatures = math.random(3,8)
   local alpha = math.random(1,100)/100
   local beta  = math.random(0,100)/100
   local k = math.random(1,3)

   for k, typename in ipairs(typenames) do
      local input = torch.rand(bs, nbfeatures, inputSize, inputSize):type(typename)

      local ctype = t2cpu[typename]
      input = input:type(ctype)
      local sconv = nn.SpatialCrossMapLRN(size, alpha, beta, k):type(ctype)
      local groundtruth = sconv:forward(input)

      input = input:type(typename)
      local gconv = nn.SpatialCrossMapLRN(size, alpha, beta, k):type(typename)
      local rescuda = gconv:forward(input)

      local error = rescuda:double() - groundtruth:double()
      mytester:assertlt(error:abs():max(), precision_forward_type(precision_forward, typename),
          string.format('error on state (forward) with %s', typename))
   end
end

function cunntest.SpatialCrossMapLRN_backward_batch()
   local bs = math.random(4,10)
   local inputSize = math.random(6,9)
   local size = math.random(1,3)*2+1
   local nbfeatures = math.random(3,8)
   local alpha = math.random(1,100)/100
   local beta  = math.random(0,100)/100
   local k = math.random(1,3)

   for k, typename in ipairs(typenames) do
      local input = torch.rand(bs, nbfeatures, inputSize, inputSize):type(typename)
      local gradOutput = torch.rand(input:size()):type(typename)

      local ctype = t2cpu[typename]
      input = input:type(ctype)
      gradOutput = gradOutput:type(ctype)
      local sconv = nn.SpatialCrossMapLRN(size, alpha, beta, k):type(ctype)
      sconv:forward(input)
      sconv:zeroGradParameters()
      local groundgrad = sconv:backward(input, gradOutput)

      input = input:type(ctype)
      gradOutput = gradOutput:type(ctype)
      local gconv = nn.SpatialCrossMapLRN(size, alpha, beta, k):type(ctype)
      gconv:forward(input)
      gconv:zeroGradParameters()
      local rescuda = gconv:backward(input, gradOutput)

      local error = rescuda:double() - groundgrad:double()

      mytester:assertlt(error:abs():max(), precision_backward_type(precision_backward),
          string.format('error on state (backward) with %s', typename))
   end
end

function cunntest.MarginCriterion_backward()
   local size = math.random(1,100)

   for k, typename in ipairs(typenames) do
      local input = ((torch.rand(size)-0.5) * 2):type(typename) -- data spread from -1 to 1
      local target = ((torch.round(torch.rand(size))*2)-1):type(typename) -- generate random labels -1, 1

      local ctype = t2cpu[typename]
      input = input:type(ctype)
      target = target:type(ctype)
      local crit = nn.MarginCriterion():type(ctype)
      crit:forward(input, target)
      local groundgrad = crit:backward(input, target)

      input = input:type(typename)
      target = target:type(typename)
      local g_crit = nn.MarginCriterion():type(typename)
      g_crit:forward(input, target)
      local rescuda = g_crit:backward(input, target)

      local error = rescuda:double() - groundgrad:double()

      mytester:assertlt(error:abs():max(), precision_backward_type(precision_backward),
         string.format('error on state (backward) with %s', typename))
   end
end

function cunntest.BCECriterion_backward()
   local size = math.random(1,100)

   local tm = {}
   local title = string.format('BCECriterion.backward, Size %d', size)
   times[title] = tm

   local input = torch.Tensor(size):uniform()
   local target = torch.Tensor(size):uniform():gt(0.5):type(torch.type(input))

   local crit = nn.BCECriterion()
   crit:forward(input, target)
   local groundgrad = crit:backward(input, target)
   local a = torch.Timer()
   for i = 1,nloop do
      groundgrad = crit:backward(input, target)
   end
   tm.cpu = a:time().real

   input = input:cuda()
   target = target:cuda()
   local g_crit = nn.BCECriterion():cuda()
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

function cunntest.BCECriterionWeights_backward()
  local size = math.random(1,100)

  local tm = {}
  local title = string.format('BCECriterionWeights.backward, Size %d', size)
  times[title] = tm

  local input = torch.Tensor(size):uniform()
  local target = torch.Tensor(size):uniform():gt(0.5):type(torch.type(input))
  local weights = torch.Tensor(size):uniform()

  local crit = nn.BCECriterion(weights)
  crit:forward(input, target)
  local groundgrad = crit:backward(input, target)
  local a = torch.Timer()
  for i = 1,nloop do
    groundgrad = crit:backward(input, target)
  end
  tm.cpu = a:time().real

  input = input:cuda()
  target = target:cuda()
  weights = weights:cuda()
  local g_crit = nn.BCECriterion(weights):cuda()
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
      for k, typename in ipairs(typenames) do
         local size = math.random(3000,5000)
         local input = torch.randn(size,1,1):type(typename)
         local target = torch.randn(size):type(typename)

         local ctype = t2cpu[typename]
         input = input:type(ctype)
         target = target:type(ctype)
         local mod = nn.MSECriterion(sizeAverage == 1):type(ctype)

         local fout = mod:forward(input,target)
         local fgin = mod:backward(input,target):clone()

         local cinput = input:type(typename)
         local ctarget = target:type(typename)
         local cmod = nn.MSECriterion(sizeAverage == 1):type(typename)
         local cout = cmod:forward(cinput,ctarget)
         local cgin = cmod:backward(cinput,ctarget)

         if (typename == 'torch.CudaHalfTensor') then
            fout = ffi.C.THC_half2float(ffi.C.THC_float2half(fout))
         end
         mytester:assertlt(math.abs(fout-cout), precision_forward_type(0.02, typename),
            string.format('error  on output with %s', typename))
         local gerr = cgin:double() - fgin:double()
         mytester:assertlt(gerr:abs():max(), precision_forward_type(precision_forward, typename),
            string.format('error  on gradInput with %s', typename))
      end
   end
end

function cunntest.SmoothL1()
   for sizeAverage = 0, 1 do
      local size = math.random(3000,5000)

      for k, typename in ipairs(typenames) do
         local input = torch.randn(size,1,1):type(typename)
         local target = torch.randn(size):type(typename)

         local ctype = t2cpu[typename]
         input = input:type(ctype)
         target = target:type(ctype)
         local mod = nn.SmoothL1Criterion(sizeAverage == 1):type(ctype)

         local fout = mod:forward(input,target)
         local fgin = mod:backward(input,target):clone()

         local cinput = input:type(typename)
         local ctarget = target:type(typename)
         local cmod = nn.SmoothL1Criterion(sizeAverage == 1):type(typename)
         local cout = cmod:forward(cinput,ctarget)
         local cgin = cmod:backward(cinput,ctarget)

         if (typename == 'torch.CudaHalfTensor') then
            fout = ffi.C.THC_half2float(ffi.C.THC_float2half(fout))
         end
         mytester:assertlt(math.abs(fout-cout), 0.01, string.format('error  on output with %s', typename))
         local gerr = cgin:double() - fgin:double()
         mytester:assertlt(gerr:abs():max(), precision_forward_type(precision_forward, typename),
            string.format('error  on gradInput with %s', typename))
      end
   end
end

function cunntest.SoftMarginCriterion()
   for sizeAverage = 0, 1 do
      for k, typename in ipairs(typenames) do
         local size = math.random(3000,5000)
         local input = torch.randn(size,1,1):type(typename)
         local target = torch.randn(size):type(typename)

         local ctype = t2cpu[typename]
         input = input:type(ctype)
         target = target:type(ctype)
         local mod = nn.SoftMarginCriterion(sizeAverage == 1):type(ctype)

         local fout = mod:forward(input,target)
         local fgin = mod:backward(input,target):clone()

         local cinput = input:type(typename)
         local ctarget = target:type(typename)
         local cmod = nn.SoftMarginCriterion(sizeAverage == 1):type(typename)
         local cout = cmod:forward(cinput,ctarget)
         local cgin = cmod:backward(cinput,ctarget)

        mytester:assertlt(math.abs(fout-cout), 0.01, 'error  on output')
        local gerr = cgin:double() - fgin:double()
        mytester:assertlt(gerr:abs():max(), precision_forward_type(precision_forward, typename),
           string.format('error  on gradInput with %s', typename))
      end
   end
end


function cunntest.distkldiv()
   for sizeAverage = 0, 1 do
      local size = math.random(3000,5000)
      local input = torch.randn(size) -- TODO, make it back to (size, 1, 1), see https://github.com/torch/cunn/issues/245#issuecomment-209260954
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

   for k, typename in ipairs(typenames) do
      local input = torch.randn(size):type(typename)
      local ctype = t2cpu[typename]
      input = input:type(ctype)
      local sconv = nn.SoftPlus():type(ctype)
      local groundtruth = sconv:forward(input)

      input = input:type(typename)
      local gconv = nn.SoftPlus():type(typename)
      local rescuda = gconv:forward(input)

      local error = rescuda:double() - groundtruth:double()
      mytester:assertlt(error:abs():max(), precision_forward_type(precision_forward,typename),
          string.format('error on state (forward) with %s', typename))
    end
end

function cunntest.SoftPlus_backward()
   local size = math.random(1,100)

   for k, typename in ipairs(typenames) do
      local input = torch.randn(size):type(typename)
      local gradOutput = torch.randn(size):type(typename)
      local ctype = t2cpu[typename]
      input = input:type(ctype)
      gradOutput = gradOutput:type(ctype)
      local sconv = nn.SoftPlus():type(ctype)
      sconv:forward(input)
      local groundgrad = sconv:backward(input, gradOutput)

      input = input:type(typename)
      gradOutput = gradOutput:type(typename)
      local gconv = sconv:clone():type(typename)
      gconv:forward(input)
      local rescuda = gconv:backward(input, gradOutput)

      local error = rescuda:double() - groundgrad:double()
      mytester:assertlt(error:abs():max(), precision_backward_type(precision_backward, typename),
          string.format('error on state (backward) with %s', typename))
    end
end

function cunntest.SpatialUpSamplingNearest_forward()
   local f = torch.random(3, 15)
   local h = torch.random(3, 15)
   local w = torch.random(3, 15)
   local scale = torch.random(2,5)

   for k, typename in ipairs(typenames) do
      local input = torch.randn(f, h, w):type(typename)

      local ctype = t2cpu[typename]
      input = input:type(ctype)
      local sconv = nn.SpatialUpSamplingNearest(scale):type(ctype)
      local groundtruth = sconv:forward(input)

      input = input:type(typename)
      local gconv = sconv:clone():type(typename)
      local rescuda = gconv:forward(input)

      local error = rescuda:double() - groundtruth:double()
      mytester:assertlt(error:abs():max(), precision_forward_type(precision_forward, typename),
          string.format('error on state (forward) with %s', typename))
   end
end

function cunntest.SpatialUpSamplingNearest_forward_batch()
   local nbatch = torch.random(3, 15)
   local f = torch.random(3, 15)
   local h = torch.random(3, 15)
   local w = torch.random(3, 15)
   local scale = torch.random(2,5)

   for k, typename in ipairs(typenames) do
      local input = torch.randn(nbatch, f, h, w):type(typename)

      local ctype = t2cpu[typename]
      input = input:type(ctype)
      local sconv = nn.SpatialUpSamplingNearest(scale):type(ctype)
      local groundtruth = sconv:forward(input)

      input = input:type(typename)
      local gconv = sconv:clone():type(typename)
      local rescuda = gconv:forward(input)

      local error = rescuda:double() - groundtruth:double()
      mytester:assertlt(error:abs():max(), precision_forward_type(precision_forward, typename),
          string.format('error on state (forward) with %s', typename))
   end
end

function cunntest.SpatialUpSamplingNearest_backward()
   local f = torch.random(3, 15)
   local h = torch.random(3, 15)
   local w = torch.random(3, 15)
   local scale = torch.random(2,5)

   for k, typename in ipairs(typenames) do
      local input = torch.randn(f, h, w):type(typename)
      local gradOutput = torch.randn(f, h*scale, w*scale):type(typename)

      local ctype = t2cpu[typename]
      input = input:type(ctype)
      gradOutput = gradOutput:type(ctype)
      local sconv = nn.SpatialUpSamplingNearest(scale):type(ctype)
      sconv:forward(input)
      sconv:zeroGradParameters()
      local groundgrad = sconv:backward(input, gradOutput)

      input = input:type(typename)
      gradOutput = gradOutput:type(typename)
      local gconv = sconv:clone():type(typename)
      gconv:forward(input)
      gconv:zeroGradParameters()
      local rescuda = gconv:backward(input, gradOutput)

      local error = rescuda:double() - groundgrad:double()

      mytester:assertlt(error:abs():max(), precision_backward_type(precision_backward, typename),
          string.format('error on state (backward) with %s', typename))
   end
end

function cunntest.SpatialUpSamplingNearest_backward_batch()
   local nbatch = torch.random(3, 15)
   local f = torch.random(3, 15)
   local h = torch.random(3, 15)
   local w = torch.random(3, 15)
   local scale = torch.random(2,5)

   for k, typename in ipairs(typenames) do
      local input = torch.randn(nbatch, f, h, w):type(typename)
      local gradOutput = torch.randn(nbatch, f, h*scale, w*scale):type(typename)

      local ctype = t2cpu[typename]
      input = input:type(ctype)
      gradOutput = gradOutput:type(ctype)
      local sconv = nn.SpatialUpSamplingNearest(scale):type(ctype)
      sconv:forward(input)
      sconv:zeroGradParameters()
      local groundgrad = sconv:backward(input, gradOutput)

      input = input:type(typename)
      gradOutput = gradOutput:type(typename)
      local gconv = sconv:clone():type(typename)
      gconv:forward(input)
      gconv:zeroGradParameters()
      local rescuda = gconv:backward(input, gradOutput)

      local error = rescuda:double() - groundgrad:double()

      mytester:assertlt(error:abs():max(), precision_backward_type(precision_backward, typename),
          string.format('error on state (backward) with %s', typename))
   end
end

function cunntest.SpatialUpSamplingBilinear_forward()
   local f = torch.random(3, 15)
   local h = torch.random(3, 15)
   local w = torch.random(3, 15)
   local scale = torch.random(2,5)

   for k, typename in ipairs(typenames) do
      local input = torch.randn(f, h, w):type(typename)

      local ctype = t2cpu[typename]
      input = input:type(ctype)
      local sconv = nn.SpatialUpSamplingBilinear(scale):type(ctype)
      local groundtruth = sconv:forward(input)

      input = input:type(typename)
      local gconv = sconv:clone():type(typename)
      local rescuda = gconv:forward(input)

      local error = rescuda:double() - groundtruth:double()
      mytester:assertlt(error:abs():max(), precision_forward_type(precision_forward, typename),
                        string.format('error on state (forward) with %s', typename))
   end
end

function cunntest.SpatialUpSamplingBilinear_forward_batch()
   local nbatch = torch.random(3, 15)
   local f = torch.random(3, 15)
   local h = torch.random(3, 15)
   local w = torch.random(3, 15)
   local scale = torch.random(2,5)

   for k, typename in ipairs(typenames) do
      local input = torch.randn(nbatch, f, h, w):type(typename)

      local ctype = t2cpu[typename]
      input = input:type(ctype)
      local sconv = nn.SpatialUpSamplingBilinear(scale):type(ctype)
      local groundtruth = sconv:forward(input)

      input = input:type(typename)
      local gconv = sconv:clone():type(typename)
      local rescuda = gconv:forward(input)

      local error = rescuda:double() - groundtruth:double()
      mytester:assertlt(error:abs():max(), precision_forward_type(precision_forward, typename),
                        string.format('error on state (forward) with %s', typename))
   end
end

function cunntest.SpatialUpSamplingBilinear_backward()
   local f = torch.random(3, 15)
   local h = torch.random(3, 15)
   local w = torch.random(3, 15)
   local scale = torch.random(2,5)

   for k, typename in ipairs(typenames) do
      local input = torch.randn(f, h, w):type(typename)
      local gradOutput = torch.randn(f, h*scale, w*scale):type(typename)

      local ctype = t2cpu[typename]
      input = input:type(ctype)
      gradOutput = gradOutput:type(ctype)
      local sconv = nn.SpatialUpSamplingBilinear(scale):type(ctype)
      sconv:forward(input)
      sconv:zeroGradParameters()
      local groundgrad = sconv:backward(input, gradOutput)

      input = input:type(typename)
      gradOutput = gradOutput:type(typename)
      local gconv = sconv:clone():type(typename)
      gconv:forward(input)
      gconv:zeroGradParameters()
      local rescuda = gconv:backward(input, gradOutput)

      local error = rescuda:double() - groundgrad:double()

      mytester:assertlt(error:abs():max(), precision_backward_type(precision_backward, typename),
                        string.format('error on state (backward) with %s', typename))
   end
end

function cunntest.SpatialUpSamplingBilinear_backward_batch()
   local nbatch = torch.random(3, 15)
   local f = torch.random(3, 15)
   local h = torch.random(3, 15)
   local w = torch.random(3, 15)
   local scale = torch.random(2,5)

   for k, typename in ipairs(typenames) do
      local input = torch.randn(nbatch, f, h, w):type(typename)
      local gradOutput = torch.randn(nbatch, f, h*scale, w*scale):type(typename)

      local ctype = t2cpu[typename]
      input = input:type(ctype)
      gradOutput = gradOutput:type(ctype)
      local sconv = nn.SpatialUpSamplingBilinear(scale):type(ctype)
      local output = sconv:forward(input)
      sconv:zeroGradParameters()
      local groundgrad = sconv:backward(input, gradOutput)

      input = input:type(typename)
      gradOutput = gradOutput:type(typename)
      local gconv = sconv:clone():type(typename)
      gconv:forward(input)
      gconv:zeroGradParameters()
      local rescuda = gconv:backward(input, gradOutput)

      local err = rescuda:double() - groundgrad:double()

      mytester:assertlt(err:abs():max(), precision_backward_type(precision_backward, typename),
                        string.format('error on state (backward) with %s', typename))
   end
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

function cunntest.SpatialClassNLLCriterion()
   local batchSize = math.random(5, 10)
   local h = math.random(300, 500)
   local w = math.random(300, 800)
   local classes = math.random(10,30)

   for k, typename in ipairs(typenames) do
      local input = torch.randn(batchSize, classes, h, w):type(typename)
      local target = torch.Tensor(batchSize, h, w)
      target:apply(function() return math.random(1, classes) end)
      local ctype = t2cpu[typename]
      input = input:type(ctype)
      local mod = nn.SpatialClassNLLCriterion():type(ctype)
      local fout = mod:forward(input, target)
      local fgin = mod:backward(input, target):clone()

      local cinput = input:type(typename)
      local ctarget = target:type(typename)

      local cmod = nn.SpatialClassNLLCriterion():type(typename)
      local cout = cmod:forward(cinput,ctarget)
      local cgin = cmod:backward(cinput,ctarget)
      cutorch.synchronize()

      mytester:assertlt(
        math.abs(fout-cout), precision_forward_type(precision_forward, typename),
          string.format('error on output with %s', typename))

      local gerr = cgin:double() - fgin:double()
      mytester:assertlt(gerr:abs():max(), precision_forward_type(precision_forward, typename),
          string.format('error  on gradInput with %s', typename))
    end
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
   local outi = math.random(1,20)
   local outj = math.random(1,20)
   local outk = math.random(1,20)
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
   local iT = math.random(kT*2, 60)
   local iH = math.random(kH*2, 60)
   local iW = math.random(kW*2, 60)
   local padT = math.random(0,kT/2-1)
   local padH = math.random(0,kH/2-1)
   local padW = math.random(0,kW/2-1)
   local iF = math.random(1, 16) -- features
   local oT = math.floor((iT - kT + 2*padT) / dT + 1)
   local oH = math.floor((iH - kH + 2*padH) / dH + 1)
   local oW = math.floor((iW - kW + 2*padW) / dW + 1)

   local tm = {}
   local title = string.format('VolumetricMaxPooling.forward %dx%dx%dx%d o %dx%dx%d (%dx%dx%d)-> %dx%dx%dx%d',
                           iF, iT, iH, iW, kT, kH, kW, dT, dH, dW, iF, oT, oH, oW)
   times[title] = tm

   local input = torch.Tensor(iF, iT, iH, iW):float():uniform(-1, 1)
   local layer = nn.VolumetricMaxPooling(kT, kW, kH, dT, dW, dH, padT, padW, padH):float()
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
   local iT = math.random(kT*2, 60)
   local iH = math.random(kH*2, 60)
   local iW = math.random(kW*2, 60)
   local padT = math.random(0,kT/2-1)
   local padH = math.random(0,kH/2-1)
   local padW = math.random(0,kW/2-1)
   local iF = math.random(1, 16) -- features
   local oT = math.floor((iT - kT + 2*padT) / dT + 1)
   local oH = math.floor((iH - kH + 2*padH) / dH + 1)
   local oW = math.floor((iW - kW + 2*padW) / dW + 1)

   local tm = {}
   local title = string.format('VolumetricMaxPooling.backward %dx%dx%dx%d o %dx%dx%d (%dx%dx%d) -> %dx%dx%dx%d',
                               iF, iT, iH, iW, kT, kH, kW, dT, dH, dW, iF, oT, oH, oW)
   times[title] = tm

   local input = torch.Tensor(iF, iT, iH, iW):float():uniform(-1, 1)
   local layer = nn.VolumetricMaxPooling(kT, kW, kH, dT, dW, dH, padT, padW, padH):float()
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

function cunntest.VolumetricDilatedMaxPooling_forward_batch()
   local bs = math.random(8,16)
   local from = math.random(8,16)
   local to = from
   local kt = math.random(2,4)
   local ki = math.random(2,4)
   local kj = math.random(2,4)
   local st = math.random(2,4)
   local si = math.random(2,4)
   local sj = math.random(2,4)
   local outt = math.random(1,10)
   local outi = math.random(1,33)
   local outj = math.random(1,33)
   local padt = math.random(0,kt/2-1)
   local padi = math.random(0,ki/2-1)
   local padj = math.random(0,kj/2-1)
   local dilationt = math.random(1,10)
   local dilationi = math.random(1,10)
   local dilationj = math.random(1,10)
   local int = (outt-1)*st+(dilationt*(kt-1)+1)-2*padt
   local ini = (outi-1)*si+(dilationi*(ki-1)+1)-2*padi
   local inj = (outj-1)*sj+(dilationj*(kj-1)+1)-2*padj
   local ceil_mode = math.random(0,1) == 1

   local tm = {}
   local title = string.format('VolumetricDilatedMaxPooling.forward %dx%dx%dx%dx%d o %dx%dx%d -> %d%dx%dx%dx%d',
                               bs, from, int, inj, ini, kt, kj, ki, bs, to, outt, outj, outi)
   times[title] = tm

   local input = torch.randn(bs,from,int,inj,ini)
   local sconv = nn.VolumetricDilatedMaxPooling(kt,ki,kj,st,si,sj,padt,padi,padj,dilationt,dilationi,dilationj)
   if ceil_mode then sconv:ceil() end
   local groundtruth = sconv:forward(input)
   local a = torch.Timer()
   for i = 1,nloop do
      groundtruth = sconv:forward(input)
   end
   tm.cpu = a:time().real

   input = input:cuda()
   local gconv = nn.VolumetricDilatedMaxPooling(kt,ki,kj,st,si,sj,padt,padi,padj,dilationt,dilationi,dilationj):cuda()
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

function cunntest.VolumetricDilatedMaxPooling_backward_batch()
   local bs = math.random(8,16)
   local from = math.random(8,16)
   local to = from
   local kt = math.random(2,4)
   local ki = math.random(2,4)
   local kj = math.random(2,4)
   local st = math.random(2,4)
   local si = math.random(2,4)
   local sj = math.random(2,4)
   local outt = math.random(32,60)
   local outi = math.random(32,60)
   local outj = math.random(32,60)
   local padt = math.random(0,kt/2-1)
   local padi = math.random(0,ki/2-1)
   local padj = math.random(0,kj/2-1)
   local dilationt = math.random(1,10)
   local dilationi = math.random(1,10)
   local dilationj = math.random(1,10)
   local int = (outt-1)*st+(dilationt*(kt-1)+1)-2*padt
   local ini = (outi-1)*si+(dilationi*(ki-1)+1)-2*padi
   local inj = (outj-1)*sj+(dilationj*(kj-1)+1)-2*padj
   local ceil_mode = math.random(0,1) == 1

   local tm = {}
   local title = string.format('VolumetricDilatedMaxPooling.forward %dx%dx%dx%dx%d o %dx%dx%d -> %d%dx%dx%dx%d',
                               bs, from, int, inj, ini, kt, kj, ki, bs, to, outt, outj, outi)
   times[title] = tm

   local input = torch.randn(bs,from,int,inj,ini)
   local gradOutput = torch.randn(bs,to,outt,outj,outi)
   local sconv = nn.VolumetricDilatedMaxPooling(kt,ki,kj,st,si,sj,padt,padi,padj,dilationt,dilationi,dilationj)
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
   local gconv = nn.VolumetricDilatedMaxPooling(kt,ki,kj,st,si,sj,padt,padi,padj,dilationt,dilationi,dilationj):cuda()
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

function cunntest.VolumetricMaxUnpooling_forward_batch()
   local bs = math.random(4,10)
   local from = math.random(1,64)
   local to = from
   local kt = math.random(3,7)
   local ki = math.random(3,7)
   local kj = math.random(3,7)
   local st, si, sj = kt, ki, kj
   local outt = math.random(32,128)
   local outi = math.random(32,128)
   local outj = math.random(32,128)
   local padt = math.random(0,kt/2-1)
   local padi = math.random(0,ki/2-1)
   local padj = math.random(0,kj/2-1)
   local it = ((outt + padt*2 - kt)/st) +1
   local ii = ((outi + padi*2 - ki)/si) +1
   local ij = ((outj + padj*2 - kj)/sj) +1

   local tm = {}
   local title = string.format('VolumetricMaxUnpooling.forward %dx%dx%dx%dx%d o %dx%dx%d -> %d%dx%dx%dx%d',
                               bs, from, it, ij, ii, kt, kj, ki, bs, to, outt, outj, outi)
   times[title] = tm

   local pooler = nn.VolumetricMaxPooling(kt,ki,kj,st,si,sj,padt,padi,padj)
   local sunpool = nn.VolumetricMaxUnpooling(pooler)

   local original = torch.randn(bs,from,it,ij,ii)
   local input = pooler:forward(original)
   local groundtruth = sunpool:forward(input)
   local a = torch.Timer()
   for i = 1,nloop do
      groundtruth = sunpool:forward(input)
   end
   tm.cpu = a:time().real

   original = original:cuda()
   pooler = nn.VolumetricMaxPooling(kt,ki,kj,st,si,sj,padt,padi,padj):cuda()
   local gunpool = nn.VolumetricMaxUnpooling(pooler):cuda()

   input = pooler:forward(original)
   local rescuda = gunpool:forward(input)
   a:reset()
   for i = 1,nloop do
      rescuda = gunpool:forward(input)
   end
   cutorch.synchronize()
   tm.gpu = a:time().real

   local error = rescuda:float() - groundtruth
   mytester:assertlt(error:abs():max(), precision_forward, 'error on state (forward) ')
end

function cunntest.VolumetricMaxUnpooling_backward_batch()
   local bs = math.random(4,10)
   local from = math.random(1,64)
   local to = from
   local kt = math.random(3,7)
   local ki = math.random(3,7)
   local kj = math.random(3,7)
   local st, si, sj = kt, ki, kj
   local outt = math.random(32,128)
   local outi = math.random(32,128)
   local outj = math.random(32,128)
   local padt = math.random(0,kt/2-1)
   local padi = math.random(0,ki/2-1)
   local padj = math.random(0,kj/2-1)
   local it = ((outt + padt*2 - kt)/st) +1
   local ii = ((outi + padi*2 - ki)/si) +1
   local ij = ((outj + padj*2 - kj)/sj) +1

   local tm = {}
   local title = string.format('VolumetricMaxUnpooling.backward %dx%dx%dx%dx%d o %dx%dx%d -> %d%dx%dx%dx%d',
                               bs, from, it, ij, ii, kt, kj, ki, bs, to, outt, outj, outi)
   times[title] = tm

   local pooler = nn.VolumetricMaxPooling(kt,ki,kj,st,si,sj,padt,padi,padj)
   local sunpool = nn.VolumetricMaxUnpooling(pooler)

   local original = torch.randn(bs,from,it,ij,ii)
   local input = pooler:forward(original)
   local gradOutput = torch.randn(original:size())
   sunpool:forward(input)
   sunpool:zeroGradParameters()
   local groundgrad = sunpool:backward(input, gradOutput)
   local a = torch.Timer()
   for i = 1,nloop do
      sunpool:zeroGradParameters()
      groundgrad = sunpool:backward(input, gradOutput)
   end
   tm.cpu = a:time().real

   pooler = nn.VolumetricMaxPooling(kt,ki,kj,st,si,sj,padt,padi,padj):cuda()
   local gunpool = nn.VolumetricMaxUnpooling(pooler):cuda()

   original = original:cuda()
   input = pooler:forward(original)
   gunpool:forward(input)

   gradOutput = gradOutput:cuda()
   gunpool:zeroGradParameters()
   local rescuda = gunpool:backward(input, gradOutput)
   a:reset()
   for i = 1,nloop do
      gunpool:zeroGradParameters()
      rescuda = gunpool:backward(input, gradOutput)
   end
   cutorch.synchronize()
   tm.gpu = a:time().real

   local error = rescuda:float() - groundgrad

   mytester:assertlt(error:abs():max(), precision_backward, 'error on state (backward) ')
end

function cunntest.VolumetricAveragePooling_forward()
   local kT = math.random(3, 7)
   local kH = math.random(3, 7)
   local kW = math.random(3, 7)
   local dT = math.random(1, 13)
   local dH = math.random(1, 13)
   local dW = math.random(1, 13)
   local oT = math.random(1, 20)
   local oH = math.random(1, 20)
   local oW = math.random(1, 20)
   local iF = math.random(1, 16) -- features
   local iT = (oT - 1) * dT + kT
   local iH = (oH - 1) * dH + kH
   local iW = (oW - 1) * dW + kW

   local tm = {}
   local title = string.format('VolumetricAveragePooling.forward %dx%dx%dx%d o %dx%dx%d (%dx%dx%d) -> %dx%dx%dx%d',
                               iF, iT, iH, iW, kT, kH, kW, dT, dH, dW, iF, oT, oH, oW)
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
   local oT = math.random(1, 20)
   local oH = math.random(1, 20)
   local oW = math.random(1, 20)
   local iF = math.random(1, 16) -- features
   local iT = (oT - 1) * dT + kT
   local iH = (oH - 1) * dH + kH
   local iW = (oW - 1) * dW + kW

   local tm = {}
   local title = string.format('VolumetricAveragePooling.backward %dx%dx%dx%d o %dx%dx%d (%dx%dx%d) -> %dx%dx%dx%d',
                           iF, iT, iH, iW, kT, kH, kW, dT, dH, dW, iF, oT, oH, oW)
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
    local input = torch.randn(nOutputPlane,h,w)

    for k, typename in ipairs(typenames) do
      local ctype = t2cpu[typename]
      local input = input:type(ctype)
      local sconv = nn.PReLU(nOutputPlane):type(ctype)
      local groundtruth = sconv:forward(input)

      input = input:type(typename)
      local gconv = sconv:type(typename)
      local rescuda = gconv:forward(input)

      local error = rescuda:double() - groundtruth:double()
      mytester:assertlt(error:abs():max(), precision_forward_type(precision_forward, typename),
          string.format('error on state with %s', typename))
    end
end

function cunntest.PReLU_backward()
    local nOutputPlane = 8
    local w = math.random(1,10)
    local h = math.random(1,10)

    for k, typename in ipairs(typenames) do
        local input = torch.randn(nOutputPlane, h, w):type(typename)
        local gradOutput = torch.randn(#input):type(typename)
        local ctype = t2cpu[typename]
        input = input:type(ctype)
        gradOutput = gradOutput:type(ctype)
        local sconv = nn.PReLU(nOutputPlane):type(ctype)
        local gconv = sconv:clone():type(typename)

        sconv:forward(input)
        sconv:zeroGradParameters()
        local groundgrad = sconv:backward(input, gradOutput)

        input = input:type(typename)
        gradOutput = gradOutput:type(typename)
        gconv:forward(input)
        gconv:zeroGradParameters()
        local rescuda = gconv:backward(input, gradOutput)

        local err = rescuda:double() - groundgrad:double()
        local weightGradError = gconv.gradWeight:double() - sconv.gradWeight:double()

        mytester:assertlt(err:abs():max(), precision_backward_type(precision_backward, typename),
            string.format('error on state %s', typename))
        mytester:assertlt(weightGradError:abs():max(), precision_backward_type(precision_backward, typename),
            string.format('error on weight %s', typename))
    end
end


function cunntest.RReLU_forward()
    local nOutputPlane = 8
    local w = math.random(1,100)
    local h = math.random(1,100)

    for k, typename in ipairs(typenames) do
       for _,train in ipairs({true,false}) do
          for _,inplace in ipairs({false,true}) do
              local input = torch.randn(nOutputPlane, h, w):type(typename) - 0.5
              local ctype = t2cpu[typename]
              input = input:type(ctype)
              local sconv = nn.RReLU(1/8, 1/3, inplace):type(ctype)
              if not train then
                  sconv:evaluate()
              end
              local groundtruth = sconv:forward(input:clone())

              input = input:type(typename)
              local gconv = sconv:type(typename)
              local rescuda = gconv:forward(input:clone())

              if not train then
                  local error = rescuda:double() - groundtruth:double()
                  mytester:assertlt(error:abs():max(), precision_forward_type(precision_forward, typename),
                      string.format('error on state %s', typename))
              end
          end
      end
    end
end

function cunntest.RReLU_backward()
    local nOutputPlane = 8
    local w = math.random(1,10)
    local h = math.random(1,10)

    for k, typename in ipairs(typenames) do
        for _,train in ipairs({true,false}) do
            for _,inplace in ipairs({false,true}) do
                local ctype = t2cpu[typename]
                local input = torch.randn(nOutputPlane, h, w):type(typename)
                local gradOutput = torch.randn(#input):type(typename) - 0.5
                input = input:type(ctype)
                gradOutput = gradOutput:type(ctype)
                local sconv = nn.RReLU(1/8, 1/3, inplace):type(ctype)
                if not train then
                  sconv:evaluate()
                end

                sconv:forward(input:clone())
                local groundgrad = sconv:backward(input, gradOutput:clone())

                local gconv = sconv:clone():type(typename)
                input = input:type(typename)
                gradOutput = gradOutput:type(typename)
                gconv:forward(input:clone())
                local rescuda = gconv:backward(input, gradOutput:clone())

                if not train then
                  local err = rescuda:double() - groundgrad:double()
                  mytester:assertlt(err:abs():max(), precision_backward_type(precision_backward, typename),
                    string.format('error on state', typename))
                end

                input = -torch.rand(1000):type(typename)
                gconv:forward(input) -- fill internal noise tensor
                local g = gconv:backward(input, torch.ones(1000):type(typename))
                local err = math.abs(g[input:le(0)]:mean()-(gconv.lower+gconv.upper)/2)
                mytester:assertlt(err, 0.05, 'mean deviation of gradient for negative inputs')
          end
       end
    end
end

function cunntest.VolumetricFullConvolution_pair_test()

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

    local module = nn.VolumetricFullConvolution(inChan, outChan, kT, kH, kW,
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

function cunntest.VolumetricFullConvolution()
    local module = nn.VolumetricFullConvolution(3, 1, 3, 3, 3, 3, 3, 3);
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

    module:zeroGradParameters()
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
                    mytester:assertlt(module.gradWeight[c][1][t][h][w] - 0.1, precision_backward,
                                      'error on backward weight gradients ')
                end
            end
        end
    end
end

function cunntest.VolumetricDilatedConvolution()
   local from = math.random(1,32)
   local to = math.random(1,8) * 8
   local ki = math.random(1,15)
   local kj = math.random(1,15)
   local kk = math.random(1,3)
   local si = math.random(1,3)
   local sj = math.random(1,3)
   local sk = math.random(1,3)
   local padW = math.random(0,1)
   local padH = math.random(0,1)
   local padT = math.random(0,1)
   local outi = math.random(ki, 64)
   local outj = math.random(kj, 64)
   local outk = math.random(kk, kk+5)
   local dilationW = math.random(1,10)
   local dilationH = math.random(1,10)
   local dilationT = math.random(1,10)
   local ini = (outi - 1) * si - 2 * padW + dilationW * (ki-1) + 1
   local inj = (outj - 1) * sj - 2 * padH + dilationH * (kj-1) + 1
   local ink = (outk - 1) * sk - 2 * padT + dilationT * (kk-1) + 1

   local input = torch.randn(from,ink,inj,ini)
   local sconv = nn.VolumetricDilatedConvolution(from,to,kk,ki,kj,sk,si,sj,padT,padW,padH,dilationT,dilationW,dilationH)
   local output = sconv:forward(input)
   local gradOutput = output:clone():normal()
   sconv:zeroGradParameters()
   local groundgrad = sconv:backward(input, gradOutput)
   local groundweight = sconv.gradWeight
   local groundbias = sconv.gradBias

   input = input:cuda()
   gradOutput = gradOutput:cuda()
   local gconv = nn.VolumetricDilatedConvolution(from,to,kk,ki,kj,sk,si,sj,padT,padW,padH,dilationT,dilationW,dilationH):cuda()
   gconv.weight = sconv.weight:cuda()
   gconv.bias = sconv.bias:cuda()
   local rescuda = gconv:forward(input)
   gconv:zeroGradParameters()
   local gradcuda = gconv:backward(input, gradOutput)
   local weightcuda = gconv.gradWeight
   local biascuda = gconv.gradBias

   local error = rescuda:float() - output
   local gerror = gradcuda:float() - groundgrad
   local werror = weightcuda:float() - groundweight
   local berror = biascuda:float() - groundbias

   mytester:assertlt(error:abs():max(), precision_forward, 'error on state (forward) ')
   mytester:assertlt(gerror:abs():max(), precision_backward, 'error on state (backward) ')
   mytester:assertlt(werror:abs():max(), precision_backward, 'error on weight (backward) ')
   mytester:assertlt(berror:abs():max(), precision_backward, 'error on bias (backward) ')
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
   local grid = {
      nInput = {10, 101, 1000, 10007},
      nVocab = {100, 10000},
      nDim = {97, 255},
      scaleGradByFreq = {false, true},
      batch = {false, true},
      paddingValue = {0, 1},
   }

   for itr = 1, 10 do
      -- Randomly sample from grid of parameters
      local s = {}
      for k, v in pairs(grid) do
         s[k] = v[torch.random(#v)]
      end

      local input, gradOutput
      if s.batch then
         input = torch.LongTensor(s.nInput, 5):random(s.nVocab)
         gradOutput = torch.randn(s.nInput, 5, s.nDim)
      else
         input = torch.LongTensor(s.nInput):random(s.nVocab)
         gradOutput = torch.randn(s.nInput, s.nDim)
      end

      local sconv = nn.LookupTable(s.nVocab, s.nDim, s.paddingValue)
      local gconv = sconv:clone():cuda()
      if s.scaleGradByFreq then
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
         'error on weight for size ' .. tostring(s.nInput) ..
          ' nVocab: ' .. tostring(s.nVocab) ..
          ' nDim ' .. tostring(s.nDim) ..
          ' scaleGradByFreq: ' .. tostring(s.scaleGradByFreq) ..
          ' batch: ' .. tostring(s.batch) ..
          ' paddingValue: ' .. tostring(s.paddingValue))
   end

   local nVocab = 10000
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

function cunntest.getParameters()
  -- tensors are non-contiguous but compact; they can be gathered
  local L = nn.Linear(10,10):cuda()
  L.weight = torch.CudaTensor(10,10):t():fill(1)
  local tmp = torch.CudaTensor(10,10):fill(2)
  L.bias = tmp:select(1,2)
  local P = L:getParameters()
  mytester:asserteq(L.weight:mean(), 1)
  mytester:asserteq(L.bias:mean(), 2)
  mytester:asserteq(L.weight:storage(), L.bias:storage())
  mytester:asserteq(P:nElement(), 110)
  mytester:asserteq(P:storage():size(), 110)
  mytester:assertlt(L.bias[{ {10} }]:storageOffset() - 1, L.bias:storage():size())
end

function cunntest.SpatialReflectionPadding_forward()
   local batch = math.random(1,3)
   local plane = math.random(1,3)
   local sizeY = math.random(7,16)
   local sizeX = math.random(7,16)
   local padL = math.random(-3,3)
   local padR = math.random(-3,3)
   local padT = math.random(-3,3)
   local padB = math.random(-3,3)

   for k, typename in ipairs(typenames) do
      local input = torch.rand(batch, plane, sizeY, sizeX):type(typename)

      local ctype = t2cpu[typename]
      input = input:type(ctype)
      local module = nn.SpatialReflectionPadding(padL, padR, padT, padB):type(ctype)
      local groundtruth = module:forward(input)

      input = input:type(typename)
      local gmodule = nn.SpatialReflectionPadding(padL, padR, padT, padB):type(typename)
      local rescuda = gmodule:forward(input)

      local error = rescuda:double() - groundtruth:double()
      mytester:assertlt(error:abs():max(),
                        precision_forward_type(precision_forward, typename),
                        string.format('error on state (forward) with %s', typename))
   end
end

function cunntest.SpatialReflectionPadding_backward()
   local batch = math.random(1,3)
   local plane = math.random(1,3)
   local sizeY = math.random(7,16)
   local sizeX = math.random(7,16)
   local padL = math.random(-3,3)
   local padR = math.random(-3,3)
   local padT = math.random(-3,3)
   local padB = math.random(-3,3)

   for k, typename in ipairs(typenames) do
      local input = torch.rand(batch, plane, sizeY, sizeX):type(typename)
      local gradOutput = torch.rand(
          batch, plane, sizeY + padT + padB, sizeX + padL + padR
       ):type(typename)

       local ctype = t2cpu[typename]
       input = input:type(ctype)
       gradOutput = gradOutput:type(ctype)
       local module = nn.SpatialReflectionPadding(padL, padR, padT, padB):type(ctype)
       module:forward(input)
       module:zeroGradParameters()
       local groundgrad = module:backward(input, gradOutput)

       input = input:type(typename)
       gradOutput = gradOutput:type(typename)
       local gmodule = nn.SpatialReflectionPadding(padL, padR, padT, padB):type(typename)
       gmodule:forward(input)
       gmodule:zeroGradParameters()
       local rescuda = gmodule:backward(input, gradOutput)

       local error = rescuda:double() - groundgrad:double()
       mytester:assertlt(error:abs():max(),
                         precision_backward_type(precision_backward, type),
                         string.format('error on state (backward) with %s', typename))
   end
end

function cunntest.SpatialReplicationPadding_forward()
   local batch = math.random(1,3)
   local plane = math.random(1,3)
   local sizeY = math.random(7,16)
   local sizeX = math.random(7,16)
   local padL = math.random(-3,3)
   local padR = math.random(-3,3)
   local padT = math.random(-3,3)
   local padB = math.random(-3,3)

   for k, typename in ipairs(typenames) do
      local input = torch.rand(batch, plane, sizeY, sizeX):type(typename)

      local ctype = t2cpu[typename]
      input = input:type(ctype)
      local module = nn.SpatialReplicationPadding(padL, padR, padT, padB):type(ctype)
      local groundtruth = module:forward(input)

      input = input:type(typename)
      local gmodule = nn.SpatialReplicationPadding(padL, padR, padT, padB):type(typename)
      local rescuda = gmodule:forward(input)

      local error = rescuda:double() - groundtruth:double()
      mytester:assertlt(error:abs():max(),
                        precision_forward_type(precision_forward, type),
                        string.format('error on state (forward) with %s', typename))
   end
end

function cunntest.SpatialReplicationPadding_backward()
   local batch = math.random(1,3)
   local plane = math.random(1,3)
   local sizeY = math.random(7,16)
   local sizeX = math.random(7,16)
   local padL = math.random(-3,3)
   local padR = math.random(-3,3)
   local padT = math.random(-3,3)
   local padB = math.random(-3,3)

   for k, typename in ipairs(typenames) do
      local input = torch.rand(batch, plane, sizeY, sizeX):type(typename)
      local gradOutput = torch.rand(
          batch, plane, sizeY + padT + padB, sizeX + padL + padR
       ):type(typename)

       local ctype = t2cpu[typename]
       input = input:type(ctype)
       gradOutput = gradOutput:type(ctype)
       local module = nn.SpatialReplicationPadding(padL, padR, padT, padB):type(ctype)
       module:forward(input)
       module:zeroGradParameters()
       local groundgrad = module:backward(input, gradOutput)

       input = input:type(typename)
       gradOutput = gradOutput:type(typename)
       local gmodule = nn.SpatialReplicationPadding(padL, padR, padT, padB):type(typename)
       gmodule:forward(input)
       gmodule:zeroGradParameters()
       local rescuda = gmodule:backward(input, gradOutput)

       local error = rescuda:double() - groundgrad:double()
       mytester:assertlt(error:abs():max(),
                         precision_backward_type(precision_backward, typename),
                         string.format('error on state (backward) with %s', typename))
   end
end

function cunntest.VolumetricReplicationPadding_forward()
   local batch = math.random(1,3)
   local plane = math.random(1,3)
   local sizeZ = math.random(7,16)
   local sizeY = math.random(7,16)
   local sizeX = math.random(7,16)
   local pleft = math.random(-3,3)
   local pright = math.random(-3,3)
   local ptop = math.random(-3,3)
   local pbottom = math.random(-3,3)
   local pfront = math.random(-3,3)
   local pback = math.random(-3,3)

   local tm = {}
   local title =
      string.format(
         'VolumetricReplicationPadding.forward %dx%dx%dx%dx%d -> ' ..
         '%dx%dx%dx%dx%d',
         batch, plane, sizeZ, sizeY, sizeX,
         batch, plane, sizeZ + pfront + pback, sizeY + ptop + pbottom,
         sizeX + pleft + pright)
   times[title] = tm

   local input = torch.rand(batch, plane, sizeZ, sizeY, sizeX)
   local module = nn.VolumetricReplicationPadding(pleft, pright, ptop, pbottom,
                                                  pfront, pback)
   local groundtruth = module:forward(input)
   local a = torch.Timer()
   for i = 1, nloop do
      groundtruth = module:forward(input)
   end
   tm.cpu = a:time().real

   input = input:cuda()
   local gmodule = nn.VolumetricReplicationPadding(pleft, pright, ptop, pbottom,
                                                   pfront, pback):cuda()
   local rescuda = gmodule:forward(input)
   a:reset()
   for i = 1, nloop do
      rescuda = gmodule:forward(input)
   end
   cutorch.synchronize()
   tm.gpu = a:time().real

   local error = rescuda:float() - groundtruth
   mytester:assertlt(error:abs():max(),
                     precision_forward, 'error on state (forward) ')
end

function cunntest.VolumetricReplicationPadding_backward()
   local batch = math.random(1,3)
   local plane = math.random(1,3)
   local sizeZ = math.random(7,16)
   local sizeY = math.random(7,16)
   local sizeX = math.random(7,16)
   local pleft = math.random(-3,3)
   local pright = math.random(-3,3)
   local ptop = math.random(-3,3)
   local pbottom = math.random(-3,3)
   local pfront = math.random(-3,3)
   local pback = math.random(-3,3)

   local tm = {}
   local title =
      string.format(
         'VolumetricReplicationPadding.backward %dx%dx%dx%dx%d -> ' ..
         '%dx%dx%dx%dx%d',
         batch, plane, sizeZ, sizeY, sizeX,
         batch, plane, sizeZ + pfront + pback, sizeY + ptop + pbottom,
         sizeX + pleft + pright)
   times[title] = tm

   local input = torch.rand(batch, plane, sizeZ, sizeY, sizeX)
   local gradOutput = torch.rand(
      batch, plane, sizeZ + pfront + pback, sizeY + ptop + pbottom,
      sizeX + pleft + pright
   )
   local module = nn.VolumetricReplicationPadding(pleft, pright, ptop, pbottom,
                                                  pfront, pback)
   module:forward(input)
   module:zeroGradParameters()
   local groundgrad = module:backward(input, gradOutput)
   local a = torch.Timer()
   for i = 1, nloop do
      module:zeroGradParameters()
      groundgrad = module:backward(input, gradOutput)
   end
   tm.cpu = a:time().real

   input = input:cuda()
   gradOutput = gradOutput:cuda()
   local gmodule = nn.VolumetricReplicationPadding(pleft, pright, ptop, pbottom,
                                                   pfront, pback):cuda()
   gmodule:forward(input)
   gmodule:zeroGradParameters()
   local rescuda = gmodule:backward(input, gradOutput)
   a:reset()
   for i = 1, nloop do
      gmodule:zeroGradParameters()
      rescuda = gmodule:backward(input, gradOutput)
   end
   cutorch.synchronize()
   tm.gpu = a:time().real

   local error = rescuda:float() - groundgrad
   mytester:assertlt(error:abs():max(),
                     precision_backward, 'error on state (backward) ')
end

function cunntest.GPU()
   local ndevice = cutorch.getDeviceCount()
   if ndevice < 2 then
      return
   end
   assert(nn.GPU, "Please update nn to latest version")

   local originaldevice = cutorch.getDevice()

   cutorch.setDevice(1)
   local linear = nn.Linear(3,4)
   local linear2 = linear:clone():float()
   linear.mybuffer = {torch.CudaTensor(3)}

   local gpu = nn.GPU(linear, 2, 1)
   gpu:cuda()

   mytester:assert(linear.mybuffer[1]:getDevice() == 2)
   mytester:assert(linear.weight:getDevice() == 2)
   mytester:assert(cutorch.getDevice() == originaldevice)

   local input = torch.CudaTensor(2,3):uniform(0,1)
   local output = gpu:forward(input)

   mytester:assert(linear.output:getDevice() == 2)
   mytester:assert(output:getDevice() == 1)
   mytester:assert(gpu._input:getDevice() == 2)

   local gradOutput = torch.CudaTensor(2,4):uniform(0,1)
   gpu:zeroGradParameters()
   mytester:assert(cutorch.getDevice() == 1)
   local gradInput = gpu:backward(input, gradOutput)

   mytester:assert(cutorch.getDevice() == 1)
   mytester:assert(gpu._gradOutput:getDevice() == 2)
   mytester:assert(linear.gradInput:getDevice() == 2)
   mytester:assert(gradInput:getDevice() == 1)

   mytester:assert(cutorch.getDevice() == 1)
   local input2, gradOutput2 = input:float(), gradOutput:float()
   local output2 = linear2:forward(input2)
   linear2:zeroGradParameters()
   local gradInput2 = linear2:backward(input2, gradOutput2)


   mytester:assertTensorEq(input2, input:float(), 0.000001)
   mytester:assertTensorEq(gradInput2, gradInput:float(), 0.000001)

   local params, gradParams = gpu:parameters()
   local params2, gradParams2 = linear2:parameters()

   for i=1,#params do
      mytester:assertTensorEq(params2[i], params[i]:float(), 0.000001)
      mytester:assertTensorEq(gradParams2[i], gradParams[i]:float(), 0.000001)
   end

   -- test serialize/deserialize

   local gpustr = torch.serialize(gpu)
   mytester:assert(cutorch.getDevice() == 1)
   local gpu2 = torch.deserialize(gpustr)
   mytester:assert(cutorch.getDevice() == 1)

   local output2 = gpu2:forward(input)

   mytester:assert(gpu2.modules[1].output:getDevice() == 2)
   mytester:assert(output2:getDevice() == 1)
   mytester:assert(gpu2._input:getDevice() == 2)

   gpu2:zeroGradParameters()
   mytester:assert(cutorch.getDevice() == 1)
   local gradInput2 = gpu2:backward(input, gradOutput)

   mytester:assert(cutorch.getDevice() == 1)
   mytester:assert(gpu2._gradOutput:getDevice() == 2)
   mytester:assert(gpu2.modules[1].gradInput:getDevice() == 2)
   mytester:assert(gradInput2:getDevice() == 1)

   mytester:assertTensorEq(input2, input2, 0.000001)
   mytester:assertTensorEq(gradInput2, gradInput2, 0.000001)

   local params, gradParams = gpu:parameters()
   local params2, gradParams2 = gpu2:parameters()

   for i=1,#params do
      mytester:assert(params2[i]:getDevice() == params[i]:getDevice())
      mytester:assert(gradParams2[i]:getDevice() == gradParams[i]:getDevice())
      mytester:assertTensorEq(params2[i]:float(), params[i]:float(), 0.000001)
      mytester:assertTensorEq(gradParams2[i]:float(), gradParams[i]:float(), 0.000001)
   end


   -- test table input/output
   local lin1, lin2 = nn.Linear(3,4), nn.Linear(3,4)
   local para = nn.ParallelTable():add(lin1):add(lin2)
   local para2 = para:clone():float()
   local gpu = nn.GPU(para, 2, 1)

   gpu:cuda()
   mytester:assert(lin1.weight:getDevice() == 2)
   mytester:assert(lin2.weight:getDevice() == 2)
   mytester:assert(cutorch.getDevice() == 1)

   local device3 = cutorch.getDeviceCount()
   local input = {
      torch.CudaTensor(2,3):uniform(0,1),
      cutorch.withDevice(device3, function() return torch.CudaTensor(2,3):uniform(0,1) end) -- tests input from multiple devices
   }
   local output = gpu:forward(input)

   mytester:assert(para.output[1]:getDevice() == 2)
   mytester:assert(para.output[2]:getDevice() == 2)
   mytester:assert(output[1]:getDevice() == 1)
   mytester:assert(output[2]:getDevice() == 1)
   mytester:assert(gpu._input[1]:getDevice() == 2)
   mytester:assert(gpu._input[2]:getDevice() == 2)

   local gradOutput = {
      torch.CudaTensor(2,4):uniform(0,1),
      cutorch.withDevice(device3, function() return torch.CudaTensor(2,4):uniform(0,1) end) -- tests gradOutput from multiple devices
   }

   gpu:zeroGradParameters()
   mytester:assert(cutorch.getDevice() == 1)
   local gradInput = gpu:backward(input, gradOutput)

   mytester:assert(cutorch.getDevice() == 1)
   mytester:assert(gpu._gradOutput[1]:getDevice() == 2)
   mytester:assert(gpu._gradOutput[2]:getDevice() == 2)
   mytester:assert(para.gradInput[1]:getDevice() == 2)
   mytester:assert(para.gradInput[2]:getDevice() == 2)
   mytester:assert(gradInput[1]:getDevice() == 1)
   mytester:assert(gradInput[2]:getDevice() == device3)

   local input2, gradOutput2 = {input[1]:float(), input[2]:float()}, {gradOutput[1]:float(), gradOutput[2]:float()}
   local output2 = para2:forward(input2)
   para2:zeroGradParameters()
   local gradInput2 = para2:backward(input2, gradOutput2)

   mytester:assertTensorEq(input2[1], input[1]:float(), 0.000001)
   mytester:assertTensorEq(input2[2], input[2]:float(), 0.000001)
   mytester:assertTensorEq(gradInput2[1], gradInput[1]:float(), 0.000001)
   mytester:assertTensorEq(gradInput2[2], gradInput[2]:float(), 0.000001)

   local params, gradParams = gpu:parameters()
   local params2, gradParams2 = para2:parameters()

   for i=1,#params do
      mytester:assertTensorEq(params2[i], params[i]:float(), 0.000001)
      mytester:assertTensorEq(gradParams2[i], gradParams[i]:float(), 0.000001)
   end

   -- test that it handles reduction in input/output size

   input[2], gradOutput[2] = nil, nil
   para.modules[2] = nil
   para.output[2] = nil
   para.gradInput[2] = nil

   local output = gpu:forward(input)

   mytester:assert(#gpu._input == 1)
   mytester:assert(#output == 1)

   local gradInput = gpu:backward(input, gradOutput)

   mytester:assert(#gpu._gradOutput == 1)
   mytester:assert(#gradInput == 1)

   -- test sequential multi-GPUs

   local mlp = nn.Sequential()
   for device=1,ndevice do
      local outdevice = device == ndevice and 1 or device
      mlp:add(nn.GPU(nn.Linear(3,3), device, outdevice))
      mytester:assert(cutorch.getDevice() == 1)
   end
   mlp:cuda()
   mytester:assert(cutorch.getDevice() == 1)

   local input = torch.CudaTensor(2,3):uniform(0,1)
   local gradOutput = torch.CudaTensor(2,3):uniform(0,1)

   local output = mlp:forward(input)
   mlp:zeroGradParameters()
   local gradInput = mlp:backward(input, gradOutput)

   -- test CPU only

   local params, gradParams = mlp:parameters()

   mlp:float()

   local input2, gradOutput2 = input:float(), gradOutput:float()

   local _cutorch = cutorch
   cutorch = nil

   local output2 = mlp:forward(input2)
   mlp:zeroGradParameters()
   local gradInput2 = mlp:backward(input2, gradOutput2)

   cutorch = _cutorch

   mytester:assertTensorEq(output:float(), output2, 0.000001)
   mytester:assertTensorEq(gradInput:float(), gradInput2, 0.000001)

   local params2, gradParams2 = mlp:parameters()

   for i=1,#params do
      mytester:assertTensorEq(params[i]:float(), params2[i], 0.000001)
      mytester:assertTensorEq(gradParams[i]:float(), gradParams2[i], 0.000001)
   end

   cutorch.setDevice(originaldevice)
end

local function setUp()
   cutorch.setDevice(1)
end

for k,v in pairs(cunntest.__tests) do
   cunntest.__tests[k] = function()
      setUp()
      v()
   end
end

local function initSeed(seed)
   seed = seed or math.floor((torch.tic() * 1e5) % 1e9)
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
   checkHalf()
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

-- add alias, in same format as eg cutorch.test()
cunn = cunn or {}
cunn.test = nn.testcuda
