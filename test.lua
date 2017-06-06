local cunntest = torch.TestSuite()
local ffi = require 'ffi'
local precision_forward = 1e-4
local precision_backward = 1e-2
local nloop = 1
local times = {}

-- load THC
local THC = ffi.os == 'Windows' and ffi.load('THC') or ffi.C

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

local function half_max_error(maxabs)
  -- arbitrarily double the precision limit
  return 2 * ((maxabs and (2^(math.floor(math.log(maxabs) / math.log(2)))) * (2^(-10))) or 0)
end

-- half has additional error on top of double/float
local function precision_forward_type(precision_f, tensor_type, maxabs)
   if (tensor_type == 'torch.CudaHalfTensor') then
      return 1e-2 + precision_f + half_max_error(maxabs)
   else
      return precision_f
   end
end

local function precision_backward_type(precision_b, tensor_type, maxabs)
   if (tensor_type == 'torch.CudaHalfTensor') then
      return 1e-1 + precision_b + half_max_error(maxabs)
   else
      return precision_b
   end
end

local function precision_backward_conv_weightbias(precision_b, tensor_type, maxabs)
   if (tensor_type == 'torch.CudaHalfTensor') then
      -- cudnn uses 8 here
      return 2 + precision_b + half_max_error(maxabs)
   else
      return precision_b
   end
end

local function makeNonContiguous(tensor)
   size = tensor:size()
   local osize = {}
   for i = 1, #size do osize[i] = size[i] end
   -- randomly inflate a few dimensions in osize
   for i = 1, 3 do
      local dim = torch.random(1,#osize)
      local add = torch.random(4, 15)
      osize[dim] = osize[dim] + add
   end
   local input = torch[tensor:type():match('torch.(%a+)')]()
   input:resize(torch.LongStorage(osize))
   -- now extract the input of correct size from 'input'
   for i = 1, #size do
      if input:size(i) ~= size[i] then
         local bounds = torch.random(1, input:size(i) - size[i] + 1)
         input = input:narrow(i, bounds, size[i])
      end
   end
   input:copy(tensor)
   return input
end

local function pointwise_forward(proto_module, name, max_error)
   local size = math.random(1,100)
   if name == 'GatedLinearUnit' then size = size*2 end

   for k, typename in ipairs(typenames) do
      local input = torch.randn(size):type(typename)
      local ctype = t2cpu[typename]
      local input = makeNonContiguous(input:type(ctype))
      if name == 'Sqrt' then input:abs() end
      local sconv = proto_module:type(ctype)
      local groundtruth = sconv:forward(input)

      input = makeNonContiguous(input:type(typename))
      local gconv = proto_module:clone():type(typename)
      local rescuda = gconv:forward(input)

      local error = rescuda:double() - groundtruth:double()
      mytester:assertlt(error:abs():max(), precision_forward_type(max_error, typename),
        string.format('error on state (forward) with %s', typename))
    end
end

local function pointwise_backward(proto_module, name, max_error)
   local size = math.random(1,100)
   if name == 'GatedLinearUnit' then size = size*2 end

   for k, typename in ipairs(typenames) do
      local input = torch.randn(size):type(typename)
      local gradOutput = torch.randn(size):type(typename)
      if name == 'GatedLinearUnit' then gradOutput = torch.randn(size/2) end

      local ctype = t2cpu[typename]
      input = makeNonContiguous(input:type(ctype))
      gradOutput = makeNonContiguous(gradOutput:type(ctype))
      if name == 'Sqrt' then input:abs() end
      local sconv = proto_module:type(ctype)
      sconv:forward(input)
      local groundgrad = sconv:backward(input, gradOutput)

      input = makeNonContiguous(input:type(typename))
      gradOutput = makeNonContiguous(gradOutput:type(typename))
      local gconv = proto_module:clone():type(typename)
      gconv:forward(input)
      local rescuda = gconv:backward(input, gradOutput)

      local error = rescuda:double() - groundgrad:double()

      mytester:assertlt(error:abs():max(),
        precision_backward_type(max_error, typename, rescuda:abs():max()),
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
      local gradOutput = makeNonContiguous(torch.randn(size))
      gradOutput = makeNonContiguous(gradOutput:type(ctype))
      local sconv = proto_module:type(ctype)
      local groundgrad = sconv:backward(input, gradOutput)
      mytester:assertTensorEq(groundgrad:double(),
                              gradOutput:double(),
                              0.000001,
                              string.format("inplace not respected for %s", ctype))

      input = makeNonContiguous(torch.randn(size))
      input = makeNonContiguous(input:type(typename))
      if name == 'Sqrt' then input:abs() end
      gradOutput = makeNonContiguous(torch.randn(size))
      gradOutput = makeNonContiguous(gradOutput:type(typename))
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

function cunntest.GatedLinearUnit_forward()
   pointwise_forward(nn.GatedLinearUnit(), 'GatedLinearUnit', precision_forward)
end

function cunntest.GatedLinearUnit_backward()
   pointwise_backward(nn.GatedLinearUnit(), 'GatedLinearUnit', precision_backward)
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
      local input_gpu = makeNonContiguous(torch.CudaTensor(size, size):zero():type(typename))
      module_gpu:forward(input_gpu)

      local gradOutput_gpu = makeNonContiguous(torch.CudaTensor(size, size):fill(1):type(typename))
      local gradInput_gpu = module_gpu:backward(input_gpu, gradOutput_gpu)

      mytester:assertTensorEq(gradInput_gpu:double(),
                              torch.DoubleTensor(size, size):zero(),
                              0.000001, "error in sqrt backward singularity")

      -- Verify CPU and GPU zero behavior equivalency
      local ctype = t2cpu[typename]
      local module_cpu = nn.Sqrt():type(ctype)
      local input_cpu = makeNonContiguous(input_gpu:type(ctype))
      module_cpu:forward(input_cpu)

      local gradOutput_cpu = makeNonContiguous(gradOutput_gpu:type(ctype))
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
  local r = math.random()
  pointwise_forward(nn.SoftShrink(r), 'SoftShrink', precision_forward)
end

function cunntest.SoftShrink_backward()
  local r = math.random()
  pointwise_backward(nn.SoftShrink(r), 'SoftShrink', precision_backward)
end

function cunntest.SoftShrink_transposed()
  local r = math.random()
  pointwise_transposed(nn.SoftShrink(r), 'SoftShrink', precision_backward)
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

   local input = makeNonContiguous(torch.randn(bs, dim, h, w))
   local sconv = nn.SpatialSoftMax()
   local groundtruth = sconv:forward(input)
   local gradOutput = makeNonContiguous(groundtruth:clone():fill(0.5))
   local gradInput = sconv:backward(input, gradOutput)

   input = makeNonContiguous(input:cuda())
   gradOutput = makeNonContiguous(gradOutput:cuda())
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
      input = makeNonContiguous(input:type(ctype))
      local sconv = nn.LogSoftMax():type(ctype)
      local groundtruth = sconv:forward(input)

      input = makeNonContiguous(input:type(typename))
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
      input = makeNonContiguous(input:type(ctype))
      gradOutput = makeNonContiguous(gradOutput:type(ctype))
      local sconv = nn.LogSoftMax():type(ctype)
      sconv:forward(input)
      local groundgrad = sconv:backward(input, gradOutput)

      input = makeNonContiguous(input:type(typename))
      gradOutput = makeNonContiguous(gradOutput:type(typename))
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
      input = makeNonContiguous(input:type(ctype))
      local sconv = nn.SpatialLogSoftMax():type(ctype)
      local groundtruth = sconv:forward(input):type(ctype)

      input = makeNonContiguous(input:type(typename))
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
      gradOutput = makeNonContiguous(gradOutput:type(ctype))
      local sconv = nn.SpatialLogSoftMax():type(ctype)
      sconv:forward(input)
      local groundgrad = sconv:backward(input, gradOutput)

      input = makeNonContiguous(input:type(typename))
      gradOutput = makeNonContiguous(gradOutput:type(typename))
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

      input = makeNonContiguous(input:type(typename))
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
      input = makeNonContiguous(input:type(ctype))
      gradOutput = makeNonContiguous(gradOutput:type(ctype))
      local sconv = nn.SpatialLogSoftMax():type(ctype)
      sconv:forward(input)
      local groundgrad = sconv:backward(input, gradOutput)

      input = makeNonContiguous(input:type(typename))
      gradOutput = makeNonContiguous(gradOutput:type(typename))
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

   local input = makeNonContiguous(torch.randn(bs, nin))
   local sconv = nn.Euclidean(nin, nout)
   local groundtruth = sconv:forward(input)
   local a = torch.Timer()
   for i = 1,nloop do
      groundtruth = sconv:forward(input)
   end
   tm.cpu = a:time().real

   input = makeNonContiguous(input:cuda())
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

   local input = makeNonContiguous(torch.randn(bs, nin))
   local gradOutput = makeNonContiguous(torch.randn(bs, nout))
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

   input = makeNonContiguous(input:cuda())
   gradOutput = makeNonContiguous(gradOutput:cuda())
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

   local input = makeNonContiguous(torch.randn(bs, nin))
   local sconv = nn.WeightedEuclidean(nin, nout)
   local groundtruth = sconv:forward(input)
   local a = torch.Timer()
   for i = 1,nloop do
      groundtruth = sconv:forward(input)
   end
   tm.cpu = a:time().real

   input = makeNonContiguous(input:cuda())
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

   local input = makeNonContiguous(torch.randn(bs, nin))
   local gradOutput = makeNonContiguous(torch.randn(bs, nout))
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

   input = makeNonContiguous(input:cuda())
   gradOutput = makeNonContiguous(gradOutput:cuda())
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

    for k, typename in ipairs(typenames) do
        if typename ~= "torch.CudaHalfTensor" then
            local ctype = t2cpu[typename]
            local module = nn.SparseLinear(ini,inj):type(ctype)
            local sslin = module
            local gslin = module:clone():type(typename)

            -- Create a random sparse vector
            local input = {}
            for i=1,inb do
                local nnz = math.random(5, 10)
                local inds = torch.randperm(ini)[{{1,nnz}}]
                input[i] = torch.Tensor(nnz, 2):type(ctype)
                input[i]:select(2,1):copy(inds)
                input[i]:select(2,2):copy(torch.rand(nnz):type(typename):type(ctype))
            end

            local groundtruth = sslin:forward(input)
            sslin:zeroGradParameters()

            for i,v in ipairs(input) do input[i] = input[i]:type(typename) end
            local rescuda = gslin:forward(input)
            gslin:zeroGradParameters()

            local error = rescuda:double() - groundtruth:double()
            mytester:assertlt(error:abs():max(), precision_forward_type(precision_forward, typename),
                string.format('error on state (forward) with %s', typename))
        end
    end
end

function cunntest.SparseLinear_backward()
    local inb = math.random(5,10)
    local ini = math.random(50,100)
    local inj = math.random(5,10)

    for k, typename in ipairs(typenames) do
        if typename ~= "torch.CudaHalfTensor" then
            local ctype = t2cpu[typename]
            local gslin = nn.SparseLinear(ini,inj):type(typename)
            local sslin = nn.Linear(ini,inj):type(ctype)
            gslin.weight = sslin.weight:clone():type(typename)
            gslin.bias = sslin.bias:clone():type(typename)

            -- Create a random sparse vector
            local input = {}
            local nonsparse = torch.zeros(inb, ini):type(ctype)
            for i=1,inb do
                local nnz = math.random(3, 5)
                local inds = torch.randperm(ini)[{{1,nnz}}]
                input[i] = torch.Tensor(nnz, 2):type(ctype)
                input[i]:select(2,1):copy(inds)
                input[i]:select(2,2):copy(torch.rand(nnz):type(typename):type(ctype))
                nonsparse[i]:scatter(1, input[i]:select(2,1):long(), input[i]:select(2,2))
            end

            local gradOutput = makeNonContiguous(torch.randn(inb, inj):type(typename):type(ctype))
            sslin:forward(nonsparse)
            local groundgrad = sslin:backward(nonsparse, gradOutput)
            sslin:zeroGradParameters()
            local groundweight = sslin.gradWeight
            local groundbias = sslin.gradBias

            for i,v in ipairs(input) do input[i] = input[i]:type(typename) end
            gradOutput = makeNonContiguous(gradOutput:type(typename))
            gslin:forward(input)
            local rescuda = gslin:backward(input, gradOutput)
            gslin:zeroGradParameters()
            local weightcuda = gslin.gradWeight
            local biascuda = gslin.gradBias

            local werror = weightcuda:double() - groundweight:double()
            local berror = biascuda:double() - groundbias:double()

            mytester:assertlt(werror:abs():max(), precision_backward_type(precision_backward, typename),
                string.format('error on weight (backward) with %s', typename))
            mytester:assertlt(berror:abs():max(), precision_backward_type(precision_backward, typename),
                string.format('error on bias (backward) with %s', typename))

            gslin:updateParameters(.1)
            sslin:updateParameters(.1)
            werror = gslin.weight:double() - sslin.weight:double()
            berror = gslin.bias:double() - sslin.bias:double()

            mytester:assertlt(werror:abs():max(), precision_backward_type(precision_backward, typename),
                string.format('error on weight (update) with %s', typename))
            mytester:assertlt(berror:abs():max(), precision_backward_type(precision_backward, typename),
                string.format('error on bias (update) with %s', typename))

            gslin:zeroGradParameters()
        end
    end
end

local function BatchNormalization_forward(moduleName, inputSize)
   local planes = inputSize[2]

   for k, typename in ipairs(typenames) do
      local input = torch.randn(table.unpack(inputSize)):type(typename)

      local ctype = t2cpu[typename]
      input = makeNonContiguous(input:type(ctype))
      local sbnorm = nn[moduleName](planes):type(ctype)
      local groundtruth = sbnorm:forward(input)

      input = makeNonContiguous(input:type(typename))
      local gbnorm = nn[moduleName](planes):type(typename)
      gbnorm.weight = sbnorm.weight:type(typename)
      gbnorm.bias = sbnorm.bias:type(typename)
      local rescuda = gbnorm:forward(input)

      local error = rescuda:double() - groundtruth:double()
      mytester:assertlt(error:abs():max(), precision_forward_type(precision_forward, typename, rescuda:abs():max()),
         string.format('error on state (forward) with %s', typename))
      mytester:assertlt((gbnorm.running_mean:double() - sbnorm.running_mean:double()):abs():max(),
         precision_forward_type(precision_forward, typename, gbnorm.running_mean:abs():max()),
         string.format('error on running_mean (forward) with %s', typenanme))
      mytester:assertlt((gbnorm.running_var:double() - sbnorm.running_var:double()):abs():max(),
         precision_forward_type(precision_forward, typename, gbnorm.running_var:abs():max()),
         string.format('error on running_var (forward) with %s', typename))
   end
end

local function BatchNormalization_forward_inference(moduleName, inputSize)
   local planes = inputSize[2]

   for k, typename in ipairs(typenames) do
      local input = torch.randn(table.unpack(inputSize)):type(typename)

      local ctype = t2cpu[typename]
      input = makeNonContiguous(input:type(ctype))
      local sbnorm = nn[moduleName](planes):type(ctype)
      sbnorm.running_mean:normal(1, 2)
      sbnorm.running_var:uniform(1e-3, 2)
      sbnorm.running_var = sbnorm.running_var:type(typename):type(ctype)
      sbnorm.running_mean = sbnorm.running_mean:type(typename):type(ctype)

      sbnorm:evaluate()
      local groundtruth = sbnorm:forward(input)

      input = makeNonContiguous(input:type(typename))
      local gbnorm = nn[moduleName](planes):type(typename)
      gbnorm:evaluate()
      gbnorm.weight = sbnorm.weight:type(typename)
      gbnorm.bias = sbnorm.bias:type(typename)
      gbnorm.running_mean = sbnorm.running_mean:type(typename)
      gbnorm.running_var = sbnorm.running_var:type(typename)
      local rescuda = gbnorm:forward(input)

      local error = rescuda:double() - groundtruth:double()
      mytester:assertlt(error:abs():max(), precision_forward_type(precision_forward, typename, rescuda:abs():max()),
         string.format('error on state (forward evaluate) with %s', typename))
   end
end

local function BatchNormalization_backward(moduleName, mode, inputSize, backwardFn)
   assert(mode == 'training' or mode == 'evaluation', 'invalid mode')

   local planes = inputSize[2]

   for k, typename in ipairs(typenames) do
      local input = torch.randn(table.unpack(inputSize)):type(typename)
      local gradOutput = torch.randn(table.unpack(inputSize)):type(typename)

      local ctype = t2cpu[typename]
      input = makeNonContiguous(input:type(ctype))
      gradOutput = makeNonContiguous(gradOutput:type(ctype))
      local sbnorm = nn[moduleName](planes):type(ctype)
      if mode == 'training' then
        sbnorm:training()
      else
        sbnorm:evaluate()
      end
      sbnorm:forward(input)
      sbnorm:zeroGradParameters()
      local groundgrad = backwardFn(sbnorm, input, gradOutput)
      local groundweight = sbnorm.gradWeight
      local groundbias = sbnorm.gradBias

      input = makeNonContiguous(input:type(typename))
      gradOutput = makeNonContiguous(gradOutput:type(typename))
      local gbnorm = nn[moduleName](planes):type(typename)
      if mode == 'training' then
        gbnorm:training()
      else
        gbnorm:evaluate()
      end
      gbnorm.weight = sbnorm.weight:type(typename)
      gbnorm.bias = sbnorm.bias:type(typename)
      gbnorm:forward(input)
      gbnorm:zeroGradParameters()
      local rescuda = backwardFn(gbnorm, input, gradOutput)
      local weightcuda = gbnorm.gradWeight
      local biascuda = gbnorm.gradBias

      local error = rescuda:double() - groundgrad:double()
      local werror = weightcuda:double() - groundweight:double()
      local berror = biascuda:double() - groundbias:double()

      local backerror = precision_backward_type(precision_backward, typename, rescuda:abs():max())
      if typename == 'torch.CudaHalfTensor' and (mode == 'training') then
        -- this correction is empirical; mean can be off by roughly 4e-4, multiplied by roughly stdval^2.
        backerror = backerror + (sbnorm.save_std:max())^2 * 4e-4
      end
      mytester:assertlt(error:abs():max(),
        backerror,
        string.format('error on state (backward) with %s', typename))
      mytester:assertlt(werror:abs():max(),
        precision_backward_type(precision_backward, typename, weightcuda:abs():max()),
        string.format('error on weight (backward) with %s', typename))
      mytester:assertlt(berror:abs():max(),
        precision_backward_type(precision_backward, typename, biascuda:abs():max()),
        string.format('error on bias (backward) with %s', typename))
    end
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
         input = makeNonContiguous(input:type(ctype))
         local sconv = nn.SpatialConvolutionMM(from,to,ki,kj,si,sj,padW,padH):type(ctype)
         if noBias then
            sconv:noBias()
         end
         local groundtruth = sconv:forward(input)

         input = makeNonContiguous(input:type(typename))
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
         input = makeNonContiguous(input:type(ctype))
         local sconv = nn.SpatialConvolutionMM(from,to,ki,kj,si,sj,padW,padH):type(ctype)
         if noBias then
            sconv:noBias()
         end
         local groundtruth = sconv:forward(input)

         input = makeNonContiguous(input:type(typename))
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
         input = makeNonContiguous(input:type(ctype))
         gradOutput = makeNonContiguous(gradOutput:type(ctype))
         local sconv = nn.SpatialConvolutionMM(from,to,ki,kj,si,sj,padW,padH):type(ctype)
         if noBias then
            sconv:noBias()
         end
         sconv:forward(input)
         sconv:zeroGradParameters()
         local groundgrad = sconv:backward(input, gradOutput)
         local groundweight = sconv.gradWeight
         local groundbias = sconv.gradBias

         input = makeNonContiguous(input:type(typename))
         gradOutput = makeNonContiguous(gradOutput:type(typename))
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
         mytester:assertlt(werror:abs():max(),
            precision_backward_conv_weightbias(precision_backward, typename, weightcuda:abs():max()),
            string.format('error on weight (backward) with %s', typename))

         if gconv.bias then
            local berror = gconv.gradBias:double() - groundbias:double()
            mytester:assertlt(berror:abs():max(),
                precision_backward_conv_weightbias(precision_backward, typename, gconv.gradBias:abs():max()),
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
         input = makeNonContiguous(input:type(ctype))
         gradOutput = makeNonContiguous(gradOutput:type(ctype))
         local sconv = nn.SpatialConvolutionMM(from,to,ki,kj,si,sj,padW,padH):type(ctype)
         if noBias then
            sconv:noBias()
         end
         sconv:forward(input)
         sconv:zeroGradParameters()
         local groundgrad = sconv:backward(input, gradOutput)
         local groundweight = sconv.gradWeight
         local groundbias = sconv.gradBias

         input = makeNonContiguous(input:type(typename))
         gradOutput = makeNonContiguous(gradOutput:type(typename))
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
         mytester:assertlt(werror:abs():max(),
            precision_backward_conv_weightbias(precision_backward, typename, weightcuda:abs():max()),
            string.format('error on weight (backward) with %s', typename))
         if gconv.bias then
            local berror = gconv.gradBias:double() - groundbias:double()
            mytester:assertlt(berror:abs():max(),
                precision_backward_conv_weightbias(precision_backward, typename, gconv.gradBias:abs():max()),
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
       if typename ~= "torch.CudaHalfTensor" then
           local input = torch.randn(from,inj,ini):type(typename)

           local ctype = t2cpu[typename]
           input = makeNonContiguous(input:type(ctype))
           local sconv = nn.SpatialConvolutionLocal(from,to,ini,inj,ki,kj,si,sj,padW,padH):type(ctype)
           local groundtruth = sconv:forward(input)

           input = makeNonContiguous(input:type(typename))
           local gconv = nn.SpatialConvolutionLocal(from,to,ini,inj,ki,kj,si,sj,padW,padH):type(typename)
           gconv.weight = sconv.weight:type(typename)
           gconv.bias = sconv.bias:type(typename)
           local rescuda = gconv:forward(input)

           local error = rescuda:double() - groundtruth:double()
           mytester:assertlt(error:abs():max(), precision_forward_type(precision_forward, typename),
                             string.format('error on state (forward) with %s', typename))
       end
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
       if typename ~= "torch.CudaHalfTensor" then
           local input = torch.randn(bs,from,inj,ini):type(typename)

           local ctype = t2cpu[typename]
           input = makeNonContiguous(input:type(ctype))
           local sconv = nn.SpatialConvolutionLocal(from,to,ini,inj,ki,kj,si,sj,padW,padH):type(ctype)
           local groundtruth = sconv:forward(input)

           input = makeNonContiguous(input:type(typename))
           local gconv = nn.SpatialConvolutionLocal(from,to,ini,inj,ki,kj,si,sj,padW,padH):type(typename)
           gconv.weight = sconv.weight:type(typename)
           gconv.bias = sconv.bias:type(typename)
           local rescuda = gconv:forward(input)

           local error = rescuda:double() - groundtruth:double()
           mytester:assertlt(error:abs():max(), precision_forward_type(precision_forward, typename),
                             string.format('error on state (forward) with %s', typename))
       end
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
       if typename ~= "torch.CudaHalfTensor" then
           local input = torch.randn(from,inj,ini):type(typename)
           local gradOutput = torch.randn(to,outj,outi):type(typename)

           local ctype = t2cpu[typename]
           input = makeNonContiguous(input:type(ctype))
           gradOutput = makeNonContiguous(gradOutput:type(ctype))
           local sconv = nn.SpatialConvolutionLocal(from,to,ini,inj,ki,kj,si,sj,padW,padH):type(ctype)
           sconv:forward(input)
           sconv:zeroGradParameters()
           local groundgrad = sconv:backward(input, gradOutput)
           local groundweight = sconv.gradWeight
           local groundbias = sconv.gradBias

           input = makeNonContiguous(input:type(typename))
           gradOutput = makeNonContiguous(gradOutput:type(typename))
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
           mytester:assertlt(werror:abs():max(),
                             precision_backward_conv_weightbias(precision_backward, typename, weightcuda:abs():max()),
                             string.format('error on weight (backward) with %s', typename))
           mytester:assertlt(berror:abs():max(),
                             precision_backward_conv_weightbias(precision_backward, typename, biascuda:abs():max()),
                             string.format('error on bias (backward) with %s', typename))
       end
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
       if typename ~= "torch.CudaHalfTensor" then
           local input = torch.randn(bs,from,inj,ini):type(typename)
           local gradOutput = torch.randn(bs,to,outj,outi):type(typename)

           local ctype = t2cpu[typename]
           input = makeNonContiguous(input:type(ctype))
           gradOutput = makeNonContiguous(gradOutput:type(ctype))
           local sconv = nn.SpatialConvolutionLocal(from,to,ini,inj,ki,kj,si,sj,padW,padH):type(ctype)
           sconv:forward(input)
           sconv:zeroGradParameters()
           local groundgrad = sconv:backward(input, gradOutput)
           local groundweight = sconv.gradWeight
           local groundbias = sconv.gradBias

           input = makeNonContiguous(input:type(typename))
           gradOutput = makeNonContiguous(gradOutput:type(typename))
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
           mytester:assertlt(werror:abs():max(),
                             precision_backward_conv_weightbias(precision_backward, typename, weightcuda:abs():max()),
                             string.format('error on weight (backward) with %s', typename))
           mytester:assertlt(berror:abs():max(),
                             precision_backward_conv_weightbias(precision_backward, typename, biascuda:abs():max()),
                             string.format('error on bias (backward) with %s', typename))
       end
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
         input = makeNonContiguous(input:type(ctype))
         local sconv = nn.SpatialFullConvolution(from,to,ki,kj,si,sj,padW,padH,adjW,adjH):type(ctype)
         if noBias then
            sconv:noBias()
         end
         local groundtruth = sconv:forward(input)

         input = makeNonContiguous(input:type(typename))
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
         input = makeNonContiguous(input:type(ctype))
         local sconv = nn.SpatialFullConvolution(from,to,ki,kj,si,sj,padW,padH,adjW,adjH):type(ctype)
         if noBias then
            sconv:noBias()
         end
         local groundtruth = sconv:forward(input)

         input = makeNonContiguous(input:type(typename))
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
         input = makeNonContiguous(input:type(ctype))
         local sconv = nn.SpatialFullConvolution(from,to,ki,kj,si,sj,padW,padH,adjW,adjH):type(ctype)
         if noBias then
            sconv:noBias()
         end
         local output = sconv:forward(input)
         local gradOutput = makeNonContiguous(output:clone():normal())
         sconv:zeroGradParameters()
         local groundgrad = sconv:backward(input, gradOutput)
         local groundweight = sconv.gradWeight
         local groundbias = sconv.gradBias

         input = (input:type(typename))
         gradOutput = makeNonContiguous(gradOutput:type(typename))
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
         mytester:assertlt(werror:abs():max(),
            precision_backward_conv_weightbias(precision_backward, typename, weightcuda:abs():max()),
            string.format('error on weight (backward) with %s', typename))

         if gconv.bias then
            local berror = gconv.gradBias:double() - groundbias:double()
            mytester:assertlt(berror:abs():max(),
               precision_backward_conv_weightbias(precision_backward, typename, gconv.gradBias:abs():max()),
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
         input = makeNonContiguous(input:type(ctype))
         local sconv = nn.SpatialFullConvolution(from,to,ki,kj,si,sj,padW,padH,adjW,adjH):type(ctype)
         if noBias then
            sconv:noBias()
         end
         local output = sconv:forward(input)
         local gradOutput = makeNonContiguous(output:clone():normal())
         sconv:zeroGradParameters()
         local groundgrad = sconv:backward(input, gradOutput)
         local groundweight = sconv.gradWeight
         local groundbias = sconv.gradBias

         input = makeNonContiguous(input:type(typename))
         gradOutput = makeNonContiguous(gradOutput:type(typename))
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
         mytester:assertlt(werror:abs():max(),
            precision_backward_conv_weightbias(precision_backward, typename, weightcuda:abs():max()),
            string.format('error on weight (backward) with %s', typename))
         if gconv.bias then
            local berror = gconv.gradBias:double() - groundbias:double()
            mytester:assertlt(berror:abs():max(),
               precision_backward_conv_weightbias(precision_backward, typename, gconv.gradBias:abs():max()),
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
      input = makeNonContiguous(input:type(ctype))
      local sconv = nn.SpatialDilatedConvolution(from,to,ki,kj,si,sj,padW,padH,dilationW,dilationH):type(ctype)
      local groundtruth = sconv:forward(input)

      input = makeNonContiguous(input:type(typename))
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
   local dilationW = math.random(1,10)
   local dilationH = math.random(1,10)
   local ini = (outi - 1) * si - 2 * padW + dilationW * (ki-1) + 1
   local inj = (outj - 1) * sj - 2 * padH + dilationH * (kj-1) + 1

   for k, typename in ipairs(typenames) do
      local input = torch.randn(bs,from,inj,ini):type(typename)

      local ctype = t2cpu[typename]
      input = makeNonContiguous(input:type(ctype))
      local sconv = nn.SpatialDilatedConvolution(from,to,ki,kj,si,sj,padW,padH,dilationW,dilationH):type(ctype)
      local groundtruth = sconv:forward(input)

      input = makeNonContiguous(input:type(typename))
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
   local dilationW = math.random(1,10)
   local dilationH = math.random(1,10)
   local ini = (outi - 1) * si - 2 * padW + dilationW * (ki-1) + 1
   local inj = (outj - 1) * sj - 2 * padH + dilationH * (kj-1) + 1

   for k, typename in ipairs(typenames) do
      local input = torch.randn(from,inj,ini):type(typename)

      local ctype = t2cpu[typename]
      input = makeNonContiguous(input:type(ctype))
      local sconv = nn.SpatialDilatedConvolution(from,to,ki,kj,si,sj,padW,padH,dilationW,dilationH):type(ctype)
      local output = sconv:forward(input)
      local gradOutput = makeNonContiguous(output:clone():normal())
      sconv:zeroGradParameters()
      local groundgrad = sconv:backward(input, gradOutput)
      local groundweight = sconv.gradWeight
      local groundbias = sconv.gradBias

      input = makeNonContiguous(input:type(typename))
      gradOutput = makeNonContiguous(gradOutput:type(typename))
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
      mytester:assertlt(werror:abs():max(),
         precision_backward_conv_weightbias(precision_backward, typename, weightcuda:abs():max()),
         string.format('error on weight (backward) with %s', typename))
      mytester:assertlt(berror:abs():max(),
         precision_backward_conv_weightbias(precision_backward, typename, biascuda:abs():max()),
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
   local dilationW = math.random(1,10)
   local dilationH = math.random(1,10)
   local ini = (outi - 1) * si - 2 * padW + dilationW * (ki-1) + 1
   local inj = (outj - 1) * sj - 2 * padH + dilationH * (kj-1) + 1

   for k, typename in ipairs(typenames) do
      local input = torch.randn(bs,from,inj,ini):type(typename)

      local ctype = t2cpu[typename]
      input = makeNonContiguous(input:type(ctype))
      local sconv = nn.SpatialDilatedConvolution(from,to,ki,kj,si,sj,padW,padH,dilationW,dilationH):type(ctype)
      local output = sconv:forward(input)
      local gradOutput = makeNonContiguous(output:clone():normal())
      sconv:zeroGradParameters()
      local groundgrad = sconv:backward(input, gradOutput)
      local groundweight = sconv.gradWeight
      local groundbias = sconv.gradBias

      input = makeNonContiguous(input:type(typename))
      gradOutput = makeNonContiguous(gradOutput:type(typename))
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
      mytester:assertlt(werror:abs():max(),
         precision_backward_conv_weightbias(precision_backward, typename, weightcuda:abs():max()),
         string.format('error on weight (backward) with %s', typename))
      mytester:assertlt(berror:abs():max(),
         precision_backward_conv_weightbias(precision_backward, typename, biascuda:abs():max()),
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
      input = makeNonContiguous(input:type(ctype))
      local sconv = nn.SpatialSubSampling(from,ki,kj,si,sj):type(ctype)
      local groundtruth = sconv:forward(input)

      input = makeNonContiguous(input:type(typename))
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
      input = makeNonContiguous(input:type(ctype))
      local sconv = nn.SpatialSubSampling(from,ki,kj,si,sj):type(ctype)
      local groundtruth = sconv:forward(input)

      input = makeNonContiguous(input:type(typename))
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
      input = makeNonContiguous(input:type(ctype))
      gradOutput = makeNonContiguous(gradOutput:type(ctype))
      local sconv = nn.SpatialSubSampling(from,ki,kj,si,sj):type(ctype)
      sconv:forward(input)
      sconv:zeroGradParameters()
      local groundgrad = sconv:backward(input, gradOutput)
      local groundweight = sconv.gradWeight
      local groundbias = sconv.gradBias

      input = makeNonContiguous(input:type(typename))
      gradOutput = makeNonContiguous(gradOutput:type(typename))
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
      local input = torch.randn(bs,from,inj,ini):type(typename)
      local gradOutput = torch.randn(bs,to,outj,outi):type(typename)

      local ctype = t2cpu[typename]
      input = makeNonContiguous(input:type(ctype))
      gradOutput = makeNonContiguous(gradOutput:type(ctype))
      local sconv = nn.SpatialSubSampling(from,ki,kj,si,sj):type(ctype)
      sconv:forward(input)
      sconv:zeroGradParameters()
      local groundgrad = sconv:backward(input, gradOutput)
      local groundweight = sconv.gradWeight
      local groundbias = sconv.gradBias

      input = makeNonContiguous(input:type(typename))
      gradOutput = makeNonContiguous(gradOutput:type(typename))
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

      -- FIXME: SpatialSubSampling accumulates directly to real, causes
      -- precision issues with half, so we double the error tolerance
      mytester:assertlt(error:abs():max(),
          2*precision_backward_type(precision_backward, typename, rescuda:abs():max()),
          string.format('error on state (backward) with %s', typename))
      mytester:assertlt(werror:abs():max(),
          2*precision_backward_type(precision_backward, typename, weightcuda:abs():max()),
          string.format('error on weight (backward) with %s', typename))
      mytester:assertlt(berror:abs():max(),
          2*precision_backward_type(precision_backward, typename, biascuda:abs():max()),
          string.format('error on bias (backward) with %s', typename))
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
   local padi = math.random(0,math.floor(ki/2)-1)
   local padj = math.random(0,math.floor(kj/2)-1)
   local ini = (outi-1)*si+ki - padi*2
   local inj = (outj-1)*sj+kj - padj*2
   local ceil_mode = math.random(0,1) == 1

   for k, typename in ipairs(typenames) do
      local input = torch.randn(from,inj,ini):type(typename)
      local ctype = t2cpu[typename]
      input = makeNonContiguous(input:type(ctype))
      local sconv = nn.SpatialMaxPooling(ki,kj,si,sj,padi,padj):type(ctype)
      if ceil_mode then sconv:ceil() end
      local groundtruth = sconv:forward(input)

      input = makeNonContiguous(input:type(typename))
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
   local padi = math.random(0,math.floor(ki/2)-1)
   local padj = math.random(0,math.floor(kj/2)-1)
   local ini = (outi-1)*si+ki - padi*2
   local inj = (outj-1)*sj+kj - padj*2
   local ceil_mode = math.random(0,1) == 1

   for k, typename in ipairs(typenames) do
      local input = torch.randn(bs,from,inj,ini):type(typename)
      local ctype = t2cpu[typename]
      input = makeNonContiguous(input:type(ctype))
      local sconv = nn.SpatialMaxPooling(ki,kj,si,sj,padi,padj):type(ctype)
      if ceil_mode then sconv:ceil() end
      local groundtruth = sconv:forward(input)

      input = makeNonContiguous(input:type(typename))
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
   local padi = math.random(0,math.floor(ki/2)-1)
   local padj = math.random(0,math.floor(kj/2)-1)
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
      original = makeNonContiguous(original:type(ctype))
      local input = pooler:forward(original)
      local groundtruth = sunpool:forward(input)

      original = makeNonContiguous(original:type(typename))
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
   local padi = math.random(0,math.floor(ki/2)-1)
   local padj = math.random(0,math.floor(kj/2)-1)
   local ini = (outi-1)*si+ki - padi*2
   local inj = (outj-1)*sj+kj - padj*2
   local ceil_mode = true--math.random(0,1) == 1

   for k, typename in ipairs(typenames) do
      local input = torch.randn(from,inj,ini):type(typename)
      local gradOutput = torch.randn(to,outj,outi):type(typename)
      local ctype = t2cpu[typename]
      input = makeNonContiguous(input:type(ctype))
      gradOutput = makeNonContiguous(gradOutput:type(ctype))

      local sconv = nn.SpatialMaxPooling(ki,kj,si,sj,padi,padj):type(ctype)
      if ceil_mode then sconv:ceil() end
      sconv:forward(input)
      sconv:zeroGradParameters()
      local groundgrad = sconv:backward(input, gradOutput)

      input = makeNonContiguous(input:type(typename))
      gradOutput = makeNonContiguous(gradOutput:type(typename))
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
   local padi = math.random(0,math.floor(ki/2)-1)
   local padj = math.random(0,math.floor(kj/2)-1)
   local ini = (outi-1)*si+ki - padi*2
   local inj = (outj-1)*sj+kj - padj*2
   local ceil_mode = math.random(0,1) == 1

   for k, typename in ipairs(typenames) do
      local input = torch.randn(bs,from,inj,ini):type(typename)
      local gradOutput = torch.randn(bs,to,outj,outi):type(typename)
      local ctype = t2cpu[typename]
      local input = makeNonContiguous(input:type(ctype))
      local gradOutput = makeNonContiguous(gradOutput:type(ctype))
      local sconv = nn.SpatialMaxPooling(ki,kj,si,sj,padi,padj):type(ctype)
      if ceil_mode then sconv:ceil() end
      sconv:forward(input)
      sconv:zeroGradParameters()
      local groundgrad = sconv:backward(input, gradOutput)

      input = makeNonContiguous(input:type(typename))
      gradOutput = makeNonContiguous(gradOutput:type(typename))
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
   local padi = math.random(0,math.floor(ki/2)-1)
   local padj = math.random(0,math.floor(kj/2)-1)
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
      original = makeNonContiguous(original:type(ctype))
      local input = pooler:forward(original)
      local gradOutput = torch.randn(original:size()):type(typename)
      gradOutput = makeNonContiguous(gradOutput:type(ctype))
      sunpool:forward(input)
      sunpool:zeroGradParameters()
      local groundgrad = sunpool:backward(input, gradOutput)

      pooler = nn.SpatialMaxPooling(ki,kj,si,sj,padi,padj):type(typename)
      if ceil_mode then pooler:ceil() end
      local gunpool = nn.SpatialMaxUnpooling(pooler):type(typename)

      original = makeNonContiguous(original:type(typename))
      input = pooler:forward(original)
      gunpool:forward(input)

      gradOutput = makeNonContiguous(gradOutput:type(typename))
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
   local padi = math.random(0,math.floor(ki/2)-1)
   local padj = math.random(0,math.floor(kj/2)-1)
   local dilationi = math.random(1,10)
   local dilationj = math.random(1,10)
   local ini = (outi-1)*si+(dilationi*(ki-1)+1)-2*padi
   local inj = (outj-1)*sj+(dilationj*(kj-1)+1)-2*padj
   local ceil_mode = math.random(0,1) == 1

   for k, typename in ipairs(typenames) do
      local input = torch.randn(from,inj,ini):type(typename)
      local ctype = t2cpu[typename]
      input = makeNonContiguous(input:type(ctype))
      local sconv = nn.SpatialDilatedMaxPooling(ki,kj,si,sj,padi,padj,dilationi,dilationj):type(ctype)
      if ceil_mode then sconv:ceil() end
      local groundtruth = sconv:forward(input)

      input = makeNonContiguous(input:type(typename))
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
   local padi = math.random(0,math.floor(ki/2)-1)
   local padj = math.random(0,math.floor(kj/2)-1)
   local dilationi = math.random(1,10)
   local dilationj = math.random(1,10)
   local ini = (outi-1)*si+(dilationi*(ki-1)+1)-2*padi
   local inj = (outj-1)*sj+(dilationj*(kj-1)+1)-2*padj
   local ceil_mode = math.random(0,1) == 1

   for k, typename in ipairs(typenames) do
      local input = torch.randn(bs,from,inj,ini):type(typename)
      local ctype = t2cpu[typename]
      input = makeNonContiguous(input:type(ctype))
      local sconv = nn.SpatialDilatedMaxPooling(ki,kj,si,sj,padi,padj,dilationi,dilationj):type(ctype)
      if ceil_mode then sconv:ceil() end
      local groundtruth = sconv:forward(input)

      input = makeNonContiguous(input:type(typename))
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
   local padi = math.random(0,math.floor(ki/2)-1)
   local padj = math.random(0,math.floor(kj/2)-1)
   local dilationi = math.random(1,10)
   local dilationj = math.random(1,10)
   local ini = (outi-1)*si+(dilationi*(ki-1)+1)-2*padi
   local inj = (outj-1)*sj+(dilationj*(kj-1)+1)-2*padj
   local ceil_mode = math.random(0,1) == 1

   for k, typename in ipairs(typenames) do
      local input = torch.randn(from,inj,ini):type(typename)
      local gradOutput = torch.randn(to,outj,outi):type(typename)
      local ctype = t2cpu[typename]
      input = makeNonContiguous(input:type(ctype))
      gradOutput = makeNonContiguous(gradOutput:type(ctype))
      local sconv = nn.SpatialDilatedMaxPooling(ki,kj,si,sj,padi,padj,dilationi,dilationj):type(ctype)
      if ceil_mode then sconv:ceil() end
      sconv:forward(input)
      sconv:zeroGradParameters()
      local groundgrad = sconv:backward(input, gradOutput)

      input = makeNonContiguous(input:type(typename))
      gradOutput = makeNonContiguous(gradOutput:type(typename))
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
   local padi = math.random(0,math.floor(ki/2)-1)
   local padj = math.random(0,math.floor(kj/2)-1)
   local dilationi = math.random(1,10)
   local dilationj = math.random(1,10)
   local ini = (outi-1)*si+(dilationi*(ki-1)+1)-2*padi
   local inj = (outj-1)*sj+(dilationj*(kj-1)+1)-2*padj
   local ceil_mode = math.random(0,1) == 1

   for k, typename in ipairs(typenames) do
      local input = torch.randn(bs,from,inj,ini):type(typename)
      local gradOutput = torch.randn(bs,to,outj,outi):type(typename)
      local ctype = t2cpu[typename]
      input = makeNonContiguous(input:type(ctype))
      gradOutput = makeNonContiguous(gradOutput:type(ctype))
      local sconv = nn.SpatialDilatedMaxPooling(ki,kj,si,sj,padi,padj,dilationi,dilationj):type(ctype)
      if ceil_mode then sconv:ceil() end
      sconv:forward(input)
      sconv:zeroGradParameters()
      local groundgrad = sconv:backward(input, gradOutput)

      input = makeNonContiguous(input:type(typename))
      gradOutput = makeNonContiguous(gradOutput:type(typename))
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
        input = makeNonContiguous(input:type(ctype))
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

        input = makeNonContiguous(input:type(typename))

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
        input = makeNonContiguous(input:type(ctype))
        gradOutput = makeNonContiguous(gradOutput:type(ctype))
        local module =
            nn.SpatialFractionalMaxPooling(poolSizeW, poolSizeH, outW, outH)
            :fixPoolingRegions():type(ctype)

        -- convert type of randomSamples and ensure we don't resample
        module:initSampleBuffer_(input)
        module:fixPoolingRegions()
        module.randomSamples = module.randomSamples:type(typename):type(ctype)
        module:forward(input)
        module:zeroGradParameters()
        local groundgrad = module:backward(input, gradOutput)

        input = makeNonContiguous(input:type(typename))
        gradOutput = makeNonContiguous(gradOutput:type(typename))

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

      local input = makeNonContiguous(torch.FloatTensor(1, 1, 1, 1):type(typename))
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
   local padi = math.random(0,math.floor(ki/2)-1)
   local padj = math.random(0,math.floor(kj/2)-1)
   local ini = (outi-1)*si+ki - padi*2
   local inj = (outj-1)*sj+kj - padj*2
   local ceil_mode = math.random(0,1) == 1
   local count_exclude_pad = math.random(0,1) == 1

   for k, typename in ipairs(typenames) do
      local input = torch.randn(from,inj,ini):type(typename)

      local ctype = t2cpu[typename]
      input = makeNonContiguous(input:type(ctype))
      local sconv = nn.SpatialAveragePooling(ki,kj,si,sj,padi,padj):type(ctype)
      if ceil_mode then sconv:ceil() end
      if count_exclude_pad then sconv:setCountExcludePad() end
      local groundtruth = sconv:forward(input)

      input = makeNonContiguous(input:type(typename))
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
   local padi = math.random(0,math.floor(ki/2)-1)
   local padj = math.random(0,math.floor(kj/2)-1)
   local ini = (outi-1)*si+ki - padi*2
   local inj = (outj-1)*sj+kj - padj*2
   local ceil_mode = math.random(0,1) == 1
   local count_exclude_pad = math.random(0,1) == 1

   for k, typename in ipairs(typenames) do
      local input = torch.randn(bs,from,inj,ini):type(typename)
      local ctype = t2cpu[typename]

      input = makeNonContiguous(input:type(ctype))
      local sconv = nn.SpatialAveragePooling(ki,kj,si,sj,padi,padj):type(ctype)
      if ceil_mode then sconv:ceil() end
      if count_exclude_pad then sconv:setCountExcludePad() end
      local groundtruth = sconv:forward(input)

      input = makeNonContiguous(input:type(typename))
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
   local padi = math.random(0,math.floor(ki/2)-1)
   local padj = math.random(0,math.floor(kj/2)-1)
   local ini = (outi-1)*si+ki - padi*2
   local inj = (outj-1)*sj+kj - padj*2
   local ceil_mode = math.random(0,1) == 1
   local count_exclude_pad = math.random(0,1) == 1

   for k, typename in ipairs(typenames) do
      local input = torch.randn(from,inj,ini):type(typename)
      local gradOutput = torch.randn(to,outj,outi):type(typename)

      local ctype = t2cpu[typename]
      input = makeNonContiguous(input:type(ctype))
      gradOutput = makeNonContiguous(gradOutput:type(ctype))
      local sconv = nn.SpatialAveragePooling(ki,kj,si,sj,padi,padj):type(ctype)
      if ceil_mode then sconv:ceil() end
      if count_exclude_pad then sconv:setCountExcludePad() end
      sconv:forward(input)
      sconv:zeroGradParameters()
      local groundgrad = sconv:backward(input, gradOutput)

      input = makeNonContiguous(input:type(typename))
      gradOutput = makeNonContiguous(gradOutput:type(typename))
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
   local padi = math.random(0,math.floor(ki/2)-1)
   local padj = math.random(0,math.floor(kj/2)-1)
   local ini = (outi-1)*si+ki - padi*2
   local inj = (outj-1)*sj+kj - padj*2
   local ceil_mode = math.random(0,1) == 1
   local count_exclude_pad = math.random(0,1) == 1

   for k, typename in ipairs(typenames) do
      local input = torch.randn(bs,from,inj,ini):type(typename)
      local gradOutput = torch.randn(bs,to,outj,outi):type(typename)

      local ctype = t2cpu[typename]
      input = makeNonContiguous(input:type(ctype))
      gradOutput = makeNonContiguous(gradOutput:type(ctype))
      local sconv = nn.SpatialAveragePooling(ki,kj,si,sj,padi,padj):type(ctype)
      if ceil_mode then sconv:ceil() end
      if count_exclude_pad then sconv:setCountExcludePad() end
      sconv:forward(input)
      sconv:zeroGradParameters()
      local groundgrad = sconv:backward(input, gradOutput)

      input = makeNonContiguous(input:type(typename))
      gradOutput = makeNonContiguous(gradOutput:type(typename))
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
      input = makeNonContiguous(input:type(ctype))
      local sconv = nn.SpatialAdaptiveMaxPooling(outi,outj):type(ctype)
      local groundtruth = sconv:forward(input):type(ctype)

      input = makeNonContiguous(input:type(typename))
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
      local input = makeNonContiguous(input0:type(ctype):transpose(2,3))
      local sconv = nn.SpatialAdaptiveMaxPooling(outi,outj):type(ctype)
      local groundtruth = sconv:forward(input)

      input = makeNonContiguous(input0:type(typename):transpose(2,3))
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
      input = makeNonContiguous(input:type(ctype))
      local sconv = nn.SpatialAdaptiveMaxPooling(outi,outj):type(ctype)
      local groundtruth = sconv:forward(input)

      input = makeNonContiguous(input:type(typename))
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
      input = makeNonContiguous(input:type(ctype))
      gradOutput = makeNonContiguous(gradOutput:type(ctype))
      local sconv = nn.SpatialAdaptiveMaxPooling(outi,outj):type(ctype)
      sconv:forward(input)
      sconv:zeroGradParameters()
      local groundgrad = sconv:backward(input, gradOutput)

      input = makeNonContiguous(input:type(typename))
      gradOutput = makeNonContiguous(gradOutput:type(typename))
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
      local input = makeNonContiguous(input0:type(ctype):transpose(2,3))
      gradOutput = makeNonContiguous(gradOutput:type(ctype))
      local sconv = nn.SpatialAdaptiveMaxPooling(outi,outj):type(ctype)
      sconv:forward(input)
      sconv:zeroGradParameters()
      local groundgrad = sconv:backward(input, gradOutput)

      input = makeNonContiguous(input0:type(typename):transpose(2,3))
      gradOutput = makeNonContiguous(gradOutput:type(typename))
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
      input = makeNonContiguous(input:type(ctype))
      gradOutput = makeNonContiguous(gradOutput:type(ctype))
      local sconv = nn.SpatialAdaptiveMaxPooling(outi,outj):type(ctype)
      sconv:forward(input)
      sconv:zeroGradParameters()
      local groundgrad = sconv:backward(input, gradOutput)

      input = makeNonContiguous(input:type(typename))
      gradOutput = makeNonContiguous(gradOutput:type(typename))
      local gconv = nn.SpatialAdaptiveMaxPooling(outi,outj):type(typename)
      gconv:forward(input)
      gconv:zeroGradParameters()
      local rescuda = gconv:backward(input, gradOutput)

      local error = rescuda:double() - groundgrad:double()

      mytester:assertlt(error:abs():max(), precision_backward_type(precision_backward, typename),
          string.format('error on state (backward) with %s', typename))
   end
end

function cunntest.SpatialAdaptiveAveragePooling_forward()
   local from = math.random(1,64)
   local to = from
   local outi = math.random(2,64)
   local outj = math.random(2,64)
   local ini = math.random(10,256)
   local inj = math.random(10,256)

   for k, typename in ipairs(typenames) do
      local input = torch.randn(from,inj,ini):type(typename)
      local ctype = t2cpu[typename]
      input = makeNonContiguous(input:type(ctype))
      local sconv = nn.SpatialAdaptiveAveragePooling(outi,outj):type(ctype)
      local groundtruth = sconv:forward(input):type(ctype)

      input = makeNonContiguous(input:type(typename))
      local gconv = nn.SpatialAdaptiveAveragePooling(outi,outj):type(typename)
      local rescuda = gconv:forward(input)

      local error = rescuda:double() - groundtruth:double()
      mytester:assertlt(error:abs():max(), precision_forward_type(precision_forward, typename),
          string.format('error on state (forward) with %s', typename))
   end
end

function cunntest.SpatialAdaptiveAveragePooling_forward_noncontig()
   local from = math.random(1,64)
   local to = from
   local outi = math.random(2,64)
   local outj = math.random(2,64)
   local ini = math.random(10,256)
   local inj = math.random(10,256)

   for k, typename in ipairs(typenames) do
      local input0 = torch.randn(from,ini,inj):type(typename)
      local ctype = t2cpu[typename]
      local input = makeNonContiguous(input0:type(ctype):transpose(2,3))
      local sconv = nn.SpatialAdaptiveAveragePooling(outi,outj):type(ctype)
      local groundtruth = sconv:forward(input)

      input = makeNonContiguous(input0:type(typename):transpose(2,3))
      local gconv = nn.SpatialAdaptiveAveragePooling(outi,outj):type(typename)
      local rescuda = gconv:forward(input)

      local error = rescuda:double() - groundtruth:double()
      mytester:assertlt(error:abs():max(), precision_forward_type(precision_forward, typename),
          string.format('error on state (forward) with %s', typename))
   end
end

function cunntest.SpatialAdaptiveAveragePooling_forward_batch()
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
      input = makeNonContiguous(input:type(ctype))
      local sconv = nn.SpatialAdaptiveAveragePooling(outi,outj):type(ctype)
      local groundtruth = sconv:forward(input)

      input = makeNonContiguous(input:type(typename))
      local gconv = nn.SpatialAdaptiveAveragePooling(outi,outj):type(typename)
      local rescuda = gconv:forward(input)

      local error = rescuda:double() - groundtruth:double()
      mytester:assertlt(error:abs():max(), precision_forward_type(precision_forward, typename),
          string.format('error on state (forward) with %s', typename))
   end
end

function cunntest.SpatialAdaptiveAveragePooling_backward()
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
      input = makeNonContiguous(input:type(ctype))
      gradOutput = makeNonContiguous(gradOutput:type(ctype))
      local sconv = nn.SpatialAdaptiveAveragePooling(outi,outj):type(ctype)
      sconv:forward(input)
      sconv:zeroGradParameters()
      local groundgrad = sconv:backward(input, gradOutput)

      input = makeNonContiguous(input:type(typename))
      gradOutput = makeNonContiguous(gradOutput:type(typename))
      local gconv = nn.SpatialAdaptiveAveragePooling(outi,outj):type(typename)
      gconv:forward(input)
      gconv:zeroGradParameters()
      local rescuda = gconv:backward(input, gradOutput)

      local error = rescuda:double() - groundgrad:double()

      mytester:assertlt(error:abs():max(), precision_backward_type(precision_backward, typename),
          string.format('error on state (backward) with %s', typename))
   end
end

function cunntest.SpatialAdaptiveAveragePooling_backward_noncontig()
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
      local input = makeNonContiguous(input0:type(ctype):transpose(2,3))
      gradOutput = makeNonContiguous(gradOutput:type(ctype))
      local sconv = nn.SpatialAdaptiveAveragePooling(outi,outj):type(ctype)
      sconv:forward(input)
      sconv:zeroGradParameters()
      local groundgrad = sconv:backward(input, gradOutput)

      input = makeNonContiguous(input0:type(typename):transpose(2,3))
      gradOutput = makeNonContiguous(gradOutput:type(typename))
      local gconv = nn.SpatialAdaptiveAveragePooling(outi,outj):type(typename)
      gconv:forward(input)
      gconv:zeroGradParameters()
      local rescuda = gconv:backward(input, gradOutput)

      local error = rescuda:double() - groundgrad:double()

      mytester:assertlt(error:abs():max(), precision_backward_type(precision_backward, typename),
          string.format('error on state (backward) with %s', typename))
   end
end

function cunntest.SpatialAdaptiveAveragePooling_backward_batch()
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
      input = makeNonContiguous(input:type(ctype))
      gradOutput = makeNonContiguous(gradOutput:type(ctype))
      local sconv = nn.SpatialAdaptiveAveragePooling(outi,outj):type(ctype)
      sconv:forward(input)
      sconv:zeroGradParameters()
      local groundgrad = sconv:backward(input, gradOutput)

      input = makeNonContiguous(input:type(typename))
      gradOutput = makeNonContiguous(gradOutput:type(typename))
      local gconv = nn.SpatialAdaptiveAveragePooling(outi,outj):type(typename)
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

   local input = makeNonContiguous(torch.randn(from,inj,ini))
   local sconv = nn.SpatialLPPooling(from,pnorm,ki,kj,si,sj)
   local groundtruth = sconv:forward(input)
   local a = torch.Timer()
   for i = 1,nloop do
      groundtruth = sconv:forward(input)
   end
   tm.cpu = a:time().real

   input = makeNonContiguous(input:cuda())
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

   local input = makeNonContiguous(torch.randn(from,inj,ini))
   local gradOutput = makeNonContiguous(torch.randn(to,outj,outi))
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

   input = makeNonContiguous(input:cuda())
   gradOutput = makeNonContiguous(gradOutput:cuda())
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

  for k, typename in ipairs(typenames) do
     local input = torch.Tensor(size):uniform():type(typename)
     local target = torch.Tensor(size):uniform():gt(0.5):type(torch.type(input))

     local ctype = t2cpu[typename]
     input = makeNonContiguous(input:type(ctype))
     target = makeNonContiguous(target:type(ctype))
     local crit = nn.BCECriterion():type(ctype)
     local rescpu = crit:forward(input, target)

     input = makeNonContiguous(input:type(typename))
     target = makeNonContiguous(target:type(typename))
     local g_crit = nn.BCECriterion():type(typename)
     local rescuda = g_crit:forward(input, target)
     local errorVal = rescuda - rescpu
     mytester:assertlt(errorVal, precision_forward_type(precision_forward, typename),
        string.format('error on state (forward) with %s', typename))

     -- test vs lua implementation
     input = makeNonContiguous(input:type(ctype))
     target = makeNonContiguous(target:type(ctype))
     buffer = input.new()
     local restruth = BCECriterion_forward_truth(buffer, input, target, nil, true)
     errorVal = rescpu - restruth
     mytester:assertlt(errorVal, precision_forward_type(precision_forward, typename),
        string.format('error on state (forward) with %s', typename))
     errorVal = rescuda - restruth
     mytester:assertlt(errorVal, precision_forward_type(precision_forward, typename),
        string.format('error on state (forward) with %s', typename))
  end
end

function cunntest.BCECriterionWeights_forward()
  local size = math.random(1,100)
  for k, typename in ipairs(typenames) do
     local input = torch.Tensor(size):uniform():type(typename)
     local target = torch.Tensor(size):uniform():gt(0.5):type(torch.type(input))
     local weights = torch.Tensor(size):uniform():type(typename)

     local ctype = t2cpu[typename]
     input = makeNonContiguous(input:type(ctype))
     target = makeNonContiguous(target:type(ctype))
     weights = makeNonContiguous(weights:type(ctype))
     local crit = nn.BCECriterion(weights):type(ctype)
     local rescpu = crit:forward(input, target)

     input = makeNonContiguous(input:type(typename))
     target = makeNonContiguous(target:type(typename))
     weights = makeNonContiguous(weights:type(typename))
     local g_crit = nn.BCECriterion(weights):type(typename)
     local rescuda = g_crit:forward(input, target)

     local errorVal = rescuda - rescpu
     mytester:assertlt(errorVal, precision_forward_type(precision_forward, typename),
        string.format('error on state (forward) with %s', typename))

     -- test vs lua implementation
     -- FIXME: half does not support dot without CUDA 8.0, so can't compare to lua implementation.
     if typename ~= 'torch.CudaHalfTensor' then
        buffer = input.new()
        restruth = BCECriterion_forward_truth(buffer, input, target, weights, true)
        errorVal = rescpu - restruth
        mytester:assertlt(errorVal, precision_forward_type(precision_forward, typename),
           string.format('error on state (forward) with %s', typename))
        errorVal = rescuda - restruth
        mytester:assertlt(errorVal, precision_forward_type(precision_forward, typename),
           string.format('error on state (forward) with %s', typename))
     end
  end
end

function cunntest.MarginCriterion_forward()
  local size = math.random(1,100)

  for k, typename in ipairs(typenames) do
    local input = ((torch.rand(size)-0.5) * 2):type(typename) -- data spread from -1 to 1
    local target = ((torch.round(torch.rand(size))*2)-1):type(typename)-- generate random labels -1, 1

    local ctype = t2cpu[typename]
    input = makeNonContiguous(input:type(ctype))
    target = makeNonContiguous(input:type(ctype))
    local crit = nn.MarginCriterion():type(ctype)
    local groundtruth= crit:forward(input, target)

    input = makeNonContiguous(input:type(typename))
    target = makeNonContiguous(target:type(typename))
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
     local input = ((torch.rand(size)-0.5) * 2):type(typename)-- data spread from -1 to 1
     local target = makeNonContiguous(torch.round(torch.rand(size)*(size-1)):add(1)) -- generate random labels > 0
     local zero = math.random(0,size) -- turn some labels into 0 targets
     if zero > 0 then
        target:sub(size-zero+1,size):zero()
     end

     local ctype = t2cpu[typename]
     input = makeNonContiguous(input:type(ctype))
     local crit = nn.MultiLabelMarginCriterion():type(ctype)
     local groundtruth= crit:forward(input, target)
     input = makeNonContiguous(input:type(typename))
     target = makeNonContiguous(target:type(typename))
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
      input = makeNonContiguous(input:type(ctype))
      local crit = nn.MultiLabelMarginCriterion():type(ctype)
      local pred = crit:forward(input, target)
      local groundgrad = crit:backward(input, target)

      input = makeNonContiguous(input:type(typename))
      target = makeNonContiguous(target:type(typename))
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
      input = makeNonContiguous(input:type(ctype))
      local sconv = nn.SpatialCrossMapLRN(size, alpha, beta, k):type(ctype)
      local groundtruth = sconv:forward(input)

      input = makeNonContiguous(input:type(typename))
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
      input = makeNonContiguous(input:type(ctype))
      gradOutput = makeNonContiguous(gradOutput:type(ctype))
      local sconv = nn.SpatialCrossMapLRN(size, alpha, beta, k):type(ctype)
      sconv:forward(input)
      sconv:zeroGradParameters()
      local groundgrad = sconv:backward(input, gradOutput)

      input = makeNonContiguous(input:type(ctype))
      gradOutput = makeNonContiguous(gradOutput:type(ctype))
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
      input = makeNonContiguous(input:type(ctype))
      target = makeNonContiguous(target:type(ctype))
      local crit = nn.MarginCriterion():type(ctype)
      crit:forward(input, target)
      local groundgrad = crit:backward(input, target)

      input = makeNonContiguous(input:type(typename))
      target = makeNonContiguous(target:type(typename))
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

   for k, typename in ipairs(typenames) do
      local input = torch.Tensor(size):uniform():type(typename)
      local target = torch.Tensor(size):uniform():gt(0.5):type(torch.type(input))

      local ctype = t2cpu[typename]
      input = makeNonContiguous(input:type(ctype))
      target = makeNonContiguous(target:type(ctype))
      local crit = nn.BCECriterion():type(ctype)
      crit:forward(input, target)
      local groundgrad = crit:backward(input, target)

      input = makeNonContiguous(input:type(typename))
      target = makeNonContiguous(target:type(typename))
      local g_crit = nn.BCECriterion():type(typename)
      g_crit:forward(input, target)
      local rescuda = g_crit:backward(input, target)

      local error = rescuda:double() - groundgrad:double()

      mytester:assertlt(error:abs():max(), precision_backward_type(precision_backward, typename),
         string.format('error on state (backward) with %s', typename))
   end
end

function cunntest.BCECriterionWeights_backward()
  local size = math.random(1,100)

  for k, typename in ipairs(typenames) do
     local input = torch.Tensor(size):uniform():type(typename)
     local target = torch.Tensor(size):uniform():gt(0.5):type(torch.type(input))
     local weights = torch.Tensor(size):uniform():type(typename)

     local ctype = t2cpu[typename]
     input = makeNonContiguous(input:type(ctype))
     target = makeNonContiguous(target:type(ctype))
     weights = makeNonContiguous(weights:type(ctype))
     local crit = nn.BCECriterion(weights):type(ctype)
     crit:forward(input, target)
     local groundgrad = crit:backward(input, target)

     input = makeNonContiguous(input:type(typename))
     target = makeNonContiguous(target:type(typename))
     weights = makeNonContiguous(weights:type(typename))
     local g_crit = nn.BCECriterion(weights):type(typename)
     g_crit:forward(input, target)
     local rescuda = g_crit:backward(input, target)

     local error = rescuda:double() - groundgrad:double()

     mytester:assertlt(error:abs():max(), precision_backward_type(precision_backward, typename),
        string.format('error on state (backward) with %s', typename))
  end
end

function cunntest.mse()
   for sizeAverage = 0, 1 do
      for k, typename in ipairs(typenames) do
         local size = math.random(3000,5000)
         local input = torch.randn(size,1,1):type(typename)
         local target = torch.randn(size):type(typename)

         local ctype = t2cpu[typename]
         input = makeNonContiguous(input:type(ctype))
         target = makeNonContiguous(target:type(ctype))
         local mod = nn.MSECriterion(sizeAverage == 1):type(ctype)

         local fout = mod:forward(input,target)
         local fgin = mod:backward(input,target):clone()

         local cinput = makeNonContiguous(input:type(typename))
         local ctarget = makeNonContiguous(target:type(typename))
         local cmod = nn.MSECriterion(sizeAverage == 1):type(typename)
         local cout = cmod:forward(cinput,ctarget)
         local cgin = cmod:backward(cinput,ctarget)

         mytester:assertlt(math.abs(fout-cout),
            precision_forward_type(0.03, typename, math.abs(fout)),
            string.format('error on output with %s', typename))
         local gerr = cgin:double() - fgin:double()
         mytester:assertlt(gerr:abs():max(),
            precision_forward_type(precision_forward, typename),
            string.format('error on gradInput with %s', typename))
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
         input = makeNonContiguous(input:type(ctype))
         target = makeNonContiguous(target:type(ctype))
         local mod = nn.SmoothL1Criterion(sizeAverage == 1):type(ctype)

         local fout = mod:forward(input,target)
         local fgin = mod:backward(input,target):clone()

         local cinput = makeNonContiguous(input:type(typename))
         local ctarget = makeNonContiguous(target:type(typename))
         local cmod = nn.SmoothL1Criterion(sizeAverage == 1):type(typename)
         local cout = cmod:forward(cinput,ctarget)
         local cgin = cmod:backward(cinput,ctarget)

         mytester:assertlt(math.abs(fout-cout),
            math.max(precision_forward_type(precision_forward, typename, math.abs(fout)), 0.01),
            string.format('error on output with %s', typename))
         local gerr = cgin:double() - fgin:double()
         mytester:assertlt(gerr:abs():max(), precision_forward_type(precision_forward, typename),
            string.format('error on gradInput with %s', typename))
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
         input = makeNonContiguous(input:type(ctype))
         target = makeNonContiguous(target:type(ctype))
         local mod = nn.SoftMarginCriterion(sizeAverage == 1):type(ctype)

         local fout = mod:forward(input,target)
         local fgin = mod:backward(input,target):clone()

         local cinput = makeNonContiguous(input:type(typename))
         local ctarget = makeNonContiguous(target:type(typename))
         local cmod = nn.SoftMarginCriterion(sizeAverage == 1):type(typename)
         local cout = cmod:forward(cinput,ctarget)
         local cgin = cmod:backward(cinput,ctarget)

        mytester:assertlt(math.abs(fout-cout), 0.01, 'error on output')
        local gerr = cgin:double() - fgin:double()
        mytester:assertlt(gerr:abs():max(), precision_forward_type(precision_forward, typename),
           string.format('error on gradInput with %s', typename))
      end
   end
end


function cunntest.distkldiv()
   for sizeAverage = 0, 1 do
      local size = math.random(3000,5000)

      for k, typename in ipairs(typenames) do
         local input = torch.randn(size):type(typename) -- TODO, make it back to (size, 1, 1), see https://github.com/torch/cunn/issues/245#issuecomment-209260954
         local target = torch.randn(size):type(typename)

         local ctype = t2cpu[typename]
         input = makeNonContiguous(input:type(ctype))
         target = makeNonContiguous(target:type(ctype))
         local mod = nn.DistKLDivCriterion(sizeAverage == 1):type(ctype)

         local fout = mod:forward(input,target)
         local fgin = mod:backward(input,target):clone()

         local cinput = makeNonContiguous(input:type(typename))
         local ctarget = makeNonContiguous(target:type(typename))
         local cmod = nn.DistKLDivCriterion(sizeAverage == 1):type(typename)
         local cout = cmod:forward(cinput,ctarget)
         local cgin = cmod:backward(cinput,ctarget)

         mytester:assertlt(math.abs(fout-cout), precision_forward_type(precision_forward, typename),
            string.format('error on output with %s', typename))
         local gerr = cgin:double() - fgin:double()
         mytester:assertlt(gerr:abs():max(), precision_backward_type(precision_backward, typename),
            string.format('error on gradInput with %s', typename))
      end
   end
end

function cunntest.TemporalConvolution_forward()
   local from = math.random(1,64) -- inputFrameSize
   local to = math.random(1,64) -- outputFrameSize
   local ki = math.random(3,15) -- kernelWidth (kW)
   local si = math.random(1,2) -- stepSize (dW)
   local outi = math.random(1,256) -- nOutputFrame
   local ini = (outi-1)*si+ki -- nInputFrame

   for k, typename in ipairs(typenames) do
      local input = torch.randn(ini,from):type(typename)

      local ctype = t2cpu[typename]
      input = makeNonContiguous(input:type(ctype))
      local sconv = nn.TemporalConvolution(from,to,ki,si):type(ctype)
      local groundtruth = sconv:forward(input)

      input = makeNonContiguous(input:type(typename))
      local gconv = nn.TemporalConvolution(from,to,ki,si):type(typename)
      gconv.weight = sconv.weight:type(typename)
      gconv.bias = sconv.bias:type(typename)
      local rescuda = gconv:forward(input)

      local error = rescuda:double() - groundtruth:double()
      mytester:assertlt(error:abs():max(), precision_forward_type(precision_forward, typename),
        string.format('error on state (forward) with %s', typename))
   end
end

function cunntest.TemporalConvolution_forward_batch()
   local bs = math.random(4,16)
   local from = math.random(1,64)
   local to = math.random(1,64)
   local ki = math.random(3,15)
   local si = math.random(1,2)
   local outi = math.random(1,256)
   local ini = (outi-1)*si+ki

   for k, typename in ipairs(typenames) do
      local input = torch.randn(bs,ini,from):type(typename)

      local ctype = t2cpu[typename]
      input = makeNonContiguous(input:type(ctype))
      local sconv = nn.TemporalConvolution(from,to,ki,si):type(ctype)
      local groundtruth = sconv:forward(input)

      input = makeNonContiguous(input:type(typename))
      local gconv = nn.TemporalConvolution(from,to,ki,si):type(typename)
      gconv.weight = sconv.weight:type(typename)
      gconv.bias = sconv.bias:type(typename)
      local rescuda = gconv:forward(input)

      local error = rescuda:double() - groundtruth:double()
      mytester:assertlt(error:abs():max(), precision_forward_type(precision_forward, typename),
        string.format('error on state (forward) with %s', typename))
   end
end

function cunntest.TemporalConvolution_backward()
  local from = math.random(1,64)
   local to = math.random(1,64)
   local ki = math.random(3,15)
   local si = math.random(1,2)
   local outi = math.random(1,256)
   local ini = (outi-1)*si+ki

   for k, typename in ipairs(typenames) do
      local input = torch.randn(ini,from):type(typename)
      local gradOutput = torch.randn(outi,to):type(typename)

      local ctype = t2cpu[typename]
      input = makeNonContiguous(input:type(ctype))
      gradOutput = makeNonContiguous(gradOutput:type(ctype))
      local sconv = nn.TemporalConvolution(from,to,ki,si):type(ctype)
      sconv:forward(input)
      sconv:zeroGradParameters()
      local groundgrad = sconv:backward(input, gradOutput)
      local groundweight = sconv.gradWeight
      local groundbias = sconv.gradBias

      input = makeNonContiguous(input:type(typename))
      gradOutput = makeNonContiguous(gradOutput:type(typename))
      local gconv = nn.TemporalConvolution(from,to,ki,si):type(typename)
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
      mytester:assertlt(werror:abs():max(),
        precision_backward_conv_weightbias(precision_backward, typename, weightcuda:abs():max()),
        string.format('error on weight (backward) with %s', typename))
      mytester:assertlt(berror:abs():max(),
        precision_backward_conv_weightbias(precision_backward, typename, biascuda:abs():max()),
        string.format('error on bias (backward) with %s', typename))
   end
end

function cunntest.TemporalConvolution_backward_batch()
   local bs = math.random(4,16)
   local from = math.random(1,64)
   local to = math.random(1,64)
   local ki = math.random(3,15)
   local si = math.random(1,2)
   local outi = math.random(1,256)
   local ini = (outi-1)*si+ki

   for k, typename in ipairs(typenames) do
      local input = torch.randn(bs,ini,from):type(typename)
      local gradOutput = torch.randn(bs,outi,to):type(typename)

      local ctype = t2cpu[typename]
      input = makeNonContiguous(input:type(ctype))
      gradOutput = makeNonContiguous(gradOutput:type(ctype))
      local sconv = nn.TemporalConvolution(from,to,ki,si):type(ctype)
      sconv:forward(input)
      sconv:zeroGradParameters()
      local groundgrad = sconv:backward(input, gradOutput)
      local groundweight = sconv.gradWeight
      local groundbias = sconv.gradBias

      input = makeNonContiguous(input:type(typename))
      gradOutput = makeNonContiguous(gradOutput:type(typename))
      local gconv = nn.TemporalConvolution(from,to,ki,si):type(typename)
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
      mytester:assertlt(werror:abs():max(),
        precision_backward_conv_weightbias(precision_backward, typename, weightcuda:abs():max()),
        string.format('error on weight (backward) with %s', typename))
      mytester:assertlt(berror:abs():max(),
        precision_backward_conv_weightbias(precision_backward, typename, biascuda:abs():max()),
        string.format('error on bias (backward) with %s', typename))
   end
end


function cunntest.TemporalRowConvolution_forward_single()
  local from = math.random(1,64) -- nFeature
  local to = from
  local ki = math.random(3,15) -- kW
  local si = math.random(1,2) -- dW
  local outi = math.random(1,256) -- nOutputFrame
  local ini = (outi-1)*si+ki -- nInputFrame

  local function jacTest(noBias, featFirst)
    noBias = noBias or false
    featFirst = featFirst or false

    for k, typename in ipairs(typenames) do
      if typename ~= "torch.CudaHalfTensor" then

        local input
        if featFirst then
          input = torch.randn(from, ini):type(typename)
        else
          input = torch.randn(ini, from):type(typename)
        end

        local ctype = t2cpu[typename]
        input = makeNonContiguous(input:type(ctype))
        local mod = nn.TemporalRowConvolution(from,ki,si):type(ctype)
        if featFirst then
          mod.featFirst = true
        end
        if noBias then
          mod:noBias()
        end
        local groundtruth = mod:forward(input)

        input = makeNonContiguous(input:type(typename))
        local cmod = nn.TemporalRowConvolution(from,ki,si):type(typename)

        if featFirst then
          cmod.featFirst = true
        end
        if noBias then
          cmod:noBias()
        end
        cmod.weight = mod.weight:type(typename)
        if mod.bias then cmod.bias = mod.bias:type(typename) end
        local rescuda = cmod:forward(input)

        local error = rescuda:double() - groundtruth:double()
        mytester:assertlt(error:abs():max(), precision_forward_type(precision_forward, typename),
          string.format('error on state (forward) with %s', typename))
      end
    end
  end
  jacTest(false,false)
  jacTest(false,true)
  jacTest(true,false)
  jacTest(true,true)
end

function cunntest.TemporalRowConvolution_forward_batch()
  local bs = math.random(4,16)
  local from = math.random(1,64)
  local to = from
  local ki = math.random(3,15)
  local si = math.random(1,2)
  local outi = math.random(1,256)
  local ini = (outi-1)*si+ki

  local function jacTest(noBias,featFirst)
    noBias = noBias or false
    featFirst = featFirst or false
    for k, typename in ipairs(typenames) do
      if typename ~= "torch.CudaHalfTensor" then

        local input
        if featFirst then
          input = torch.randn(bs, from, ini):type(typename)
        else
          input = torch.randn(bs, ini, from):type(typename)
        end

        local ctype = t2cpu[typename]
        input = makeNonContiguous(input:type(ctype))
        local mod = nn.TemporalRowConvolution(from,ki,si):type(ctype)
        if featFirst then
          mod.featFirst = true
        end
        if noBias then
          mod:noBias()
        end
        local groundtruth = mod:forward(input)

        input = makeNonContiguous(input:type(typename))
        local cmod = nn.TemporalRowConvolution(from,ki,si):type(typename)
        if featFirst then
          cmod.featFirst = true
        end
        if noBias then
          cmod:noBias()
        end
        cmod.weight = mod.weight:type(typename)
        if mod.bias then
          cmod.bias = mod.bias:type(typename)
        end
        local rescuda = cmod:forward(input)

        local error = rescuda:double() - groundtruth:double()
        mytester:assertlt(error:abs():max(), precision_forward_type(precision_forward, typename),
          string.format('error on state (forward) with %s', typename))
      end
    end
  end
  jacTest(false,false)
  jacTest(false,true)
  jacTest(true,false)
  jacTest(true,true)
end

function cunntest.TemporalRowConvolution_backward_single()
  local from = math.random(1,64) -- nFeature
  local to = from
  local ki = math.random(3,15) -- kW
  local si = math.random(1,2) -- dW
  local outi = math.random(1,256) -- nOutputFrame
  local ini = (outi-1)*si+ki -- nInputFrame

  local function jacTest(noBias,featFirst)
    noBias = noBias or false
    featFirst = featFirst or false
    for k, typename in ipairs(typenames) do
      if typename ~= "torch.CudaHalfTensor" then

        local input, gradOutput
        if featFirst then
          input = torch.randn(from, ini):type(typename)
          gradOutput = torch.randn(to, outi):type(typename)
        else
          input = torch.randn(ini, from):type(typename)
          gradOutput = torch.rand(outi, to):type(typename)
        end

        local ctype = t2cpu[typename]
        input = makeNonContiguous(input:type(ctype))
        gradOutput = makeNonContiguous(gradOutput:type(ctype))
        local mod = nn.TemporalRowConvolution(from,ki,si):type(ctype)
        if featFirst then mod.featFirst = true end
        if noBias then mod:noBias() end
        mod:forward(input)
        mod:zeroGradParameters()
        local groundgrad = mod:backward(input, gradOutput)
        local groundweight = mod.gradWeight
        local groundbias = mod.gradBias

        input = makeNonContiguous(input:type(typename))
        gradOutput = makeNonContiguous(gradOutput:type(typename))
        local cmod = nn.TemporalRowConvolution(from,ki,si):type(typename)
        if featFirst then cmod.featFirst = true end
        if noBias then cmod:noBias() end
        cmod.weight = mod.weight:type(typename)
        if cmod.bias then cmod.bias = mod.bias:type(typename) end
        cmod:forward(input)
        cmod:zeroGradParameters()
        local rescuda = cmod:backward(input, gradOutput)
        local weightcuda = cmod.gradWeight

        local error = rescuda:double() - groundgrad:double()
        local werror = weightcuda:double() - groundweight:double()

        mytester:assertlt(error:abs():max(), precision_backward_type(precision_backward, typename),
          string.format('error on state (backward) with %s', typename))
        mytester:assertlt(werror:abs():max(),
          precision_backward_conv_weightbias(precision_backward, typename, weightcuda:abs():max()),
          string.format('error on weight (backward) with %s', typename))

        if cmod.bias then
          local berror = cmod.gradBias:double() - groundbias:double()
          mytester:assertlt(berror:abs():max(),
            precision_backward_conv_weightbias(precision_backward, typename, cmod.gradBias:abs():max()),
            string.format('error on bias (backward) with %s', typename))
        end
      end
    end
  end
  jacTest(false,false)
  jacTest(false,true)
  jacTest(true,false)
  jacTest(true,true)
end

function cunntest.TemporalRowConvolution_backward_batch()
  local bs = math.random(4,16)
  local from = math.random(1,64) -- nFeature
  local to = from
  local ki = math.random(3,15) -- kW
  local si = math.random(1,2) -- dW
  local outi = math.random(1,256) -- nOutputFrame
  local ini = (outi-1)*si+ki -- nInputFrame

  local function jacTest(noBias,featFirst)
    for k, typename in ipairs(typenames) do
      if typename ~= "torch.CudaHalfTensor" then

        local input, gradOutput
        if featFirst then
          input = torch.randn(bs, from, ini):type(typename)
          gradOutput = torch.randn(bs, to, outi):type(typename)
        else
          input = torch.randn(bs, ini, from):type(typename)
          gradOutput = torch.rand(bs, outi, to):type(typename)
        end

        local ctype = t2cpu[typename]
        input = makeNonContiguous(input:type(ctype))
        gradOutput = makeNonContiguous(gradOutput:type(ctype))
        local mod = nn.TemporalRowConvolution(from,ki,si):type(ctype)
        if featFirst then
          mod.featFirst = true
        end
        if noBias then
          mod:noBias()
        end
        mod:forward(input)
        mod:zeroGradParameters()
        local groundgrad = mod:backward(input, gradOutput)
        local groundweight = mod.gradWeight
        local groundbias = mod.gradBias

        input = makeNonContiguous(input:type(typename))
        gradOutput = makeNonContiguous(gradOutput:type(typename))
        local cmod = nn.TemporalRowConvolution(from,ki,si):type(typename)
        if featFirst then
          cmod.featFirst = true
        end
        if noBias then
          cmod:noBias()
        end
        cmod.weight = mod.weight:type(typename)
        if cmod.bias then
          cmod.bias = mod.bias:type(typename)
        end
        cmod:forward(input)
        cmod:zeroGradParameters()
        local rescuda = cmod:backward(input, gradOutput)
        local weightcuda = cmod.gradWeight

        local error = rescuda:double() - groundgrad:double()
        local werror = weightcuda:double() - groundweight:double()

        mytester:assertlt(error:abs():max(), precision_backward_type(precision_backward, typename),
          string.format('error on state (backward) [batch] with %s', typename))
        mytester:assertlt(werror:abs():max(),
          precision_backward_conv_weightbias(precision_backward, typename, weightcuda:abs():max()),
          string.format('error on weight (backward) [batch] with %s', typename))

        if cmod.bias then
          local berror = cmod.gradBias:double() - groundbias:double()
          mytester:assertlt(berror:abs():max(),
            precision_backward_conv_weightbias(precision_backward, typename, cmod.gradBias:abs():max()),
            string.format('error on bias (backward) [batch] with %s', typename))
        end
      end
    end
  end
  jacTest(false,false)
  jacTest(false,true)
  jacTest(true,false)
  jacTest(true,true)
end

function cunntest.Dropout()
   local p = 0.2 --prob of droping out a neuron
   local input = makeNonContiguous(torch.CudaTensor(1000):fill((1-p)))
   local module = nn.Dropout(p)
   module:cuda()
   -- version 2
   local output = module:forward(input)
   mytester:assert(math.abs(output:mean() - (1-p)) < 0.05, 'dropout output')
   local gradInput = module:backward(input, input)
   mytester:assert(math.abs(gradInput:mean() - (1-p)) < 0.05, 'dropout gradInput')
   -- version 1 (old nnx version)
   local input = makeNonContiguous(input:fill(1))
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

   local input = makeNonContiguous(torch.randn(size))
   local sconv = nn.Dropout()
   local groundtruth = sconv:forward(input)
   local a = torch.Timer()
   for i = 1,nloop do
      groundtruth = sconv:forward(input)
   end
   tm.cpu = a:time().real

   input = makeNonContiguous(input:cuda())
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
      input = makeNonContiguous(input:type(ctype))
      local sconv = nn.SoftPlus():type(ctype)
      local groundtruth = sconv:forward(input)

      input = makeNonContiguous(input:type(typename))
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
      input = makeNonContiguous(input:type(ctype))
      gradOutput = makeNonContiguous(gradOutput:type(ctype))
      local sconv = nn.SoftPlus():type(ctype)
      sconv:forward(input)
      local groundgrad = sconv:backward(input, gradOutput)

      input = makeNonContiguous(input:type(typename))
      gradOutput = makeNonContiguous(gradOutput:type(typename))
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
      input = makeNonContiguous(input:type(ctype))
      local sconv = nn.SpatialUpSamplingNearest(scale):type(ctype)
      local groundtruth = sconv:forward(input)

      input = makeNonContiguous(input:type(typename))
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
      input = makeNonContiguous(input:type(ctype))
      local sconv = nn.SpatialUpSamplingNearest(scale):type(ctype)
      local groundtruth = sconv:forward(input)

      input = makeNonContiguous(input:type(typename))
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
      input = makeNonContiguous(input:type(ctype))
      gradOutput = makeNonContiguous(gradOutput:type(ctype))
      local sconv = nn.SpatialUpSamplingNearest(scale):type(ctype)
      sconv:forward(input)
      sconv:zeroGradParameters()
      local groundgrad = sconv:backward(input, gradOutput)

      input = makeNonContiguous(input:type(typename))
      gradOutput = makeNonContiguous(gradOutput:type(typename))
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
      input = makeNonContiguous(input:type(ctype))
      gradOutput = makeNonContiguous(gradOutput:type(ctype))
      local sconv = nn.SpatialUpSamplingNearest(scale):type(ctype)
      sconv:forward(input)
      sconv:zeroGradParameters()
      local groundgrad = sconv:backward(input, gradOutput)

      input = makeNonContiguous(input:type(typename))
      gradOutput = makeNonContiguous(gradOutput:type(typename))
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
      input = makeNonContiguous(input:type(ctype))
      local sconv = nn.SpatialUpSamplingBilinear(scale):type(ctype)
      local groundtruth = sconv:forward(input)

      input = makeNonContiguous(input:type(typename))
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
      input = makeNonContiguous(input:type(ctype))
      local sconv = nn.SpatialUpSamplingBilinear(scale):type(ctype)
      local groundtruth = sconv:forward(input)

      input = makeNonContiguous(input:type(typename))
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
      input = makeNonContiguous(input:type(ctype))
      gradOutput = makeNonContiguous(gradOutput:type(ctype))
      local sconv = nn.SpatialUpSamplingBilinear(scale):type(ctype)
      sconv:forward(input)
      sconv:zeroGradParameters()
      local groundgrad = sconv:backward(input, gradOutput)

      input = makeNonContiguous(input:type(typename))
      gradOutput = makeNonContiguous(gradOutput:type(typename))
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
      input = makeNonContiguous(input:type(ctype))
      gradOutput = makeNonContiguous(gradOutput:type(ctype))
      local sconv = nn.SpatialUpSamplingBilinear(scale):type(ctype)
      local output = sconv:forward(input)
      sconv:zeroGradParameters()
      local groundgrad = sconv:backward(input, gradOutput)

      input = makeNonContiguous(input:type(typename))
      gradOutput = makeNonContiguous(gradOutput:type(typename))
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

   for k, typename in ipairs(typenames) do
     local input = torch.randn(size):type(typename)

     local ctype = t2cpu[typename]
     input = makeNonContiguous(input:type(ctype))
     local mod = nn.L1Cost():type(ctype)

     local fout = mod:forward(input)
     local fgin = mod:backward(input):clone()

     local cinput = makeNonContiguous(input:type(typename))
     local cmod = nn.L1Cost():type(typename)
     local cout = cmod:forward(cinput)
     local cgin = cmod:backward(cinput)

     mytester:assertlt(math.abs(fout-cout),
        precision_forward_type(precision_forward, typename, math.abs(fout)),
        string.format('error on output with %s', typename))
     local gerr = cgin:double() - fgin:double()
     mytester:assertlt(gerr:abs():max(),
        precision_forward_type(precision_forward, typename),
        string.format('error on gradInput with %s', typename))
   end
end


function cunntest.ClassNLLCriterionSingleTarget()
   local size = math.random(3000,5000)

   for k, typename in ipairs(typenames) do
      local input = torch.randn(size):type(typename)
      local target = 1

      local ctype = t2cpu[typename]
      input = makeNonContiguous(input:type(ctype))
      local mod = nn.ClassNLLCriterion():type(ctype)

      local fout = mod:forward(input, target)
      local fgin = mod:backward(input, target):clone()

      local cinput = makeNonContiguous(input:type(typename))
      local ctarget = makeNonContiguous(torch.CudaTensor(1):fill(target))
      local cmod = nn.ClassNLLCriterion():type(typename)
      local cout = cmod:forward(cinput,ctarget)
      local cgin = cmod:backward(cinput,ctarget)

      mytester:assertlt(
         math.abs(fout-cout), precision_forward_type(precision_forward, typename),
            string.format('error on output with %s', typename))
      local gerr = cgin:double() - fgin:double()
      mytester:assertlt(gerr:abs():max(), precision_forward_type(precision_forward, typename),
         string.format('error on gradInput with %s', typename))
   end
end

function cunntest.ClassNLLCriterionSingleTargetWeights()
   local size = math.random(3000,5000)

   for k, typename in ipairs(typenames) do
      local input = torch.randn(size):type(typename)
      local target = 1
      local weights = torch.rand(size):type(typename)

      local ctype = t2cpu[typename]
      input = makeNonContiguous(input:type(ctype))
      weights = makeNonContiguous(weights:type(ctype))
      local mod = nn.ClassNLLCriterion(weights):type(ctype)

      local fout = mod:forward(input, target)
      local fgin = mod:backward(input, target):clone()

      local cinput = makeNonContiguous(input:type(typename))
      local cweights = makeNonContiguous(weights:type(typename))
      local ctarget = makeNonContiguous(torch.CudaTensor(1):fill(target))
      local cmod = nn.ClassNLLCriterion(cweights):type(typename)
      local cout = cmod:forward(cinput,ctarget)
      local cgin = cmod:backward(cinput,ctarget)

      mytester:assertlt(
         math.abs(fout-cout), precision_forward_type(precision_forward, typename),
            string.format('error on output with %s', typename))
      local gerr = cgin:double() - fgin:double()
      mytester:assertlt(gerr:abs():max(), precision_forward_type(precision_forward, typename),
         string.format('error on gradInput with %s', typename))
   end
end

function cunntest.ClassNLLCriterionMultipleTarget()
   local size = math.random(3000,5000)

   for k, typename in ipairs(typenames) do
      local input = torch.randn(size, size):type(typename)
      local target = makeNonContiguous(torch.randperm(size))

      local ctype = t2cpu[typename]
      input = makeNonContiguous(input:type(ctype))
      local mod = nn.ClassNLLCriterion():type(ctype)

      local fout = mod:forward(input, target)
      local fgin = mod:backward(input, target):clone()

      local cinput = makeNonContiguous(input:type(typename))
      local ctarget = makeNonContiguous(target:cuda())

      local cmod = nn.ClassNLLCriterion():type(typename)
      local cout = cmod:forward(cinput,ctarget)
      local cgin = cmod:backward(cinput,ctarget)

      mytester:assertlt(
        math.abs(fout-cout), precision_forward_type(precision_forward, typename),
          string.format('error on output with %s', typename))

      local gerr = cgin:double() - fgin:double()
      mytester:assertlt(gerr:abs():max(), precision_forward_type(precision_forward, typename),
        string.format('error on gradInput with %s', typename))
   end
end

function cunntest.SpatialClassNLLCriterion()
   local batchSize = math.random(5, 10)
   local h = math.random(300, 500)
   local w = math.random(300, 800)
   local classes = math.random(10,30)

   for k, typename in ipairs(typenames) do
      local input = torch.randn(batchSize, classes, h, w):type(typename)
      local target = makeNonContiguous(torch.Tensor(batchSize, h, w))
      target:apply(function() return math.random(1, classes) end)
      local ctype = t2cpu[typename]
      input = makeNonContiguous(input:type(ctype))
      local mod = nn.SpatialClassNLLCriterion():type(ctype)
      local fout = mod:forward(input, target)
      local fgin = mod:backward(input, target):clone()

      local cinput = makeNonContiguous(input:type(typename))
      local ctarget = makeNonContiguous(target:type(typename))

      local cmod = nn.SpatialClassNLLCriterion():type(typename)
      local cout = cmod:forward(cinput,ctarget)
      local cgin = cmod:backward(cinput,ctarget)
      cutorch.synchronize()

      mytester:assertlt(
        math.abs(fout-cout), precision_forward_type(precision_forward, typename),
          string.format('error on output with %s', typename))

      local gerr = cgin:double() - fgin:double()
      mytester:assertlt(gerr:abs():max(), precision_forward_type(precision_forward, typename),
          string.format('error on gradInput with %s', typename))
    end
end

function cunntest.ClassNLLCriterionMultipleTargetWeights()
   local size = math.random(3000,5000)

   for k, typename in ipairs(typenames) do
      local input = torch.randn(size, size):type(typename)
      local target = makeNonContiguous(torch.randperm(size))
      local weights = torch.rand(size):type(typename)

      local ctype = t2cpu[typename]
      input = makeNonContiguous(input:type(ctype))
      weights = makeNonContiguous(weights:type(ctype))
      local mod = nn.ClassNLLCriterion(weights):type(ctype)

      local fout = mod:forward(input, target)
      local fgin = mod:backward(input, target):clone()

      local cinput = makeNonContiguous(input:type(typename))
      local ctarget = makeNonContiguous(target:cuda())
      local cweights = makeNonContiguous(weights:type(typename))

      local cmod = nn.ClassNLLCriterion(cweights):type(typename)
      local cout = cmod:forward(cinput,ctarget)
      local cgin = cmod:backward(cinput,ctarget)

      mytester:assertlt(
        math.abs(fout-cout), precision_forward_type(precision_forward, typename),
          string.format('error on output with %s', typename))

      local gerr = cgin:double() - fgin:double()
      mytester:assertlt(gerr:abs():max(), precision_forward_type(precision_forward, typename),
        string.format('error on gradInput with %s', typename))
   end
end

function cunntest.ClassNLLCriterion_ignoreIndex()
   local numLabels = 10
   local batchsize = 4
   local ignoreIndex = -1
   local cri = nn.ClassNLLCriterion(nil, nil, ignoreIndex):cuda()
   local input = torch.randn(numLabels):cuda()
   local target = ignoreIndex
   mytester:assert(cri:forward(input, target) == 0)
   mytester:assert(cri:backward(input, target):abs():sum() == 0)
   local input = torch.randn(batchsize, numLabels):cuda()
   local target = torch.LongTensor(batchsize):random(1,numLabels)
   target[1] = ignoreIndex
   target = target:cudaLong()
   local output = cri:forward(input, target)
   local gradInput = cri:backward(input, target):clone()
   mytester:assert(gradInput[1]:abs():sum() == 0)
   local input, target = input:sub(2,batchsize), target:sub(2,batchsize)
   local output2 = cri:forward(input, target)
   mytester:assert(math.abs(output2 - output) < 0.0000001)
   local gradInput2 = cri:backward(input, target)
   mytester:assertTensorEq(gradInput2, gradInput:sub(2,batchsize), 0.0000001)
end

function cunntest.TemporalMaxPooling()
   local settings = {{2, 2}, {3, 3}, {4, 2}, {2, 4}, {3, 5}}

   for i, setting in ipairs(settings) do
      for k, typename in ipairs(typenames) do
        local input = torch.rand(16, 18, 3):type(typename)

        local ctype = t2cpu[typename]
        input = makeNonContiguous(input:type(ctype))
        local mod = nn.TemporalMaxPooling(setting[1], setting[2]):type(ctype)

        local fout = mod:forward(input)
        local fgout = makeNonContiguous(torch.rand(fout:size()):type(typename):type(ctype))
        local fgin = mod:backward(input, fgout):clone()

        local cinput = makeNonContiguous(input:type(typename))
        local cgout = makeNonContiguous(fgout:type(typename))
        local cmod = nn.TemporalMaxPooling(setting[1], setting[2]):type(typename)
        local cout = cmod:forward(cinput)
        local cgin = cmod:backward(cinput, cgout)

        local outerror = cout:double() - fout:double()
        mytester:assertlt(outerror:abs():max(), precision_forward_type(precision_forward, typename),
          string.format('error on output with %s', typename))

        local ginerror = cgin:double() - fgin:double()
        mytester:assertlt(ginerror:abs():max(), precision_backward_type(precision_backward, typename),
          string.format('error on gradInput with %s', typename))
      end
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

   for k, typename in ipairs(typenames) do
      local input = torch.randn(from,ini,inj,ink):type(typename)

      local ctype = t2cpu[typename]
      input = makeNonContiguous(input:type(ctype))
      local sconv = nn.VolumetricConvolution(from,to,ki,kk,kj,si,sk,sj):type(ctype)
      local groundtruth = sconv:forward(input)

      input = makeNonContiguous(input:type(typename))
      local gconv = nn.VolumetricConvolution(from,to,ki,kk,kj,si,sk,sj):type(typename)
      gconv.weight = sconv.weight:type(typename)
      gconv.bias = sconv.bias:type(typename)
      local rescuda = gconv:forward(input)

      local error = rescuda:double() - groundtruth:double()
      mytester:assertlt(error:abs():max(), precision_forward_type(precision_forward, typename),
        string.format('error on state (forward) with %s', typename))
      mytester:assert(groundtruth:isSize(rescuda:size()),
        string.format('size mismatch on state (forward) with %s', typename))
   end
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

   for k, typename in ipairs(typenames) do
      local input = torch.randn(bs,from,ini,inj, ink):type(typename)

      local ctype = t2cpu[typename]
      input = makeNonContiguous(input:type(ctype))
      local sconv = nn.VolumetricConvolution(from,to,ki,kk,kj,si,sj,sk):type(ctype)
      local groundtruth = sconv:forward(input)

      input = makeNonContiguous(input:type(typename))
      local gconv = nn.VolumetricConvolution(from,to,ki,kk,kj,si,sj,sk):type(typename)
      gconv.weight = sconv.weight:type(typename)
      gconv.bias = sconv.bias:type(typename)
      local rescuda = gconv:forward(input)

      local error = rescuda:double() - groundtruth:double()
      mytester:assertlt(error:abs():max(), precision_forward_type(precision_forward, typename),
        string.format('error on state (forward) with %s', typename))
      mytester:assert(groundtruth:isSize(rescuda:size()),
        string.format('size mismatch on state (forward) with %s', typename))
   end
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

   for k, typename in ipairs(typenames) do
      local input = torch.randn(from, ini, inj, ink):type(typename)
      local gradOutput = torch.randn(to, outi, outj, outk):type(typename)

      local ctype = t2cpu[typename]
      input = makeNonContiguous(input:type(ctype))
      gradOutput = makeNonContiguous(gradOutput:type(ctype))
      local sconv = nn.VolumetricConvolution(from,to,ki,kk,kj,si,sk,sj):type(ctype)
      sconv:forward(input)
      sconv:zeroGradParameters()
      local groundgrad = sconv:backward(input, gradOutput)
      local groundweight = sconv.gradWeight
      local groundbias = sconv.gradBias

      input = makeNonContiguous(input:type(typename))
      gradOutput = makeNonContiguous(gradOutput:type(typename))
      local gconv = nn.VolumetricConvolution(from,to,ki,kk,kj,si,sk,sj):type(typename)
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
      mytester:assert(groundgrad:isSize(rescuda:size()),
        string.format('size mismatch on state (forward) with %s', typename))
      mytester:assertlt(error:abs():max(), precision_backward_type(precision_backward, typename),
        string.format('error on state (backward) with %s', typename))
      mytester:assertlt(werror:abs():max(),
        precision_backward_conv_weightbias(precision_backward, typename, weightcuda:abs():max()),
        string.format('error on weight (backward) with %s', typename))
      mytester:assertlt(berror:abs():max(),
        precision_backward_conv_weightbias(precision_backward, typename, biascuda:abs():max()),
        string.format('error on bias (backward) with %s', typename))
   end
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

   for k, typename in ipairs(typenames) do
      local input = torch.randn(bs, from, ini, inj, ink):type(typename)
      local gradOutput = torch.randn(bs, to, outi, outj, outk):type(typename)

      local ctype = t2cpu[typename]
      input = makeNonContiguous(input:type(ctype))
      gradOutput = makeNonContiguous(gradOutput:type(ctype))
      local sconv = nn.VolumetricConvolution(from,to,ki,kk,kj,si,sk,sj):type(ctype)
      sconv:forward(input)
      sconv:zeroGradParameters()
      local groundgrad = sconv:backward(input, gradOutput)
      local groundweight = sconv.gradWeight
      local groundbias = sconv.gradBias

      input = makeNonContiguous(input:type(typename))
      gradOutput = makeNonContiguous(gradOutput:type(typename))
      local gconv = nn.VolumetricConvolution(from,to,ki,kk,kj,si,sk,sj):type(typename)
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
      mytester:assert(groundgrad:isSize(rescuda:size()),
        string.format('size mismatch on state (forward) with %s', typename))
      mytester:assertlt(error:abs():max(), precision_backward_type(precision_backward, typename),
        string.format('error on state (backward) with %s', typename))
      mytester:assertlt(werror:abs():max(),
        precision_backward_conv_weightbias(precision_backward, typename, weightcuda:abs():max()),
        string.format('error on weight (backward) with %s', typename))
      mytester:assertlt(berror:abs():max(),
        precision_backward_conv_weightbias(precision_backward, typename, biascuda:abs():max()),
        string.format('error on bias (backward) with %s', typename))
   end
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
   local padT = math.random(0,math.floor(kT/2)-1)
   local padH = math.random(0,math.floor(kH/2)-1)
   local padW = math.random(0,math.floor(kW/2)-1)
   local iF = math.random(1, 16) -- features
   local oT = math.floor((iT - kT + 2*padT) / dT + 1)
   local oH = math.floor((iH - kH + 2*padH) / dH + 1)
   local oW = math.floor((iW - kW + 2*padW) / dW + 1)

   for k, typename in ipairs(typenames) do
      local input = torch.Tensor(iF, iT, iH, iW):float():uniform(-1, 1):type(typename)

      local ctype = t2cpu[typename]
      input = makeNonContiguous(input:type(ctype))
      local layer = nn.VolumetricMaxPooling(kT, kW, kH, dT, dW, dH, padT, padW, padH):type(ctype)
      local output = layer:forward(input)

      local inputCUDA = makeNonContiguous(input:type(typename))
      local layerCUDA = layer:clone():type(typename)
      local outputCUDA = layerCUDA:forward(inputCUDA)

      local error = outputCUDA:double() - output:double()
      mytester:assertlt(error:abs():max(), precision_forward_type(precision_forward, typename),
        string.format('error on state (forward) with %s', typename))
   end
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
   local padT = math.random(0,math.floor(kT/2)-1)
   local padH = math.random(0,math.floor(kH/2)-1)
   local padW = math.random(0,math.floor(kW/2)-1)
   local iF = math.random(1, 16) -- features
   local oT = math.floor((iT - kT + 2*padT) / dT + 1)
   local oH = math.floor((iH - kH + 2*padH) / dH + 1)
   local oW = math.floor((iW - kW + 2*padW) / dW + 1)

   for k, typename in ipairs(typenames) do
      local input = torch.Tensor(iF, iT, iH, iW):float():uniform(-1, 1):type(typename)

      local ctype = t2cpu[typename]
      input = makeNonContiguous(input:type(ctype))
      local layer = nn.VolumetricMaxPooling(kT, kW, kH, dT, dW, dH, padT, padW, padH):type(ctype)
      local output = layer:forward(input)
      local gradOutput = makeNonContiguous(output:clone():uniform(-1, 1))

      local gradInput = layer:backward(input, gradOutput)

      local inputCUDA = makeNonContiguous(input:type(typename))
      local layerCUDA = layer:clone():type(typename)
      local outputCUDA = layerCUDA:forward(inputCUDA)
      local gradOutputCUDA = makeNonContiguous(gradOutput:type(typename))
      local gradInputCUDA = layerCUDA:backward(inputCUDA, gradOutputCUDA)

      local error = gradInputCUDA:double() - gradInput:double()
      mytester:assertlt(error:abs():max(), precision_forward_type(precision_forward, typename),
        string.format('error on state (backward) with %s', typename))
   end
end

function cunntest.VolumetricDilatedMaxPooling_forward_batch()
   local bs = math.random(4,8)
   local from = math.random(4,8)
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
   local padt = math.random(0,math.floor(kt/2)-1)
   local padi = math.random(0,math.floor(ki/2)-1)
   local padj = math.random(0,math.floor(kj/2)-1)
   local dilationt = math.random(1,10)
   local dilationi = math.random(1,10)
   local dilationj = math.random(1,10)
   local int = math.max((outt-1)*st+(dilationt*(kt-1)+1)-2*padt, kt)
   local ini = math.max((outi-1)*si+(dilationi*(ki-1)+1)-2*padi, ki)
   local inj = math.max((outj-1)*sj+(dilationj*(kj-1)+1)-2*padj, kj)
   local ceil_mode = math.random(0,1) == 1

   for k, typename in ipairs(typenames) do
      local input = torch.randn(bs,from,int,inj,ini):type(typename)

      local ctype = t2cpu[typename]
      input = makeNonContiguous(input:type(ctype))
      local sconv = nn.VolumetricDilatedMaxPooling(kt,ki,kj,st,si,sj,padt,padi,padj,dilationt,dilationi,dilationj):type(ctype)
      if ceil_mode then sconv:ceil() end
      local groundtruth = sconv:forward(input)

      input = makeNonContiguous(input:type(typename))
      local gconv = nn.VolumetricDilatedMaxPooling(kt,ki,kj,st,si,sj,padt,padi,padj,dilationt,dilationi,dilationj):type(typename)
      if ceil_mode then gconv:ceil() end
      local rescuda = gconv:forward(input)

      local error = rescuda:double() - groundtruth:double()
      mytester:assertlt(error:abs():max(), precision_forward_type(precision_forward, typename),
        string.format('error on state (forward) with %s', typename))
   end
end

function cunntest.VolumetricDilatedMaxPooling_backward_batch()
   local bs = math.random(4,8)
   local from = math.random(4,8)
   local to = from
   local kt = math.random(2,4)
   local ki = math.random(2,4)
   local kj = math.random(2,4)
   local st = math.random(2,4)
   local si = math.random(2,4)
   local sj = math.random(2,4)
   local outt = math.random(8,16)
   local outi = math.random(8,16)
   local outj = math.random(8,16)
   local padt = math.random(0,math.floor(kt/2)-1)
   local padi = math.random(0,math.floor(ki/2)-1)
   local padj = math.random(0,math.floor(kj/2)-1)
   local dilationt = math.random(1,10)
   local dilationi = math.random(1,10)
   local dilationj = math.random(1,10)
   local int = math.max((outt-1)*st+(dilationt*(kt-1)+1)-2*padt, kt)
   local ini = math.max((outi-1)*si+(dilationi*(ki-1)+1)-2*padi, ki)
   local inj = math.max((outj-1)*sj+(dilationj*(kj-1)+1)-2*padj, kj)
   local ceil_mode = math.random(0,1) == 1

   for k, typename in ipairs(typenames) do
      local input = torch.randn(bs,from,int,inj,ini):type(typename)
      local gradOutput = torch.randn(bs,to,outt,outj,outi):type(typename)

      local ctype = t2cpu[typename]
      input = makeNonContiguous(input:type(ctype))
      gradOutput = makeNonContiguous(gradOutput:type(ctype))
      local sconv = nn.VolumetricDilatedMaxPooling(kt,ki,kj,st,si,sj,padt,padi,padj,dilationt,dilationi,dilationj):type(ctype)
      if ceil_mode then sconv:ceil() end
      sconv:forward(input)
      sconv:zeroGradParameters()
      local groundgrad = sconv:backward(input, gradOutput)

      input = makeNonContiguous(input:type(typename))
      gradOutput = makeNonContiguous(gradOutput:type(typename))
      local gconv = nn.VolumetricDilatedMaxPooling(kt,ki,kj,st,si,sj,padt,padi,padj,dilationt,dilationi,dilationj):type(typename)
      if ceil_mode then gconv:ceil() end
      gconv:forward(input)
      gconv:zeroGradParameters()
      local rescuda = gconv:backward(input, gradOutput)

      local error = rescuda:double() - groundgrad:double()

      mytester:assertlt(error:abs():max(), precision_backward_type(precision_backward, typename),
        string.format('error on state (backward) with %s', typename))
   end
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
   local padt = math.random(0,math.floor(kt/2)-1)
   local padi = math.random(0,math.floor(ki/2)-1)
   local padj = math.random(0,math.floor(kj/2)-1)
   local it = math.max(((outt + padt*2 - kt)/st) +1, kt)
   local ii = math.max(((outi + padi*2 - ki)/si) +1, ki)
   local ij = math.max(((outj + padj*2 - kj)/sj) +1, kj)

   for k, typename in ipairs(typenames) do
      local ctype = t2cpu[typename]

      local pooler = nn.VolumetricMaxPooling(kt,ki,kj,st,si,sj,padt,padi,padj):type(ctype)
      local sunpool = nn.VolumetricMaxUnpooling(pooler):type(ctype)

      local original = makeNonContiguous(torch.randn(bs,from,it,ij,ii):type(typename):type(ctype))
      local input = makeNonContiguous(pooler:forward(original))
      local groundtruth = sunpool:forward(input)

      original = makeNonContiguous(original:type(typename))
      pooler = nn.VolumetricMaxPooling(kt,ki,kj,st,si,sj,padt,padi,padj):type(typename)
      local gunpool = nn.VolumetricMaxUnpooling(pooler):type(typename)

      input = makeNonContiguous(pooler:forward(original))
      local rescuda = gunpool:forward(input)

      local error = rescuda:double() - groundtruth:double()
      mytester:assertlt(error:abs():max(), precision_forward_type(precision_forward, typename),
        string.format('error on state (forward) with %s', typename))
   end
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
   local padt = math.random(0,math.floor(kt/2)-1)
   local padi = math.random(0,math.floor(ki/2)-1)
   local padj = math.random(0,math.floor(kj/2)-1)
   local it = math.max(((outt + padt*2 - kt)/st) +1, kt)
   local ii = math.max(((outi + padi*2 - ki)/si) +1, ki)
   local ij = math.max(((outj + padj*2 - kj)/sj) +1, kj)

   for k, typename in ipairs(typenames) do
      local ctype = t2cpu[typename]

      local pooler = nn.VolumetricMaxPooling(kt,ki,kj,st,si,sj,padt,padi,padj):type(ctype)
      local sunpool = nn.VolumetricMaxUnpooling(pooler):type(ctype)

      local original = makeNonContiguous(torch.randn(bs,from,it,ij,ii):type(typename):type(ctype))
      local input = makeNonContiguous(pooler:forward(original))
      local gradOutput = makeNonContiguous(torch.randn(original:size()):type(typename):type(ctype))
      sunpool:forward(input)
      sunpool:zeroGradParameters()
      local groundgrad = sunpool:backward(input, gradOutput)

      pooler = nn.VolumetricMaxPooling(kt,ki,kj,st,si,sj,padt,padi,padj):type(typename)
      local gunpool = nn.VolumetricMaxUnpooling(pooler):type(typename)

      original = makeNonContiguous(original:type(typename))
      input = makeNonContiguous(pooler:forward(original))
      gunpool:forward(input)

      gradOutput = makeNonContiguous(gradOutput:type(typename))
      gunpool:zeroGradParameters()
      local rescuda = gunpool:backward(input, gradOutput)

      local error = rescuda:double() - groundgrad:double()

      mytester:assertlt(error:abs():max(), precision_backward_type(precision_backward, typename),
        string.format('error on state (backward) with %s', typename))
   end
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

   for k, typename in ipairs(typenames) do
      local input = torch.Tensor(iF, iT, iH, iW):float():uniform(-1, 1):type(typename)

      local ctype = t2cpu[typename]
      input = makeNonContiguous(input:type(ctype))
      local layer = nn.VolumetricAveragePooling(kT, kW, kH, dT, dW, dH):type(ctype)
      local output = layer:forward(input)

      local inputCUDA = makeNonContiguous(input:type(typename))
      local layerCUDA = layer:clone():type(typename)
      local outputCUDA = layerCUDA:forward(inputCUDA)

      local error = outputCUDA:double() - output:double()
      mytester:assertlt(error:abs():max(), precision_forward_type(precision_forward, typename),
        string.format('error on state (forward) with %s', typename))
   end
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

   for k, typename in ipairs(typenames) do
      local input = torch.Tensor(iF, iT, iH, iW):float():uniform(-1, 1):type(typename)

      local ctype = t2cpu[typename]
      input = makeNonContiguous(input:type(ctype))
      local layer = nn.VolumetricAveragePooling(kT, kW, kH, dT, dW, dH):type(ctype)
      local output = layer:forward(input)
      local gradOutput = makeNonContiguous(output:clone():uniform(-1, 1))

      local gradInput = layer:backward(input, gradOutput)

      local inputCUDA = makeNonContiguous(input:type(typename))  local layerCUDA = layer:clone():type(typename)
      local outputCUDA = layerCUDA:forward(inputCUDA)   local gradOutputCUDA = makeNonContiguous(gradOutput:type(typename))
      local gradInputCUDA = layerCUDA:backward(inputCUDA, gradOutputCUDA)

      local error = gradInputCUDA:double() - gradInput:double()
      mytester:assertlt(error:abs():max(), precision_forward_type(precision_forward, typename),
        string.format('error on state (backward) with %s', typename))
   end
end

function cunntest.CMul_forward_batch()
   local bs = math.random(8,32)
   local nini = math.random(1,100)
   local ninj = math.random(1,100)
   local nink = math.random(1,100)

   local tm = {}
   local title = string.format('CMul forward %d %d %d %d', bs, nini, ninj, nink)
   times[title] = tm

   local input = makeNonContiguous(torch.randn(bs, nini, ninj, nink))
   local sconv = nn.CMul(nini, ninj, nink)
   local groundtruth = sconv:forward(input)
   local a = torch.Timer()
   for i = 1,nloop do
      groundtruth = sconv:forward(input)
   end
   tm.cpu = a:time().real

   input = makeNonContiguous(input:cuda())
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

   local input = makeNonContiguous(torch.randn(bs, nini, ninj, nink))
   local gradOutput = makeNonContiguous(torch.randn(bs, nini, ninj, nink))
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

   input = makeNonContiguous(input:cuda())
   gradOutput = makeNonContiguous(gradOutput:cuda())
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

    for k, typename in ipairs(typenames) do
      local input = torch.randn(nOutputPlane,h,w):type(typename)

      local ctype = t2cpu[typename]
      input = makeNonContiguous(input:type(ctype))
      local sconv = nn.PReLU(nOutputPlane):type(ctype)
      local groundtruth = sconv:forward(input)

      input = makeNonContiguous(input:type(typename))
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
        input = makeNonContiguous(input:type(ctype))
        gradOutput = makeNonContiguous(gradOutput:type(ctype))
        local sconv = nn.PReLU(nOutputPlane):type(ctype)
        local gconv = sconv:clone():type(typename)

        sconv:forward(input)
        sconv:zeroGradParameters()
        local groundgrad = sconv:backward(input, gradOutput)

        input = makeNonContiguous(input:type(typename))
        gradOutput = makeNonContiguous(gradOutput:type(typename))
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
              input = makeNonContiguous(input:type(ctype))
              local sconv = nn.RReLU(1/8, 1/3, inplace):type(ctype)
              if not train then
                  sconv:evaluate()
              end
              local groundtruth = sconv:forward(input:clone())

              input = makeNonContiguous(input:type(typename))
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
                input = makeNonContiguous(input:type(ctype))
                gradOutput = makeNonContiguous(gradOutput:type(ctype))
                local sconv = nn.RReLU(1/8, 1/3, inplace):type(ctype)
                if not train then
                  sconv:evaluate()
                end

                sconv:forward(input:clone())
                local groundgrad = sconv:backward(input, gradOutput:clone())

                local gconv = sconv:clone():type(typename)
                input = makeNonContiguous(input:type(typename))
                gradOutput = makeNonContiguous(gradOutput:type(typename))
                gconv:forward(input:clone())
                local rescuda = gconv:backward(input, gradOutput:clone())

                if not train then
                  local err = rescuda:double() - groundgrad:double()
                  mytester:assertlt(err:abs():max(), precision_backward_type(precision_backward, typename),
                    string.format('error on state', typename))
                end

                input = makeNonContiguous(-torch.rand(1000):type(typename))
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
    local pT = math.floor((kT-1)/2)
    local pH = math.floor((kH-1)/2)
    local pW = pH

    local inChan = math.random(1,32)
    local outChan = math.random(1,32)

    for k, typename in ipairs(typenames) do
      local ctype = t2cpu[typename]
      local module = nn.VolumetricFullConvolution(inChan, outChan, kT, kH, kW,
                                                  dT, dH, dW, pT, pH, pW):type(ctype);
      module.weight:fill(1);
      module.bias:fill(0.1);
      module.weight = module.weight:type(typename):type(ctype)
      module.bias = module.bias:type(typename):type(ctype)

      local bs = math.random(8,32)
      local inD = math.random(8,32)
      local inH = math.random(8,32)
      local inW = math.random(8,32)
      local outD = (inD - 1) * dT - 2 * pT + kT
      local outH = (inH - 1) * dH - 2 * pH + kH
      local outW = (inW - 1) * dW - 2 * pW + kW
      local input = makeNonContiguous(torch.Tensor(bs, inChan, inD, inH, inW):fill(1):type(typename):type(ctype))
      local gradOut = makeNonContiguous(torch.randn(bs, outChan, outD, outH, outW):type(typename):type(ctype))

      local outcpu = module:forward(input)
      local gradcpu = module:backward(input, gradOut)
      module:type(typename)
      local outgpu = module:forward(makeNonContiguous(input:type(typename)))
      local gradgpu = module:backward(makeNonContiguous(input:type(typename)), makeNonContiguous(gradOut:type(typename)))

      local error = outgpu:type(typename) - outcpu:type(typename)
      mytester:assertlt(error:abs():max(),
                        precision_forward_type(precision_forward, typename, outgpu:abs():max()),
                        string.format('error on state (forward) with %s', typename))

      local error = gradgpu:type(typename) - gradcpu:type(typename)
      mytester:assertlt(error:abs():max(),
                        precision_backward_type(precision_backward, typename),
                        string.format('error on state (backward) with %s', typename))
    end
end

function cunntest.VolumetricFullConvolution()
    for k, typename in ipairs(typenames) do
        local ctype = t2cpu[typename]
        local module = nn.VolumetricFullConvolution(3, 1, 3, 3, 3, 3, 3, 3):type(ctype);
        module.weight:fill(1);
        module.bias:fill(0.1);
        module:type(typename);

        local input = makeNonContiguous(torch.Tensor(1, 3, 2, 2, 2):zero());
        for c = 1,3 do
            input[1][c][1][1][1] = 1
        end
        local output = module:forward(input:type(typename))
        for t = 1,6 do
            for h = 1,6 do
                for w = 1,6 do
                    if t <= 3 and h <= 3 and w <= 3 then
                        mytester:assertlt(output[1][1][t][h][w] - 3.1, precision_forward_type(precision_forward, typename),
                          string.format('error on forward with %s', typename))
                    else
                        mytester:assertlt(output[1][1][t][h][w] - 0.1, precision_forward_type(precision_forward, typename),
                          string.format('error on forward with %s', typename))
                    end
                end
            end
        end

        module:zeroGradParameters()
        local gradOut = makeNonContiguous(torch.Tensor(1, 1, 6, 6, 6):fill(0.1));
        local gradIn = module:backward(makeNonContiguous(input:type(typename)), makeNonContiguous(gradOut:type(typename)))
        for t = 1,2 do
            for h = 1,2 do
                for w = 1,2 do
                    mytester:assertlt(gradIn[1][1][t][h][w] - 2.7, precision_backward_type(precision_backward, typename),
                                      string.format('error on backward input gradients with %s', typename))
                end
            end
        end

        mytester:assertlt(module.gradBias[1] - 21.6, precision_backward_type(precision_backward, typename),
                          string.format('error on backward gradBias with %s', typename))
        for c = 1,3 do
            for t = 1,3 do
                for h = 1,3 do
                    for w = 1,3 do
                        mytester:assertlt(module.gradWeight[c][1][t][h][w] - 0.1, precision_backward_type(precision_backward, typename),
                                          string.format('error on backward weight gradients with %s', typename))
                    end
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
   local ini = math.max((outi - 1) * si - 2 * padW + dilationW * (ki-1) + 1, ki)
   local inj = math.max((outj - 1) * sj - 2 * padH + dilationH * (kj-1) + 1, kj)
   local ink = math.max((outk - 1) * sk - 2 * padT + dilationT * (kk-1) + 1, kk)

   for k, typename in ipairs(typenames) do
      local input = torch.randn(from,ink,inj,ini):type(typename)
      input = makeNonContiguous(input)

      local ctype = t2cpu[typename]
      input = makeNonContiguous(input:type(ctype))
      local sconv = nn.VolumetricDilatedConvolution(from,to,kk,ki,kj,sk,si,sj,padT,padW,padH,dilationT,dilationW,dilationH):type(ctype)
      local output = sconv:forward(input)
      local gradOutput = makeNonContiguous(output:clone():normal())
      sconv:zeroGradParameters()
      local groundgrad = sconv:backward(input, gradOutput)
      local groundweight = sconv.gradWeight
      local groundbias = sconv.gradBias

      input = makeNonContiguous(input:type(typename))
      gradOutput = makeNonContiguous(gradOutput:type(typename))
      local gconv = nn.VolumetricDilatedConvolution(from,to,kk,ki,kj,sk,si,sj,padT,padW,padH,dilationT,dilationW,dilationH):type(typename)
      gconv.weight = sconv.weight:type(typename)
      gconv.bias = sconv.bias:type(typename)
      local rescuda = gconv:forward(input)
      gconv:zeroGradParameters()
      local gradcuda = gconv:backward(input, gradOutput)
      local weightcuda = gconv.gradWeight
      local biascuda = gconv.gradBias

      local error = rescuda:double() - output:double()
      local gerror = gradcuda:double() - groundgrad:double()
      local werror = weightcuda:double() - groundweight:double()
      local berror = biascuda:double() - groundbias:double()

      mytester:assertlt(error:abs():max(), precision_forward_type(precision_forward, typename),
        string.format('error on state (forward) with %s', typename))
      mytester:assertlt(gerror:abs():max(), precision_backward_type(precision_backward, typename),
        string.format('error on state (backward) with %s', typename))
      mytester:assertlt(werror:abs():max(),
        precision_backward_conv_weightbias(precision_backward, typename, weightcuda:abs():max()),
        string.format('error on weight (backward) with %s', typename))
      mytester:assertlt(berror:abs():max(),
        precision_backward_conv_weightbias(precision_backward, typename, biascuda:abs():max()),
        string.format('error on bias (backward) with %s', typename))
   end
end

function cunntest.LookupTable_forward()
   local nVocab = 10000
   local nDim = 100
   local nInput = 1000

   for k, typename in ipairs(typenames) do
      local input = makeNonContiguous(torch.LongTensor(nInput):random(nVocab))

      local ctype = t2cpu[typename]
      local sconv = nn.LookupTable(nVocab, nDim):type(ctype)
      local groundtruth = sconv:forward(input)

      input = makeNonContiguous(input:cuda())
      local gconv = sconv:type(typename)
      local rescuda = gconv:forward(input)

      local error = rescuda:double() - groundtruth:double()
      mytester:assertlt(error:abs():max(), precision_forward_type(precision_forward, typename),
        string.format('error on state with %s', typename))
   end
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

      for k, typename in ipairs(typenames) do
          local ctype = t2cpu[typename]
          local input, gradOutput
          if s.batch then
              input = makeNonContiguous(torch.LongTensor(s.nInput, 5):random(s.nVocab))
              gradOutput = makeNonContiguous(torch.randn(s.nInput, 5, s.nDim):type(typename):type(ctype))
          else
              input = makeNonContiguous(torch.LongTensor(s.nInput):random(s.nVocab))
              gradOutput = makeNonContiguous(torch.randn(s.nInput, s.nDim):type(typename):type(ctype))
          end

          local sconv = nn.LookupTable(s.nVocab, s.nDim, s.paddingValue):type(ctype)
          local gconv = sconv:clone():type(typename)
          if s.scaleGradByFreq then
              sconv = sconv:scaleGradByFreq()
              gconv = gconv:scaleGradByFreq()
          end

          sconv:forward(input)
          sconv:backward(input, gradOutput)

          input = makeNonContiguous(input:cuda())
          gradOutput = makeNonContiguous(gradOutput:type(typename))
          gconv:forward(input)
          gconv:backward(input, gradOutput)

          local weightGradError = gconv.gradWeight:double() - sconv.gradWeight:double()
          mytester:assertlt(weightGradError:abs():max(),
              precision_backward_conv_weightbias(precision_backward, typename, gconv.gradWeight:abs():max()),
              'error on weight for size ' .. tostring(s.nInput) ..
              ' nVocab: ' .. tostring(s.nVocab) ..
              ' nDim ' .. tostring(s.nDim) ..
              ' scaleGradByFreq: ' .. tostring(s.scaleGradByFreq) ..
              ' batch: ' .. tostring(s.batch) ..
              ' paddingValue: ' .. tostring(s.paddingValue) ..
              ' type:' .. typename)
      end
   end

   local nVocab = 10000
   local nDim = 128
   local nInput = 1000

   for k, typename in ipairs(typenames) do
      local input = makeNonContiguous(torch.LongTensor(nInput):random(nVocab))

      local ctype = t2cpu[typename]
      local gradOutput = makeNonContiguous(torch.randn(nInput, nDim):type(ctype))
      local sconv = nn.LookupTable(nVocab, nDim):type(ctype)
      local gconv = sconv:clone():type(typename)

      sconv:forward(input)
      sconv:backward(input, gradOutput)

      input = makeNonContiguous(input:cuda())
      gradOutput = makeNonContiguous(gradOutput:type(typename))
      gconv:forward(input)
      gconv:backward(input, gradOutput)

      local weightGradError = gconv.gradWeight:double() - sconv.gradWeight:double()
      mytester:assertlt(weightGradError:abs():max(), precision_backward_type(precision_backward, typename),
          string.format('error on weight with %s', typename))
   end
end

function cunntest.getParameters()
  -- tensors are non-contiguous but compact; they can be gathered
  for k, typename in ipairs(typenames) do
    local L = nn.Linear(10,10):type(typename)
    L.weight = torch[typename:match('torch.(%a+)')](10,10):t():fill(1)
    local tmp = torch[typename:match('torch.(%a+)')](10,10):fill(2)
    L.bias = tmp:select(1,2)
    local P = L:getParameters()
    mytester:asserteq(L.weight:mean(), 1)
    mytester:asserteq(L.bias:mean(), 2)
    mytester:asserteq(L.weight:storage(), L.bias:storage())
    mytester:asserteq(P:nElement(), 110)
    mytester:asserteq(P:storage():size(), 110)
    mytester:assertlt(L.bias[{ {10} }]:storageOffset() - 1, L.bias:storage():size())
  end
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
      input = makeNonContiguous(input:type(ctype))
      local module = nn.SpatialReflectionPadding(padL, padR, padT, padB):type(ctype)
      local groundtruth = module:forward(input)

      input = makeNonContiguous(input:type(typename))
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
       input = makeNonContiguous(input:type(ctype))
       gradOutput = makeNonContiguous(gradOutput:type(ctype))
       local module = nn.SpatialReflectionPadding(padL, padR, padT, padB):type(ctype)
       module:forward(input)
       module:zeroGradParameters()
       local groundgrad = module:backward(input, gradOutput)

       input = makeNonContiguous(input:type(typename))
       gradOutput = makeNonContiguous(gradOutput:type(typename))
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
      input = makeNonContiguous(input:type(ctype))
      local module = nn.SpatialReplicationPadding(padL, padR, padT, padB):type(ctype)
      local groundtruth = module:forward(input)

      input = makeNonContiguous(input:type(typename))
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
       input = makeNonContiguous(input:type(ctype))
       gradOutput = makeNonContiguous(gradOutput:type(ctype))
       local module = nn.SpatialReplicationPadding(padL, padR, padT, padB):type(ctype)
       module:forward(input)
       module:zeroGradParameters()
       local groundgrad = module:backward(input, gradOutput)

       input = makeNonContiguous(input:type(typename))
       gradOutput = makeNonContiguous(gradOutput:type(typename))
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

   for k, typename in ipairs(typenames) do
      local input = torch.rand(batch, plane, sizeZ, sizeY, sizeX):type(typename)

      local ctype = t2cpu[typename]
      input = makeNonContiguous(input:type(ctype))
      local module = nn.VolumetricReplicationPadding(pleft, pright, ptop, pbottom,
                                                     pfront, pback):type(ctype)
      local groundtruth = module:forward(input)

      input = makeNonContiguous(input:type(typename))
      local gmodule = nn.VolumetricReplicationPadding(pleft, pright, ptop, pbottom,
                                                      pfront, pback):type(typename)
      local rescuda = gmodule:forward(input)

      local error = rescuda:double() - groundtruth:double()
      mytester:assertlt(error:abs():max(),
                        precision_forward_type(precision_forward, typename),
                        string.format('error on state (forward) with %s', typename))
   end
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

   for k, typename in ipairs(typenames) do
      local input = torch.rand(batch, plane, sizeZ, sizeY, sizeX):type(typename)
      local gradOutput = torch.rand(
        batch, plane, sizeZ + pfront + pback, sizeY + ptop + pbottom,
        sizeX + pleft + pright
      ):type(typename)

      local ctype = t2cpu[typename]
      input = makeNonContiguous(input:type(ctype))
      gradOutput = makeNonContiguous(gradOutput:type(ctype))
      local module = nn.VolumetricReplicationPadding(pleft, pright, ptop, pbottom,
                                                     pfront, pback):type(ctype)
      module:forward(input)
      module:zeroGradParameters()
      local groundgrad = module:backward(input, gradOutput)

      input = makeNonContiguous(input:type(typename))
      gradOutput = makeNonContiguous(gradOutput:type(typename))
      local gmodule = nn.VolumetricReplicationPadding(pleft, pright, ptop, pbottom,
                                                      pfront, pback):type(typename)
      gmodule:forward(input)
      gmodule:zeroGradParameters()
      local rescuda = gmodule:backward(input, gradOutput)

      local error = rescuda:double() - groundgrad:double()
      mytester:assertlt(error:abs():max(),
                        precision_backward_type(precision_backward, typename),
                        string.format('error on state (backward) with %s', typename))
   end
end

function cunntest.ModuleConversionFunctions()
   local module = nn.Tanh() -- arbitrary module
   local input = torch.randn(10)

   module:cuda()
   mytester:assert(module:type() == 'torch.CudaTensor')
   module:forward(input:type('torch.CudaTensor'))

   module:cudaDouble()
   mytester:assert(module:type() == 'torch.CudaDoubleTensor')
   module:forward(input:type('torch.CudaDoubleTensor'))

   if cutorch.hasHalf then
      module:cudaHalf()
      mytester:assert(module:type() == 'torch.CudaHalfTensor')
      module:forward(input:type('torch.CudaHalfTensor'))
   end
end

function cunntest.IndexLinear()
   local isize = 500E3
   local osize = 250
   local weightDecay = 0.01
   local nnzMin = 1000
   local nnzMax = 1500
   local idxMin = 1
   local idxMax = isize
   local batchSize = 128
   local lr = 0.01
   local ntests = 1

   local errNorm = function(a, b)
      return torch.Tensor(1):fill(torch.cdiv((a - b):abs(), a:abs()):max())
   end

   local ilc = nn.IndexLinear(isize, osize):float()
   local ilg = nn.IndexLinear(isize, osize):float():cuda()

   local ilc2 = nn.IndexLinear(isize, osize):float()
   local ilg2 = nn.IndexLinear(isize, osize):float():cuda()

   local tot = 0
   local samples = 0
   local inputCPU = {{}, {}}
   local inputGPU = {{}, {}}
   local flatInputCPU = {torch.LongTensor(), torch.FloatTensor(), torch.LongTensor()}
   local flatInputGPU = {torch.CudaLongTensor(), torch.CudaTensor(), torch.CudaLongTensor()}
   local sizes = torch.LongTensor(batchSize)
   for i=1,batchSize do
      local n = torch.random(nnzMin, nnzMax)
      local indices = idxMin + torch.LongTensor():randperm(idxMax - idxMin)
      inputCPU[1][i] = indices[{{1,n}}]
      inputCPU[2][i] = torch.FloatTensor(n):uniform()
      inputGPU[1][i] = torch.CudaLongTensor(n):copy(inputCPU[1][i])
      inputGPU[2][i] = torch.CudaTensor(n):copy(inputCPU[2][i])
      sizes[i] = n
      tot = tot + n
   end
   flatInputCPU[1]:cat(inputCPU[1], 1)
   flatInputCPU[2]:cat(inputCPU[2], 1)
   flatInputCPU[3] = sizes

   flatInputGPU[1]:cat(inputGPU[1], 1)
   flatInputGPU[2]:cat(inputGPU[2], 1)
   flatInputGPU[3] = sizes:cudaLong()

   local inputSize = #inputCPU[1]
   local gradOutsCPU = torch.FloatTensor(inputSize, osize):uniform()
   local gradOutsGPU = torch.CudaTensor(inputSize, osize):copy(gradOutsCPU)

   local outputCPU, outputGPU
   local flatOutputCPU, flatOutputGPU

   ilc.weightDecay = weightDecay
   ilg.weightDecay = weightDecay
   ilc2.weightDecay = weightDecay
   ilg2.weightDecay = weightDecay

   ilc.weight:uniform()
   ilc.bias:fill(1)
   ilc2.weight:uniform()
   ilc2.bias:fill(1)

   ilg.weight:copy(ilc.weight)
   ilg.bias:copy(ilc.bias)
   ilg2.weight:copy(ilc2.weight)
   ilg2.bias:copy(ilc2.bias)

   ilc:zeroGradParameters()
   outputCPU = ilc:forward(inputCPU)
   ilc:backward(inputCPU, gradOutsCPU);
   ilc:updateParameters(lr)

   ilc2:zeroGradParameters()
   flatOutputCPU = ilc2:forward(flatInputCPU)
   ilc2:backward(flatInputCPU, gradOutsCPU);
   ilc2:updateParameters(lr)

   ilg:zeroGradParameters()
   outputGPU = ilg:forward(inputGPU)
   ilg:backward(inputGPU, gradOutsGPU);
   ilg:updateParameters(lr)

   ilg2:zeroGradParameters()
   flatOutputGPU = ilg2:forward(flatInputGPU)
   ilg2:backward(flatInputGPU, gradOutsGPU);
   ilg2:updateParameters(lr)

   mytester:assertTensorEq(errNorm(outputCPU, outputGPU:float()),
                           torch.Tensor(1):fill(0),
                           1E-5, "cunn.IndexLinear:forward failed for output")

   mytester:assertTensorEq(errNorm(flatOutputCPU, flatOutputGPU:float()),
                           torch.Tensor(1):fill(0),
                           1E-5, "cunn.IndexLinear:forward failed for flatOutput")

   mytester:assertTensorEq(ilc.bias,
                           ilg.bias:float(),
                           1E-5, "cunn.IndexLinear:backward+update failed for bias for tensor array")

   mytester:assertTensorEq(ilc.weight,
                           ilg.weight:float(),
                           1E-5, "cunn.IndexLinear:backward+update failed for weight for tensor array")

   mytester:assertTensorEq(ilc2.bias,
                           ilg2.bias:float(),
                           1E-5, "cunn.IndexLinear:backward+update failed for bias for flat input")

   mytester:assertTensorEq(ilc2.weight,
                           ilg2.weight:float(),
                           1E-5, "cunn.IndexLinear:backward+update failed for weight for flat input")

   ilc.weight:uniform()
   ilc.bias:fill(1)

   ilg.weight:copy(ilc.weight)
   ilg.bias:copy(ilc.bias)

   ilc2.weight:uniform()
   ilc2.bias:fill(1)

   ilg2.weight:copy(ilc2.weight)
   ilg2.bias:copy(ilc2.bias)

   outputCPU = ilc:forward(inputCPU)
   ilc:backwardUpdate(inputCPU, gradOutsCPU, lr);

   outputGPU = ilg:forward(inputGPU)
   ilg:backwardUpdate(inputGPU, gradOutsGPU, lr);

   flatOutputCPU = ilc2:forward(flatInputCPU)
   ilc2:backwardUpdate(flatInputCPU, gradOutsCPU, lr);

   flatOutputGPU = ilg2:forward(flatInputGPU)
   ilg2:backwardUpdate(flatInputGPU, gradOutsGPU, lr);

   mytester:assertTensorEq(errNorm(outputCPU, outputGPU:float()),
                           torch.Tensor(1):fill(0),
                           1E-5, "cunn.IndexLinear:forward failed for output")

   mytester:assertTensorEq(errNorm(flatOutputCPU, flatOutputGPU:float()),
                           torch.Tensor(1):fill(0),
                           1E-5, "cunn.IndexLinear:forward failed for flatOutput")

   mytester:assertTensorEq(ilc.bias,
                           ilg.bias:float(),
                           1E-5, "cunn.IndexLinear:backward+update failed for bias for tensor array")

   mytester:assertTensorEq(ilc.weight,
                           ilg.weight:float(),
                           1E-5, "cunn.IndexLinear:backward+update failed for weight for tensor array")

   mytester:assertTensorEq(ilc2.bias,
                           ilg2.bias:float(),
                           1E-5, "cunn.IndexLinear:backward+update failed for bias for flat input")

   mytester:assertTensorEq(ilc2.weight,
                           ilg2.weight:float(),
                           1E-5, "cunn.IndexLinear:backward+update failed for weight for flat input")
end

function cunntest.IndexLinearMaxNorm()
   local isize = 500E3
   local osize = 250
   local weightDecay = 0
   local nnzMin = 1000
   local nnzMax = 1500
   local idxMin = 1
   local idxMax = isize
   local batchSize = 128
   local lr = 0.01
   local ntests = 1

   local errNorm = function(a, b)
      return torch.Tensor(1):fill(torch.cdiv((a - b):abs(), a:abs()):max())
   end

   local ilc = nn.IndexLinear(isize, osize, nil, nil, nil, nil, 1):float()
   local ilg = nn.IndexLinear(isize, osize, nil, nil, nil, nil, 1):float():cuda()

   local tot = 0
   local samples = 0
   local inputCPU = {{}, {}}
   local inputGPU = {{}, {}}
   for i=1,batchSize do
      local n = torch.random(nnzMin, nnzMax)
      local indices = idxMin + torch.LongTensor():randperm(idxMax - idxMin)
      inputCPU[1][i] = indices[{{1,n}}]
      inputCPU[2][i] = torch.FloatTensor(n):uniform()
      inputGPU[1][i] = torch.CudaLongTensor(n):copy(inputCPU[1][i])
      inputGPU[2][i] = torch.CudaTensor(n):copy(inputCPU[2][i])
      tot = tot + n
   end

   local inputSize = #inputCPU[1]
   local gradOutsCPU = torch.FloatTensor(inputSize, osize):uniform()
   local gradOutsGPU = torch.CudaTensor(inputSize, osize):copy(gradOutsCPU)

   ilc.weightDecay = weightDecay
   ilg.weightDecay = weightDecay

   ilc.weight:uniform()
   ilc.weight:narrow(2,2,1):fill(1.0):cdiv(ilc.weight:narrow(2,1,1))
   ilc.bias:fill(1)

   ilg.weight:copy(ilc.weight)
   ilg.bias:copy(ilc.bias)

   outputCPU = ilc:forward(inputCPU)
   outputGPU = ilg:forward(inputGPU)

   mytester:assertTensorEq(errNorm(outputCPU, outputGPU:float()),
                           torch.Tensor(1):fill(0),
                           1E-5, "cunn.IndexLinear:forward failed for output")
end

function cunntest.GPU()
   local ndevice = cutorch.getDeviceCount()
   if ndevice < 2 then
      return
   end
   assert(nn.GPU, "Please update nn to latest version")

   for k, typename in ipairs(typenames) do
      local tolerance = 1e-6
      if typename == 'torch.CudaHalfTensor' then tolerance = 1e-3 end
      local originaldevice = cutorch.getDevice()

      local ctype = t2cpu[typename]
      cutorch.setDevice(1)
      local linear = nn.Linear(3,4):type(ctype)
      local linear2 = linear:clone():type(ctype)
      linear.mybuffer = {torch[typename:match('torch.(%a+)')](3)}

      local gpu = nn.GPU(linear, 2, 1)
      gpu:type(typename)

      mytester:assert(linear.mybuffer[1]:getDevice() == 2)
      mytester:assert(linear.weight:getDevice() == 2)
      mytester:assert(cutorch.getDevice() == originaldevice)

      local input = torch[typename:match('torch.(%a+)')](2,3):uniform(0,1)
      local output = gpu:forward(input)

      mytester:assert(linear.output:getDevice() == 2)
      mytester:assert(output:getDevice() == 1)
      mytester:assert(gpu._input:getDevice() == 2)

      local gradOutput = torch[typename:match('torch.(%a+)')](2,4):uniform(0,1)
      gpu:zeroGradParameters()
      mytester:assert(cutorch.getDevice() == 1)
      local gradInput = gpu:backward(input, gradOutput)

      mytester:assert(cutorch.getDevice() == 1)
      mytester:assert(gpu._gradOutput:getDevice() == 2)
      mytester:assert(linear.gradInput:getDevice() == 2)
      mytester:assert(gradInput:getDevice() == 1)

      mytester:assert(cutorch.getDevice() == 1)
      local input2, gradOutput2 = input:type(ctype), gradOutput:type(ctype)
      local output2 = linear2:forward(input2)
      linear2:zeroGradParameters()
      local gradInput2 = linear2:backward(input2, gradOutput2)


      mytester:assertTensorEq(input2:double(), input:double(), tolerance)
      mytester:assertTensorEq(gradInput2:double(), gradInput:double(), tolerance)

      local params, gradParams = gpu:parameters()
      local params2, gradParams2 = linear2:parameters()

      for i=1,#params do
        mytester:assertTensorEq(params2[i]:double(), params[i]:double(), tolerance)
        mytester:assertTensorEq(gradParams2[i]:double(), gradParams[i]:double(), tolerance)
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

      mytester:assertTensorEq(input2:double(), input2:double(), tolerance)
      mytester:assertTensorEq(gradInput2:double(), gradInput2:double(), tolerance)

      local params, gradParams = gpu:parameters()
      local params2, gradParams2 = gpu2:parameters()

      for i=1,#params do
        mytester:assert(params2[i]:getDevice() == params[i]:getDevice())
        mytester:assert(gradParams2[i]:getDevice() == gradParams[i]:getDevice())
        mytester:assertTensorEq(params2[i]:double(), params[i]:double(), tolerance)
        mytester:assertTensorEq(gradParams2[i]:double(), gradParams[i]:double(), tolerance)
      end


      -- test table input/output
      local lin1, lin2 = nn.Linear(3,4), nn.Linear(3,4)
      local para = nn.ParallelTable():add(lin1):add(lin2)
      local para2 = para:clone():type(ctype)
      local gpu = nn.GPU(para, 2, 1)

      gpu:type(typename)
      mytester:assert(lin1.weight:getDevice() == 2)
      mytester:assert(lin2.weight:getDevice() == 2)
      mytester:assert(cutorch.getDevice() == 1)

      local device3 = cutorch.getDeviceCount()
      local input = {
        torch[typename:match('torch.(%a+)')](2,3):uniform(0,1),
        cutorch.withDevice(device3, function() return torch[typename:match('torch.(%a+)')](2,3):uniform(0,1) end) -- tests input from multiple devices
      }
      local output = gpu:forward(input)

      mytester:assert(para.output[1]:getDevice() == 2)
      mytester:assert(para.output[2]:getDevice() == 2)
      mytester:assert(output[1]:getDevice() == 1)
      mytester:assert(output[2]:getDevice() == 1)
      mytester:assert(gpu._input[1]:getDevice() == 2)
      mytester:assert(gpu._input[2]:getDevice() == 2)

      local gradOutput = {
        torch[typename:match('torch.(%a+)')](2,4):uniform(0,1),
        cutorch.withDevice(device3, function() return torch[typename:match('torch.(%a+)')](2,4):uniform(0,1) end) -- tests gradOutput from multiple devices
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

      local input2, gradOutput2 = {input[1]:type(ctype), input[2]:type(ctype)}, {gradOutput[1]:type(ctype), gradOutput[2]:type(ctype)}
      local output2 = para2:forward(input2)
      para2:zeroGradParameters()
      local gradInput2 = para2:backward(input2, gradOutput2)

      mytester:assertTensorEq(input2[1]:double(), input[1]:double(), tolerance)
      mytester:assertTensorEq(input2[2]:double(), input[2]:double(), tolerance)
      mytester:assertTensorEq(gradInput2[1]:double(), gradInput[1]:double(), tolerance)
      mytester:assertTensorEq(gradInput2[2]:double(), gradInput[2]:double(), tolerance)

      local params, gradParams = gpu:parameters()
      local params2, gradParams2 = para2:parameters()

      for i=1,#params do
        mytester:assertTensorEq(params2[i]:double(), params[i]:double(), tolerance)
        mytester:assertTensorEq(gradParams2[i]:double(), gradParams[i]:double(), tolerance)
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
      mlp:type(typename)
      mytester:assert(cutorch.getDevice() == 1)

      local input = torch[typename:match('torch.(%a+)')](2,3):uniform(0,1)
      local gradOutput =   torch[typename:match('torch.(%a+)')](2,3):uniform(0,1)

      local output = mlp:forward(input)
      mlp:zeroGradParameters()
      local gradInput = mlp:backward(input, gradOutput)

      -- test CPU only

      local params, gradParams = mlp:parameters()

      mlp:type(ctype)

     local input2, gradOutput2 = input:type(ctype), gradOutput:type(ctype)

     local _cutorch = cutorch
     cutorch = nil

     local output2 = mlp:forward(input2)
     mlp:zeroGradParameters()
     local gradInput2 = mlp:backward(input2, gradOutput2)

     cutorch = _cutorch

     mytester:assertTensorEq(output:double(), output2:double(), tolerance)
     mytester:assertTensorEq(gradInput:double(), gradInput2:double(), tolerance)

     local params2, gradParams2 = mlp:parameters()

     for i=1,#params do
        mytester:assertTensorEq(params[i]:double(), params2[i]:double(), tolerance)
        mytester:assertTensorEq(gradParams[i]:double(), gradParams2[i]:double(), tolerance)
     end

     cutorch.setDevice(originaldevice)
   end
end

function cunntest.SpatialDepthWiseConvolution()
   local epsilon = 0.00001

   local SC = nn.SpatialConvolution
   local SDWC = nn.SpatialDepthWiseConvolution

   local function spatialDepthWiseConv(
         nInputPlane, multiplier, kernel, stride, padding, inputSize, weight, bias
      )
      local conv = SDWC(nInputPlane, multiplier, kernel, kernel, stride, stride, padding, padding)
      conv.weight = weight
      conv.bias = bias
      return conv
   end

   -- Utility spatialDepthWiseConv_util() function --------------------------------
   -- By Alfredo Canziani, alfredo.canziani@gmail.com -----------------------------
   local function spatialDepthWiseConv_util(
         nInputPlane, multiplier, kernel, stride, padding, inputSize, weight, bias
      )

      local conv = nn.Sequential()
      conv:add(nn.Contiguous())
      conv:add(nn.View(-1, 1, inputSize, inputSize))
      conv:add(SC(1, multiplier, kernel, kernel, stride, stride, padding, padding))

      local depthWiseConv = nn.Parallel(2, 2)
      for channel = 1, nInputPlane do
         local tempConv = conv:clone()
         tempConv:get(3).weight = weight:narrow(2, channel, 1):clone()
         tempConv:get(3).bias = bias:select(2, channel):clone()
        depthWiseConv:add(tempConv)
      end
      depthWiseConv:add(nn.Contiguous())
      return depthWiseConv
   end

   local n = 3 -- nInputPlane
   local s = 28 -- input height and width
   local b = 3 -- batch size
   local m = 4 -- multiplier
   local k = 3 -- kernel size
   local p = 1 -- padding
   local st = 1 -- stride

   local testBatch = 1e3 -- number of repetition

   local X = torch.rand(b, n, s, s):cuda() -- 1x3x299x299 images
   local weight = torch.rand(m, n, k, k):cuda() -- weight
   local bias = torch.rand(m, n):cuda() -- bias

   local model = spatialDepthWiseConv(n, m, k, st, p, s, weight, bias):cuda()
   local model_util = spatialDepthWiseConv_util(n, m, k, st, p, s, weight, bias):cuda()

   local Y_util = model_util:forward(X)
   local Y = model:forward(X)

   local abs_diff = Y_util:clone():csub(Y):abs()
   mytester:assert(torch.all(abs_diff:lt(epsilon)))
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
