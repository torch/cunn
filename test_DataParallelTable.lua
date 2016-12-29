require 'cunn'
require 'optim'

-- If fbcunn and fbnn exists we'll do a profile of DataParallel
local profileDp = pcall(function() require 'fbcunn'; require 'fbnn' end)

local baseGpu = 1  -- First GPU to use
local numGpus = cutorch.getDeviceCount()
torch.setdefaulttensortype('torch.DoubleTensor')
torch.setnumthreads(8)
cutorch.setDevice(baseGpu)
cutorch.reserveStreams(1)

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
       t2cpu['torch.CudaHalfTensor'] = 'torch.HalfTensor'
   end
end

local function half_max_error(maxabs)
  -- arbitrarily double the precision limit
  return 2 * ((maxabs and (2^(math.floor(math.log(maxabs) / math.log(2)))) * (2^(-10))) or 0)
end

-- Create an instance of the test framework
function precision(typename, max_error)
   if typename == 'torch.CudaHalfTensor' then
      return 5e-2 + half_max_error(max_error)
   else
      return 1e-5
   end
end

-- Create an instance of the test framework
local mytester = torch.Tester()
local test = torch.TestSuite()

local function copyTable(x)  -- Shallow copy
   local ret = {}
   for key, value in pairs(x) do ret[key] = value end
   return ret
end

local function createSplitNetwork(dim, dimSize)
   local split = nn.ConcatTable()
   for i = 1, dimSize do
      split:add(nn.Narrow(dim, i, 1))
   end
   return split
end

-- Build a binary classifier that takes in a table of tensors and outputs
-- a table of tensors.  We will split the BATCHES across GPUs.
local function buildNet(width, height, pool, feat, filt, tableInOut, numConvs)
   local net = nn.Sequential()
   if tableInOut then
      net:add(nn.JoinTable(2))  -- Join R,G,B tensors into RGB
   end
   assert(math.fmod(filt,2) == 1)
   for i = 1, numConvs do
      local fin = 3
      if (i > 1) then fin = feat end
      net:add(nn.SpatialConvolutionMM(fin, feat, filt, filt, 1, 1, (filt-1)/2))
      net:add(nn.Threshold())
   end
   net:add(nn.SpatialMaxPooling(pool, pool))
   net:add(nn.Reshape(width * height * feat / (pool * pool)))
   net:add(nn.Linear(width * height * feat / (pool * pool), 2))
   -- net:add(nn.SoftMax())  -- This is fake anyway, so just do regression :-)
   if tableInOut then
      net:add(createSplitNetwork(2,2))
   end
   return net
end

local function serialize(net)
   net:clearState()
   local uniq = sys.execute('echo "$(($(date +%s%N)/1000000))"')
   local f = torch.DiskFile(string.format('/tmp/%s', uniq), 'w')
   f:binary()
   f:writeObject(net)
   f:close()
   return string.format('/tmp/%s', uniq)
end

local function deserialize(file)
   local f = torch.DiskFile(file)
   f:binary()
   local net = f:readObject()
   f:close()
   os.execute(string.format('rm %s', file))
   return net
end


function test.DataParallelTable()
   for k, typename in ipairs(typenames) do
     test_DataParallelTable(typename)
   end
end

function test_DataParallelTable(gtype)
   local width = 16
   local height = 16
   local pool = 4
   local feat = 8
   local filt = 5
   local numConvs = 2
   local numSgdSteps = 10
   local syncGpuCpuParamsEvery = 4
   assert(numGpus > 1)

   -- test for various batchSizes, not necessarily multiples of nGpus:
   for _,batchSize in ipairs {2 * numGpus, 9, 15} do
      collectgarbage()

      -- Build a CPU model
      local cpuClassifier = buildNet(width, height, pool, feat, filt, true,
      numConvs)

      -- Build a multi-GPU model
      local gClassifier = nn.DataParallelTable(1):type(gtype)
      for i = 1, numGpus do
         local curGpu = math.fmod(baseGpu+(i-1)-1, cutorch.getDeviceCount()) + 1
         cutorch.setDevice(curGpu)
         gClassifier:add(cpuClassifier:clone():type(gtype), curGpu)
      end
      cutorch.setDevice(baseGpu)

      -- Now wrap them in layers that will split up the input tensor and join the
      -- output tensor (I know this seems stupid - and it is - but we need to test
      -- DataParallelTable under table inputs and when it is embedded in a network.
      local cNet = nn.Sequential()
      cNet:add(createSplitNetwork(2,3))
      cNet:add(cpuClassifier)
      cNet:add(nn.JoinTable(2))
      cNet:type(gtype)

      local gNet = nn.Sequential()
      gNet:add(createSplitNetwork(2,3))
      gNet:add(gClassifier)
      gNet:add(nn.JoinTable(2):type(gtype))
      gNet:get(1):type(gtype)
      gNet:get(3):type(gtype)

      -- Force in a serialization / deserialization pass ------------
      local file = serialize(gNet)
      gNet = nil
      collectgarbage()
      collectgarbage()
      gNet = deserialize(file)
      ----------------------------------------------------------------

      local cInput = torch.rand(batchSize, 3, height, width):type(gtype)
      local gInput = cInput:type(gtype)
      local cTarget = torch.rand(batchSize, 2):type(gtype)
      local gTarget = cTarget:type(gtype):type(gtype)

      local cParams, cGradParams = cNet:getParameters()
      local gParams, gGradParams = gNet:getParameters()

      assert(cutorch.getDevice() == baseGpu,
      'getParameters: didnt restore GPU state')

      -- Set up an MSE optimizer on the GPU and CPU
      local optimStateCpu = {
         learningRate = 0.1,  -- Artificially big learning rate
         weightDecay = 0,
         momentum = 0.9,
         dampening = 0,
         learningRateDecay = 0,
         nesterov = true,
      }
      local optimStateGpu = copyTable(optimStateCpu)
      local optimMethod = optim.sgd

      local criterionCpu = nn.MSECriterion():type(gtype)
      local criterionGpu = criterionCpu:clone():type(gtype)

      for i = 1, numSgdSteps do
         collectgarbage()
         local fevalCpu = function(x)
            if x ~= cParams then cParams:copy(x) end
            cNet:zeroGradParameters()
            -- FPROP + BPROP on CPU
            local output = cNet:forward(cInput)
            local err = criterionCpu:forward(output, cTarget)
            local gradOutput = criterionCpu:backward(output, cTarget)
            local gradInput = cNet:backward(cInput, gradOutput)
            return err, cGradParams
         end

         local fevalGpu = function(x)
            if x ~= gParams then gParams:copy(x) end
            gNet:zeroGradParameters()
            assert(cutorch.getDevice() == baseGpu,
            'zeroGradParameters: didnt restore GPU state')
            -- FPROP + BPROP on GPU
            local output = gNet:forward(gInput)
            assert(cutorch.getDevice() == baseGpu,
            'DataParallelTable:forward didnt restore GPU state')
            local err = criterionGpu:forward(output, gTarget)
            local gradOutput = criterionGpu:backward(output, gTarget)
            local gradInput = gNet:backward(gInput, gradOutput)
            assert(cutorch.getDevice() == baseGpu,
            'DataParallelTable:add didnt restore GPU state')
            return err, gGradParams
         end

         -- Perform an SGD step on the GPU and CPU
         optimMethod(fevalCpu, cParams, optimStateCpu)
         optimMethod(fevalGpu, gParams, optimStateGpu)
         gNet:findModules('nn.DataParallelTable')[1]:syncParameters()
         assert(cutorch.getDevice() == baseGpu,
         'DataParallelTable:syncParameters didnt restore GPU state')

         -- Now make sure that everything is the same
         local cOutput = cNet.output
         local gOutput = gNet.output
         local cGradInput = cNet.gradInput
         local gGradInput = gNet.gradInput

         mytester:assertlt((cOutput:double() - gOutput:double()):abs():max(),
         precision(gtype, cOutput:clone():double():abs():max()), 'fprop error ' .. gtype)
         mytester:assertlt((criterionCpu.gradInput:double() -
         criterionCpu.gradInput:double()):abs():max(),
         precision(gtype, criterionGpu.gradInput:clone():double():abs():max()),
         'CRITERION BPROP error ' .. gtype)
         mytester:assertlt((cParams:double() - gParams:double()):abs():max(),
         precision(gtype, cParams:clone():double():abs():max()), 'parameters error ' .. gtype)
         mytester:assertlt((cGradParams:double() - gGradParams:double()):abs():max(),
         precision(gtype, cGradParams:clone():double():abs():max()), 'BPROP error (gradParams) ' .. gtype)
         mytester:assertlt((cGradInput:double() - gGradInput:double()):abs():max(),
         precision(gtype, cGradInput:clone():double():abs():max()), 'BPROP error (gradInput) ' .. gtype)

         -- Sync the CPU and GPU weights every few "epochs" to prevent floating point
         -- drift between SGD iterations (ie, they will eventually be divergent after
         -- enough iters)
         if math.fmod(i, syncGpuCpuParamsEvery) == 0 then
            local cp = cNet:parameters()
            local gp = gNet:get(2):get(1):parameters()
            assert(#cp == #gp)
            for j = 1, #cp do
               cp[j]:copy(gp[j])
            end
         end
      end
   end
end

function test.DataParallelTable_smallBatch()
   for k, typename in ipairs(typenames) do
     test_DataParallelTable_smallBatch(typename)
   end
end

function test_DataParallelTable_smallBatch(gtype)
   local net = nn.SpatialConvolution(3, 3, 3, 5):type(gtype)

   local dpt = nn.DataParallelTable(1):type(gtype)
   for i=1,numGpus do
      cutorch.withDevice(i, function()
         dpt:add(net:clone():type(gtype), i)
      end)
   end

   -- Check for batches that are smaller than numGpus or don't divide evenly
   for _,batchSize in ipairs{numGpus-1,2*numGpus-1} do
      local input = torch[gtype:match('torch.(%a+)')](batchSize,3,10,10):uniform(-1, 1)

      -- Check that forward works as expected
      local output = dpt:forward(input)
      local expected = net:forward(input)
      assert((expected - output):abs():max() < precision(gtype, expected:clone():abs():max()), 'unexpected output')

      local gradOutput = output:clone():uniform(-1, 1)
      local gradInput = dpt:updateGradInput(input, gradOutput)
      local expected = net:updateGradInput(input, gradOutput)
      assert((expected - gradInput):abs():max() < precision(gtype, expected:clone():abs():max()), 'unexpected gradInput')
   end
end


function test.DataParallelTable_emptyTensor()
   for k, typename in ipairs(typenames) do
     test_DataParallelTable_emptyTensor(typename)
   end
end

function test_DataParallelTable_emptyTensor(gtype)
   local net = nn.Sequential():add(nn.SelectTable(2)):add(nn.Linear(10,2)):type(gtype)

   local dpt = nn.DataParallelTable(1):type(gtype)
   for i=1,numGpus do
      cutorch.withDevice(i, function()
         dpt:add(net:clone():type(gtype), i)
      end)
   end

   local input      = {torch[gtype:match('torch.(%a+)')](0), torch[gtype:match('torch.(%a+)')](numGpus, 10):fill(1)}
   local output     = dpt:forward(input)
   local expected   = net:forward(input)
   assert((output   - expected ):abs():max() < precision(gtype, expected:clone():abs():max()), 'unexpected output')
   local gradOutput = output:clone():uniform(-1,1)
   local gradInput  = dpt:backward(input, gradOutput)
   local expected   = net:backward(input, gradOutput)
   assert((expected[2] - gradInput[2]):abs():max() < precision(gtype, expected[2]:clone():abs():max()), 'unexpected gradInput')
end

function test.DataParallelTable_type()
   for k, typename in ipairs(typenames) do
     test_DataParallelTable_type(typename)
   end
end

function test_DataParallelTable_type(gtype)
   local ctype = t2cpu[gtype]
   local net = nn.SpatialConvolution(3, 3, 3, 5):type(ctype)

   local dpt = nn.DataParallelTable(1):type(gtype)
   for i=1,numGpus do
      cutorch.withDevice(i, function()
         dpt:add(net:clone(), i)
      end)
   end

   dpt:type(gtype)

   ok = pcall(function() dpt:type(ctype) end)
   assert(not ok, 'should not be able to call DataParallelTable:type(' .. ctype .. ')')
end

function test.DataParallelTable_sync()
   for k, typename in ipairs(typenames) do
     test_DataParallelTable_sync(typename)
   end
end

function test_DataParallelTable_sync(gtype)
   -- Test that DataParallelTable automatically syncParameters in updateOutput
   -- if you forget to call :syncParameters()
   local nSteps = 10
   local net = nn.Sequential()
      :add(nn.Linear(10, 10))
      :add(nn.ReLU(true))
      :add(nn.Linear(10, 10))
      :type(gtype)

   local dpt = nn.DataParallelTable(1):type(gtype)
   for i=1,numGpus do
      cutorch.withDevice(i, function()
         dpt:add(net:clone(), i)
      end)
   end

   local criterion = nn.MSECriterion():type(gtype)

   local optimState = {
      learningRate = 1,
      momentum = 0,
   }

   local input = torch[gtype:match('torch.(%a+)')](numGpus,10)
   local target = torch[gtype:match('torch.(%a+)')](numGpus,10)

   local function feval(net)
      local params, gradParams = net:getParameters()
      return params, function(x)
         net:zeroGradParameters()
         local output = net:forward(input)
         local err = criterion:forward(output, target)
         local gradOutput = criterion:backward(output, target)
         local gradInput = net:backward(input, gradOutput)
         return err, gradParams
      end
   end

   local paramsDpt, fevalDpt = feval(dpt)
   local paramsBase, fevalBase = feval(net)

   for i=1,nSteps do
      input:uniform(-1, 1)
      target:uniform(-1, 1)
      optim.sgd(fevalDpt, paramsDpt, optimState)
      optim.sgd(fevalBase, paramsBase, optimState)
   end

   assert((paramsDpt - paramsBase):abs():max() < precision(gtype, paramsDpt:clone():abs():max()),
      'parameters do not match')
end

function test.DataParallelTable_serialize()
   for k, typename in ipairs(typenames) do
     test_DataParallelTable_serialize(typename)
   end
end

function test_DataParallelTable_serialize(gtype)
   -- Test serialization after getParameters()
   local net = nn.Linear(10, 10):type(gtype)

   local dpt = nn.DataParallelTable(1):type(gtype)
   for i=1,numGpus do
      cutorch.withDevice(i, function()
         dpt:add(net:clone():type(gtype), i)
      end)
   end

   dpt:getParameters()
   dpt = deserialize(serialize(dpt))

   local input = torch[gtype:match('torch.(%a+)')](numGpus,10):uniform(-1, 1)

   -- Check that forward works as expected
   local output = dpt:forward(input)
   assert(output and output:sum() ~= 0, 'unexpected output')

   -- Zero the weights on the first tower and sync paramteters
   -- to check that Tensors are pointing to the proper storages
   dpt.flattenedParams[1][1]:zero()
   dpt:syncParameters()

   output = dpt:forward(input)
   assert(output:sum() == 0, 'weights not zeroed')
end


function test.DataParallelTable_flattenParameters()
   for k, typename in ipairs(typenames) do
     test_DataParallelTable_flattenParameters(typename)
   end
end

function test_DataParallelTable_flattenParameters(gtype)
    -- Wrap only a part of a network with data parallel table and
    -- check if the correct number of parameters have been copied
    local seq = nn.Sequential()
    local layer1 = nn.Linear(10, 10):type(gtype)
    local layer2 = nn.Linear(10, 5):type(gtype)
    local dpt = nn.DataParallelTable(1, true, true):threads():type(gtype)
    dpt:add(layer2, torch.range(1, numGpus):totable())
    seq:add(layer1):add(dpt)

    seq:getParameters()
    local input = torch.randn(7, 10):type(gtype)
    seq:forward(input)
    -- There are 55 parameters in layer 2 (50 + 5 bias weights)
    assert(dpt.flattenedParams[1][1]:size(1) == 55, "Incorrect number of " ..
        "parameters copied")
    -- Check grad weights
    assert(dpt.flattenedParams[1][2]:size(1) == 55, "Incorrect number of " ..
        "parameters copied")
end

function test.DataParallelTable_misc()
   for k, typename in ipairs(typenames) do
     test_DataParallelTable_misc(typename)
   end
end

function test_DataParallelTable_misc(gtype)
   local net = nn.Sequential()
      :add(nn.Linear(3, 10))
      :add(nn.ReLU())
      :add(nn.Linear(10, 7))

   local dpt = nn.DataParallelTable(1):type(gtype)
      :add(net, torch.range(1, numGpus):totable())
      :threads()
      :type(gtype)

   local input = torch.randn(8, 3):type(gtype)
   local output = dpt:forward(input)

   -- check that clone works
   dpt = dpt:clone()
   local output2 = dpt:forward(input)
   assert((output2 - output):abs():max() == 0)

   -- check findModules and listModules
   local modules = dpt:listModules()
   assert(#modules == #net:listModules() + 1)
   assert(torch.type(modules[1]) == 'nn.DataParallelTable')
   assert(torch.type(modules[2]) == 'nn.Sequential')

   assert(#dpt:findModules('nn.ReLU') == 1)
end

function test.DataParallelTable_noGradInput()
   for k, typename in ipairs(typenames) do
     test_DataParallelTable_noGradInput(typename)
   end
end

function test_DataParallelTable_noGradInput(gtype)
   local net = nn.Sequential()
      :add(nn.LookupTable(10, 10))
      :add(nn.Linear(10, 7))
      :add(nn.ReLU())
      :type(gtype)

   local dpt = nn.DataParallelTable(1)
      :add(net, torch.range(1, numGpus):totable())
      :threads()
      :type(gtype)

   local input = torch.Tensor(5):random(10):type(gtype)
   local output1 = net:forward(input):clone()
   local gradOutput = output1:clone():uniform(-1, 1)
   local gradInput1 = net:backward(input, gradOutput):clone()

   local output2 = dpt:forward(input)
   local gradInput2 = dpt:backward(input, gradOutput)
   mytester:assertlt((output1 - output2):abs():max(), precision(gtype, output1:clone():abs():max()),
      'forward prop error')
   mytester:asserteq(gradInput2:nElement(), gradInput1:nElement())
end

function test.DataParallelTable_accGradParameters()
   for k, typename in ipairs(typenames) do
     test_DataParallelTable_accGradParameters(typename)
   end
end

function test_DataParallelTable_accGradParameters(gtype)
   local net = nn.Sequential()
      :add(nn.Linear(3, 10))
      :add(nn.ReLU())
      :add(nn.Linear(10, 7))
      :type(gtype)

   local inputs = {}
   local gradOutputs = {}
   for i=1,3 do
      inputs[i] = torch.randn(8, 3):type(gtype)
      gradOutputs[i] = torch.randn(8, 7):type(gtype)
   end

   local configs = {
      {1, false, false},
      {1, true,  false},
   }

   local function accumulateGradient(m)
      m:zeroGradParameters()
      for i=1,#inputs do
         m:forward(inputs[i])
         m:backward(inputs[i], gradOutputs[i])
      end
      m:updateParameters(0.5)
   end

   local base = net:clone()
   accumulateGradient(base)
   local expected = base:forward(inputs[1])

   for _, config in ipairs(configs) do
      local dpt = nn.DataParallelTable(table.unpack(config))
         :add(net:clone(), torch.range(1, numGpus):totable()):type(gtype)
      accumulateGradient(dpt)
      local output = dpt:forward(inputs[1])
      mytester:assertlt((output - expected):abs():max(), precision(gtype, expected:clone():abs():max()), 'invalid output ' .. gtype)
   end
end

function test.DataParallelTable_apply()
   for k, typename in ipairs(typenames) do
     test_DataParallelTable_apply(typename)
   end
end

function test_DataParallelTable_apply(gtype)
   local net = nn.Sequential()
      :add(nn.Linear(3, 10))
      :add(nn.ReLU())
      :add(nn.Linear(10, 7))
      :type(gtype)

   local inputs = {}
   local gradOutputs = {}
   for i=1,3 do
      inputs[i] = torch.randn(8, 3):type(gtype)
      gradOutputs[i] = torch.randn(8, 7):type(gtype)
   end

   local configs = {
      {1, false, false},
      {1, true,  false},
   }

   local function trainNetwork(m)
      -- Test that apply doesn't break everything. This will be very slow
      -- in the training loop, but should still be correct.
      local function emptyFn() end
      m:apply(emptyFn)
      for i=1,#inputs do
         m:zeroGradParameters()
         m:forward(inputs[i])
         m:backward(inputs[i], gradOutputs[i])
         m:updateParameters(0.1)
         m:apply(emptyFn)
      end
   end

   local base = net:clone()
   trainNetwork(base)
   local expected = base:forward(inputs[1])

   for _, usethreads in ipairs{false,true} do
      for _, config in ipairs(configs) do
         local dpt = nn.DataParallelTable(table.unpack(config))
            :add(net:clone(), torch.range(1, numGpus):totable()):type(gtype)
         if usethreads then
            dpt:threads()
         end
         trainNetwork(dpt)
         local output = dpt:forward(inputs[1])
         mytester:assertlt((output - expected):abs():max(), precision(gtype, expected:clone():abs():max()),
            'invalid output: flatten=' .. tostring(config[2]) ..
            ' threads=' .. tostring(usethreads))
      end
   end
end

function test.DataParallelTable_streams()
   for k, typename in ipairs(typenames) do
     test_DataParallelTable_streams(typename)
   end
end

function test_DataParallelTable_streams(gtype)
   local net = nn.Sequential()
      :add(nn.Linear(3, 10))
      :add(nn.ReLU())
      :add(nn.Linear(10, 7))
      :type(gtype)

   local input = torch.randn(8, 3):type(gtype)
   local gradOutput = torch.randn(8, 7):type(gtype)
   local gOutput = net:forward(input):clone()
   net:zeroGradParameters()
   local gGradInput = net:backward(input, gradOutput):clone()

   local configs = {
      {1, false, false},
      {1, true,  false},
      {1, true,  true},
   }

   local function test(dpt)
      local output = dpt:forward(input)
      dpt:zeroGradParameters()
      local gradInput = dpt:backward(input, gradOutput)

      mytester:assert((output - gOutput):abs():max() == 0, 'invalid output')
      mytester:assert((gradInput - gGradInput):abs():max() == 0,
         'invalid gradInput')
   end

   for _, stream in ipairs{0, 1} do
      cutorch.setStream(stream)
      for _, config in ipairs(configs) do
         for _, threads in ipairs{false, true} do
            local dpt = nn.DataParallelTable(table.unpack(config))
               :add(net, torch.range(1, numGpus):totable())
               :type(gtype)
            if threads then
               dpt:threads(function()
                  cutorch.reserveStreams(1)
                  cutorch.setStream(stream)
               end)
            end
            test(dpt)
         end
      end
   end
   cutorch.setStream(0)
end

function test.DataParallelTable_emptyData()
   for k, typename in ipairs(typenames) do
     test_DataParallelTable_emptyData(typename)
   end
end

function test_DataParallelTable_emptyData(gtype)
   local function eq(a,b)
      if not torch.isTensor(a) then
         local res = true
         for i = 1, #a do
            res = res and eq(a[i], b[i])
         end
         return res
      end
      return a:clone():add(-b):abs():max() == 0
   end

   local identity = nn.Linear(5,5)
   identity.bias:zero()
   identity.weight=torch.eye(5)

   local a = nn.DataParallelTable(1)
   a:add(identity, torch.range(1,numGpus):totable())
   a:type(gtype)

   local inputs = {torch.range(1,numGpus*5):reshape(numGpus,5):type(gtype),
                   torch.range(1,5):reshape(1,5):type(gtype),
                   torch.range(1,10):reshape(2,5):type(gtype),
                  }

   for _, input in ipairs(inputs) do
      local output = a:forward(input)
      local gradInput = a:backward(input, output)
      mytester:assert(eq(input, output))
      mytester:assert(eq(input, gradInput))
   end

   a = nn.DataParallelTable(1)
   a:add(nn.ParallelTable():add(identity):add(identity), torch.range(1,numGpus):totable())
   a:type(gtype)

   for _, input in ipairs(inputs) do
      input = {input, input}
      local output = a:forward(input)
      local gradInput = a:backward(input, output)
      mytester:assert(eq(input, output))
      mytester:assert(eq(input, gradInput))
   end
end


function test.ProfileDataParallelTable()
   for k, typename in ipairs(typenames) do
     test_ProfileDataParallelTable(typename)
   end
end

function test_ProfileDataParallelTable(gtype)
   local width = 32
   local height = 32
   local pool = 4
   local feat = 128
   local filt = 7
   local numConvs = 4
   local numRepeats = 10

   local modulesToTest = {}
   modulesToTest['DataParallelTable'] = nn.DataParallelTable
   if profileDp then
      modulesToTest['DataParallel'] = nn.DataParallel
   end

   local deviceCount = numGpus
   assert(deviceCount > 1)

   for moduleName, module in pairs(modulesToTest) do
      for numGpus = 1, deviceCount do
         collectgarbage()
         print('Profiling ' .. moduleName .. ' with ' .. numGpus .. ' gpus')
         local batchSize = 2 * 3 * 4
         assert(math.fmod(batchSize, numGpus) == 0)

         -- Build a CPU model
         local cNet = buildNet(width, height, pool, feat, filt, false, numConvs)

         -- Build a multi-GPU model
         local gNet = module(1)
         if (moduleName == 'DataParallel') then
            cutorch.setDevice(baseGpu)
            gNet:type(gtype)
         elseif (moduleName == 'DataParallelTable') then
            gNet:type(gtype)
         end
         for i = 1, numGpus do
            local curGpu = math.fmod(baseGpu+(i-1)-1, cutorch.getDeviceCount())+1
            cutorch.setDevice(curGpu)
            gNet:add(cNet:clone():type(gtype), curGpu)
         end
         cutorch.setDevice(baseGpu)

         local input = torch.rand(batchSize, 3, height, width):type(gtype)
         local target = torch.rand(batchSize, 2):type(gtype)

         local gParams, gGradParams
         if (moduleName == 'DataParallelTable') then
            -- Force in a serialization / deserialization pass ------------
            local file = serialize(gNet)
            gNet = nil
            collectgarbage()
            collectgarbage()
            gNet = deserialize(file)
            ----------------------------------------------------------------
            gParams, gGradParams = gNet:getParameters()
         end

         -- Set up an MSE optimizer on the GPU
         local optimState = {
            learningRate = 0.1,
            weightDecay = 0,
            momentum = 0.9,
            dampening = 0,
            learningRateDecay = 0,
            nesterov = true,
         }
         local optimMethod = optim.sgd
         local criterion = nn.MSECriterion():type(gtype)
         local timeGpuNet = 0

         local opt
         if (moduleName == 'DataParallel') then
            opt = nn.Optim(gNet, optimState)
         end

         -- Call forward and backward once to hide allocations in profile
         do
            local output = gNet:forward(input)
            gNet:backward(input, output)
         end

         for i = 1, numRepeats do
            collectgarbage()

            local fevalGpu = function(x)
               if x ~= gParams then gParams:copy(x) end
               gNet:zeroGradParameters()
               local output = gNet:forward(input)
               local err = criterion:forward(output, target)
               local gradOutput = criterion:backward(output, target)
               local gradInput = gNet:backward(input, gradOutput)
               return err, gGradParams
            end

            -- Perform an SGD step and profile it
            sys.tic()
            if (moduleName == 'DataParallelTable') then
               optimMethod(fevalGpu, gParams, optimState)
               gNet:findModules('nn.DataParallelTable')[1]:syncParameters()
            else
               opt:optimize(optim.sgd, input, target, criterion)
            end
            cutorch.synchronize()
            timeGpuNet = timeGpuNet + sys.toc()

            collectgarbage()
         end

         print('  Time per FPROP+BPROP: ' .. timeGpuNet / numRepeats)
      end
   end
end

-- Now run the test above
--checkHalf() -- half not enabled yet for DataParallelTable
mytester:add(test)
mytester:run()
