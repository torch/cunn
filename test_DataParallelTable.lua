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

-- Create an instance of the test framework
local precision = 1e-5
local mytester = torch.Tester()
local test = {}

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
      local gClassifier = nn.DataParallelTable(1)
      for i = 1, numGpus do
         local curGpu = math.fmod(baseGpu+(i-1)-1, cutorch.getDeviceCount()) + 1
         cutorch.setDevice(curGpu)
         gClassifier:add(cpuClassifier:clone():cuda(), curGpu)
      end
      cutorch.setDevice(baseGpu)

      -- Now wrap them in layers that will split up the input tensor and join the
      -- output tensor (I know this seems stupid - and it is - but we need to test
      -- DataParallelTable under table inputs and when it is embedded in a network.
      local cNet = nn.Sequential()
      cNet:add(createSplitNetwork(2,3))
      cNet:add(cpuClassifier)
      cNet:add(nn.JoinTable(2))
      cNet:cuda()

      local gNet = nn.Sequential()
      gNet:add(createSplitNetwork(2,3))
      gNet:add(gClassifier)
      gNet:add(nn.JoinTable(2):cuda())
      gNet:get(1):cuda()
      gNet:get(3):cuda()

      -- Force in a serialization / deserialization pass ------------
      local file = serialize(gNet)
      gNet = nil
      collectgarbage()
      collectgarbage()
      gNet = deserialize(file)
      ----------------------------------------------------------------

      local cInput = torch.rand(batchSize, 3, height, width):cuda()
      local gInput = cInput:cuda()
      local cTarget = torch.rand(batchSize, 2):cuda()
      local gTarget = cTarget:cuda():cuda()

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

      local criterionCpu = nn.MSECriterion():cuda()
      local criterionGpu = criterionCpu:clone():cuda()

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

         mytester:assertlt((cOutput:float() - gOutput:float()):abs():max(),
         precision, 'fprop error ')
         mytester:assertlt((criterionCpu.gradInput:float() -
         criterionGpu.gradInput:float()):abs():max(), precision,
         'CRITERION BPROP error ')
         mytester:assertlt((cParams:float() - gParams:float()):abs():max(),
         precision, 'parameters error ')
         mytester:assertlt((cGradParams:float() - gGradParams:float()):abs():max(),
         precision, 'BPROP error (gradParams)')
         mytester:assertlt((cGradInput:float() - gGradInput:float()):abs():max(),
         precision, 'BPROP error (gradInput)')

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
   local net = nn.SpatialConvolution(3, 3, 3, 5):cuda()

   local dpt = nn.DataParallelTable(1)
   for i=1,numGpus do
      cutorch.withDevice(i, function()
         dpt:add(net:clone():cuda(), i)
      end)
   end

   -- Check for batches that are smaller than numGpus or don't divide evenly
   for _,batchSize in ipairs{numGpus-1,2*numGpus-1} do
      local input = torch.CudaTensor(batchSize,3,10,10):uniform(-1, 1)

      -- Check that forward works as expected
      local output = dpt:forward(input)
      local expected = net:forward(input)
      assert((expected - output):abs():max() < precision, 'unexpected output')

      local gradOutput = output:clone():uniform(-1, 1)
      local gradInput = dpt:updateGradInput(input, gradOutput)
      local expected = net:updateGradInput(input, gradOutput)
      assert((expected - gradInput):abs():max() < precision, 'unexpected gradInput')
   end
end


function test.DataParallelTable_emptyTensor()
   local net = nn.Sequential():add(nn.SelectTable(2)):add(nn.Linear(10,2)):cuda()

   local dpt = nn.DataParallelTable(1)
   for i=1,numGpus do
      cutorch.withDevice(i, function()
         dpt:add(net:clone():cuda(), i)
      end)
   end

   local input      = {torch.CudaTensor(0), torch.CudaTensor(numGpus, 10):fill(1)}
   local output     = dpt:forward(input)
   local expected   = net:forward(input)
   assert((output   - expected ):abs():max() < precision, 'unexpected output')
   local gradOutput = output:clone():uniform(-1,1)
   local gradInput  = dpt:backward(input, gradOutput)
   local expected   = net:backward(input, gradOutput)
   assert((expected[2] - gradInput[2]):abs():max() < precision, 'unexpected gradInput')
end

function test.DataParallelTable_type()
   local net = nn.SpatialConvolution(3, 3, 3, 5)

   local dpt = nn.DataParallelTable(1)
   for i=1,numGpus do
      cutorch.withDevice(i, function()
         dpt:add(net:clone(), i)
      end)
   end

   dpt:cuda()

   ok = pcall(function() dpt:float() end)
   assert(not ok, 'should not be able to call DataParallelTable:float()')
end

function test.DataParallelTable_sync()
   -- Test that DataParallelTable automatically syncParameters in updateOutput
   -- if you forget to call :syncParameters()
   local nSteps = 10
   local net = nn.Sequential()
      :add(nn.Linear(10, 10))
      :add(nn.ReLU(true))
      :add(nn.Linear(10, 10))
      :cuda()

   local dpt = nn.DataParallelTable(1)
   for i=1,numGpus do
      cutorch.withDevice(i, function()
         dpt:add(net:clone(), i)
      end)
   end

   local criterion = nn.MSECriterion():cuda()

   local optimState = {
      learningRate = 1,
      momentum = 0,
   }

   local input = torch.CudaTensor(numGpus,10)
   local target = torch.CudaTensor(numGpus,10)

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

   assert((paramsDpt - paramsBase):abs():max() < precision,
      'parameters do not match')
end

function test.DataParallelTable_serialize()
   -- Test serialization after getParameters()
   local net = nn.Linear(10, 10):cuda()

   local dpt = nn.DataParallelTable(1)
   for i=1,numGpus do
      cutorch.withDevice(i, function()
         dpt:add(net:clone():cuda(), i)
      end)
   end

   dpt:getParameters()
   dpt = deserialize(serialize(dpt))

   local input = torch.CudaTensor(numGpus,10):uniform(-1, 1)

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

function test.DataParallelTable_misc()
   local net = nn.Sequential()
      :add(nn.Linear(3, 10))
      :add(nn.ReLU())
      :add(nn.Linear(10, 7))

   local dpt = nn.DataParallelTable(1)
      :add(net, torch.range(1, numGpus):totable())
      :threads()
      :cuda()

   local input = torch.randn(8, 3):cuda()
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
   local net = nn.Sequential()
      :add(nn.LookupTable(10, 10))
      :add(nn.Linear(10, 7))
      :add(nn.ReLU())
      :cuda()

   local dpt = nn.DataParallelTable(1)
      :add(net, torch.range(1, numGpus):totable())
      :threads()
      :cuda()

   local input = torch.Tensor(5):random(10):cuda()
   local output1 = net:forward(input):clone()
   local gradOutput = output1:clone():uniform(-1, 1)
   local gradInput1 = net:backward(input, gradOutput):clone()

   local output2 = dpt:forward(input)
   local gradInput2 = dpt:backward(input, gradOutput)
   mytester:assertlt((output1 - output2):abs():max(), precision,
      'forward prop error')
   mytester:asserteq(gradInput2:nElement(), gradInput1:nElement())
end

function test.DataParallelTable_accGradParameters()
   local net = nn.Sequential()
      :add(nn.Linear(3, 10))
      :add(nn.ReLU())
      :add(nn.Linear(10, 7))
      :cuda()

   local inputs = {}
   local gradOutputs = {}
   for i=1,3 do
      inputs[i] = torch.randn(8, 3):cuda()
      gradOutputs[i] = torch.randn(8, 7):cuda()
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
         :add(net:clone(), torch.range(1, numGpus):totable())
      accumulateGradient(dpt)
      local output = dpt:forward(inputs[1])
      mytester:assertlt((output - expected):abs():max(), 1e-5, 'invalid output')
   end
end

function test.DataParallelTable_apply()
   local net = nn.Sequential()
      :add(nn.Linear(3, 10))
      :add(nn.ReLU())
      :add(nn.Linear(10, 7))
      :cuda()

   local inputs = {}
   local gradOutputs = {}
   for i=1,3 do
      inputs[i] = torch.randn(8, 3):cuda()
      gradOutputs[i] = torch.randn(8, 7):cuda()
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
            :add(net:clone(), torch.range(1, numGpus):totable())
         if usethreads then
            dpt:threads()
         end
         trainNetwork(dpt)
         local output = dpt:forward(inputs[1])
         mytester:assertlt((output - expected):abs():max(), 1e-5,
            'invalid output: flatten=' .. tostring(config[2]) ..
            ' threads=' .. tostring(usethreads))
      end
   end
end

function test.DataParallelTable_streams()
   local net = nn.Sequential()
      :add(nn.Linear(3, 10))
      :add(nn.ReLU())
      :add(nn.Linear(10, 7))
      :cuda()

   local input = torch.randn(8, 3):cuda()
   local gradOutput = torch.randn(8, 7):cuda()
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
               :cuda()
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
   a:cuda()

   local inputs = {torch.range(1,numGpus*5):reshape(numGpus,5):cuda(),
                   torch.range(1,5):reshape(1,5):cuda(),
                   torch.range(1,10):reshape(2,5):cuda(),
                  }

   for _, input in ipairs(inputs) do
      local output = a:forward(input)
      local gradInput = a:backward(input, output)
      mytester:assert(eq(input, output))
      mytester:assert(eq(input, gradInput))
   end

   a = nn.DataParallelTable(1)
   a:add(nn.ParallelTable():add(identity):add(identity), torch.range(1,numGpus):totable())
   a:cuda()

   for _, input in ipairs(inputs) do
      input = {input, input}
      local output = a:forward(input)
      local gradInput = a:backward(input, output)
      mytester:assert(eq(input, output))
      mytester:assert(eq(input, gradInput))
   end
end


function test.ProfileDataParallelTable()
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
            gNet:cuda()
         end
         for i = 1, numGpus do
            local curGpu = math.fmod(baseGpu+(i-1)-1, cutorch.getDeviceCount())+1
            cutorch.setDevice(curGpu)
            gNet:add(cNet:clone():cuda(), curGpu)
         end
         cutorch.setDevice(baseGpu)

         local input = torch.rand(batchSize, 3, height, width):cuda()
         local target = torch.rand(batchSize, 2):cuda()

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
         local criterion = nn.MSECriterion():cuda()
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
mytester:add(test)
mytester:run()
