require 'cunn'
require 'optim'

-- If fbcunn and fbnn exists we'll do a profile of DataParallel
local profileDp = pcall(function() require 'fbcunn'; require 'fbnn' end)

local baseGpu = 1  -- First GPU to use
local numGpus = cutorch.getDeviceCount()
torch.setdefaulttensortype('torch.DoubleTensor')
torch.setnumthreads(8)
cutorch.setDevice(baseGpu)

-- Create an instance of the test framework
local precision = 1e-5
local loosePrecision = 1e-4
local mytester = torch.Tester()
local jac = nn.Jacobian
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
   print('')

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
