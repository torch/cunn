require 'cunn'
require 'optim'

-- If fbcunn and fbnn exists we'll do a profile of DataParallel
profile_dp = pcall(function() require 'fbcunn'; require 'fbnn' end)

local base_gpu = 1  -- First GPU to use
local num_gpus = cutorch.getDeviceCount()
torch.setdefaulttensortype('torch.DoubleTensor')
torch.setnumthreads(8)
cutorch.setDevice(base_gpu)

-- Create an instance of the test framework
local precision = 1e-5
local loose_precision = 1e-4
local mytester = torch.Tester()
local jac = nn.Jacobian
local test = {}

function copyTable(x)  -- Shallow copy
  local ret = {}
  for key, value in pairs(x) do ret[key] = value end
  return ret
end

function createSplitNetwork(dim, dim_size)
  local split = nn.ConcatTable()
  for i = 1, dim_size do 
    split:add(nn.Narrow(dim, i, 1)) 
  end
  return split
end

-- Build a binary classifier that takes in a table of tensors and outputs
-- a table of tensors.  We will split the BATCHES across GPUs.
function buildNet(width, height, pool, feat, filt, table_in_out, num_convs)
  local net = nn.Sequential()
  if table_in_out then
    net:add(nn.JoinTable(2))  -- Join R,G,B tensors into RGB
  end
  assert(math.fmod(filt,2) == 1)
  for i = 1, num_convs do
    local fin = 3
    if (i > 1) then fin = feat end
    net:add(nn.SpatialConvolutionMM(fin, feat, filt, filt, 1, 1, (filt-1)/2))
    net:add(nn.Threshold())
  end
  net:add(nn.SpatialMaxPooling(pool, pool))
  net:add(nn.Reshape(width * height * feat / (pool * pool)))
  net:add(nn.Linear(width * height * feat / (pool * pool), 2))
  -- net:add(nn.SoftMax())  -- This is fake anyway, so just do regression :-)
  if table_in_out then
    net:add(createSplitNetwork(2,2))
  end
  return net
end

function test.DataParallelTable()
  collectgarbage()
  local width = 16
  local height = 16
  local pool = 4
  local feat = 8
  local filt = 5
  local num_convs = 2
  local num_sgd_steps = 10
  local sync_gpu_cpu_params_every = 4
  
  assert(num_gpus > 1)
  local batch_size = 2 * num_gpus
  
  -- Build a CPU model
  local cpu_classifier = buildNet(width, height, pool, feat, filt, true, 
    num_convs)

  -- Build a multi-GPU model
  local g_classifier = nn.DataParallelTable(1)
  for i = 1, num_gpus do
    local cur_gpu = math.fmod(base_gpu+(i-1)-1, cutorch.getDeviceCount()) + 1
    cutorch.setDevice(cur_gpu)
    g_classifier:add(cpu_classifier:clone():cuda(), cur_gpu)
  end
  cutorch.setDevice(base_gpu)

  -- Now wrap them in layers that will split up the input tensor and join the
  -- output tensor (I know this seems stupid - and it is - but we need to test
  -- DataParallelTable under table inputs and when it is embedded in a network.
  local c_net = nn.Sequential()
  c_net:add(createSplitNetwork(2,3))
  c_net:add(cpu_classifier)
  c_net:add(nn.JoinTable(2))
  c_net:cuda()

  local g_net = nn.Sequential()
  g_net:add(createSplitNetwork(2,3))
  g_net:add(g_classifier)
  g_net:add(nn.JoinTable(2):cuda())
  g_net:get(1):cuda()
  g_net:get(3):cuda()
  
  local c_input = torch.rand(batch_size, 3, height, width):cuda()
  local g_input = c_input:cuda()
  local c_target = torch.rand(batch_size, 2):cuda()
  local g_target = c_target:cuda():cuda()
  
  local c_params, c_gradParams = c_net:getParameters()
  local g_params, g_gradParams = g_net:getParameters()
  
  assert(cutorch.getDevice() == base_gpu, 
    'getParameters: didnt restore GPU state')
  
  -- Set up an MSE optimizer on the GPU and CPU
  local optim_state_cpu = {
    learningRate = 0.1,  -- Artificially big learning rate
    weightDecay = 0,
    momentum = 0.9,
    dampening = 0,
    learningRateDecay = 0,
    nesterov = true,
  }
  local optim_state_gpu = copyTable(optim_state_cpu)
  local optim_method = optim.sgd
  
  local criterion_cpu = nn.MSECriterion():cuda()
  local criterion_gpu = criterion_cpu:clone():cuda()
  
  for i = 1, num_sgd_steps do
    collectgarbage()
    local feval_cpu = function(x)
      if x ~= c_params then c_params:copy(x) end
      c_net:zeroGradParameters()
      -- FPROP + BPROP on CPU
      local output = c_net:forward(c_input)
      local err = criterion_cpu:forward(output, c_target)
      local gradOutput = criterion_cpu:backward(output, c_target)
      local gradInput = c_net:backward(c_input, gradOutput)
      return err, c_gradParams
    end
    
    local feval_gpu = function(x)
      if x ~= g_params then g_params:copy(x) end
      g_net:zeroGradParameters()
      assert(cutorch.getDevice() == base_gpu, 
        'zeroGradParameters: didnt restore GPU state')
      -- FPROP + BPROP on GPU
      local output = g_net:forward(g_input)
      assert(cutorch.getDevice() == base_gpu, 
        'DataParallelTable:forward didnt restore GPU state')
      local err = criterion_gpu:forward(output, g_target)
      local gradOutput = criterion_gpu:backward(output, g_target)
      local gradInput = g_net:backward(g_input, gradOutput)
      assert(cutorch.getDevice() == base_gpu, 
        'DataParallelTable:add didnt restore GPU state')
      return err, g_gradParams
    end
    
    -- Perform an SGD step on the GPU and CPU
    optim_method(feval_cpu, c_params, optim_state_cpu)
    optim_method(feval_gpu, g_params, optim_state_gpu)
    g_net:findModules('nn.DataParallelTable')[1]:syncParameters()
    assert(cutorch.getDevice() == base_gpu, 
      'DataParallelTable:syncParameters didnt restore GPU state')
    
    -- Now make sure that everything is the same
    local c_output = c_net.output
    local g_output = g_net.output
    local c_gradInput = c_net.gradInput
    local g_gradInput = g_net.gradInput
  
    mytester:assertlt((c_output:float() - g_output:float()):abs():max(), 
      precision, 'fprop error ')
    mytester:assertlt((criterion_cpu.gradInput:float() - 
      criterion_gpu.gradInput:float()):abs():max(), precision, 
      'CRITERION BPROP error ')
    mytester:assertlt((c_params:float() - g_params:float()):abs():max(),
      precision, 'parameters error ')
    mytester:assertlt((c_gradParams:float() - g_gradParams:float()):abs():max(), 
      precision, 'BPROP error (gradParams)')
    mytester:assertlt((c_gradInput:float() - g_gradInput:float()):abs():max(),
      precision, 'BPROP error (gradInput)')
    
    -- Sync the CPU and GPU weights every few "epochs" to prevent floating point
    -- drift between SGD iterations (ie, they will eventually be divergent after
    -- enough iters)
    if math.fmod(i, sync_gpu_cpu_params_every) == 0 then
      local cp = c_net:parameters()
      local gp = g_net:get(2):get(1):parameters()
      assert(#cp == #gp)
      for j = 1, #cp do 
        cp[j]:copy(gp[j]) 
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
  local num_convs = 4
  local num_repeats = 10
  
  local modules_to_test = {}
  modules_to_test['DataParallelTable'] = nn.DataParallelTable
  if profile_dp then 
    modules_to_test['DataParallel'] = nn.DataParallel 
  end 
  
  local device_count = num_gpus
  assert(device_count > 1)
  print('')
  
  for module_name, module in pairs(modules_to_test) do
    for num_gpus = 1, device_count do 
      collectgarbage()
      print('Profiling ' .. module_name .. ' with ' .. num_gpus .. ' gpus')
      local batch_size = 2 * 3 * 4
      assert(math.fmod(batch_size, num_gpus) == 0)
      
      -- Build a CPU model
      local c_net = buildNet(width, height, pool, feat, filt, false, num_convs)

      -- Build a multi-GPU model
      local g_net = module(1)
      if (module_name == 'DataParallel') then
        cutorch.setDevice(base_gpu)
        g_net:cuda()
      end
      for i = 1, num_gpus do
        local cur_gpu = math.fmod(base_gpu+(i-1)-1, cutorch.getDeviceCount())+1
        cutorch.setDevice(cur_gpu)
        g_net:add(c_net:clone():cuda(), cur_gpu)
      end
      cutorch.setDevice(base_gpu)
      
      local input = torch.rand(batch_size, 3, height, width):cuda()
      local target = torch.rand(batch_size, 2):cuda()
      
      local g_params, g_gradParams
      if (module_name == 'DataParallelTable') then
        g_params, g_gradParams = g_net:getParameters()
      end
      
      -- Set up an MSE optimizer on the GPU
      local optim_state = {
        learningRate = 0.1,
        weightDecay = 0,
        momentum = 0.9,
        dampening = 0,
        learningRateDecay = 0,
        nesterov = true,
      }
      local optim_method = optim.sgd 
      local criterion = nn.MSECriterion():cuda()
      local time_gpu_net = 0
      
      local opt
      if (module_name == 'DataParallel') then
        opt = nn.Optim(g_net, optim_state)
      end
      
      -- Call forward and backward once to hide allocations in profile
      do
        local output = g_net:forward(input)
        g_net:backward(input, output)
      end
      
      for i = 1, num_repeats do
        collectgarbage()
         
        local feval_gpu = function(x)
          if x ~= g_params then g_params:copy(x) end
          g_net:zeroGradParameters()
          local output = g_net:forward(input)
          local err = criterion:forward(output, target)
          local gradOutput = criterion:backward(output, target)
          local gradInput = g_net:backward(input, gradOutput)
          return err, g_gradParams
        end
        
        -- Perform an SGD step and profile it 
        sys.tic()
        if (module_name == 'DataParallelTable') then
          optim_method(feval_gpu, g_params, optim_state)
          g_net:findModules('nn.DataParallelTable')[1]:syncParameters()
        else
          opt:optimize(optim.sgd, input, target, criterion)
        end
        cutorch.synchronize()
        time_gpu_net = time_gpu_net + sys.toc()
        
        collectgarbage()
      end
      
      print('  Time per FPROP+BPROP: ' .. time_gpu_net / num_repeats)
    end
  end
end

-- Now run the test above
mytester:add(test)
mytester:run()
