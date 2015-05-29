local gpu_local_copy_buffers = {}
local base_gpu_index = 1  -- A constant

-- *****************************************************************************
-- Helper Functions
-- *****************************************************************************
-- queryGPUDeviceId - Function to query a tensor or table for the 
-- GPUID.  For tables we will search the table for CudaTensors, query their 
-- device and make sure the deviceIds of ALL CudaTensors are on the same GPU.
local function queryGPUDeviceId(object)
  if torch.type(object) == 'torch.CudaTensor' then
    return object:getDevice()
  end

  local deviceId

  -- Try finding a parameter
  local stack = {}  -- explicit stack to recurse on tables
  for key, param in pairs(object) do
    if key ~= 'modules' then
      stack[#stack+1] = param  -- Push onto the stack
    end
  end
  while #stack > 0 do
    local param = stack[#stack]; stack[#stack] = nil  -- Pop the stack
    if (torch.type(param) == 'table') then
      for i = 1, #param do stack[#stack+1] = param[i] end  -- Push onto stack
    elseif (torch.type(param) == 'torch.CudaTensor') then
      if (torch.numel(param) > 0) then
        -- Empty tensors are always on GPU "0"
        local cur_id = param:getDevice()
        if deviceId == nil then
          deviceId = cur_id
        else
          assert(deviceId == cur_id, 
            'Found CudaTensor instances from different devices')
        end
      end
    end
  end

  return deviceId
end

-- Get an avaliable GPU buffer for asyncGPUCopy.  It is used when the GPU tensor
-- is not contiguous.
local function getBuffer()
  local device = cutorch.getDevice()
  if not gpu_local_copy_buffers[device] then
    gpu_local_copy_buffers[device] = torch.CudaTensor()
  end
  return gpu_local_copy_buffers[device]
end

-- setDeviceSafe - Avoid redundant calls to setDevice
local function setDevice(gpuid)
  if (cutorch.getDevice() ~= gpuid) then
    cutorch.setDevice(gpuid)
  end
end

-- Asynchronous copy from source to dest from GPU to GPU.
-- This is borrowed (with modifications) from fbcunn.
local function asyncGPUCopy(dest, source)
  assert(torch.typename(dest) == 'torch.CudaTensor')
  assert(torch.typename(source) == 'torch.CudaTensor')
  local prev_device = cutorch.getDevice()
  
  local dest_gpuid = dest:getDevice()
  local source_gpuid = source:getDevice()
  
  if source_gpuid == dest_gpuid then
    -- if both tensors are on the same gpu normal operation works
    setDevice(dest_gpuid)
    dest:copy(source)
    setDevice(prev_device)
    return
  end
  
  if source:isContiguous() and dest:isContiguous() then
    -- if both tensors are contiguous operation across gpus works
    setDevice(dest_gpuid)
    dest:copy(source)
    setDevice(prev_device)
    return
  end

  -- Either the dest or the source are not contiguous.  we will need to do
  -- intermediate copies.
  local tmp_source = source
  if not source:isContiguous() then
    setDevice(source_gpuid)
    tmp_source = getBuffer()
    tmp_source:resizeAs(source)
    tmp_source:copy(source)  -- Make contiguous using a copy
  end

  setDevice(dest_gpuid)
  local tmp_dest = dest
  if not dest:isContiguous() then
    local tmp_dest = getBuffer()
    tmp_dest:resizeAs(tmp_source)
    tmp_dest:copy(tmp_source)
  end
  
  dest:copy(tmp_dest)
  
  cutorch.synchronize()  -- Ensures we keep the buffer for the copy duration
  
  -- Put the device back to what it was.
  setDevice(prev_device)
end

local function equalSize(size_table1, size_table2)
  if (#size_table1 ~= #size_table2) then
    return false
  end
  for i = 1, #size_table1 do
    if size_table1[i] ~= size_table2[i] then return false end
  end
  return true
end

-- deepTensorsCopy - perform an elementwise copy of the tensors in the nested 
-- table. We assume that the tables are properly initialized (ie same size and 
-- structure), although we will assert it.
local function deepTensorsCopy(dst, src)
  if (torch.type(src) == 'table') then
    assert(torch.type(dst) == 'table' and #src == #dst)
    for i = 1, #src do deepTensorsCopy(dst[i], src[i]) end
  elseif torch.type(src):find('torch%..+Tensor') then
    assert(torch.type(dst):find('torch%..+Tensor'))
    assert(dst:isSameSizeAs(src))
    asyncGPUCopy(dst, src)
  else
    error('input must be a nested table of tensors!')
  end 
end

-- deepTensorsAdd - perform an elementwise add of the tensors in the nested 
-- table. We assume that the tables are properly initialized (ie same size and 
-- structure), although we will assert it.
--
-- Note: this is necessary because add() will malloc new memory on the cuda
-- driver side every time we want to get new memory!  Therefore, we actually
-- need to copy src to the dst gpu
local function deepTensorsAdd(dst, src)
  if (torch.type(src) == 'table') then
    assert(torch.type(dst) == 'table' and #src == #dst)
    for i = 1, #src do deepTensorsAdd(dst[i], src[i]) end
  elseif torch.type(src):find('torch%..+Tensor') then
    assert(torch.type(dst):find('torch%..+Tensor'))
    assert(dst:isSameSizeAs(src))
    
    local dst_gpuid = dst:getDevice()
    local src_gpuid = src:getDevice()
    local cur_gpuid = cutorch:getDevice()
    setDevice(dst_gpuid)
    
    -- Copy src over to a buffer on the dst GPU
    local src_buffer_on_dst_gpu = src
    if (dst_gpuid ~= src_gpuid) then
      src_buffer_on_dst_gpu = getBuffer()
      src_buffer_on_dst_gpu:resizeAs(src)
      assert(src:isContiguous())
      src_buffer_on_dst_gpu:copy(src)
    end
    
    -- Perform the actual add
    dst:add(src_buffer_on_dst_gpu)
    if (dst_gpuid ~= src_gpuid) then
      -- Ensures we get to keep the buffer for the duration of the add
      cutorch.synchronize()
    end
    
    setDevice(cur_gpuid)  -- Put the GPU id back to what it was
  else
    error('input must be a nested table of tensors!')
  end 
end

-- *****************************************************************************
-- DataParallelTable
-- *****************************************************************************
local DataParallelTable, parent = torch.class('nn.DataParallelTable', 
  'nn.Container')

function DataParallelTable:__init(dimension)
  parent.__init(self)
  if not dimension then
    error "must specify a dimension!"
  end

  self.dimension = dimension
  self.modules = {}
  self.gpu_assignments = {}  -- Which gpuid each module sits on
  self.input_gpu = {}  -- inputs for each gpu
  self.gradOutput_gpu = {} -- gradOutputs for each gpu
  self.output_gpu = {} -- outputs for each gpu
  self.gradInput_gpu = {} -- gradInput for each gpu
end

-- NOTE: The input should be on the FIRST added GPU device, and similarly the 
-- output will be on the FIRST GPU device.
function DataParallelTable:add(module, gpuid)
  assert(gpuid <= cutorch.getDeviceCount() and gpuid >= 1)
  assert(#self.modules == #self.gpu_assignments)

  self.modules[#self.modules + 1] = module
  self.gpu_assignments[#self.gpu_assignments + 1] = gpuid

  return self
end

function DataParallelTable:get(index)
  return self.modules[index]
end

function DataParallelTable:updateOutput(input)
  local base_gpuid = self.gpu_assignments[base_gpu_index]
  assert(queryGPUDeviceId(input) == base_gpuid, 'Input is not on gpu ' ..
    base_gpuid)

  local prev_gpuid = cutorch.getDevice()

  -- distribute the input to GPUs
  for i = 1, #self.modules do
    local gpuid = self.gpu_assignments[i]
    -- Split the tensors in the input nested table to the GPU with gpuid
    -- _distributeTensorRecursive(src,dst,src_gpuid,src_ind,dst_gpuid,dst_ind)
    self.input_gpu[gpuid] = self:_distributeTensorRecursive(input, 
      self.input_gpu[gpuid], base_gpuid, base_gpu_index, gpuid, i)  
  end
  
  cutorch.synchronize()

  -- update output for each module asynchronously
  for i, module in ipairs(self.modules) do   
    local gpuid = self.gpu_assignments[i]
    setDevice(gpuid)
    self.output_gpu[gpuid] = module:updateOutput(self.input_gpu[gpuid])
  end
  
  cutorch.synchronize()

  -- concatenate the outputs to the base GPU
  for i = 1, #self.modules do
    local gpuid = self.gpu_assignments[i]
    -- Merge the tensors in the input nested table to the GPU with gpuid
    -- _ConcatTensorRecursive(src,dst,src_gpuid,src_ind,dst_gpuid,dst_ind)
    self.output = self:_ConcatTensorRecursive(self.output_gpu[gpuid], 
      self.output, gpuid, i, base_gpuid, base_gpu_index)
  end

  setDevice(prev_gpuid)

  return self.output
end

function DataParallelTable:updateGradInput(input, gradOutput)
  -- We assume that updateOutput has already been called (therefore input_gpu
  -- has been populated)
  local base_gpuid = self.gpu_assignments[base_gpu_index]
  assert(queryGPUDeviceId(gradOutput) == base_gpuid, 
    'gradOutput is not on gpu ' .. base_gpuid)

  local prev_gpuid = cutorch.getDevice()

  -- distribute the gradOutput to GPUs
  for i = 1, #self.modules do
    local gpuid = self.gpu_assignments[i]
    -- Split the tensors in the input nested table to the GPU with gpuid
    -- _distributeTensorRecursive(src,dst,src_gpuid,src_ind,dst_gpuid,dst_ind)
    self.gradOutput_gpu[gpuid] = self:_distributeTensorRecursive(gradOutput, 
      self.gradOutput_gpu[gpuid], base_gpuid, base_gpu_index, gpuid, i)
  end
  
  cutorch.synchronize()

  -- update gradInput for each module asynchronously
  for i, module in ipairs(self.modules) do   
    local gpuid = self.gpu_assignments[i]
    setDevice(gpuid)
    self.gradInput_gpu[gpuid] = module:updateGradInput(self.input_gpu[gpuid],
      self.gradOutput_gpu[gpuid])
  end
  
  cutorch.synchronize()

  -- concatenate the outputs to the base GPU
  for i = 1, #self.modules do
    local gpuid = self.gpu_assignments[i]
    -- Merge the tensors in the input nested table to the GPU with gpuid
    -- _ConcatTensorRecursive(src,dst,src_gpuid,src_ind,dst_gpuid,dst_ind)
    self.gradInput = self:_ConcatTensorRecursive(self.gradInput_gpu[gpuid], 
      self.gradInput, gpuid, i, base_gpuid, base_gpu_index)
  end
  
  cutorch.synchronize()

  setDevice(prev_gpuid)

  return self.gradInput
end

function DataParallelTable:accGradParameters(input, gradOutput, scale)
  -- We assume updateGradInput has already been called (so gradOutput has
  -- already been populated)
  local prev_gpuid = cutorch.getDevice()
  local base_gpuid = self.gpu_assignments[base_gpu_index]
  
  scale = scale or 1
  -- Calculate the gradWeight + gradBias on each sub-module
  for i, module in ipairs(self.modules) do
    local gpuid = self.gpu_assignments[i]
    setDevice(gpuid)
    module:accGradParameters(self.input_gpu[gpuid], self.gradOutput_gpu[gpuid],
      scale)
  end
  
  cutorch.synchronize()  -- We have to wait until accGradParameters has finished
  
  -- Accumulate the gradients onto one GPU (the first one)
  -- TODO: Parallelize this (ie a parallel merge)
  base_params, base_grad_params = self.modules[base_gpu_index]:parameters()
  --print(base_grad_params)  -- TODO: Temp code
  for i, module in ipairs(self.modules) do
    if (i ~= base_gpu_index) then
      params, grad_params = self.modules[i]:parameters()
      deepTensorsAdd(base_grad_params, grad_params)  -- dst, src
      cutorch.synchronize()
    end
  end

  setDevice(prev_gpuid)
end

function DataParallelTable:syncParameters()
  local prev_gpuid = cutorch.getDevice()
  base_params, base_grad_params = self.modules[base_gpu_index]:parameters()
  -- TODO: Parallelize this (ie a parallel copy)
  for i, module in ipairs(self.modules) do
    if (i ~= base_gpu_index) then
      params, grad_params = self.modules[i]:parameters()
      deepTensorsCopy(params, base_params)  -- dst, src
    end
  end
  cutorch.synchronize()
  
  setDevice(prev_gpuid)
end

-- For compatability with nn.Optim from fbcunn
function DataParallelTable:_mixGrads()
  self:syncParameters()
end

function DataParallelTable:accUpdateGradParameters(input, gradOutput, lr)
  error("accUpdateGradParameters not supported for DataParallelTable.")
end

function DataParallelTable:zeroGradParameters()
  local prev_gpuid = cutorch.getDevice()
  for i, module in ipairs(self.modules) do
    setDevice(self.gpu_assignments[i])
    module:zeroGradParameters()
  end
  setDevice(prev_gpuid)
end

function DataParallelTable:updateParameters(learningRate)
  error("updateParameters not supported for DataParallelTable.")
end

function DataParallelTable:parameters()
  local prev_gpuid = cutorch.getDevice()
  setDevice(self.gpu_assignments[1])
  local ret = {self.modules[1]:parameters()}
  setDevice(prev_gpuid)
  return unpack(ret)
end

function DataParallelTable:share(mlp,...)
  error("Share not supported for DataParallelTable.")
end

function DataParallelTable:clone()
  error("clone not supported for DataParallelTable.")
end

function DataParallelTable:reset(stdv)
  local prev_gpuid = cutorch.getDevice()
  for i, module in ipairs(self.modules) do
    setDevice(self.gpu_assignments[i])
    module:reset(stdv)
  end
  setDevice(prev_gpuid)
end

function DataParallelTable:name()
  return 'DataParallelTable'
end

function DataParallelTable:type(type_str)
  error("type() not supported for DataParallelTable.")
end

function DataParallelTable:_calculateSliceRange(tensor, id)
  local outerDim = tensor:size(self.dimension)
  if outerDim % #self.modules ~= 0 then
    error("cannot evenly divide " .. outerDim .. " inputs to " ..
      #self.modules .. " modules")
  end
  local eltsPerMod = outerDim / #self.modules
  local rangeStart = (id - 1) * eltsPerMod + 1
  local rangeEnd = rangeStart + eltsPerMod - 1
  return {rangeStart, rangeEnd}
end

-- _distributeTensorRecursive - if the src is a tensor then the function slices
-- it long self.dimension and copies each portion into each child module. 
-- Otherwise it does a recursive call on tables.
function DataParallelTable:_distributeTensorRecursive(src, dst, src_gpuid, 
  src_index, dst_gpuid, dst_index)
  if (torch.type(src) == 'table') then
    if torch.type(dst) ~= 'table' or #src ~= #dst then
      dst = {}
    end

    -- Recurse on the table
    for i = 1, #src do
      dst[i] = self:_distributeTensorRecursive(src[i], dst[i], src_gpuid, 
        src_index, dst_gpuid, dst_index)
    end

  elseif torch.type(src):find('torch%..+Tensor') then  
    if (dst == nil or torch.type(dst) ~= 'torch.CudaTensor') then
      -- Allocate only on startup or when input table structure changes. 
      -- Otherwise we will just resize the tensor below.
      setDevice(dst_gpuid)
      dst = torch.CudaTensor()
    end

    -- Split the tensor
    assert(torch.typename(src) == 'torch.CudaTensor')
    local slice = src[{ self:_calculateSliceRange(src, dst_index) }]

    if not dst:isSameSizeAs(slice) then
      setDevice(dst_gpuid)
      dst:resizeAs(slice)
    end

    asyncGPUCopy(dst, slice)  -- dst, src
  else
    error('input must be a nested table of tensors!')
  end

  return dst  
end

-- _ConcatTensorRecursive - if the src is a tensor then the function copies it
-- into the dst slice along self.dimension. 
-- Otherwise it does a recursive call on tables.
function DataParallelTable:_ConcatTensorRecursive(src, dst, src_gpuid, 
  src_index, dst_gpuid, dst_index)
  if (torch.type(src) == 'table') then
    if torch.type(dst) ~= 'table' or #src ~= #dst then
      dst = {}
    end

    -- Recurse on the table
    for i = 1, #src do
      dst[i] = self:_ConcatTensorRecursive(src[i], dst[i], src_gpuid, 
        src_index, dst_gpuid, dst_index)
    end

  elseif torch.type(src):find('torch%..+Tensor') then  
    if (dst == nil or torch.type(dst) ~= 'torch.CudaTensor') then
      -- Allocate only on startup or when input table structure changes. 
      -- Otherwise we will just resize the tensor below.
      setDevice(dst_gpuid)
      dst = torch.CudaTensor()
    end
    
    if (torch.numel(src) > 0) then
      -- Some modules return empty gradInputs if they don't actually return 
      -- anything.
      local dst_size = src:size():totable()
      dst_size[self.dimension] = dst_size[self.dimension] * #self.modules
      if not (equalSize(dst:size():totable(), dst_size)) then
        setDevice(dst_gpuid)
        dst:resize(unpack(dst_size))
      end

      -- Split the tensor
      assert(torch.typename(src) == 'torch.CudaTensor')
      local slice = dst[{ self:_calculateSliceRange(dst, src_index) }]

      asyncGPUCopy(slice, src)  -- dst, src
    end
  else
    error('input must be a nested table of tensors!')
  end

  return dst  
end
