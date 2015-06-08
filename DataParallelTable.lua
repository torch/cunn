local gpuLocalCopyBuffers = {}
local baseGpuIndex = 1  -- A constant

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
            local curId = param:getDevice()
            if deviceId == nil then
               deviceId = curId
            else
               assert(deviceId == curId,
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
   if not gpuLocalCopyBuffers[device] then
      gpuLocalCopyBuffers[device] = torch.CudaTensor()
   end
   return gpuLocalCopyBuffers[device]
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
   local prevDevice = cutorch.getDevice()

   local destGpuid = dest:getDevice()
   local sourceGpuid = source:getDevice()

   if sourceGpuid == destGpuid then
      -- if both tensors are on the same gpu normal operation works
      setDevice(destGpuid)
      dest:copy(source)
      setDevice(prevDevice)
      return
   end

   if source:isContiguous() and dest:isContiguous() then
      -- if both tensors are contiguous operation across gpus works
      setDevice(destGpuid)
      dest:copy(source)
      setDevice(prevDevice)
      return
   end

   -- Either the dest or the source are not contiguous.  we will need to do
   -- intermediate copies.
   local tmpSource = source
   if not source:isContiguous() then
      setDevice(sourceGpuid)
      tmpSource = getBuffer()
      tmpSource:resizeAs(source)
      tmpSource:copy(source)  -- Make contiguous using a copy
   end

   setDevice(destGpuid)
   local tmpDest = dest
   if not dest:isContiguous() then
      local tmpDest = getBuffer()
      tmpDest:resizeAs(tmpSource)
      tmpDest:copy(tmpSource)
   end

   dest:copy(tmpDest)

   cutorch.synchronize()  -- Ensures we keep the buffer for the copy duration

   -- Put the device back to what it was.
   setDevice(prevDevice)
end

local function equalSize(sizeTable1, sizeTable2)
   if (#sizeTable1 ~= #sizeTable2) then
      return false
   end
   for i = 1, #sizeTable1 do
      if sizeTable1[i] ~= sizeTable2[i] then return false end
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

      local dstGpuid = dst:getDevice()
      local srcGpuid = src:getDevice()
      local curGpuid = cutorch:getDevice()
      setDevice(dstGpuid)

      -- Copy src over to a buffer on the dst GPU
      local srcBufferOnDstGpu = src
      if (dstGpuid ~= srcGpuid) then
         srcBufferOnDstGpu = getBuffer()
         srcBufferOnDstGpu:resizeAs(src)
         assert(src:isContiguous())
         srcBufferOnDstGpu:copy(src)
      end

      -- Perform the actual add
      dst:add(srcBufferOnDstGpu)
      if (dstGpuid ~= srcGpuid) then
         -- Ensures we get to keep the buffer for the duration of the add
         cutorch.synchronize()
      end

      setDevice(curGpuid)  -- Put the GPU id back to what it was
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
   self.gpuAssignments = {}  -- Which gpuid each module sits on
   self.inputGpu = {}  -- inputs for each gpu
   self.gradOutputGpu = {} -- gradOutputs for each gpu
   self.outputGpu = {} -- outputs for each gpu
   self.gradInputGpu = {} -- gradInput for each gpu
end

-- NOTE: The input should be on the FIRST added GPU device, and similarly the
-- output will be on the FIRST GPU device.
function DataParallelTable:add(module, gpuid)
   assert(gpuid <= cutorch.getDeviceCount() and gpuid >= 1)
   assert(#self.modules == #self.gpuAssignments)

   self.modules[#self.modules + 1] = module
   self.gpuAssignments[#self.gpuAssignments + 1] = gpuid

   return self
end

function DataParallelTable:__tostring()
   return 'DataParallelTable: ' .. #self.modules .. ' x ' .. tostring(self.modules[1])
end

function DataParallelTable:get(index)
   return self.modules[index]
end

function DataParallelTable:updateOutput(input)
   local baseGpuid = self.gpuAssignments[baseGpuIndex]
   assert(queryGPUDeviceId(input) == baseGpuid, 'Input is not on gpu ' ..
   baseGpuid)

   local prevGpuid = cutorch.getDevice()

   -- distribute the input to GPUs
   for i = 1, #self.modules do
      local gpuid = self.gpuAssignments[i]
      -- Split the tensors in the input nested table to the GPU with gpuid
      -- _distributeTensorRecursive(src,dst,srcGpuid,srcInd,dstGpuid,dstInd)
      self.inputGpu[gpuid] = self:_distributeTensorRecursive(
         input, self.inputGpu[gpuid],
         baseGpuid, baseGpuIndex, gpuid, i,
         #self.modules
      )
   end

   cutorch.synchronize()

   -- update output for each module asynchronously
   for i, module in ipairs(self.modules) do
      local gpuid = self.gpuAssignments[i]
      setDevice(gpuid)
      self.outputGpu[gpuid] = module:updateOutput(self.inputGpu[gpuid])
   end

   cutorch.synchronize()

   -- concatenate the outputs to the base GPU
   for i = 1, #self.modules do
      local gpuid = self.gpuAssignments[i]
      -- Merge the tensors in the input nested table to the GPU with gpuid
      -- _concatTensorRecursive(src,dst,srcGpuid,srcInd,dstGpuid,dstInd)
      self.output = self:_concatTensorRecursive(
         self.outputGpu[gpuid], self.output,
         gpuid, i, baseGpuid, baseGpuIndex,
         #self.modules
      )
   end

   setDevice(prevGpuid)

   return self.output
end

function DataParallelTable:updateGradInput(input, gradOutput)
   -- We assume that updateOutput has already been called (therefore inputGpu
   -- has been populated)
   local baseGpuid = self.gpuAssignments[baseGpuIndex]
   assert(queryGPUDeviceId(gradOutput) == baseGpuid,
   'gradOutput is not on gpu ' .. baseGpuid)

   local prevGpuid = cutorch.getDevice()

   -- distribute the gradOutput to GPUs
   for i = 1, #self.modules do
      local gpuid = self.gpuAssignments[i]
      -- Split the tensors in the input nested table to the GPU with gpuid
      -- _distributeTensorRecursive(src,dst,srcGpuid,srcInd,dstGpuid,dstInd)
      self.gradOutputGpu[gpuid] = self:_distributeTensorRecursive(gradOutput,
         self.gradOutputGpu[gpuid], baseGpuid, baseGpuIndex, gpuid, i, #self.modules)
   end

   cutorch.synchronize()

   -- update gradInput for each module asynchronously
   for i, module in ipairs(self.modules) do
      local gpuid = self.gpuAssignments[i]
      setDevice(gpuid)
      self.gradInputGpu[gpuid] = module:updateGradInput(self.inputGpu[gpuid],
      self.gradOutputGpu[gpuid])
   end

   cutorch.synchronize()

   -- concatenate the outputs to the base GPU
   for i = 1, #self.modules do
      local gpuid = self.gpuAssignments[i]
      -- Merge the tensors in the input nested table to the GPU with gpuid
      -- _concatTensorRecursive(src,dst,srcGpuid,srcInd,dstGpuid,dstInd)
      self.gradInput = self:_concatTensorRecursive(self.gradInputGpu[gpuid],
         self.gradInput, gpuid, i, baseGpuid, baseGpuIndex, #self.modules)
   end

   cutorch.synchronize()

   setDevice(prevGpuid)

   return self.gradInput
end

function DataParallelTable:accGradParameters(input, gradOutput, scale)
   -- We assume updateGradInput has already been called (so gradOutput has
   -- already been populated)
   local prevGpuid = cutorch.getDevice()
   local baseGpuid = self.gpuAssignments[baseGpuIndex]

   scale = scale or 1
   -- Calculate the gradWeight + gradBias on each sub-module
   for i, module in ipairs(self.modules) do
      local gpuid = self.gpuAssignments[i]
      setDevice(gpuid)
      module:accGradParameters(self.inputGpu[gpuid], self.gradOutputGpu[gpuid],
      scale)
   end

   cutorch.synchronize()  -- We have to wait until accGradParameters has finished

   -- Accumulate the gradients onto one GPU (the first one)
   -- TODO: Parallelize this (ie a parallel merge)
   local baseParams, baseGradParams = self.modules[baseGpuIndex]:parameters()
   for i, module in ipairs(self.modules) do
      if (i ~= baseGpuIndex) then
         local params, gradParams = self.modules[i]:parameters()
         deepTensorsAdd(baseGradParams, gradParams)  -- dst, src
         cutorch.synchronize()
      end
   end

   setDevice(prevGpuid)
end

function DataParallelTable:syncParameters()
   local prevGpuid = cutorch.getDevice()
   local baseParams, baseGradParams = self.modules[baseGpuIndex]:parameters()
   -- TODO: Parallelize this (ie a parallel copy)
   for i, module in ipairs(self.modules) do
      if (i ~= baseGpuIndex) then
         local params, gradParams = self.modules[i]:parameters()
         deepTensorsCopy(params, baseParams)  -- dst, src
      end
   end
   cutorch.synchronize()

   setDevice(prevGpuid)
end

-- For compatability with nn.Optim from fbcunn
function DataParallelTable:MixGrads()
   self:syncParameters()
end

function DataParallelTable:accUpdateGradParameters(input, gradOutput, lr)
   error("accUpdateGradParameters not supported for DataParallelTable.")
end

function DataParallelTable:zeroGradParameters()
   local prevGpuid = cutorch.getDevice()
   for i, module in ipairs(self.modules) do
      setDevice(self.gpuAssignments[i])
      module:zeroGradParameters()
   end
   setDevice(prevGpuid)
end

function DataParallelTable:updateParameters(learningRate)
   error("updateParameters not supported for DataParallelTable.")
end

function DataParallelTable:parameters()
   local prevGpuid = cutorch.getDevice()
   setDevice(self.gpuAssignments[1])
   local ret = {self.modules[1]:parameters()}
   setDevice(prevGpuid)
   return unpack(ret)
end

function DataParallelTable:share(mlp,...)
   error("Share not supported for DataParallelTable.")
end

function DataParallelTable:clone()
   error("clone not supported for DataParallelTable.")
end

function DataParallelTable:reset(stdv)
   local prevGpuid = cutorch.getDevice()
   for i, module in ipairs(self.modules) do
      setDevice(self.gpuAssignments[i])
      module:reset(stdv)
   end
   setDevice(prevGpuid)
end

function DataParallelTable:name()
   return 'DataParallelTable'
end

function DataParallelTable:type(typeStr)
   error("type() not supported for DataParallelTable.")
end

function DataParallelTable:_calculateSliceRange(tensor, id, total)
   local outerDim = tensor:size(self.dimension)
   local eltsPerMod = torch.round( outerDim / #self.modules )
   local rangeStart = (id - 1) * eltsPerMod + 1
   local rangeEnd = rangeStart + eltsPerMod - 1
   if id == total then
      rangeEnd = outerDim
   end
   self.batchSize = outerDim -- TODO: this is a hack to propagate batchSize to line 494
                             --       but might not be generic enough
   return {rangeStart, rangeEnd}
end

-- _distributeTensorRecursive - if the src is a tensor then the function slices
-- it long self.dimension and copies each portion into each child module.
-- Otherwise it does a recursive call on tables.
function DataParallelTable:_distributeTensorRecursive(src, dst,
   srcGpuid, srcIndex, dstGpuid, dstIndex, nModules)
   if (torch.type(src) == 'table') then
      if torch.type(dst) ~= 'table' or #src ~= #dst then
         dst = {}
      end

      -- Recurse on the table
      for i = 1, #src do
         dst[i] = self:_distributeTensorRecursive(src[i], dst[i], srcGpuid,
         srcIndex, dstGpuid, dstIndex, nModules)
      end

   elseif torch.type(src):find('torch%..+Tensor') then
      if (dst == nil or torch.type(dst) ~= 'torch.CudaTensor') then
         -- Allocate only on startup or when input table structure changes.
         -- Otherwise we will just resize the tensor below.
         setDevice(dstGpuid)
         dst = torch.CudaTensor()
      end

      -- Split the tensor
      assert(torch.typename(src) == 'torch.CudaTensor')
      local slice = src[{self:_calculateSliceRange(src, dstIndex, nModules)}]

      if not dst:isSameSizeAs(slice) then
         setDevice(dstGpuid)
         dst:resizeAs(slice)
      end

      asyncGPUCopy(dst, slice)  -- dst, src
   else
      error('input must be a nested table of tensors!')
   end

   return dst
end

-- _concatTensorRecursive - if the src is a tensor then the function copies it
-- into the dst slice along self.dimension.
-- Otherwise it does a recursive call on tables.
function DataParallelTable:_concatTensorRecursive(src, dst, srcGpuid,
   srcIndex, dstGpuid, dstIndex, nModules)
   if (torch.type(src) == 'table') then
      if torch.type(dst) ~= 'table' or #src ~= #dst then
         dst = {}
      end

      -- Recurse on the table
      for i = 1, #src do
         dst[i] = self:_concatTensorRecursive(src[i], dst[i], srcGpuid,
            srcIndex, dstGpuid, dstIndex, nModules)
      end

   elseif torch.type(src):find('torch%..+Tensor') then
      if (dst == nil or torch.type(dst) ~= 'torch.CudaTensor') then
         -- Allocate only on startup or when input table structure changes.
         -- Otherwise we will just resize the tensor below.
         setDevice(dstGpuid)
         dst = torch.CudaTensor()
      end

      if (torch.numel(src) > 0) then
         -- Some modules return empty gradInputs if they don't actually return
         -- anything.
         local dstSize = src:size():totable()
         dstSize[self.dimension] = self.batchSize
         if not (equalSize(dst:size():totable(), dstSize)) then
            setDevice(dstGpuid)
            dst:resize(unpack(dstSize))
         end

         -- Split the tensor
         assert(torch.typename(src) == 'torch.CudaTensor')
         local slice = dst[{ self:_calculateSliceRange(dst, srcIndex, nModules) }]

         asyncGPUCopy(slice, src)  -- dst, src
      end
   else
      error('input must be a nested table of tensors!')
   end

   return dst
end
