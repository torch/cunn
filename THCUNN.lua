local ffi = require 'ffi'
local THNN = require 'nn.THNN'

local THCUNN = {}

local THCState_ptr = ffi.typeof('THCState*')

function THCUNN.getState()
  return THCState_ptr(cutorch.getState());
end

local THCUNN_h = [[
typedef void THCState;

TH_API void THNN_CudaAbs_updateOutput(
          THCState *state,
          THCudaTensor *input,
          THCudaTensor *output);
TH_API void THNN_CudaAbs_updateGradInput(
          THCState *state,
          THCudaTensor *input,
          THCudaTensor *gradOutput,
          THCudaTensor *gradInput);

TH_API void THNN_CudaAbsCriterion_updateOutput(
          THCState *state,
          THCudaTensor *input,
          THCudaTensor *target,
          float *output,
          bool sizeAverage);
TH_API void THNN_CudaAbsCriterion_updateGradInput(
          THCState *state,
          THCudaTensor *input,
          THCudaTensor *target,
          THCudaTensor *gradInput,
          bool sizeAverage);
]]


local preprocessed = string.gsub(THCUNN_h, 'TH_API ', '')
ffi.cdef(preprocessed)

local ok,result
if ffi.os == "OSX" then
  ok,result = pcall(ffi.load, 'libTHCUNN.dylib')
else
  ok,result = pcall(ffi.load, 'THCUNN')
end
if not ok then
  print(result)
  error("Ops, could not load 'THCUNN' GPU backend library.")
else
  THCUNN.C = result
end

local function extract_function_names(s)
  local t = {}
  for n in string.gmatch(s, 'TH_API void THNN_Cuda([%a%d_]+)') do
    t[#t+1] = n
  end
  return t
end

-- build function table
local function_names = extract_function_names(THCUNN_h)

THNN.kernels['torch.CudaTensor'] = THNN.bind(THCUNN.C, function_names, 'Cuda', THCUNN.getState)
torch.getmetatable('torch.CudaTensor').THNN = THNN.kernels['torch.CudaTensor']

return THCUNN
