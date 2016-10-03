local ffi = require 'ffi'
local THNN = require 'nn.THNN'

local THCUNN = {}

-- load libTHCUNN
THCUNN.C = ffi.load(package.searchpath('libTHCUNN', package.cpath))

local THCState_ptr = ffi.typeof('THCState*')

function THCUNN.getState()
   return THCState_ptr(cutorch.getState());
end

local THCUNN_h = require 'cunn.THCUNN_h'
-- strip all lines starting with #
-- to remove preprocessor directives originally present
-- in THNN.h
THCUNN_h = THCUNN_h:gsub("\n#[^\n]*", "")
THCUNN_h = THCUNN_h:gsub("^#[^\n]*\n", "")

local THCUNN_generic_h = require 'cunn.THCUNN_generic_h'
-- strip all lines starting with #
-- to remove preprocessor directives originally present
-- in THNN.h
THCUNN_generic_h = THCUNN_generic_h:gsub("\n#[^\n]*", "")
THCUNN_generic_h = THCUNN_generic_h:gsub("^#[^\n]*\n", "")

ffi.cdef("half THC_float2half(float a);")

local preprocessed = string.gsub(THCUNN_h, 'TH_API ', '')
local preprocessed_generic = string.gsub(THCUNN_generic_h, 'TH_API void THNN_%(([%a%d_]+)%)', 'void THNN_TYPE%1')

local replacements =
{
   {
      ['THTensor'] = 'THCudaTensor',
      ['THIndexTensor'] = 'THCudaLongTensor',
      ['THIndex_t'] = 'long',
      ['THInteger_t'] = 'float'
   }
}

local cct2lt = {
   ['THCudaFloatTensor'] = 'torch.CudaTensor',
   ['THCudaDoubleTensor'] = 'torch.CudaDoubleTensor',
}

local replacements_generic =
{
  {
    ['THCTensor'] = 'THCudaTensor',
    ['THIndexTensor'] = 'THCudaLongTensor',
    ['TYPE'] = 'Cuda',
    ['real'] = 'float'
  },
  {
    ['THCTensor'] = 'THCudaDoubleTensor',
    ['THIndexTensor'] = 'THCudaLongTensor',
    ['TYPE'] = 'CudaDouble',
    ['real'] = 'double',
   }
}

if cutorch.hasHalf then
  cct2lt['THCudaHalfTensor'] = 'torch.CudaHalfTensor'
  local half_replacement = {
    ['THCTensor'] = 'THCudaHalfTensor',
    ['THIndexTensor'] = 'THCudaLongTensor',
    ['TYPE'] = 'CudaHalf',
    ['real'] = 'half'
  }
  table.insert(replacements_generic, half_replacement)
end

for i=1,#replacements do
   local r = replacements[i]
   local s = preprocessed
   for k,v in pairs(r) do
      s = string.gsub(s, k, v)
   end
   ffi.cdef(s)
end

for i=1,#replacements_generic do
    local r = replacements_generic[i]
    local s = preprocessed_generic
    for k,v in pairs(r) do
        s = string.gsub(s, k, v)
    end
    ffi.cdef(s)
end

local function extract_function_names(s)
   local t = {}
   for n in string.gmatch(s, 'TH_API void THNN_Cuda([%a%d_]+)') do
      t[#t+1] = n
   end
   return t
end

local function extract_function_names_generic(s)
   local t = {}
   for n in string.gmatch(s, 'TH_API void THNN_%(([%a%d_]+)%)') do
       t[#t+1] = n
   end
   return t
end

-- build function table
local function_names = extract_function_names(THCUNN_h)
local function_names_generic = extract_function_names_generic(THCUNN_generic_h)

-- combine function names for CudaTensor
for k,v in pairs(function_names_generic) do
  function_names[#function_names+1] = v
end

THNN.kernels['torch.CudaTensor'] = THNN.bind(THCUNN.C, function_names, 'Cuda', THCUNN.getState)
torch.getmetatable('torch.CudaTensor').THNN = THNN.kernels['torch.CudaTensor']

THNN.kernels['torch.CudaDoubleTensor'] = THNN.bind(THCUNN.C, function_names_generic, 'CudaDouble', THCUNN.getState)
torch.getmetatable('torch.CudaDoubleTensor').THNN = THNN.kernels['torch.CudaDoubleTensor']

-- in order to call 'half' functions from lua, convert number arguments from
-- float to half
local transform_args_to_half = function(t)
    for k,v in pairs(t) do
        if torch.type(v) == 'number' then
          t[k] = ffi.C.THC_float2half(t[k])
        end
    end
    return t
end

local raw_half_functions = THNN.bind(THCUNN.C, function_names_generic, 'CudaHalf', THCUNN.getState)
for k,v in pairs(raw_half_functions) do
    raw_half_functions[k] = function(...) v(unpack(transform_args_to_half({...}))) end
end
THNN.kernels['torch.CudaHalfTensor'] = raw_half_functions
torch.getmetatable('torch.CudaHalfTensor').THNN = THNN.kernels['torch.CudaHalfTensor']

return THCUNN
