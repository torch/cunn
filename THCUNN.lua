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

local preprocessed = string.gsub(THCUNN_h, 'TH_API ', '')

local replacements =
{
   {
      ['THTensor'] = 'THCudaTensor',
      ['THIndexTensor'] = 'THCudaTensor',
      ['THIntegerTensor'] = 'THCudaTensor',
      ['THIndex_t'] = 'float',
      ['THInteger_t'] = 'float'
   }
}

for i=1,#replacements do
   local r = replacements[i]
   local s = preprocessed
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

-- build function table
local function_names = extract_function_names(THCUNN_h)

THNN.kernels['torch.CudaTensor'] = THNN.bind(THCUNN.C, function_names, 'Cuda', THCUNN.getState)
torch.getmetatable('torch.CudaTensor').THNN = THNN.kernels['torch.CudaTensor']

return THCUNN
