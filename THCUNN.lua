local ffi = require 'ffi'
local THNN = require 'nn.THNN'

local THCUNN = {}

-- load libTHCUNN
THCUNN.C = ffi.load(package.searchpath('libTHCUNN', package.cpath))

-- load THC
local THC = ffi.os == 'Windows' and ffi.load('THC') or ffi.C

local THCState_ptr = ffi.typeof('THCState*')

function THCUNN.getState()
   return THCState_ptr(cutorch.getState());
end

local THCUNN_generic_h = require 'cunn.THCUNN_generic_h'
-- strip all lines starting with #
-- to remove preprocessor directives originally present
-- in THNN.h
THCUNN_generic_h = THCUNN_generic_h:gsub("\n#[^\n]*", "")
THCUNN_generic_h = THCUNN_generic_h:gsub("^#[^\n]*\n", "")

local preprocessed_generic = string.gsub(THCUNN_generic_h, 'TH_API void THNN_%(([%a%d_]+)%)', 'void THNN_TYPE%1')

local replacements =
{
   {
      ['THTensor'] = 'THCudaTensor',
      ['THCIndexTensor'] = 'THCudaLongTensor',
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
    ['THCIndexTensor'] = 'THCudaLongTensor',
    ['TYPE'] = 'Cuda',
    ['accreal'] = 'float',
  },
  {
    ['THCTensor'] = 'THCudaDoubleTensor',
    ['THCIndexTensor'] = 'THCudaLongTensor',
    ['TYPE'] = 'CudaDouble',
    ['accreal'] = 'double',
   }
}

if cutorch.hasHalf then
  ffi.cdef("half THC_float2half(float a);")
  ffi.cdef("float THC_half2float(half a);")
  cct2lt['THCudaHalfTensor'] = 'torch.CudaHalfTensor'
  local half_replacement = {
    ['THCTensor'] = 'THCudaHalfTensor',
    ['THCIndexTensor'] = 'THCudaLongTensor',
    ['TYPE'] = 'CudaHalf',
    ['accreal'] = 'float',
  }
  table.insert(replacements_generic, half_replacement)
end

for i=1,#replacements_generic do
    local r = replacements_generic[i]
    local s = preprocessed_generic
    for k,v in pairs(r) do
        s = string.gsub(s, k, v)
    end
    ffi.cdef(s)
end

local function extract_function_names_generic(s)
   local t = {}
   for n in string.gmatch(s, 'TH_API void THNN_%(([%a%d_]+)%)') do
       t[#t+1] = n
   end
   return t
end

local function find_positions(s, p)
   local begin = 0
   local positions = {}
   while true do
      local start, stop = string.find(s, p, begin)
      if (start == nil) then break end
      positions[#positions+1] = start
      begin = stop + 1
   end
   return positions
end

local function extract_function_names_and_real_args(s)
   local t = {}
   for n in string.gmatch(s, 'TH_API ([^;]+)') do
      local func_name = string.match(n, 'void THNN_%(([%a%d_]+)%)')
      local param_positions = find_positions(n, ',')
      local positions = {}
      for x,y in ipairs(find_positions(n, 'real')) do
          local found = false
          for cn,cp in ipairs(param_positions) do
              if cp > y then
                positions[#positions+1] = cn
                found = true
                break
              end
          end
          -- it is the last param
          if not found then positions[#positions+1] = #param_positions + 1 end
      end

   t[func_name] = positions
   end
   return t
end

local real_args = extract_function_names_and_real_args(THCUNN_generic_h)

-- build function table
local function_names_generic = extract_function_names_generic(THCUNN_generic_h)

THNN.kernels['torch.CudaTensor'] = THNN.bind(THCUNN.C, function_names_generic, 'Cuda', THCUNN.getState)
torch.getmetatable('torch.CudaTensor').THNN = THNN.kernels['torch.CudaTensor']

THNN.kernels['torch.CudaDoubleTensor'] = THNN.bind(THCUNN.C, function_names_generic, 'CudaDouble', THCUNN.getState)
torch.getmetatable('torch.CudaDoubleTensor').THNN = THNN.kernels['torch.CudaDoubleTensor']

if cutorch.hasHalf then
   local raw_half_functions = THNN.bind(THCUNN.C, function_names_generic, 'CudaHalf', THCUNN.getState)
   THNN.kernels['torch.CudaHalfTensor'] = raw_half_functions
   torch.getmetatable('torch.CudaHalfTensor').THNN = THNN.kernels['torch.CudaHalfTensor']
end

local function Module__converter(type)
    return function(self)
            return self:type(type)
    end
end

rawset(torch.getmetatable('nn.Module'), 'cudaDouble', Module__converter('torch.CudaDoubleTensor'))
if cutorch.hasHalf then
    rawset(torch.getmetatable('nn.Module'), 'cudaHalf', Module__converter('torch.CudaHalfTensor'))
end
return THCUNN
