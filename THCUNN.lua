local ffi = require 'ffi'
local THNN = require 'nn.THNN'

local THCUNN = {}

-- load libTHCUNN
THCUNN.C = ffi.load(package.searchpath('libTHCUNN', package.cpath))

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
    ['real'] = 'float'
  },
  {
    ['THCTensor'] = 'THCudaDoubleTensor',
    ['THCIndexTensor'] = 'THCudaLongTensor',
    ['TYPE'] = 'CudaDouble',
    ['real'] = 'double',
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
    ['real'] = 'half'
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

-- in order to call 'half' functions from lua, convert real arguments from
-- to half since there is no other defined conversion
local transform_reals_to_half = function(func_name, real_args, ...)
    t = {}
    -- this select logic is necessary to deal with nil arguments
    for i = 1, select('#', ...) do
        t[i] = select(i, ...)
    end
    for k,v in ipairs(real_args[func_name]) do
        -- first argument (THCState) is added implicitly by bind
        t[v-1] = ffi.C.THC_float2half(t[v-1])
    end
    return t
end

local raw_half_functions = THNN.bind(THCUNN.C, function_names_generic, 'CudaHalf', THCUNN.getState)
for k,v in pairs(raw_half_functions) do
    -- select required in case there are trailing nils
    raw_half_functions[k] = function(...) v(unpack(transform_reals_to_half(k, real_args, ...), 1, select("#",...)))
end
end
THNN.kernels['torch.CudaHalfTensor'] = raw_half_functions
torch.getmetatable('torch.CudaHalfTensor').THNN = THNN.kernels['torch.CudaHalfTensor']

return THCUNN
