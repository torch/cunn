#include "utils.h"
#include "THCApply.cuh"

struct hardtanhupdateOutput_functor
{
  __device__ void operator()(float* output, const float* input) const
  {
    if(*input < -1)
      *output = -1;
    else if(*input <= 1)
      *output = *input;
    else
      *output = 1;
  }
};

static int cunn_HardTanh_updateOutput(lua_State *L)
{
  THCState *state = getCutorchState(L);
  THCudaTensor *input = (THCudaTensor*)luaT_checkudata(L, 2, "torch.CudaTensor");
  THCudaTensor *output = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "output", "torch.CudaTensor");
  THAssert(THCudaTensor_checkGPU(state, 2, input, output));
  THCudaTensor_resizeAs(state, output, input);
  THCudaTensor_pointwiseApply2(state, output, input, hardtanhupdateOutput_functor());
  return 1;
}

struct hardtanhupdateGradInput_functor
{
  __device__ void operator()(float* gradInput, const float* input, const float* gradOutput) const
  {
    if(*input < -1 || *input > 1)
      *gradInput = 0;
    else
      *gradInput = *gradOutput;
  }
};

static int cunn_HardTanh_updateGradInput(lua_State *L)
{
  THCState *state = getCutorchState(L);
  THCudaTensor *input = (THCudaTensor*)luaT_checkudata(L, 2, "torch.CudaTensor");
  THCudaTensor *gradOutput = (THCudaTensor*)luaT_checkudata(L, 3, "torch.CudaTensor");
  THCudaTensor *gradInput = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "gradInput", "torch.CudaTensor");
  THAssert(THCudaTensor_checkGPU(state, 3, input, gradOutput, gradInput));

  THCudaTensor_resizeAs(state, gradInput, input);
  THCudaTensor_pointwiseApply3(state, gradInput, input, gradOutput, hardtanhupdateGradInput_functor());
  return 1;
}

static const struct luaL_Reg cunn_HardTanh__ [] = {
  {"HardTanh_updateOutput", cunn_HardTanh_updateOutput},
  {"HardTanh_updateGradInput", cunn_HardTanh_updateGradInput},
  {NULL, NULL}
};

static void cunn_HardTanh_init(lua_State *L)
{
  luaT_pushmetatable(L, "torch.CudaTensor");
  luaT_registeratname(L, cunn_HardTanh__, "nn");
  lua_pop(L,1);
}
