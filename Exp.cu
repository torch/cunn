#include "utils.h"

struct expupdateOutput_functor
{
  __device__ void operator()(float* output, const float* input) const
  {
    *output = exp(*input);
  }
};

static int cunn_Exp_updateOutput(lua_State *L)
{
  THCState *state = getCutorchState(L);
  THCudaTensor *input = (THCudaTensor*)luaT_checkudata(L, 2, "torch.CudaTensor");
  THCudaTensor *output = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "output", "torch.CudaTensor");
  THAssert(THCudaTensor_checkGPU(state, 2, input, output));
  THCudaTensor_resizeAs(state, output, input);
  THCudaTensor_pointwiseApply2(state, output, input, expupdateOutput_functor());
  return 1;
}

struct expupdateGradInput_functor
{
  __device__ void operator()(float* gradInput, const float* output, const float* gradOutput) const
  {
    *gradInput = *gradOutput * *output;
  }
};

static int cunn_Exp_updateGradInput(lua_State *L)
{
  THCState *state = getCutorchState(L);
  THCudaTensor *output = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "output", "torch.CudaTensor");
  THCudaTensor *gradOutput = (THCudaTensor*)luaT_checkudata(L, 3, "torch.CudaTensor");
  THCudaTensor *gradInput = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "gradInput", "torch.CudaTensor");
  THAssert(THCudaTensor_checkGPU(state, 3, output, gradOutput, gradInput));
  THCudaTensor_resizeAs(state, gradInput, output);
  THCudaTensor_pointwiseApply3(state, gradInput, output, gradOutput, expupdateGradInput_functor());
  return 1;
}

static const struct luaL_Reg cunn_Exp__ [] = {
  {"Exp_updateOutput", cunn_Exp_updateOutput},
  {"Exp_updateGradInput", cunn_Exp_updateGradInput},
  {NULL, NULL}
};

static void cunn_Exp_init(lua_State *L)
{
  luaT_pushmetatable(L, "torch.CudaTensor");
  luaT_registeratname(L, cunn_Exp__, "nn");
  lua_pop(L,1);
}
