#include "utils.h"

struct sqrtupdateOutput_functor
{
  const double bias;

  sqrtupdateOutput_functor(double bias_) : bias(bias_) {}

  __device__ void operator()(float* output, const float* input) const
  {
    *output = sqrt(*input + bias);
  }
};

static int cunn_Sqrt_updateOutput(lua_State *L)
{
  THCState *state = getCutorchState(L);
  double bias = luaT_getfieldchecknumber(L,1,"eps");
  THCudaTensor *input = (THCudaTensor*)luaT_checkudata(L, 2, "torch.CudaTensor");
  THCudaTensor *output = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "output", "torch.CudaTensor");

  THCudaTensor_resizeAs(state, output, input);
  THCudaTensor_pointwiseApply2(state, output, input, sqrtupdateOutput_functor(bias));
  return 1;
}

struct sqrtupdateGradInput_functor
{
  const double bias;

  sqrtupdateGradInput_functor(double bias_) : bias(bias_) {}

  __device__ void operator()(float* gradInput, const float* output, const float* gradOutput) const
  {
    *gradInput = 0.5 * *gradOutput / *output;
  }
};

static int cunn_Sqrt_updateGradInput(lua_State *L)
{
  THCState *state = getCutorchState(L);
  double bias = luaT_getfieldchecknumber(L,1,"eps");
  THCudaTensor *output = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "output", "torch.CudaTensor");
  THCudaTensor *gradOutput = (THCudaTensor*)luaT_checkudata(L, 3, "torch.CudaTensor");
  THCudaTensor *gradInput = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "gradInput", "torch.CudaTensor");

  THCudaTensor_resizeAs(state, gradInput, output);
  THCudaTensor_pointwiseApply3(state, gradInput, output, gradOutput, sqrtupdateGradInput_functor(bias));
  return 1;
}

static const struct luaL_Reg cunn_Sqrt__ [] = {
  {"Sqrt_updateOutput", cunn_Sqrt_updateOutput},
  {"Sqrt_updateGradInput", cunn_Sqrt_updateGradInput},
  {NULL, NULL}
};

static void cunn_Sqrt_init(lua_State *L)
{
  luaT_pushmetatable(L, "torch.CudaTensor");
  luaT_registeratname(L, cunn_Sqrt__, "nn");
  lua_pop(L,1);
}
