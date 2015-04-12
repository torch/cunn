#include "THCApply.cuh"
#include "utils.h"


struct PReLUUpdateOutput {
  const float weight_;

  PReLUUpdateOutput(float weight): weight_(weight) {}

  __device__ __forceinline__ void operator()(float* out, float* in) {
    float x = *in;
    *out = (x > 0) ? x : weight_ * x;
  }
};

static int cunn_PReLU_updateOutput(lua_State *L)
{
  THCState *state = getCutorchState(L);
  // Input:
  THCudaTensor *input = (THCudaTensor*) luaT_checkudata(L, 2, "torch.CudaTensor");
  // Params:
  THCudaTensor *weight = (THCudaTensor*) luaT_getfieldcheckudata(L, 1, "weight", "torch.CudaTensor");
  THCudaTensor *output = (THCudaTensor*) luaT_getfieldcheckudata(L, 1, "output", "torch.CudaTensor");
  long nOutputPlane = luaT_getfieldchecknumber(L, 1, "nOutputPlane");

  THCudaTensor_resizeAs(state, output, input);

  if (nOutputPlane == 0)
  {
    // handle shared parameter case
    float w = THCudaTensor_get1d(state, weight, 0);
    THCudaTensor_pointwiseApply2(state, output, input, PReLUUpdateOutput(w));
  } else {
    printf("PReLU cuda version only supports shared parameter case\n");
    THError("aborting");
  }

  THCudaCheck(cudaGetLastError());
  return 1;
}

static int cunn_PReLU_updateGradInput(lua_State *L)
{
  THCState *state = getCutorchState(L);
  // Input:
  THCudaTensor *input = (THCudaTensor*)luaT_checkudata(L, 2, "torch.CudaTensor");
  THCudaTensor *gradOutput = (THCudaTensor*)luaT_checkudata(L, 3, "torch.CudaTensor");



  THCudaCheck(cudaGetLastError());
  return 1;
}

static int cunn_PReLU_accGradParameters(lua_State *L)
{
  THCState *state = getCutorchState(L);
  // Input:
  THCudaTensor *input = (THCudaTensor*)luaT_checkudata(L, 2, "torch.CudaTensor");
  THCudaTensor *gradOutput = (THCudaTensor*)luaT_checkudata(L, 3, "torch.CudaTensor");



  THCudaCheck(cudaGetLastError());
  return 1;
}

static const struct luaL_Reg cunn_PRelu__ [] = {
  {"PReLU_updateOutput", cunn_PReLU_updateOutput},
  {"PReLU_updateGradInput", cunn_PReLU_updateGradInput},
  {"PReLU_accGradParameters", cunn_PReLU_accGradParameters},
  {NULL, NULL}
};

static void cunn_PReLU_init(lua_State *L)
{
  luaT_pushmetatable(L, "torch.CudaTensor");
  luaT_registeratname(L, cunn_PRelu__, "nn");
  lua_pop(L,1);
}