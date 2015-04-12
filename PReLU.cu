#include "THCApply.cuh"
#include "utils.h"


static int cunn_PReLU_updateOutput(lua_State *L)
{
  THCState *state = getCutorchState(L);
  // Input:
  THCudaTensor *input = (THCudaTensor*)luaT_checkudata(L, 2, "torch.CudaTensor");



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