#include "THCApply.cuh"
#include "utils.h"

struct SquareUpdateOutput {
  __device__ __forceinline__ void operator()(float* out, float* in) {
    *out = *in * *in;
  }
};

static int cunn_Square_updateOutput(lua_State *L) {
  THCState *state = getCutorchState(L);
  THCudaTensor *input = (THCudaTensor*)luaT_checkudata(L, 2, "torch.CudaTensor");
  THCudaTensor *output = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "output", "torch.CudaTensor");

  THCudaTensor_resizeAs(state, output, input);
  THCudaTensor_pointwiseApply2(state, output, input,
                               SquareUpdateOutput());

  THCudaCheck(cudaGetLastError());
  return 1;
}

struct SquareUpdateGradInput {
  __device__ __forceinline__ void operator()(float* gradInput, float* input,
                                             float* gradOutput) const {
    *gradInput = 2.0f * *gradOutput * *input;
  }
};

static int cunn_Square_updateGradInput(lua_State *L) {
  THCState *state = getCutorchState(L);
  THCudaTensor *input = (THCudaTensor*)luaT_checkudata(L, 2, "torch.CudaTensor");
  THCudaTensor *gradOutput = (THCudaTensor*)luaT_checkudata(L, 3, "torch.CudaTensor");
  THCudaTensor *gradInput = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "gradInput", "torch.CudaTensor");
  THCudaTensor_resizeAs(state, gradInput, input);
  THCudaTensor_pointwiseApply3(state, gradInput, input, gradOutput,
                               SquareUpdateGradInput());

  THCudaCheck(cudaGetLastError());
  return 1;
}


static const struct luaL_Reg cunn_Square__ [] = {
  {"Square_updateOutput", cunn_Square_updateOutput},
  {"Square_updateGradInput", cunn_Square_updateGradInput},
  {NULL, NULL}
};

void cunn_Square_init(lua_State *L) {
  luaT_pushmetatable(L, "torch.CudaTensor");
  luaT_registeratname(L, cunn_Square__, "nn");
  lua_pop(L,1);
}
