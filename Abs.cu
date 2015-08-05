#include "THCApply.cuh"
#include "utils.h"

struct AbsUpdateOutput {
  __device__ __forceinline__ void operator()(float* out, float* in) {
    *out = fabsf(*in);
  }
};

static int cunn_Abs_updateOutput(lua_State *L) {
  THCState *state = getCutorchState(L);
  THCudaTensor *input = (THCudaTensor*)luaT_checkudata(L, 2, "torch.CudaTensor");
  THCudaTensor *output = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "output", "torch.CudaTensor");

  THCudaTensor_resizeAs(state, output, input);
  THCudaTensor_pointwiseApply2(state, output, input,
                               AbsUpdateOutput());

  THCudaCheck(cudaGetLastError());
  return 1;
}

struct AbsUpdateGradInput {
  __device__ __forceinline__ void operator()(float* gradInput, float* input,
                                             float* gradOutput) const {
    *gradInput = (*input < 0.0f) ? -(*gradOutput) : *gradOutput;
  }
};

static int cunn_Abs_updateGradInput(lua_State *L) {
  THCState *state = getCutorchState(L);
  THCudaTensor *input = (THCudaTensor*)luaT_checkudata(L, 2, "torch.CudaTensor");
  THCudaTensor *gradOutput = (THCudaTensor*)luaT_checkudata(L, 3, "torch.CudaTensor");
  THCudaTensor *gradInput = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "gradInput", "torch.CudaTensor");
  THCudaTensor_resizeAs(state, gradInput, input);
  THCudaTensor_pointwiseApply3(state, gradInput, input, gradOutput,
                               AbsUpdateGradInput());

  THCudaCheck(cudaGetLastError());
  return 1;
}


static const struct luaL_Reg cunn_Abs__ [] = {
  {"Abs_updateOutput", cunn_Abs_updateOutput},
  {"Abs_updateGradInput", cunn_Abs_updateGradInput},
  {NULL, NULL}
};

void cunn_Abs_init(lua_State *L) {
  luaT_pushmetatable(L, "torch.CudaTensor");
  luaT_registeratname(L, cunn_Abs__, "nn");
  lua_pop(L,1);
}
