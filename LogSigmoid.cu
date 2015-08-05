#include "THCApply.cuh"
#include "utils.h"
#include "THCApply.cuh"

struct LogSigmoidUpdateOutput {
  __device__ __forceinline__ void operator()(float* out, float* in) {
    float z = expf(-(*in));
    *out = -logf(1.0f + z);
  }
};

static int cunn_LogSigmoid_updateOutput(lua_State *L) {
  THCState *state = getCutorchState(L);
  THCudaTensor *input = (THCudaTensor*)luaT_checkudata(L, 2, "torch.CudaTensor");
  THCudaTensor *output = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "output", "torch.CudaTensor");

  THCudaTensor_resizeAs(state, output, input);
  THCudaTensor_pointwiseApply2(state, output, input,
                               LogSigmoidUpdateOutput());

  THCudaCheck(cudaGetLastError());
  return 1;
}

struct LogSigmoidUpdateGradInput {
  __device__ __forceinline__ void operator()(float* gradInput, float* input,
                                             float* gradOutput) const {
    float z = expf(-(*input));
    *gradInput = *gradOutput * z / (1.0f + z);
  }
};

static int cunn_LogSigmoid_updateGradInput(lua_State *L) {
  THCState *state = getCutorchState(L);
  THCudaTensor *input = (THCudaTensor*)luaT_checkudata(L, 2, "torch.CudaTensor");
  THCudaTensor *gradOutput = (THCudaTensor*)luaT_checkudata(L, 3, "torch.CudaTensor");
  THCudaTensor *gradInput = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "gradInput", "torch.CudaTensor");
  THCudaTensor_resizeAs(state, gradInput, input);
  THCudaTensor_pointwiseApply3(state, gradInput, input, gradOutput,
                               LogSigmoidUpdateGradInput());

  THCudaCheck(cudaGetLastError());
  return 1;
}


static const struct luaL_Reg cunn_LogSigmoid__ [] = {
  {"LogSigmoid_updateOutput", cunn_LogSigmoid_updateOutput},
  {"LogSigmoid_updateGradInput", cunn_LogSigmoid_updateGradInput},
  {NULL, NULL}
};

void cunn_LogSigmoid_init(lua_State *L) {
  luaT_pushmetatable(L, "torch.CudaTensor");
  luaT_registeratname(L, cunn_LogSigmoid__, "nn");
  lua_pop(L,1);
}
