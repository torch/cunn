#include "THCApply.cuh"
#include "utils.h"
#include "THCApply.cuh"

struct SigmoidUpdateOutput {
  __device__ __forceinline__ void operator()(float* out, float* in) {
    *out = 1.0f / (1.0f + expf(-(*in)));
  }
};

// in-place variant
struct SigmoidUpdateOutputIP {
  __device__ __forceinline__ void operator()(float* x) {
    *x = 1.0f / (1.0f + expf(-(*x)));
  }
};

static int cunn_Sigmoid_updateOutput(lua_State *L) {
  THCState *state = getCutorchState(L);
  THCudaTensor *input = (THCudaTensor*)luaT_checkudata(L, 2, "torch.CudaTensor");
  bool inPlace = luaT_getfieldcheckboolean(L, 1, "inplace");
  THCudaTensor *output = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "output", "torch.CudaTensor");

  if (inPlace) {
    THCudaTensor_pointwiseApply1(state, input,
                                 SigmoidUpdateOutputIP());
    THCudaTensor_set(state, output, input);
  } else {
    THCudaTensor_resizeAs(state, output, input);
    THCudaTensor_pointwiseApply2(state, output, input,
                                 SigmoidUpdateOutput());
  }

  THCudaCheck(cudaGetLastError());
  return 1;
}


struct SigmoidUpdateGradInput {
  __device__ __forceinline__ void operator()(float* gradInput, float* output,
                                             float* gradOutput) const {
    *gradInput = *gradOutput * (1.0f - *output) * *output;
  }
};

struct SigmoidUpdateGradInputIP {
  __device__ __forceinline__ void operator()(float* gradOutput,
                                             float* output) const {
    *gradOutput = *gradOutput * (1.0f - *output) * *output;
  }
};

static int cunn_Sigmoid_updateGradInput(lua_State *L) {
  THCState *state = getCutorchState(L);
  THCudaTensor *output = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "output", "torch.CudaTensor");
  THCudaTensor *gradOutput = (THCudaTensor*)luaT_checkudata(L, 3, "torch.CudaTensor");
  bool inPlace = luaT_getfieldcheckboolean(L, 1, "inplace");
  THCudaTensor *gradInput = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "gradInput", "torch.CudaTensor");

  if (inPlace) {
    THCudaTensor_pointwiseApply2(state, gradOutput, output,
                                 SigmoidUpdateGradInputIP());
    THCudaTensor_set(state, gradInput, gradOutput);
  } else {
    THCudaTensor_resizeAs(state, gradInput, output);
    THCudaTensor_pointwiseApply3(state, gradInput, output, gradOutput,
                                 SigmoidUpdateGradInput());
  }

  THCudaCheck(cudaGetLastError());
  return 1;
}


static const struct luaL_Reg cunn_Sigmoid__ [] = {
  {"Sigmoid_updateOutput", cunn_Sigmoid_updateOutput},
  {"Sigmoid_updateGradInput", cunn_Sigmoid_updateGradInput},
  {NULL, NULL}
};

void cunn_Sigmoid_init(lua_State *L) {
  luaT_pushmetatable(L, "torch.CudaTensor");
  luaT_registeratname(L, cunn_Sigmoid__, "nn");
  lua_pop(L,1);
}
