#include "THCApply.cuh"
#include "THCApply.cuh"
#include "utils.h"

struct SqrtUpdateOutput {
  __device__ __forceinline__ void operator()(float* out, float* in) {
    *out = sqrtf(*in);
  }
};

// in-place variant
struct SqrtUpdateOutputIP {
  __device__ __forceinline__ void operator()(float* x) {
    *x = sqrtf(*x);
  }
};

static int cunn_Sqrt_updateOutput(lua_State *L) {
  THCState *state = getCutorchState(L);
  THCudaTensor *input = (THCudaTensor*)luaT_checkudata(L, 2, "torch.CudaTensor");
  bool inPlace = luaT_getfieldcheckboolean(L, 1, "inplace");
  THCudaTensor *output = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "output", "torch.CudaTensor");

  if (inPlace) {
    THCudaTensor_pointwiseApply1(state, input,
                                 SqrtUpdateOutputIP());
    THCudaTensor_set(state, output, input);
  } else {
    THCudaTensor_resizeAs(state, output, input);
    THCudaTensor_pointwiseApply2(state, output, input,
                                 SqrtUpdateOutput());
  }

  THCudaCheck(cudaGetLastError());
  return 1;
}

struct SqrtUpdateGradInput {
  __device__ __forceinline__ void operator()(float* gradInput, float* output,
                                             float* gradOutput) const {
    *gradInput = (*output == 0.0f) ? 0.0f : ((0.5f * *gradOutput) / *output);
  }
};

struct SqrtUpdateGradInputIP {
  __device__ __forceinline__ void operator()(float* gradOutput,
                                             float* output) const {
    *gradOutput = (*output == 0.0f) ? 0.0f : ((0.5f * *gradOutput) / *output);
  }
};

static int cunn_Sqrt_updateGradInput(lua_State *L) {
  THCState *state = getCutorchState(L);
  THCudaTensor *output = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "output", "torch.CudaTensor");
  THCudaTensor *gradOutput = (THCudaTensor*)luaT_checkudata(L, 3, "torch.CudaTensor");
  bool inPlace = luaT_getfieldcheckboolean(L, 1, "inplace");
  THCudaTensor *gradInput = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "gradInput", "torch.CudaTensor");

  if (inPlace) {
    THCudaTensor_pointwiseApply2(state, gradOutput, output,
                                 SqrtUpdateGradInputIP());
    THCudaTensor_set(state, gradInput, gradOutput);
  } else {
    THCudaTensor_resizeAs(state, gradInput, output);
    THCudaTensor_pointwiseApply3(state, gradInput, output, gradOutput,
                                 SqrtUpdateGradInput());
  }

  THCudaCheck(cudaGetLastError());
  return 1;
}

static const struct luaL_Reg cunn_Sqrt__ [] = {
  {"Sqrt_updateOutput", cunn_Sqrt_updateOutput},
  {"Sqrt_updateGradInput", cunn_Sqrt_updateGradInput},
  {NULL, NULL}
};

void cunn_Sqrt_init(lua_State *L) {
  luaT_pushmetatable(L, "torch.CudaTensor");
  luaT_registeratname(L, cunn_Sqrt__, "nn");
  lua_pop(L,1);
}
