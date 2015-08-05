#include "THCApply.cuh"
#include "utils.h"
#include "THCApply.cuh"

struct TanhUpdateOutput {
  __device__ __forceinline__ void operator()(float* out, float* in) {
    *out = tanhf(*in);
  }
};

// in-place variant
struct TanhUpdateOutputIP {
  __device__ __forceinline__ void operator()(float* x) {
    *x = tanhf(*x);
  }
};

static int cunn_Tanh_updateOutput(lua_State *L) {
  THCState *state = getCutorchState(L);
  THCudaTensor *input = (THCudaTensor*)luaT_checkudata(L, 2, "torch.CudaTensor");
  bool inPlace = luaT_getfieldcheckboolean(L, 1, "inplace");
  THCudaTensor *output = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "output", "torch.CudaTensor");

  if (inPlace) {
    THCudaTensor_pointwiseApply1(state, input,
                                 TanhUpdateOutputIP());
    THCudaTensor_set(state, output, input);
  } else {
    THCudaTensor_resizeAs(state, output, input);
    THCudaTensor_pointwiseApply2(state, output, input,
                                 TanhUpdateOutput());
  }

  THCudaCheck(cudaGetLastError());
  return 1;
}

struct TanhUpdateGradInput {
  __device__ __forceinline__ void operator()(float* gradInput, float* output,
                                             float* gradOutput) const {
    *gradInput = *gradOutput * (1.0f - *output * *output);
  }
};

struct TanhUpdateGradInputIP {
  __device__ __forceinline__ void operator()(float* gradOutput,
                                             float* output) const {
    *gradOutput = *gradOutput * (1.0f - *output * *output);
  }
};

static int cunn_Tanh_updateGradInput(lua_State *L) {
  THCState *state = getCutorchState(L);
  THCudaTensor *output = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "output", "torch.CudaTensor");
  THCudaTensor *gradOutput = (THCudaTensor*)luaT_checkudata(L, 3, "torch.CudaTensor");
  bool inPlace = luaT_getfieldcheckboolean(L, 1, "inplace");
  THCudaTensor *gradInput = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "gradInput", "torch.CudaTensor");

  if (inPlace) {
    THCudaTensor_pointwiseApply2(state, gradOutput, output,
                                 TanhUpdateGradInputIP());
    THCudaTensor_set(state, gradInput, gradOutput);
  } else {
    THCudaTensor_resizeAs(state, gradInput, output);
    THCudaTensor_pointwiseApply3(state, gradInput, output, gradOutput,
                                 TanhUpdateGradInput());
  }

  THCudaCheck(cudaGetLastError());
  return 1;
}


static const struct luaL_Reg cunn_Tanh__ [] = {
  {"Tanh_updateOutput", cunn_Tanh_updateOutput},
  {"Tanh_updateGradInput", cunn_Tanh_updateGradInput},
  {NULL, NULL}
};

void cunn_Tanh_init(lua_State *L) {
  luaT_pushmetatable(L, "torch.CudaTensor");
  luaT_registeratname(L, cunn_Tanh__, "nn");
  lua_pop(L,1);
}
