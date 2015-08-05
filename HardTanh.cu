#include "THCApply.cuh"
#include "utils.h"

struct HardTanhUpdateOutput {
  __device__ __forceinline__ void operator()(float* out, float* in) {
    if (*in < -1.0f) {
      *out = -1.0f;
    } else {
      if (*in <= 1.0f) {
        *out = *in;
      } else {
        *out = 1.0f;
      }
    }
  }
};

static int cunn_HardTanh_updateOutput(lua_State *L) {
  THCState *state = getCutorchState(L);
  THCudaTensor *input = (THCudaTensor*)luaT_checkudata(L, 2, "torch.CudaTensor");
  THCudaTensor *output = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "output", "torch.CudaTensor");

  THCudaTensor_resizeAs(state, output, input);
  THCudaTensor_pointwiseApply2(state, output, input,
                               HardTanhUpdateOutput());

  THCudaCheck(cudaGetLastError());
  return 1;
}

struct HardTanhUpdateGradInput {
  __device__ __forceinline__ void operator()(float* gradInput, float* input,
                                             float* gradOutput) const {
    if (*input < -1.0f || *input > 1.0f) {
      *gradInput = 0.0f;
    } else {
      *gradInput = *gradOutput;
    }
  }
};

static int cunn_HardTanh_updateGradInput(lua_State *L) {
  THCState *state = getCutorchState(L);
  THCudaTensor *input = (THCudaTensor*)luaT_checkudata(L, 2, "torch.CudaTensor");
  THCudaTensor *gradOutput = (THCudaTensor*)luaT_checkudata(L, 3, "torch.CudaTensor");
  THCudaTensor *gradInput = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "gradInput", "torch.CudaTensor");
  THCudaTensor_resizeAs(state, gradInput, input);
  THCudaTensor_pointwiseApply3(state, gradInput, input, gradOutput,
                               HardTanhUpdateGradInput());

  THCudaCheck(cudaGetLastError());
  return 1;
}


static const struct luaL_Reg cunn_HardTanh__ [] = {
  {"HardTanh_updateOutput", cunn_HardTanh_updateOutput},
  {"HardTanh_updateGradInput", cunn_HardTanh_updateGradInput},
  {NULL, NULL}
};

void cunn_HardTanh_init(lua_State *L) {
  luaT_pushmetatable(L, "torch.CudaTensor");
  luaT_registeratname(L, cunn_HardTanh__, "nn");
  lua_pop(L,1);
}
