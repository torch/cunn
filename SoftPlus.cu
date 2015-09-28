#include "THCApply.cuh"
#include "utils.h"

struct SoftPlusUpdateOutput {
  SoftPlusUpdateOutput(float threshold, float beta): threshold_(threshold)
                                                   , beta_(beta)
                                                   , invBeta_(1.0f / beta) {
  }

  __device__ __forceinline__ void operator()(float* out, float* in) {
    float discriminant = *in * beta_;
    if (discriminant > threshold_) {
      *out = *in;
    } else {
      *out = log1pf(exp(discriminant)) * invBeta_;
    }
  }

 private:
  float threshold_;
  float beta_;
  float invBeta_;
};

// in-place variant
struct SoftPlusUpdateOutputIP {
  SoftPlusUpdateOutputIP(float threshold, float beta): threshold_(threshold)
                                                     , beta_(beta)
                                                     , invBeta_(1.0f / beta) {
  }

  __device__ __forceinline__ void operator()(float* x) {
    float discriminant = *x * beta_;
    if (discriminant <= threshold_) {
      *x = log1pf(exp(discriminant)) * invBeta_;
    }
  }

 private:
  float threshold_;
  float beta_;
  float invBeta_;
};

static int cunn_SoftPlus_updateOutput(lua_State *L) {
  THCState *state = getCutorchState(L);
  THCudaTensor *input = (THCudaTensor*)luaT_checkudata(L, 2, "torch.CudaTensor");
  float beta = luaT_getfieldchecknumber(L, 1, "beta");
  float threshold = luaT_getfieldchecknumber(L, 1, "threshold");
  bool inPlace = luaT_getfieldcheckboolean(L, 1, "inplace");
  THCudaTensor *output = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "output", "torch.CudaTensor");

  if (inPlace) {
    THCudaTensor_pointwiseApply1(state, input,
                                 SoftPlusUpdateOutputIP(threshold, beta));
    THCudaTensor_set(state, output, input);
  } else {
    THCudaTensor_resizeAs(state, output, input);
    THCudaTensor_pointwiseApply2(state, output, input,
                                 SoftPlusUpdateOutput(threshold, beta));
  }

  THCudaCheck(cudaGetLastError());
  return 1;
}

struct SoftPlusUpdateGradInput {
  SoftPlusUpdateGradInput(float threshold, float beta): threshold_(threshold)
                                                      , beta_(beta) {
  }

  __device__ __forceinline__ void operator()(float* gradInput, float* output,
                                             float* gradOutput) const {
    float discriminant = *output * beta_;
    float exp_disc = expf(discriminant);
    if (discriminant > threshold_) {
      *gradInput = *gradOutput;
    } else {
      *gradInput = *gradOutput * (exp_disc - 1.0f) / exp_disc;
    }
  }

 private:
  float threshold_;
  float beta_;
};

struct SoftPlusUpdateGradInputIP {
  SoftPlusUpdateGradInputIP(float threshold, float beta): threshold_(threshold)
                                                        , beta_(beta) {
  }

  __device__ __forceinline__ void operator()(float* gradOutput,
                                             float* output) const {
    float discriminant = *output * beta_;
    float exp_disc = expf(discriminant);
    if (discriminant <= threshold_) {
      *gradOutput = *gradOutput * (exp_disc - 1.0f) / exp_disc;
    }
  }

 private:
  float threshold_;
  float beta_;
};

static int cunn_SoftPlus_updateGradInput(lua_State *L) {
  THCState *state = getCutorchState(L);
  THCudaTensor *output = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "output", "torch.CudaTensor");
  THCudaTensor *gradOutput = (THCudaTensor*)luaT_checkudata(L, 3, "torch.CudaTensor");
  float beta = luaT_getfieldchecknumber(L, 1, "beta");
  float threshold = luaT_getfieldchecknumber(L, 1, "threshold");
  bool inPlace = luaT_getfieldcheckboolean(L, 1, "inplace");
  THCudaTensor *gradInput = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "gradInput", "torch.CudaTensor");

  if (inPlace) {
    THCudaTensor_pointwiseApply2(state, gradOutput, output,
                                 SoftPlusUpdateGradInputIP(threshold, beta));
    THCudaTensor_set(state, gradInput, gradOutput);
  } else {
    THCudaTensor_resizeAs(state, gradInput, output);
    THCudaTensor_pointwiseApply3(state, gradInput, output, gradOutput,
                                 SoftPlusUpdateGradInput(threshold, beta));
  }

  THCudaCheck(cudaGetLastError());
  return 1;
}


static const struct luaL_Reg cunn_SoftPlus__ [] = {
  {"SoftPlus_updateOutput", cunn_SoftPlus_updateOutput},
  {"SoftPlus_updateGradInput", cunn_SoftPlus_updateGradInput},
  {NULL, NULL}
};

void cunn_SoftPlus_init(lua_State *L) {
  luaT_pushmetatable(L, "torch.CudaTensor");
  luaT_registeratname(L, cunn_SoftPlus__, "nn");
  lua_pop(L,1);
}
