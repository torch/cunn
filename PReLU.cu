#include "THCApply.cuh"
#include "utils.h"


struct PReLUUpdateOutput {
  float* weight_;

  PReLUUpdateOutput(float* weight): weight_(weight) {}

  __device__ __forceinline__ void operator()(float* out, float* in) {
    float x = *in;
    *out = (x > 0) ? x : weight_[0] * x;
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

  assert(nOutputPlane == 0 && "PReLU cuda version only supports shared parameter case\n");
  THCudaTensor_resizeAs(state, output, input);

  float* w = THCudaTensor_data(state, weight);
  THCudaTensor_pointwiseApply2(state, output, input, PReLUUpdateOutput(w));

  THCudaCheck(cudaGetLastError());
  return 1;
}

struct PReLUUpdateGradInput {
  float *weight_;

  PReLUUpdateGradInput(float* weight): weight_(weight) {}

  __device__ __forceinline__ void operator()(float* gradInput_data,
                                             float* gradOutput_data,
                                             float* input_data) {
    if ((*input_data) > 0)
    {
      *gradInput_data = *gradOutput_data;
    }
    else
    {
      *gradInput_data = weight_[0] * (*gradOutput_data);
    }
  }
};

static int cunn_PReLU_updateGradInput(lua_State *L)
{
  THCState *state = getCutorchState(L);
  // Input:
  THCudaTensor *input = (THCudaTensor*)luaT_checkudata(L, 2, "torch.CudaTensor");
  THCudaTensor *gradOutput = (THCudaTensor*)luaT_checkudata(L, 3, "torch.CudaTensor");
  // Params:
  THCudaTensor *weight = (THCudaTensor*) luaT_getfieldcheckudata(L, 1, "weight", "torch.CudaTensor");
  THCudaTensor *gradInput = (THCudaTensor*) luaT_getfieldcheckudata(L, 1, "gradInput", "torch.CudaTensor");
  long nOutputPlane = luaT_getfieldchecknumber(L, 1, "nOutputPlane");

  assert(nOutputPlane == 0 &&"PReLU cuda version only supports shared parameter case\n");
  THCudaTensor_resizeAs(state, gradInput, input);

  float* w = THCudaTensor_data(state, weight);
  THCudaTensor_pointwiseApply3(state, gradInput, gradOutput, input, PReLUUpdateGradInput(w));

  THCudaCheck(cudaGetLastError());
  return 1;
}

struct PReLUAccGradParameters {
  __device__ __forceinline__ void operator()(float* input_data,
                                             float* gradOutput_data) {
    if ((*input_data) <= 0)
    {
      *input_data = (*input_data) * (*gradOutput_data);
    }
  }
};

static int cunn_PReLU_accGradParameters(lua_State *L)
{
  THCState *state = getCutorchState(L);
  // Input:
  THCudaTensor *input = (THCudaTensor*)luaT_checkudata(L, 2, "torch.CudaTensor");
  THCudaTensor *gradOutput = (THCudaTensor*)luaT_checkudata(L, 3, "torch.CudaTensor");
  // Params:
  THCudaTensor *weight = (THCudaTensor*) luaT_getfieldcheckudata(L, 1, "weight", "torch.CudaTensor");
  THCudaTensor *gradWeight = (THCudaTensor*) luaT_getfieldcheckudata(L, 1, "gradWeight", "torch.CudaTensor");
  THCudaTensor *gradInput = (THCudaTensor*) luaT_getfieldcheckudata(L, 1, "gradInput", "torch.CudaTensor");
  long nOutputPlane = luaT_getfieldchecknumber(L, 1, "nOutputPlane");
  float scale = luaL_optnumber(L, 4, 1);

  assert(nOutputPlane == 0 && "PReLU cuda version only supports shared parameter case\n");

  // use grad input for temporary storage, then call updateGradInput again
  PReLUAccGradParameters functor;
  THCudaTensor_pointwiseApply2(state, gradInput, gradOutput, functor);
  // introduces a sync point ...
  float sum = THCudaTensor_sumall(state, gradInput);
  THCudaTensor_set1d(state, gradWeight, 0, sum);

  // restore gradInput
  float* w = THCudaTensor_data(state, weight);
  THCudaTensor_pointwiseApply3(state, gradInput, gradOutput, input, PReLUUpdateGradInput(w));

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
