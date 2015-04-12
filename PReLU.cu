#include "THCApply.cuh"
#include "utils.h"


struct PReLUUpdateOutput {
  const float weight_;

  PReLUUpdateOutput(float weight): weight_(weight) {}

  __device__ __forceinline__ void operator()(float* out, float* in) {
    float x = *in;
    *out = (x > 0) ? x : weight_ * x;
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

  THCudaTensor_resizeAs(state, output, input);

  if (nOutputPlane == 0)
  {
    // handle shared parameter case
    float w = THCudaTensor_get1d(state, weight, 0);
    THCudaTensor_pointwiseApply2(state, output, input, PReLUUpdateOutput(w));
  } else {
    printf("PReLU cuda version only supports shared parameter case\n");
    THError("aborting");
  }

  THCudaCheck(cudaGetLastError());
  return 1;
}

struct PReLUUpdateGradInput {
  const float weight_;

  PReLUUpdateGradInput(float weight): weight_(weight) {}

  __device__ __forceinline__ void operator()(float* gradInput_data,
                                             float* gradOutput_data,
                                             float* input_data) {
    if ((*input_data) > 0)
    {
      *gradInput_data = *gradOutput_data;
    }
    else
    {
      *gradInput_data = weight_ * (*gradOutput_data);
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

  THCudaTensor_resizeAs(state, gradInput, input);
  if (nOutputPlane == 0)
  {
    // handle shared parameter case
    float w = THCudaTensor_get1d(state, weight, 0);
    THCudaTensor_pointwiseApply3(state, gradInput, gradOutput, input, PReLUUpdateGradInput(w));
  } else {
    printf("PReLU cuda version only supports shared parameter case\n");
    THError("aborting");
  }

  THCudaCheck(cudaGetLastError());
  return 1;
}

struct PReLUAccGradParameters {
  float sum;

  PReLUAccGradParameters(): sum(0) {}

  __device__ __forceinline__ void operator()(float* input_data,
                                             float* gradOutput_data) {
    if ((*input_data) <= 0)
    {
      sum += (*input_data) * (*gradOutput_data);
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
  THCudaTensor *gradWeight = (THCudaTensor*) luaT_getfieldcheckudata(L, 1, "gradWeight", "torch.CudaTensor");
  long nOutputPlane = luaT_getfieldchecknumber(L, 1, "nOutputPlane");
  float scale = luaL_optnumber(L, 4, 1);

  if (nOutputPlane == 0)
  {
    // handle shared parameter case
    PReLUAccGradParameters functor;
    THCudaTensor_pointwiseApply2(state, input, gradOutput, functor);
    THCudaTensor_set1d(state, gradWeight, 0, functor.sum);
  } else {
    printf("PReLU cuda version only supports shared parameter case\n");
    THError("aborting");
  }

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