#include "utils.h"

#include <thrust/fill.h>
#include <thrust/functional.h>
#include <thrust/device_ptr.h>
#include <thrust/reduce.h>
#include <thrust/inner_product.h>
#if CUDA_VERSION >= 7000
#include <thrust/system/cuda/execution_policy.h>
#endif

struct abs_functor
{
  abs_functor() {}

  __host__ __device__ float operator()(const float& x, const float& y) const
    {
      float z = x-y;
      return z >= 0 ? z : -z;
    }
};


static int cunn_AbsCriterion_updateOutput(lua_State *L)
{
  THCState *state = getCutorchState(L);
  THCudaTensor *input = (THCudaTensor*)luaT_checkudata(L, 2, "torch.CudaTensor");
  THCudaTensor *target = (THCudaTensor*)luaT_checkudata(L, 3, "torch.CudaTensor");
  int sizeAverage = luaT_getfieldcheckboolean(L, 1, "sizeAverage");
  THAssert(THCudaTensor_checkGPU(state, 2, input, target));
  float sum;

  long size = THCudaTensor_nElement(state, input);

  input = THCudaTensor_newContiguous(state, input);
  target = THCudaTensor_newContiguous(state, target);

  thrust::device_ptr<float> input_data(THCudaTensor_data(state, input));
  thrust::device_ptr<float> target_data(THCudaTensor_data(state, target));
  sum = thrust::inner_product(
#if CUDA_VERSION >= 7000
    thrust::cuda::par.on(THCState_getCurrentStream(state)),
#endif
    input_data, input_data+size, target_data, (float) 0,
    thrust::plus<float>(), abs_functor());

  if(sizeAverage)
    sum /= size;

  THCudaTensor_free(state, input);
  THCudaTensor_free(state, target);

  lua_pushnumber(L, sum);
  lua_setfield(L, 1, "output");

  lua_pushnumber(L, sum);
  return 1;
}


struct abs_updateGradInput_functor
{
  const float norm;

  abs_updateGradInput_functor(float norm_) : norm(norm_) {}

  __host__ __device__ float operator()(const float& x, const float& y) const
    {
      return (x - y) >= 0 ? norm : -norm;
    }
};

static int cunn_AbsCriterion_updateGradInput(lua_State *L)
{
  THCState *state = getCutorchState(L);
  THCudaTensor *input = (THCudaTensor*)luaT_checkudata(L, 2, "torch.CudaTensor");
  THCudaTensor *target = (THCudaTensor*)luaT_checkudata(L, 3, "torch.CudaTensor");
  int sizeAverage = luaT_getfieldcheckboolean(L, 1, "sizeAverage");
  THCudaTensor *gradInput = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "gradInput", "torch.CudaTensor");
  THAssert(THCudaTensor_checkGPU(state, 3, input, target, gradInput));

  long size = THCudaTensor_nElement(state, input);
  float norm = (sizeAverage ? 1./size : 1.);

  input = THCudaTensor_newContiguous(state, input);
  target = THCudaTensor_newContiguous(state, target);

  THCudaTensor_resizeAs(state, gradInput, input);

  thrust::device_ptr<float> input_data(THCudaTensor_data(state, input));
  thrust::device_ptr<float> target_data(THCudaTensor_data(state, target));
  thrust::device_ptr<float> gradInput_data(THCudaTensor_data(state, gradInput));

  thrust::transform(
#if CUDA_VERSION >= 7000
    thrust::cuda::par.on(THCState_getCurrentStream(state)),
#endif
    input_data, input_data+size, target_data, gradInput_data,
    abs_updateGradInput_functor(norm));

  THCudaTensor_free(state, input);
  THCudaTensor_free(state, target);
  return 1;
}

static const struct luaL_Reg cunn_AbsCriterion__ [] = {
  {"AbsCriterion_updateOutput", cunn_AbsCriterion_updateOutput},
  {"AbsCriterion_updateGradInput", cunn_AbsCriterion_updateGradInput},
  {NULL, NULL}
};

void cunn_AbsCriterion_init(lua_State *L)
{
  luaT_pushmetatable(L, "torch.CudaTensor");
  luaT_registeratname(L, cunn_AbsCriterion__, "nn");
  lua_pop(L,1);
}
