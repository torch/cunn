#include <thrust/reduce.h>
#include <thrust/transform.h>

struct l1cost_functor
{
  l1cost_functor() {}

  __host__ __device__ float operator()(float x, float y) const
    {
      return abs(x)+abs(y);
  }
};

static int cunn_L1Cost_updateOutput(lua_State *L)
{
  THCudaTensor *input = (THCudaTensor*)luaT_checkudata(L, 2, "torch.CudaTensor");

  float sum;
  long size = THCudaTensor_nElement(input);
  input = THCudaTensor_newContiguous(input);
  thrust::device_ptr<float> input_data(THCudaTensor_data(input));
  sum = thrust::reduce(input_data, input_data+size, (float) 0, l1cost_functor());

  THCudaTensor_free(input);

  lua_pushnumber(L, sum);
  lua_setfield(L, 1, "output");

  lua_pushnumber(L, sum);
  return 1;
}

struct l1cost_updateGradInput_functor
{
  l1cost_updateGradInput_functor() {}

  __host__ __device__ float operator()(float x) const
    {
      if(x > 0)
        return 1;
      else if(x < 0)
        return -1;
      else
        return 0;
  }
};

static int cunn_L1Cost_updateGradInput(lua_State *L)
{
  THCudaTensor *input = (THCudaTensor*)luaT_checkudata(L, 2, "torch.CudaTensor");
  THCudaTensor *gradInput = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "gradInput", "torch.CudaTensor");

  long size = THCudaTensor_nElement(input);

  input = THCudaTensor_newContiguous(input);
  THCudaTensor_resizeAs(gradInput, input);

  thrust::device_ptr<float> input_data(THCudaTensor_data(input));
  thrust::device_ptr<float> gradInput_data(THCudaTensor_data(gradInput));

  thrust::transform(input_data, input_data+size, gradInput_data, l1cost_updateGradInput_functor());

  THCudaTensor_free(input);
  return 1;
}

static const struct luaL_Reg cunn_L1Cost__ [] = {
  {"L1Cost_updateOutput", cunn_L1Cost_updateOutput},
  {"L1Cost_updateGradInput", cunn_L1Cost_updateGradInput},
  {NULL, NULL}
};

static void cunn_L1Cost_init(lua_State *L)
{
  luaT_pushmetatable(L, "torch.CudaTensor");
  luaT_registeratname(L, cunn_L1Cost__, "nn");
  lua_pop(L,1);
}
