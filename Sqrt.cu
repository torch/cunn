#include "utils.h"

struct sqrtupdateOutput_functor
{
  const double bias;

  sqrtupdateOutput_functor(double bias_) : bias(bias_) {}

  __host__ __device__ float operator()(const float& input) const
  {
    return sqrt(input+bias);
  }
};

static int cunn_Sqrt_updateOutput(lua_State *L)
{
  THCState *state = getCutorchState(L);
  double bias = luaT_getfieldchecknumber(L,1,"eps");
  THCudaTensor *input = (THCudaTensor*)luaT_checkudata(L, 2, "torch.CudaTensor");
  THCudaTensor *output = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "output", "torch.CudaTensor");
  long size = THCudaTensor_nElement(state, input);

  input = THCudaTensor_newContiguous(state, input);

  THCudaTensor_resizeAs(state, output, input);

  thrust::device_ptr<float> output_data(THCudaTensor_data(state, output));
  thrust::device_ptr<float> input_data(THCudaTensor_data(state, input));
  thrust::transform(input_data, input_data+size, output_data, sqrtupdateOutput_functor(bias));

  THCudaTensor_free(state, input);
  return 1;
}

struct sqrtupdateGradInput_functor
{
  const double bias;

  sqrtupdateGradInput_functor(double bias_) : bias(bias_) {}

  __host__ __device__ float operator()(const float& output, const float& gradOutput) const
  {
    return 0.5 * gradOutput / output;
  }
};

static int cunn_Sqrt_updateGradInput(lua_State *L)
{
  THCState *state = getCutorchState(L);
  double bias = luaT_getfieldchecknumber(L,1,"eps");
  THCudaTensor *output = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "output", "torch.CudaTensor");
  THCudaTensor *gradOutput = (THCudaTensor*)luaT_checkudata(L, 3, "torch.CudaTensor");
  THCudaTensor *gradInput = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "gradInput", "torch.CudaTensor");
  long size = THCudaTensor_nElement(state, output);

  gradOutput = THCudaTensor_newContiguous(state, gradOutput);
  THCudaTensor_resizeAs(state, gradInput, output);

  thrust::device_ptr<float> output_data(THCudaTensor_data(state, output));
  thrust::device_ptr<float> gradOutput_data(THCudaTensor_data(state, gradOutput));
  thrust::device_ptr<float> gradInput_data(THCudaTensor_data(state, gradInput));
  thrust::transform(output_data, output_data+size, gradOutput_data, gradInput_data, sqrtupdateGradInput_functor(bias));

  THCudaTensor_free(state, gradOutput);
  return 1;
}

static const struct luaL_Reg cunn_Sqrt__ [] = {
  {"Sqrt_updateOutput", cunn_Sqrt_updateOutput},
  {"Sqrt_updateGradInput", cunn_Sqrt_updateGradInput},
  {NULL, NULL}
};

static void cunn_Sqrt_init(lua_State *L)
{
  luaT_pushmetatable(L, "torch.CudaTensor");
  luaT_registeratname(L, cunn_Sqrt__, "nn");
  lua_pop(L,1);
}
