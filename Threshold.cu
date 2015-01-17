#include "utils.h"

struct thresholdupdateOutput_functor
{
  const double threshold;
  const double val;

  thresholdupdateOutput_functor(double threshold_, double val_) : threshold(threshold_), val(val_) {}

  __host__ __device__ float operator()(const float& input) const
  {
    return (input > threshold) ? input : val;
  }
};

static int cunn_Threshold_updateOutput(lua_State *L)
{
  THCState *state = getCutorchState(L);
  THCudaTensor *input = (THCudaTensor*)luaT_checkudata(L, 2, "torch.CudaTensor");
  THCudaTensor *output = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "output", "torch.CudaTensor");
  double val = luaT_getfieldchecknumber(L, 1, "val");
  double threshold = luaT_getfieldchecknumber(L, 1, "threshold");
  long size = THCudaTensor_nElement(state, input);

  input = THCudaTensor_newContiguous(state, input);

  THCudaTensor_resizeAs(state, output, input);

  thrust::device_ptr<float> output_data(THCudaTensor_data(state, output));
  thrust::device_ptr<float> input_data(THCudaTensor_data(state, input));
  thrust::transform(input_data, input_data+size, output_data,
                    thresholdupdateOutput_functor(threshold, val));

  THCudaTensor_free(state, input);
  return 1;
}

struct thresholdupdateGradInput_functor
{
  const double threshold;
  const double val;

  thresholdupdateGradInput_functor(double threshold_, double val_) : threshold(threshold_), val(val_) {}

  __host__ __device__ float operator()(const float& input, const float& gradOutput) const
  {
    return (input > threshold) ? gradOutput : 0;
  }
};

static int cunn_Threshold_updateGradInput(lua_State *L)
{
  THCState *state = getCutorchState(L);
  THCudaTensor *output = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "output", "torch.CudaTensor");
  THCudaTensor *input = (THCudaTensor*)luaT_checkudata(L, 2, "torch.CudaTensor");
  THCudaTensor *gradOutput = (THCudaTensor*)luaT_checkudata(L, 3, "torch.CudaTensor");
  THCudaTensor *gradInput = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "gradInput", "torch.CudaTensor");
  double val = luaT_getfieldchecknumber(L, 1, "val");
  double threshold = luaT_getfieldchecknumber(L, 1, "threshold");
  long size = THCudaTensor_nElement(state, output);

  gradOutput = THCudaTensor_newContiguous(state, gradOutput);
  input = THCudaTensor_newContiguous(state, input);
  THCudaTensor_resizeAs(state, gradInput, output);

  thrust::device_ptr<float> input_data(THCudaTensor_data(state, input));
  thrust::device_ptr<float> gradOutput_data(THCudaTensor_data(state, gradOutput));
  thrust::device_ptr<float> gradInput_data(THCudaTensor_data(state, gradInput));
  thrust::transform(input_data, input_data+size, gradOutput_data, gradInput_data,
                    thresholdupdateGradInput_functor(threshold, val));

  THCudaTensor_free(state, input);
  THCudaTensor_free(state, gradOutput);
  return 1;
}

static const struct luaL_Reg cunn_Threshold__ [] = {
  {"Threshold_updateOutput", cunn_Threshold_updateOutput},
  {"Threshold_updateGradInput", cunn_Threshold_updateGradInput},
  {NULL, NULL}
};

static void cunn_Threshold_init(lua_State *L)
{
  luaT_pushmetatable(L, "torch.CudaTensor");
  luaT_registeratname(L, cunn_Threshold__, "nn");
  lua_pop(L,1);
}

