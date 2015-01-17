#include "utils.h"

#include "SpatialPoolingCUDA/updateOutput.cu"
#include "SpatialPoolingCUDA/updateGradInput.cu"

static int cunn_SpatialMaxPoolingCUDA_updateOutput(lua_State *L)
{
  THCState *state = getCutorchState(L);
  THCudaTensor *input = (THCudaTensor *)luaT_checkudata(L, 2, "torch.CudaTensor");
  int kW = luaT_getfieldcheckint(L, 1, "kW");
  int kH = luaT_getfieldcheckint(L, 1, "kH");
  int dW = luaT_getfieldcheckint(L, 1, "dW");
  int dH = luaT_getfieldcheckint(L, 1, "dH");

  THCudaTensor *output = (THCudaTensor *)luaT_getfieldcheckudata(L, 1, "output", "torch.CudaTensor");

  luaL_argcheck(L, input->nDimension == 4, 2, "4D (batch) tensor expected");

  long nInputCols = input->size[2];
  long nInputRows = input->size[1];
  long nInputPlane = input->size[0];
  long batchSize = input->size[3];
  long nOutputCols = (nInputCols - kW) / dW + 1;
  long nOutputRows = (nInputRows - kH) / dH + 1;

  luaL_argcheck(L, nInputCols >= kW && nInputRows >= kH, 2, "input image smaller than kernel size");

  luaL_argcheck(L, THCudaTensor_isContiguous(state, input), 2, "input must be contiguous");
  float *input_data = THCudaTensor_data(state, input);

  THCudaTensor_resize4d(state, output, nInputPlane, nOutputRows, nOutputCols, batchSize);
  float *output_data = THCudaTensor_data(state, output);

  spatialMaxPooling_updateOutput<MaxPooler>
    (input_data, output_data,
     nInputPlane, nInputRows, nInputCols, batchSize,
     nOutputRows, nOutputCols,
     kH, kW,
     0, dW);

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("error in SpatialMaxPoolingCUDA.updateOutput: %s\n", cudaGetErrorString(err));
    THError("aborting");
  }
  return 1;
}

static int cunn_SpatialMaxPoolingCUDA_updateGradInput(lua_State *L)
{
  THCState *state = getCutorchState(L);
  THCudaTensor *input = (THCudaTensor *)luaT_checkudata(L, 2, "torch.CudaTensor");
  THCudaTensor *gradOutput = (THCudaTensor *)luaT_checkudata(L, 3, "torch.CudaTensor");
  int kW = luaT_getfieldcheckint(L, 1, "kW");
  int kH = luaT_getfieldcheckint(L, 1, "kH");
  int dW = luaT_getfieldcheckint(L, 1, "dW");
  int dH = luaT_getfieldcheckint(L, 1, "dH");

  THCudaTensor *gradInput = (THCudaTensor *)luaT_getfieldcheckudata(L, 1, "gradInput", "torch.CudaTensor");
  THCudaTensor *output = (THCudaTensor *)luaT_getfieldcheckudata(L, 1, "output", "torch.CudaTensor");

  long nInputCols = input->size[2];
  long nInputRows = input->size[1];
  long nInputPlane = input->size[0];
  long batchSize = input->size[3];
  long nOutputCols = (nInputCols - kW) / dW + 1;
  long nOutputRows = (nInputRows - kH) / dH + 1;

  THCudaTensor_resizeAs(state, gradInput, input);
  THCudaTensor_zero(state, gradInput);

  float *input_data = THCudaTensor_data(state, input);
  float *output_data = THCudaTensor_data(state, output);
  float *gradOutput_data = THCudaTensor_data(state, gradOutput);
  float *gradInput_data = THCudaTensor_data(state, gradInput);

  spatialMaxPooling_updateGradInput
    (input_data, gradOutput_data, output_data, gradInput_data,
     nInputPlane, nInputRows, nInputCols, batchSize,
     nOutputRows, nOutputCols,
     kH, kW,
     0, dW);

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("error in SpatialMaxPoolingCUDA.updateGradInput: %s\n", cudaGetErrorString(err));
    THError("aborting");
  }
  return 1;
}

static const struct luaL_Reg cunn_SpatialMaxPoolingCUDA__ [] = {
  {"SpatialMaxPoolingCUDA_updateOutput", cunn_SpatialMaxPoolingCUDA_updateOutput},
  {"SpatialMaxPoolingCUDA_updateGradInput", cunn_SpatialMaxPoolingCUDA_updateGradInput},
  {NULL, NULL}
};

static void cunn_SpatialMaxPoolingCUDA_init(lua_State *L)
{
  luaT_pushmetatable(L, "torch.CudaTensor");
  luaT_registeratname(L, cunn_SpatialMaxPoolingCUDA__, "nn");
  lua_pop(L,1);
}
