#include "utils.h"

__global__ void min_gradInput(float *input, float *output, float *indices,
                              long nrows, long ncols)
{
  // output offset:
  long o = threadIdx.x + blockDim.x * blockIdx.x;
  if (o >= nrows) return;

  // input offset:
  long i = o * ncols;

  // bprop min gradient:
  long idx = indices[o]-1;
  input[i+idx] = output[o];
}

static int cunn_Min_updateOutput(lua_State *L)
{
  THCState *state = getCutorchState(L);
  THCudaTensor *input = (THCudaTensor *)luaT_checkudata(L, 2, "torch.CudaTensor");
  int dimension = luaT_getfieldcheckint(L, 1, "dimension")-1;
  THCudaTensor *indices = (THCudaTensor *)luaT_getfieldcheckudata(L, 1, "indices", "torch.CudaTensor");
  THCudaTensor *output = (THCudaTensor *)luaT_getfieldcheckudata(L, 1, "output", "torch.CudaTensor");

  THAssert(THCudaTensor_checkGPU(state, 3, input, indices, output));
  luaL_argcheck(L, dimension >= 0 && dimension < input->nDimension, 2, "dimension out of range");
  luaL_argcheck(L, dimension == input->nDimension-1, 2, "only supported dimension is innermost (CUDA kernel only)");

  THCudaTensor_min(state, output, indices, input, dimension);
  THCudaTensor_select(state, output, NULL, dimension, 0);

  return 1;
}

static int cunn_Min_updateGradInput(lua_State *L)
{
  THCState *state = getCutorchState(L);
  THCudaTensor *input = (THCudaTensor *)luaT_checkudata(L, 2, "torch.CudaTensor");
  THCudaTensor *gradOutput = (THCudaTensor *)luaT_checkudata(L, 3, "torch.CudaTensor");
  THCudaTensor *indices = (THCudaTensor *)luaT_getfieldcheckudata(L, 1, "indices", "torch.CudaTensor");
  int dimension  = luaT_getfieldcheckint(L, 1, "dimension")-1;
  THCudaTensor *gradInput  = (THCudaTensor *)luaT_getfieldcheckudata(L, 1, "gradInput", "torch.CudaTensor");

  THAssert(THCudaTensor_checkGPU(state, 4, input, indices, gradOutput, gradInput));
  THCudaTensor_resizeAs(state, gradInput, input);
  THCudaTensor_zero(state, gradInput);

  float *gradInput_data = THCudaTensor_data(state, gradInput);
  float *gradOutput_data = THCudaTensor_data(state, gradOutput);
  float *indices_data = THCudaTensor_data(state, indices);

  long nrows = THCudaTensor_nElement(state, gradOutput);
  long ncols = gradInput->size[dimension];

  // cuda blocks & threads:
  long nthreads = 256;
  long nblocks = ceil((float)nrows / nthreads);
  dim3 blocks(nblocks);
  dim3 threads(nthreads);

  // kernel:
  min_gradInput <<<blocks, threads,
    0, THCState_getCurrentStream(state)>>> (gradInput_data, gradOutput_data, indices_data, nrows, ncols);

  // check for errors
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("error in Min.updateOutput: %s\n", cudaGetErrorString(err));
    THError("aborting");
  }

  return 1;
}

static const struct luaL_Reg cunn_Min__ [] = {
  {"Min_updateOutput", cunn_Min_updateOutput},
  {"Min_updateGradInput", cunn_Min_updateGradInput},
  {NULL, NULL}
};

void cunn_Min_init(lua_State *L)
{
  luaT_pushmetatable(L, "torch.CudaTensor");
  luaT_registeratname(L, cunn_Min__, "nn");
  lua_pop(L,1);
}
