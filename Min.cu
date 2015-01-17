#include "utils.h"

/*
 * Description:
 *    this function finds the min along the innermost dimension
 *    Nd input, (N-1)d output, (N-1)d argmin
 */
__global__ void min_output(float *input, float *output, float *indices,
                           long nrows, long ncols)
{
  // output offset:
  long o = threadIdx.x + blockDim.x * blockIdx.x;
  if (o >= nrows) return;

  // input offset:
  long i = o * ncols;

  // move pointers
  input = input + i;

  // compute min:
  float min = input[0];
  long argmin = 0;
  long ii;
  for (ii=1; ii<ncols; ii++) {
      float val = input[ii];
      if (val < min) {
          min = val;
          argmin = ii;
      }
  }

  // store
  output[o] = min;
  indices[o] = argmin+1;
}

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

  luaL_argcheck(L, dimension >= 0 && dimension < input->nDimension, 2, "dimension out of range");
  luaL_argcheck(L, dimension == input->nDimension-1, 2, "only supported dimension is innermost (CUDA kernel only)");

  input = THCudaTensor_newContiguous(state, input);

  THLongStorage *dim = THLongStorage_newWithSize(input->nDimension);
  long i;
  for(i = 0; i < input->nDimension; i++)
    dim->data[i] = input->size[i];
  dim->data[dimension] = 1;
  THCudaTensor_resize(state, output, dim, NULL);
  THCudaTensor_resize(state, indices, dim, NULL);
  THLongStorage_free(dim);

  float *input_data = THCudaTensor_data(state, input);
  float *output_data = THCudaTensor_data(state, output);
  float *indices_data = THCudaTensor_data(state, indices);

  long nrows = THCudaTensor_nElement(state, output);
  long ncols = input->size[dimension];

  // cuda blocks & threads:
  long nthreads = 256;
  long nblocks = ceil((float)nrows / nthreads);
  dim3 blocks(nblocks);
  dim3 threads(nthreads);

  // kernel:
  min_output <<<blocks, threads>>> (input_data, output_data, indices_data, nrows, ncols);

  // check for errors
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("error in Min.updateOutput: %s\n", cudaGetErrorString(err));
    THError("aborting");
  }

  // final cut:
  THCudaTensor_free(state, input);
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
  min_gradInput <<<blocks, threads>>> (gradInput_data, gradOutput_data, indices_data, nrows, ncols);

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

static void cunn_Min_init(lua_State *L)
{
  luaT_pushmetatable(L, "torch.CudaTensor");
  luaT_registeratname(L, cunn_Min__, "nn");
  lua_pop(L,1);
}
