#include "utils.h"

#define MULTILABELMARGIN_THREADS 128

__global__ void cunn_MultiLabelMarginCriterion_updateOutput_kernel(float *output, float *input, float *target, int nframe, int dim, int sizeaverage)
{
  // Temporary sums (for mapreduce)
  __shared__ double sums[MULTILABELMARGIN_THREADS];

  // vectors:
  int k = blockIdx.x;
  float *input_k = input + k*dim;
  float *target_k = target + k*dim;
  float *output_k = output + k;

  // iterate over targets
  sums[threadIdx.x] = 0;
  for (int dt = 0; dt < dim; dt++) {
    // next target:
    int target_idx = (int)target_k[dt];
    if (target_idx == 0) break;

    // current value for target
    float input_target_k = input_k[target_idx-1];

    // compare to all inputs (multithreaded):
    for (int d = threadIdx.x; d < dim; d += blockDim.x) {
      // loop over targets again:
      int istarget = 0;
      for (int ddt = 0; ddt < dim; ddt++) {
        if (!target_k[ddt]) break;
        if ( (int)(target_k[ddt]) - 1 == d ) istarget = 1;
      }

      // only if not a target, then contribute to loss:
      if (!istarget) {
        float z = 1 - input_target_k + input_k[d];
        if (z > 0)
          sums[threadIdx.x] += z;
      }
    }
  }
  __syncthreads();

  // reduce
  if (threadIdx.x == 0)
  {
    double sum = 0;
    for (int i=0; i<blockDim.x; i++)
      sum += sums[i];

    if(sizeaverage)
      *output_k = sum/dim;
    else
      *output_k = sum;
  }
}

__global__ void cunn_MultiLabelMarginCriterion_updateGradInput_kernel(float *gradInput, float *input, float *target, int nframe, int dim, int sizeaverage)
{
  // Temporary sums (for mapreduce)
  __shared__ double sums[MULTILABELMARGIN_THREADS];

  // vectors:
  int k = blockIdx.x;
  float *input_k = input + k*dim;
  float *gradInput_k = gradInput + k*dim;
  float *target_k = target + k*dim;

  // gain:
  float g = (sizeaverage ? 1./((float)dim) : 1.);

  // zero gradients:
  for (int d = threadIdx.x; d < dim; d += blockDim.x) {
    gradInput_k[d] = 0;
  }
  __syncthreads();

  // iterate over targets
  for (int dt = 0; dt < dim; dt++) {
    // next target:
    int target_idx = (int)target_k[dt];
    if (target_idx == 0) break;

    // current value for target
    float input_target_k = input_k[target_idx-1];

    // compare to all inputs (multithreaded):
    sums[threadIdx.x] = 0;
    for (int d = threadIdx.x; d < dim; d += blockDim.x) {
      // loop over targets again:
      int istarget = 0;
      for (int ddt = 0; ddt < dim; ddt++) {
        if (!target_k[ddt]) break;
        if ( (int)(target_k[ddt]) - 1 == d ) istarget = 1;
      }

      // only if not a target, then contribute to loss:
      if (!istarget) {
        float z = 1 - input_target_k + input_k[d];
        if (z > 0) {
          sums[threadIdx.x] -= g;
          gradInput_k[d] += g;
        }
      }
    }
    __syncthreads();

    // reduce sum
    if (threadIdx.x == 0) {
      double sum = 0;
      for (int i = 0; i < blockDim.x; i++ ) {
        sum += sums[i];
      }
      gradInput_k[target_idx-1] += sum;
    }
    __syncthreads();
  }
}

static int cunn_MultiLabelMarginCriterion_updateOutput(lua_State *L)
{
  THCState *state = getCutorchState(L);
  THCudaTensor *input = (THCudaTensor*)luaT_checkudata(L, 2, "torch.CudaTensor");
  THCudaTensor *target = (THCudaTensor*)luaT_checkudata(L, 3, "torch.CudaTensor");

  int sizeaverage = luaT_getfieldcheckboolean(L, 1, "sizeAverage");

  input = THCudaTensor_newContiguous(state, input);

  if(input->nDimension == 1)
  {
    THCudaTensor *output = THCudaTensor_newWithSize1d(state, 1);

    dim3 blocks(1);
    dim3 threads(MULTILABELMARGIN_THREADS);

    cunn_MultiLabelMarginCriterion_updateOutput_kernel<<<blocks,threads>>>(
        THCudaTensor_data(state, output),
        THCudaTensor_data(state, input),
        THCudaTensor_data(state, target),
        1, input->size[0],
        sizeaverage
        );
    lua_pushnumber(L, THCudaTensor_sumall(state, output));
    THCudaTensor_free(state, output);
  }
  else if(input->nDimension == 2)
  {
    THCudaTensor *output = THCudaTensor_newWithSize1d(state, input->size[0]);

    dim3 blocks(input->size[0]);
    dim3 threads(MULTILABELMARGIN_THREADS);

    cunn_MultiLabelMarginCriterion_updateOutput_kernel<<<blocks,threads>>>(
        THCudaTensor_data(state, output),
        THCudaTensor_data(state, input),
        THCudaTensor_data(state, target),
        input->size[0], input->size[1],
        sizeaverage
        );
    lua_pushnumber(L, THCudaTensor_sumall(state, output));
    THCudaTensor_free(state, output);
  }
  else
    THError("vector or matrix expected");

  cudaError errcode = cudaGetLastError();
  if(errcode != cudaSuccess)
    THError(cudaGetErrorString(errcode));

  lua_pushstring(L, "output");
  lua_pushvalue(L, -2);
  lua_rawset(L, 1);

  THCudaTensor_free(state, input);
  return 1;
}

static int cunn_MultiLabelMarginCriterion_updateGradInput(lua_State *L)
{
  THCState *state = getCutorchState(L);
  THCudaTensor *input = (THCudaTensor*)luaT_checkudata(L, 2, "torch.CudaTensor");
  int sizeaverage = luaT_getfieldcheckboolean(L, 1, "sizeAverage");
  THCudaTensor *gradInput = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "gradInput", "torch.CudaTensor");
  THCudaTensor *target = (THCudaTensor*)luaT_checkudata(L, 3, "torch.CudaTensor");

  THCudaTensor_resizeAs(state, gradInput, input);

  if(gradInput->nDimension == 1)
  {
    dim3 blocks(1);
    dim3 threads(MULTILABELMARGIN_THREADS);

    cunn_MultiLabelMarginCriterion_updateGradInput_kernel<<<blocks,threads>>>(THCudaTensor_data(state, gradInput),
        THCudaTensor_data(state, input),
        THCudaTensor_data(state, target),
        1, gradInput->size[0],
        sizeaverage);

  }
  else if(gradInput->nDimension == 2)
  {
    dim3 blocks(gradInput->size[0]);
    dim3 threads(MULTILABELMARGIN_THREADS);

    cunn_MultiLabelMarginCriterion_updateGradInput_kernel<<<blocks,threads>>>(THCudaTensor_data(state, gradInput),
        THCudaTensor_data(state, input),
        THCudaTensor_data(state, target),
        gradInput->size[0], gradInput->size[1],
        sizeaverage);
  }
  else
    THError("vector or matrix expected");

  cudaError errcode = cudaGetLastError();
  if(errcode != cudaSuccess)
    THError(cudaGetErrorString(errcode));

  return 1;
}

static const struct luaL_Reg cunn_MultiLabelMarginCriterion__ [] = {
  {"MultiLabelMarginCriterion_updateOutput", cunn_MultiLabelMarginCriterion_updateOutput},
  {"MultiLabelMarginCriterion_updateGradInput", cunn_MultiLabelMarginCriterion_updateGradInput},
  {NULL, NULL}
};

static void cunn_MultiLabelMarginCriterion_init(lua_State *L)
{
  luaT_pushmetatable(L, "torch.CudaTensor");
  luaT_registeratname(L, cunn_MultiLabelMarginCriterion__, "nn");
  lua_pop(L,1);
}
