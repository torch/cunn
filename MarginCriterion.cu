#include "utils.h"

#define MARGIN_THREADS 128

__global__ void cunn_MarginCriterion_updateOutput_kernel(float *output, float *input, float *target, int inputSize, int sizeaverage, float margin)
{
  __shared__ float buffer[MARGIN_THREADS];
  int k = blockIdx.x;

  float *input_k = input + k;
  float *output_k = output + k;
  float *target_k = target + k;

  int i_start = threadIdx.x;
  int i_end = inputSize;
  int i_step = blockDim.x;

  buffer[threadIdx.x] = 0;
  for(int i = i_start; i < i_end; i += i_step)
  {
    float z = margin - target_k[i] * input_k[i];
    buffer[threadIdx.x] += z>0 ? z : 0;
  }
  __syncthreads();

  // reduce
  if (threadIdx.x == 0)
  {
    float sum = 0;
    for (int i=0; i<blockDim.x; i++)
      sum += buffer[i];

    if(sizeaverage)
      *output_k = sum/(float) inputSize;
    else
      *output_k = sum;
  }
}

__global__ void cunn_MarginCriterion_updateGradInput_kernel(float *gradInput, float *input, float *target, int inputSize, int sizeaverage, float margin)
{
  int k = blockIdx.x;
  float *input_k = input + k;
  float *target_k = target + k;
  float *gradInput_k = gradInput + k;
  float g = (sizeaverage ? 1./((float)inputSize) : 1.); // for sizeAverage

  int i_start = threadIdx.x;
  int i_end = inputSize;
  int i_step = blockDim.x;

  for (int i=i_start; i<i_end; i+=i_step)
  {
    if(target_k[i] * input_k[i] < margin)
    {
      gradInput_k[i] = -g * target_k[i];
    }
    else
      gradInput_k[i] = 0;
  }

  __syncthreads();
}

static int cunn_MarginCriterion_updateOutput(lua_State *L)
{
  THCState *state = getCutorchState(L);
  THCudaTensor *input = (THCudaTensor*)luaT_checkudata(L, 2, "torch.CudaTensor");
  int sizeaverage = luaT_getfieldcheckboolean(L, 1, "sizeAverage");
  float margin = luaT_getfieldchecknumber(L, 1, "margin");

  THAssert(input->nDimension == 1);


  THCudaTensor *target = (THCudaTensor*)luaT_checkudata(L, 3, "torch.CudaTensor");   
  THAssert(THCudaTensor_checkGPU(state, 2, input, target));

  input = THCudaTensor_newContiguous(state, input);
  THCudaStorage *output = THCudaStorage_newWithSize(state, 1);

  dim3 blocks(1);
  dim3 threads(MARGIN_THREADS);

  cunn_MarginCriterion_updateOutput_kernel <<<blocks,threads,
        0, THCState_getCurrentStream(state)>>>(output->data,
                                               THCudaTensor_data(state, input),
                                               THCudaTensor_data(state, target),
                                               input->size[0],
                                               sizeaverage, margin);
  lua_pushnumber(L, THCudaStorage_get(state, output, 0));

  THCudaStorage_free(state, output);

  cudaError errcode = cudaGetLastError();
  if(errcode != cudaSuccess)
    THError(cudaGetErrorString(errcode));

  lua_pushstring(L, "output");
  lua_pushvalue(L, -2);
  lua_rawset(L, 1);

  THCudaTensor_free(state, input);
  return 1;
}

static int cunn_MarginCriterion_updateGradInput(lua_State *L)
{
  THCState *state = getCutorchState(L);
  THCudaTensor *input = (THCudaTensor*)luaT_checkudata(L, 2, "torch.CudaTensor");
  int sizeaverage = luaT_getfieldcheckboolean(L, 1, "sizeAverage");
  float margin = luaT_getfieldchecknumber(L, 1, "margin");

  THAssert(input->nDimension == 1);

  THCudaTensor *gradInput = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "gradInput", "torch.CudaTensor");
  THCudaTensor *target = (THCudaTensor*)luaT_checkudata(L, 3, "torch.CudaTensor");
  
  THAssert(THCudaTensor_checkGPU(state, 3, input, target, gradInput));
 
  input = THCudaTensor_newContiguous(state, input);
  THCudaTensor_resizeAs(state, gradInput, input);
  dim3 blocks(1);
  dim3 threads(MARGIN_THREADS);

  cunn_MarginCriterion_updateGradInput_kernel <<<blocks,threads,
        0, THCState_getCurrentStream(state)>>>(THCudaTensor_data(state, gradInput),
                                               THCudaTensor_data(state, input),
                                               THCudaTensor_data(state, target),
                                               gradInput->size[0],
                                               sizeaverage, margin);
                     


  cudaError errcode = cudaGetLastError();
  if(errcode != cudaSuccess)
    THError(cudaGetErrorString(errcode));

  THCudaTensor_free(state, input);
  return 1;
}

static const struct luaL_Reg cunn_MarginCriterion__ [] = {
  {"MarginCriterion_updateOutput", cunn_MarginCriterion_updateOutput},
  {"MarginCriterion_updateGradInput", cunn_MarginCriterion_updateGradInput},
  {NULL, NULL}
};

void cunn_MarginCriterion_init(lua_State *L)
{
  luaT_pushmetatable(L, "torch.CudaTensor");
  luaT_registeratname(L, cunn_MarginCriterion__, "nn");
  lua_pop(L,1);
}
