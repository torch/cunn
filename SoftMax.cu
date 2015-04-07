#include "utils.h"

#define MINUS_LOG_THRESHOLD -18.42
#define SOFTMAX_THREADS 128

__global__ void cunn_SoftMax_updateOutput_kernel(float *output, float *input, int nframe, int dim)
{
  __shared__ float buffer[SOFTMAX_THREADS+1];
  int k = blockIdx.x;
  float *input_k = input + k*dim;
  float *output_k = output + k*dim;

  int i_start = threadIdx.x;
  int i_end = dim;
  int i_step = blockDim.x;

  // max?
  buffer[threadIdx.x] = -FLT_MAX;
  for (int i=i_start; i<i_end; i+=i_step)
  {
    float z = input_k[i];
    if(buffer[threadIdx.x] < z)
      buffer[threadIdx.x] = z;
  }

  __syncthreads();

  // reduce
  if (threadIdx.x == 0)
  {
    float max_k = -FLT_MAX;
    for (int i=0; i<blockDim.x; i++)
    {
      if(max_k < buffer[i])
        max_k = buffer[i];
    }
    buffer[SOFTMAX_THREADS] = max_k;
  }

  __syncthreads();

  // sum?
  float max_k = buffer[SOFTMAX_THREADS];
  buffer[threadIdx.x] = 0;
  for (int i=i_start; i<i_end; i+=i_step) {
    float z = __expf(input_k[i]-max_k);
    buffer[threadIdx.x] += z;
    output_k[i] = z;
  }

  __syncthreads();

  // reduce
  if (threadIdx.x == 0)
  {
    float sum_k = 0;
    for (int i=0; i<blockDim.x; i++)
      sum_k += buffer[i];
    buffer[SOFTMAX_THREADS] = sum_k;
  }

  __syncthreads();

  // softmax
  float sum_k = buffer[SOFTMAX_THREADS];
  for (int i=i_start; i<i_end; i+=i_step)
    output_k[i] = output_k[i] / sum_k;
}


__global__ void cunn_SoftMax_updateGradInput_kernel(float *gradInput, float *output, float *gradOutput, int nframe, int dim)
{
  __shared__ float buffer[SOFTMAX_THREADS];
  int k = blockIdx.x;
  float *gradInput_k = gradInput + k*dim;
  float *output_k = output + k*dim;
  float *gradOutput_k = gradOutput + k*dim;

  int i_start = threadIdx.x;
  int i_end = dim;
  int i_step = blockDim.x;

  // sum?
  buffer[threadIdx.x] = 0;
  for (int i=i_start; i<i_end; i+=i_step)
    buffer[threadIdx.x] += gradOutput_k[i] * output_k[i];

  __syncthreads();

  // reduce
  if (threadIdx.x == 0)
  {
    float sum_k = 0;
    for (int i=0; i<blockDim.x; i++)
      sum_k += buffer[i];
    buffer[0] = sum_k;
  }

  __syncthreads();

  float sum_k = buffer[0];
  for (int i=i_start; i<i_end; i+=i_step)
    gradInput_k[i] = output_k[i] * (gradOutput_k[i] - sum_k);
}

static int cunn_SoftMax_updateOutput(lua_State *L)
{
  THCState *state = getCutorchState(L);
  THCudaTensor *input = (THCudaTensor*)luaT_checkudata(L, 2, "torch.CudaTensor");
  THCudaTensor *output = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "output", "torch.CudaTensor");
  THAssert(THCudaTensor_checkGPU(state, 2, input, output));

  input = THCudaTensor_newContiguous(state, input);
  THCudaTensor_resizeAs(state, output, input);

  if(input->nDimension == 1)
  {
    dim3 blocks(1);
    dim3 threads(SOFTMAX_THREADS);
    cunn_SoftMax_updateOutput_kernel<<<blocks,threads,
      0, THCState_getCurrentStream(state)>>>(THCudaTensor_data(state, output),
                                             THCudaTensor_data(state, input),
                                             1, input->size[0]);
  }
  else if(input->nDimension == 2)
  {
    dim3 blocks(input->size[0]);
    dim3 threads(SOFTMAX_THREADS);
    cunn_SoftMax_updateOutput_kernel<<<blocks,threads,
      0, THCState_getCurrentStream(state)>>>(THCudaTensor_data(state, output),
                                             THCudaTensor_data(state, input),
                                             input->size[0], input->size[1]);
  }
  else
    THError("vector or matrix expected");

  cudaError errcode = cudaGetLastError();
  if(errcode != cudaSuccess)
    THError(cudaGetErrorString(errcode));

  THCudaTensor_free(state, input);
  return 1;
}

struct softmaxupdateGradInput_functor
{
  float value;

  softmaxupdateGradInput_functor(float value_) : value(value_) {}

  __host__ __device__ float operator()(const float& output, const float& gradOutput) const
  {
    return gradOutput - exp(output)*value;
  }
};

static int cunn_SoftMax_updateGradInput(lua_State *L)
{
  THCState *state = getCutorchState(L);
  THCudaTensor *gradOutput = (THCudaTensor*)luaT_checkudata(L, 3, "torch.CudaTensor");
  THCudaTensor *output = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "output", "torch.CudaTensor");
  THCudaTensor *gradInput = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "gradInput", "torch.CudaTensor");
  THAssert(THCudaTensor_checkGPU(state, 3, output, gradOutput, gradInput));

  output = THCudaTensor_newContiguous(state, output);
  gradOutput = THCudaTensor_newContiguous(state, gradOutput);

  THCudaTensor_resizeAs(state, gradInput, output);

  if(gradInput->nDimension == 1)
  {
    dim3 blocks(1);
    dim3 threads(SOFTMAX_THREADS);

    cunn_SoftMax_updateGradInput_kernel<<<blocks,threads,
      0, THCState_getCurrentStream(state)>>>(THCudaTensor_data(state, gradInput),
                                             THCudaTensor_data(state, output),
                                             THCudaTensor_data(state, gradOutput),
                                             1, gradInput->size[0]);
  }
  else if(gradInput->nDimension == 2)
  {
    dim3 blocks(gradInput->size[0]);
    dim3 threads(SOFTMAX_THREADS);

    cunn_SoftMax_updateGradInput_kernel<<<blocks,threads,
      0, THCState_getCurrentStream(state)>>>(THCudaTensor_data(state, gradInput),
                                             THCudaTensor_data(state, output),
                                             THCudaTensor_data(state, gradOutput),
                                             gradInput->size[0], gradInput->size[1]);
  }
  else
    THError("vector or matrix expected");

  cudaError errcode = cudaGetLastError();
  if(errcode != cudaSuccess)
    THError(cudaGetErrorString(errcode));

  THCudaTensor_free(state, gradOutput);
  THCudaTensor_free(state, output);
  return 1;
}

static const struct luaL_Reg cunn_SoftMax__ [] = {
  {"SoftMax_updateOutput", cunn_SoftMax_updateOutput},
  {"SoftMax_updateGradInput", cunn_SoftMax_updateGradInput},
  {NULL, NULL}
};

static void cunn_SoftMax_init(lua_State *L)
{
  luaT_pushmetatable(L, "torch.CudaTensor");
  luaT_registeratname(L, cunn_SoftMax__, "nn");
  lua_pop(L,1);
}
