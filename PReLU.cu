#include "THCApply.cuh"
#include "THCReduce.cuh"
#include "utils.h"

// This is bad, because the following is defined in SpatialConvolution.cu ...
/*
// Use 1024 threads per block, which requires cuda sm_2x or above
const int CUDA_NUM_THREADS = 1024;

// CUDA: number of blocks for threads.
inline int GET_BLOCKS(const int N) {
  return (N + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS;
}

// CUDA: grid stride looping
#define CUDA_KERNEL_LOOP(i, n)                        \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; \
      i < (n);                                       \
      i += blockDim.x * gridDim.x)
*/

struct PReLUUpdateOutput {
  float* weight_;

  PReLUUpdateOutput(float* weight): weight_(weight) {}

  __device__ __forceinline__ void operator()(float* out, float* in) {
    float x = *in;
    *out = (x > 0) ? x : weight_[0] * x;
  }
};

__global__ void preluForward(float *output, const float *input, const float *weight,
    int n, int nOutputPlane, int dim)
{
  CUDA_KERNEL_LOOP(i, n)
  {
    int j = (i / dim) % nOutputPlane;
    output[i] = input[i] > 0 ? input[i] : input[i] * weight[j];
  }
}


static int cunn_PReLU_updateOutput(lua_State *L)
{
  THCState *state = getCutorchState(L);
  // Input:
  THCudaTensor *input = (THCudaTensor*) luaT_checkudata(L, 2, "torch.CudaTensor");
  // Params:
  THCudaTensor *weight = (THCudaTensor*) luaT_getfieldcheckudata(L, 1, "weight", "torch.CudaTensor");
  THCudaTensor *output = (THCudaTensor*) luaT_getfieldcheckudata(L, 1, "output", "torch.CudaTensor");
  long nOutputPlane = luaT_getfieldchecknumber(L, 1, "nOutputPlane");

  THCudaTensor_resizeAs(state, output, input);

  float* w = THCudaTensor_data(state, weight);

  if(nOutputPlane == 0)
    THCudaTensor_pointwiseApply2(state, output, input, PReLUUpdateOutput(w));
  else
  {
    input = THCudaTensor_newContiguous(state, input);

    int n = THCudaTensor_nElement(state, input);
    int dim = n / nOutputPlane / input->size[0];
    preluForward<<<GET_BLOCKS(n), CUDA_NUM_THREADS, 0, THCState_getCurrentStream(state)>>> (
	THCudaTensor_data(state, output),
	THCudaTensor_data(state, input),
	w,
	n, nOutputPlane, dim);
    THCudaTensor_free(state, input);
  }

  return 1;
}

struct PReLUUpdateGradInput {
  float *weight_;

  PReLUUpdateGradInput(float* weight): weight_(weight) {}

  __device__ __forceinline__ void operator()(float* gradInput,
                                             float* gradOutput,
                                             float* input) {
    *gradInput = *input > 0 ? *gradOutput : *gradOutput * *weight_;
  }
};

__global__ void preluBackward(
    float *gradInput,
    const float *input,
    const float *weight,
    const float *gradOutput,
    int n, int nOutputPlane, int dim)
{
  CUDA_KERNEL_LOOP(i, n)
  {
    int j = (i / dim) % nOutputPlane;
    gradInput[i] = input[i] > 0 ? gradOutput[i] : gradOutput[i] * weight[j];
  }
}

static int cunn_PReLU_updateGradInput(lua_State *L)
{
  THCState *state = getCutorchState(L);
  // Input:
  THCudaTensor *input = (THCudaTensor*)luaT_checkudata(L, 2, "torch.CudaTensor");
  THCudaTensor *gradOutput = (THCudaTensor*)luaT_checkudata(L, 3, "torch.CudaTensor");
  // Params:
  THCudaTensor *weight = (THCudaTensor*) luaT_getfieldcheckudata(L, 1, "weight", "torch.CudaTensor");
  THCudaTensor *gradInput = (THCudaTensor*) luaT_getfieldcheckudata(L, 1, "gradInput", "torch.CudaTensor");
  long nOutputPlane = luaT_getfieldchecknumber(L, 1, "nOutputPlane");

  THCudaTensor_resizeAs(state, gradInput, input);

  float* w = THCudaTensor_data(state, weight);
  if(nOutputPlane == 0)
    THCudaTensor_pointwiseApply3(state, gradInput, gradOutput, input, PReLUUpdateGradInput(w));
  else
  {
    input = THCudaTensor_newContiguous(state, input);
    gradOutput = THCudaTensor_newContiguous(state, gradOutput);

    int n = THCudaTensor_nElement(state, input);
    int dim = n / nOutputPlane / input->size[0];
    preluBackward<<<GET_BLOCKS(n), CUDA_NUM_THREADS, 0, THCState_getCurrentStream(state)>>> (
	THCudaTensor_data(state, gradInput),
	THCudaTensor_data(state, input),
	w,
	THCudaTensor_data(state, gradOutput),
	n, nOutputPlane, dim);

    THCudaTensor_free(state, input);
    THCudaTensor_free(state, gradOutput);
  }
  return 1;
}

struct PReLUAccGradParameters {
  float scale;
  PReLUAccGradParameters(float scale) : scale(scale) {}

  __device__ __forceinline__ void operator()(float* gradInput,
      					     float* input,
                                             float* gradOutput) {
    if ((*input) <= 0)
    {
      *gradInput = (*input) * (*gradOutput) * scale;
    }
  }
};

static int cunn_PReLU_accGradParameters(lua_State *L)
{
  THCState *state = getCutorchState(L);
  // Input:
  THCudaTensor *input = (THCudaTensor*)luaT_checkudata(L, 2, "torch.CudaTensor");
  THCudaTensor *gradOutput = (THCudaTensor*)luaT_checkudata(L, 3, "torch.CudaTensor");
  // Params:
  THCudaTensor *weight = (THCudaTensor*) luaT_getfieldcheckudata(L, 1, "weight", "torch.CudaTensor");
  THCudaTensor *gradWeight = (THCudaTensor*) luaT_getfieldcheckudata(L, 1, "gradWeight", "torch.CudaTensor");
  THCudaTensor *gradInput = (THCudaTensor*) luaT_getfieldcheckudata(L, 1, "gradInput", "torch.CudaTensor");
  long nOutputPlane = luaT_getfieldchecknumber(L, 1, "nOutputPlane");
  float scale = luaL_optnumber(L, 4, 1);

  // use grad input for temporary storage, then call updateGradInput again
  THCudaTensor_pointwiseApply3(state, gradInput, input, gradOutput, PReLUAccGradParameters(scale));

  if(nOutputPlane == 0)
  {
    // introduces a sync point
    float sum = THCudaTensor_sumall(state, gradInput);
    THCudaTensor_set1d(state, gradWeight, 0, sum);
  }
  else
  {
    int input_ndim = THCudaTensor_nDimension(state, input);
    int reduce_dim = (input_ndim - 1) % 2;
    THCudaTensor_reduceDim(state, gradOutput, gradInput, thrust::identity<float>(), thrust::plus<float>(), 0, reduce_dim);
  }

  // restore gradInput
  cunn_PReLU_updateGradInput(L);
  return 1;
}

static const struct luaL_Reg cunn_PRelu__ [] = {
  {"PReLU_updateOutput", cunn_PReLU_updateOutput},
  {"PReLU_updateGradInput", cunn_PReLU_updateGradInput},
  {"PReLU_accGradParameters", cunn_PReLU_accGradParameters},
  {NULL, NULL}
};

static void cunn_PReLU_init(lua_State *L)
{
  luaT_pushmetatable(L, "torch.CudaTensor");
  luaT_registeratname(L, cunn_PRelu__, "nn");
  lua_pop(L,1);
}
