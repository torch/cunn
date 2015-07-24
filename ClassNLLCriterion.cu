#include "utils.h"
#include <assert.h>

static const int NTHREADS = 32;

__global__ void cunn_ClassNLLCriterion_updateOutput_kernel1(float *output,
                                                            float *input,
                                                            float *target,
                                                            int ntarget,
                                                            float *weights,
                                                            bool apply_weights) {
  assert(threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0);

  // TODO: T4951791 Reuse code between updateOutput_kernel1 and
  // updateOutput_kernel.
  // Verify whether `register` does anything here.
  register int i, t;
  for (i = 0; i < ntarget; i++) {
    t = target[i] - 1;
    if (t >= 0) {
      if (apply_weights) {
        *output = -(input[t] * weights[t]);
      } else {
        *output = -input[t];
      }
    }
  }
}

__global__ void cunn_ClassNLLCriterion_updateOutput_kernel(float *output,
                                                           float *input,
                                                           float *target,
                                                           int nframe,
                                                           int ndim,
                                                           int sizeAverage,
                                                           int ntarget,
                                                           float* weights,
                                                           bool apply_weights) {
  __shared__ float shInputs[NTHREADS];
  // Verify whether `register` does anything here.
  register int i, j, t;

  shInputs[threadIdx.x] = .0;
  for (i = threadIdx.x; i < nframe; i += NTHREADS) {
    for (j = 0; j < ntarget; ++j) {
      t = (int)target[i * ntarget + j] - 1;
      if (t >= 0) {
        if (apply_weights) {
          shInputs[threadIdx.x] += (input[i * ndim + t] * weights[t]);
        } else {
          shInputs[threadIdx.x] += input[i * ndim + t];
        }
      }
    }
  }
  __syncthreads();

  // TODO: T4951791 Reuse code between updateOutput_kernel1 and
  // updateOutput_kernel
  if (threadIdx.x == 0) {
    *output = .0;
    for (i = 0; i < NTHREADS; ++i)
      *output += shInputs[i];
    if (sizeAverage)
      *output /= nframe;
    *output = -(*output);
  }
}

__global__ void cunn_ClassNLLCriterion_updateGradInput_kernel(float *gradInput,
                                                              float *target,
                                                              int nframe,
                                                              int ndim,
                                                              float grad,
                                                              int ntarget,
                                                              float* weights,
                                                              bool apply_weights) {
  register int i, j, t;
  for (i = threadIdx.x; i < nframe; i += NTHREADS) {
    for (j = 0; j < ntarget; ++j) {
      t = (int)target[i * ntarget + j] - 1;
      if (t >= 0) {
        if (apply_weights) {
          gradInput[i * ndim + t] = grad * weights[t];
        } else {
          gradInput[i * ndim + t] = grad;
        }
      }
    }
  }
}

static int cunn_ClassNLLCriterion_updateOutput(lua_State *L) {
  THCState *state = getCutorchState(L);
  THCudaTensor *input =
    (THCudaTensor *)luaT_checkudata(L, 2, "torch.CudaTensor");
  THCudaTensor *target =
      (THCudaTensor *)luaT_checkudata(L, 3, "torch.CudaTensor");
  THCudaTensor *output = (THCudaTensor *)luaT_getfieldcheckudata(
    L, 1, "outputTensor", "torch.CudaTensor");
  THCudaTensor *weights = (THCudaTensor *)luaT_getfieldcheckudata(
      L, 1, "weights", "torch.CudaTensor");

  THAssert(THCudaTensor_checkGPU(state, 3, input, target, output));
  input = THCudaTensor_newContiguous(state, input);
  float *input_data = THCudaTensor_data(state, input);

  target = THCudaTensor_newContiguous(state, target);
  float *target_data = THCudaTensor_data(state, target);
  int ntarget = 1;
  if (target->nDimension > 1)
    ntarget = target->size[1];

  output = THCudaTensor_newContiguous(state, output);
  float *output_data = THCudaTensor_data(state, output);

  const bool apply_weights = weights->nDimension > 0;
  float* weights_data = NULL;
  if (apply_weights) {
    weights = THCudaTensor_newContiguous(state, weights);
    weights_data = THCudaTensor_data(state, weights);
  }

  if (input->nDimension == 1) {
    cunn_ClassNLLCriterion_updateOutput_kernel1 <<<1, 1,
      0, THCState_getCurrentStream(state)>>>
        (output_data, input_data, target_data, ntarget, weights_data, 
         apply_weights);
  } else if (input->nDimension == 2) {
    dim3 blocks(1);
    dim3 threads(NTHREADS);
    int sizeAverage = luaT_getfieldcheckboolean(L, 1, "sizeAverage");
    cunn_ClassNLLCriterion_updateOutput_kernel <<<blocks, threads,
      0, THCState_getCurrentStream(state)>>>
        (output_data,
         input_data,
         target_data,
         input->size[0],
         input->size[1],
         sizeAverage,
         ntarget,
         weights_data,
         apply_weights);
  } else
    THArgCheck(0, 2, "vector or matrix expected");

  cudaError errcode = cudaGetLastError();
  if (errcode != cudaSuccess)
    THError(cudaGetErrorString(errcode));

  THCudaTensor_free(state, output);
  THCudaTensor_free(state, target);
  THCudaTensor_free(state, input);
  if (apply_weights) {
    THCudaTensor_free(state, weights);
  }

  return 1;
}

static int cunn_ClassNLLCriterion_updateGradInput(lua_State *L) {
  THCState *state = getCutorchState(L);

  THCudaTensor *input =
      (THCudaTensor *)luaT_checkudata(L, 2, "torch.CudaTensor");
  THCudaTensor *target =
    (THCudaTensor *)luaT_checkudata(L, 3, "torch.CudaTensor");
  THCudaTensor *gradInput = (THCudaTensor *)luaT_getfieldcheckudata(
      L, 1, "gradInput", "torch.CudaTensor");
  THCudaTensor *weights = (THCudaTensor *)luaT_getfieldcheckudata(
        L, 1, "weights", "torch.CudaTensor");
  THAssert(THCudaTensor_checkGPU(state, 3, input, target, gradInput));

  input = THCudaTensor_newContiguous(state, input);
  target = THCudaTensor_newContiguous(state, target);
  gradInput = THCudaTensor_newContiguous(state, gradInput);

  float *target_data = THCudaTensor_data(state, target);
  int ntarget = 1;
  if (target->nDimension > 1)
    ntarget = target->size[1];

  float *gradInput_data = THCudaTensor_data(state, gradInput);

  const bool apply_weights = weights->nDimension > 0;
  float* weights_data = NULL;
  if (apply_weights) {
    weights = THCudaTensor_newContiguous(state, weights);
    weights_data = THCudaTensor_data(state, weights);
  }

  float grad = -1.0;
  if (input->nDimension == 1) {
    // TODO(tompson): Fix this
    if (ntarget > 1)
      THArgCheck(0, 2, "multi-target not implemented");
    float tid;
    cudaMemcpy(&tid, target_data, sizeof(float), cudaMemcpyDeviceToHost);
    if (apply_weights) {
      float weight;
      cudaMemcpy(&weight, weights_data + (int)tid - 1, sizeof(float), 
                 cudaMemcpyDeviceToHost);
      grad *= weight;
    }
    cudaMemcpy(gradInput_data + (int)tid - 1,
               &grad,
               sizeof(float),
               cudaMemcpyHostToDevice);
  } else if (input->nDimension == 2) {
    int nframe = input->size[0];
    int ndim = input->size[1];
    int sizeAverage = luaT_getfieldcheckboolean(L, 1, "sizeAverage");
    if (sizeAverage)
      grad /= nframe;
    dim3 blocks(1);
    dim3 threads(NTHREADS);
    cunn_ClassNLLCriterion_updateGradInput_kernel <<<blocks, threads,
      0, THCState_getCurrentStream(state)>>>
      (gradInput_data, target_data, nframe, ndim, grad, ntarget, weights_data, 
       apply_weights);
  } else
    THArgCheck(0, 2, "vector or matrix expected");

  cudaError errcode = cudaGetLastError();
  if (errcode != cudaSuccess)
    THError(cudaGetErrorString(errcode));

  THCudaTensor_free(state, gradInput);
  THCudaTensor_free(state, target);
  THCudaTensor_free(state, input);
  if (apply_weights) {
    THCudaTensor_free(state, weights);
  }

  return 1;
}

static const struct luaL_Reg cunn_ClassNLLCriterion__[] = {
    {"ClassNLLCriterion_updateOutput", cunn_ClassNLLCriterion_updateOutput},
    {"ClassNLLCriterion_updateGradInput",
     cunn_ClassNLLCriterion_updateGradInput},
    {NULL, NULL}};

void cunn_ClassNLLCriterion_init(lua_State *L) {
  luaT_pushmetatable(L, "torch.CudaTensor");
  luaT_registeratname(L, cunn_ClassNLLCriterion__, "nn");
  lua_pop(L, 1);
}
