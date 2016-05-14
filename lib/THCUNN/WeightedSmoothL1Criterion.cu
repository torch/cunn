#include "THCUNN.h"
#include "common.h"

#include <stdio.h>
#include <assert.h>

static const int NTHREADS = 32;

__global__ void cunn_WeightedSmoothL1Criterion_updateOutput_kernel(float *output,
                                                           float *total_weight,
                                                           float *input,
                                                           float *target,
                                                           float *weights,
                                                           int size_average,
                                                           int numel) {
  __shared__ float shInputs[NTHREADS], acc_weight[NTHREADS];
  int i;
  float cur_weight;

  shInputs[threadIdx.x] = 0.0f;
  acc_weight[threadIdx.x] = 0.0f;
  for (i = threadIdx.x; i < numel; i += NTHREADS) {
      cur_weight = weights ? weights[i] : 1.0f;
      float z = fabsf(input[i]-target[i]);
      shInputs[threadIdx.x] += (z < 1.f ? 0.5f*z*z : z - 0.5f) * cur_weight;
      acc_weight[threadIdx.x] += cur_weight;
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    *output = *total_weight = 0;
    for (i = 0; i < NTHREADS; ++i){
      *output += shInputs[i];
      *total_weight += acc_weight[i];
    }
    if (size_average && *total_weight > 0) {
      *output /= *total_weight;
    }
  }
}

__global__ void cunn_WeightedSmoothL1Criterion_updateGradInput_kernel(
  float *gradInput,
  float *input, 
  float *target,
  float *weights,
  float *total_weight,
  int size_average,
  int numel)
{
  if (*total_weight <= 0) {
    return;
  }
  int i;
  float norm = size_average ? (1.0f / *total_weight) : 1.0f;

  for (i = threadIdx.x; i < numel; i += NTHREADS) {
    float z = input[i] - target[i]; 
    if (z < -1.f)
      gradInput[i] = -(weights ? weights[i] : 1.0f) * norm;
    else if (z > 1.f)
      gradInput[i] = (weights ? weights[i] : 1.0f) * norm;
    else
      gradInput[i] = z * (weights ? weights[i] : 1.0f) * norm;
  }
}

void THNN_CudaWeightedSmoothL1Criterion_updateOutput(
    THCState *state, 
    THCudaTensor *input, 
    THCudaTensor *target, 
    THCudaTensor *output, 
    bool sizeAverage, 
    THCudaTensor *weights, 
    THCudaTensor *total_weight) {

  if (weights) {
    THCUNN_assertSameGPU(
      state, 5, input, target, weights, output, total_weight
    );
  } else {
    THCUNN_assertSameGPU(
      state, 4, input, target, output, total_weight
    );
  }

  input = THCudaTensor_newContiguous(state, input);
  weights = weights ? THCudaTensor_newContiguous(state, weights) : NULL;
  target = THCudaTensor_newContiguous(state, target);

  float *input_data = THCudaTensor_data(state, input);
  float *weights_data = weights ? THCudaTensor_data(state, weights) : NULL;
  float *target_data = THCudaTensor_data(state, target);
  float *output_data = THCudaTensor_data(state, output);
  float *total_weight_data = THCudaTensor_data(state, total_weight);

  cunn_WeightedSmoothL1Criterion_updateOutput_kernel
    <<<1, NTHREADS, 0, THCState_getCurrentStream(state)>>>(
      output_data,
      total_weight_data,
      input_data,
      target_data,
      weights_data,
      sizeAverage,
      THCudaTensor_nElement(state, input)
  );

  cudaError errcode = cudaGetLastError();
  if (errcode != cudaSuccess) {
    THError(cudaGetErrorString(errcode));
  }
  if (weights) {
    THCudaTensor_free(state, weights);
  }
  THCudaTensor_free(state, target);
  THCudaTensor_free(state, input);
}

void THNN_CudaWeightedSmoothL1Criterion_updateGradInput(
    THCState *state, 
    THCudaTensor *input, 
    THCudaTensor *target, 
    THCudaTensor *gradInput, 
    bool sizeAverage, 
    THCudaTensor *weights, 
    THCudaTensor *total_weight) {

  THArgCheck(THCudaTensor_isContiguous(state, gradInput), 4, "gradInput must be contiguous");

  if (weights) {
    THCUNN_assertSameGPU(
      state, 5, weights, input, target, gradInput, total_weight
    );
  }
  else {
    THCUNN_assertSameGPU(
      state, 4, input, target, gradInput, total_weight
    );
  }

  input = THCudaTensor_newContiguous(state, input);
  weights = weights ? THCudaTensor_newContiguous(state, weights) : NULL;
  target = THCudaTensor_newContiguous(state, target);

  float *input_data = THCudaTensor_data(state, input);
  float *weights_data = weights ? THCudaTensor_data(state, weights) : NULL;
  float *gradInput_data = THCudaTensor_data(state, gradInput);
  float *target_data = THCudaTensor_data(state, target);
  float *total_weight_data = THCudaTensor_data(state, total_weight);

  cunn_WeightedSmoothL1Criterion_updateGradInput_kernel
    <<<1, NTHREADS, 0, THCState_getCurrentStream(state)>>>(
      gradInput_data,
      input_data,
      target_data,
      weights_data,
      total_weight_data,
      sizeAverage,
      THCudaTensor_nElement(state, input)
  );

  cudaError errcode = cudaGetLastError();
  if (errcode != cudaSuccess) {
    THError(cudaGetErrorString(errcode));
  }
  if (weights) {
    THCudaTensor_free(state, weights);
  }
  THCudaTensor_free(state, target);
  THCudaTensor_free(state, input);
}
