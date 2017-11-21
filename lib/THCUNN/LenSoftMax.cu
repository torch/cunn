#include "THCUNN.h"
#include "THCHalf.h"
#include "THCHalfAutoNumerics.cuh"

#define LENSOFTMAX_THREADS 128

template <typename T, typename AccumT, typename IndexT>
__global__ void cunn_LenSoftMax_updateOutput_kernel(
  T *output, T *input, int nframe, int dim, IndexT *len)
{
  __shared__ AccumT buffer[LENSOFTMAX_THREADS+1];
  T *input_k  = input  + blockIdx.x*dim + blockIdx.y + blockIdx.z;
  T *output_k = output + blockIdx.x*dim + blockIdx.y + blockIdx.z;

  int i_start = threadIdx.x;
  int i_end = ScalarConvert<IndexT, int>::to(len[blockIdx.x]);
  int i_step = blockDim.x;

  // max?
  buffer[threadIdx.x] = -THCNumerics<AccumT>::max();
  for (int i=i_start; i<i_end; i+=i_step)
  {
    T z = input_k[i];
    AccumT zAcc = ScalarConvert<T, AccumT>::to(z);
    if (buffer[threadIdx.x] < zAcc)
      buffer[threadIdx.x] = zAcc;
  }


  __syncthreads();

  // reduce
  if (threadIdx.x == 0)
  {
    AccumT max_k = -THCNumerics<AccumT>::max();
    for (int i=0; i<blockDim.x; i++)
    {
      if (max_k < buffer[i])
        max_k = buffer[i];
    }
    buffer[LENSOFTMAX_THREADS] = max_k;
  }

  __syncthreads();

  // sum?
  T max_k = ScalarConvert<AccumT, T>::to(buffer[LENSOFTMAX_THREADS]);
  buffer[threadIdx.x] = ScalarConvert<int, AccumT>::to(0);
  for (int i=i_start; i<i_end; i+=i_step) {
    T z = THCNumerics<T>::exp(input_k[i]-max_k);
    buffer[threadIdx.x] += ScalarConvert<T, AccumT>::to(z);
    output_k[i] = z;
  }
  T vz = ScalarConvert<int, T>::to(0);
  for (int i=i_end; i<dim; i+=i_step) {
    output_k[i] = vz;
  }

  __syncthreads();

  // reduce
  if (threadIdx.x == 0)
  {
    AccumT sum_k = ScalarConvert<int, AccumT>::to(0);
    for (int i=0; i<blockDim.x; i++)
      sum_k += buffer[i];
    buffer[LENSOFTMAX_THREADS] = sum_k;
  }

  __syncthreads();

  // softmax
  T sum_k = ScalarConvert<AccumT, T>::to(buffer[LENSOFTMAX_THREADS]);
  for (int i=i_start; i<i_end; i+=i_step)
    output_k[i] = output_k[i] / sum_k;
}

template <typename T, typename AccumT, typename IndexT>
__global__ void cunn_LenSoftMax_updateGradInput_kernel(
  T *gradInput, T *output, T *gradOutput, int nframe, int dim, IndexT *len)
{
  __shared__ AccumT buffer[LENSOFTMAX_THREADS];
  T *gradInput_k  = gradInput  + blockIdx.x*dim + blockIdx.y + blockIdx.z;
  T *output_k     = output     + blockIdx.x*dim + blockIdx.y + blockIdx.z;
  T *gradOutput_k = gradOutput + blockIdx.x*dim + blockIdx.y + blockIdx.z;

  int i_start = threadIdx.x;
  int i_end = ScalarConvert<IndexT, int>::to(len[blockIdx.x]);
  int i_step = blockDim.x;

  // sum?
  buffer[threadIdx.x] = ScalarConvert<int, AccumT>::to(0);
  for (int i=i_start; i<i_end; i+=i_step)
    buffer[threadIdx.x] += ScalarConvert<T, AccumT>::to(gradOutput_k[i] * output_k[i]);

  __syncthreads();

  // reduce
  if (threadIdx.x == 0)
  {
    AccumT sum_k = ScalarConvert<int, AccumT>::to(0);
    for (int i=0; i<blockDim.x; i++)
      sum_k += buffer[i];
    buffer[0] = sum_k;
  }

  __syncthreads();

  T sum_k = ScalarConvert<AccumT, T>::to(buffer[0]);
  for (int i=i_start; i<i_end; i+=i_step)
    gradInput_k[i] = output_k[i] * (gradOutput_k[i] - sum_k);
  T vz = ScalarConvert<int, T>::to(0);
  for (int i=i_end; i<dim; i+=i_step)
    gradInput_k[i] = vz;
}

#include "generic/LenSoftMax.cu"
#include "THCGenerateFloatTypes.h"

#undef LENSOFTMAX_THREADS
