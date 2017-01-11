#include "THCUNN.h"
#include "common.h"

#include "THCThrustAllocator.cuh"
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/transform_reduce.h>
#if CUDA_VERSION >= 7000
#include <thrust/system/cuda/execution_policy.h>
#endif
#include <thrust/unique.h>
#include "THCHalf.h"
#include "THCHalfAutoNumerics.cuh"

#ifndef DIVUP
#define DIVUP(x, y) (((x) + (y) - 1) / (y))
#endif

const int WARP_SIZE = 32;

__device__ __forceinline__ bool warpHasCollision(int val)
{
  // Compare our value to the values stored in the next 16 lanes,
  // wrapping around at 32. If any pair of values is the same than
  // there is a collision in the warp.
  bool dup = 0;
  const int laneId = threadIdx.x % 32;

#if __CUDA_ARCH__ >= 300

  #pragma unroll
  for (int i = 1; i <= 16; i++)
  {
    dup |= (__shfl(val, (laneId + i) % 32) == val);
  }

#else

  volatile __shared__ int values[128];
  values[threadIdx.x] = val;
  const int offset = threadIdx.x - laneId;

  #pragma unroll
  for (int i = 1; i <= 16; i++)
  {
    dup |= (values[offset + ((laneId + i) % 32)] == val);
  }

#endif

  return __any(dup) != 0;
}

template <typename Dtype>
__global__ void cunn_LookupTable_accGradParametersKernelByFeature(
  long *input, Dtype *gradOutput, Dtype *gradWeight, Dtype scale, ptrdiff_t numel,
  long stride, int paddingValue) {

  const int featureDim = blockIdx.x * 4 + threadIdx.x / 32;
  if (featureDim >= stride) {
    return;
  }

  // The strategy here is that each warp handles a single feature
  // dimension.
  // Within that feature dimension, points in the [batch][element]
  // dimension can overlap, and we need to determine if threads want
  // to add to the gradient in a colliding manner.
  // Typically one would use floating-point atomicAdd() to resolve
  // these collisions, but that is non-deterministic if there are
  // collisions. Non-determinism for this code is really bad,
  // especially in RNNs, and is prone to snowballing error.
  // In order to get a deterministic order of execution, we handle
  // non-colliding updates separately from colliding ones. Colliding
  // updates are serialized in their order of execution by using the
  // warp-wide collision detector `warpHasCollision`.
  const int laneId = threadIdx.x % 32;
  for (ptrdiff_t i = laneId; i < numel; i += WARP_SIZE) {
    const int weightIndex = (int) (input[i] - TH_INDEX_BASE);
    if (weightIndex == paddingValue - TH_INDEX_BASE) {
      continue;
    }

    Dtype update = gradOutput[i*stride + featureDim] * scale;

    // FIXME: should we accumulate as accreal?
    // Check for collision
    if (warpHasCollision(weightIndex)) {
      // Run all lanes sequentially; warp divergence
      for (int i = 0; i < WARP_SIZE; ++i) {
        if (laneId == i) {
          gradWeight[weightIndex*stride + featureDim] += update;
        }
      }
    } else {
      // No collision; warp coherence
      gradWeight[weightIndex*stride + featureDim] += update;
    }
  }
}

template <typename Dtype, typename Acctype>
__global__ void cunn_LookupTable_accGradParametersKernel(
  long *input, long *indices, Dtype *gradOutput, Dtype *gradWeight,
  long *count, Dtype defaultScale, ptrdiff_t numel, long stride, int paddingValue) {

  int idx = blockIdx.x * 4 + threadIdx.y;

  // Each warp is responsible for an input into the LookupTable.
  // If the preceeding input has the same as this input, then the warp
  // exits immediately. The warp also processes subsequent inputs with the
  // same value.
  //
  // Input Warp
  // 1     <warp 1>
  // 1     <warp 1> (<warp 2> exits without doing any work)
  // 5     <warp 3>
  // 8     <warp 4>

  // Number of values proceessed by each thread (grain size)
  const int SZ = 4;

  if (idx < numel
      && (idx == 0 || input[idx] != input[idx - 1])
      && input[idx] != paddingValue) {
    do {
      const int startFeature = threadIdx.x + blockIdx.y * blockDim.x * SZ;
      const int weightRow = ((int) input[idx] - TH_INDEX_BASE) * stride;
      const int gradOutputRow = ((int) indices[idx] - TH_INDEX_BASE) * stride;
      const Acctype scale = count ? ScalarConvert<Dtype, Acctype>::to(defaultScale) / count[idx] : ScalarConvert<Dtype, Acctype>::to(defaultScale);

      Acctype gradient[SZ];
      Acctype weight[SZ];

      #pragma unroll
      for (int ii = 0; ii < SZ; ii++)
      {
        int featureDim = startFeature + ii * WARP_SIZE;
        if (featureDim < stride)
        {
          gradient[ii] = ScalarConvert<Dtype, Acctype>::to(gradOutput[gradOutputRow + featureDim]);
          weight[ii] = ScalarConvert<Dtype, Acctype>::to(gradWeight[weightRow + featureDim]);
        }
      }

      #pragma unroll
      for (int ii = 0; ii < SZ; ii++)
      {
        weight[ii] += gradient[ii] * scale;
      }

      #pragma unroll
      for (int ii = 0; ii < SZ; ii++)
      {
        int featureDim = startFeature + ii * WARP_SIZE;
        if (featureDim < stride)
        {
          gradWeight[weightRow + featureDim] = ScalarConvert<Acctype, Dtype>::to(weight[ii]);
        }
      }

      idx++;
    } while (idx < numel && input[idx] == input[idx - 1]);
  }
}

/*
 * Keep the norm of weight smaller than maxNorm
 */
template <typename Dtype, typename Acctype>
struct pow_v
{
  Acctype normType;
  pow_v(Dtype v) : normType(ScalarConvert<Dtype, Acctype>::to(v)) {}
  __host__ __device__
  Acctype operator()(const Dtype& x) const {
    Acctype xA = ScalarConvert<Dtype, Acctype>::to(x);
    if (normType == 1)
      return std::abs(xA);
    else if (normType == 2)
      return xA * xA;
    else
      return std::pow(std::abs(xA), normType);
  }
};

template <typename T>
struct multiply_s
{
  T scale;
  multiply_s(T s) : scale(s) {}
  __host__ __device__
  T operator()(const T& x) const {
    return x * scale;
  }
};

#include "generic/LookupTable.cu"
#include "THCGenerateFloatTypes.h"
