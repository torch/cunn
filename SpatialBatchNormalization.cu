// Copyright 2004-present Facebook. All Rights Reserved.
// Only compile for arch 3.5 and higher because __shfl_xor
#include "utils.h"

#include <THC/THC.h>
#include <THC/THCDeviceTensor.cuh>
#include <THC/THCDeviceTensorUtils.cuh>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

/**
   Returns the index of the most significant 1 bit in `val`.
*/

__device__ __forceinline__ int getMSB(int val) {
  return
    ((val >= 1024 && val < 2048) ? 10 :
     ((val >= 512) ? 9 :
      ((val >= 256) ? 8 :
       ((val >= 128) ? 7 :
        ((val >= 64) ? 6 :
         ((val >= 32) ? 5 :
          ((val >= 16) ? 4 :
           ((val >= 8) ? 3 :
            ((val >= 4) ? 2 :
             ((val >= 2) ? 1 :
              ((val == 1) ? 0 : -1)))))))))));
}

/**
     Computes floor(a / b)
     */
template <typename T>
__host__ __device__ __forceinline__ T floor(T a, T b) {
    return (a - b + 1) / b;
}

__device__ inline bool inBounds(int y, int x, const THCDeviceTensor<float, 4>& t) {
  // Rely on unsigned integer arithmetic to test both 0 <= and < t.getSize()
  // in one shot.
  return ((unsigned)(y) < (unsigned)(t.getSize(2)) &&
          (unsigned)(x) < (unsigned)(t.getSize(3)));
}

template<typename T, int NumThreads, bool affine, typename ComputeT>
__global__ void SpatialBatchNormalizationUpdateOutputInferenceUnrolled_kernel(
    const THCDeviceTensor<T, 4> input,
    THCDeviceTensor<T, 4> output,
    THCDeviceTensor<T, 1> runningMean,
    THCDeviceTensor<T, 1> runningStddev,
    const THCDeviceTensor<T, 1> weight,
    const THCDeviceTensor<T, 1> bias) {

  //static_assert(std::is_same<ComputeT, double>::value , "type");

  int x = threadIdx.x;
  int y = threadIdx.y;
  int plane = blockIdx.x;
  int batch = blockIdx.y;

  // stddev is actually 1 / stddev
  T stddev = runningStddev[plane].ldg();
  T mean = runningMean[plane].ldg();
  T inp = input[batch][plane][y][x].ldg();
  if (affine) {
    // multiply with gamma and add beta
    // TODO: everyone pulling this, optimize by reusing better
    T beta =  bias[plane].ldg();
    T gamma = weight[plane].ldg();
    output[batch][plane][y][x] = gamma * (inp - mean) * (stddev) + beta;
  } else {
    output[batch][plane][y][x] = (inp - mean) * (stddev);
  }
}

template<typename T, int NumThreads, bool affine, typename ComputeT>
__global__ void SpatialBatchNormalizationUpdateOutputInference_kernel(
    const THCDeviceTensor<T, 4> input,
    THCDeviceTensor<T, 4> output,
    THCDeviceTensor<T, 1> runningMean,
    THCDeviceTensor<T, 1> runningStddev,
    const THCDeviceTensor<T, 1> weight,
    const THCDeviceTensor<T, 1> bias) {

  //static_assert(std::is_same<ComputeT, double>::value , "type");

  int x = threadIdx.x;
  int plane = blockIdx.x;
  int batch = blockIdx.y;

  // stddev is actually 1 / stddev
  T stddev = runningStddev[plane].ldg();
  T mean = runningMean[plane].ldg();
  T beta, gamma;
  if (affine) {
    beta =  bias[plane].ldg();
    gamma = weight[plane].ldg();
  }

  for (int y = threadIdx.y; y < output.getSize(2); y += blockDim.y) {
    T inp = input[batch][plane][y][x].ldg();
    if (affine) {
      // multiply with gamma and add beta
      // TODO: everyone pulling this, optimize by reusing better
      output[batch][plane][y][x] = gamma * (inp - mean) * (stddev) + beta;
    } else {
      output[batch][plane][y][x] = (inp - mean) * (stddev);
    }
  }

}


template<typename T, int NumThreads, bool affine, typename ComputeT>
__global__ void SpatialBatchNormalizationUpdateOutput_kernel(
    const THCDeviceTensor<T, 4> input,
    THCDeviceTensor<T, 4> output,
    THCDeviceTensor<T, 4> centered,
    THCDeviceTensor<T, 1> std,
    THCDeviceTensor<T, 4> normalized,
    THCDeviceTensor<T, 1> runningMean,
    THCDeviceTensor<T, 1> runningStddev,
    const THCDeviceTensor<T, 1> weight,
    const THCDeviceTensor<T, 1> bias,
    T epsilon,
    T momentum) {

  //static_assert(std::is_same<ComputeT, double>::value , "type");

  // Assert powers of 2 for proper intra-warp shuffle reduction
  assert(blockDim.x == NumThreads);
  assert(blockDim.y == NumThreads);
  //static_assert((NumThreads & (NumThreads - 1)) == 0,
  //              "NumThreads must be a power of 2 for proper warp shuffling");
  int plane = blockIdx.x;
  int numBatches = input.getSize(0);

  T norm = (T)0;
  if (threadIdx.y == 0) {
    norm = input.getSize(0) * input.getSize(2) * input.getSize(3);
    norm = (T)1 / norm;
  }

  // 1. Compute the mean across (batch, y, x), save it and update the
  // runningMean with momentum
  T batchMeanGlobal = (T)0;
  for (int y = threadIdx.y; y < input.getSize(2); y += NumThreads) {
    T batchMeanLocal = (T)0;
    for (int batch = 0; batch < numBatches; ++batch) {
      for (int x = threadIdx.x; x < input.getSize(3); x += NumThreads) {
        int inp = (inBounds(y, x, input)) ?
          input[batch][plane][y][x].ldg() : 0.0f;
        batchMeanLocal += inp;
      }
    }
    // Reduce within warp
    for (int i = 0; i < getMSB(NumThreads); ++i) {
      batchMeanLocal += __shfl_xor(batchMeanLocal, 1 << i, NumThreads);
    }
    // thread 0 has it
    batchMeanGlobal += batchMeanLocal;
  }

  __shared__ T shared[NumThreads];
  // thx == 0 stores into smem
  if (threadIdx.x == 0) {
    shared[threadIdx.y] = batchMeanGlobal;
  }

  __syncthreads();
  // 'transpose', and reduce within warp again
  if (threadIdx.y == 0) {
    T batchMeanLocal = shared[threadIdx.x];
    // Reduce within warp again
    for (int i = 0; i < getMSB(NumThreads); ++i) {
      batchMeanLocal += __shfl_xor(batchMeanLocal, 1 << i, NumThreads);
    }
    // We did an allreduce with xors, this should reduce contention on
    // shared memory.
    batchMeanGlobal = batchMeanLocal * norm;
    // Save the non momentum-altered version to share with everyone
    shared[threadIdx.x] = batchMeanGlobal;
  }
  __syncthreads();

  // Everyone picks it up
  batchMeanGlobal = shared[threadIdx.x];
  if (threadIdx.y == 0 && threadIdx.x == 0) {
    // Momentum based writeback
    runningMean[plane] =
      (1 - momentum) * runningMean[plane] + momentum * batchMeanGlobal;
  }


  // 2. Compute the stddev across (batch, y, x),
  //      save it
  //      update the runningStddev with momentum
  //      save a copy
  // All threads have the batchMean now, compute the stddev
  T batchStddevGlobal = (T)0;
  for (int y = threadIdx.y; y < input.getSize(2); y += NumThreads) {
    T batchStddevLocal = (T)0;
    for (int batch = 0; batch < numBatches; ++batch) {
      for (int x = threadIdx.x; x < input.getSize(3); x += NumThreads) {
        T inp = 0.0f;
        if (inBounds(y, x, input)) {
          inp = input[batch][plane][y][x].ldg();
          batchStddevLocal +=
            (inp - batchMeanGlobal) * (inp - batchMeanGlobal);
          centered[batch][plane][y][x] = inp - batchMeanGlobal;
        }
      }
    }
    // Reduce within warp
    for (int i = 0; i < getMSB(NumThreads); ++i) {
      batchStddevLocal += __shfl_xor(batchStddevLocal, 1 << i, NumThreads);
    }
    // thread 0 has it
    batchStddevGlobal += batchStddevLocal;
  }

  // thx == 0 stores into smem, reuse the same smem region, be sure to kill
  // WAR / WAW dependences even if they are extremely unlikely.
  __syncthreads();
  if (threadIdx.x == 0) {
    shared[threadIdx.y] = batchStddevGlobal;
  }

  __syncthreads();
  // 'transpose', and reduce within warp again
  if (threadIdx.y == 0) {
    int batchStddevLocal = shared[threadIdx.x];
    // Reduce within warp again
    for (int i = 0; i < getMSB(NumThreads); ++i) {
      batchStddevLocal += __shfl_xor(batchStddevLocal, 1 << i, NumThreads);
    }
    // We did an allreduce with xors, this should reduce contention on
    // shared memory.
    batchStddevLocal *= norm;
    batchStddevGlobal = 1 / sqrt(batchStddevLocal + epsilon);
    // Save the non momentum-altered version to share with everyone
    shared[threadIdx.x] = batchStddevGlobal;
  }
  __syncthreads();

  // Everyone picks it up
  batchStddevGlobal = shared[threadIdx.x];
  // Momentum based writeback
  if (threadIdx.y == 0 && threadIdx.x == 0) {
    std[plane] = batchStddevGlobal;
    runningStddev[plane] =
      (1 - momentum) * runningStddev[plane] + momentum * batchStddevGlobal;
  }

  // Write normalized and update the output
  T beta =  bias[plane];
  T gamma =  weight[plane];
  for (int y = threadIdx.y; y < input.getSize(2); y += NumThreads) {
    for (int x = threadIdx.x; x < input.getSize(3); x += NumThreads) {
      if(inBounds(y, x, output)) {
        for (int batch = 0; batch < numBatches; ++batch) {
          T inp = input[batch][plane][y][x].ldg();
          normalized[batch][plane][y][x] =
            (inp - batchMeanGlobal) * (batchStddevGlobal);
          if (affine) {
            // multiply with gamma and add beta
            output[batch][plane][y][x] =
              gamma * (inp - batchMeanGlobal) * (batchStddevGlobal) + beta;
          } else {
            output[batch][plane][y][x] =
            (inp - batchMeanGlobal) * (batchStddevGlobal);
          }
        }
      }
    }
  }

}


template<typename T, int BatchDims, int ImageDims, bool train, bool affine, typename ComputeT>
void SpatialBatchNormalizationUpdateOutput(
    THCState *state,
    const THCDeviceTensor<T, BatchDims + ImageDims> input,
    THCDeviceTensor<T, BatchDims + ImageDims> output,
    THCDeviceTensor<T, BatchDims + ImageDims> centered,
    THCDeviceTensor<T, 1> std,
    THCDeviceTensor<T, BatchDims + ImageDims> normalized,
    THCDeviceTensor<T, 1> runningMean,
    THCDeviceTensor<T, 1> runningStddev,
    const THCDeviceTensor<T, 1> weight,
    const THCDeviceTensor<T, 1> bias,
    T epsilon,
    T momentum,
    cudaStream_t s)
{
  //static_assert(BatchDims == 2, "BatchDims == 2 only atm");

  cudaDeviceProp *prop = THCState_getCurrentDeviceProperties(state);
  int maxThreadsPerBlock = prop->maxThreadsPerBlock;     
  
  if (!train) {
    if (input.getSize(3) * input.getSize(2) < maxThreadsPerBlock) {
      dim3 blocks(input.getSize(1), input.getSize(0));
      dim3 threads(input.getSize(3), input.getSize(2));
      SpatialBatchNormalizationUpdateOutputInferenceUnrolled_kernel
        <T, 1, affine, ComputeT>
        <<<blocks, threads, 0, s>>>
        (input, output, runningMean, runningStddev, weight, bias);
    } else {
      dim3 blocks(input.getSize(1),
                  input.getSize(0));
      dim3 threads(input.getSize(3),
                   min(input.getSize(2),
                       floor(maxThreadsPerBlock, input.getSize(3)))
                  );
      SpatialBatchNormalizationUpdateOutputInference_kernel
        <T, 1, affine, ComputeT>
        <<<blocks, threads, 0, s>>>
        (input, output, runningMean, runningStddev, weight, bias);
    }
  } else {
    dim3 blocks(input.getSize(1));
    if (input.getSize(3) >= 16 && input.getSize(2) >= 16) {
      dim3 threads(16, 16);
      SpatialBatchNormalizationUpdateOutput_kernel
        <T, 16, affine, ComputeT>
        <<<blocks, threads, 0, s>>>(input,
                                    output,
                                    centered,
                                    std,
                                    normalized,
                                    runningMean,
                                    runningStddev,
                                    weight,
                                    bias,
                                    epsilon,
                                    momentum);
    } else {
      dim3 threads(8, 8);
      SpatialBatchNormalizationUpdateOutput_kernel
        <T, 8, affine, ComputeT>
        <<<blocks, threads, 0, s>>>(input,
                                    output,
                                    centered,
                                    std,
                                    normalized,
                                    runningMean,
                                    runningStddev,
                                    weight,
                                    bias,
                                    epsilon,
                                    momentum);
    }
  }

}



static int cunn_SpatialBatchNormalization_updateOutput(lua_State *L) {
#if __CUDA_ARCH__ >= 300
  THCState *state = getCutorchState(L);
  THCudaTensor *input = (THCudaTensor*)luaT_checkudata(L, 2, "torch.CudaTensor");

  THCudaTensor *output = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "output", "torch.CudaTensor");
  THCudaTensor *centered = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "centered", "torch.CudaTensor");
  THCudaTensor *std = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "std", "torch.CudaTensor");
  THCudaTensor *normalized = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "normalized", "torch.CudaTensor");
  THCudaTensor *runningMean = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "running_mean", "torch.CudaTensor");
  THCudaTensor *runningStddev = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "running_std", "torch.CudaTensor");
  THCudaTensor *weight = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "weight", "torch.CudaTensor");
  THCudaTensor *bias = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "bias", "torch.CudaTensor");
  float epsilon = luaT_getfieldchecknumber(L, 1, "eps");
  float momentum = luaT_getfieldchecknumber(L, 1, "momentum");
  bool train = luaT_getfieldcheckboolean(L, 1, "train");
  bool affine = luaT_getfieldcheckboolean(L, 1, "affine");

  // The SpatialBatchNormalization lua module is designed for
  // 4-D only: batch, plane, y, x
  const int BatchDims = 2;
  const int ImageDims = 2;
  typedef double ComputeT;
  if (!train) {
    if (!affine) {
      // Collapse
      SpatialBatchNormalizationUpdateOutput
        <float, BatchDims, ImageDims, false, false, ComputeT>
        (
          state,
          toDeviceTensor<float, BatchDims + ImageDims>(state, input),
          toDeviceTensor<float, BatchDims + ImageDims>(state, output),
          THCDeviceTensor<float, BatchDims + ImageDims>(),
          THCDeviceTensor<float, 1>(),
          THCDeviceTensor<float, BatchDims + ImageDims>(),
          toDeviceTensor<float, 1>(state, runningMean),
          toDeviceTensor<float, 1>(state, runningStddev),
          THCDeviceTensor<float, 1>(),
          THCDeviceTensor<float, 1>(),
          epsilon,
          momentum,
          THCState_getCurrentStream(state)
        );
    } else {
      // Collapse
      SpatialBatchNormalizationUpdateOutput
        <float, BatchDims, ImageDims, false, true, ComputeT>
        (
          state,
          toDeviceTensor<float, BatchDims + ImageDims>(state, input),
          toDeviceTensor<float, BatchDims + ImageDims>(state, output),
          THCDeviceTensor<float, BatchDims + ImageDims>(),
          THCDeviceTensor<float, 1>(),
          THCDeviceTensor<float, BatchDims + ImageDims>(),
          toDeviceTensor<float, 1>(state, runningMean),
          toDeviceTensor<float, 1>(state, runningStddev),
          toDeviceTensor<float, 1>(state, weight),
          toDeviceTensor<float, 1>(state, bias),
          epsilon,
          momentum,
          THCState_getCurrentStream(state)
        );
    }
  } else {
    if (!affine) {
      SpatialBatchNormalizationUpdateOutput
        <float, BatchDims, ImageDims, true, false, ComputeT>
      (
        state,
        toDeviceTensor<float, BatchDims + ImageDims>(state, input),
        toDeviceTensor<float, BatchDims + ImageDims>(state, output),
        toDeviceTensor<float, BatchDims + ImageDims>(state, centered),
        toDeviceTensor<float, 1>(state, std),
        toDeviceTensor<float, BatchDims + ImageDims>(state, normalized),
        toDeviceTensor<float, 1>(state, runningMean),
        toDeviceTensor<float, 1>(state, runningStddev),
        THCDeviceTensor<float, 1>(),
        THCDeviceTensor<float, 1>(),
        epsilon,
        momentum,
        THCState_getCurrentStream(state)
      );
    } else {
      SpatialBatchNormalizationUpdateOutput
        <float, BatchDims, ImageDims, true, true, ComputeT>
      (
        state,
        toDeviceTensor<float, BatchDims + ImageDims>(state, input),
        toDeviceTensor<float, BatchDims + ImageDims>(state, output),
        toDeviceTensor<float, BatchDims + ImageDims>(state, centered),
        toDeviceTensor<float, 1>(state, std),
        toDeviceTensor<float, BatchDims + ImageDims>(state, normalized),
        toDeviceTensor<float, 1>(state, runningMean),
        toDeviceTensor<float, 1>(state, runningStddev),
        toDeviceTensor<float, 1>(state, weight),
        toDeviceTensor<float, 1>(state, bias),
        epsilon,
        momentum,
        THCState_getCurrentStream(state)
      );
    }
  }

  THCudaCheck(cudaGetLastError());
#endif

  return 1;
}



template<typename T, int NumThreads, bool affine, typename ComputeT>
__global__ void SpatialBatchNormalizationUpdateGradInput_kernel(
    THCDeviceTensor<T, 4> gradInput,
    const THCDeviceTensor<T, 4> gradOutput,
    THCDeviceTensor<T, 4> centered,
    THCDeviceTensor<T, 1> std,
    const THCDeviceTensor<T, 1> weight) {

  //static_assert(std::is_same<ComputeT, double>::value , "type");

  // Assert powers of 2 for proper intra-warp shuffle reduction
  assert(blockDim.x == NumThreads);
  assert(blockDim.y == NumThreads);
  //static_assert((NumThreads & (NumThreads - 1)) == 0,
  //              "NumThreads must be a power of 2 for proper warp shuffling");
  int plane = blockIdx.x;
  int numBatches = gradInput.getSize(0);

  T norm = (T)0;
  if (threadIdx.y == 0) {
    norm = gradInput.getSize(0) * gradInput.getSize(2) * gradInput.getSize(3);
    norm = (T)1 / norm;
  }

  // 1. Compute means across (batch, y, x)
  T gradMeanGlobal = (T)0;
  T centeredGradMeanGlobal = (T)0;
  for (int y = threadIdx.y; y < gradInput.getSize(2); y += NumThreads) {
    T gradMeanLocal = (T)0;
    T centeredGradMeanLocal = (T)0;
    for (int batch = 0; batch < numBatches; ++batch) {
      for (int x = threadIdx.x; x < gradInput.getSize(3); x += NumThreads) {
        int g = (inBounds(y, x, gradOutput)) ?
          gradOutput[batch][plane][y][x].ldg() : 0.0f;
        int c = (inBounds(y, x, centered)) ?
          centered[batch][plane][y][x].ldg()   : 0.0f;
        gradMeanLocal += g;
        centeredGradMeanLocal += c * g;
      }
    }
    // Reduce within warp
    for (int i = 0; i < getMSB(NumThreads); ++i) {
      gradMeanLocal +=
        __shfl_xor(gradMeanLocal, 1 << i, NumThreads);
      centeredGradMeanLocal +=
        __shfl_xor(centeredGradMeanLocal, 1 << i, NumThreads);
    }
    // thread 0 has it
    gradMeanGlobal += gradMeanLocal;
    centeredGradMeanGlobal += centeredGradMeanLocal;
  }

  __shared__ T shared[2][NumThreads];
  // thx == 0 stores into smem
  if (threadIdx.x == 0) {
    shared[0][threadIdx.y] = gradMeanGlobal;
    shared[1][threadIdx.y] = centeredGradMeanGlobal;
  }

  __syncthreads();
  // 'transpose', and reduce within warp again
  if (threadIdx.y == 0) {
    T gradMeanLocal = shared[0][threadIdx.x];
    T centeredGradMeanLocal = shared[1][threadIdx.x];
    // Reduce within warp again
    for (int i = 0; i < getMSB(NumThreads); ++i) {
      gradMeanLocal +=
        __shfl_xor(gradMeanLocal, 1 << i, NumThreads);
      centeredGradMeanLocal +=
        __shfl_xor(centeredGradMeanLocal, 1 << i, NumThreads);
    }
    // We did an allreduce with xors, this should reduce contention on
    // shared memory.
    gradMeanGlobal = gradMeanLocal * norm;
    centeredGradMeanGlobal = centeredGradMeanLocal * norm;
    // Save the non momentum-altered version to share with everyone
    shared[0][threadIdx.x] = gradMeanGlobal;
    shared[1][threadIdx.x] = centeredGradMeanGlobal;
  }
  __syncthreads();

  // Everyone picks it up, should be broadcast into the whole gradInput
  gradMeanGlobal = shared[0][threadIdx.x];
  centeredGradMeanGlobal = shared[1][threadIdx.x];
  T stdVal = std[plane];
  for (int y = threadIdx.y; y < gradInput.getSize(2); y += NumThreads) {
    for (int batch = 0; batch < numBatches; ++batch) {
      for (int x = threadIdx.x; x < gradInput.getSize(3); x += NumThreads) {
        if (affine) {
          gradInput[batch][plane][y][x] =
            ( - centeredGradMeanGlobal *
                centered[batch][plane][y][x] *
                stdVal *
                stdVal
              +
                gradOutput[batch][plane][y][x]
              -
                gradMeanGlobal
            )
            * stdVal * weight[plane];
        } else {
          gradInput[batch][plane][y][x] =
            ( - centeredGradMeanGlobal *
                centered[batch][plane][y][x] *
                stdVal *
                stdVal
              +
                gradOutput[batch][plane][y][x]
              -
                gradMeanGlobal
            )
            * stdVal;
        }
      }
    }
  }

}



template<typename T, int BatchDims, int ImageDims, bool affine, typename ComputeT>
void SpatialBatchNormalizationUpdateGradInput(
    THCDeviceTensor<T, BatchDims + ImageDims> gradInput,
    const THCDeviceTensor<T, BatchDims + ImageDims> gradOutput,
    THCDeviceTensor<T, BatchDims + ImageDims> centered,
    THCDeviceTensor<T, 1> std,
    const THCDeviceTensor<T, 1> weight,
    cudaStream_t s)
{
  //static_assert(BatchDims == 2, "BatchDims == 2 only atm");

  dim3 blocks(gradInput.getSize(1));
  if (gradInput.getSize(3) >= 16 && gradInput.getSize(2) >= 16) {
    dim3 threads(16, 16);
    SpatialBatchNormalizationUpdateGradInput_kernel
      <T, 16, affine, ComputeT>
      <<<blocks, threads, 0, s>>>(gradInput,
                                  gradOutput,
                                  centered,
                                  std,
                                  weight);
  } else {
    dim3 threads(8, 8);
    SpatialBatchNormalizationUpdateGradInput_kernel
      <T, 8, affine, ComputeT>
      <<<blocks, threads, 0, s>>>(gradInput,
                                  gradOutput,
                                  centered,
                                  std,
                                  weight);
  }

}


static int cunn_SpatialBatchNormalization_updateGradInput(lua_State *L) {
#if __CUDA_ARCH__ >= 300

  THCState *state = getCutorchState(L);
  // Inputs
  THCudaTensor *input = (THCudaTensor *)luaT_checkudata(L, 2, "torch.CudaTensor");
  THCudaTensor *gradOutput = (THCudaTensor *)luaT_checkudata(L, 3, "torch.CudaTensor");

  THCudaTensor *gradInput = (THCudaTensor *)luaT_getfieldcheckudata(L, 1, "gradInput", "torch.CudaTensor");
  THCudaTensor *weight = (THCudaTensor *)luaT_getfieldcheckudata(L, 1, "weight", "torch.CudaTensor");
  THCudaTensor *centered = (THCudaTensor *)luaT_getfieldcheckudata(L, 1, "centered", "torch.CudaTensor");
  THCudaTensor *std = (THCudaTensor *)luaT_getfieldcheckudata(L, 1, "std", "torch.CudaTensor");
  bool affine = luaT_getfieldcheckboolean(L, 1, "affine");

  // The SpatialBatchNormalization lua module is designed for
  // 4-D only: batch, plane, y, x
  const int BatchDims = 2;
  const int ImageDims = 2;
  typedef double ComputeT;
  if (!affine) {
    // Collapse
    SpatialBatchNormalizationUpdateGradInput
      <float, BatchDims, ImageDims, false, ComputeT>
      (
        toDeviceTensor<float, BatchDims + ImageDims>(state, gradInput),
        toDeviceTensor<float, BatchDims + ImageDims>(state, gradOutput),
        toDeviceTensor<float, BatchDims + ImageDims>(state, centered),
        toDeviceTensor<float, 1>(state, std),
        THCDeviceTensor<float, 1>(),
        THCState_getCurrentStream(state)
      );
  } else {
    // Collapse
    SpatialBatchNormalizationUpdateGradInput
      <float, BatchDims, ImageDims, true, ComputeT>
      (
        toDeviceTensor<float, BatchDims + ImageDims>(state, gradInput),
        toDeviceTensor<float, BatchDims + ImageDims>(state, gradOutput),
        toDeviceTensor<float, BatchDims + ImageDims>(state, centered),
        toDeviceTensor<float, 1>(state, std),
        toDeviceTensor<float, 1>(state, weight),
        THCState_getCurrentStream(state)
      );
  }

  THCudaCheck(cudaGetLastError());
#endif

  return 1;
}




template<typename T, int NumThreads, typename ComputeT>
__global__  void SpatialBatchNormalizationAccGradParameters_kernel(
    const THCDeviceTensor<T, 4> gradOutput,
    const THCDeviceTensor<T, 4> normalized,
    THCDeviceTensor<T, 1> gradWeight,
    THCDeviceTensor<T, 1> gradBias,
    T scale)
{

  //static_assert(std::is_same<ComputeT, double>::value , "type");

  // Assert powers of 2 for proper intra-warp shuffle reduction
  assert(blockDim.x == NumThreads);
  assert(blockDim.y == NumThreads);
  //static_assert((NumThreads & (NumThreads - 1)) == 0,
  //              "NumThreads must be a power of 2 for proper warp shuffling");
  int plane = blockIdx.x;
  int numBatches = gradOutput.getSize(0);

  // 1. Compute sums across (batch, y, x)
  T gradMeanGlobal = (T)0;
  T normalizedGradMeanGlobal = (T)0;
  for (int y = threadIdx.y; y < gradOutput.getSize(2); y += NumThreads) {
    T gradMeanLocal = (T)0;
    T normalizedGradMeanLocal = (T)0;
    for (int batch = 0; batch < numBatches; ++batch) {
      for (int x = threadIdx.x; x < gradOutput.getSize(3); x += NumThreads) {
        int g = (inBounds(y, x, gradOutput)) ?
          gradOutput[batch][plane][y][x].ldg() : 0.0f;
        int n = (inBounds(y, x, normalized)) ?
          normalized[batch][plane][y][x].ldg() : 0.0f;
        gradMeanLocal += g;
        normalizedGradMeanLocal += n * g;
      }
    }
    // Reduce within warp
    for (int i = 0; i < getMSB(NumThreads); ++i) {
      gradMeanLocal +=
        __shfl_xor(gradMeanLocal, 1 << i, NumThreads);
      normalizedGradMeanLocal +=
        __shfl_xor(normalizedGradMeanLocal, 1 << i, NumThreads);
    }
    // thread 0 has it
    gradMeanGlobal += gradMeanLocal;
    normalizedGradMeanGlobal += normalizedGradMeanLocal;
  }

  __shared__ T shared[2][NumThreads];
  // thx == 0 stores into smem
  if (threadIdx.x == 0) {
    shared[0][threadIdx.y] = gradMeanGlobal;
    shared[1][threadIdx.y] = normalizedGradMeanGlobal;
  }

  __syncthreads();
  // 'transpose', and reduce within warp again
  if (threadIdx.y == 0) {
    T gradMeanLocal = shared[0][threadIdx.x];
    T normalizedGradMeanLocal = shared[1][threadIdx.x];
    // Reduce within warp again
    for (int i = 0; i < getMSB(NumThreads); ++i) {
      gradMeanLocal +=
        __shfl_xor(gradMeanLocal, 1 << i, NumThreads);
      normalizedGradMeanLocal +=
        __shfl_xor(normalizedGradMeanLocal, 1 << i, NumThreads);
    }
    // We did an allreduce with xors, this should reduce contention on
    // shared memory.
    gradMeanGlobal = gradMeanLocal;
    normalizedGradMeanGlobal = normalizedGradMeanLocal;

    // thread 0 has it
    if (threadIdx.x == 0) {
      gradBias[plane] += scale * gradMeanGlobal;
      gradWeight[plane] += scale * normalizedGradMeanGlobal;
    }
  }
}



template<typename T, int BatchDims, int ImageDims, typename ComputeT>
void SpatialBatchNormalizationAccGradParameters(
    const THCDeviceTensor<T, BatchDims + ImageDims> gradOutput,
    const THCDeviceTensor<T, BatchDims + ImageDims> normalized,
    THCDeviceTensor<T, 1> gradWeight,
    THCDeviceTensor<T, 1> gradBias,
    T scale,
    cudaStream_t s)
{
  //static_assert(BatchDims == 2, "BatchDims == 2 only atm");

  dim3 blocks(gradOutput.getSize(1));
  if (gradOutput.getSize(3) >= 16 && gradOutput.getSize(2) >= 16) {
    dim3 threads(16, 16);
    SpatialBatchNormalizationAccGradParameters_kernel<T, 16, ComputeT>
      <<<blocks, threads, 0, s>>>(gradOutput,
                                  normalized,
                                  gradWeight,
                                  gradBias,
                                  scale);
  } else {
    dim3 threads(8, 8);
    SpatialBatchNormalizationAccGradParameters_kernel<T, 8, ComputeT>
      <<<blocks, threads, 0, s>>>(gradOutput,
                                  normalized,
                                  gradWeight,
                                  gradBias,
                                  scale);
  }

}


static int cunn_SpatialBatchNormalization_accGradParameters(lua_State *L) {
#if __CUDA_ARCH__ >= 300
  THCState *state = getCutorchState(L);
  // Inputs
  THCudaTensor *input = (THCudaTensor *)luaT_checkudata(L, 2, "torch.CudaTensor");
  THCudaTensor *gradOutput = (THCudaTensor *)luaT_checkudata(L, 3, "torch.CudaTensor");
  float scale = luaL_optnumber(L, 4, 1);

  THCudaTensor *normalized = (THCudaTensor *)luaT_getfieldcheckudata(L, 1, "normalized", "torch.CudaTensor");
  THCudaTensor *gradWeight = (THCudaTensor *)luaT_getfieldcheckudata(L, 1, "gradWeight", "torch.CudaTensor");
  THCudaTensor *gradBias = (THCudaTensor *)luaT_getfieldcheckudata(L, 1, "gradBias", "torch.CudaTensor");

  // The SpatialBatchNormalization lua module is designed for
  // 4-D only: batch, plane, y, x
  const int BatchDims = 2;
  const int ImageDims = 2;
  typedef double ComputeT;
  // Collapse
  SpatialBatchNormalizationAccGradParameters
    <float, BatchDims, ImageDims, ComputeT>
    (
      toDeviceTensor<float, BatchDims + ImageDims>(state, gradOutput),
      toDeviceTensor<float, BatchDims + ImageDims>(state, normalized),
      toDeviceTensor<float, 1>(state, gradWeight),
      toDeviceTensor<float, 1>(state, gradBias),
      scale,
      THCState_getCurrentStream(state)
    );

  THCudaCheck(cudaGetLastError());
#endif

  return 1;
}



static const struct luaL_Reg cunn_SpatialBatchNormalization__ [] = {
  {"SpatialBatchNormalization_updateOutput", cunn_SpatialBatchNormalization_updateOutput},
  {"SpatialBatchNormalization_updateGradInput", cunn_SpatialBatchNormalization_updateGradInput},
  {"SpatialBatchNormalization_accGradParameters", cunn_SpatialBatchNormalization_accGradParameters},
  {NULL, NULL}
};

void cunn_SpatialBatchNormalization_init(lua_State *L)
{
  luaT_pushmetatable(L, "torch.CudaTensor");
  luaT_registeratname(L, cunn_SpatialBatchNormalization__, "nn");
  lua_pop(L,1);
}
