#ifndef __SBN_CUH__
#define __SBN_CUH__

#define MAX_THREADS 1024
#define MAX_CHUNK_SIZE 256

/*
  Optimized reductions based on
  http://devblogs.nvidia.com/parallelforall/faster-parallel-reductions-kepler/
*/

__inline__ __device__
float warpReduceSum(float val, int width) {
  int lane = threadIdx.x % warpSize;

  float ret_val;
  for (unsigned int offset = warpSize/2; offset > 0; offset /= 2) {
    ret_val = __shfl_down(val, offset, width);
    val = lane + offset < width ? val + ret_val : val;
  }
  return val;
}

__inline__ __device__
void multiWarpReduceSum(float val, int width, int numValsToReduce,
                        int chunkSize, float* reducedChunkVals) {

  static __shared__ float shared[32][MAX_CHUNK_SIZE];

  int lane = threadIdx.x % warpSize;
  int wid = threadIdx.x / warpSize;

  int mySpatialId = threadIdx.x / numValsToReduce;

  val = warpReduceSum(val, width);

  // special case here, one warp doing multiple spatial reductions
  // write reduced value to shared memory for each chunk
  if(lane == 0) {
     shared[wid][mySpatialId] = val;
  }
  __syncthreads();

  // at least one warp (for number of vals to reduce < warpSize)
  int warpsInReduction = numValsToReduce/warpSize == 0 ? 1 : numValsToReduce/warpSize;
  bool isEdgeWarp = (wid % warpsInReduction == 0);

  if(isEdgeWarp) {
     val = ( lane < warpsInReduction ) ? shared[lane + wid][mySpatialId] : float(0);
     val = warpReduceSum(val, width);

     if(lane == 0) {
        reducedChunkVals[mySpatialId] = val;
     }
  }
  return;
}

__inline__ __device__
float blockReduceSum(float val, int width, int numThreads) {
  static __shared__ float shared[32];
  int lane = threadIdx.x%warpSize;
  int wid = threadIdx.x/warpSize;
  int warpsInReduction = numThreads/warpSize;
  val = warpReduceSum(val, width);

  __syncthreads();

  //write reduced value to shared memory
  if(lane==0) {
    shared[wid] = val;
  }
  __syncthreads();

  if (numThreads / warpSize == 0) {// special case if number of threads < warp size
    val = shared[wid]; // one warp, already reduced
  } else {
    //ensure we only grab a value from shared memory if that warp existed
    if(wid == 0) {
      val = (lane < warpsInReduction) ? shared[lane] : float(0);
      val = warpReduceSum(val, width);
    }
  }
  return val;
}

/*
  Computes either the sum or mean (divides the sum by nBatch) if divideByNBatch = true
  across the batch dimension.
*/
__global__ void spatialBatchNormalization_batchMean_kernel(float* input, float* mean_output,
                                                           int nBatch, int nFeature, int nSpatial,
                                                           int chunkSize, bool divideByNBatch) {
   int tid = threadIdx.x;

   __shared__ float reducedChunkVals[MAX_CHUNK_SIZE];
   int lane = tid % warpSize;
   int wid = tid / warpSize;
   int width = (nBatch < warpSize) ? nBatch : warpSize;
   int numValsToReduce = (nBatch < warpSize) ? warpSize : nBatch;

   int globalInputIndex = blockIdx.y * nBatch * nSpatial + (blockIdx.x * nBatch * chunkSize) +
                          (lane + wid * width);
   int globalOutputIndex = blockIdx.y * nSpatial + blockIdx.x * chunkSize;

   // if nBatch < 32, need to pad with 0 all elements that are > nBatch
   float sum = lane < nBatch && globalInputIndex < nBatch * nFeature * nSpatial ? input[globalInputIndex] : 0.0f;

   multiWarpReduceSum(sum, width, numValsToReduce, chunkSize, reducedChunkVals);

   __syncthreads();

   if (tid < chunkSize) {
      // normalize if divideByNBatch is true
      float res = divideByNBatch ? reducedChunkVals[tid] * 1.0f / (float)nBatch : reducedChunkVals[tid];

      // write out to output
      mean_output[globalOutputIndex + tid] = res;
   }
}

__global__ void spatialBatchNormalization_spatialAgg_kernel(float* input, float* output,
                                                            int nBatch, int nFeature,
                                                            int nSpatial) {
   int globalTid = blockIdx.x * blockDim.x + threadIdx.x;

   int globalInputIndex = globalTid + blockIdx.y * nSpatial;
   int globalOutputIndex = blockIdx.y;

   // init accumulators to 0.0f
   if (threadIdx.x == 0) {
      if (blockIdx.x == 0) {
         output[globalOutputIndex] = 0.0f;
      }
   }
   __syncthreads();

   float sum = 0.0f;

   if (globalTid < nSpatial) {
      sum = input[globalInputIndex];
      int width = (nSpatial < warpSize) ? nSpatial : warpSize;
      sum = blockReduceSum(sum, width, nSpatial);
   }
   __syncthreads();

   if (threadIdx.x == 0) {
      float ret = atomicAdd(&(output[globalOutputIndex]), sum);
   }
}

__global__ void spatialBatchNormalization_batchVar_kernel(float* input, float* mean,
                                                          float *std,
                                                          int nBatch, int nFeature,
                                                          int nSpatial, int chunkSize) {
   __shared__ float reducedChunkVals[MAX_CHUNK_SIZE];

   int tid = threadIdx.x;
   int lane = tid % warpSize;
   int wid = tid/warpSize;
   int width = (nBatch < warpSize) ? nBatch : warpSize;
   int numValsToReduce = (nBatch < warpSize) ? warpSize : nBatch;

   int globalInputIndex = blockIdx.y * nBatch * nSpatial +
                          (blockIdx.x * nBatch * chunkSize) + (lane + wid * width);
   int globalOutputIndex = blockIdx.y * nSpatial + blockIdx.x * chunkSize;
   int nFeatureIndex = blockIdx.y;

   float diff = lane < nBatch && globalInputIndex < nBatch * nFeature * nSpatial ? input[globalInputIndex] - mean[nFeatureIndex] * (1.0f/(float)nSpatial) : 0.0f;
   float sum = diff * diff;

   multiWarpReduceSum(sum, width, numValsToReduce, chunkSize, reducedChunkVals);

   __syncthreads();

   if (tid < chunkSize) {
      // normalize
      float res = reducedChunkVals[tid] * 1.0f / (float)nBatch;

      // write out to output
      std[globalOutputIndex + tid] = res;
   }
}

__global__ void spatialBatchNormalization_normalizeAffine_kernel(float* input, float* mean,
                                                                 float *std, float *output,
                                                                 float* gamma, float* beta,
                                                                 float* centered,
                                                                 float* normalized,
                                                                 float epsilon,
                                                                 int nBatch, int nFeature,
                                                                 int nSpatial, int chunkSize) {
   int tid = threadIdx.x;
   int lane = tid % warpSize;
   int wid = tid/warpSize;
   int width = (nBatch < warpSize) ? nBatch : warpSize;
   int numValsInWarp = (nBatch < warpSize) ? warpSize : nBatch;

   int globalIndex = blockIdx.y * nBatch * nSpatial +
                     (blockIdx.x * nBatch * chunkSize) + (lane + wid*width);

   int nFeatureIndex = blockIdx.y;

   // read ormalized mean & unnormalized std & normalize it locally
   __shared__ float localNormStd;
   __shared__ float localNormMean;

   if (tid == 0) {
      float normStd = std[nFeatureIndex] * (1.0f/(float)nSpatial);
      localNormStd = 1.0f / sqrt(normStd + epsilon);
      localNormMean = mean[nFeatureIndex] * 1.0f / (float)nSpatial;
   }
   __syncthreads();

   // compute final normalized value
   if (lane < nBatch && tid < numValsInWarp * chunkSize) {
      float diff = input[globalIndex] - localNormMean;
      // write out the centered data for use in updateGradInput
      centered[globalIndex] = diff;
      normalized[globalIndex] = diff * localNormStd;
      output[globalIndex] = gamma[nFeatureIndex] *  diff * localNormStd + beta[nFeatureIndex];
   }
}

__global__ void spatialBatchNormalization_normalizeNoAffine_kernel(float* input, float* mean,
                                                                   float *std, float *output,
                                                                   float* centered,
                                                                   float* normalized,
                                                                   float epsilon,
                                                                   int nBatch, int nFeature,
                                                                   int nSpatial, int chunkSize) {
   int tid = threadIdx.x;
   int lane = tid % warpSize;
   int wid = tid/warpSize;
   int width = (nBatch < warpSize) ? nBatch : warpSize;
   int numValsInWarp = (nBatch < warpSize) ? warpSize : nBatch;

   int globalIndex = blockIdx.y * nBatch * nSpatial +
                     (blockIdx.x * nBatch * chunkSize) + (lane + wid*width);

   int nFeatureIndex = blockIdx.y;

   // read ormalized mean & unnormalized std & normalize it locally
   __shared__ float localNormStd;
   __shared__ float localNormMean;

   if (tid == 0) {
      float normStd = std[nFeatureIndex] * (1.0f/(float)nSpatial);
      localNormStd = 1.0f / sqrt(normStd + epsilon);
      localNormMean = mean[nFeatureIndex] * 1.0f / (float)nSpatial;
   }
   __syncthreads();

   // compute final normalized value
   if (lane < nBatch && tid < numValsInWarp * chunkSize) {
      float diff = input[globalIndex] - localNormMean;
      // write out the centered data for use in updateGradInput
      centered[globalIndex] = diff;
      normalized[globalIndex] = diff * localNormStd;
      output[globalIndex] = diff * localNormStd;
   }
}

__global__ void spatialBatchNormalization_normalizeForwardInfAffine_kernel(float* input,
                                                                           float* mean, float *std,
                                                                           float *weight, float *bias,
                                                                           float *output,
                                                                           int nBatch, int nFeature,
                                                                           int nSpatial, int chunkSize) {
   int tid = threadIdx.x;
   int lane = tid % warpSize;
   int wid = tid/warpSize;
   int width = (nBatch < warpSize) ? nBatch : warpSize;
   int numValsInWarp = (nBatch < warpSize) ? warpSize : nBatch;

   int globalIndex = blockIdx.y * nBatch * nSpatial +
                     (blockIdx.x * nBatch * chunkSize) + (lane + wid*width);

   int nFeatureIndex = blockIdx.y;

   // compute normalized value
   if (lane < nBatch && tid < numValsInWarp * chunkSize) {
      output[globalIndex] =
         weight[nFeatureIndex] * (input[globalIndex] - mean[nFeatureIndex]) * std[nFeatureIndex] + bias[nFeatureIndex];
   }
}

__global__ void spatialBatchNormalization_normalizeForwardInfNoAffine_kernel(float* input,
                                                                            float* mean, float *std,
                                                                            float *output,
                                                                            int nBatch, int nFeature,
                                                                            int nSpatial, int chunkSize) {
   int tid = threadIdx.x;
   int lane = tid % warpSize;
   int wid = tid/warpSize;
   int width = (nBatch < warpSize) ? nBatch : warpSize;
   int numValsInWarp = (nBatch < warpSize) ? warpSize : nBatch;

   int globalIndex = blockIdx.y * nBatch * nSpatial +
                     (blockIdx.x * nBatch * chunkSize) + (lane + wid*width);

   int nFeatureIndex = blockIdx.y;

   // compute normalized value
   if (lane < nBatch && tid < numValsInWarp * chunkSize) {
      output[globalIndex] = (input[globalIndex] - mean[nFeatureIndex]) * std[nFeatureIndex];
   }
}

__global__ void spatialBatchNormalization_normalizeForwardInf_kernel(float* input,
                                                                     float* mean, float *std,
                                                                     float *output,
                                                                     int nBatch, int nFeature,
                                                                     int nSpatial, int chunkSize) {
   int tid = threadIdx.x;
   int lane = tid % warpSize;
   int wid = tid/warpSize;
   int width = (nBatch < warpSize) ? nBatch : warpSize;
   int numValsInWarp = (nBatch < warpSize) ? warpSize : nBatch;

   int globalIndex = blockIdx.y * nBatch * nSpatial +
                     (blockIdx.x * nBatch * chunkSize) + (lane + wid*width);

   int nFeatureIndex = blockIdx.y;

   // compute normalized value
   if (lane < nBatch && tid < numValsInWarp * chunkSize) {
      output[globalIndex] = (input[globalIndex] - mean[nFeatureIndex]) * std[nFeatureIndex];
   }
}

__global__ void spatialBatchNormalization_normalizeMeanStd_kernel(float* mean, float* std,
                                                                  float factor, float epsilon,
                                                                  int nFeature) {
   int tid = threadIdx.x;

   if (tid < nFeature) {
      if (blockIdx.x == 0) {
         mean[tid] = mean[tid] * factor;
      } else {
         float nstd = std[tid] * factor;
         std[tid] = 1.0f / sqrt(nstd + epsilon);
      }
   }
}

__global__ void spatialBatchNormalization_elementWiseMultiply_kernel(float* input1, float* input2,
                                                                     float* output,
                                                                     int nBatch, int nFeature,
                                                                     int nSpatial, int chunkSize) {

   int tid = threadIdx.x;
   int lane = tid % warpSize;
   int wid = tid/warpSize;
   int width = (nBatch < warpSize) ? nBatch : warpSize;
   int numValsToReduce = (nBatch < warpSize) ? warpSize : nBatch;

   int globalIndex = blockIdx.y * nBatch * nSpatial +
                     (blockIdx.x * nBatch * chunkSize) + (lane + wid*width);

   if (lane < nBatch && tid < numValsToReduce * chunkSize) {
      output[globalIndex] = input1[globalIndex] * input2[globalIndex];
   }
}

__global__ void spatialBatchNormalization_normGradInput_kernel(float* centered, float* mean,
                                                               float *std,
                                                               float* gradInput,
                                                               int nBatch, int nFeature,
                                                               int nSpatial,
                                                               int chunkSize) {
   __shared__ float localNormMean;
   __shared__ float localStd;

   int tid = threadIdx.x;
   int lane = tid % warpSize;
   int wid = tid/warpSize;
   int width = (nBatch < warpSize) ? nBatch : warpSize;
   int numValsToReduce = (nBatch < warpSize) ? warpSize : nBatch;

   int globalIndex = blockIdx.y * nBatch * nSpatial +
                     (blockIdx.x * nBatch * chunkSize) + (lane + wid*width);

   int nFeatureIndex = blockIdx.y;

   if (tid == 0) {
      localStd = std[nFeatureIndex];
      localNormMean = mean[nFeatureIndex] * (1.0f/(float)nSpatial);
   }
   __syncthreads();

   if (lane < nBatch && tid < numValsToReduce * chunkSize) {
      gradInput[globalIndex] = centered[globalIndex] * localNormMean * -1 * localStd * localStd;
   }
}

__global__ void spatialBatchNormalization_updateFinalGradInputAffine_kernel(float* gradOutput, float* gradInput,
                                                                            float* mean,
                                                                            float* std,
                                                                            float* weight,
                                                                            int nBatch, int nFeature,
                                                                            int nSpatial,
                                                                            int chunkSize) {
   __shared__ float localNormMean;

   int tid = threadIdx.x;
   int lane = tid % warpSize;
   int wid = tid/warpSize;
   int width = (nBatch < warpSize) ? nBatch : warpSize;
   int numValsToReduce = (nBatch < warpSize) ? warpSize : nBatch;

   int globalIndex = blockIdx.y * nBatch * nSpatial +
                     (blockIdx.x * nBatch * chunkSize) + (lane + wid*width);
   int nFeatureIndex = blockIdx.y;

   if (tid == 0) {
      localNormMean = mean[nFeatureIndex] * (1.0f/(float)nSpatial);
   }
   __syncthreads();

   if (lane < nBatch && tid < numValsToReduce * chunkSize) {
      gradInput[globalIndex] =
        (gradInput[globalIndex] + gradOutput[globalIndex] - localNormMean) * std[nFeatureIndex] * weight[nFeatureIndex];
   }
}

__global__ void spatialBatchNormalization_updateFinalGradInputNoAffine_kernel(float* gradOutput, float* gradInput,
                                                                            float* mean,
                                                                            float* std,
                                                                            int nBatch, int nFeature,
                                                                            int nSpatial,
                                                                            int chunkSize) {
   __shared__ float localNormMean;

   int tid = threadIdx.x;
   int lane = tid % warpSize;
   int wid = tid/warpSize;
   int width = (nBatch < warpSize) ? nBatch : warpSize;
   int numValsToReduce = (nBatch < warpSize) ? warpSize : nBatch;

   int globalIndex = blockIdx.y * nBatch * nSpatial +
                     (blockIdx.x * nBatch * chunkSize) + (lane + wid*width);
   int nFeatureIndex = blockIdx.y;

   if (tid == 0) {
      localNormMean = mean[nFeatureIndex] * (1.0f/(float)nSpatial);
   }
   __syncthreads();

   if (lane < nBatch && tid < numValsToReduce * chunkSize) {
      gradInput[globalIndex] =
        (gradInput[globalIndex] + gradOutput[globalIndex] - localNormMean) * std[nFeatureIndex];
   }
}

__global__ void spatialBatchNormalization_updateWeightBias_kernel(float* weight, float* w_update, float* bias,
                                                                  float* b_update, float scale, int nFeature) {
   int tid = threadIdx.x;

   if (tid < nFeature) {
      if (blockIdx.x == 0) {
         weight[tid] = weight[tid] + w_update[tid] * scale;
      } else {
         bias[tid] = bias[tid] + b_update[tid] * scale;
      }
   }
}

/* Transpose kernels */
__global__ void transposeInputClsdRead_kernel(float* in, float* transp, int nBatch, int nFeature, int nSpatial) {

   int numValsInChunk = blockDim.x;
   int totalSize = nBatch * nFeature * nSpatial;

   int inputIndex = blockIdx.x * nFeature * nSpatial + blockIdx.y * nSpatial + blockIdx.z * numValsInChunk + threadIdx.x;
   int outputIndex = blockIdx.y * nSpatial * nBatch + threadIdx.x * nBatch + blockIdx.z * nBatch * numValsInChunk + blockIdx.x;

   if (outputIndex < totalSize && inputIndex < totalSize) {
     transp[outputIndex] = in[inputIndex];
   }
}

__global__ void transposeInputClsdWrite_kernel(float* in, float* transp, int nBatch, int nFeature, int nSpatial, int batchesInBlock) {

   int totalSize = nBatch * nFeature * nSpatial;

   int tidInBatch = threadIdx.x % nBatch;
   int localSpatialIndex = threadIdx.x / nBatch;

   int inputIndex = tidInBatch * nSpatial * nFeature + blockIdx.y * nSpatial + blockIdx.x * batchesInBlock + localSpatialIndex;
   int outputIndex = blockIdx.y * nSpatial * nBatch + blockIdx.x * nBatch * batchesInBlock + threadIdx.x;

   if (outputIndex < totalSize && inputIndex < totalSize) {
     transp[outputIndex] = in[inputIndex];
   }
}

__global__ void transposeOutput_kernel(float* transp, float* out, int nBatch, int nFeature, int nSpatial, int batchesInBlock) {

   int totalSize = nBatch * nFeature * nSpatial;

   int tidInBatch = threadIdx.x % nBatch;
   int localSpatialIndex = threadIdx.x / nBatch;

   int inputIndex = blockIdx.y * nSpatial * nBatch + blockIdx.x * nBatch * batchesInBlock + threadIdx.x;
   int outputIndex = tidInBatch * nSpatial * nFeature + blockIdx.y * nSpatial + blockIdx.x * batchesInBlock + localSpatialIndex;

   if (outputIndex < totalSize && inputIndex < totalSize) {
     out[outputIndex] = transp[inputIndex];
   }
}

#endif