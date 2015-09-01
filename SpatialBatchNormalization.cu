#include <stdio.h>
#include <sys/time.h>

#include "SpatialBatchNormalization.cuh"

/*
  NOTE: This correctly handles only 3D/4D tensors with:
  1. nBatch <= 32, and powers of two >=32 (64, 128, 256, 512, 1024)
  2. nFeature any value <= 1024
  3. nSpatial (iH x iW) in powers of two

  This check is performed in nn/SpatialBatchNormalization.lua
*/

/* Helpers */
inline int getSpatialChunkSize(int nBatch, int nSpatial) {
   int threadsPerBatchDim = nBatch >= WARP_SIZE ? nBatch : WARP_SIZE;

   return MAX_THREADS/threadsPerBatchDim > nSpatial ? nSpatial : MAX_THREADS/threadsPerBatchDim;
}

inline int getNumSpatialBlocks(int nSpatial) {
   return nSpatial/MAX_THREADS == 0 ? 1 : nSpatial/MAX_THREADS;
}

static void cunn_SpatialBatchNormalization_transposeInput(THCState *state, THCudaTensor *input,
                                                          THCudaTensor *transpose, int nBatch, int nFeature,
                                                          int nSpatial) {
   /*
     Transpose

     Two cases for input transpose:
      - use coalesced write implementation for small spatial dim (< MAX_THREADS)
      - use coalesced read implementation optimized for large spatial dimensions,
        (uses all threads in a block)
   */
   int batchesInBlockTransp = MAX_THREADS/nBatch;
   int spatialBlocksTransp = nSpatial/batchesInBlockTransp == 0 ? 1 : nSpatial/batchesInBlockTransp;

   dim3 dimBlockTransp(MAX_THREADS, 1);
   dim3 dimGridTransp(spatialBlocksTransp, nFeature);

   if (nSpatial < MAX_THREADS) {
      transposeInputClsdWrite_kernel<<<dimGridTransp, dimBlockTransp>>>(THCudaTensor_data(state, input),
                                                                        THCudaTensor_data(state, transpose),
                                                                        nBatch, nFeature, nSpatial,
                                                                        batchesInBlockTransp);
   } else {
      spatialBlocksTransp = getNumSpatialBlocks(nSpatial);
      dim3 dimBlockTransp2(MAX_THREADS, 1);
      dim3 dimGridTransp2(nBatch, nFeature, spatialBlocksTransp);

      transposeInputClsdRead_kernel<<<dimGridTransp2, dimBlockTransp2>>>(THCudaTensor_data(state, input),
                                                                         THCudaTensor_data(state, transpose),
                                                                         nBatch, nFeature, nSpatial);
   }

   cudaError errcode = cudaGetLastError();
   if(errcode != cudaSuccess)
     THError(cudaGetErrorString(errcode));

   return;
 }

 static void cunn_SpatialBatchNormalization_transposeOutput(THCState *state, THCudaTensor *transposed,
                                                           THCudaTensor *output, int nBatch, int nFeature,
                                                           int nSpatial) {

   int batchesInBlockTransp = MAX_THREADS/nBatch;
   int spatialBlocksTransp = nSpatial/batchesInBlockTransp == 0 ? 1 : nSpatial/batchesInBlockTransp;
   // remainder
   spatialBlocksTransp = nSpatial % batchesInBlockTransp == 0 ? spatialBlocksTransp : spatialBlocksTransp + 1;

   dim3 dimBlockTransp(MAX_THREADS, 1);
   dim3 dimGridTransp(spatialBlocksTransp, nFeature);

   transposeOutput_kernel<<<dimGridTransp, dimBlockTransp>>>(THCudaTensor_data(state, transposed),
                                                             THCudaTensor_data(state, output),
                                                             nBatch, nFeature, nSpatial, batchesInBlockTransp);
   cudaError errcode = cudaGetLastError();
   if(errcode != cudaSuccess)
     THError(cudaGetErrorString(errcode));

   return;
}

/*
  NOTE: This correctly handles only 3D/4D tensors with:
  1. nBatch <= 32, and powers of two >=32 (64, 128, 256, 512, 1024)
  2. nFeature any value <= 1024
  3. nSpatial (iH x iW) in powers of two
*/

static int cunn_SpatialBatchNormalization_forwardInferenceAffine(lua_State *L)
{
   THCState *state = getCutorchState(L);

   // Params:
   int nBatch = (int)luaL_checknumber(L, 3);
   int nFeature = (int)luaL_checknumber(L, 4);
   int nSpatial = (int)luaL_checknumber(L, 5);

   // Input
   THCudaTensor *input = (THCudaTensor*)luaT_checkudata(L, 2, "torch.CudaTensor");

   // Pre-allocated buffers for the transposes
   THCudaTensor *transposedInput = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "transposedInput", "torch.CudaTensor");
   THCudaTensor *transposedOutput = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "transposedOutput", "torch.CudaTensor");

   // Output
   THCudaTensor *output = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "output", "torch.CudaTensor");

   THCudaTensor *running_mean = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "running_mean", "torch.CudaTensor");
   THCudaTensor *running_std = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "running_std", "torch.CudaTensor");
   THCudaTensor *weight = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "weight", "torch.CudaTensor");
   THCudaTensor *bias = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "bias", "torch.CudaTensor");

   // assert
   THAssert(THCudaTensor_checkGPU(state, 7, transposedInput, transposedOutput, output, weight, bias, running_mean, running_std));
   luaL_argcheck(L, input->nDimension == 3 || input->nDimension == 4, 2, "3D or 4D (batch mode) tensor is expected");

   /* Transpose input */
   cunn_SpatialBatchNormalization_transposeInput(state, input, transposedInput, nBatch, nFeature, nSpatial);

   /*
    -----
    Block/thread dimensions for [nBatch x nFeature x nSpatial]
    -----
    blockIdx.x - gets a chunk of nSpatial dimension, depending on MAX_THREADS and the nBatch paramter.
                 That is, each blockIdx.x handles as many nSpatial features as fits in its MAX_TREADS.
                 If nBatch < 32, zero-pad to 32 so that one warp handles all batches in one spatial dim.
    blockIdx.y - over nFeature dimension
    threadIdx.x - over nBatch dimension

    For reductions, each block performs a local reduction of its values, across all the warps that span
    the nBatch dimension. It then writes out the reduced (and normalized by nBatch) value to output[blockIdx.y].
   */
   int chunkSize = getSpatialChunkSize(nBatch, nSpatial);
   int numSpatialBlocks = nSpatial/chunkSize;

   dim3 dimBlockBatch(MAX_THREADS, 1);
   dim3 dimGridBatch(numSpatialBlocks, nFeature);

   /*
     in: transposedInput, running_mean, running_std
     out: transposedOutput

     Normalize the transposedInput based on running statistics.

     transposedOutput[globalIndex] = (transposedInput[i] - mean) * std
   */
   spatialBatchNormalization_normalizeForwardInfAffine_kernel<<<dimGridBatch, dimBlockBatch>>>(THCudaTensor_data(state, transposedInput),
                                                                                               THCudaTensor_data(state, running_mean),
                                                                                               THCudaTensor_data(state, running_std),
                                                                                               THCudaTensor_data(state, weight),
                                                                                               THCudaTensor_data(state, bias),
                                                                                               THCudaTensor_data(state, transposedOutput),
                                                                                               nBatch, nFeature, nSpatial,
                                                                                               chunkSize);
   /* Transpose output */
   cunn_SpatialBatchNormalization_transposeOutput(state, transposedOutput, output, nBatch, nFeature, nSpatial);

   cudaError errcode = cudaGetLastError();
   if(errcode != cudaSuccess)
     THError(cudaGetErrorString(errcode));

   return 1;
}

static int cunn_SpatialBatchNormalization_forwardInference(lua_State *L)
{
   THCState *state = getCutorchState(L);

   // Params:
   int nBatch = (int)luaL_checknumber(L, 3);
   int nFeature = (int)luaL_checknumber(L, 4);
   int nSpatial = (int)luaL_checknumber(L, 5);

   // Input
   THCudaTensor *input = (THCudaTensor*)luaT_checkudata(L, 2, "torch.CudaTensor");

   // Pre-allocated buffers for the transposes
   THCudaTensor *transposedInput = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "transposedInput", "torch.CudaTensor");
   THCudaTensor *transposedOutput = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "transposedOutput", "torch.CudaTensor");

   // Output
   THCudaTensor *output = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "output", "torch.CudaTensor");

   THCudaTensor *running_mean = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "running_mean", "torch.CudaTensor");
   THCudaTensor *running_std = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "running_std", "torch.CudaTensor");

   // assert
   THAssert(THCudaTensor_checkGPU(state, 5, transposedInput, transposedOutput, output, running_mean, running_std));
   luaL_argcheck(L, input->nDimension == 3 || input->nDimension == 4, 2, "3D or 4D (batch mode) tensor is expected");

   /* Transpose input */
   cunn_SpatialBatchNormalization_transposeInput(state, input, transposedInput, nBatch, nFeature, nSpatial);

   /*
    -----
    Block/thread dimensions for [nBatch x nFeature x nSpatial]
    -----
    blockIdx.x - gets a chunk of nSpatial dimension, depending on MAX_THREADS and the nBatch paramter.
                 That is, each blockIdx.x handles as many nSpatial features as fits in its MAX_TREADS.
                 If nBatch < 32, zero-pad to 32 so that one warp handles all batches in one spatial dim.
    blockIdx.y - over nFeature dimension
    threadIdx.x - over nBatch dimension

    For reductions, each block performs a local reduction of its values, across all the warps that span
    the nBatch dimension. It then writes out the reduced (and normalized by nBatch) value to output[blockIdx.y].
   */
   int chunkSize = getSpatialChunkSize(nBatch, nSpatial);
   int numSpatialBlocks = nSpatial/chunkSize;

   dim3 dimBlockBatch(MAX_THREADS, 1);
   dim3 dimGridBatch(numSpatialBlocks, nFeature);

   /*
     in: transposedInput, running_mean, running_std
     out: transposedOutput

     Normalize the transposedInput based on running statistics.

     transposedOutput[globalIndex] = (transposedInput[i] - mean) * std
   */
   spatialBatchNormalization_normalizeForwardInfNoAffine_kernel<<<dimGridBatch, dimBlockBatch>>>(THCudaTensor_data(state, transposedInput),
                                                                                                 THCudaTensor_data(state, running_mean),
                                                                                                 THCudaTensor_data(state, running_std),
                                                                                                 THCudaTensor_data(state, transposedOutput),
                                                                                                 nBatch, nFeature, nSpatial,
                                                                                                 chunkSize);
   /* Transpose output */
   cunn_SpatialBatchNormalization_transposeOutput(state, transposedOutput, output, nBatch, nFeature, nSpatial);

   cudaError errcode = cudaGetLastError();
   if(errcode != cudaSuccess)
     THError(cudaGetErrorString(errcode));

   return 1;
}


static void cunn_SpatialBatchNormalization_updateOutputComputeStats(THCState *state,
                                                                    THCudaTensor *input,
                                                                    THCudaTensor *transposedInput,
                                                                    THCudaTensor *batchAgg,
                                                                    THCudaTensor *meanBuffer,
                                                                    THCudaTensor *stdBuffer,
                                                                    THCudaTensor *centered,
                                                                    THCudaTensor *normalized,
                                                                    int nBatch, int nFeature,
                                                                    int nSpatial) {
   /* Transpose input */
   cunn_SpatialBatchNormalization_transposeInput(state, input, transposedInput, nBatch, nFeature, nSpatial);

   /*
    -----
    Block/thread dimensions for [nBatch x nFeature x nSpatial]
    -----
    blockIdx.x - gets a chunk of nSpatial dimension, depending on MAX_THREADS and the nBatch paramter.
                 That is, each blockIdx.x handles as many nSpatial features as fits in its MAX_TREADS.
                 If nBatch < 32, zero-pad to 32 so that one warp handles all batches in one spatial dim.
    blockIdx.y - over nFeature dimension
    threadIdx.x - over nBatch dimension

    For reductions, each block performs a local reduction of its values, across all the warps that span
    the nBatch dimension. It then writes out the reduced (and normalized by nBatch) value to output[blockIdx.y].
   */
   int chunkSize = getSpatialChunkSize(nBatch, nSpatial);
   int numSpatialBlocks = nSpatial/chunkSize;

   dim3 dimBlockBatch(MAX_THREADS, 1);
   dim3 dimGridBatch(numSpatialBlocks, nFeature);

   /*
    -----
    Block/thread dimensions for [nFeature x nSpatial]
    -----
    blockIdx.x - chunk of nSpatial dimension (chunk size bound by MAX_THREADS)
    blockIdx.y - over nFeature dimension
    threadIdx.x - over nSpatial dimension

    In reduction, first all blocks do a local reduction of their chunk, then we reduce across blocks
    using atomic operations.
   */
   int spatialBlocks = getNumSpatialBlocks(nSpatial);

   dim3 dimBlockSpatial(MAX_THREADS, 1);
   dim3 dimGridSpatial(spatialBlocks, nFeature);

   /*
     in: transposedInput
     out: batchAgg

     Compute mean across nBatch dimension of transposedInput.
   */
   spatialBatchNormalization_batchMean_kernel<<<dimGridBatch, dimBlockBatch>>>(THCudaTensor_data(state, transposedInput),
                                                                               THCudaTensor_data(state, batchAgg),
                                                                               nBatch, nFeature, nSpatial,
                                                                               chunkSize, true);
   /*
     in: batchAgg
     out: meanBuffer

     Compute *sum* across nSpatial dimension of batchAgg. Normalize by nSpatial in kernel(s) below.
   */
   spatialBatchNormalization_spatialAgg_kernel<<<dimGridSpatial, dimBlockSpatial>>>(THCudaTensor_data(state, batchAgg),
                                                                                    THCudaTensor_data(state, meanBuffer),
                                                                                    nBatch, nFeature, nSpatial);
   /*
     in: transposedInput, meanBuffer
     out: batchAgg

     Compute va across nBatch dimension of transposedInput.
   */
   spatialBatchNormalization_batchVar_kernel<<<dimGridBatch, dimBlockBatch>>>(THCudaTensor_data(state, transposedInput),
                                                                              THCudaTensor_data(state, meanBuffer),
                                                                              THCudaTensor_data(state, batchAgg),
                                                                              nBatch, nFeature, nSpatial, chunkSize);
   /*
     in: batchAgg
     out: stdBuffer

     Compute *sum* across nSpatial dimension of batchAgg. Normalize by nSpatial in kernel(s) below.
   */
   spatialBatchNormalization_spatialAgg_kernel<<<dimGridSpatial, dimBlockSpatial>>>(THCudaTensor_data(state, batchAgg),
                                                                                    THCudaTensor_data(state, stdBuffer),
                                                                                    nBatch, nFeature, nSpatial);
   cudaError errcode = cudaGetLastError();
   if(errcode != cudaSuccess)
     THError(cudaGetErrorString(errcode));

   return;
}

static void cunn_SpatialBatchNormalization_updateOutputNormStatsAndTranspose(THCState *state,
                                                                             THCudaTensor *transposedOutput,
                                                                             THCudaTensor *output,
                                                                             THCudaTensor *meanBuffer,
                                                                             THCudaTensor *stdBuffer,
                                                                             float epsilon,
                                                                             int nBatch, int nFeature,
                                                                             int nSpatial) {
   /*
    -----
    Block/thread dimensions for [nFeature]
    -----
    blockIdx.x - 2 blocks, id 0 for mean, 1 for std
    threadIdx.x - over nFeature dimension
   */
   dim3 gridBlockMeanStd(nFeature);
   dim3 dimGridMeanStd(2, 1);
   float factor = 1.0f / (float)nSpatial;

   /*
     in: meanBuffer, stdBuffer
     out: meanBuffer, stdBuffer

     Normalize both the spatial mean and std (divide by nSpatial) and write out

     mean[i] = mean[i] * factor
     std[i] = std[i] * factor
   */
   spatialBatchNormalization_normalizeMeanStd_kernel<<<dimGridMeanStd, gridBlockMeanStd>>>(THCudaTensor_data(state, meanBuffer),
                                                                                           THCudaTensor_data(state, stdBuffer),
                                                                                           factor, epsilon, nFeature);

   /* Transpose output back */
   cunn_SpatialBatchNormalization_transposeOutput(state, transposedOutput, output, nBatch, nFeature, nSpatial);

   return;
}

static int cunn_SpatialBatchNormalization_updateOutputAffine(lua_State *L)
{
   THCState *state = getCutorchState(L);

   // Params:
   int nBatch = (int)luaL_checknumber(L, 3);
   int nFeature = (int)luaL_checknumber(L, 4);
   int nSpatial = (int)luaL_checknumber(L, 5);

   // Input
   THCudaTensor *input = (THCudaTensor*)luaT_checkudata(L, 2, "torch.CudaTensor");

   // Pre-allocated buffers for the transposes
   THCudaTensor *transposedInput = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "transposedInput", "torch.CudaTensor");
   THCudaTensor *transposedOutput = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "transposedOutput", "torch.CudaTensor");

   // Output
   THCudaTensor *output = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "output", "torch.CudaTensor");

   // Temp structure for nFeature x nSpatial (aggregated over batch)
   THCudaTensor *batchAgg = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "batchAgg", "torch.CudaTensor");

   float epsilon = (float)luaT_getfieldchecknumber(L, 1, "eps");
   THCudaTensor *weight = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "weight", "torch.CudaTensor");
   THCudaTensor *bias = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "bias", "torch.CudaTensor");
   THCudaTensor *meanBuffer = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "meanBuffer", "torch.CudaTensor");
   THCudaTensor *stdBuffer = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "stdBuffer", "torch.CudaTensor");
   THCudaTensor *centered = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "centered", "torch.CudaTensor");
   THCudaTensor *normalized = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "normalized", "torch.CudaTensor");

   // assert
   THAssert(THCudaTensor_checkGPU(state, 10, transposedInput, transposedOutput, output, batchAgg,
                                  weight, bias, meanBuffer, stdBuffer, centered, normalized));
   luaL_argcheck(L, input->nDimension == 3 || input->nDimension == 4, 2, "3D or 4D (batch mode) tensor is expected");

   /* Compute stats */
   cunn_SpatialBatchNormalization_updateOutputComputeStats(state, input, transposedInput, batchAgg, meanBuffer,
                                                           stdBuffer, centered, normalized,
                                                           nBatch, nFeature, nSpatial);

   /* Parallelization same as for compute stats kernels above */
   int chunkSize = getSpatialChunkSize(nBatch, nSpatial);
   int numSpatialBlocks = nSpatial/chunkSize;

   dim3 dimBlockBatch(MAX_THREADS, 1);
   dim3 dimGridBatch(numSpatialBlocks, nFeature);

   /*
     in: transposedInput, meanBuffer, stdBuffer, weight, bias
     out: transposedOutput, centered (transposedInput[i] - mean), normalized (not multiplied by weight & bias)

     Normalize the transposedInput based on statistics.

     transposedOutput[globalIndex] = weight * (transposedInput[i] - mean) * std + bias
   */
   spatialBatchNormalization_normalizeAffine_kernel<<<dimGridBatch, dimBlockBatch>>>(THCudaTensor_data(state, transposedInput),
                                                                                     THCudaTensor_data(state, meanBuffer),
                                                                                     THCudaTensor_data(state, stdBuffer),
                                                                                     THCudaTensor_data(state, transposedOutput),
                                                                                     THCudaTensor_data(state, weight),
                                                                                     THCudaTensor_data(state, bias),
                                                                                     THCudaTensor_data(state, centered),
                                                                                     THCudaTensor_data(state, normalized),
                                                                                     epsilon, nBatch, nFeature, nSpatial,
                                                                                     chunkSize);
   /* Do final division by nBatch and tranpose output */
   cunn_SpatialBatchNormalization_updateOutputNormStatsAndTranspose(state, transposedOutput, output, meanBuffer, stdBuffer,
                                                                    epsilon, nBatch, nFeature, nSpatial);
   cudaError errcode = cudaGetLastError();
   if(errcode != cudaSuccess)
     THError(cudaGetErrorString(errcode));

   return 1;
}

static int cunn_SpatialBatchNormalization_updateOutput(lua_State *L)
{
   THCState *state = getCutorchState(L);

   // Params:
   int nBatch = (int)luaL_checknumber(L, 3);
   int nFeature = (int)luaL_checknumber(L, 4);
   int nSpatial = (int)luaL_checknumber(L, 5);

   // Input
   THCudaTensor *input = (THCudaTensor*)luaT_checkudata(L, 2, "torch.CudaTensor");

   // Pre-allocated buffers for the transposes
   THCudaTensor *transposedInput = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "transposedInput", "torch.CudaTensor");
   THCudaTensor *transposedOutput = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "transposedOutput", "torch.CudaTensor");

   // Output
   THCudaTensor *output = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "output", "torch.CudaTensor");

   // Temp structure for nFeature x nSpatial (aggregated over batch)
   THCudaTensor *batchAgg = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "batchAgg", "torch.CudaTensor");

   float epsilon = (float)luaT_getfieldchecknumber(L, 1, "eps");
   THCudaTensor *meanBuffer = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "meanBuffer", "torch.CudaTensor");
   THCudaTensor *stdBuffer = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "stdBuffer", "torch.CudaTensor");
   THCudaTensor *centered = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "centered", "torch.CudaTensor");
   THCudaTensor *normalized = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "normalized", "torch.CudaTensor");

   // assert
   THAssert(THCudaTensor_checkGPU(state, 8, transposedInput, transposedOutput, output, batchAgg,
                                  meanBuffer, stdBuffer, centered, normalized));
   luaL_argcheck(L, input->nDimension == 3 || input->nDimension == 4, 2, "3D or 4D (batch mode) tensor is expected");

   /* Compute stats */
   cunn_SpatialBatchNormalization_updateOutputComputeStats(state, input, transposedInput, batchAgg, meanBuffer,
                                                           stdBuffer, centered, normalized,
                                                           nBatch, nFeature, nSpatial);

   /* Parallelization same as for compute stats kernels above */
   int chunkSize = getSpatialChunkSize(nBatch, nSpatial);
   int numSpatialBlocks = nSpatial/chunkSize;

   dim3 dimBlockBatch(MAX_THREADS, 1);
   dim3 dimGridBatch(numSpatialBlocks, nFeature);

   /*
     in: transposedInput, meanBuffer, stdBuffer
     out: transposedOutput, centered (transposedInput[i] - mean), normalized

     Normalize the transposedInput based on statistics.

     transposedOutput[globalIndex] = (transposedInput[i] - mean) * std
   */
   spatialBatchNormalization_normalizeNoAffine_kernel<<<dimGridBatch, dimBlockBatch>>>(THCudaTensor_data(state, transposedInput),
                                                                                       THCudaTensor_data(state, meanBuffer),
                                                                                       THCudaTensor_data(state, stdBuffer),
                                                                                       THCudaTensor_data(state, transposedOutput),
                                                                                       THCudaTensor_data(state, centered),
                                                                                       THCudaTensor_data(state, normalized),
                                                                                       epsilon, nBatch, nFeature, nSpatial,
                                                                                       chunkSize);

   /* Do final division by nBatch and tranpose output */
   cunn_SpatialBatchNormalization_updateOutputNormStatsAndTranspose(state, transposedOutput, output, meanBuffer, stdBuffer,
                                                                    epsilon, nBatch, nFeature, nSpatial);
   cudaError errcode = cudaGetLastError();
   if(errcode != cudaSuccess)
     THError(cudaGetErrorString(errcode));

   return 1;
}

static void cunn_SpatialBatchNormalization_updateGradInputComputeStats(THCState *state,
                                                                       THCudaTensor *gradOutput,
                                                                       THCudaTensor *transposedGradOutput,
                                                                       THCudaTensor *transposedGradInput,
                                                                       THCudaTensor *centered,
                                                                       THCudaTensor *batchAgg,
                                                                       THCudaTensor *meanBuffer,
                                                                       THCudaTensor *stdBuffer,
                                                                       int nBatch, int nFeature,
                                                                       int nSpatial) {
   /* Transpose input */
   cunn_SpatialBatchNormalization_transposeInput(state, gradOutput, transposedGradOutput, nBatch, nFeature, nSpatial);

   /*
    -----
    Block/thread dimensions for [nBatch x nFeature x nSpatial]
    -----
    blockIdx.x - gets a chunk of nSpatial dimension, depending on MAX_THREADS and the nBatch paramter.
                 That is, each blockIdx.x handles as many nSpatial features as fits in its MAX_TREADS.
                 If nBatch < 32, zero-pad to 32 so that one warp handles all batches in one spatial dim.
    blockIdx.y - over nFeature dimension
    threadIdx.x - over nBatch dimension

    For reductions, each block performs a local reduction of its values, across all the warps that span
    the nBatch dimension. It then writes out the reduced (and normalized by nBatch) value to output[blockIdx.y].
   */
   int chunkSize = getSpatialChunkSize(nBatch, nSpatial);
   int numSpatialBlocks = nSpatial/chunkSize;

   dim3 dimBlockBatch(MAX_THREADS, 1);
   dim3 dimGridBatch(numSpatialBlocks, nFeature);

   /*
    -----
    Block/thread dimensions for [nFeature x nSpatial]
    -----
    blockIdx.x - chunk of nSpatial dimension (chunk size bound by MAX_THREADS)
    blockIdx.y - over nFeature dimension
    threadIdx.x - over nSpatial dimension

    In reduction, first all blocks do a local reduction of their chunk, then we reduce across blocks
    using atomic operations.
   */
   int spatialBlocks = getNumSpatialBlocks(nSpatial);

   dim3 dimBlockSpatial(MAX_THREADS, 1);
   dim3 dimGridSpatial(spatialBlocks, nFeature);


   /*
     in: centered, transposedGradOutput
     out: transposedGradInput

     Element-wise mulitply centered by transposedGradOutput, write out to transposedGradInput.
   */
   spatialBatchNormalization_elementWiseMultiply_kernel<<<dimGridBatch, dimBlockBatch>>>(THCudaTensor_data(state, centered),
                                                                                         THCudaTensor_data(state, transposedGradOutput),
                                                                                         THCudaTensor_data(state, transposedGradInput),
                                                                                         nBatch, nFeature, nSpatial, chunkSize);
   /*
     in: transposedGradInput
     out: batchAgg

     Compute mean across nBatch dimension of transposedGradInput.
   */
   spatialBatchNormalization_batchMean_kernel<<<dimGridBatch, dimBlockBatch>>>(THCudaTensor_data(state, transposedGradInput),
                                                                               THCudaTensor_data(state, batchAgg),
                                                                               nBatch, nFeature, nSpatial, chunkSize,
                                                                               true);
   /*
     in: batchAgg
     out: meanBuffer

     Compute *sum* across nSpatial dimension of batchAgg. Normalize by nSpatial in kernel(s) below.
   */
   spatialBatchNormalization_spatialAgg_kernel<<<dimGridSpatial, dimBlockSpatial>>>(THCudaTensor_data(state, batchAgg),
                                                                                    THCudaTensor_data(state, meanBuffer),
                                                                                    nBatch, nFeature, nSpatial);
   /*
     in: centered, meanBuffer, stdBuffer (from forward pass)
     out: transposedGradInput

     transposedGradInput[i] = centered[i] * mean * -1 * std * std;
   */
   spatialBatchNormalization_normGradInput_kernel<<<dimGridBatch, dimBlockBatch>>>(THCudaTensor_data(state, centered),
                                                                                   THCudaTensor_data(state, meanBuffer),
                                                                                   THCudaTensor_data(state, stdBuffer),
                                                                                   THCudaTensor_data(state, transposedGradInput),
                                                                                   nBatch, nFeature, nSpatial, chunkSize);
   /*
     in: transposedGradOutput
     out: batchAgg

     Compute mean across nBatch dimension of transposedGradOutput.
   */
   spatialBatchNormalization_batchMean_kernel<<<dimGridBatch, dimBlockBatch>>>(THCudaTensor_data(state, transposedGradOutput),
                                                                               THCudaTensor_data(state, batchAgg),
                                                                               nBatch, nFeature, nSpatial,
                                                                               chunkSize, true);
   /*
     in: batchAgg
     out: meanBuffer

     Compute *sum* across nSpatial dimension of batchAgg. Normalize by nSpatial in kernel(s) below.
   */
   spatialBatchNormalization_spatialAgg_kernel<<<dimGridSpatial, dimBlockSpatial>>>(THCudaTensor_data(state, batchAgg),
                                                                                    THCudaTensor_data(state, meanBuffer),
                                                                                    nBatch, nFeature, nSpatial);
   cudaError errcode = cudaGetLastError();
   if(errcode != cudaSuccess)
     THError(cudaGetErrorString(errcode));

   return;
}

static int cunn_SpatialBatchNormalization_updateGradInputAffine(lua_State *L)
{
   THCState *state = getCutorchState(L);

   // Params:
   int nBatch = (int)luaL_checknumber(L, 3);
   int nFeature = (int)luaL_checknumber(L, 4);
   int nSpatial = (int)luaL_checknumber(L, 5);

   // Input
   THCudaTensor *gradOutput = (THCudaTensor*)luaT_checkudata(L, 2, "torch.CudaTensor");

   // Pre-allocated buffers for the transposes, reused across forward/backward
   THCudaTensor *transposedGradOutput = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "transposedInput", "torch.CudaTensor");
   THCudaTensor *transposedGradInput = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "transposedOutput", "torch.CudaTensor");

   // Output
   THCudaTensor *gradInput = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "gradInput", "torch.CudaTensor");

   THCudaTensor *batchAgg = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "batchAgg", "torch.CudaTensor");
   THCudaTensor *weight = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "weight", "torch.CudaTensor");
   THCudaTensor *centered = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "centered", "torch.CudaTensor");
   THCudaTensor *meanBuffer = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "meanBuffer", "torch.CudaTensor");
   THCudaTensor *stdBuffer = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "stdBuffer", "torch.CudaTensor");

   // assert
   THAssert(THCudaTensor_checkGPU(state, 7, batchAgg, gradOutput, gradInput, weight, centered, meanBuffer, stdBuffer));
   luaL_argcheck(L, gradOutput->nDimension == 3 || gradOutput->nDimension == 4, 2, "3D or 4D (batch mode) tensor is expected");
   luaL_argcheck(L, gradInput->nDimension == 3 || gradInput->nDimension == 4, 2, "3D or 4D (batch mode) tensor is expected");

   cunn_SpatialBatchNormalization_updateGradInputComputeStats(state, gradOutput, transposedGradOutput, transposedGradInput,
                                                              centered, batchAgg, meanBuffer, stdBuffer, nBatch, nFeature, nSpatial);

   /* Parallelization same as for compute stats kernels above */
   int chunkSize = getSpatialChunkSize(nBatch, nSpatial);
   int numSpatialBlocks = nSpatial/chunkSize;

   dim3 dimBlockBatch(MAX_THREADS, 1);
   dim3 dimGridBatch(numSpatialBlocks, nFeature);
   /*
     in: transposedGradOutput, transposedGradInput, meanBuffer, stdBuffer, weight
     out: transposedGradInput

      transposedGradInput[i] = (transposedGradInput[i] + transposedGradOutput[i] - mean) * std * weight
    */
   spatialBatchNormalization_updateFinalGradInputAffine_kernel<<<dimGridBatch, dimBlockBatch>>>(THCudaTensor_data(state, transposedGradOutput),
                                                                                                THCudaTensor_data(state, transposedGradInput),
                                                                                                THCudaTensor_data(state, meanBuffer),
                                                                                                THCudaTensor_data(state, stdBuffer),
                                                                                                THCudaTensor_data(state, weight),
                                                                                                nBatch, nFeature, nSpatial, chunkSize);
   /* Transpose output back */
   cunn_SpatialBatchNormalization_transposeOutput(state, transposedGradInput, gradInput, nBatch, nFeature, nSpatial);

   cudaError errcode = cudaGetLastError();
   if(errcode != cudaSuccess)
     THError(cudaGetErrorString(errcode));

   return 1;
}

static int cunn_SpatialBatchNormalization_updateGradInput(lua_State *L)
{
   THCState *state = getCutorchState(L);

   // Params:
   int nBatch = (int)luaL_checknumber(L, 3);
   int nFeature = (int)luaL_checknumber(L, 4);
   int nSpatial = (int)luaL_checknumber(L, 5);

   // Input
   THCudaTensor *gradOutput = (THCudaTensor*)luaT_checkudata(L, 2, "torch.CudaTensor");

   // Pre-allocated buffers for the transposes, reused across forward/backward
   THCudaTensor *transposedGradOutput = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "transposedInput", "torch.CudaTensor");
   THCudaTensor *transposedGradInput = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "transposedOutput", "torch.CudaTensor");

   // Output
   THCudaTensor *gradInput = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "gradInput", "torch.CudaTensor");

   THCudaTensor *batchAgg = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "batchAgg", "torch.CudaTensor");
   THCudaTensor *centered = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "centered", "torch.CudaTensor");
   THCudaTensor *meanBuffer = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "meanBuffer", "torch.CudaTensor");
   THCudaTensor *stdBuffer = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "stdBuffer", "torch.CudaTensor");

   // assert
   THAssert(THCudaTensor_checkGPU(state, 6, batchAgg, gradOutput, gradInput, centered, meanBuffer, stdBuffer));
   luaL_argcheck(L, gradOutput->nDimension == 3 || gradOutput->nDimension == 4, 2, "3D or 4D (batch mode) tensor is expected");
   luaL_argcheck(L, gradInput->nDimension == 3 || gradInput->nDimension == 4, 2, "3D or 4D (batch mode) tensor is expected");

   cunn_SpatialBatchNormalization_updateGradInputComputeStats(state, gradOutput, transposedGradOutput, transposedGradInput,
                                                              centered, batchAgg, meanBuffer, stdBuffer, nBatch, nFeature, nSpatial);

   /* Parallelization same as for compute stats kernels above */
   int chunkSize = getSpatialChunkSize(nBatch, nSpatial);
   int numSpatialBlocks = nSpatial/chunkSize;

   dim3 dimBlockBatch(MAX_THREADS, 1);
   dim3 dimGridBatch(numSpatialBlocks, nFeature);

   /*
     in: transposedGradOutput, transposedGradInput, meanBuffer, stdBuffer, weight
     out: transposedGradInput

      transposedGradInput[i] = (transposedGradInput[i] + transposedGradOutput[i] - mean) * std * weight
    */
   spatialBatchNormalization_updateFinalGradInputNoAffine_kernel<<<dimGridBatch, dimBlockBatch>>>(THCudaTensor_data(state, transposedGradOutput),
                                                                                                  THCudaTensor_data(state, transposedGradInput),
                                                                                                  THCudaTensor_data(state, meanBuffer),
                                                                                                  THCudaTensor_data(state, stdBuffer),
                                                                                                  nBatch, nFeature, nSpatial, chunkSize);
   /* Transpose output back */
   cunn_SpatialBatchNormalization_transposeOutput(state, transposedGradInput, gradInput, nBatch, nFeature, nSpatial);

   cudaError errcode = cudaGetLastError();
   if(errcode != cudaSuccess)
     THError(cudaGetErrorString(errcode));

   return 1;
}

static int cunn_SpatialBatchNormalization_accGradParameters(lua_State *L)
{
   THCState *state = getCutorchState(L);

   // Params:
   int nBatch = (int)luaL_checknumber(L, 2);
   int nFeature = (int)luaL_checknumber(L, 3);
   int nSpatial = (int)luaL_checknumber(L, 4);
   float scale = (float)luaL_checknumber(L, 5);

   // Input
   THCudaTensor *gradOutput = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "transposedInput", "torch.CudaTensor");

   THCudaTensor *gradBias = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "gradBias", "torch.CudaTensor");
   THCudaTensor *normalized = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "normalized", "torch.CudaTensor");
   THCudaTensor *gradWeight = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "gradWeight", "torch.CudaTensor");

   // here these two are just buffers for the aggregates sum for gradOutput * normalized and gradOutput
   THCudaTensor *buffer1 = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "meanBuffer", "torch.CudaTensor");
   THCudaTensor *buffer2 = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "stdBuffer", "torch.CudaTensor");

   // Temp structure for nFeature x nSpatial (aggregated over batch)
   THCudaTensor *batchAgg = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "batchAgg", "torch.CudaTensor");

   // flag to indicate whether the forward pass was called before this call
   bool forwardDone = (bool)luaT_getfieldcheckboolean(L, 1, "forwardDone");

   // assert
   THAssert(THCudaTensor_checkGPU(state, 7, gradOutput, gradBias, normalized, gradWeight, buffer1, buffer2, batchAgg));
   luaL_argcheck(L, gradOutput->nDimension == 3 || gradOutput->nDimension == 4, 2, "3D or 4D (batch mode) tensor is expected");
   THAssert(forwardDone);

   /*
    -----
    Block/thread dimensions for [nBatch x nFeature x nSpatial]
    -----
    blockIdx.x - gets a chunk of nSpatial dimension, depending on MAX_THREADS and the nBatch paramter.
                 That is, each blockIdx.x handles as many nSpatial features as fits in its MAX_TREADS.
                 If nBatch < 32, zero-pad to 32 so that one warp handles all batches in one spatial dim.
    blockIdx.y - over nFeature dimension
    threadIdx.x - over nBatch dimension

    For reductions, each block performs a local reduction of its values, across all the warps that span
    the nBatch dimension. It then writes out the reduced (and normalized by nBatch) value to output[blockIdx.y].
   */
   int chunkSize = getSpatialChunkSize(nBatch, nSpatial);
   int numSpatialBlocks = nSpatial/chunkSize;

   dim3 dimBlockBatch(MAX_THREADS, 1);
   dim3 dimGridBatch(numSpatialBlocks, nFeature);

   /*
    -----
    Block/thread dimensions for [nFeature x nSpatial]
    -----
    blockIdx.x - chunk of nSpatial dimension (chunk size bound by MAX_THREADS)
    blockIdx.y - over nFeature dimension
    threadIdx.x - over nSpatial dimension

    In reduction, first all blocks do a local reduction of their chunk, then we reduce across blocks
    using atomic operations.
   */
   int spatialBlocks = getNumSpatialBlocks(nSpatial);

   dim3 dimBlockSpatial(MAX_THREADS, 1);
   dim3 dimGridSpatial(spatialBlocks, nFeature);

   /*
    -----
    Block/thread dimensions for [nFeature]
    -----
    blockIdx.x - 2 blocks, id 0 for mean, 1 for std
    threadIdx.x - over nFeature dimension
   */
   dim3 gridBlockMeanStd(nFeature);
   dim3 dimGridMeanStd(2, 1);

   /*
     in: normalized, gradOutput
     out: normalized

     Element-wise multiply normalized by gradOutput, write out to normalized
     Note: this overwrites the normalized buffer
   */
   spatialBatchNormalization_elementWiseMultiply_kernel<<<dimGridBatch, dimBlockBatch>>>(THCudaTensor_data(state, normalized),
                                                                                         THCudaTensor_data(state, gradOutput),
                                                                                         THCudaTensor_data(state, normalized),
                                                                                         nBatch, nFeature, nSpatial, chunkSize);
   /*
     in: normalized
     out: batchAgg

     Compute *sum* across nBatch dimension of normalized (divideByNBatch = false)
   */
   spatialBatchNormalization_batchMean_kernel<<<dimGridBatch, dimBlockBatch>>>(THCudaTensor_data(state, normalized),
                                                                               THCudaTensor_data(state, batchAgg),
                                                                               nBatch, nFeature, nSpatial,
                                                                               chunkSize, false);
   /*
     in: batchAgg
     out: buffer1

     Compute *sum* across nSpatial dimension of batchAgg. Normalize by nSpatial in kernel(s) below.
   */
   spatialBatchNormalization_spatialAgg_kernel<<<dimGridSpatial, dimBlockSpatial>>>(THCudaTensor_data(state, batchAgg),
                                                                                    THCudaTensor_data(state, buffer1),
                                                                                    nBatch, nFeature, nSpatial);
   /*
     in: gradOutput
     out: batchAgg

     Compute *sum* across nBatch dimension of gradOutput (divideByNBatch = false)
   */
   spatialBatchNormalization_batchMean_kernel<<<dimGridBatch, dimBlockBatch>>>(THCudaTensor_data(state, gradOutput),
                                                                               THCudaTensor_data(state, batchAgg),
                                                                               nBatch, nFeature, nSpatial,
                                                                               chunkSize,
                                                                               false);
   /*
     in: batchAgg
     out: buffer2

     Compute *sum* across nSpatial dimension of batchAgg. Normalize by nSpatial in kernel(s) below.
   */
   spatialBatchNormalization_spatialAgg_kernel<<<dimGridSpatial, dimBlockSpatial>>>(THCudaTensor_data(state, batchAgg),
                                                                                    THCudaTensor_data(state, buffer2),
                                                                                    nBatch, nFeature, nSpatial);
   /*
     in: gradWeight, buffer1, gradBias, buffer2
     out: gradWeight, gradBias

     Scales weight and bias.
     weight[i] = weight[i] + buffer1[i] * scale;
     bias[i] = bias[i] + buffer2[i] * scale;
   */
   spatialBatchNormalization_updateWeightBias_kernel<<<dimGridMeanStd, gridBlockMeanStd>>>(THCudaTensor_data(state, gradWeight),
                                                                                           THCudaTensor_data(state, buffer1),
                                                                                           THCudaTensor_data(state, gradBias),
                                                                                           THCudaTensor_data(state, buffer2),
                                                                                           scale, nFeature);
   cudaError errcode = cudaGetLastError();
   if(errcode != cudaSuccess)
     THError(cudaGetErrorString(errcode));

   return 1;
}
static const struct luaL_Reg cunn_SpatialBatchNormalization__ [] = {
  {"SpatialBatchNormalization_updateOutputAffine", cunn_SpatialBatchNormalization_updateOutputAffine},
  {"SpatialBatchNormalization_updateOutput", cunn_SpatialBatchNormalization_updateOutput},
  {"SpatialBatchNormalization_updateGradInputAffine", cunn_SpatialBatchNormalization_updateGradInputAffine},
  {"SpatialBatchNormalization_updateGradInput", cunn_SpatialBatchNormalization_updateGradInput},
  {"SpatialBatchNormalization_accGradParameters", cunn_SpatialBatchNormalization_accGradParameters},
  {"SpatialBatchNormalization_forwardInferenceAffine", cunn_SpatialBatchNormalization_forwardInferenceAffine},
  {"SpatialBatchNormalization_forwardInference", cunn_SpatialBatchNormalization_forwardInference},
  {NULL, NULL}
};

static void cunn_SpatialBatchNormalization_init(lua_State *L)
{
  luaT_pushmetatable(L, "torch.CudaTensor");
  luaT_registeratname(L, cunn_SpatialBatchNormalization__, "nn");
  lua_pop(L,1);
}
