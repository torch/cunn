#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "generic/SpatialDepthWiseConvolution.cu"
#else

#include "common.h"
#include <vector>

// Helper for `updateOutput`. Fills `output` with constant (bias) values
__global__ void fillOutputWithBias(
  real *output, const int batchSize, const int elementsPerPlane,
  const real *bias, const int nInputPlane, const int nOutputPlane) {

  // `output` is of size
  // (batchSize) x (nInputPlane) x (nOutputPlane) x (outputHeight*outputWidth)
  int index = blockIdx.x * blockDim.x + threadIdx.x;

  if (index < batchSize * nInputPlane * nOutputPlane * elementsPerPlane) {
    real *outputPixel = &output[index];

    index /= elementsPerPlane; index %= nInputPlane * nOutputPlane; // index of the output channel
    int outPlaneIdx = index % nOutputPlane;
    int  inPlaneIdx = index / nOutputPlane;

    // bias is of size (nOutputPlane) x (nInputPlane)
    *outputPixel = bias[outPlaneIdx*nInputPlane + inPlaneIdx];
  }
}

// Transposes `tensor` pseudo-in-place using `buffer`.
// `buffer` is enlarged if needed.
// This function REQUIRES `buffer` to be non-NULL.
void transposeWithBuffer(
  THCState *state, THCTensor *tensor, THCStorage *buffer,
  const int dim1, const int dim2) {

  THAssert(buffer != NULL);

  THCTensor *tensor_viewT = THCTensor_(newTranspose)(state, tensor, dim1, dim2);
  
  // Size of `buffer` (== `columns`, for example) should be
  // enough in general, but who knows, so let `setStorageNd` ensure
  THCTensor *bufferTensorT = THCTensor_(new)(state);
  THCTensor_(setStorageNd)(state, bufferTensorT, buffer, 0, 
    tensor_viewT->nDimension, tensor_viewT->size, NULL);

  // This makes a contiguous tensor from `tensor_viewT`, i.e. does the actual transpose
  THCTensor_(copy)(state, bufferTensorT, tensor_viewT);
  // Copy the transposed data back
  THCTensor_(copy)(state, tensor, bufferTensorT);

  // Now reshape `tensor` to match the transposed size
  std::vector<long> newSize(tensor->size, tensor->size + tensor->nDimension);
  std::swap(newSize[dim1], newSize[dim2]);
  THCTensor_(setStorageNd)(state, tensor, tensor->storage,
    tensor->storageOffset, tensor->nDimension, newSize.data(), NULL);

  THCTensor_(free)(state, tensor_viewT);
  THCTensor_(free)(state, bufferTensorT);
}

static inline void THNN_(SpatialDepthWiseConvolution_shapeCheck)(
                         THCState *state,
                         THCTensor *input, THCTensor *gradOutput,
                         THCTensor *weight, THCTensor *bias,
                         int kH, int kW, int dH, int dW, int padH, int padW) {
  THArgCheck(kW > 0 && kH > 0, 9,
             "kernel size should be greater than zero, but got kH: %d kW: %d", kH, kW);
  THArgCheck(dW > 0 && dH > 0, 11,
             "stride should be greater than zero, but got dH: %d dW: %d", dH, dW);
  THCUNN_argCheck(state, weight->nDimension == 4, 5, weight,
                  "2D or 4D weight tensor expected, but got: %s");

  if (bias != NULL) {
    THCUNN_check_dim_size(state, bias, 2, 0, weight->size[0]);
    THCUNN_check_dim_size(state, bias, 2, 1, weight->size[1]);
  }

  int ndim = input->nDimension;
  int dimf = 0;
  int dimh = 1;
  int dimw = 2;

  if (ndim == 4) {
    dimf++;
    dimh++;
    dimw++;
  }

  THCUNN_argCheck(state, ndim == 3 || ndim == 4, 2, input,
                  "3D or 4D input tensor expected but got: %s");

  long nInputPlane  = weight->size[1];
  long inputHeight  = input->size[dimh];
  long inputWidth   = input->size[dimw];
  long nOutputPlane = weight->size[0];
  long outputHeight = (inputHeight + 2*padH - kH) / dH + 1;
  long outputWidth  = (inputWidth + 2*padW - kW) / dW + 1;

  if (outputWidth < 1 || outputHeight < 1)
      THError("Given input size: (%d x %d x %d). "
              "Calculated output size: (%d x %d x %d). Output size is too small",
              nInputPlane,inputHeight,inputWidth,nOutputPlane*nInputPlane,outputHeight,outputWidth);

  THCUNN_check_dim_size(state, input, ndim, dimf, nInputPlane);

  if (gradOutput != NULL) {
    THCUNN_check_dim_size(state, gradOutput, ndim + 1, dimf, nInputPlane);
    THCUNN_check_dim_size(state, gradOutput, ndim + 1, dimh, nOutputPlane);
    THCUNN_check_dim_size(state, gradOutput, ndim + 1, dimw, outputHeight);
    THCUNN_check_dim_size(state, gradOutput, ndim + 1, dimw + 1, outputWidth);
  }
}

__global__ void updateOutputKernel(
      real *output          , const real *input     , 
      const real *weight    , const int batchSize   ,
      const int inputHeight , const int inputWidth  ,
      const int nInputPlane , const int nOutputPlane,
      const int outputHeight, const int outputWidth ,
      const int kH          , const int kW          ,
      const int padH        , const int padW        ,
      const int strideH     , const int strideW     ,
      const int dilationH   , const int dilationW) {

  int index = blockIdx.x * blockDim.x + threadIdx.x;

  if (index < batchSize * nInputPlane * nOutputPlane * outputHeight * outputWidth) {

    const int outIndex = index;
    // grid is (batchSize) x (nInputPlane) x (nOutputPlane) x (outputHeight) x (outputWidth)
    const int wOut        = index % outputWidth ; index /= outputWidth ;
    const int hOut        = index % outputHeight; index /= outputHeight;
    const int outPlaneIdx = index % nOutputPlane; index /= nOutputPlane;
    const int inPlaneIdx  = index % nInputPlane ; index /= nInputPlane ;
    // `index` is now the index of the sample in batch
    const int & sampleIdx = index;

    const int hInStart = hOut * strideH - padH;
    const int wInStart = wOut * strideW - padW;
    input += ((sampleIdx 
      * nInputPlane + inPlaneIdx)
        * inputHeight + hInStart) 
          * inputWidth + wInStart;

    // weight is (nInputPlane) x (nOutputPlane) x (kH) x (kW)
    weight += (inPlaneIdx * nOutputPlane + outPlaneIdx) * kH * kW;

    accreal result = 0;

    for (int i = 0; i < kH; ++i) {
      for (int j = 0; j < kW; ++j) {
        const int h = hInStart + i * dilationH;
        const int w = wInStart + j * dilationW;
        result += 
          (h >= 0 && w >= 0 && h < inputHeight && w < inputWidth) ?
          input[i * dilationH * inputWidth + j * dilationW] * (*weight) :
          ScalarConvert<int, real>::to(0);
        ++weight;
      }
    }

    output[outIndex] += ScalarConvert<accreal, real>::to(result);
  }
}

void THNN_(SpatialDepthWiseConvolution_updateOutput)(
           THCState *state,
           THCTensor *input,
           THCTensor *output,
           THCTensor *weight,
           THCTensor *bias,
           THCTensor *columns,
           THCTensor *ones,
           int kW, int kH,
           int dW, int dH,
           int padW, int padH) {

  THCUNN_assertSameGPU(state, 5, input, output, weight, columns, ones);
  if (bias) {
    THCUNN_assertSameGPU(state, 2, weight, bias);
  }

  // Params:
  int nInputPlane = weight->nDimension == 2 ? weight->size[1]/(kH*kW) : weight->size[1];
  int nOutputPlane = weight->size[0];
  if (weight->nDimension == 2) {
    THCTensor_(resize4d)(state, weight, nOutputPlane, nInputPlane, kH, kW);
  }

  THNN_(SpatialDepthWiseConvolution_shapeCheck)
       (state, input, NULL, weight, bias, kH, kW, dH, dW, padH, padW);

  long inputWidth   = input->size[3];
  long inputHeight  = input->size[2];
  long outputWidth  = (inputWidth + 2*padW - kW) / dW + 1;
  long outputHeight = (inputHeight + 2*padH - kH) / dH + 1;

  input = THCTensor_(newContiguous)(state, input);

  int batch = 1;
  if (input->nDimension == 3) {
    // Force batch
    batch = 0;
    THCTensor_(resize4d)(state, input, 1, input->size[0], input->size[1], input->size[2]);
  }

  // Batch size + input planes
  long batchSize = input->size[0];

  // Resize output
  THCTensor_(resize5d)(state, output, batchSize, nInputPlane, nOutputPlane, outputHeight, outputWidth);
  // Resize temporary columns
  THCTensor_(resizeAs)(state, columns, weight); // reserve size for transpose

  transposeWithBuffer(state, weight, columns->storage, 0, 1);

  // Helpers
  THCTensor *outputTransposed_i = THCTensor_(new)(state);
  THCTensor *weight_i = THCTensor_(new)(state);

  // Make sure `ones` buffer is at least as large as `output`
  // THCTensor *outputTransposed = ones;
  THCTensor_(resize4d)(state, ones, 
    nInputPlane, nOutputPlane, batchSize, outputHeight*outputWidth);

  // Do bias first (fill the output)
  if (bias) {
    // fillOutputTransposedWithBias
    fillOutputWithBias
      <<<GET_BLOCKS(batchSize*nInputPlane*nOutputPlane*outputHeight*outputWidth), 
      CUDA_NUM_THREADS, 0, THCState_getCurrentStream(state)>>> (
        THCTensor_(data)(state, output), batchSize, outputHeight*outputWidth,
        THCTensor_(data)(state, bias), nInputPlane, nOutputPlane);
    THCudaCheck(cudaGetLastError());
  } else {
    THCTensor_(zero)(state, output);
  }

  updateOutputKernel
    <<<GET_BLOCKS(THCTensor_(nElement)(state, output)), 
    CUDA_NUM_THREADS, 0, THCState_getCurrentStream(state)>>>(
      THCTensor_(data)(state, output), THCTensor_(data)(state, input),
      THCTensor_(data)(state, weight), batchSize,
      inputHeight, inputWidth, nInputPlane, nOutputPlane, 
      outputHeight, outputWidth, kH, kW, padH, padW, dH, dW, 1, 1);
  THCudaCheck(cudaGetLastError());

  // transpose back
  transposeWithBuffer(state, weight, columns->storage, 0, 1);

  // Free
  THCTensor_(free)(state, outputTransposed_i);
  THCTensor_(free)(state, weight_i);

  // Merge first dims of the output
  THCTensor_(resize4d)(state, output, batchSize, nInputPlane * nOutputPlane, outputHeight, outputWidth);

  // Make a contiguous copy of output (OPTIONAL)
  // THCTensor *_output = THCTensor_(newContiguous)(state, output);

  // Resize output
  if (batch == 0) {
    THCTensor_(select)(state, output, NULL, 0, 0);
    THCTensor_(select)(state, input, NULL, 0, 0);
  }
  //else
    //THCTensor_(resize5d)(state, output, batchSize, nOutputPlane, nInputPlane, outputHeight, outputWidth);

  // Copy output back
  // THCTensor_(freeCopyTo)(state, _output, output);

  THCTensor_(free)(state, input);
}

__global__ void updateGradInputKernel(
      real *gradInput       , const real *gradOutput,
      const real *weight    , const int batchSize   ,
      const int inputHeight , const int inputWidth  ,
      const int nInputPlane , const int nOutputPlane,
      const int outputHeight, const int outputWidth ,
      const int kH          , const int kW          ,
      const int padH        , const int padW        ,
      const int strideH     , const int strideW     ,
      const int dilationH   , const int dilationW) {

  int index = blockIdx.x * blockDim.x + threadIdx.x;

  if (index < batchSize * nInputPlane * inputHeight * inputWidth) {

    const int gradInputIdx = index;
    // grid is (batchSize) x (nInputPlane) x (inputHeight) x (inputWidth)
    const int wIn         = index % inputWidth  + padW; index /= inputWidth ;
    const int hIn         = index % inputHeight + padH; index /= inputHeight;
    const int inPlaneIdx  = index % nInputPlane       ; index /= nInputPlane;
    // `index` is now the index of the sample in batch
    const int & sampleIdx = index;

    // gradOutput is (batchSize) x (nInputPlane) x (nOutputPlane) x (outputHeight) x (outputWidth)
    gradOutput += (((sampleIdx 
      * nInputPlane + inPlaneIdx)
        * nOutputPlane)
          * outputHeight)
            * outputWidth;

    // weight is (nInputPlane) x (nOutputPlane) x (kH) x (kW)
    weight += inPlaneIdx * nOutputPlane * kH * kW;

    accreal result = 0;

    const int kernelExtentW = (kW - 1) * dilationW + 1;
    const int kernelExtentH = (kH - 1) * dilationH + 1;
    // compute the start and end of the output
    const int wOutStart =
      (wIn < kernelExtentW) ? 0 : (wIn - kernelExtentW) / strideW + 1;
    const int wOutEnd = min(wIn / strideW + 1, outputHeight);
    const int hOutStart =
      (hIn < kernelExtentH) ? 0 : (hIn - kernelExtentH) / strideH + 1;
    const int hOutEnd = min(hIn / strideH + 1, outputHeight);

    for (int outPlaneIdx = 0; outPlaneIdx < nOutputPlane; ++outPlaneIdx) {
      for (int hOut = hOutStart; hOut < hOutEnd; ++hOut) {
        for (int wOut = wOutStart; wOut < wOutEnd; ++wOut) {
          int hWeight = (hIn - hOut * strideH);
          int wWeight = (wIn - wOut * strideW);
          
          // TODO: use LCM of stride and dilation to avoid unnecessary loops
          if (hWeight % dilationH == 0 && wWeight % dilationW == 0) {
            hWeight /= dilationH;
            wWeight /= dilationW;  

            result += 
              gradOutput[hOut * outputWidth + wOut] *
              weight[hWeight * kW + wWeight];
          }
        }
      }
      gradOutput += outputHeight * outputWidth;
    }

    gradInput[gradInputIdx] = ScalarConvert<accreal, real>::to(result);
  }
}

void THNN_(SpatialDepthWiseConvolution_updateGradInput)(
           THCState *state,
           THCTensor *input,
           THCTensor *gradOutput,
           THCTensor *gradInput,
           THCTensor *weight,
           THCTensor *gradColumns,
           THCTensor *ones,
           int kW, int kH,
           int dW, int dH,
           int padW, int padH) {

  THCUNN_assertSameGPU(state, 5, input, gradOutput, weight,
                       gradColumns, gradInput);

  // Params:
  int nInputPlane = weight->nDimension == 2 ? weight->size[1]/(kH*kW) : weight->size[1];
  int nOutputPlane = weight->size[0];
  if (weight->nDimension == 2) {
    THCTensor_(resize4d)(state, weight, nOutputPlane, nInputPlane, kH, kW);
  }

  gradOutput = THCTensor_(newWithTensor)(state, gradOutput);

  if (input->nDimension == 3) {
    if (gradOutput->nDimension == 3) {
      THCTensor_(resize4d)(state, gradOutput, nInputPlane, nOutputPlane, gradOutput->size[1], gradOutput->size[2]);
    }
  }
  else
  {
    if (gradOutput->nDimension == 4) {
      THCTensor_(resize5d)(state, gradOutput, gradOutput->size[0], nInputPlane, nOutputPlane, gradOutput->size[2], gradOutput->size[3]);
    }
  }

  THNN_(SpatialDepthWiseConvolution_shapeCheck)
       (state, input, gradOutput, weight, NULL, kH, kW, dH, dW, padH, padW);

  input = THCTensor_(newContiguous)(state, input);

  int batch = 1;
  if (input->nDimension == 3) {
    // Force batch
    batch = 0;
    THCTensor_(resize4d)(state, input, 1, input->size[0], input->size[1], input->size[2]);
    THCTensor_(resize5d)(state, gradOutput, 1, gradOutput->size[0], gradOutput->size[1], gradOutput->size[2], gradOutput->size[3]);
  }

  long inputWidth   = input->size[3];
  long inputHeight  = input->size[2];
  long outputWidth  = (inputWidth + 2*padW - kW) / dW + 1;
  long outputHeight = (inputHeight + 2*padH - kH) / dH + 1;

  // Batch size + input planes
  long batchSize = input->size[0];

  // Resize output
  THCTensor_(resize4d)(state, gradInput, batchSize, nInputPlane, inputHeight, inputWidth);

  updateGradInputKernel
    <<<GET_BLOCKS(THCTensor_(nElement)(state, gradInput)), 
    CUDA_NUM_THREADS, 0, THCState_getCurrentStream(state)>>>(
      THCTensor_(data)(state, gradInput), THCTensor_(data)(state, gradOutput),
      THCTensor_(data)(state, weight), batchSize,
      inputHeight, inputWidth, nInputPlane, nOutputPlane, 
      outputHeight, outputWidth, kH, kW, padH, padW, dH, dW, 1, 1);
  THCudaCheck(cudaGetLastError());

  // Resize output
  if (batch == 0) {
    THCTensor_(select)(state, gradOutput, NULL, 0, 0);
    THCTensor_(select)(state, input, NULL, 0, 0);
    THCTensor_(select)(state, gradInput, NULL, 0, 0);
  }

  THCTensor_(free)(state, input);
  THCTensor_(free)(state, gradOutput);
}

void THNN_(SpatialDepthWiseConvolution_accGradParameters)(
           THCState *state,
           THCTensor *input,
           THCTensor *gradOutput,
           THCTensor *gradWeight,
           THCTensor *gradBias,
           THCTensor *columns,
           THCTensor *ones,
           int kW, int kH,
           int dW, int dH,
           int padW, int padH,
           accreal scale_) {

  real scale = ScalarConvert<accreal, real>::to(scale_);

  THCUNN_assertSameGPU(state, 5, input, gradOutput, gradWeight, columns, ones);
  if (gradBias) {
    THCUNN_assertSameGPU(state, 2, gradWeight, gradBias);
  }

  input = THCTensor_(newContiguous)(state, input);

  int batch = 1;
  if (input->nDimension == 3) {
    // Force batch
    batch = 0;
    THCTensor_(resize4d)(state, input, 1, input->size[0], input->size[1], input->size[2]);
    THCTensor_(resize5d)(state, gradOutput, 1, gradOutput->size[0], gradOutput->size[1], gradOutput->size[2], gradOutput->size[3]);
  }

  // Batch size + input planes
  long batchSize = input->size[0];

  long inputWidth   = input->size[3];
  long inputHeight  = input->size[2];
  long outputWidth  = (inputWidth + 2*padW - kW) / dW + 1;
  long outputHeight = (inputHeight + 2*padH - kH) / dH + 1;

  // Params
  int nInputPlane = gradWeight->nDimension == 2 ? gradWeight->size[1]/(kW*kH) : gradWeight->size[1];
  int nOutputPlane = gradWeight->size[0];
  if (gradWeight->nDimension == 2) {
    THCTensor_(resize4d)(state, gradWeight, nOutputPlane, nInputPlane, kH, kW);
  }

  gradOutput = THCTensor_(newWithTensor)(state, gradOutput);
  if (input->nDimension == 3) {
    if (gradOutput->nDimension == 3) {
      THCTensor_(resize4d)(state, gradOutput, nInputPlane, nOutputPlane, gradOutput->size[1], gradOutput->size[2]);
    }
  }
  else
  {
    if (gradOutput->nDimension == 4) {
      THCTensor_(resize5d)(state, gradOutput, gradOutput->size[0], nInputPlane, nOutputPlane, gradOutput->size[2], gradOutput->size[3]);
    }
  }

  THNN_(SpatialDepthWiseConvolution_shapeCheck)
       (state, input, gradOutput, gradWeight, gradBias, kH, kW, dH, dW, padH, padW);

  // Do `gradBias` first:

  transposeWithBuffer(state, gradBias, columns->storage, 0, 1);

  // Define a buffer of ones, for bias accumulation
  if (ones->nDimension != 2 || ones->size[0]*ones->size[1] < outputHeight*outputWidth) {
    // Resize plane and fill with ones...
    THCTensor_(resize2d)(state, ones, outputHeight, outputWidth);
    THCTensor_(fill)(state, ones, ScalarConvert<int, real>::to(1));
  }

  // Helpers
  THCTensor *input_n = THCTensor_(new)(state);
  THCTensor *gradOutput_n = THCTensor_(new)(state);

  // For each elt in batch, do:
  for (int elt = 0; elt < batchSize; elt ++) {
    // Matrix mulitply per output:
    THCTensor_(select)(state, input_n, input, 0, elt);
    THCTensor_(select)(state, gradOutput_n, gradOutput, 0, elt);

    // Do Bias:
    // (see http://docs.nvidia.com/cuda/cublas/#cublas-lt-t-gt-gemm)
    long m_ = nInputPlane * nOutputPlane;
    long k_ = outputHeight * outputWidth;

    // Do GEMV (note: this is a bit confusing because gemv assumes column-major matrices)
    if (gradBias) {
      #if defined(THC_REAL_IS_FLOAT) || defined(THC_REAL_IS_DOUBLE)
      #ifdef THC_REAL_IS_FLOAT
      THCudaBlas_Sgemv(
      #elif defined(THC_REAL_IS_DOUBLE)
      THCudaBlas_Dgemv(
      #endif
          state,
          't',
          k_, m_,
          scale,
          THCTensor_(data)(state, gradOutput_n), k_,
          THCTensor_(data)(state, ones), 1,
          ScalarConvert<int, real>::to(1),
          THCTensor_(data)(state, gradBias), 1
      );
      #endif
      #ifdef THC_REAL_IS_HALF
      THCudaBlas_Hgemm(
          state,
          't', 'n',
          m_, 1, k_,
          scale,
          THCTensor_(data)(state, gradOutput_n), k_,
          THCTensor_(data)(state, ones), k_,
          ScalarConvert<int, real>::to(1),
          THCTensor_(data)(state, gradBias), m_
      );
      #endif
    }
  }

  transposeWithBuffer(state, gradBias, columns->storage, 0, 1);

  // OK, bias done, now let's do `gradWeight`:

  // merge two last (spatial) dimensions
  THCTensor_(setStorage3d)(state, gradWeight, 
    gradWeight->storage, gradWeight->storageOffset,
    nOutputPlane, -1, nInputPlane, -1, kH*kW, -1);

  transposeWithBuffer(state, gradWeight, columns->storage, 0, 1);
  // transpose for proper accumulation in GEMM
  transposeWithBuffer(state, gradWeight, columns->storage, 1, 2);

  // Resize temporary columns
  THCTensor_(resize3d)(state, columns, kW*kH, batchSize, outputHeight*outputWidth);

  THCTensor_(resize3d)(state, ones, nOutputPlane, batchSize, outputHeight*outputWidth);
  THCTensor *gradOutputGrouped = ones;

  THCTensor *gradWeight_i = THCTensor_(new)(state);
  THCTensor *gradOutput_i = THCTensor_(new)(state);

  for (int inPlaneIdx = 0; inPlaneIdx < nInputPlane; ++inPlaneIdx) {

    // group gradOutput planes by input plane index and transpose
    // `gradOutputGrouped` has size (nOutputPlane) x (batchSize) x (outputHeight) x (outputWidth)
    THCTensor_(select)(state, gradOutput_i, gradOutput, 1, inPlaneIdx);
    THCTensor_(transpose)(state, gradOutput_i, gradOutput_i, 0, 1);
    THCTensor_(copy)(state, gradOutputGrouped, gradOutput_i);

    // columns: (kW*kH) x (batchSize) x (outputHeight*outputWidth)
    // gradOutputTransposed: (nInputPlane) x (nOutputPlane) x (batchSize) x (outputHeight*outputWidth)
    // gradWeight: (nInputPlane) x (nOutputPlane) x (kH*kW)
    THCTensor_(select)(state, gradWeight_i, gradWeight, 0, inPlaneIdx);
    
    // Extract columns:
    im2col_depthwise(
      THCState_getCurrentStream(state),
      THCTensor_(data)(state, input), batchSize, nInputPlane,
      inPlaneIdx, inputHeight, inputWidth, kH, kW, padH, padW, dH, dW,
      1, 1, THCTensor_(data)(state, columns)
    );

    // M,N,K are dims of matrix A and B
    // (see http://docs.nvidia.com/cuda/cublas/#cublas-lt-t-gt-gemm)
    long m = nOutputPlane;
    long n = kH * kW;
    long k = batchSize * outputHeight * outputWidth;

    // Do GEMM (note: this is a bit confusing because gemm assumes column-major matrices)
    // (m x k) * (n x k)^T = (m x n)
    #ifdef THC_REAL_IS_FLOAT
    THCudaBlas_Sgemm(
    #elif defined(THC_REAL_IS_HALF)
    THCudaBlas_Hgemm(
    #elif defined(THC_REAL_IS_DOUBLE)
    THCudaBlas_Dgemm(
    #endif
        state,
        't', 'n',
        m, n, k,
        scale,
        THCTensor_(data)(state, gradOutputGrouped), k,
        THCTensor_(data)(state, columns), k,
        ScalarConvert<int, real>::to(1),
        THCTensor_(data)(state, gradWeight_i), m
    );
  }

  // transpose back
  transposeWithBuffer(state, gradWeight, columns->storage, 1, 2);
  transposeWithBuffer(state, gradWeight, columns->storage, 0, 1);

  // un-merge two last (spatial) dimensions back
  THCTensor_(setStorage4d)(state, gradWeight,
    gradWeight->storage, gradWeight->storageOffset,
    nOutputPlane, -1, nInputPlane, -1, kH, -1, kW, -1);

  // Free
  THCTensor_(free)(state, input_n);
  THCTensor_(free)(state, gradOutput_n);

  THCTensor_(free)(state, gradWeight_i);
  THCTensor_(free)(state, gradOutput_i);

  // Resize
  if (batch == 0) {
    THCTensor_(select)(state, gradOutput, NULL, 0, 0);
    THCTensor_(select)(state, input, NULL, 0, 0);
  }

  THCTensor_(free)(state, input);
  THCTensor_(free)(state, gradOutput);
}

#endif
