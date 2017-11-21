#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "generic/LenSoftMax.cu"
#else

#include "../common.h"

void THNN_(LenSoftMax_updateOutput)(
           THCState *state,
           THCTensor *input,
           THCTensor *output,
           THCIndexTensor *len)
{
  THCUNN_assertSameGPU(state, 2, input, output);

  if ((input->nDimension != 2) && (len->nDimension != 1))
  {
    THError("2D tensor expected for input, 1D tensor expected for len");
  }

  input = THCTensor_(newContiguous)(state, input);
  THCTensor_(resizeAs)(state, output, input);
  long batchSize = input->size[0], dim = input->size[1];
  long blocksY = 1, blocksZ = 1;

  dim3 blocks(batchSize, blocksY, blocksZ);
  dim3 threads(LENSOFTMAX_THREADS);
  cunn_LenSoftMax_updateOutput_kernel<real, accreal, THCIndex_t><<<blocks, threads, 0, THCState_getCurrentStream(state)>>>(
    THCTensor_(data)(state, output),
    THCTensor_(data)(state, input),
    batchSize, dim, THCIndexTensor_(data)(state, len)
  );
  THCudaCheck(cudaGetLastError());

  THCTensor_(free)(state, input);
}

void THNN_(LenSoftMax_updateGradInput)(
           THCState *state,
           THCTensor *input,
           THCTensor *gradOutput,
           THCTensor *gradInput,
           THCTensor *output,
           THCIndexTensor *len)
{
  THCUNN_check_nElement(state, input, gradOutput);
  THCUNN_assertSameGPU(state, 3, output, gradOutput, gradInput);

  if ((gradInput->nDimension != 2) && (len->nDimension != 1))
  {
    THError("2D tensor expected for input, 1D tensor expected for len");
  }


  output = THCTensor_(newContiguous)(state, output);
  gradOutput = THCTensor_(newContiguous)(state, gradOutput);

  THCTensor_(resizeAs)(state, gradInput, output);
  long batchSize = gradInput->size[0], dim = gradInput->size[1];
  long blocksY = 1, blocksZ = 1;

  dim3 blocks(batchSize, blocksY, blocksZ);
  dim3 threads(LENSOFTMAX_THREADS);
  cunn_LenSoftMax_updateGradInput_kernel<real, accreal, THCIndex_t><<<blocks, threads, 0, THCState_getCurrentStream(state)>>>(
    THCTensor_(data)(state, gradInput),
    THCTensor_(data)(state, output),
    THCTensor_(data)(state, gradOutput),
    batchSize, dim, THCIndexTensor_(data)(state, len)
  );
  THCudaCheck(cudaGetLastError());

  THCTensor_(free)(state, gradOutput);
  THCTensor_(free)(state, output);
}

#endif
