#include "THCUNN.h"

struct absupdateOutput_functor
{
  __device__ void operator()(float* output, const float* input) const
  {
    *output = abs(*input);
  }
};

void THNN_CudaAbs_updateOutput(THCState *state, THCudaTensor *input, THCudaTensor *output)
{
  THAssert(THCudaTensor_checkGPU(state, 2, input, output));
  THCudaTensor_resizeAs(state, output, input);
  THCudaTensor_pointwiseApply2(state, output, input, absupdateOutput_functor());
}

struct absupdateGradInput_functor
{
  __device__ void operator()(float* gradInput, const float* input, const float* gradOutput) const
  {
    *gradInput = *input < 0 ? - *gradOutput : *gradOutput;
  }
};

void THNN_CudaAbs_updateGradInput(THCState *state, THCudaTensor *input, THCudaTensor *gradOutput, THCudaTensor *gradInput)
{
  THAssert(THCudaTensor_checkGPU(state, 3, input, gradOutput, gradInput));
  THCudaTensor_resizeAs(state, gradInput, input);
  THCudaTensor_pointwiseApply3(state, gradInput, input, gradOutput, absupdateGradInput_functor());
}
