#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "generic/THCUNN.h"
#else
TH_API void THNN_(Abs_updateOutput)(
          THCState *state,
          THCTensor *input,
          THCTensor *output);
TH_API void THNN_(Abs_updateGradInput)(
          THCState *state,
          THCTensor *input,
          THCTensor *gradOutput,
          THCTensor *gradInput);
#endif
