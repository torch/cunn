#include "THCUNN.h"
#include "common.h"
#include "im2col.h"

#include "THCHalf.h"
#include "THCHalfAutoNumerics.cuh"

#include "generic/SpatialDepthWiseConvolution.cu"
#include "THCGenerateFloatTypes.h"
