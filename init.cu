#include "luaT.h"
#include "TH.h"
#include "THC.h"
#include "THLogAdd.h" /* DEBUG: WTF */

#include <thrust/transform.h>
#include <thrust/reduce.h>
#include <thrust/transform_reduce.h>
#include <thrust/functional.h>
#include <thrust/device_ptr.h>

#include "utils.h"

LUA_EXTERNC DLL_EXPORT int luaopen_libcunn(lua_State *L);

int luaopen_libcunn(lua_State *L)
{
  lua_newtable(L);
  cunn_ClassNLLCriterion_init(L);
  cunn_Tanh_init(L);
  cunn_ELU_init(L);
  cunn_Sigmoid_init(L);
  cunn_HardTanh_init(L);
  cunn_L1Cost_init(L);
  cunn_LogSoftMax_init(L);
  cunn_SoftMax_init(L);
  cunn_TemporalConvolution_init(L);
  cunn_TemporalMaxPooling_init(L);
  cunn_SpatialConvolutionMM_init(L);
  cunn_SpatialMaxPooling_init(L);
  cunn_SpatialFractionalMaxPooling_init(L);
  cunn_SpatialAdaptiveMaxPooling_init(L);
  cunn_SpatialSubSampling_init(L);
  cunn_SpatialAveragePooling_init(L);
  cunn_MultiMarginCriterion_init(L);
  cunn_MarginCriterion_init(L);
  cunn_Square_init(L);
  cunn_Sqrt_init(L);
  cunn_Threshold_init(L);
  cunn_MSECriterion_init(L);
  cunn_SmoothL1Criterion_init(L);
  cunn_AbsCriterion_init(L);
  cunn_DistKLDivCriterion_init(L);
  cunn_Abs_init(L);
  cunn_SoftPlus_init(L);
  cunn_SpatialUpSamplingNearest_init(L);
  cunn_VolumetricConvolution_init(L);
  cunn_VolumetricDeconvolution_init(L);
  cunn_VolumetricMaxPooling_init(L);
  cunn_VolumetricAveragePooling_init(L);
  cunn_LogSigmoid_init(L);
  cunn_PReLU_init(L);
  cunn_RReLU_init(L);
  cunn_LookupTable_init(L);

  return 1;
}
