#include "luaT.h"
#include "THC.h"
#include "THLogAdd.h" /* DEBUG: WTF */

#include <thrust/transform.h>
#include <thrust/reduce.h>
#include <thrust/transform_reduce.h>
#include <thrust/functional.h>
#include <thrust/device_ptr.h>

#include "utils.c"

#include "ClassNLLCriterion.cu"
#include "HardTanh.cu"
#include "L1Cost.cu"
#include "Tanh.cu"
#include "Max.cu"
#include "Min.cu"
#include "LogSoftMax.cu"
#include "SoftMax.cu"
#include "TemporalConvolution.cu"
#include "TemporalMaxPooling.cu"
#include "SpatialConvolutionMM.cu"
#include "SpatialSubSampling.cu"
#include "SpatialMaxPooling.cu"
#include "SpatialAveragePooling.cu"
#include "SpatialAdaptiveMaxPooling.cu"
#include "Square.cu"
#include "Sqrt.cu"
#include "MultiMarginCriterion.cu"
#include "MSECriterion.cu"
#include "DistKLDivCriterion.cu"
#include "Threshold.cu"
#include "Sigmoid.cu"
#include "AbsCriterion.cu"
#include "Abs.cu"
#include "SoftPlus.cu"
#include "SpatialUpSamplingNearest.cu"
#include "VolumetricConvolution.cu"
#include "LogSigmoid.cu"
#include "PReLU.cu"

LUA_EXTERNC DLL_EXPORT int luaopen_libcunn(lua_State *L);

int luaopen_libcunn(lua_State *L)
{
  lua_newtable(L);
  cunn_ClassNLLCriterion_init(L);
  cunn_Tanh_init(L);
  cunn_Sigmoid_init(L);
  cunn_Max_init(L);
  cunn_Min_init(L);
  cunn_HardTanh_init(L);
  cunn_L1Cost_init(L);
  cunn_LogSoftMax_init(L);
  cunn_SoftMax_init(L);
  cunn_TemporalConvolution_init(L);
  cunn_TemporalMaxPooling_init(L);
  cunn_SpatialConvolutionMM_init(L);
  cunn_SpatialMaxPooling_init(L);
  cunn_SpatialAdaptiveMaxPooling_init(L);
  cunn_SpatialSubSampling_init(L);
  cunn_SpatialAveragePooling_init(L);
  cunn_MultiMarginCriterion_init(L);
  cunn_Square_init(L);
  cunn_Sqrt_init(L);
  cunn_Threshold_init(L);
  cunn_MSECriterion_init(L);
  cunn_AbsCriterion_init(L);
  cunn_DistKLDivCriterion_init(L);
  cunn_Abs_init(L);
  cunn_SoftPlus_init(L);
  cunn_SpatialUpSamplingNearest_init(L);
  cunn_VolumetricConvolution_init(L);
  cunn_LogSigmoid_init(L);
  cunn_PReLU_init(L);

  return 1;
}
