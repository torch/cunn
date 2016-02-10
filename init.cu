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
  cunn_SpatialCrossMapLRN_init(L);
  cunn_SpatialBatchNormalization_init(L);
  cunn_SpatialConvolutionLocal_init(L);
  cunn_SpatialFullConvolution_init(L);
  cunn_SpatialMaxUnpooling_init(L);
  cunn_SpatialFractionalMaxPooling_init(L);
  cunn_SpatialSubSampling_init(L);
  cunn_SpatialUpSamplingNearest_init(L);
  return 1;
}
