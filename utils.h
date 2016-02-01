#ifndef CUNN_UTILS_H
#define CUNN_UTILS_H

extern "C"
{
#include <lua.h>
}
#include <luaT.h>
#include <THC/THC.h>

THCState* getCutorchState(lua_State* L);

void cunn_SpatialCrossMapLRN_init(lua_State *L);
void cunn_TemporalConvolution_init(lua_State *L);
void cunn_TemporalMaxPooling_init(lua_State *L);
void cunn_SpatialBatchNormalization_init(lua_State *L);
void cunn_SpatialConvolutionLocal_init(lua_State *L);
void cunn_SpatialFullConvolution_init(lua_State *L);
void cunn_SpatialMaxUnpooling_init(lua_State *L);
void cunn_SpatialFractionalMaxPooling_init(lua_State *L);
void cunn_SpatialSubSampling_init(lua_State *L);
void cunn_SpatialUpSamplingNearest_init(lua_State *L);

#endif
