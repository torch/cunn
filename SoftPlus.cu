struct softPlusupdateOutput_functor
{
  const float threshold;
  const float beta;

  softPlusupdateOutput_functor(float threshold_, float beta_) : threshold(threshold_), beta(beta_) {}

  __host__ __device__ float operator()(const float& input) const
  {
    float betain = beta * input;
    return ((betain) > threshold) ? input : (1/beta) * log1p(exp(betain));
  }
};

static int cunn_SoftPlus_updateOutput(lua_State *L)
{
  THCudaTensor *input = (THCudaTensor*)luaT_checkudata(L, 2, "torch.CudaTensor");
  THCudaTensor *output = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "output", "torch.CudaTensor");
  float beta = luaT_getfieldchecknumber(L, 1, "beta");
  float threshold = luaT_getfieldchecknumber(L, 1, "threshold");
  long size = THCudaTensor_nElement(input);

  input = THCudaTensor_newContiguous(input);

  THCudaTensor_resizeAs(output, input);

  thrust::device_ptr<float> output_data(THCudaTensor_data(output));
  thrust::device_ptr<float> input_data(THCudaTensor_data(input));
  thrust::transform(input_data, input_data+size, output_data, 
                    softPlusupdateOutput_functor(threshold, beta));

  THCudaTensor_free(input);
  return 1;
}

struct softPlusupdateGradInput_functor
{
  const float threshold;
  const float beta;

  softPlusupdateGradInput_functor(float threshold_, float beta_) : threshold(threshold_), beta(beta_) {}

  __host__ __device__ float operator()(const float& output, const float& gradOutput) const
  {
    float betaout = beta * output;
    float exp_bo = exp(betaout);
    return ((betaout) > threshold) ? gradOutput : gradOutput * (exp_bo - 1) / exp_bo;
  }
};

static int cunn_SoftPlus_updateGradInput(lua_State *L)
{
  THCudaTensor *output = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "output", "torch.CudaTensor");
  THCudaTensor *input = (THCudaTensor*)luaT_checkudata(L, 2, "torch.CudaTensor");
  THCudaTensor *gradOutput = (THCudaTensor*)luaT_checkudata(L, 3, "torch.CudaTensor");
  THCudaTensor *gradInput = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "gradInput", "torch.CudaTensor");
  float beta = luaT_getfieldchecknumber(L, 1, "beta");
  float threshold = luaT_getfieldchecknumber(L, 1, "threshold");
  long size = THCudaTensor_nElement(output);

  gradOutput = THCudaTensor_newContiguous(gradOutput);
  THCudaTensor_resizeAs(gradInput, output);

  thrust::device_ptr<float> output_data(THCudaTensor_data(output));
  thrust::device_ptr<float> gradOutput_data(THCudaTensor_data(gradOutput));
  thrust::device_ptr<float> gradInput_data(THCudaTensor_data(gradInput));
  thrust::transform(output_data, output_data+size, gradOutput_data, gradInput_data, 
                    softPlusupdateGradInput_functor(threshold, beta));

  THCudaTensor_free(gradOutput);
  return 1;
}

static const struct luaL_Reg cunn_SoftPlus__ [] = {
  {"SoftPlus_updateOutput", cunn_SoftPlus_updateOutput},
  {"SoftPlus_updateGradInput", cunn_SoftPlus_updateGradInput},
  {NULL, NULL}
};

void cunn_SoftPlus_init(lua_State *L)
{
  luaT_pushmetatable(L, "torch.CudaTensor");
  luaT_registeratname(L, cunn_SoftPlus__, "nn");
  lua_pop(L,1);
}

