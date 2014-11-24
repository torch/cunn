
static int cunn_Linear_updateOutput(lua_State *L) {
  THCudaTensor *input = (THCudaTensor*)luaT_checkudata(L, 2, "torch.CudaTensor");
  
  THCudaTensor *weight = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "weight", "torch.CudaTensor");
  THCudaTensor *bias = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "bias", "torch.CudaTensor");
  THCudaTensor *ones = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "_ones", "torch.CudaTensor");
  THCudaTensor *output = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "output", "torch.CudaTensor");

  luaL_argcheck(L, input->nDimension == 1 || input->nDimension == 2, 2, "1D or 2D tensor expected");

  if (input->nDimension == 1) 
  {
    luaL_argcheck(L, input->size[0] == weight->size[1], 2, "invalid number of input units (input:size(1))");
    
    THCudaTensor_resize1d(output, bias->size[0]);
    THCudaTensor_copy(output, bias);
    THCudaTensor_addmv(output, 1, output, 1, weight, input);
  }
  else if ( input->nDimension == 2 ) 
  {
    long batchSize = input->size[0];
    long inputSize = weight->size[1];
    long outputSize = weight->size[0];
    
    luaL_argcheck(L, input->size[1] == inputSize, 2, "invalid number of input units (input:size(2))");

    THCudaTensor_resize2d(output, batchSize, outputSize);
    if (ones->size[0] != batchSize)
    {
      THCudaTensor_resize1d(ones, batchSize);
      THCudaTensor_fill(ones, 1);
    }
      
    THCudaTensor_zero(output);
    THCudaTensor_addr(output, 1, output, 1, ones, bias);    
    
    THCudaTensor_transpose(weight, NULL, 0, 1);
    THCudaTensor_addmm(output, 1, output, 1, input, weight);
    THCudaTensor_transpose(weight, NULL, 0, 1);
  }
  
  return 1;
}

static int cunn_Linear_updateGradInput(lua_State *L) {
  THCudaTensor *input = (THCudaTensor *)luaT_checkudata(L, 2, "torch.CudaTensor");
  THCudaTensor *gradOutput = (THCudaTensor *)luaT_checkudata(L, 3, "torch.CudaTensor");
  
  THCudaTensor *weight = (THCudaTensor *)luaT_getfieldcheckudata(L, 1, "weight", "torch.CudaTensor");
  THCudaTensor *gradInput = (THCudaTensor *)luaT_getfieldcheckudata(L, 1, "gradInput", "torch.CudaTensor");
  
  long nElement = THCudaTensor_nElement(gradInput);
  
  luaL_argcheck(L, input->nDimension == 1 || input->nDimension == 2, 2, "1D or 2D tensor expected");
  luaL_argcheck(L, gradOutput->nDimension == input->nDimension, 2, "input and gradOutput should have same number of dimensions");
  
  THCudaTensor_resizeAs(gradInput, input);
  if (THCudaTensor_nElement(gradInput) != nElement)
    THCudaTensor_zero(gradInput);
      
  if (input->nDimension == 1) 
  {    
    luaL_argcheck(L, input->size[0] == weight->size[1], 2, "invalid number of input units (input:size(1))");
    luaL_argcheck(L, gradOutput->size[0] == weight->size[0], 2, "invalid number of output units (gradOutput:size(1))");
    
    THCudaTensor_transpose(weight, NULL, 0, 1);
    THCudaTensor_addmv(gradInput, 0, gradInput, 1, weight, gradOutput);
    THCudaTensor_transpose(weight, NULL, 0, 1);
  }
  else
  {
    long inputSize = weight->size[1];
    long outputSize = weight->size[0];
    
    luaL_argcheck(L, input->size[1] == inputSize, 2, "invalid number of input units (input:size(2))");
    luaL_argcheck(L, gradOutput->size[1] == outputSize, 2, "invalid number of output units (gradOutput:size(2))");
    
    THCudaTensor_addmm(gradInput, 0, gradInput, 1, gradOutput, weight);
  }
  
  return 1;
}

static int cunn_Linear_accGradParameters(lua_State *L) {
  THCudaTensor *input = (THCudaTensor *)luaT_checkudata(L, 2, "torch.CudaTensor");
  THCudaTensor *gradOutput = (THCudaTensor *)luaT_checkudata(L, 3, "torch.CudaTensor");
  double scale = luaL_optnumber(L, 4, 1);

  THCudaTensor *gradWeight = (THCudaTensor *)luaT_getfieldcheckudata(L, 1, "gradWeight", "torch.CudaTensor");
  THCudaTensor *gradBias = (THCudaTensor *)luaT_getfieldcheckudata(L, 1, "gradBias", "torch.CudaTensor");
  THCudaTensor *ones = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "_ones", "torch.CudaTensor");

  luaL_argcheck(L, input->nDimension == 1 || input->nDimension == 2, 2, "1D or 2D tensor expected");
  luaL_argcheck(L, gradOutput->nDimension == input->nDimension, 2, "input and gradOutput should have same number of dimensions");
  
  if (input->nDimension == 1) 
  {
    luaL_argcheck(L, input->size[0] == gradWeight->size[1], 2, "invalid number of input units (input:size(1))");
    luaL_argcheck(L, gradOutput->size[0] == gradWeight->size[0], 2, "invalid number of output units (gradOutput:size(1))");
    
    THCudaTensor_addr(gradWeight, 1, gradWeight, scale, gradOutput, input);
    THCudaTensor_cadd(gradBias, gradBias, scale, gradOutput);
  }
  else 
  {
    long nframe = input->size[0];
    
    luaL_argcheck(L, input->size[1] == gradWeight->size[1], 2, "invalid number of input units (input:size(2))");
    luaL_argcheck(L, gradOutput->size[1] == gradWeight->size[0], 2, "invalid number of output units (gradOutput:size(2))");

    if (ones->size[0] != nframe)
    {
      THCudaTensor_resize1d(ones, nframe);
      THCudaTensor_fill(ones, 1);
    }
    
    THCudaTensor_transpose(gradOutput, NULL, 0, 1);
    THCudaTensor_addmm(gradWeight, 1, gradWeight, scale, gradOutput, input);
    THCudaTensor_addmv(gradBias, 1, gradBias, scale, gradOutput, ones);
    THCudaTensor_transpose(gradOutput, NULL, 0, 1);
    
  }

  return 0;
}

static const struct luaL_Reg cunn_Linear__ [] = {
  {"Linear_updateOutput", cunn_Linear_updateOutput},
  {"Linear_updateGradInput", cunn_Linear_updateGradInput},
  {"Linear_accGradParameters", cunn_Linear_accGradParameters},
  {NULL, NULL}
};

static void cunn_Linear_init(lua_State *L)
{
  luaT_pushmetatable(L, "torch.CudaTensor");
  luaT_registeratname(L, cunn_Linear__, "nn");
  lua_pop(L,1);
}
