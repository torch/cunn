
static int cunn_Linear_updateOutput(lua_State *L) {
  THCudaTensor *input = (THCudaTensor*)luaT_checkudata(L, 2, "torch.CudaTensor");
  
  int forceAsync = luaT_getfieldcheckboolean(L, 1, "forceAsync");
  THCudaTensor *weight = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "weight", "torch.CudaTensor");
  THCudaTensor *bias = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "bias", "torch.CudaTensor");
  THCudaTensor *ones = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "_ones", "torch.CudaTensor");
  THCudaTensor *_input = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "_input", "torch.CudaTensor");
  THCudaTensor *output = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "output", "torch.CudaTensor");

  luaL_argcheck(L, input->nDimension == 1 || input->nDimension == 2, 2, "1D or 2D tensor expected");

  if (input->nDimension == 1) 
  {
    luaL_argcheck(L, input->size[0] == weight->size[1], 2, "invalid number of input units (input:size(1))");
    
    THCudaTensor_resize1d(output, bias->size[0]);
    THCudaTensor_copy(output, bias);
    THCudaTensor_addmv(output, 1, 1, weight, input);
  }
  else if ( input->nDimension == 2 ) 
  {
    long batchSize = input->size[0];
    long inputSize = weight->size[1];
    long outputSize = weight->size[0];
    THCudaTensor* weightT = THCudaTensor_newTranspose(weight, 0, 1);
    
    luaL_argcheck(L, input->size[1] == inputSize, 2, "invalid number of input units (input:size(2))");

    THCudaTensor_resize2d(output, batchSize, outputSize);
    if (ones->size[0] != batchSize)
    {
      THCudaTensor_resize1d(ones, batchSize);
      THCudaTensor_fill(ones, 1);
    }
      
    THCudaTensor_zero(output);
    THCudaTensor_addr(output, 1, ones, bias);    
    
    long newBatchSize = -1;
    if (forceAsync == 1) 
    {
      // in such cases cublasSgemm is blocking...
      if ( (batchSize < 65) && ( (outputSize > 1399) || (inputSize > 1399) ) )
      {
        if ( ( (outputSize < 385) || (inputSize < 385) ) && batchSize < 65 )
          newBatchSize = 65;
        else
          newBatchSize = 39;
      }
    }
    
    if (newBatchSize > 0)
    {
      THCudaTensor_resize2d(output, newBatchSize, outputSize); 
      THCudaTensor_resize2d(_input, batchSize, inputSize);
      THCudaTensor_copy(_input, input);
      THCudaTensor_resize2d(_input, newBatchSize, inputSize);
      THCudaTensor_addmm(output, 1, 1, _input, weightT);
      THCudaTensor_resize2d(output, batchSize, outputSize);
    }
    else
    {
      THCudaTensor_addmm(output, 1, 1, input, weightT);
    }
    
    
    
    THCudaTensor_free(weightT);
  }
  
  return 1;
}

static int cunn_Linear_updateGradInput(lua_State *L) {
  THCudaTensor *input = (THCudaTensor *)luaT_checkudata(L, 2, "torch.CudaTensor");
  THCudaTensor *gradOutput = (THCudaTensor *)luaT_checkudata(L, 3, "torch.CudaTensor");
  
  int forceAsync = luaT_getfieldcheckboolean(L, 1, "forceAsync");
  THCudaTensor *weight = (THCudaTensor *)luaT_getfieldcheckudata(L, 1, "weight", "torch.CudaTensor");
  THCudaTensor *_gradOutput = (THCudaTensor *)luaT_getfieldcheckudata(L, 1, "_gradOutput", "torch.CudaTensor");
  THCudaTensor *gradInput = (THCudaTensor *)luaT_getfieldcheckudata(L, 1, "gradInput", "torch.CudaTensor");
  
  long nElement = THCudaTensor_nElement(gradInput);
  
  luaL_argcheck(L, input->nDimension == 1 || input->nDimension == 2, 2, "1D or 2D tensor expected");
  luaL_argcheck(L, gradOutput->nDimension == input->nDimension, 2, "input and gradOutput should have same number of dimensions");
  
  if (input->nDimension == 1) 
  {
    THCudaTensor *weightT = THCudaTensor_newTranspose(weight, 0, 1);
    
    luaL_argcheck(L, input->size[0] == weight->size[1], 2, "invalid number of input units (input:size(1))");
    luaL_argcheck(L, gradOutput->size[0] == weight->size[0], 2, "invalid number of output units (gradOutput:size(1))");
    
    THCudaTensor_resizeAs(gradInput, input);
    if (THCudaTensor_nElement(gradInput) != nElement)
      THCudaTensor_zero(gradInput);
    
    THCudaTensor_addmv(gradInput, 0, 1, weightT, gradOutput);
    THCudaTensor_free(weightT);
  }
  else
  {
    long batchSize = input->size[0];
    long inputSize = weight->size[1];
    long outputSize = weight->size[0];
    
    luaL_argcheck(L, input->size[1] == weight->size[1], 2, "invalid number of input units (input:size(2))");
    luaL_argcheck(L, gradOutput->size[1] == weight->size[0], 2, "invalid number of output units (gradOutput:size(2))");
    
    long newBatchSize = -1;
    if (forceAsync == 1) 
    {
      // in such cases cublasSgemm is blocking...
      if ( (batchSize < 65) && ( (outputSize > 1399) || (inputSize > 1399) ) )
      {
        if ( ( (outputSize < 385) || (inputSize < 385) ) && batchSize < 65 )
          newBatchSize = 65;
        else
          newBatchSize = 39;
      }
    }
    
    if (newBatchSize > 0)
    {
      THCudaTensor_resize2d(gradInput, newBatchSize, inputSize); 
      THCudaTensor_zero(gradInput);
      THCudaTensor_resize2d(_gradOutput, batchSize, outputSize);
      THCudaTensor_copy(_gradOutput, gradOutput);
      THCudaTensor_resize2d(_gradOutput, newBatchSize, outputSize);
      THCudaTensor_addmm(gradInput, 0, 1, _gradOutput, weight);
      THCudaTensor_resize2d(gradInput, batchSize, inputSize);
    }
    else
    {
      THCudaTensor_resizeAs(gradInput, input);
      if (THCudaTensor_nElement(gradInput) != nElement)
        THCudaTensor_zero(gradInput);
      THCudaTensor_addmm(gradInput, 0, 1, gradOutput, weight);
    }
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
    
    THCudaTensor_addr(gradWeight, scale, gradOutput, input);
    THCudaTensor_cadd(gradBias, scale, gradOutput);
  }
  else 
  {
    long nframe = input->size[0];
    THCudaTensor* gradOutputT = THCudaTensor_newTranspose(gradOutput, 0, 1);
    
    luaL_argcheck(L, input->size[1] == gradWeight->size[1], 2, "invalid number of input units (input:size(2))");
    luaL_argcheck(L, gradOutput->size[1] == gradWeight->size[0], 2, "invalid number of output units (gradOutput:size(2))");

    if (ones->size[0] != nframe)
    {
      THCudaTensor_resize1d(ones, nframe);
      THCudaTensor_fill(ones, 1);
    }
    THCudaTensor_addmm(gradWeight, 1, scale, gradOutputT, input);
    THCudaTensor_addmv(gradBias, 1, scale, gradOutputT, ones);
    
    THCudaTensor_free(gradOutputT);
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
