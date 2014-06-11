#define MINUS_LOG_THRESHOLD -18.42
#define LOGSOFTMAX_THREADS 128

__global__ void cunn_LogSoftMax_updateOutput_kernel(float *output, float *input, int nframe, int dim)
{
  __shared__ float buffer[LOGSOFTMAX_THREADS+1];
  int k = blockIdx.x;
  float *input_k = input + k*dim;
  float *output_k = output + k*dim;
  int tx = threadIdx.x;

  int i_start = threadIdx.x;
  int i_end = dim;
  int i_step = blockDim.x;

  // max?
  buffer[tx] = -FLT_MAX;
  for (int i=i_start; i<i_end; i+=i_step)
  {
    float z = input_k[i];
    if(buffer[tx] < z)
      buffer[tx] = z;
  }

  // reduce
  for (unsigned int stride = blockDim.x >> 1; stride > 0; stride >>= 1)
  {
    __syncthreads();
    if ((tx < stride) && (buffer[tx] < buffer[tx+stride]))
      buffer[tx] = buffer[tx+stride];
  }
  if (tx == 0)
  {
    float max_k = -FLT_MAX;
    if(max_k < buffer[0])
      max_k = buffer[0];
    buffer[LOGSOFTMAX_THREADS] = max_k;
  }

  __syncthreads();

  // logadd?
  float max_k = buffer[LOGSOFTMAX_THREADS];
  buffer[tx] = 0;
  for (int i=i_start; i<i_end; i+=i_step)
    buffer[tx] += expf(input_k[i]-max_k);

  // reduce
  for (unsigned int stride = blockDim.x >> 1; stride > 0; stride >>= 1)
  {
    __syncthreads();
    if (tx < stride)
      buffer[tx] += buffer[tx+stride];
  }
  if (tx == 0)
    buffer[LOGSOFTMAX_THREADS] = max_k + logf(buffer[0]);

  __syncthreads();

  // logsoftmax
  float logsum_k = buffer[LOGSOFTMAX_THREADS];
  for (int i=i_start; i<i_end; i+=i_step)
    output_k[i] = input_k[i] - logsum_k;
}


__global__ void cunn_LogSoftMax_updateGradInput_kernel(float *gradInput, float *output, float *gradOutput, int nframe, int dim)
{
  __shared__ float buffer[LOGSOFTMAX_THREADS];
  int k = blockIdx.x;
  float *gradInput_k = gradInput + k*dim;
  float *output_k = output + k*dim;
  float *gradOutput_k = gradOutput + k*dim;
  int tx = threadIdx.x;

  int i_end = dim;
  int i_step = blockDim.x;

  // sum?
  buffer[tx] = 0;
  for (int i=tx; i<i_end; i+=i_step)
    buffer[tx] += gradOutput_k[i];

  // reduce
  for (unsigned int stride = blockDim.x >> 1; stride > 0; stride >>= 1)
  {
    __syncthreads();
    if (tx < stride)
      buffer[tx] += buffer[tx+stride];
  }

  __syncthreads();

  float sum_k = buffer[0];
  for (int i=tx; i<i_end; i+=i_step)
    gradInput_k[i] = gradOutput_k[i] - __expf(output_k[i])*sum_k;
}

static int cunn_LogSoftMax_updateOutput(lua_State *L)
{
  THCudaTensor *input = (THCudaTensor*)luaT_checkudata(L, 2, "torch.CudaTensor");
  THCudaTensor *output = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "output", "torch.CudaTensor");

  input = THCudaTensor_newContiguous(input);
  THCudaTensor_resizeAs(output, input);

  if(input->nDimension == 1)
  {
    dim3 blocks(1);
    dim3 threads(LOGSOFTMAX_THREADS);
    cunn_LogSoftMax_updateOutput_kernel<<<blocks,threads>>>(THCudaTensor_data(output), THCudaTensor_data(input), 1, input->size[0]);
  }
  else if(input->nDimension == 2)
  {
    dim3 blocks(input->size[0]);
    dim3 threads(LOGSOFTMAX_THREADS);
    cunn_LogSoftMax_updateOutput_kernel<<<blocks,threads>>>(THCudaTensor_data(output), THCudaTensor_data(input), input->size[0], input->size[1]);
  }
  else
    THError("vector or matrix expected");

  cudaError errcode = cudaGetLastError();
  if(errcode != cudaSuccess)
    THError(cudaGetErrorString(errcode));

  THCudaTensor_free(input);
  return 1;
}

static int cunn_LogSoftMax_updateGradInput(lua_State *L)
{
  THCudaTensor *gradOutput = (THCudaTensor*)luaT_checkudata(L, 3, "torch.CudaTensor");
  THCudaTensor *output = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "output", "torch.CudaTensor");
  THCudaTensor *gradInput = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "gradInput", "torch.CudaTensor");

  output = THCudaTensor_newContiguous(output);
  gradOutput = THCudaTensor_newContiguous(gradOutput);

  THCudaTensor_resizeAs(gradInput, output);

  if(gradInput->nDimension == 1)
  {
    dim3 blocks(1);
    dim3 threads(LOGSOFTMAX_THREADS);

    cunn_LogSoftMax_updateGradInput_kernel<<<blocks,threads>>>(THCudaTensor_data(gradInput),
                                                        THCudaTensor_data(output),
                                                        THCudaTensor_data(gradOutput),
                                                        1, gradInput->size[0]);
  }
  else if(gradInput->nDimension == 2)
  {
    dim3 blocks(gradInput->size[0]);
    dim3 threads(LOGSOFTMAX_THREADS);

    cunn_LogSoftMax_updateGradInput_kernel<<<blocks,threads>>>(THCudaTensor_data(gradInput),
                                                        THCudaTensor_data(output),
                                                        THCudaTensor_data(gradOutput),
                                                        gradInput->size[0], gradInput->size[1]);
  }
  else
    THError("vector or matrix expected");

  cudaError errcode = cudaGetLastError();
  if(errcode != cudaSuccess)
    THError(cudaGetErrorString(errcode));

  THCudaTensor_free(gradOutput);
  THCudaTensor_free(output);
  return 1;
}

static const struct luaL_Reg cunn_LogSoftMax__ [] = {
  {"LogSoftMax_updateOutput", cunn_LogSoftMax_updateOutput},
  {"LogSoftMax_updateGradInput", cunn_LogSoftMax_updateGradInput},
  {NULL, NULL}
};

static void cunn_LogSoftMax_init(lua_State *L)
{
  luaT_pushmetatable(L, "torch.CudaTensor");
  luaT_registeratname(L, cunn_LogSoftMax__, "nn");
  lua_pop(L,1);
}
