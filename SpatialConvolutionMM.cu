// CUDA: grid stride looping
#define CUDA_KERNEL_LOOP(i, n)                        \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; \
       i < (n);                                       \
       i += blockDim.x * gridDim.x)

// Use 1024 threads per block, which requires cuda sm_2x or above
const int CUDA_NUM_THREADS = 1024;

// CUDA: number of blocks for threads.
inline int GET_BLOCKS(const int N) {
  return (N + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS;
}

// Kernel for fast unfold+copy
// (borrowed from Caffe: https://github.com/BVLC/caffe/blob/master/src/caffe/layers/conv_layer.cu)
__global__ void im2col_kernel(const int n, const float* data_im,
                              const int height, const int width, const int ksize_h, const int ksize_w, const int pad_h,
			      const int pad_w, const int stride_h, const int stride_w, const int height_col, const int width_col,
                              float* data_col) {
  CUDA_KERNEL_LOOP(index, n) {
    int w_out = index % width_col;
    index /= width_col;
    int h_out = index % height_col;
    int channel_in = index / height_col;
    int channel_out = channel_in * ksize_h * ksize_w;
    int h_in = h_out * stride_h - pad_h;
    int w_in = w_out * stride_w - pad_w;
    data_col += (channel_out * height_col + h_out) * width_col + w_out;
    data_im += (channel_in * height + h_in) * width + w_in;
    for (int i = 0; i < ksize_h; ++i) {
      for (int j = 0; j < ksize_w; ++j) {
        int h = h_in + i;
        int w = w_in + j;
        *data_col = (h >= 0 && w >= 0 && h < height && w < width) ?
                                             data_im[i * width + j] : 0;
        data_col += height_col * width_col;
      }
    }
  }
}

void im2col(const float* data_im, const int channels,
            const int height, const int width, const int ksize_h, const int ksize_w, const int pad_h,
	    const int pad_w, const int stride_h, const int stride_w, float* data_col) {
  // We are going to launch channels * height_col * width_col kernels, each
  // kernel responsible for copying a single-channel grid.
  int height_col = (height + 2 * pad_h - ksize_h) / stride_h + 1;
  int width_col = (width + 2 * pad_w - ksize_w) / stride_w + 1;
  int num_kernels = channels * height_col * width_col;
  // Launch
  im2col_kernel <<<GET_BLOCKS(num_kernels), CUDA_NUM_THREADS>>> (
                                                                 num_kernels, data_im, height, width, ksize_h, ksize_w, 
                                                                 pad_h, pad_w, stride_h, stride_w, 
                                                                 height_col, width_col, data_col
                                                                 );
}

__global__ void col2im_kernel(const int n, const float* data_col,
                              const int height, const int width, const int channels, const int patch_h, const int patch_w,
                              const int pad_h, const int pad_w, const int stride_h, const int stride_w, const int height_col, const int width_col,
                              float* data_im) {
  CUDA_KERNEL_LOOP(index, n) {
    float val = 0;
    int w = index % width + pad_w;
    int h = (index / width) % height + pad_h;
    int c = index / (width * height);
    // compute the start and end of the output
    int w_col_start = (w < patch_w) ? 0 : (w - patch_w) / stride_w + 1;
    int w_col_end = min(w / stride_w + 1, width_col);
    int h_col_start = (h < patch_h) ? 0 : (h - patch_h) / stride_h + 1;
    int h_col_end = min(h / stride_h + 1, height_col);
    /*
      for (int h_col = h_col_start; h_col < h_col_end; ++h_col) {
      for (int w_col = w_col_start; w_col < w_col_end; ++w_col) {
      // the col location: [c * width * height + h_out, w_out]
      int c_col = c * patch_h * patch_w + (h - h_col * stride_h) * ksize + (w - w_col * stride_w);
      val += data_col[(c_col * height_col + h_col) * width_col + w_col];
      }
      }
    */
    // equivalent implementation
    int offset = (c * patch_h * patch_w + h * patch_w + w) * height_col * width_col;
    int coeff_h_col = (1 - stride_h * patch_w * height_col) * width_col;
    int coeff_w_col = (1 - stride_w * height_col * width_col);
    for (int h_col = h_col_start; h_col < h_col_end; ++h_col) {
      for (int w_col = w_col_start; w_col < w_col_end; ++w_col) {
        val += data_col[offset + h_col * coeff_h_col + w_col * coeff_w_col];
      }
    }
    data_im[index] = val;
  }
}

void col2im(const float* data_col, const int channels,
            const int height, const int width, const int patch_h, const int patch_w, const int pad_h,
            const int pad_w, const int stride_h, const int stride_w, float* data_im) {
  int height_col = (height + 2 * pad_h - patch_h) / stride_h + 1;
  int width_col = (width + 2 * pad_w - patch_w) / stride_w + 1;
  int num_kernels = channels * height * width;
  // To avoid involving atomic operations, we will launch one kernel per
  // bottom dimension, and then in the kernel add up the top dimensions.
  col2im_kernel <<<GET_BLOCKS(num_kernels), CUDA_NUM_THREADS>>> (
                                                                 num_kernels, data_col, height, width, channels, 
								 patch_h, patch_w, pad_h, pad_w, stride_h, stride_w,
                                                                 height_col, width_col, data_im
                                                                 );
}

static int cunn_SpatialConvolutionMM_updateOutput(lua_State *L) {
  // Input
  THCudaTensor *input = (THCudaTensor*)luaT_checkudata(L, 2, "torch.CudaTensor");

  // Params:
  int dW = luaT_getfieldcheckint(L, 1, "dW");
  int dH = luaT_getfieldcheckint(L, 1, "dH");
  int kW = luaT_getfieldcheckint(L, 1, "kW");
  int kH = luaT_getfieldcheckint(L, 1, "kH");
  int nInputPlane = luaT_getfieldcheckint(L, 1, "nInputPlane");
  int nOutputPlane = luaT_getfieldcheckint(L, 1, "nOutputPlane");
  int padding = luaT_getfieldcheckint(L, 1, "padding");

  THCudaTensor *weight = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "weight", "torch.CudaTensor");
  THCudaTensor *bias = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "bias", "torch.CudaTensor");
  THCudaTensor *columns = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "finput", "torch.CudaTensor");
  THCudaTensor *ones = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "fgradInput", "torch.CudaTensor");
  THCudaTensor *output = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "output", "torch.CudaTensor");

  luaL_argcheck(L, input->nDimension == 3 || input->nDimension == 4, 2, "3D or 4D (batch mode) tensor is expected");

  int batch = 1;
  if (input->nDimension == 3) {
    // Force batch
    batch = 0;
    THCudaTensor_resize4d(input, 1, input->size[0], input->size[1], input->size[2]);
  }

  long inputWidth   = input->size[3];
  long inputHeight  = input->size[2];
  long outputWidth  = (inputWidth + 2*padding - kW) / dW + 1;
  long outputHeight = (inputHeight + 2*padding - kH) / dH + 1;

  
  // Batch size + input planes
  long batchSize = input->size[0];

  // Resize output
  THCudaTensor_resize4d(output, batchSize, nOutputPlane, outputHeight, outputWidth);

  // Resize temporary columns
  THCudaTensor_resize2d(columns, nInputPlane*kW*kH, outputHeight*outputWidth);

  // Define a buffer of ones, for bias accumulation
  // Note: this buffer can be shared with other modules, it only ever gets increased, 
  // and always contains ones.
  if (ones->nDimension != 2 || ones->size[0]*ones->size[1] < outputHeight*outputWidth) {
    // Resize plane and fill with ones...
    THCudaTensor_resize2d(ones, outputHeight, outputWidth);
    THCudaTensor_fill(ones, 1);
  }

  // Helpers
  THCudaTensor *input_n = THCudaTensor_new();
  THCudaTensor *output_n = THCudaTensor_new();

  // For each elt in batch, do:
  for (int elt = 0; elt < batchSize; elt ++) {
    // Matrix mulitply per output:
    THCudaTensor_select(input_n, input, 0, elt);
    THCudaTensor_select(output_n, output, 0, elt);
   
    // Do Bias first: 
    // M,N,K are dims of matrix A and B
    // (see http://docs.nvidia.com/cuda/cublas/#cublas-lt-t-gt-gemm)
    long m_ = nOutputPlane;
    long n_ = outputHeight * outputWidth;
    long k_ = 1;

    // Do GEMM (note: this is a bit confusing because gemm assumes column-major matrices)
    cublasSgemm(
      't', 'n',
      n_, m_, k_,
      1, 
      THCudaTensor_data(ones), k_,
      THCudaTensor_data(bias), k_,
      0,
      THCudaTensor_data(output_n), n_
    );

    // Extract columns:
    im2col(
           THCudaTensor_data(input_n),
           nInputPlane, inputHeight, inputWidth, kH, kW, padding, padding, dH, dW, 
           THCudaTensor_data(columns)
           );

    // M,N,K are dims of matrix A and B
    // (see http://docs.nvidia.com/cuda/cublas/#cublas-lt-t-gt-gemm)
    long m = weight->size[0];
    long n = columns->size[1];
    long k = weight->size[1];

    // Do GEMM (note: this is a bit confusing because gemm assumes column-major matrices)
    cublasSgemm(
                'n', 'n',
                n, m, k,
                1, 
                THCudaTensor_data(columns), n,
                THCudaTensor_data(weight), k,
                1,
                THCudaTensor_data(output_n), n
                );
    THCublasCheck();
  }

  // Free
  THCudaTensor_free(input_n);
  THCudaTensor_free(output_n);
  
  // Resize output
  if (batch == 0) {
    THCudaTensor_resize3d(output, nOutputPlane, outputHeight, outputWidth);
    THCudaTensor_resize3d(input, nInputPlane, inputHeight, inputWidth);
  }

  // return output
  return 1;
}

static int cunn_SpatialConvolutionMM_updateGradInput(lua_State *L) {
  // Inputs
  THCudaTensor *input = (THCudaTensor *)luaT_checkudata(L, 2, "torch.CudaTensor");
  THCudaTensor *gradOutput = (THCudaTensor *)luaT_checkudata(L, 3, "torch.CudaTensor");

  // Params
  int dW = luaT_getfieldcheckint(L, 1, "dW");
  int dH = luaT_getfieldcheckint(L, 1, "dH");
  int kW = luaT_getfieldcheckint(L, 1, "kW");
  int kH = luaT_getfieldcheckint(L, 1, "kH");
  int nInputPlane = luaT_getfieldcheckint(L, 1, "nInputPlane");
  int nOutputPlane = luaT_getfieldcheckint(L, 1, "nOutputPlane");
  int padding = luaT_getfieldcheckint(L, 1, "padding");

  THCudaTensor *weight = (THCudaTensor *)luaT_getfieldcheckudata(L, 1, "weight", "torch.CudaTensor");
  THCudaTensor *gradColumns = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "finput", "torch.CudaTensor");
  THCudaTensor *gradInput = (THCudaTensor *)luaT_getfieldcheckudata(L, 1, "gradInput", "torch.CudaTensor");
    
  luaL_argcheck(L, input->nDimension == 3 || input->nDimension == 4, 2, "3D or 4D (batch mode) tensor is expected");
  
  int batch = 1;
  if (input->nDimension == 3) {
    // Force batch
    batch = 0;
    THCudaTensor_resize4d(input, 1, input->size[0], input->size[1], input->size[2]);
    THCudaTensor_resize4d(gradOutput, 1, gradOutput->size[0], gradOutput->size[1], gradOutput->size[2]);
  }
    
  long inputWidth   = input->size[3];
  long inputHeight  = input->size[2];
  long outputWidth  = (inputWidth + 2*padding - kW) / dW + 1;
  long outputHeight = (inputHeight + 2*padding - kH) / dH + 1;

  // Batch size + input planes
  long batchSize = input->size[0];

  // Resize output
  THCudaTensor_resize4d(gradInput, batchSize, nInputPlane, inputHeight, inputWidth);

  // Resize temporary columns
  THCudaTensor_resize2d(gradColumns, nInputPlane*kW*kH, outputHeight*outputWidth);
      
  // Helpers
  THCudaTensor *input_n = THCudaTensor_new();
  THCudaTensor *gradInput_n = THCudaTensor_new();
  THCudaTensor *gradOutput_n = THCudaTensor_new();

  // For each elt in batch, do:
  for (int elt = 0; elt < batchSize; elt ++) {
    // Matrix mulitply per sample:
    THCudaTensor_select(input_n, input, 0, elt);
    THCudaTensor_select(gradInput_n, gradInput, 0, elt);
    THCudaTensor_select(gradOutput_n, gradOutput, 0, elt);
          
    // M,N,K are dims of matrix A and B
    // (see http://docs.nvidia.com/cuda/cublas/#cublas-lt-t-gt-gemm)
    long m = weight->size[1];
    long n = gradColumns->size[1];
    long k = weight->size[0];
         
    // Do GEMM (note: this is a bit confusing because gemm assumes column-major matrices)
    cublasSgemm(
                'n', 't',
                n, m, k,
                1, 
                THCudaTensor_data(gradOutput_n), n,
                THCudaTensor_data(weight), m,
                0,
                THCudaTensor_data(gradColumns), n
                );
    THCublasCheck();
          
    // Unpack columns back into input:
    col2im(
           THCudaTensor_data(gradColumns),
           nInputPlane, inputHeight, inputWidth, kH, kW, padding, padding, dH, dW, 
           THCudaTensor_data(gradInput_n)
           );
  }

  // Free
  THCudaTensor_free(input_n);
  THCudaTensor_free(gradInput_n);
  THCudaTensor_free(gradOutput_n);

  // Resize output
  if (batch == 0) {
    THCudaTensor_resize3d(gradOutput, nOutputPlane, outputHeight, outputWidth);
    THCudaTensor_resize3d(input, nInputPlane, inputHeight, inputWidth);
    THCudaTensor_resize3d(gradInput, nInputPlane, inputHeight, inputWidth);
  }

  // Return gradInput
  return 1;
}

static int cunn_SpatialConvolutionMM_accGradParameters(lua_State *L) {
  // Inputs
  THCudaTensor *input = (THCudaTensor *)luaT_checkudata(L, 2, "torch.CudaTensor");
  THCudaTensor *gradOutput = (THCudaTensor *)luaT_checkudata(L, 3, "torch.CudaTensor");

  // Params
  int dW = luaT_getfieldcheckint(L, 1, "dW");
  int dH = luaT_getfieldcheckint(L, 1, "dH");
  int kW = luaT_getfieldcheckint(L, 1, "kW");
  int kH = luaT_getfieldcheckint(L, 1, "kH");
  int nInputPlane = luaT_getfieldcheckint(L, 1, "nInputPlane");
  int nOutputPlane = luaT_getfieldcheckint(L, 1, "nOutputPlane");
  int padding = luaT_getfieldcheckint(L, 1, "padding");
  float scale = luaL_optnumber(L, 4, 1);

  THCudaTensor *gradWeight = (THCudaTensor *)luaT_getfieldcheckudata(L, 1, "gradWeight", "torch.CudaTensor");
  THCudaTensor *gradBias = (THCudaTensor *)luaT_getfieldcheckudata(L, 1, "gradBias", "torch.CudaTensor");
  THCudaTensor *columns = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "finput", "torch.CudaTensor");
  THCudaTensor *ones = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "fgradInput", "torch.CudaTensor");

  luaL_argcheck(L, input->nDimension == 3 || input->nDimension == 4, 2, "3D or 4D (batch mode) tensor is expected");
  
  int batch = 1;
  if (input->nDimension == 3) {
    // Force batch
    batch = 0;
    THCudaTensor_resize4d(input, 1, input->size[0], input->size[1], input->size[2]);
    THCudaTensor_resize4d(gradOutput, 1, gradOutput->size[0], gradOutput->size[1], gradOutput->size[2]);
  }

  long inputWidth   = input->size[3];
  long inputHeight  = input->size[2];
  long outputWidth  = (inputWidth + 2*padding - kW) / dW + 1;
  long outputHeight = (inputHeight + 2*padding - kH) / dH + 1;



  // Batch size + input planes
  long batchSize = input->size[0];
  
  // Define a buffer of ones, for bias accumulation
  if (ones->nDimension != 2 || ones->size[0]*ones->size[1] < outputHeight*outputWidth) {
    // Resize plane and fill with ones...
    THCudaTensor_resize2d(ones, outputHeight, outputWidth);
    THCudaTensor_fill(ones, 1);
  }

  // Resize temporary columns
  THCudaTensor_resize2d(columns, nInputPlane*kW*kH, outputHeight*outputWidth);
      
  // Helpers
  THCudaTensor *input_n = THCudaTensor_new();
  THCudaTensor *gradOutput_n = THCudaTensor_new();

  // For each elt in batch, do:
  for (int elt = 0; elt < batchSize; elt ++) {
    // Matrix mulitply per output:
    THCudaTensor_select(input_n, input, 0, elt);
    THCudaTensor_select(gradOutput_n, gradOutput, 0, elt);

    // Extract columns:
    im2col(
           THCudaTensor_data(input_n),
           nInputPlane, inputHeight, inputWidth, kH, kW, padding, padding, dH, dW, 
           THCudaTensor_data(columns)
           );

    // M,N,K are dims of matrix A and B
    // (see http://docs.nvidia.com/cuda/cublas/#cublas-lt-t-gt-gemm)
    long m = gradWeight->size[0];
    long n = gradWeight->size[1];
    long k = columns->size[1];

    // Do GEMM (note: this is a bit confusing because gemm assumes column-major matrices)
    cublasSgemm(
                't', 'n',
                n, m, k,
                scale,
                THCudaTensor_data(columns), k,
                THCudaTensor_data(gradOutput_n), k,
                1,
                THCudaTensor_data(gradWeight), n
                );
    THCublasCheck();
    
    // Do Bias: 
    // M,N,K are dims of matrix A and B
    // (see http://docs.nvidia.com/cuda/cublas/#cublas-lt-t-gt-gemm)
    long m_ = nOutputPlane;
    long n_ = 1;
    long k_ = outputHeight * outputWidth;

    // Do GEMM (note: this is a bit confusing because gemm assumes column-major matrices)
    cublasSgemm(
      'n', 'n',
      n_, m_, k_,
      scale, 
      THCudaTensor_data(ones), n_,
      THCudaTensor_data(gradOutput_n), k_,
      1,
      THCudaTensor_data(gradBias), n_
    );
  }

  // Free
  THCudaTensor_free(input_n);
  THCudaTensor_free(gradOutput_n);
  
  // Resize 
  if (batch == 0) {
    THCudaTensor_resize3d(gradOutput, nOutputPlane, outputHeight, outputWidth);
    THCudaTensor_resize3d(input, nInputPlane, inputHeight, inputWidth);
  }

  // Return nothing
  return 0;
}

static const struct luaL_Reg cunn_SpatialConvolutionMM__ [] = {
  {"SpatialConvolutionMM_updateOutput", cunn_SpatialConvolutionMM_updateOutput},
  {"SpatialConvolutionMM_updateGradInput", cunn_SpatialConvolutionMM_updateGradInput},
  {"SpatialConvolutionMM_accGradParameters", cunn_SpatialConvolutionMM_accGradParameters},
  {NULL, NULL}
};

static void cunn_SpatialConvolutionMM_init(lua_State *L)
{
  luaT_pushmetatable(L, "torch.CudaTensor");
  luaT_registeratname(L, cunn_SpatialConvolutionMM__, "nn");
  lua_pop(L,1);
}
