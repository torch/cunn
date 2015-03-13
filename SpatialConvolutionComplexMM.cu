// CUDA: grid stride looping
#define CUDA_KERNEL_LOOP(i, n)                        \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; \
      i < (n);                                       \
      i += blockDim.x * gridDim.x)

// Use 1024 threads per block, which requires cuda sm_2x or above
const int CUDA_NUM_THREADS_COMPLEX = 1024;

// CUDA: number of blocks for threads.
inline int GET_BLOCKS_COMPLEX(const int N) {
  return (N + CUDA_NUM_THREADS_COMPLEX - 1) / CUDA_NUM_THREADS_COMPLEX;
}

// Kernel for fast unfold+copy
// (borrowed from Caffe: https://github.com/BVLC/caffe/blob/master/src/caffe/layers/conv_layer.cu)
__global__ void im2col_kernel_complex(const int n, const cuComplex* data_im,
    const int height, const int width, const int ksize_h, const int ksize_w, 
    const int pad_h, const int pad_w, const int stride_h, const int stride_w, 
    const int height_col, const int width_col,
    cuComplex* data_col) {
  cuComplex zeros = make_float2(0, 0);
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
          data_im[i * width + j] : zeros;
        data_col += height_col * width_col;
      }
    }
  }
}

void im2col_complex(const cuComplex* data_im, const int channels,
    const int height, const int width, const int ksize_h, const int ksize_w, 
    const int pad_h, const int pad_w, const int stride_h, const int stride_w, 
    cuComplex* data_col) {
  // We are going to launch channels * height_col * width_col kernels, each
  // kernel responsible for copying a single-channel grid.
  int height_col = (height + 2 * pad_h - ksize_h) / stride_h + 1;
  int width_col = (width + 2 * pad_w - ksize_w) / stride_w + 1;
  int num_kernels = channels * height_col * width_col;
  // Launch
  im2col_kernel_complex <<<GET_BLOCKS_COMPLEX(num_kernels), CUDA_NUM_THREADS_COMPLEX>>> (
      num_kernels, data_im, height, width, ksize_h, ksize_w,
      pad_h, pad_w, stride_h, stride_w,
      height_col, width_col, data_col
  );
}

__global__ void col2im_kernel_complex(const int n, const cuComplex* data_col,
    const int height, const int width, const int channels, const int patch_h, 
    const int patch_w, const int pad_h, const int pad_w, const int stride_h, 
    const int stride_w, const int height_col, const int width_col,
    cuComplex* data_im) {
  CUDA_KERNEL_LOOP(index, n) {
    cuComplex val;
    val.x = 0;
    val.y = 0;
    int w = index % width + pad_w;
    int h = (index / width) % height + pad_h;
    int c = index / (width * height);
    // compute the start and end of the output
    int w_col_start = (w < patch_w) ? 0 : (w - patch_w) / stride_w + 1;
    int w_col_end = min(w / stride_w + 1, width_col);
    int h_col_start = (h < patch_h) ? 0 : (h - patch_h) / stride_h + 1;
    int h_col_end = min(h / stride_h + 1, height_col);
    int offset = (c * patch_h * patch_w + h * patch_w + w) * height_col * 
      width_col;
    int coeff_h_col = (1 - stride_h * patch_w * height_col) * width_col;
    int coeff_w_col = (1 - stride_w * height_col * width_col);
    for (int h_col = h_col_start; h_col < h_col_end; ++h_col) {
      for (int w_col = w_col_start; w_col < w_col_end; ++w_col) {
        val.x += 
          data_col[offset + h_col * coeff_h_col + w_col * coeff_w_col].x;
        val.y += 
          data_col[offset + h_col * coeff_h_col + w_col * coeff_w_col].y;
      }
    }
    data_im[index] = val;
  }
}

void col2im_complex(const cuComplex* data_col, const int channels, 
    const int height, const int width, const int patch_h, const int patch_w, 
    const int pad_h, const int pad_w, const int stride_h, const int stride_w, 
    cuComplex* data_im) {
  int height_col = (height + 2 * pad_h - patch_h) / stride_h + 1;
  int width_col = (width + 2 * pad_w - patch_w) / stride_w + 1;
  int num_kernels = channels * height * width;
  // To avoid involving atomic operations, we will launch one kernel per
  // bottom dimension, and then in the kernel add up the top dimensions.
  col2im_kernel_complex <<<GET_BLOCKS_COMPLEX(num_kernels), CUDA_NUM_THREADS_COMPLEX>>> (
      num_kernels, data_col, height, width, channels,
      patch_h, patch_w, pad_h, pad_w, stride_h, stride_w,
      height_col, width_col, data_im
  );
}

static int cunn_SpatialConvolutionComplexMM_updateOutput(lua_State *L) {
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

  THCudaTensor *weight = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "weight",
    "torch.CudaTensor");
  THCudaTensor *bias = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "bias", 
    "torch.CudaTensor");
  THCudaTensor *columns = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, 
    "finput", "torch.CudaTensor");
  THCudaTensor *ones = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, 
    "fgradInput", "torch.CudaTensor");
  THCudaTensor *output = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "output",
    "torch.CudaTensor");

  const int device = THCudaTensor_getDevice(weight);
  luaL_argcheck(L, THCudaTensor_getDevice(bias) == device, 1,
                "weight and bias need to be on the same device");
  luaL_argcheck(L, THCudaTensor_getDevice(output) == device ||
                THCudaTensor_getDevice(output) == -1, 1,
                "weight and output need to be on the same device");
  luaL_argcheck(L, THCudaTensor_getDevice(input) == device, 2,
                "weight and input need to be on the same device");

  luaL_argcheck(L, input->nDimension == 4 || input->nDimension == 5, 2, 
    "4D or 5D (batch mode) tensor is expected");
  luaL_argcheck(L, input->size[input->nDimension-1] == 2, 2, 
    "last input dimension size is not 2");

  int batch = 1;
  if (input->nDimension == 4) {
    // Force batch
    batch = 0;
    THCudaTensor_resize5d(input, 1, input->size[0], input->size[1], 
      input->size[2], 2);
  }

  long inputWidth   = input->size[3];
  long inputHeight  = input->size[2];
  long outputWidth  = (inputWidth + 2*padding - kW) / dW + 1;
  long outputHeight = (inputHeight + 2*padding - kH) / dH + 1;


  // Batch size + input planes
  long batchSize = input->size[0];

  // Resize output
  THCudaTensor_resize5d(output, batchSize, nOutputPlane, outputHeight, 
    outputWidth, 2);

  // Resize temporary columns
  THCudaTensor_resize2d(columns, nInputPlane*kW*kH*2, 
    outputHeight*outputWidth*2);

  // Define a buffer of ones, for bias accumulation
  // Note: this buffer can be shared with other modules, it only ever gets 
  // increased, and always contains ones.
  if (ones->nDimension != 3 || 
      ones->size[0]*ones->size[1] < outputHeight*outputWidth) {
    // Resize plane and fill with ones (real) and zeros (imag)...
    THCudaTensor_resize3d(ones, outputHeight, outputWidth, 2);
    THCudaTensor *real = THCudaTensor_new();
    THCudaTensor *imag = THCudaTensor_new();
    THCudaTensor_narrow(real, ones, 2, 0, 1);
    THCudaTensor_narrow(imag, ones, 2, 1, 1);
    THCudaTensor_fill(real, 1);
    THCudaTensor_fill(imag, 0);
    THCudaTensor_free(real);
    THCudaTensor_free(imag);
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

    // Do GEMM (note: this is a bit confusing because gemm assumes 
    // column-major matrices)
    cuComplex alpha = make_float2(1, 0);
    cuComplex beta = make_float2(0, 0);
    THCudaBlas_cgemm(
        't', 'n',
        n_, m_, k_,
        alpha,
        (cuComplex*)THCudaTensor_data(ones), k_,
        (cuComplex*)THCudaTensor_data(bias), k_,
        beta,
        (cuComplex*)THCudaTensor_data(output_n), n_
    );

    // Extract columns:
    im2col_complex(
        (cuComplex*)THCudaTensor_data(input_n),
        nInputPlane, inputHeight, inputWidth, kH, kW, padding, padding, dH, dW,
        (cuComplex*)THCudaTensor_data(columns)
    );

    // M,N,K are dims of matrix A and B
    // (see http://docs.nvidia.com/cuda/cublas/#cublas-lt-t-gt-gemm)
    long m = weight->size[0];
    long n = columns->size[1] / 2;
    long k = weight->size[1] / 2;

    // Do GEMM (note: this is a bit confusing because gemm assumes 
    // column-major matrices)
    THCudaBlas_cgemm(
        'n', 'n',
        n, m, k,
        alpha,
        (cuComplex*)THCudaTensor_data(columns), n,
        (cuComplex*)THCudaTensor_data(weight), k,
        alpha,
        (cuComplex*)THCudaTensor_data(output_n), n
    );
  }

  // Free
  THCudaTensor_free(input_n);
  THCudaTensor_free(output_n);

  // Resize output
  if (batch == 0) {
    THCudaTensor_resize4d(output, nOutputPlane, outputHeight, outputWidth, 2);
    THCudaTensor_resize4d(input, nInputPlane, inputHeight, inputWidth, 2);
  }

  // return output
  return 1;
}

static int cunn_SpatialConvolutionComplexMM_updateGradInput(lua_State *L) {
  // Inputs
  THCudaTensor *input = (THCudaTensor *)luaT_checkudata(L, 2, 
    "torch.CudaTensor");
  THCudaTensor *gradOutput = (THCudaTensor *)luaT_checkudata(L, 3, 
    "torch.CudaTensor");

  // Params
  int dW = luaT_getfieldcheckint(L, 1, "dW");
  int dH = luaT_getfieldcheckint(L, 1, "dH");
  int kW = luaT_getfieldcheckint(L, 1, "kW");
  int kH = luaT_getfieldcheckint(L, 1, "kH");
  int nInputPlane = luaT_getfieldcheckint(L, 1, "nInputPlane");
  int nOutputPlane = luaT_getfieldcheckint(L, 1, "nOutputPlane");
  int padding = luaT_getfieldcheckint(L, 1, "padding");

  THCudaTensor *weight = (THCudaTensor *)luaT_getfieldcheckudata(L, 1, 
    "weight", "torch.CudaTensor");
  THCudaTensor *gradColumns = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, 
    "finput", "torch.CudaTensor");
  THCudaTensor *gradInput = (THCudaTensor *)luaT_getfieldcheckudata(L, 1, 
    "gradInput", "torch.CudaTensor");

  const int device = THCudaTensor_getDevice(weight);
  luaL_argcheck(L, THCudaTensor_getDevice(input) == device, 2,
                "weight and input need to be on the same device");
  luaL_argcheck(L, THCudaTensor_getDevice(gradInput) == device
                || THCudaTensor_getDevice(gradInput) == -1, 2,
                "weight and gradInput need to be on the same device");
  luaL_argcheck(L, THCudaTensor_getDevice(gradOutput) == device
                || THCudaTensor_getDevice(gradOutput) == -1, 2,
                "weight and gradOutput need to be on the same device");


  luaL_argcheck(L, input->nDimension == 4 || input->nDimension == 5, 2, 
    "4D or 5D (batch mode) tensor is expected");
  luaL_argcheck(L, input->size[input->nDimension-1] == 2, 2, 
    "last input dimension size is not 2");

  int batch = 1;
  if (input->nDimension == 4) {
    // Force batch
    batch = 0;
    THCudaTensor_resize5d(input, 1, input->size[0], input->size[1], 
      input->size[2], 2);
    THCudaTensor_resize5d(gradOutput, 1, gradOutput->size[0], 
      gradOutput->size[1], gradOutput->size[2], 2);
  }

  long inputWidth   = input->size[3];
  long inputHeight  = input->size[2];
  long outputWidth  = (inputWidth + 2*padding - kW) / dW + 1;
  long outputHeight = (inputHeight + 2*padding - kH) / dH + 1;

  // Batch size + input planes
  long batchSize = input->size[0];

  // Resize output
  THCudaTensor_resize5d(gradInput, batchSize, nInputPlane, inputHeight, 
    inputWidth, 2);

  // Resize temporary columns
  THCudaTensor_resize2d(gradColumns, nInputPlane*kW*kH*2, 
    outputHeight*outputWidth*2);

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
    long m = weight->size[1] / 2;
    long n = gradColumns->size[1] / 2;
    long k = weight->size[0];

    // Do GEMM (note: this is a bit confusing because gemm assumes 
    // column-major matrices)
    cuComplex alpha = make_float2(1, 0);
    cuComplex beta = make_float2(0, 0);
    THCudaBlas_cgemm(
        'n', 'c',
        n, m, k,
        alpha,
        (cuComplex*)THCudaTensor_data(gradOutput_n), n,
        (cuComplex*)THCudaTensor_data(weight), m,
        beta,
        (cuComplex*)THCudaTensor_data(gradColumns), n
    );

    // Unpack columns back into input:
    col2im_complex(
        (cuComplex*)THCudaTensor_data(gradColumns),
        nInputPlane, inputHeight, inputWidth, kH, kW, padding, padding, dH, dW,
        (cuComplex*)THCudaTensor_data(gradInput_n)
    );
  }

  // Free
  THCudaTensor_free(input_n);
  THCudaTensor_free(gradInput_n);
  THCudaTensor_free(gradOutput_n);

  // Resize output
  if (batch == 0) {
    THCudaTensor_resize4d(gradOutput, nOutputPlane, outputHeight, outputWidth,
      2);
    THCudaTensor_resize4d(input, nInputPlane, inputHeight, inputWidth, 2);
    THCudaTensor_resize4d(gradInput, nInputPlane, inputHeight, inputWidth, 2);
  }
  
  // Return gradInput
  return 1;
}

static int cunn_SpatialConvolutionComplexMM_accGradParameters(lua_State *L) {
  // Inputs
  THCudaTensor *input = (THCudaTensor *)luaT_checkudata(L, 2, 
    "torch.CudaTensor");
  THCudaTensor *gradOutput = (THCudaTensor *)luaT_checkudata(L, 3, 
    "torch.CudaTensor");

  // Params
  int dW = luaT_getfieldcheckint(L, 1, "dW");
  int dH = luaT_getfieldcheckint(L, 1, "dH");
  int kW = luaT_getfieldcheckint(L, 1, "kW");
  int kH = luaT_getfieldcheckint(L, 1, "kH");
  int nInputPlane = luaT_getfieldcheckint(L, 1, "nInputPlane");
  int nOutputPlane = luaT_getfieldcheckint(L, 1, "nOutputPlane");
  int padding = luaT_getfieldcheckint(L, 1, "padding");
  float scale = luaL_optnumber(L, 4, 1);

  THCudaTensor *gradWeight = (THCudaTensor *)luaT_getfieldcheckudata(L, 1, 
    "gradWeight", "torch.CudaTensor");
  THCudaTensor *gradBias = (THCudaTensor *)luaT_getfieldcheckudata(L, 1, 
    "gradBias", "torch.CudaTensor");
  THCudaTensor *columns = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, 
    "finput", "torch.CudaTensor");
  THCudaTensor *ones = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, 
    "fgradInput", "torch.CudaTensor");

  const int device = THCudaTensor_getDevice(gradWeight);
  luaL_argcheck(L, THCudaTensor_getDevice(gradBias) == device, 1,
                "gradWeight and gradBias need to be on the same device");
  luaL_argcheck(L, THCudaTensor_getDevice(input) == device, 1,
                "gradWeight and input need to be on the same device");
  luaL_argcheck(L, THCudaTensor_getDevice(gradOutput) == device, 1,
                "gradWeight and gradOutput need to be on the same device");

  luaL_argcheck(L, input->nDimension == 4 || input->nDimension == 5, 2, 
    "4D or 5D (batch mode) tensor is expected");
  luaL_argcheck(L, input->size[input->nDimension-1] == 2, 2,
    "last input dimension size is not 2");

  int batch = 1;
  if (input->nDimension == 4) {
    // Force batch
    batch = 0;
    THCudaTensor_resize5d(input, 1, input->size[0], input->size[1], 
      input->size[2], 2);
    THCudaTensor_resize5d(gradOutput, 1, gradOutput->size[0], 
      gradOutput->size[1], gradOutput->size[2], 2);
  }

  long inputWidth   = input->size[3];
  long inputHeight  = input->size[2];
  long outputWidth  = (inputWidth + 2*padding - kW) / dW + 1;
  long outputHeight = (inputHeight + 2*padding - kH) / dH + 1;

  // Batch size + input planes
  long batchSize = input->size[0];

  // Define a buffer of ones, for bias accumulation
  if (ones->nDimension != 3 ||
      ones->size[0]*ones->size[1] < outputHeight*outputWidth) {
    // Resize plane and fill with ones (real) and zeros (imag)...
    THCudaTensor_resize3d(ones, outputHeight, outputWidth, 2);
    THCudaTensor *real = THCudaTensor_new();
    THCudaTensor *imag = THCudaTensor_new();
    THCudaTensor_narrow(real, ones, 2, 0, 1);
    THCudaTensor_narrow(imag, ones, 2, 1, 1);
    THCudaTensor_fill(real, 1);
    THCudaTensor_fill(imag, 0);
    THCudaTensor_free(real);
    THCudaTensor_free(imag);
  }

  // Resize temporary columns
  THCudaTensor_resize2d(columns, nInputPlane*kW*kH*2, 
    outputHeight*outputWidth*2);

  // Helpers
  THCudaTensor *input_n = THCudaTensor_new();
  THCudaTensor *gradOutput_n = THCudaTensor_new();

  // For each elt in batch, do:
  for (int elt = 0; elt < batchSize; elt ++) {
    // Matrix mulitply per output:
    THCudaTensor_select(input_n, input, 0, elt);
    THCudaTensor_select(gradOutput_n, gradOutput, 0, elt);

    // Extract columns:
    im2col_complex(
        (cuComplex*)THCudaTensor_data(input_n),
        nInputPlane, inputHeight, inputWidth, kH, kW, padding, padding, dH, dW,
        (cuComplex*)THCudaTensor_data(columns)
    );

    // M,N,K are dims of matrix A and B
    // (see http://docs.nvidia.com/cuda/cublas/#cublas-lt-t-gt-gemm)
    long m = gradWeight->size[0];
    long n = gradWeight->size[1] / 2;
    long k = columns->size[1] / 2;

    // Do GEMM (note: this is a bit confusing because gemm assumes 
    // column-major matrices)
    cuComplex alpha = make_float2(scale, 0);
    cuComplex beta = make_float2(1, 0);
    THCudaBlas_cgemm(
        'c', 'n',
        n, m, k,
        alpha,
        (cuComplex*)THCudaTensor_data(columns), k,
        (cuComplex*)THCudaTensor_data(gradOutput_n), k,
        beta,
        (cuComplex*)THCudaTensor_data(gradWeight), n
    );

    // Do Bias:
    // M,N,K are dims of matrix A and B
    // (see http://docs.nvidia.com/cuda/cublas/#cublas-lt-t-gt-gemm)
    long m_ = nOutputPlane;
    long k_ = outputHeight * outputWidth;

    // Do GEMV (note: this is a bit confusing because gemv assumes 
    // column-major matrices)
    THCudaBlas_cgemv(
        't',
        k_, m_,
        alpha,
        (cuComplex*)THCudaTensor_data(gradOutput_n), k_,
        (cuComplex*)THCudaTensor_data(ones), 1,
        beta,
        (cuComplex*)THCudaTensor_data(gradBias), 1
    );
  }

  // Free
  THCudaTensor_free(input_n);
  THCudaTensor_free(gradOutput_n);

  // Resize
  if (batch == 0) {
    THCudaTensor_resize4d(gradOutput, nOutputPlane, outputHeight, outputWidth,
      2);
    THCudaTensor_resize4d(input, nInputPlane, inputHeight, inputWidth, 2);
  }
  // Return nothing
  return 0;
}

static const struct luaL_Reg cunn_SpatialConvolutionComplexMM__ [] = {
  {"SpatialConvolutionComplexMM_updateOutput", cunn_SpatialConvolutionComplexMM_updateOutput},
  {"SpatialConvolutionComplexMM_updateGradInput", cunn_SpatialConvolutionComplexMM_updateGradInput},
  {"SpatialConvolutionComplexMM_accGradParameters", cunn_SpatialConvolutionComplexMM_accGradParameters},
  {NULL, NULL}
};

static void cunn_SpatialConvolutionComplexMM_init(lua_State *L)
{
  luaT_pushmetatable(L, "torch.CudaTensor");
  luaT_registeratname(L, cunn_SpatialConvolutionComplexMM__, "nn");
  lua_pop(L,1);
}
