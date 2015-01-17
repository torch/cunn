#include "utils.h"

// CUDA: grid stride looping
#define CUDA_KERNEL_LOOP(i, n)                        \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; \
       i < (n);                                       \
       i += blockDim.x * gridDim.x)

// Kernel for fast unfold+copy
// Borrowed from Theano
// Authors: Arjun Jain, Frédéric Bastien, Jan Schlüter, Nicolas Ballas
__global__ void im3d2col_kernel(const int n, const float* data_im,
                                const int height, const int width, const int depth,
                                const int kernel_h, const int kernel_w, const int kernel_d,
                                const int pad_h, const int pad_w, const int pad_d,
                                const int stride_h, const int stride_w, const int stride_d,
                                const int height_col, const int width_col, const int depth_col,
                                float* data_col)
{
  CUDA_KERNEL_LOOP(index, n)
  {
    int d_out = index % depth_col;
    int w_index = index / depth_col;
    int w_out = w_index % width_col;
    int h_index = w_index / width_col;
    int h_out = h_index % height_col;

    int channel_in = h_index / height_col;
    //channel_in = 1;

    int channel_out = channel_in * kernel_h * kernel_w * kernel_d;

    int h_in = h_out * stride_h - pad_h;
    int w_in = w_out * stride_w - pad_w;
    int d_in = d_out * stride_d - pad_d;

    float* data_col_ptr = data_col;
    data_col_ptr += channel_out * (height_col * width_col * depth_col) +
      h_out * (width_col * depth_col) + w_out * depth_col + d_out;

    const float* data_im_ptr = data_im;
    data_im_ptr += channel_in * (height * width * depth) +
      h_in * (width * depth) + w_in * depth + d_in;

    for (int i = 0; i < kernel_h; ++i)
    {
      int h = h_in + i;
      for (int j = 0; j < kernel_w; ++j)
      {
        int w = w_in + j;
        for (int k = 0; k < kernel_d; ++k)
        {
          int d = d_in + k;
          *data_col_ptr = (h >= 0 && w >= 0 && d >= 0 &&
                           h < height && w < width && d < depth) ?
                           data_im_ptr[i * (width * depth) + j *depth + k] : 0;
          data_col_ptr += height_col * width_col * depth_col;
        }
      }
    }
  }
}

void im3d2col(const float* data_im, const int channels,
              const int height, const int width, const int depth,
              const int kernel_h, const int kernel_w, const int kernel_d,
              const int pad_h, const int pad_w, const int pad_d,
              const int stride_h, const int stride_w, const int stride_d,
              float* data_col)
{
  // We are going to launch channels * height_col * width_col * depth_col kernels, each
  // kernel responsible for copying a single-channel grid.
  int height_col = (height + 2 * pad_h - kernel_h) / stride_h + 1;
  int width_col = (width + 2 * pad_w - kernel_w) / stride_w + 1;
  int depth_col = (depth + 2 * pad_d - kernel_d) / stride_d + 1;
  int num_kernels = channels * height_col * width_col * depth_col;
  im3d2col_kernel<<<GET_BLOCKS(num_kernels),
                    CUDA_NUM_THREADS>>>(num_kernels, data_im,
                                        height, width, depth,
                                        kernel_h, kernel_w, kernel_d,
                                        pad_h, pad_w, pad_d,
                                        stride_h, stride_w, stride_d,
                                        height_col, width_col, depth_col,
                                        data_col);
}


__global__ void col2im3d_kernel(const int n, const float* data_col,
                                const int height, const int width, const int depth,
                                const int channels,
                                const int patch_h, const int patch_w, const int patch_d,
                                const int pad_h, const int pad_w, const int pad_d,
                                const int stride_h, const int stride_w, const int stride_d,
                                const int height_col, const int width_col, const int depth_col,
                                float* data_im)
{
  CUDA_KERNEL_LOOP(index, n)
  {
    float val = 0;
    int d = index % depth + pad_d;
    int w_index = index / depth;
    int w = w_index % width + pad_w;
    int h_index = w_index / width;
    int h = h_index % height + pad_h;
    int c = h_index / height;

    // compute the start and end of the output
    int d_col_start = (d < patch_d) ? 0 : (d - patch_d) / stride_d + 1;
    int d_col_end = min(d / stride_d + 1, depth_col);
    int w_col_start = (w < patch_w) ? 0 : (w - patch_w) / stride_w + 1;
    int w_col_end = min(w / stride_w + 1, width_col);
    int h_col_start = (h < patch_h) ? 0 : (h - patch_h) / stride_h + 1;
    int h_col_end = min(h / stride_h + 1, height_col);

    int offset =
      (c * patch_h * patch_w * patch_d + h * patch_w * patch_d + w * patch_d + d) * height_col * width_col * depth_col;

    int coeff_h_col = (1 - stride_h * patch_w * patch_d * height_col) * width_col * depth_col;
    int coeff_w_col = (1 - stride_w * patch_d * height_col * width_col) * depth_col;
    int coeff_d_col = (1 - stride_d * height_col * width_col * depth_col);
    for (int d_col = d_col_start; d_col < d_col_end; ++d_col)
      for (int h_col = h_col_start; h_col < h_col_end; ++h_col) {
        for (int w_col = w_col_start; w_col < w_col_end; ++w_col) {
          val += data_col[offset + h_col * coeff_h_col + w_col * coeff_w_col + d_col * coeff_d_col];
      }
   }
    data_im[index] = val;
  }
}

void col2im3d(const float* data_col, const int channels,
              const int height, const int width, const int depth,
              const int patch_h, const int patch_w, const int patch_d,
              const int pad_h, const int pad_w, const int pad_d,
              const int stride_h, const int stride_w, const int stride_d,
              float* data_im)
{
  int height_col = (height + 2 * pad_h - patch_h) / stride_h + 1;
  int width_col = (width + 2 * pad_w - patch_w) / stride_w + 1;
  int depth_col = (depth + 2 * pad_d - patch_d) / stride_d + 1;
  int num_kernels = channels * height * width * depth;

  // To avoid involving atomic operations, we will launch one kernel per
  // bottom dimension, and then in the kernel add up the top dimensions.
  col2im3d_kernel<<<GET_BLOCKS(num_kernels),
                    CUDA_NUM_THREADS>>>(num_kernels, data_col,
                                        height, width, depth, channels,
                                        patch_h, patch_w, patch_d,
                                        pad_h, pad_w, pad_d,
                                        stride_h, stride_w, stride_d,
                                        height_col, width_col, depth_col,
                                        data_im);
}


static int cunn_VolumetricConvolution_updateOutput(lua_State *L) {
  THCState *state = getCutorchState(L);

  // Input
  THCudaTensor *input = (THCudaTensor*)luaT_checkudata(L, 2, "torch.CudaTensor");
  // Params:
  int dD = luaT_getfieldcheckint(L, 1, "dW");
  int dW = luaT_getfieldcheckint(L, 1, "dH");
  int dH = luaT_getfieldcheckint(L, 1, "dT");
  int kD = luaT_getfieldcheckint(L, 1, "kW");
  int kW = luaT_getfieldcheckint(L, 1, "kH");
  int kH = luaT_getfieldcheckint(L, 1, "kT");
  int nInputPlane = luaT_getfieldcheckint(L, 1, "nInputPlane");
  int nOutputPlane = luaT_getfieldcheckint(L, 1, "nOutputPlane");

  THCudaTensor *weight = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "weight", "torch.CudaTensor");
  THCudaTensor *bias = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "bias", "torch.CudaTensor");
  THCudaTensor *columns = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "finput", "torch.CudaTensor");
  THCudaTensor *ones = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "fgradInput", "torch.CudaTensor");
  THCudaTensor *output = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "output", "torch.CudaTensor");

  const int device = THCudaTensor_getDevice(state, weight);
  luaL_argcheck(L, THCudaTensor_getDevice(state, bias) == device, 1,
                "weight and bias need to be on the same device");
  luaL_argcheck(L, THCudaTensor_getDevice(state, output) == device ||
                THCudaTensor_getDevice(state, output) == -1, 1,
                "weight and output need to be on the same device");
  luaL_argcheck(L, THCudaTensor_getDevice(state, input) == device, 2,
                "weight and input need to be on the same device");

  luaL_argcheck(L, input->nDimension == 4 || input->nDimension == 5, 2, "4D or 5D (batch mode) tensor is expected");

  int batch = 1;
  if (input->nDimension == 4) {
    // Force batch
    batch = 0;
    THCudaTensor_resize5d(state, input, 1, input->size[0], input->size[1],
                          input->size[2], input->size[3]);
  }

  long inputWidth   = input->size[3];
  long inputHeight  = input->size[2];
  long inputDepth   = input->size[4];
  long outputWidth  = (inputWidth  - kW) / dW + 1;
  long outputHeight = (inputHeight - kH) / dH + 1;
  long outputDepth  = (inputDepth - kD) / dD + 1;

  // Batch size + input planes
  long batchSize = input->size[0];

  // Resize output
  THCudaTensor_resize5d(state, output, batchSize, nOutputPlane, outputDepth,
                        outputHeight, outputWidth);

  // Resize temporary columns
  THCudaTensor_resize2d(state, columns, nInputPlane*kD*kW*kH, outputDepth*outputHeight*outputWidth);

  // Define a buffer of ones, for bias accumulation
  // Note: this buffer can be shared with other modules, it only ever gets increased,
  // and always contains ones.
  if (ones->nDimension != 3 || ones->size[0]*ones->size[1]*ones->size[2] < outputDepth*outputHeight*outputWidth) {
    // Resize plane and fill with ones...
    THCudaTensor_resize3d(state, ones, outputDepth, outputHeight, outputWidth);
    THCudaTensor_fill(state, ones, 1);
  }

  // Helpers
  THCudaTensor *input_n = THCudaTensor_new(state);
  THCudaTensor *output_n = THCudaTensor_new(state);

  // For each elt in batch, do:
  for (int elt = 0; elt < batchSize; elt ++) {
    // Matrix mulitply per output:
    THCudaTensor_select(state, input_n, input, 0, elt);
    THCudaTensor_select(state, output_n, output, 0, elt);

    // Do Bias first:
    // M,N,K are dims of matrix A and B
    // (see http://docs.nvidia.com/cuda/cublas/#cublas-lt-t-gt-gemm)
    long m_ = nOutputPlane;
    long n_ = outputDepth * outputHeight * outputWidth;
    long k_ = 1;

    // Do GEMM (note: this is a bit confusing because gemm assumes column-major matrices)
    THCudaBlas_gemm(
        state,
        't', 'n',
        n_, m_, k_,
        1,
        THCudaTensor_data(state, ones), k_,
        THCudaTensor_data(state, bias), k_,
        0,
        THCudaTensor_data(state, output_n), n_
    );

    // Extract columns:
    im3d2col(
        THCudaTensor_data(state, input_n),
        nInputPlane, inputHeight, inputWidth, inputDepth, kH, kW, kD, 0, 0, 0, dH, dW, dD,
        THCudaTensor_data(state, columns)
    );

    // M,N,K are dims of matrix A and B
    // (see http://docs.nvidia.com/cuda/cublas/#cublas-lt-t-gt-gemm)
    long m = weight->size[0];
    long n = columns->size[1];
    long k = weight->size[1]*weight->size[2]*weight->size[3]*weight->size[4];

    // Do GEMM (note: this is a bit confusing because gemm assumes column-major matrices)
    THCudaBlas_gemm(
        state,
        'n', 'n',
        n, m, k,
        1,
        THCudaTensor_data(state, columns), n,
        THCudaTensor_data(state, weight), k,
        1,
        THCudaTensor_data(state, output_n), n
    );
  }

  // Free
  THCudaTensor_free(state, input_n);
  THCudaTensor_free(state, output_n);

  // Resize output
  if (batch == 0) {
    THCudaTensor_resize4d(state, output, nOutputPlane, outputHeight, outputWidth, outputDepth);
    THCudaTensor_resize4d(state, input, nInputPlane, inputHeight, inputWidth, inputDepth);
  }

  // return output
  return 1;
}

static int cunn_VolumetricConvolution_updateGradInput(lua_State *L) {
  THCState *state = getCutorchState(L);

  // Inputs
  THCudaTensor *input = (THCudaTensor *)luaT_checkudata(L, 2, "torch.CudaTensor");
  THCudaTensor *gradOutput = (THCudaTensor *)luaT_checkudata(L, 3, "torch.CudaTensor");

  // Params
  int dD = luaT_getfieldcheckint(L, 1, "dW");
  int dW = luaT_getfieldcheckint(L, 1, "dH");
  int dH = luaT_getfieldcheckint(L, 1, "dT");
  int kD = luaT_getfieldcheckint(L, 1, "kW");
  int kW = luaT_getfieldcheckint(L, 1, "kH");
  int kH = luaT_getfieldcheckint(L, 1, "kT");
  int nInputPlane = luaT_getfieldcheckint(L, 1, "nInputPlane");
  int nOutputPlane = luaT_getfieldcheckint(L, 1, "nOutputPlane");

  THCudaTensor *weight = (THCudaTensor *)luaT_getfieldcheckudata(L, 1, "weight", "torch.CudaTensor");
  THCudaTensor *gradColumns = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "finput", "torch.CudaTensor");
  THCudaTensor *gradInput = (THCudaTensor *)luaT_getfieldcheckudata(L, 1, "gradInput", "torch.CudaTensor");

  const int device = THCudaTensor_getDevice(state, weight);
  luaL_argcheck(L, THCudaTensor_getDevice(state, input) == device, 2,
                "weight and input need to be on the same device");
  luaL_argcheck(L, THCudaTensor_getDevice(state, gradInput) == device
                || THCudaTensor_getDevice(state, gradInput) == -1, 2,
                "weight and gradInput need to be on the same device");
  luaL_argcheck(L, THCudaTensor_getDevice(state, gradOutput) == device
                || THCudaTensor_getDevice(state, gradOutput) == -1, 2,
                "weight and gradOutput need to be on the same device");


  luaL_argcheck(L, input->nDimension == 4 || input->nDimension == 5, 2, "4D or 5D (batch mode) tensor is expected");

  int batch = 1;
  if (input->nDimension == 4) {
    // Force batch
    batch = 0;
    THCudaTensor_resize5d(state, input, 1, input->size[0], input->size[1], input->size[2], input->size[3]);
    THCudaTensor_resize5d(state, gradOutput, 1, gradOutput->size[0], gradOutput->size[1], gradOutput->size[2], gradOutput->size[3]);
  }

  long inputWidth   = input->size[3];
  long inputHeight  = input->size[2];
  long inputDepth   = input->size[4];
  long outputWidth  = (inputWidth - kW) / dW + 1;
  long outputHeight = (inputHeight - kH) / dH + 1;
  long outputDepth  = (inputDepth - kD) / dD + 1;

  // Batch size + input planes
  long batchSize = input->size[0];

  // Resize output
  THCudaTensor_resize5d(state, gradInput, batchSize, nInputPlane, inputDepth, inputHeight, inputWidth);

  // Resize temporary columns
  THCudaTensor_resize2d(state, gradColumns, nInputPlane*kW*kH*kD, outputDepth*outputHeight*outputWidth);

  // Helpers
  THCudaTensor *input_n = THCudaTensor_new(state);
  THCudaTensor *gradInput_n = THCudaTensor_new(state);
  THCudaTensor *gradOutput_n = THCudaTensor_new(state);

  // For each elt in batch, do:
  for (int elt = 0; elt < batchSize; elt ++) {
    // Matrix mulitply per sample:
    THCudaTensor_select(state, input_n, input, 0, elt);
    THCudaTensor_select(state, gradInput_n, gradInput, 0, elt);
    THCudaTensor_select(state, gradOutput_n, gradOutput, 0, elt);

    // M,N,K are dims of matrix A and B
    // (see http://docs.nvidia.com/cuda/cublas/#cublas-lt-t-gt-gemm)
    long m = weight->size[1]*weight->size[2]*weight->size[3]*weight->size[4];
    long n = gradColumns->size[1];
    long k = weight->size[0];

    // Do GEMM (note: this is a bit confusing because gemm assumes column-major matrices)
    THCudaBlas_gemm(
        state,
        'n', 't',
        n, m, k,
        1,
        THCudaTensor_data(state, gradOutput_n), n,
        THCudaTensor_data(state, weight), m,
        0,
        THCudaTensor_data(state, gradColumns), n
    );

    // Unpack columns back into input:
    col2im3d(
      THCudaTensor_data(state, gradColumns),
      nInputPlane, inputHeight, inputWidth, inputDepth, kH, kW, kD, 0, 0, 0, dH, dW, dD,
      THCudaTensor_data(state, gradInput_n)
    );
  }

  // Free
  THCudaTensor_free(state, input_n);
  THCudaTensor_free(state, gradInput_n);
  THCudaTensor_free(state, gradOutput_n);

  // Resize output
  if (batch == 0) {
    THCudaTensor_resize4d(state, gradOutput, nOutputPlane, outputHeight, outputWidth, outputDepth);
    THCudaTensor_resize4d(state, input, nInputPlane, inputHeight, inputWidth, inputDepth);
    THCudaTensor_resize4d(state, gradInput, nInputPlane, inputHeight, inputWidth, inputDepth);
  }

  // Return gradInput
  return 1;
}

static int cunn_VolumetricConvolution_accGradParameters(lua_State *L) {
  THCState *state = getCutorchState(L);

  // Inputs
  THCudaTensor *input = (THCudaTensor *)luaT_checkudata(L, 2, "torch.CudaTensor");
  THCudaTensor *gradOutput = (THCudaTensor *)luaT_checkudata(L, 3, "torch.CudaTensor");

  // Params
  int dD = luaT_getfieldcheckint(L, 1, "dW");
  int dW = luaT_getfieldcheckint(L, 1, "dH");
  int dH = luaT_getfieldcheckint(L, 1, "dT");
  int kD = luaT_getfieldcheckint(L, 1, "kW");
  int kW = luaT_getfieldcheckint(L, 1, "kH");
  int kH = luaT_getfieldcheckint(L, 1, "kT");
  int nInputPlane = luaT_getfieldcheckint(L, 1, "nInputPlane");
  int nOutputPlane = luaT_getfieldcheckint(L, 1, "nOutputPlane");
  float scale = luaL_optnumber(L, 4, 1);

  THCudaTensor *gradWeight = (THCudaTensor *)luaT_getfieldcheckudata(L, 1, "gradWeight", "torch.CudaTensor");
  THCudaTensor *gradBias = (THCudaTensor *)luaT_getfieldcheckudata(L, 1, "gradBias", "torch.CudaTensor");
  THCudaTensor *columns = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "finput", "torch.CudaTensor");
  THCudaTensor *ones = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "fgradInput", "torch.CudaTensor");

  const int device = THCudaTensor_getDevice(state, gradWeight);
  luaL_argcheck(L, THCudaTensor_getDevice(state, gradBias) == device, 1,
                "gradWeight and gradBias need to be on the same device");
  luaL_argcheck(L, THCudaTensor_getDevice(state, input) == device, 1,
                "gradWeight and input need to be on the same device");
  luaL_argcheck(L, THCudaTensor_getDevice(state, gradOutput) == device, 1,
                "gradWeight and gradOutput need to be on the same device");

  luaL_argcheck(L, input->nDimension == 4 || input->nDimension == 5, 2, "3D or 4D (batch mode) tensor is expected");

  int batch = 1;
  if (input->nDimension == 4) {
    // Force batch
    batch = 0;
    THCudaTensor_resize5d(state, input, 1, input->size[0], input->size[1], input->size[2], input->size[3]);
    THCudaTensor_resize5d(state, gradOutput, 1, gradOutput->size[0], gradOutput->size[1], gradOutput->size[2], gradOutput->size[3]);
  }

  long inputWidth   = input->size[3];
  long inputHeight  = input->size[2];
  long inputDepth   = input->size[4];
  long outputWidth  = (inputWidth - kW) / dW + 1;
  long outputHeight = (inputHeight - kH) / dH + 1;
  long outputDepth  = (inputDepth - kD) / dD + 1;

  // Batch size + input planes
  long batchSize = input->size[0];

  // Define a buffer of ones, for bias accumulation
  if (ones->nDimension != 3 || ones->size[0]*ones->size[1]*ones->size[2] < outputDepth*outputHeight*outputWidth) {
    // Resize plane and fill with ones...
    THCudaTensor_resize3d(state, ones, outputDepth, outputHeight, outputWidth);
    THCudaTensor_fill(state, ones, 1);
  }

  // Resize temporary columns
  THCudaTensor_resize2d(state, columns, nInputPlane*kW*kH*kD, outputDepth*outputHeight*outputWidth);

  // Helpers
  THCudaTensor *input_n = THCudaTensor_new(state);
  THCudaTensor *gradOutput_n = THCudaTensor_new(state);

  // For each elt in batch, do:
  for (int elt = 0; elt < batchSize; elt ++) {
    // Matrix mulitply per output:
    THCudaTensor_select(state, input_n, input, 0, elt);
    THCudaTensor_select(state, gradOutput_n, gradOutput, 0, elt);

    // Extract columns:
    im3d2col(
        THCudaTensor_data(state, input_n),
        nInputPlane, inputHeight, inputWidth, inputDepth, kH, kW, kD, 0, 0, 0, dH, dW, dD,
        THCudaTensor_data(state, columns)
    );

    // M,N,K are dims of matrix A and B
    // (see http://docs.nvidia.com/cuda/cublas/#cublas-lt-t-gt-gemm)
    long m = gradWeight->size[0];
    long n = gradWeight->size[1]*gradWeight->size[2]*gradWeight->size[3]*gradWeight->size[4];
    long k = columns->size[1];

    // Do GEMM (note: this is a bit confusing because gemm assumes column-major matrices)
    THCudaBlas_gemm(
        state,
        't', 'n',
        n, m, k,
        scale,
        THCudaTensor_data(state, columns), k,
        THCudaTensor_data(state, gradOutput_n), k,
        1,
        THCudaTensor_data(state, gradWeight), n
    );

    // Do Bias:
    // M,N,K are dims of matrix A and B
    // (see http://docs.nvidia.com/cuda/cublas/#cublas-lt-t-gt-gemm)
    long m_ = nOutputPlane;
    long k_ = outputDepth * outputHeight * outputWidth;

    // Do GEMV (note: this is a bit confusing because gemv assumes column-major matrices)
    THCudaBlas_gemv(
        state,
        't',
        k_, m_,
        scale,
        THCudaTensor_data(state, gradOutput_n), k_,
        THCudaTensor_data(state, ones), 1,
        1,
        THCudaTensor_data(state, gradBias), 1
    );
  }

  // Free
  THCudaTensor_free(state, input_n);
  THCudaTensor_free(state, gradOutput_n);

  // Resize
  if (batch == 0) {
    THCudaTensor_resize4d(state, gradOutput, nOutputPlane, outputHeight, outputWidth, outputDepth);
    THCudaTensor_resize4d(state, input, nInputPlane, inputHeight, inputWidth, inputDepth);
  }

  // Return nothing
  return 0;
}

static const struct luaL_Reg cunn_VolumetricConvolution__ [] = {
  {"VolumetricConvolution_updateOutput", cunn_VolumetricConvolution_updateOutput},
  {"VolumetricConvolution_updateGradInput", cunn_VolumetricConvolution_updateGradInput},
  {"VolumetricConvolution_accGradParameters", cunn_VolumetricConvolution_accGradParameters},
  {NULL, NULL}
};

static void cunn_VolumetricConvolution_init(lua_State *L)
{
  luaT_pushmetatable(L, "torch.CudaTensor");
  luaT_registeratname(L, cunn_VolumetricConvolution__, "nn");
  lua_pop(L,1);
}
