// WARNING: this module is incomplete - and just meant for reference for now.

// Kernel for fast unfold+copy
// (borrowed from Caffe: https://github.com/BVLC/caffe/blob/master/src/caffe/layers/conv_layer.cu)
__global__ void imt2col_kernel(const int n, const float* data_im,
        const int height, const int width, const int ksize, const int pad,
        const int stride, const int channels,
        const int height_col, const int width_col,
        const int bidx, const int batch,
        float* data_col) {
    CUDA_KERNEL_LOOP(index, n) {
        int w_out = index % width_col;
        index /= width_col;
        int h_out = index % height_col;
        int channel_in = index / height_col;
        int channel_out = channel_in * ksize * ksize;
        int h_in = h_out * stride - pad;
        int w_in = w_out * stride - pad;
        data_col += ((channel_out * batch + bidx) * height_col + h_out) * width_col + w_out;
        data_im += ((bidx * height + h_in) * width + w_in) * channels + channel_in;
        for (int i = 0; i < ksize; ++i) {
            for (int j = 0; j < ksize; ++j) {
                int h = h_in + i;
                int w = w_in + j;
                *data_col = (h >= 0 && w >= 0 && h < height && w < width) ?
                    data_im[(i * width + j) * channels] : 0;
                data_col += batch * height_col * width_col;
            }
        }
    }
}

void imt2col(const float* data_im, const int channels,
        const int height, const int width, const int ksize, const int pad,
        const int stride, const int batch, float* data_col) {
    // We are going to launch channels * height_col * width_col kernels, each
    // kernel responsible for copying a single-channel grid.
    int height_col = (height + 2 * pad - ksize) / stride + 1;
    int width_col = (width + 2 * pad - ksize) / stride + 1;
    int num_kernels = channels * height_col * width_col;
    // Launch
    for (int bidx = 0; bidx < batch; bidx++) {
        imt2col_kernel <<<GET_BLOCKS(num_kernels), CUDA_NUM_THREADS>>> (
            num_kernels, data_im, height, width, ksize,
            pad, stride, channels,
            height_col, width_col, bidx, batch,
            data_col
        );
    }
}

static int cunn_SpatialConvolutionMM_BHWD_updateOutput(lua_State *L) {
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
    THCudaTensor *output = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "output", "torch.CudaTensor");

    luaL_argcheck(L, input->nDimension == 3 || input->nDimension == 4, 2, "3D or 4D (batch mode) tensor is expected");

    int dimw = 1;
    int dimh = 0;
    if (input->nDimension == 4) {
        dimw++;
        dimh++;
    }
    long inputWidth   = input->size[dimw];
    long inputHeight  = input->size[dimh];
    long outputWidth  = (inputWidth - kW) / dW + 1;
    long outputHeight = (inputHeight - kH) / dH + 1;

    luaL_argcheck(L, kW == kH, 1, "filters must be square (kW == kH)");
    luaL_argcheck(L, dW == dH, 1, "stride must be square (dW == dH)");

    if (input->nDimension == 3) {

        // implementation in progress...

    } else {
        // Batch size + input planes
        long batchSize = input->size[0];
        luaL_argcheck(L, batchSize == 1 || batchSize % 4 == 0, 1, "batch size should be a multiple of 4 or equal to 1");
        luaL_argcheck(L, nOutputPlane % 8 == 0, 1, "nOutputPlane should be a multiple of 8");

        // Step batch (inner loop)
        // This variable defines how many samples are processed in //, in the inner loop
        int stepBatchSize = 1;
        if (batchSize % 4 == 0) {
            stepBatchSize = 4;
        }

        // Resize output
        THCudaTensor_resize4d(output, batchSize, outputHeight, outputWidth, nOutputPlane);

        // Resize temporary columns
        THCudaTensor_resize2d(columns, kH*kW*nInputPlane, stepBatchSize*outputHeight*outputWidth);

        // Add bias first
        // TODO: replace this by more efficient, custom kernel
        long k;
        THCudaTensor *outputPlane = THCudaTensor_new();
        for(k=0; k<nOutputPlane; k++) {
            THCudaTensor_select(outputPlane, output, 3, k);
            THCudaTensor_fill(outputPlane, THCudaTensor_get1d(bias, k));
        }
        THCudaTensor_free(outputPlane);

        // Helper
        THCudaTensor *output_n = THCudaTensor_new();

        // For each elt in batch, do:
        for (int elt = 0; elt < batchSize; elt += stepBatchSize) {
            // Extract columns:
            imt2col(
                THCudaTensor_data(input) + elt * inputHeight * inputWidth * nInputPlane,
                nInputPlane, inputHeight, inputWidth, kW, padding, dW, stepBatchSize,
                THCudaTensor_data(columns)
            );

            // Matrix mulitply per output:
            THCudaTensor_narrow(output_n, output, 0, elt, stepBatchSize);

            // M,N,K are dims of matrix A and B
            // (see http://docs.nvidia.com/cuda/cublas/#cublas-lt-t-gt-gemm)
            long m = weight->size[0];
            long n = columns->size[1];
            long k = weight->size[1];

            // Do GEMM_BHWD (note: this is a bit confusing because gemm assumes column-major matrices)
            THCudaBlas_gemm(
                't', 't',
                m, n, k,
                1,
                THCudaTensor_data(weight), k,
                THCudaTensor_data(columns), n,
                1,
                THCudaTensor_data(output_n), m
            );
        }

        // Free
        THCudaTensor_free(output_n);
    }

    // return output
    return 1;
}

static int cunn_SpatialConvolutionMM_BHWD_updateGradInput(lua_State *L) {
    // implementation in progress
    return 1;
}

static int cunn_SpatialConvolutionMM_BHWD_accGradParameters(lua_State *L) {
    // implementation in progress
    return 0;
}

static const struct luaL_Reg cunn_SpatialConvolutionMM_BHWD__ [] = {
    {"SpatialConvolutionMM_BHWD_updateOutput", cunn_SpatialConvolutionMM_BHWD_updateOutput},
    {"SpatialConvolutionMM_BHWD_updateGradInput", cunn_SpatialConvolutionMM_BHWD_updateGradInput},
    {"SpatialConvolutionMM_BHWD_accGradParameters", cunn_SpatialConvolutionMM_BHWD_accGradParameters},
    {NULL, NULL}
};

static void cunn_SpatialConvolutionMM_BHWD_init(lua_State *L)
{
    luaT_pushmetatable(L, "torch.CudaTensor");
    luaT_registeratname(L, cunn_SpatialConvolutionMM_BHWD__, "nn");
    lua_pop(L,1);
}
