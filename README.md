# CUDA backend for the Neural Network Package #

**This branch of 'cunn' package adds support for 'SpatialDepthWiseConvolution', which is similar to TensorFlow's '[depthwise_conv2d](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/g3doc/api_docs/python/functions_and_classes/shard4/tf.nn.depthwise_conv2d.md)'.**

This package provides a CUDA implementation this nn package: [nn](https://github.com/stooloveu/nn/blob/master/)

## Installing from source
```bash
git clone https://github.com/stooloveu/cunn
cd cunn
luarocks make rocks/cunn-scm-1.rockspec
```

## To use 'SpatialDepthWiseConvolution'

See <https://github.com/stooloveu/nn/blob/master/README.md>

## To use 'cunn' package

See <https://github.com/torch/cunn/blob/master/README.md>
