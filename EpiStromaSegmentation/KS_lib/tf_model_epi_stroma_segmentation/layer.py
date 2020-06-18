"""
layer.py

This file contains activation definitions for various types of
convolutional neural network layers.
"""

import tensorflow as tf
import numpy as np

##############################################################################
def bilinear_filter(size):
    """
    bilnear_filter creates a bilnear_filter using the provided upscale
    or downscale ratio

    param: size
    return: bilinear filter array
    """
    factor = (size + 1) // 2
    if size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:size, :size]
    return (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor)

##############################################################################
def upsample_filt(kernel_size, out_channels, in_channels):
    """
    upsample_filt uses bilinear_filter to upsample the
    activation from a previous layer

    param: kernel_size
    param: out_channels
    param: in_channels
    return: bilinear upsample filter with specified shape
    """

    if (kernel_size[0] != kernel_size[1]) or (out_channels != in_channels):
        raise ValueError('kernel_size_row != kernel_size_col or out_channels != in channels')

    filt = np.zeros((kernel_size[0], kernel_size[1], out_channels, in_channels), dtype=np.float32)

    for i in range(in_channels):
        filt[:, :, i, i] = bilinear_filter(kernel_size[0])

    return filt

##############################################################################
def identity_initializer():
    """
    identity_initializer returns an identity tensor of the specified shape

    param: shape
    param: dtype
    param: partition_info
    return: identity tensor
    """
    def _initializer(shape, dtype=tf.float32, partition_info=None):
        if len(shape) == 1: #1D identity is just 0
            return tf.constant_op.constant(0., dtype=dtype, shape=shape)
        elif len(shape) == 2 and shape[0] == shape[1]: #2D identity must be square
            return tf.constant_op.constant(np.identity(shape[0], dtype))
        elif len(shape) == 4 and shape[2] == shape[3]: #4D identity works if last 2 dimensions are square

            array = np.zeros(shape, dtype=float)
            cx, cy = shape[0]/2, shape[1]/2
            for i in range(shape[2]):
                array[cx, cy, i, i] = 1
            return tf.constant(array, dtype=dtype)

        else:
            raise Exception('Shape not valid')
    return _initializer

##############################################################################
def weight_variable(name, shape, stddev):
    """
    weight_variable returns a tensor of random weights with the specified
    name, shape, and standard deviation

    param: name
    param: shape
    param: stddev
    return: weight tensor
    """
    return tf.get_variable(name, shape, initializer=tf.truncated_normal_initializer(stddev=stddev))

#############################################################################
def identity_variable(name, shape):
    """
    identity_variable uses identity_initializer to return
    an identity tensor of the specified name and shape

    param: name
    param: shape
    return: identity tensor
    """
    return tf.get_variable(name, shape, initializer=identity_initializer())

##############################################################################
def bias_variable(name, shape, constant):
    """
    bias_variable initializes and returns a tensor
    of constant biases with the specified name and shape

    param: name
    param: shape
    param: constant
    return: bias tensor
    """
    return tf.get_variable(name, shape, initializer=tf.constant_initializer(constant))

##############################################################################
def relu(inTensor):
    """
    relu applies the ReLU activation function
    to a tensor

    param: inTensor
    return: tensor that results from applying element wise ReLU to input
    """
    return tf.nn.relu(inTensor)

##############################################################################
def sigmoid(inTensor):
    """
    sigmoid applies the sigmoid activation function
    to a tensor

    param: inTensor
    return: tensor that results from applying element wise sigmoid to input
    """
    return tf.sigmoid(inTensor)

##############################################################################
def max_pool(inTensor, name):
    """
    max_pool uses 2x2 max pooling to downsize a layer

    param: inTensor
    param: name
    return: downsized tensor that results from applying 2x2 max pool to input
    """
    return tf.nn.max_pool(inTensor, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID', name=name)

##############################################################################
def down_conv_relu_same(images, kernel_size, in_channels, out_channels):
    """
    down_conv_relu_same creates a downsampling convolutional layer with the ReLU activation function
    that utilizes padding to maintain an identical length and width to the input tensor

    param: images
    param: kernel_size
    param: in_channels
    param: out_channels
    return: convolutional layer tensor with specified parameters
    """

    stddev = tf.sqrt(2 / tf.to_float(kernel_size[0] * kernel_size[1] * in_channels))
    kernel = weight_variable(name='weights', shape=[kernel_size[0], kernel_size[1], in_channels, out_channels], stddev=stddev)
    conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME')
    biases = bias_variable(name='biases', shape=[out_channels],constant=0.0)
    bias = tf.nn.bias_add(conv, biases)
    activate = relu(bias)
    return activate

##############################################################################
def down_conv_relu_valid(images, kernel_size, in_channels, out_channels):
    """
    down_conv_relu_valid creates a downsampling convolutional layer with the ReLU activation function
    that does not utilize padding and slightly shrinks the length and width of the input tensor

    param: images
    param: kernel_size
    param: in_channels
    param: out_channels
    return: convolutional layer tensor with specified parameters
    """

    stddev = tf.sqrt(2 / tf.to_float(kernel_size[0] * kernel_size[1] * in_channels))
    kernel = weight_variable(name='weights', shape=[kernel_size[0], kernel_size[1], in_channels, out_channels], stddev=stddev)
    conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='VALID')
    biases = bias_variable(name='biases', shape=[out_channels], constant=0.0)
    bias = tf.nn.bias_add(conv, biases)
    activate = relu(bias)
    return activate

##############################################################################
def up_conv_relu(images, kernel_size, stride, out_img_size, in_channels, out_channels, flags):
    """
    up_conv_relu creates an upsampling convolutional with the ReLU activation function

    param: images
    param: kernel_size
    param: stride
    param: out_img_size
    param: in_channels
    param: out_channels
    param: batch_size
    return: convolutional layer tensor with specified parameters
    """

    stddev = tf.sqrt(2 / tf.to_float(kernel_size[0] * kernel_size[1] * in_channels))
    kernel = weight_variable(name='weights', shape=[kernel_size[0], kernel_size[1], out_channels, in_channels], stddev=stddev)
    convt = tf.nn.conv2d_transpose(images, kernel, output_shape=[flags['batch_size'], out_img_size[0], out_img_size[1], out_channels], strides=[1, stride[0], stride[1], 1], padding='VALID')
    biases = bias_variable(name='biases', shape=[out_channels], constant=0.0)
    bias = tf.nn.bias_add(convt, biases)
    activate = relu(bias)
    return activate

##############################################################################
def up_conv_sigmoid(images, kernel_size, stride, out_img_size, in_channels, out_channels,flags):
    """
    up_conv_sigmoid creates an upsampling convolutional with the sigmoid activation function

    param: images
    param: kernel_size
    param: stride
    param: out_img_size
    param: in_channels
    param: out_channels
    param: batch_size
    return: convolutional layer tensor with specified parameters
    """

    stddev = tf.sqrt(2 / tf.to_float(kernel_size[0] * kernel_size[1] * in_channels))
    kernel = weight_variable(name='weights', shape=[kernel_size[0], kernel_size[1], out_channels, in_channels], stddev=stddev)
    convt = tf.nn.conv2d_transpose(images, kernel, output_shape=[flags['batch_size'], out_img_size[0], out_img_size[1], out_channels], strides=[1, stride[0], stride[1], 1], padding='VALID')
    biases = bias_variable(name='biases', shape=[out_channels], constant=0.0)
    bias = tf.nn.bias_add(convt, biases)
    activate = sigmoid(bias)
    return activate

##############################################################################
def full_conv_relu_valid(reshape, out_channels):
    """
    full_conv_relu_valid uses the reshape operator to create a fully convolutional
    layer of the specified length with the ReLU activation function

    param: reshape
    param: out_channels
    return: fully convolutional layer tensor of the specified length
    """

    dim = reshape.get_shape()[1].value
    stddev = tf.sqrt(2 / tf.to_float(dim))
    weights = weight_variable(name='weights', shape=[dim, out_channels], stddev=stddev)
    biases = bias_variable(name='biases', shape=[out_channels], constant=0.0)
    images = tf.nn.relu(tf.matmul(reshape, weights) + biases)
    return images

##############################################################################
def full_conv_valid(reshape, out_channels):
    """
    full_conv_valid uses the reshape operator to create a fully convolutional
    layer of the specified length

    param: reshape
    param: out_channels
    return: fully convolutional layer tensor of the specified length
    """

    dim = reshape.get_shape()[1].value
    stddev = tf.sqrt(2 / tf.to_float(dim))
    weights = weight_variable(name='weights', shape=[dim, out_channels], stddev=stddev)
    biases = bias_variable(name='biases', shape=[out_channels], constant=0.0)
    images = tf.matmul(reshape, weights) + biases
    return images

##############################################################################
def soft_max(logits):
    """
    soft_max applies a multi-class sigmoid operator to all of the elements in tensor

    param: logits
    return: predicted class probabilities
    """

    max_ = tf.reduce_max(logits, axis=[3], keep_dims=True)
    numerator = tf.exp(logits - max_)
    denominator = tf.reduce_sum(numerator, axis=[3], keep_dims=True)
    softmax = tf.div(numerator, denominator)
    return softmax

##############################################################################
def dropout(images,keep_prob):
    """
    dropout inactivates a certain portion of neurons in a layer as specified by keep_prob

    param: images
    param: keep_prob
    return: layer with dropout applied
    """

    images = tf.nn.dropout(images,keep_prob)
    images = tf.nn.dropout(images,keep_prob)
    return images
