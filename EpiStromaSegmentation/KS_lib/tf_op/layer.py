####################################
#All the layer activation definitions
####################################


import tensorflow as tf
import numpy as np

##############################################################################
#smoothes textures in image when they are displayed larger or smaller than actual size
def bilinear_filter(size):
    factor = (size + 1) // 2
    if size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:size, :size]
    return (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor)


##############################################################################
#just creates a filter using bilinear(?)
def upsample_filt(kernel_size, out_channels, in_channels):
    if (kernel_size[0] != kernel_size[1]) or (out_channels != in_channels):
        raise ValueError('kernel_size_row != kernel_size_col or out_channels != in channels')

    filt = np.zeros((kernel_size[0], kernel_size[1], out_channels, in_channels),
                    dtype=np.float32)

    for i in range(in_channels):
        filt[:, :, i, i] = bilinear_filter(kernel_size[0])

    return filt


############################################################################## ADDED BELOW
#creates n-dimensional identity tensor
def identity_initializer():
    def _initializer(shape, dtype=tf.float32, partition_info=None):
        if len(shape) == 1: #1D identity is just 0
            return tf.constant_op.constant(0., dtype=dtype, shape=shape)
        elif len(shape) == 2 and shape[0] == shape[1]: #2D identity must be square
            return tf.constant_op.constant(np.identity(shape[0], dtype))
        elif len(shape) == 4 and shape[2] == shape[3]: #4D identity works if last 2 dimensions are square
            #example: 2 x 6 x 4 x 4 tensor --> identity is just 1s along the middle 4 x 4 patch's diagonal(1 x 3 x 4 x 4)

            array = np.zeros(shape, dtype=float)
            cx, cy = shape[0]/2, shape[1]/2
            for i in range(shape[2]):
                array[cx, cy, i, i] = 1
            return tf.constant(array, dtype=dtype)
        #elif len(shape) == 4 and shape[2] != shape[3]:
            #array = np.zeros(shape, dtype=float)
            #cx, cy = shape[0]/2, shape[1]/2
            #for i in range(min(shape[2], shape[3]):
            #    array[cx, cy, i, i] = 1
            #for i in range(shape[2], shape[3]):
            #    array[:, :, :, i
        else:
            raise Exception('Shape not valid')
    return _initializer

##############################################################################
#just returns random weights with the specified standard deviation
def weight_variable(name, shape, stddev):
    return tf.get_variable(name, shape,
                           initializer=tf.truncated_normal_initializer(stddev=stddev))
############################################################################# BROKEN
def identity_variable(name, shape):
    #array = np.zeros(shape, dtype=float)
    #cx, cy = shape[0]/2, shape[1]/2
    #for i in range(shape[2]):
    #    array[cx, cy, i, i] = 1.0
    #init = tf.constant_op.constant(array, dtype=tf.float32)
    #init = tf.constant_initializer(array, dtype=tf.float32)
    return tf.get_variable(name, shape, 
                           initializer=identity_initializer())

##############################################################################
def bias_variable(name, shape, constant):
    return tf.get_variable(name, shape, initializer=tf.constant_initializer(constant))


##############################################################################
def relu(inTensor):
    return tf.nn.relu(inTensor)


##############################################################################
def sigmoid(inTensor):
    return tf.sigmoid(inTensor)


##############################################################################
#2 x 2 max pool
def max_pool(inTensor, name):
    return tf.nn.max_pool(inTensor, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                          padding='VALID', name=name)





##############################################################################
#ACTIVATION DEFINITONS BELOW
##############################################################################
#DOWN = downsampling layers, UP = upsampling layers
##############################################################################
#executes normal conv layer + relu, uses padding = same(ex: of 13 inputs, pad with 3 0s for a length 6 filter with stride 5 to work)
def down_conv_relu_same(images, kernel_size, in_channels, out_channels):
    stddev = tf.sqrt(2 / tf.to_float(kernel_size[0] * kernel_size[1] * in_channels))
    kernel = weight_variable(name='weights', shape=[kernel_size[0], kernel_size[1], in_channels, out_channels], stddev=stddev)
    conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME')
    biases = bias_variable(name='biases', shape=[out_channels],constant=0.0)
    bias = tf.nn.bias_add(conv, biases)
    activate = relu(bias)
    return activate

##############################################################################
#executes dilated conv layer + relu with padding = same
def down_conv_relu_dilated_same(images, kernel_size, in_channels, out_channels, dilation_factor):
    stddev = tf.sqrt(2 / tf.to_float(kernel_size[0] * kernel_size[1] * in_channels))
    kernel = weight_variable(name='weights', shape=[kernel_size[0], kernel_size[1], in_channels, out_channels], stddev=stddev)
    #conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME')
    conv = tf.nn.atrous_conv2d(images, kernel, dilation_factor, padding="SAME") #dilated convolutions executed with given filter and specified dilation factor
    biases = bias_variable(name='biases', shape=[out_channels], constant=0.0)
    bias = tf.nn.bias_add(conv, biases)
    activate = relu(bias)
    return activate

##############################################################################
#executes conv layer + relu with padding = valid(ex: of 13 inputs, no padding so for length 6 filter with stride 5 the final 2 entries are dropped)
def down_conv_relu_valid(images, kernel_size, in_channels, out_channels):
    stddev = tf.sqrt(2 / tf.to_float(kernel_size[0] * kernel_size[1] * in_channels))
    kernel = weight_variable(name='weights', shape=[kernel_size[0], kernel_size[1], in_channels, out_channels], stddev=stddev)
    conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='VALID')
    biases = bias_variable(name='biases', shape=[out_channels], constant=0.0)
    bias = tf.nn.bias_add(conv, biases)
    activate = relu(bias)
    return activate

##############################################################################
#executes dilated conv layer + relu with padding = valid
def down_conv_relu_dilated_valid(images, kernel_size, in_channels, out_channels, dilation_factor):
    stddev = tf.sqrt(2 / tf.to_float(kernel_size[0] * kernel_size[1] * in_channels))
    kernel = weight_variable(name='weights', shape=[kernel_size[0], kernel_size[1], in_channels, out_channels],stddev=stddev)
    # conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME')
    conv = tf.nn.atrous_conv2d(images, kernel, dilation_factor, padding="VALID")
    biases = bias_variable(name='biases', shape=[out_channels], constant=0.0)
    bias = tf.nn.bias_add(conv, biases)
    activate = relu(bias)
    return activate

##############################################################################
#executes dilated conv layer + relu with padding = same, just does not use std dev parameter
def down_conv_relu_dilated_same(images, kernel_size, in_channels, out_channels, dilation_factor):
    stddev = tf.sqrt(2 / tf.to_float(kernel_size[0] * kernel_size[1] * in_channels))
    kernel = identity_variable(name='weights', shape=[kernel_size[0], kernel_size[1], in_channels, out_channels])#, #stddev=stddev)
    # conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME')
    conv = tf.nn.atrous_conv2d(images, kernel, dilation_factor, padding="SAME")
    biases = bias_variable(name='biases', shape=[out_channels], constant=0.0)
    bias = tf.nn.bias_add(conv, biases)
    activate = relu(bias)
    return activate


##############################################################################
#should just be FC layer activation + relu, check what "reshape" stands for
def full_conv_relu_valid(reshape, out_channels):
    dim = reshape.get_shape()[1].value
    stddev = tf.sqrt(2 / tf.to_float(dim))
    weights = weight_variable(name='weights', shape=[dim, out_channels], stddev=stddev)
    biases = bias_variable(name='biases', shape=[out_channels], constant=0.0)
    images = tf.nn.relu(tf.matmul(reshape, weights) + biases)
    return images


##############################################################################
#should just be FC layer activation(Wx + b), check what "reshape" stands for
def full_conv_valid(reshape, out_channels):
    dim = reshape.get_shape()[1].value
    stddev = tf.sqrt(2 / tf.to_float(dim))
    weights = weight_variable(name='weights', shape=[dim, out_channels], stddev=stddev)
    biases = bias_variable(name='biases', shape=[out_channels], constant=0.0)
    images = tf.matmul(reshape, weights) + biases
    return images


##############################################################################
#activation of normal conv layer with SAME Padding, just no relu
def down_conv_same(images, kernel_size, in_channels, out_channels):
    stddev = tf.sqrt(2 / tf.to_float(kernel_size[0] * kernel_size[1] * in_channels))
    kernel = weight_variable(name='weights', shape=[kernel_size[0], kernel_size[1], in_channels, out_channels], stddev=stddev)
    conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME')
    biases = bias_variable(name='biases', shape=[out_channels], constant=0.0)
    bias = tf.nn.bias_add(conv, biases)
    return bias

###############################################################################
#activation of dilated conv layer with VALID padding, just no relu
def down_conv_dilated_valid(images, kernel_size, in_channels, out_channels, dilation_factor):
    stddev = tf.sqrt(2 / tf.to_float(kernel_size[0] * kernel_size[1] * in_channels))
    kernel = weight_variable(name='weights', shape=[kernel_size[0], kernel_size[1], in_channels, out_channels], stddev=stddev)
    # conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME')
    conv = tf.nn.atrous_conv2d(images, kernel, dilation_factor, padding="VALID")
    biases = bias_variable(name='biases', shape=[out_channels], constant=0.0)
    bias = tf.nn.bias_add(conv, biases)
    return bias

###############################################################################
#activation of dilated conv layer with SAME padding, no relu, no std dev
def down_conv_dilated_same(images, kernel_size, in_channels, out_channels, dilation_factor):
    stddev = tf.sqrt(2 / tf.to_float(kernel_size[0] * kernel_size[1] * in_channels))
    kernel = identity_variable(name='weights', shape=[kernel_size[0], kernel_size[1], in_channels, out_channels])#, #stddev=stddev)
    # conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME')
    conv = tf.nn.atrous_conv2d(images, kernel, dilation_factor, padding="SAME")
    biases = bias_variable(name='biases', shape=[out_channels], constant=0.0)
    bias = tf.nn.bias_add(conv, biases)
    return bias


##############################################################################
#activation of normal conv layer with VALID padding, no relu
def down_conv_valid(images, kernel_size, in_channels, out_channels):
    stddev = tf.sqrt(2 / tf.to_float(kernel_size[0] * kernel_size[1] * in_channels))
    kernel = weight_variable(name='weights', shape=[kernel_size[0], kernel_size[1], in_channels, out_channels], stddev=stddev)
    conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='VALID')
    biases = bias_variable(name='biases', shape=[out_channels], constant=0.0)
    bias = tf.nn.bias_add(conv, biases)
    return bias


##############################################################################
#activation of upsampling conv layer with no relu and no padding
def up_conv(images, kernel_size, stride, out_img_size, in_channels, out_channels, flags):
    stddev = tf.sqrt(2 / tf.to_float(kernel_size[0] * kernel_size[1] * in_channels))
    kernel = weight_variable(name='weights', shape=[kernel_size[0], kernel_size[1], out_channels, in_channels], stddev=stddev)
    #uses conv2d_transpose to implement upsampling layer
    convt = tf.nn.conv2d_transpose(images, kernel, output_shape=[flags['batch_size'], out_img_size[0], out_img_size[1], out_channels], strides=[1, stride[0], stride[1], 1], padding='VALID')
    biases = bias_variable(name='biases', shape=[out_channels], constant=0.0)
    bias = tf.nn.bias_add(convt, biases)
    return bias


##############################################################################
#activation of upsampling conv layer with relu and no padding
def up_conv_relu(images, kernel_size, stride, out_img_size, in_channels, out_channels, flags):
    stddev = tf.sqrt(2 / tf.to_float(kernel_size[0] * kernel_size[1] * in_channels))
    kernel = weight_variable(name='weights', shape=[kernel_size[0], kernel_size[1], out_channels, in_channels], stddev=stddev)
    convt = tf.nn.conv2d_transpose(images, kernel, output_shape=[flags['batch_size'], out_img_size[0], out_img_size[1], out_channels], strides=[1, stride[0], stride[1], 1], padding='VALID')
    biases = bias_variable(name='biases', shape=[out_channels], constant=0.0)
    bias = tf.nn.bias_add(convt, biases)
    activate = relu(bias)
    return activate


##############################################################################
#activation of upsampling conv layer with sigmoid and no padding
def up_conv_sigmoid(images, kernel_size, stride, out_img_size, in_channels, out_channels,flags):
    stddev = tf.sqrt(2 / tf.to_float(kernel_size[0] * kernel_size[1] * in_channels))
    kernel = weight_variable(name='weights', shape=[kernel_size[0], kernel_size[1], out_channels, in_channels], stddev=stddev)
    convt = tf.nn.conv2d_transpose(images, kernel, output_shape=[flags['batch_size'], out_img_size[0], out_img_size[1], out_channels], strides=[1, stride[0], stride[1], 1], padding='VALID')
    biases = bias_variable(name='biases', shape=[out_channels], constant=0.0)
    bias = tf.nn.bias_add(convt, biases)
    activate = sigmoid(bias)
    return activate


##############################################################################
# def up_conv_bilinear(images, kernel_size, stride, out_img_size, in_channels, out_channels):
#     filt = upsample_filt(kernel_size, out_channels, in_channels)
#     filt = tf.convert_to_tensor(filt, dtype=tf.float32)
#     kernel = tf.get_variable(name='weights',
#                              initializer=filt,
#                              trainable=True)
#     convt = tf.nn.conv2d_transpose(images, kernel,
#                                    output_shape=[flags.batch_size, out_img_size[0], out_img_size[1], out_channels],
#                                    strides=[1, stride[0], stride[1], 1],
#                                    padding='VALID')
#     biases = bias_variable(name='biases',
#                            shape=[out_channels],
#                            constant=0.0)
#     bias = tf.nn.bias_add(convt, biases)
#     return bias


##############################################################################
#helper function for softmax
def soft_max(logits):
    max_ = tf.reduce_max(logits, axis=[3], keep_dims=True)
    numerator = tf.exp(logits - max_)
    denominator = tf.reduce_sum(numerator, axis=[3], keep_dims=True)
    softmax = tf.div(numerator, denominator)
    return softmax

##############################################################################
#helper function for dropout(keepprob = 0.5)
def dropout(images,keep_prob):
    images = tf.nn.dropout(images,keep_prob)
    images = tf.nn.dropout(images,keep_prob)
    return images
