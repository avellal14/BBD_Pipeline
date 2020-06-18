import tensorflow as tf
import numpy as np

##############################################################################
def bilinear_filter(size):
    factor = (size + 1) // 2
    if size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:size, :size]
    return (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor)


##############################################################################
def upsample_filt(kernel_size, out_channels, in_channels):
    if (kernel_size[0] != kernel_size[1]) or (out_channels != in_channels):
        raise ValueError('kernel_size_row != kernel_size_col or out_channels != in channels')

    filt = np.zeros((kernel_size[0], kernel_size[1], out_channels, in_channels),
                    dtype=np.float32)

    for i in range(in_channels):
        filt[:, :, i, i] = bilinear_filter(kernel_size[0])

    return filt


##############################################################################
def weight_variable(name, shape, stddev):
    return tf.get_variable(name, shape,
                           initializer=tf.truncated_normal_initializer(stddev=stddev))


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
def max_pool(inTensor, name):
    return tf.nn.max_pool(inTensor, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                          padding='VALID', name=name)


##############################################################################
def down_conv_relu_same(images, kernel_size, in_channels, out_channels):
    stddev = tf.sqrt(2 / tf.to_float(kernel_size[0] * kernel_size[1] * in_channels))
    kernel = weight_variable(name='weights',
                             shape=[kernel_size[0], kernel_size[1], in_channels, out_channels],
                             stddev=stddev)
    conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME')
    biases = bias_variable(name='biases',
                           shape=[out_channels],
                           constant=0.0)
    bias = tf.nn.bias_add(conv, biases)
    activate = relu(bias)
    return activate


##############################################################################
def down_conv_relu_valid(images, kernel_size, in_channels, out_channels):
    stddev = tf.sqrt(2 / tf.to_float(kernel_size[0] * kernel_size[1] * in_channels))
    kernel = weight_variable(name='weights',
                             shape=[kernel_size[0], kernel_size[1], in_channels, out_channels],
                             stddev=stddev)
    conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='VALID')
    biases = bias_variable(name='biases',
                           shape=[out_channels],
                           constant=0.0)
    bias = tf.nn.bias_add(conv, biases)
    activate = relu(bias)
    return activate


##############################################################################
def full_conv_relu_valid(reshape, out_channels):
    dim = reshape.get_shape()[1].value
    stddev = tf.sqrt(2 / tf.to_float(dim))
    weights = weight_variable(name='weights', shape=[dim, out_channels], stddev=stddev)
    biases = bias_variable(name='biases', shape=[out_channels], constant=0.0)
    images = tf.nn.relu(tf.matmul(reshape, weights) + biases)
    return images


##############################################################################
def full_conv_valid(reshape, out_channels):
    dim = reshape.get_shape()[1].value
    stddev = tf.sqrt(2 / tf.to_float(dim))
    weights = weight_variable(name='weights', shape=[dim, out_channels], stddev=stddev)
    biases = bias_variable(name='biases', shape=[out_channels], constant=0.0)
    images = tf.matmul(reshape, weights) + biases
    return images


##############################################################################
def down_conv_same(images, kernel_size, in_channels, out_channels):
    stddev = tf.sqrt(2 / tf.to_float(kernel_size[0] * kernel_size[1] * in_channels))

    kernel = weight_variable(name='weights',
                             shape=[kernel_size[0], kernel_size[1], in_channels, out_channels],
                             stddev=stddev)
    conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME')
    biases = bias_variable(name='biases',
                           shape=[out_channels],
                           constant=0.0)
    bias = tf.nn.bias_add(conv, biases)
    return bias


##############################################################################
def down_conv_valid(images, kernel_size, in_channels, out_channels):
    stddev = tf.sqrt(2 / tf.to_float(kernel_size[0] * kernel_size[1] * in_channels))

    kernel = weight_variable(name='weights',
                             shape=[kernel_size[0], kernel_size[1], in_channels, out_channels],
                             stddev=stddev)
    conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='VALID')
    biases = bias_variable(name='biases',
                           shape=[out_channels],
                           constant=0.0)
    bias = tf.nn.bias_add(conv, biases)
    return bias


##############################################################################
def up_conv(images, kernel_size, stride, out_img_size, in_channels, out_channels, flags):
    stddev = tf.sqrt(2 / tf.to_float(kernel_size[0] * kernel_size[1] * in_channels))
    kernel = weight_variable(name='weights',
                             shape=[kernel_size[0], kernel_size[1], out_channels, in_channels],
                             stddev=stddev)
    convt = tf.nn.conv2d_transpose(images, kernel,
                                   output_shape=[flags['batch_size'], out_img_size[0], out_img_size[1], out_channels],
                                   strides=[1, stride[0], stride[1], 1],
                                   padding='VALID')
    biases = bias_variable(name='biases',
                           shape=[out_channels],
                           constant=0.0)
    bias = tf.nn.bias_add(convt, biases)
    return bias


##############################################################################
def up_conv_relu(images, kernel_size, stride, out_img_size, in_channels, out_channels, flags):
    stddev = tf.sqrt(2 / tf.to_float(kernel_size[0] * kernel_size[1] * in_channels))
    kernel = weight_variable(name='weights',
                             shape=[kernel_size[0], kernel_size[1], out_channels, in_channels],
                             stddev=stddev)
    convt = tf.nn.conv2d_transpose(images, kernel,
                                   output_shape=[flags['batch_size'], out_img_size[0], out_img_size[1], out_channels],
                                   strides=[1, stride[0], stride[1], 1],
                                   padding='VALID')
    biases = bias_variable(name='biases',
                           shape=[out_channels],
                           constant=0.0)
    bias = tf.nn.bias_add(convt, biases)
    activate = relu(bias)
    return activate


##############################################################################
def up_conv_sigmoid(images, kernel_size, stride, out_img_size, in_channels, out_channels,flags):
    stddev = tf.sqrt(2 / tf.to_float(kernel_size[0] * kernel_size[1] * in_channels))
    kernel = weight_variable(name='weights',
                             shape=[kernel_size[0], kernel_size[1], out_channels, in_channels],
                             stddev=stddev)
    convt = tf.nn.conv2d_transpose(images, kernel,
                                   output_shape=[flags['batch_size'], out_img_size[0], out_img_size[1], out_channels],
                                   strides=[1, stride[0], stride[1], 1],
                                   padding='VALID')
    biases = bias_variable(name='biases',
                           shape=[out_channels],
                           constant=0.0)
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
def soft_max(logits):
    max_ = tf.reduce_max(logits, reduction_indices=[3], keep_dims=True)
    numerator = tf.exp(logits - max_)
    denominator = tf.reduce_sum(numerator, reduction_indices=[3], keep_dims=True)
    softmax = tf.div(numerator, denominator)
    return softmax

##############################################################################
def dropout(images,keep_prob):
    images = tf.nn.dropout(images,keep_prob)
    return images
