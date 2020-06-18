"""
tf_model.py

This file contains the convolutional neural network architecture definition,
as well as the loss function definition and evaluation metrics. It also
contains the high level definition for model training.
"""

import tensorflow as tf
from KS_lib.tf_op import layer
from sklearn import metrics
import numpy as np

##############################################################################
def inference(images, keep_prob, flags):
    """
    inference contains the activations for the custom 21-layer
    fully convolutional VGG-inspired network architecture

    param: RGB images (144 x 144 x 3)
    param: keep_prob
    param: n_classes
    return: pixelwise class probability predictions
    """

    #downsampling block 1
    with tf.variable_scope('down1'):

        #conv layer 1_1: 144 x 144 x 32
        with tf.variable_scope('conv1'):
            down1_conv1 = layer.down_conv_relu_same(images, [3, 3], flags['size_input_patch'][2], 32)

        #conv layer 1_2: 144 x 144 x 32
        with tf.variable_scope('conv2'):
            down1_conv2 = layer.down_conv_relu_same(down1_conv1, [3, 3], 32, 32)

        #max pool layer 1_3: 144 x 144 x 32 --> 72 x 72 x 32
        with tf.variable_scope('pool'):
            down1_pool = layer.max_pool(down1_conv2, name='pool')

    #downsampling block 2
    with tf.variable_scope('down2'):

        #conv layer 2_1: 72 x 72 x 64
        with tf.variable_scope('conv1'):
            down2_conv1 = layer.down_conv_relu_same(down1_pool, [3, 3], 32, 64)

        #conv layer 2_2: 72 x 72 x 64
        with tf.variable_scope('conv2'):
            down2_conv2 = layer.down_conv_relu_same(down2_conv1, [3, 3], 64, 64)

        #max pool layer 2_3: 72 x 72 x 64 --> 36 x 36 x 64
        with tf.variable_scope('pool'):
            down2_pool = layer.max_pool(down2_conv2, name='pool')

    #downsampling block 3
    with tf.variable_scope('down3'):

        #conv layer 3_1: 36 x 36 x 128
        with tf.variable_scope('conv1'):
            down3_conv1 = layer.down_conv_relu_same(down2_pool, [3, 3], 64, 128)

        #conv layer 3_2: 36 x 36 x 128
        with tf.variable_scope('conv2'):
            down3_conv2 = layer.down_conv_relu_same(down3_conv1, [3, 3], 128, 128)

        #max pool layer 3_3: 36 x 36 x 128 --> 18 x 18 x 128
        with tf.variable_scope('pool'):
            down3_pool = layer.max_pool(down3_conv2, name='pool')

    #downsampling block 4
    with tf.variable_scope('down4'):

        #conv layer 4_1: 18 x 18 x 256
        with tf.variable_scope('conv1'):
            down4_conv1 = layer.down_conv_relu_same(down3_pool, [3, 3], 128, 256)

        #conv layer 4_2: 18 x 18 x 256
        with tf.variable_scope('conv2'):
            down4_conv2 = layer.down_conv_relu_same(down4_conv1, [3, 3], 256, 256)

        #max pool 4_3: 18 x 18 x 256 --> 9 x 9 x 256
        with tf.variable_scope('pool'):
            down4_pool = layer.max_pool(down4_conv2, name='pool')

    #downsampling block 5
    with tf.variable_scope('down5'):

        #conv layer 5_1: 9 x 9 x 512
        with tf.variable_scope('conv1'):
            down5_conv1 = layer.down_conv_relu_same(down4_pool, [3, 3], 256, 512)

        #dropout layer 5_1
        with tf.variable_scope('dropout1'):
            down5_drop1 = layer.dropout(down5_conv1,keep_prob)

        #conv layer 5_2: 9 x 9 x 512
        with tf.variable_scope('conv2'):
            down5_conv2 = layer.down_conv_relu_same(down5_drop1, [3, 3], 512, 512)

        #dropout layer 5_2
        with tf.variable_scope('dropout2'):
            down5_drop2 = layer.dropout(down5_conv2,keep_prob)

        #nearest neighbor resize layer 5_3: 9 x 9 x 512 --> 18 x 18 x 512
        with tf.variable_scope('tconv1'):
            down5_tconv = tf.image.resize_nearest_neighbor(down5_drop2, [18, 18])

        #conv layer 5_4: 18 x 18 x 256
        with tf.variable_scope('tconv1_'):
            down5_tconv = layer.down_conv_same(down5_tconv, [3, 3], 512, 256)


    #upsampling block 1
    with tf.variable_scope('up1'):

        #dropout layer and concatentation layer
        with tf.variable_scope('concat'):
            down4_conv2_down = layer.dropout(down4_conv2, keep_prob)
            up1_concat = tf.concat(3, [down5_tconv, down4_conv2_down])

        #conv layer 1_1: 18 x 18 x 256
        with tf.variable_scope('conv1'):
            up1_conv1 = layer.down_conv_relu_same(up1_concat, [3, 3], 512, 256)

        #conv layer 1_2: 18 x 18 x 256
        with tf.variable_scope('conv2'):
            up1_conv2 = layer.down_conv_relu_same(up1_conv1, [3, 3], 256, 256)

        #nearest neighbor resize layer 1_3: 18 x 18 x 256 --> 36 x 36 x 256
        with tf.variable_scope('tconv1'):
            up1_tconv = tf.image.resize_nearest_neighbor(up1_conv2, [36, 36])

        #conv layer 1_4: 36 x 36 x 128
        with tf.variable_scope('tconv1_'):
            up1_tconv = layer.down_conv_same(up1_tconv, [3, 3], 256, 128)


    #upsampling block 2
    with tf.variable_scope('up2'):

        # dropout layer and concatentation layer
        with tf.variable_scope('concat'):
            down3_conv2_down = layer.dropout(down3_conv2, keep_prob)
            up2_concat = tf.concat(3, [up1_tconv, down3_conv2_down])

        #conv layer 2_1: 36 x 36 x 128
        with tf.variable_scope('conv1'):
            up2_conv1 = layer.down_conv_relu_same(up2_concat, [3, 3], 256, 128)

        #conv layer 2_2: 36 x 36 x 128
        with tf.variable_scope('conv2'):
            up2_conv2 = layer.down_conv_relu_same(up2_conv1, [3, 3], 128, 128)

        #bilinear resize layer 2_3: 36 x 36 x 128 --> 72 x 72 x 128
        with tf.variable_scope('tconv1'):
            up2_tconv = tf.image.resize_images(up2_conv2, [72, 72])

        #conv layer 2_4: 72 x 72 x 64
        with tf.variable_scope('tconv1_'):
            up2_tconv = layer.down_conv_same(up2_tconv, [3, 3], 128, 64)


    #upsampling block 3
    with tf.variable_scope('up3'):

        #dropout layer and concatenation layer
        with tf.variable_scope('concat'):
            down2_conv2_down = layer.dropout(down2_conv2, keep_prob)
            up3_concat = tf.concat(3, [up2_tconv, down2_conv2_down])

        #conv layer 3_1: 72 x 72 x 64
        with tf.variable_scope('conv1'):
            up3_conv1 = layer.down_conv_relu_same(up3_concat, [3, 3], 128, 64)

        #conv layer 3_2: 72 x 72 x 64
        with tf.variable_scope('conv2'):
            up3_conv2 = layer.down_conv_relu_same(up3_conv1, [3, 3], 64, 64)

        #nearest neighbor resize layer 3_3: 72 x 72 x 64 --> 144 x 144 x 64
        with tf.variable_scope('tconv1'):
            up3_tconv = tf.image.resize_nearest_neighbor(up3_conv2, [144, 144])

        #conv layer 3_4: 144 x 144 x 32
        with tf.variable_scope('tconv1_'):
            up3_tconv = layer.down_conv_same(up3_tconv, [3, 3], 64, 32)

    #upsizing block 4
    with tf.variable_scope('up4'):

        #dropout layer and concatenation layer
        with tf.variable_scope('concat'):
            down1_conv2_down = layer.dropout(down1_conv2, keep_prob)
            up4_concat = tf.concat(3, [up3_tconv, down1_conv2_down])

        #conv layer 4_1: 144 x 144 x 32
        with tf.variable_scope('conv1'):
            up4_conv1 = layer.down_conv_relu_same(up4_concat, [3, 3], 64, 32)

        #conv layer 4_2: 144 x 144 x 32
        with tf.variable_scope('conv2'):
            up4_conv2 = layer.down_conv_relu_same(up4_conv1, [3, 3], 32, 32)

        #conv layer 4_3: 144 x 144 x 3
        with tf.variable_scope('conv3'):
            up4_conv3 = layer.down_conv_same(up4_conv2, [3, 3], 32, 3)

        #softmax layer 4_4: 144 x 144 x 3
        with tf.variable_scope('softmax'):
            softmax = tf.nn.softmax(up4_conv3)

    return softmax, {'softmax':softmax}


##############################################################################
def loss(softmax, labels, weights, curr_epoch , flags):
    """
    loss contains the definition of the cross-entropy loss function, which uses
    normalization based on class area to eliminate class imbalance

    param: softmax
    param: labels
    param: weights
    param: curr_epoch
    param: alpha
    param: n_classes
    return: pixelwise cross-entropy loss function
    """

    # number of postives and negatives
    neg_mask = tf.to_float(tf.equal(labels,0)) #whether the labels = 0 or not
    pos_mask = tf.to_float(tf.equal(labels, 1)) #whether the labels = 1 or not
    bd_mask = tf.to_float(tf.equal(labels,2)) #whether the labels = 2 or not

    # entries in the 4D boolean tensors are summed together to get total counts
    n_pos = tf.to_float(tf.reduce_sum(pos_mask, [0, 1, 2, 3]))
    n_neg = tf.to_float(tf.reduce_sum(neg_mask, [0, 1, 2, 3]))
    n_bd = tf.to_float(tf.reduce_sum(bd_mask, [0, 1, 2, 3]))

    max_val = tf.reduce_max(tf.stack([n_pos,n_neg,n_bd])) #find predicted class with highest probability for each pixel

    # apply empirically determined weights, bg = 1, nuclei = 1, border = 1
    pos_weights = (pos_mask/n_pos)*max_val*1.0
    neg_weights = (neg_mask/n_neg)*max_val*1.0
    bd_weights = (bd_mask/n_bd)*max_val*1.0

    # sum all weights and add the bias term (alpha * weights)
    class_weights = pos_weights + neg_weights + bd_weights
    total_weights = class_weights + flags['alpha'] * weights

    # labels turned into one-hot format for softmax
    labels = tf.squeeze(labels)
    labels = tf.cast(labels, tf.int64)
    onehot = tf.one_hot(labels, depth=flags['n_classes'], on_value=1.0, off_value=0.0, axis=3)

    epsilon = 1e-6
    truncated_softmax = tf.clip_by_value(softmax, epsilon, 1.0 - epsilon)

    # average cross_entropy_log_loss calculated using the truncated_softmax function
    cross_entropy_log_loss = -tf.reduce_sum(onehot * tf.log(truncated_softmax), reduction_indices=[3], keep_dims=True)
    cross_entropy_log_loss = (total_weights) * cross_entropy_log_loss #loss is weighted to combat class imbalance
    avg_cross_entropy_log_loss = tf.reduce_mean(cross_entropy_log_loss, reduction_indices=[0, 1, 2])

    return {'truncated_softmax': truncated_softmax,
            'cross_entropy_log_loss': cross_entropy_log_loss,
            'avg_cross_entropy_log_loss': avg_cross_entropy_log_loss,
            'labels': labels,
            'onehot': onehot,
            'pos_mask': pos_mask,
            'neg_mask': neg_mask,
            'bd_mask':bd_mask,
            'n_pos':n_pos,
            'n_neg':n_neg,
            'n_bd':n_bd,
            'class_weights':class_weights,
            'total_weights':total_weights}

##############################################################################
def train(total_loss, global_step, parameters, flags):
    """
    train adjusts the network's weights by minimizing the loss function
    using Adam optimizer

    param: total_loss
    param: global_step
    param: parameters
    param: initial_learning_rate
    return: Adam based training operator (gradient update)
    """

    optimizer = tf.train.AdamOptimizer(learning_rate=flags['initial_learning_rate'])
    grads_and_vars = optimizer.compute_gradients(total_loss)
    train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

    return train_op

##############################################################################
def accuracy(predicts, labels, flags):
    """
    accuracy computes numerous useful accuracy metrics including TP, FP, TN, FN, F-1 score,
    precision, and recall

    param: predicts
    param: labels
    param: n_classes
    return: TP, FP, TN, FN, precision, recall, F-1 score
    """

    predicts = tf.reshape(predicts, [-1])
    labels = tf.reshape(labels, [-1])

    TP = [0] * flags['n_classes']
    FP = [0] * flags['n_classes']
    FN = [0] * flags['n_classes']
    TN = [0] * flags['n_classes']
    precision = [0] * flags['n_classes']
    recall = [0] * flags['n_classes']
    f1score = [0] * flags['n_classes']

    for iclass in range(flags['n_classes']):
        TP[iclass] = tf.reduce_sum(tf.to_float(tf.logical_and(tf.equal(predicts, iclass), tf.equal(labels, iclass))))
        FP[iclass] = tf.reduce_sum(tf.to_float(tf.logical_and(tf.equal(predicts, iclass), tf.not_equal(labels, iclass))))
        FN[iclass] = tf.reduce_sum(tf.to_float(tf.logical_and(tf.not_equal(predicts, iclass), tf.equal(labels, iclass))))
        TN[iclass] = tf.reduce_sum(tf.to_float(tf.logical_and(tf.not_equal(predicts, iclass), tf.not_equal(labels, iclass))))

        precision[iclass] = TP[iclass] / tf.to_float(TP[iclass] + FP[iclass])
        recall[iclass] = TP[iclass] / tf.to_float(TP[iclass] + FN[iclass])
        f1score[iclass] = 2 * precision[iclass] * recall[iclass] / tf.to_float(precision[iclass] + recall[iclass])

    return {'TP': TP,
            'FP': FP,
            'FN': FN,
            'TN': TN,
            'precision': precision,
            'recall': recall,
            'f1score': f1score}
