"""
tf_model.py

This file contains the convolutional neural network architecture definition,
as well as the loss function definition and evaluation metrics. It also
contains the high level definition for model training.
"""

import tensorflow as tf
from KS_lib.tf_op import layer

##############################################################################
def inference(images, keep_prob, flags):
    """
    inference contains the activations for the custom 21-layer
    fully convolutional VGG-inspired network architecture

    param: RGB images (128 x 128 x 3)
    param: keep_prob
    param: n_classes
    return: pixelwise class probability predictions
    """

    print ("entered inference in tf_model.py")

    #conv layer 1_1: 128 x 128 x 64
    with tf.variable_scope('conv1_1') as scope:
        stddev = tf.sqrt(2 / tf.to_float(3 * 3 * 3))
        kernel1_1 = tf.get_variable(name='weights', shape = [3, 3, 3, 64], initializer=tf.truncated_normal_initializer(stddev=stddev))
        conv = tf.nn.conv2d(images, kernel1_1, [1, 1, 1, 1], padding='SAME')
        biases1_1 = tf.get_variable(name='biases', shape=[64], initializer=tf.constant_initializer(0.0))
        out = tf.nn.bias_add(conv, biases1_1)
        conv1_1 = tf.nn.relu(out, name=scope.name)

    #conv layer 1_1: 128 x 128 x 64
    with tf.variable_scope('conv1_2') as scope:
        stddev = tf.sqrt(2 / tf.to_float(3 * 3 * 64))
        kernel1_2 = tf.get_variable(name='weights', shape = [3, 3, 64, 64], initializer=tf.truncated_normal_initializer(stddev=stddev))
        conv = tf.nn.conv2d(conv1_1, kernel1_2, [1, 1, 1, 1], padding='SAME')
        biases1_2 = tf.get_variable(name='biases', shape=[64], initializer=tf.constant_initializer(0.0))
        out = tf.nn.bias_add(conv, biases1_2)
        conv1_2 = tf.nn.relu(out, name=scope.name)


    #max pool layer 1_3: 128 x 128 x 64 --> 64 x 64 x 64
    pool1 = tf.nn.max_pool(conv1_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool1')

    #conv layer 2_1: 64 x 64 x 128
    with tf.variable_scope('conv2_1') as scope:
        stddev = tf.sqrt(2 / tf.to_float(3 * 3 * 64))
        kernel2_1 = tf.get_variable(name='weights', shape=[3, 3, 64, 128], initializer=tf.truncated_normal_initializer(stddev=stddev))
        conv = tf.nn.conv2d(pool1, kernel2_1, [1, 1, 1, 1], padding='SAME')
        biases2_1 = tf.get_variable(name='biases', shape=[128], initializer=tf.constant_initializer(0.0))
        out = tf.nn.bias_add(conv, biases2_1)
        conv2_1 = tf.nn.relu(out, name=scope.name)

    #conv layer 2_2: 64 x 64 x 128
    with tf.variable_scope('conv2_2') as scope:
        stddev = tf.sqrt(2 / tf.to_float(3 * 3 * 128))
        kernel2_2 = tf.get_variable(name='weights', shape=[3, 3, 128, 128], initializer=tf.truncated_normal_initializer(stddev=stddev))
        conv = tf.nn.conv2d(conv2_1, kernel2_2, [1, 1, 1, 1], padding='SAME')
        biases2_2 = tf.get_variable(name='biases', shape=[128], initializer=tf.constant_initializer(0.0))
        out = tf.nn.bias_add(conv, biases2_2)
        conv2_2 = tf.nn.relu(out, name=scope.name)

    #max pool layer 2_3: 64 x 64 x 128 --> 32 x 32 x 128
    pool2 = tf.nn.max_pool(conv2_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool2')

    #conv layer 3_1: 32 x 32 x 256
    with tf.variable_scope('conv3_1') as scope:
        stddev = tf.sqrt(2 / tf.to_float(3 * 3 * 128))
        kernel3_1 = tf.get_variable(name='weights', shape=[3, 3, 128, 256], initializer=tf.truncated_normal_initializer(stddev=stddev))
        conv = tf.nn.conv2d(pool2, kernel3_1, [1, 1, 1, 1], padding='SAME')
        biases3_1 = tf.get_variable(name='biases', shape=[256], initializer=tf.constant_initializer(0.0))
        out = tf.nn.bias_add(conv, biases3_1)
        conv3_1 = tf.nn.relu(out, name=scope.name)

    #conv layer 3_2: 32 x 32 x 256
    with tf.variable_scope('conv3_2') as scope:
        stddev = tf.sqrt(2 / tf.to_float(3 * 3 * 256))
        kernel3_2 = tf.get_variable(name='weights', shape=[3, 3, 256, 256], initializer=tf.truncated_normal_initializer(stddev=stddev))
        conv = tf.nn.conv2d(conv3_1, kernel3_2, [1, 1, 1, 1], padding='SAME')
        biases3_2 = tf.get_variable(name='biases', shape=[256], initializer=tf.constant_initializer(0.0))
        out = tf.nn.bias_add(conv, biases3_2)
        conv3_2 = tf.nn.relu(out, name=scope.name)

    #max pool layer 3_3 : 32 x 32 x 256 --> 16 x 16 x 256
    pool3 = tf.nn.max_pool(conv3_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool3')

    #conv layer 4_1: 16 x 16 x 512
    with tf.variable_scope('conv4_1') as scope:
        stddev = tf.sqrt(2 / tf.to_float(3 * 3 * 256))
        kernel4_1 = tf.get_variable(name='weights', shape=[3, 3, 256, 512], initializer=tf.truncated_normal_initializer(stddev=stddev))
        conv = tf.nn.conv2d(pool3, kernel4_1, [1, 1, 1, 1], padding='SAME')
        biases4_1 = tf.get_variable(name='biases', shape=[512], initializer=tf.constant_initializer(0.0))
        out = tf.nn.bias_add(conv, biases4_1)
        conv4_1 = tf.nn.relu(out, name=scope.name)

    #conv layer 4_2: 16 x 16 x 512
    with tf.variable_scope('conv4_2') as scope:
        stddev = tf.sqrt(2 / tf.to_float(3 * 3 * 512))
        kernel4_2 = tf.get_variable(name='weights', shape=[3, 3, 512, 512], initializer=tf.truncated_normal_initializer(stddev=stddev))
        conv = tf.nn.conv2d(conv4_1, kernel4_2, [1, 1, 1, 1], padding='SAME')
        biases4_2 = tf.get_variable(name='biases', shape=[512], initializer=tf.constant_initializer(0.0))
        out = tf.nn.bias_add(conv, biases4_2)
        conv4_2 = tf.nn.relu(out, name=scope.name)

    #conv layer 5_1: 16 x 16 x 512
    with tf.variable_scope('conv5_1') as scope:
        stddev = tf.sqrt(2 / tf.to_float(3 * 3 * 512))
        kernel5_1 = tf.get_variable(name='weights', shape=[3, 3, 512, 512], initializer=tf.truncated_normal_initializer(stddev=stddev))
        conv = tf.nn.conv2d(conv4_2, kernel5_1, [1, 1, 1, 1], padding='SAME')
        biases5_1 = tf.get_variable(name='biases', shape=[512], initializer=tf.constant_initializer(0.0))
        out = tf.nn.bias_add(conv, biases5_1)
        conv5_1 = tf.nn.relu(out, name=scope.name)

    #conv layer 5_2: 16 x 16 x 512
    with tf.variable_scope('conv5_2') as scope:
        stddev = tf.sqrt(2 / tf.to_float(3 * 3 * 512))
        kernel5_2 = tf.get_variable(name='weights', shape=[3, 3, 512, 512], initializer=tf.truncated_normal_initializer(stddev=stddev))
        conv = tf.nn.conv2d(conv5_1, kernel5_2, [1, 1, 1, 1], padding='SAME')
        biases5_2 = tf.get_variable(name='biases', shape=[512], initializer=tf.constant_initializer(0.0))
        out = tf.nn.bias_add(conv, biases5_2)
        conv5_2 = tf.nn.relu(out, name=scope.name)

    #upsample layer 1: 16 x 16 x 512 --> 32 x 32 x 512
    with tf.variable_scope('upsample_1') as scope:
        upsample_1 = tf.image.resize_bilinear(conv5_2, [32,32])

    #conv layer 6_1: 32 x 32 x 256
    with tf.variable_scope('conv6_1') as scope:
        stddev = tf.sqrt(2 / tf.to_float(3 * 3 * 512))
        kernel6_1 = tf.get_variable(name='weights', shape=[3, 3, 512, 256], initializer=tf.truncated_normal_initializer(stddev=stddev))
        conv = tf.nn.conv2d(upsample_1, kernel6_1, [1, 1, 1, 1], padding='SAME')
        biases6_1 = tf.get_variable(name='biases', shape=[256], initializer=tf.constant_initializer(0.0))
        out = tf.nn.bias_add(conv, biases6_1)
        conv6_1 = tf.nn.relu(out, name=scope.name)

    #upsample layer 2: 32 x 32 x 256 --> 64 x 64 x 256
    with tf.variable_scope('upsample_2') as scope:
        upsample_2 = tf.image.resize_bilinear(conv6_1, [64,64])
   	
    #conv layer 7_1: 64 x 64 x 192
    with tf.variable_scope('conv7_1') as scope:
        stddev = tf.sqrt(2 / tf.to_float(3 * 3 * 256))
        kernel7_1 = tf.get_variable(name='weights', shape=[3, 3, 256, 192], initializer=tf.truncated_normal_initializer(stddev=stddev))
        conv = tf.nn.conv2d(upsample_2, kernel7_1, [1, 1, 1, 1], padding='SAME')
        biases7_1 = tf.get_variable(name='biases', shape=[192], initializer=tf.constant_initializer(0.0))
        out = tf.nn.bias_add(conv, biases7_1)
        conv7_1 = tf.nn.relu(out, name=scope.name)

    #upsample layer 3: 64 x 64 x 192 --> 128 x 128 x 192
    with tf.variable_scope('upsample_3') as scope:
        upsample_3 = tf.image.resize_bilinear(conv7_1, [128,128])

    #conv layer 8_1: 128 x 128 x 64
    with tf.variable_scope('conv8_1') as scope:
        stddev = tf.sqrt(2 / tf.to_float(3 * 3 * 192))
        kernel8_1 = tf.get_variable(name='weights', shape=[3, 3, 192, 64], initializer=tf.truncated_normal_initializer(stddev=stddev))
        conv = tf.nn.conv2d(upsample_3, kernel8_1, [1, 1, 1, 1], padding='SAME')
        biases8_1 = tf.get_variable(name='biases', shape=[64], initializer=tf.constant_initializer(0.0))
        out = tf.nn.bias_add(conv, biases8_1)
        conv8_1 = tf.nn.relu(out, name=scope.name)

        print("FINAL CONV: " + str(tf.shape(conv8_1))) 

    #fully connected layer 2: 1 x 1 x 64
    with tf.variable_scope('fc2') as scope:
        fc2 = layer.down_conv_relu_valid(conv8_1, [1,1], 64, flags['n_classes'])
        print("FINAL SHAPE: " + str(tf.shape(fc2)))

    #softmax layer: 1 x 1 x 64
    with tf.variable_scope('softmax') as scope:
        softmax = tf.nn.softmax(fc2)
    
    print ("exited inference in tf_model.py")
    return softmax, {'softmax' : softmax} #return pixel level predicted class probabilities


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

    print ("entered loss in tf_model.py")

    bg_mask = tf.to_float(tf.equal(labels, 0))  # whether the label = 0 or not
    epi_mask = tf.to_float(tf.equal(labels,1)) #whether the labels = 1 or not
    stroma_mask = tf.to_float(tf.equal(labels,2)) #whether the labels = 2 or not
    fat_mask = tf.to_float(tf.equal(labels,3))  #whether the labels = 3 or not

    #entries in the 4D boolean tensors are summed together to get total counts
    n_bg = tf.to_float(tf.reduce_sum(bg_mask, [0, 1, 2, 3]))
    n_epi = tf.to_float(tf.reduce_sum(epi_mask, [0, 1, 2, 3]))
    n_stroma = tf.to_float(tf.reduce_sum(stroma_mask, [0, 1, 2, 3]))
    n_fat = tf.to_float(tf.reduce_sum(fat_mask, [0, 1, 2, 3]))

    max_val = tf.reduce_max(tf.stack([n_epi,n_stroma,n_fat,n_bg]))	#find predicted class with highest probability for each pixel

    #apply empirically determined weights, bg = 0.5, epi = 0.7, stroma = 1.6, fat = 1.4
    bg_weights = tf.cond(tf.greater(n_bg, 0.0), lambda: (bg_mask / n_bg) * max_val * 0.5, lambda: bg_mask * max_val * 0.5)
    epi_weights = tf.cond(tf.greater(n_epi,0.0), lambda: (epi_mask/n_epi) * max_val * 0.7, lambda: epi_mask * max_val * 0.7)
    stroma_weights = tf.cond(tf.greater(n_stroma, 0.0), lambda: (stroma_mask / n_stroma) * max_val * 1.6, lambda: stroma_mask * max_val * 1.6)
    fat_weights = tf.cond(tf.greater(n_fat, 0.0), lambda: (fat_mask / n_fat) * max_val * 1.4, lambda: fat_mask * max_val * 1.4)

    #sum all weights and add the bias term (alpha * weights)
    class_weights = epi_weights + stroma_weights + fat_weights + bg_weights
    total_weights = class_weights + flags['alpha'] * weights

    #labels turned into one-hot format for softmax
    labels = tf.squeeze(labels)
    labels = tf.cast(labels, tf.int64)
    onehot = tf.one_hot(labels, depth=flags['n_classes'], on_value=1.0, off_value=0.0, axis=3)

    epsilon = 1e-6
    truncated_softmax = tf.clip_by_value(softmax, epsilon, 1.0 - epsilon) #forces all values to be in range [epsilon, 1-epsilon] to prevent issues with gradients

    #average cross_entropy_log_loss calculated using the truncated_softmax function
    cross_entropy_log_loss = -tf.reduce_sum(onehot * tf.log(truncated_softmax), axis=[3], keep_dims=True)
    cross_entropy_log_loss = (total_weights) * cross_entropy_log_loss #loss is weighted to combat class imbalance
    avg_cross_entropy_log_loss = tf.reduce_mean(cross_entropy_log_loss, axis=[0, 1, 2])

    print ("finished loss in tf_model.py")
    return {'truncated_softmax': truncated_softmax,
            'cross_entropy_log_loss': cross_entropy_log_loss,
            'avg_cross_entropy_log_loss': avg_cross_entropy_log_loss,
            'labels': labels,
            'onehot': onehot,
            'epi_mask': epi_mask,
            'stroma_mask': stroma_mask,
            'fat_mask':fat_mask,
            'bg_mask':bg_mask,
            'n_epi':n_epi,
            'n_stroma':n_stroma,
            'n_bg':n_bg,
            'n_fat':n_fat,
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

    print ("entered train in tf_model.py")
    optimizer = tf.train.AdamOptimizer(learning_rate=flags['initial_learning_rate'])
    grads_and_vars = optimizer.compute_gradients(total_loss)
    train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)
    print ("finished train in tf_model.py")
  
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

    print ("entered accuracy in tf_model.py")
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

    print ("finished accuracy in tf_model.py")
    return {'TP': TP,
            'FP': FP,
            'FN': FN,
            'TN': TN,
            'precision': precision,
            'recall': recall,
            'f1score': f1score}
