"""
tf_model_train.py

This file contains the code which defines the computational graph of the
segmentation network and executes the training loop.
"""

from datetime import datetime
import os.path
import time
import copy

import numpy as np
import scipy.io as sio
import tensorflow as tf
import matplotlib.pyplot as plt

from KS_lib.tf_model_he_cell_segmentation import tf_model_input
from KS_lib.tf_model_he_cell_segmentation import tf_model
from KS_lib.prepare_data import routine
from KS_lib.general import matlab

########################################################################################################################
def define_graph(object_folder,checkpoint_folder, flags):
    """
    define_graph defines the computational graph in tensorflow
    which represents the entire network

    param: object_folder
    param: checkpoint_folder
    param: flags
    return: train operation dictionary
    """

    # global step
    global_step = tf.Variable(0, trainable=False, name='global_step')

    # Epoch counter
    curr_epoch = tf.Variable(0, trainable=False, name='curr_epoch')
    update_curr_epoch = tf.assign(curr_epoch, tf.add(curr_epoch, tf.constant(1)))

    # drop out
    keep_prob = tf.placeholder(tf.float32)


    # Load network stats
    mat_contents = matlab.load(os.path.join(checkpoint_folder, 'network_stats.mat'))
    mean_img = np.float32(mat_contents['mean_image'])
    variance_img = np.float32(mat_contents['variance_image'])

    if mean_img.ndim == 2:
        mean_img = np.expand_dims(mean_img, axis=2)
    if variance_img.ndim == 2:
        variance_img = np.expand_dims(variance_img, axis=2)

    mean_image = tf.Variable(mean_img, trainable=False, name='mean_image')
    variance_image = tf.Variable(variance_img, trainable=False, name='variance_image')

    # Get images and labels.
    out_content_train = tf_model_input.inputs(mean_image, variance_image, object_folder, 'train', flags)
    out_content_val = tf_model_input.inputs(mean_image, variance_image, object_folder, 'val', flags)

    images_train = out_content_train['images']
    labels_train = out_content_train['labels']
    weights_train = out_content_train['weights']

    images_val = out_content_val['images']
    labels_val = out_content_val['labels']
    weights_val = out_content_val['weights']

    # Build a Graph that computes the logits predictions from the inference model.
    with tf.variable_scope("network") as scope:
        sigmoid_all_train, parameters = tf_model.inference(images_train, keep_prob, flags)
        scope.reuse_variables()
        sigmoid_all_val, _ = tf_model.inference(images_val, keep_prob, flags)

    # Calculate loss
    loss_train = tf_model.loss(sigmoid_all_train, labels_train, weights_train, curr_epoch, flags)
    loss_val = tf_model.loss(sigmoid_all_val, labels_val, weights_val, curr_epoch, flags)

    # Accuracy train
    predict_train = tf.squeeze(tf.argmax(sigmoid_all_train, dimension=3))
    actual_train = tf.squeeze(labels_train)
    accuracy_train_output = tf_model.accuracy(predict_train, actual_train, flags)

    # Accuracy val
    predict_val = tf.squeeze(tf.argmax(sigmoid_all_val, dimension=3))
    actual_val = tf.squeeze(labels_val)
    accuracy_val_output = tf_model.accuracy(predict_val, actual_val, flags)

    # Build a Graph that trains the model with one batch of examples and updates the model parameters.
    train_op = tf_model.train(loss_train['avg_cross_entropy_log_loss'], global_step, parameters, flags)

    return {'global_step':global_step,
            'curr_epoch':curr_epoch,
            'update_curr_epoch':update_curr_epoch,
            'keep_prob':keep_prob,
            'loss_train':loss_train,
            'loss_val':loss_val,
            'predict_train':predict_train,
            'actual_train':actual_train,
            'predict_val':predict_val,
            'actual_val':actual_val,
            'train_op':train_op,
            'accuracy_train_output':accuracy_train_output,
            'accuracy_val_output':accuracy_val_output,
            'parameters':parameters,
            'out_content_train':out_content_train,
            'out_content_val':out_content_val,
            'sigmoid_all_train':sigmoid_all_train,
            'sigmoid_all_val':sigmoid_all_val
            # 'random_field': random_field
            }
########################################################################################################################
def load_checkpoint(sess, saver, curr_epoch, checkpoint_folder, parameters, flags):
    """
    load_checkpoint loads a saved version of the network to
    continue training

    param: sess
    param: saver
    param: curr_epoch
    param: checkpoint_folder
    param: parameters
    param: pretrain_path
    return: loss, precision, recall, and F1 score dictionary
    """

    ckpt = tf.train.get_checkpoint_state(checkpoint_folder)
    if ckpt and ckpt.model_checkpoint_path:
        # Restores from checkpoint
        saver.restore(sess, ckpt.model_checkpoint_path)

        if os.path.isfile(os.path.join(checkpoint_folder, 'variables.mat')):
            mat_contents = sio.loadmat(os.path.join(checkpoint_folder, 'variables.mat'))

            # loss
            all_avg_train_loss = mat_contents['all_avg_train_loss']
            all_avg_train_loss = all_avg_train_loss[:, 0:sess.run(curr_epoch)]

            all_avg_validation_loss = mat_contents['all_avg_validation_loss']
            all_avg_validation_loss = all_avg_validation_loss[:, 0:sess.run(curr_epoch)]

            # precision
            all_avg_train_precision = mat_contents['all_avg_train_precision']
            all_avg_train_precision = all_avg_train_precision[0:sess.run(curr_epoch)]

            all_avg_validation_precision = mat_contents['all_avg_validation_precision']
            all_avg_validation_precision = all_avg_validation_precision[0:sess.run(curr_epoch)]

            # recall
            all_avg_train_recall = mat_contents['all_avg_train_recall']
            all_avg_train_recall = all_avg_train_recall[0:sess.run(curr_epoch)]

            all_avg_validation_recall = mat_contents['all_avg_validation_recall']
            all_avg_validation_recall = all_avg_validation_recall[0:sess.run(curr_epoch)]

            # f1score
            all_avg_train_f1score = mat_contents['all_avg_train_f1score']
            all_avg_train_f1score = all_avg_train_f1score[0:sess.run(curr_epoch)]

            all_avg_validation_f1score = mat_contents['all_avg_validation_f1score']
            all_avg_validation_f1score = all_avg_validation_f1score[0:sess.run(curr_epoch)]
        else:
            all_avg_train_loss = list()
            all_avg_validation_loss = list()
            all_avg_train_precision = list()
            all_avg_validation_precision = list()
            all_avg_train_recall = list()
            all_avg_validation_recall = list()
            all_avg_train_f1score = list()
            all_avg_validation_f1score = list()

    else:
        print('No checkpoint file found')
        all_avg_train_loss = list()
        all_avg_validation_loss = list()
        all_avg_train_precision = list()
        all_avg_validation_precision = list()
        all_avg_train_recall = list()
        all_avg_validation_recall = list()
        all_avg_train_f1score = list()
        all_avg_validation_f1score = list()

        # load pretrained model
        if flags['pretrain_path'] and os.path.isfile(flags['pretrain_path']):
            tf_model.load_pretrain_model(sess, parameters, flags['pretrain_path'])

            checkpoint_path = os.path.join(checkpoint_folder, 'model_pretrain.ckpt')
            saver.save(sess, checkpoint_path)

    return {'all_avg_train_loss':all_avg_train_loss,
            'all_avg_validation_loss':all_avg_validation_loss,
            'all_avg_train_precision':all_avg_train_precision,
            'all_avg_validation_precision':all_avg_validation_precision,
            'all_avg_train_recall':all_avg_train_recall,
            'all_avg_validation_recall':all_avg_validation_recall,
            'all_avg_train_f1score':all_avg_train_f1score,
            'all_avg_validation_f1score':all_avg_validation_f1score}

########################################################################################################################
def update_training_validation_variables(train_val_variables, checkpoint_output, nTrainBatches, nValBatches, epoch,flags):
    """
    update_training_validation_variables is used to iteratively update the network's weights

    param: train_val_variables
    param: checkpoint_output
    param: nTrainBatches
    param: nValBatches
    param: epoch
    param: n_classes
    return: checkpoint_output
    """

    avg_train_loss_per_epoch = np.mean(np.asarray(train_val_variables['avg_train_loss']))
    avg_train_precision_per_epoch = [0]*flags['n_classes']
    avg_train_recall_per_epoch = [0]*flags['n_classes']
    avg_train_f1score_per_epoch = [0]*flags['n_classes']

    for iclass in range(flags['n_classes']):
        avg_train_precision_per_epoch[iclass] = np.mean(np.asarray(train_val_variables['avg_train_precision'][iclass]))
        avg_train_recall_per_epoch[iclass] = np.mean(np.asarray(train_val_variables['avg_train_recall'][iclass]))
        avg_train_f1score_per_epoch[iclass] = np.mean(np.asarray(train_val_variables['avg_train_f1score'][iclass]))

    avg_validation_loss_per_epoch = np.mean(np.asarray(train_val_variables['avg_val_loss']))
    avg_validation_precision_per_epoch = [0]*flags['n_classes']
    avg_validation_recall_per_epoch = [0]*flags['n_classes']
    avg_validation_f1score_per_epoch = [0]*flags['n_classes']
    for iclass in range(flags['n_classes']):
        avg_validation_precision_per_epoch[iclass] = np.mean(np.asarray(train_val_variables['avg_val_precision'][iclass]))
        avg_validation_recall_per_epoch[iclass] = np.mean(np.asarray(train_val_variables['avg_val_recall'][iclass]))
        avg_validation_f1score_per_epoch[iclass] = np.mean(np.asarray(train_val_variables['avg_val_f1score'][iclass]))

    all_avg_train_loss = checkpoint_output['all_avg_train_loss']
    all_avg_validation_loss = checkpoint_output['all_avg_validation_loss']
    all_avg_train_precision = checkpoint_output['all_avg_train_precision']
    all_avg_validation_precision = checkpoint_output['all_avg_validation_precision']
    all_avg_train_recall = checkpoint_output['all_avg_train_recall']
    all_avg_validation_recall = checkpoint_output['all_avg_validation_recall']
    all_avg_train_f1score = checkpoint_output['all_avg_train_f1score']
    all_avg_validation_f1score = checkpoint_output['all_avg_validation_f1score']

    if epoch == 0:
        all_avg_train_loss.append(avg_train_loss_per_epoch)
        all_avg_train_precision.append(avg_train_precision_per_epoch)
        all_avg_train_recall.append(avg_train_recall_per_epoch)
        all_avg_train_f1score.append(avg_train_f1score_per_epoch)
    else:
        all_avg_train_loss = np.append(all_avg_train_loss,avg_train_loss_per_epoch)
        all_avg_train_precision = np.append(all_avg_train_precision,
                                            np.expand_dims(avg_train_precision_per_epoch, axis = 0),axis=0)
        all_avg_train_recall = np.append(all_avg_train_recall,
                                         np.expand_dims(avg_train_recall_per_epoch, axis = 0),axis=0)
        all_avg_train_f1score = np.append(all_avg_train_f1score,
                                          np.expand_dims(avg_train_f1score_per_epoch, axis = 0),axis=0)


    if epoch == 0:
        all_avg_validation_loss.append(avg_validation_loss_per_epoch)
        all_avg_validation_precision.append(avg_validation_precision_per_epoch)
        all_avg_validation_recall.append(avg_validation_recall_per_epoch)
        all_avg_validation_f1score.append(avg_validation_f1score_per_epoch)
    else:
        all_avg_validation_loss = np.append(all_avg_validation_loss, avg_validation_loss_per_epoch)
        all_avg_validation_precision = np.append(all_avg_validation_precision,
                                                 np.expand_dims(avg_validation_precision_per_epoch, axis = 0), axis=0)
        all_avg_validation_recall = np.append(all_avg_validation_recall,
                                              np.expand_dims(avg_validation_recall_per_epoch, axis = 0),axis=0)
        all_avg_validation_f1score = np.append(all_avg_validation_f1score,
                                               np.expand_dims(avg_validation_f1score_per_epoch, axis = 0), axis=0)

    checkpoint_output['all_avg_train_loss'] = all_avg_train_loss
    checkpoint_output['all_avg_validation_loss'] = all_avg_validation_loss
    checkpoint_output['all_avg_train_precision'] = all_avg_train_precision
    checkpoint_output['all_avg_validation_precision'] = all_avg_validation_precision
    checkpoint_output['all_avg_train_recall'] = all_avg_train_recall
    checkpoint_output['all_avg_validation_recall'] = all_avg_validation_recall
    checkpoint_output['all_avg_train_f1score'] = all_avg_train_f1score
    checkpoint_output['all_avg_validation_f1score'] = all_avg_validation_f1score

    return checkpoint_output

########################################################################################################################
def training_loop(sess, define_graph_output, train_val_variables, nTrainBatches, epoch, checkpoint_folder, flags):
    """
    training_loop contains the definition for the loop where the weights
    are iteratively modified to optimize the loss function for the training set

    param: sess
    param: define_graph_output
    param: train_val_variables
    param: nTrainBatches
    param: epoch
    param: checkpoint_folder
    param: n_classes
    return: none
    """

    for step in xrange(nTrainBatches):
        start_time = time.time()
        # run session
        _, loss_value_train, precision, recall, f1score, TP, FP, FN, TN,\
            out_train, pred_train = \
            sess.run([define_graph_output['train_op'],
                      define_graph_output['loss_train']['avg_cross_entropy_log_loss'],
                      define_graph_output['accuracy_train_output']['precision'],
                      define_graph_output['accuracy_train_output']['recall'],
                      define_graph_output['accuracy_train_output']['f1score'],
                      define_graph_output['accuracy_train_output']['TP'],
                      define_graph_output['accuracy_train_output']['FP'],
                      define_graph_output['accuracy_train_output']['FN'],
                      define_graph_output['accuracy_train_output']['TN'],
                      define_graph_output['out_content_train'],
                      define_graph_output['sigmoid_all_train']
                      ],
                     feed_dict={define_graph_output['keep_prob']: 0.5})

        duration = time.time() - start_time
        assert not np.isnan(loss_value_train), 'Model diverged with loss = NaN'

        if step % 100 == 0:
            matlab.save(os.path.join(checkpoint_folder, 'train_content.mat'),
                        {'out_train':out_train,'pred_train':pred_train})

        # evaluate
        if not np.isnan(loss_value_train):
            train_val_variables['avg_train_loss'].append(loss_value_train)
        for iclass in range(flags['n_classes']):
            if not np.isnan(precision[iclass]):
                train_val_variables['avg_train_precision'][iclass].append(precision[iclass])
            if not np.isnan(recall[iclass]):
                train_val_variables['avg_train_recall'][iclass].append(recall[iclass])
            if not np.isnan(f1score[iclass]):
                train_val_variables['avg_train_f1score'][iclass].append(f1score[iclass])

        # print
        format_str = ('%s: epoch %d, step %d/ %d (%.2f sec/step)')
        print(format_str % (datetime.now(), epoch, step + 1, nTrainBatches, duration))
        format_str = ('Training Loss = %.2f, Precision = %.2f, Recall = %.2f, F1 = %.2f, ' +
                      'TP = %.2f, FP = %.2f, FN = %.2f, TN = %.2f')
        for iclass in range(flags['n_classes']):
            print(format_str % (loss_value_train, precision[iclass], recall[iclass],
                                f1score[iclass], TP[iclass], FP[iclass], FN[iclass], TN[iclass]))

########################################################################################################################
def validation_loop(sess,define_graph_output,train_val_variables,nValBatches,epoch,checkpoint_folder,flags):
    """
    validation_loop contains the definition for the loop where the weights
    are iteratively modified to optimize the loss function for the validation set

    param: sess
    param: define_graph_output
    param: train_val_variables
    param: nTrainBatches
    param: epoch
    param: checkpoint_folder
    param: n_classes
    return: none
    """

    for step in xrange(nValBatches):
        start_time = time.time()

        # run session
        loss_value_val, precision, recall, f1score, TP, FP, FN, TN,\
            out_val, pred_val = \
            sess.run([define_graph_output['loss_val']['avg_cross_entropy_log_loss'],
                      define_graph_output['accuracy_val_output']['precision'],
                      define_graph_output['accuracy_val_output']['recall'],
                      define_graph_output['accuracy_val_output']['f1score'],
                      define_graph_output['accuracy_val_output']['TP'],
                      define_graph_output['accuracy_val_output']['FP'],
                      define_graph_output['accuracy_val_output']['FN'],
                      define_graph_output['accuracy_val_output']['TN'],
                      define_graph_output['out_content_val'],
                      define_graph_output['sigmoid_all_val']
                      ],
                     feed_dict={define_graph_output['keep_prob']: 1.0})

        duration = time.time() - start_time
        assert not np.isnan(loss_value_val), 'Model diverged with loss = NaN'

        if step % 100 == 0:
            matlab.save(os.path.join(checkpoint_folder, 'val_content.mat'),
                        {'out_val':out_val,'pred_val':pred_val})

        # evaluate
        if not np.isnan(loss_value_val):
            train_val_variables['avg_val_loss'].append(loss_value_val)
        for iclass in range(flags['n_classes']):
            if not np.isnan(precision[iclass]):
                train_val_variables['avg_val_precision'][iclass].append(precision[iclass])
            if not np.isnan(recall[iclass]):
                train_val_variables['avg_val_recall'][iclass].append(recall[iclass])
            if not np.isnan(f1score[iclass]):
                train_val_variables['avg_val_f1score'][iclass].append(f1score[iclass])

        # print
        format_str = ('%s: epoch %d, step %d/ %d (%.2f sec/step)')
        print(format_str % (datetime.now(), epoch, step + 1, nValBatches, duration))
        format_str = ('Validation Loss = %.2f, Precision = %.2f, Recall = %.2f, F1 = %.2f, ' +
                      'TP = %.2f, FP = %.2f, FN = %.2f, TN = %.2f')
        for iclass in range(flags['n_classes']):
            print(format_str % (loss_value_val, precision[iclass], recall[iclass],
                            f1score[iclass], TP[iclass], FP[iclass], FN[iclass], TN[iclass]))

########################################################################################################################
def save_model(sess, saver, define_graph_output, checkpoint_folder, checkpoint_output):
    """
    save_model saves all of the current weights into a checkpoint file

    param: sess
    param: saver
    param: define_graph_output
    param: checkpoint_folder
    param: checkpoint_output
    return: none
    """

    sess.run(define_graph_output['update_curr_epoch'])

    checkpoint_path = os.path.join(checkpoint_folder, 'model.ckpt')
    saver.save(sess, checkpoint_path, global_step=define_graph_output['global_step'])

    sio.savemat(os.path.join(checkpoint_folder, 'variables.mat'),
                {'all_avg_train_loss': checkpoint_output['all_avg_train_loss'],
                 'all_avg_train_precision': checkpoint_output['all_avg_train_precision'],
                 'all_avg_train_recall': checkpoint_output['all_avg_train_recall'],
                 'all_avg_train_f1score': checkpoint_output['all_avg_train_f1score'],
                 'all_avg_validation_loss': checkpoint_output['all_avg_validation_loss'],
                 'all_avg_validation_precision': checkpoint_output['all_avg_validation_precision'],
                 'all_avg_validation_recall': checkpoint_output['all_avg_validation_recall'],
                 'all_avg_validation_f1score': checkpoint_output['all_avg_validation_f1score']
                })

########################################################################################################################
def train(object_folder, flags):
    """
    train defines the graph, then runs the training loop and validation loop and
    saves the model

    param: object_folder
    param: gpu
    param: gpu_memory_fraction
    param: num_examples_per_epoch_for_train
    param: num_examples_per_epoch_for_val
    param: batch_size
    param: num_epochs
    param: n_classes
    return: none
    """

    checkpoint_folder = os.path.join(object_folder, 'checkpoint')
    routine.create_dir(checkpoint_folder)

    with tf.Graph().as_default(), tf.device(flags['gpu']):
        # define a graph.
        define_graph_output = define_graph(object_folder, checkpoint_folder, flags)

        # Create a saver.
        saver = tf.train.Saver(max_to_keep=0)
        # saver = tf.train.Saver(tf.global_variables(), max_to_keep=0)

        # Build an initialization operation to run below.
        # init = tf.initialize_all_variables()
        init = tf.global_variables_initializer()

        # Start running operations on the Graph.
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=flags['gpu_memory_fraction'])
        config = tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options)

        with tf.Session(config=config) as sess:

            # Start the queue runners
            sess.run(init)
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            # load checkpoint
            checkpoint_output = load_checkpoint(sess, saver, define_graph_output['curr_epoch'], checkpoint_folder,
                                                define_graph_output['parameters'], flags)

            # epoch
            num_examples_per_epoch_for_train = flags['num_examples_per_epoch_for_train']
            num_examples_per_epoch_for_val = flags['num_examples_per_epoch_for_val']

            nTrainBatches = int((num_examples_per_epoch_for_train / float(flags['batch_size'])) + 1)
            nValBatches = int((num_examples_per_epoch_for_val / float(flags['batch_size'])) + 1)

            for epoch in xrange(sess.run(define_graph_output['curr_epoch']), flags['num_epochs'] + 1):
                train_val_variables = {'avg_train_loss':[],
                                       'avg_train_precision':[[] for iclass in xrange(flags['n_classes'])],
                                       'avg_train_recall':[[] for iclass in xrange(flags['n_classes'])],
                                       'avg_train_f1score':[[] for iclass in xrange(flags['n_classes'])],
                                       'avg_val_loss':[],
                                       'avg_val_precision':[[] for iclass in xrange(flags['n_classes'])],
                                       'avg_val_recall':[[] for iclass in xrange(flags['n_classes'])],
                                       'avg_val_f1score':[[] for iclass in xrange(flags['n_classes'])]}

                # Training loop
                training_loop(sess, define_graph_output, train_val_variables,
                              nTrainBatches, epoch, checkpoint_folder,flags)

                # Validation loop
                validation_loop(sess, define_graph_output, train_val_variables,
                                nValBatches, epoch, checkpoint_folder,flags)

                # Average loss on training and validation
                checkpoint_output = update_training_validation_variables(train_val_variables, checkpoint_output,
                                                                         nTrainBatches, nValBatches, epoch,flags)

                # Save the model after each epoch.
                save_model(sess, saver, define_graph_output, checkpoint_folder, checkpoint_output)

            coord.request_stop()
            coord.join(threads)
            plt.close()
