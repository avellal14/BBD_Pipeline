"""
tf_model_test.py

This file contains all of the functions involved with segmenting
test images using the network.
"""

import time
import numpy as np
import tensorflow as tf
import os
import cv2
import glob

from KS_lib.tf_model_epi_stroma_segmentation import tf_model_input_test
from KS_lib.tf_model_epi_stroma_segmentation import tf_model
from KS_lib.general import matlab
from itertools import izip
from KS_lib.image import KSimage

###################################################
def whole_image_processing(filename, sess, logits_test, parameters, images_test, keep_prob, mean_image, variance_image, flags):
    """
    whole_image_processing reads in a whole image, preprocesses it, passes it
    through the network, and returns the results

    param: filename
    param: sess
    param: logits_test
    param: parameters
    param: images_test
    param: keep_prob
    param: mean_image
    param: variance_image
    return: pred
    """

    print("image read beginning")
    image, image_mask, image_size = tf_model_input_test.read_whole_image_test(filename,flags) #should be n x n x 3

    mask1 = np.argmin(image, axis=2)
    mask3 = image[:, :, 0]  # r value
    mask4 = image[:, :, 2]  # b value
    mask1 = mask1 == 1  # want to make sure green is always the minimum in RGB
    mask3 = mask3 > 100  # force R > 100
    mask4 = mask4 > 100  # force B > 100
    mask = mask1 & mask3 & mask4
    mask = np.uint8(mask)  # turn into 1s and 0s
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))  # kernel to get rid of all the crap

    result1 = cv2.bitwise_and(image, image, mask=mask)
    result2 = cv2.morphologyEx(result1, cv2.MORPH_CLOSE, kernel)
    result2 = result2[:,:,0] #bold test, hope it works!!!!!!!
    result2 = tf.squeeze(result2)
    result2 = np.asarray(result2.eval())

    print("image read done")

    epsilon = 1e-6

    print("normalization + dim expand beginning")
    image = image - mean_image
    image = image / np.sqrt(variance_image + epsilon)
    image = np.expand_dims(image,axis=0)

    print("data going into network")
    print("IMAGE SHAPE: " + str(image.shape))
    pred, paras = sess.run([logits_test, parameters], feed_dict={images_test: image, keep_prob: 1.0})
    print("data coming out of network")

    print("result squeezed")
    pred = tf.squeeze(pred)
    pred = np.asarray(pred.eval())

    print("Success with BG!(hopefully)")
    pred[result2==0] = 0 #set a bunch of shit to bg

    pred = pred * 255.0
    pred = np.argmax(pred, axis=2)

    print("PRED SHAPE(at 10x): " + str(pred.shape))
    print("RESULT SHAPE: " + str(result2.shape))

    return pred

################################################################
def batch_processing(filename, sess, logits_test, parameters, images_test, keep_prob, mean_image, variance_image, flags):
    """
    batch_processing reads in an image, splits it up into patches,
    preprocesses the patches, passes them through the network, and returns
    the results

    param: filename
    param: sess
    param: logits_test
    param: parameters
    param: images_test
    param: keep_prob
    param: mean_image
    param: variance_image
    param: test_batch_size
    param: size_output_patch
    return: result
    """

    # Read image and extract patches
    patches, patches_mask, image_size, nPatches = tf_model_input_test.read_data_test(filename, flags)

    def batches(generator, size):
        source = generator
        while True:
            chunk = [val for _, val in izip(xrange(size), source)]
            if not chunk:
                raise StopIteration
            yield chunk

    # Construct batch indices
    batch_index = range(0, nPatches, flags['test_batch_size'])
    if nPatches not in batch_index:
        batch_index.append(nPatches)

    # Process all_patches
    shape = np.hstack([nPatches, flags['size_output_patch']])
    shape[-1] = logits_test.get_shape()[3].value
    all_patches = np.zeros(shape, dtype=np.float32)

    for ipatch, chunk in enumerate(zip(batches(patches, flags['test_batch_size']),
                                       batches(patches_mask, flags['test_batch_size']))):
        start_time = time.time()
        start_idx = batch_index[ipatch]
        end_idx = batch_index[ipatch + 1]

        tmp = list()
        for i in range(len(chunk[1])):
            tmp.append(np.sum(chunk[1][i]==255.0)/float(chunk[1][i].size))

        # process batch if any patch within it has >=50% uncovered by mask --> make sure to understand. white = uncovered by mask
        if np.any(np.array(tmp) > 0.5):
            temp = tf_model_input_test.inputs_test(chunk[0], mean_image, variance_image)

            if temp.shape[0] < flags['test_batch_size']:
                rep = np.tile(temp[-1, :, :, :], [flags['test_batch_size'] - temp.shape[0], 1, 1, 1])
                temp = np.vstack([temp, rep])

            pred, paras = sess.run([logits_test, parameters], feed_dict={images_test: temp, keep_prob: 1.0})

        else:
            shape = np.hstack([flags['test_batch_size'], flags['size_output_patch']])
            shape[-1] = logits_test.get_shape()[3].value
            pred = np.zeros(shape, dtype=np.float32)
            for j in range(flags['test_batch_size']):
                x = pred[j,:,:,:]
                x[:,:,0] = 1.0
                pred[j,:,:,:] = x

        all_patches[start_idx:end_idx, :, :, :] = pred[range(end_idx - start_idx), :, :, :]

        duration = time.time() - start_time
        print('processing step %d/%d (%.2f sec/step)' % (ipatch + 1, len(batch_index) - 1, duration))

    #this is where all the patches are combined. the issue is --> I NEED A CERTAINTY FOR EVERY INDIVIDUAL PATCH
    result = tf_model_input_test.MergePatches_test(all_patches, flags['stride_test'], image_size, flags['size_input_patch'], flags['size_output_patch'], flags)

    result = tf.squeeze(result)
    result = np.asarray(result.eval())

    result = result * 255.0
    result = result.astype(np.uint8) 
    result = np.argmax(result, axis=2) #TODO: Changed from argmax to max
    result = KSimage.imresize(result, 4.0)
 
    return result 

##########################################################
def create_dir(dir_name):
    """
    create a directory if not exist.
    param: dir_name
    return: none
    """
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

#######################################################################################################
def test(object_folder, model_path, filename_list, flags):
    """
    test uses either whole image segmentation or patch based
    segmentation to segment an entire directory of test images

    param: object_folder
    param: model_path
    param: filename_list
    param: gpu
    param: gpu_memory_fraction
    return: writes segmentation result to appropriate image file
    """

    checkpoint_dir = os.path.join(object_folder, 'checkpoint')
    mat_contents = matlab.load(os.path.join(checkpoint_dir, 'network_stats.mat'))
    mean_image = np.float32(mat_contents['mean_image'])
    variance_image = np.float32(mat_contents['variance_image'])

    # turns 256 x 256 x 3 into 1 x 1 x 3
    mean_image_new = np.array([mean_image[:, :, 0].mean(), mean_image[:, :, 1].mean(), mean_image[:, :, 2].mean()])
    variance_image_new = np.array([variance_image[:, :, 0].mean(), variance_image[:, :, 1].mean(), variance_image[:, :, 2].mean()])

    with tf.Graph().as_default(), tf.device(flags['gpu']):
        keep_prob = tf.placeholder(tf.float32)

        # Place holder for patches
        images_test = tf.placeholder(tf.float32)

        # Network
        with tf.variable_scope("network") as scope:
            logits_test, parameters = tf_model.inference(images_test, keep_prob, flags)

        # Saver and initialisation
        saver = tf.train.Saver()
        init = tf.global_variables_initializer()

        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=flags['gpu_memory_fraction'])
        config = tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options)

        with tf.Session(config=config) as sess:
            # Initialise and load variables
            sess.run(init)
            saver.restore(sess, model_path)

            result_dir = os.path.join(object_folder, 'result')
            create_dir(result_dir)

            start_time = time.time()
            for iImage, file in enumerate(filename_list):
                file = file[0]
                basename = os.path.basename(file)
                basename = os.path.splitext(basename)[0]
                savename = os.path.join(result_dir, basename + '.png')

                
        print('processing image %d/%d' % (iImage + 1, len(filename_list)))
        print("FILE!!!!!!!!!!!" + str(file))

        if (flags['use_patches'] == False):
            result = whole_image_processing(file, sess, logits_test, parameters, images_test, keep_prob, mean_image_new, variance_image_new, flags)
        else:
            result = batch_processing(file, sess, logits_test, parameters, images_test, keep_prob mean_image, variance_image, flags)

        print("Image processed")
        KSimage.imwrite(result, savename) #Write result back to image once segmentation is fixed
        print("TOTAL DURATION : " + str(time.time() - start_time))


def testWSI(object_folder, model_path, directory, flags):
    """
    testWSI segments all of the WSIs in a given directory

    param: object_folder
    param: model_path
    param: directory
    param: gpu
    param: gpu_memory_fraction
    param: test_batch_size
    param: size_input_patch
    return: writes segmentation result to corresponding segmentation result directory
    """

    checkpoint_dir = os.path.join(object_folder, 'checkpoint')
    mat_contents = matlab.load(os.path.join(checkpoint_dir, 'network_stats.mat'))
    mean_image = np.float32(mat_contents['mean_image'])
    variance_image = np.float32(mat_contents['variance_image'])
    startTime = time.time()
    with tf.Graph().as_default(), tf.device(flags['gpu']):
        keep_prob = tf.placeholder(tf.float32)

        # Place holder for patches
        images_test = tf.placeholder(tf.float32, shape=(np.hstack([flags['test_batch_size'], flags['size_input_patch']])))

        # Network
        with tf.variable_scope("network") as scope:
            logits_test, parameters = tf_model.inference(images_test, keep_prob, flags)

        # Saver and initialisation
        saver = tf.train.Saver()
        init = tf.global_variables_initializer()

        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=flags['gpu_memory_fraction'])
        config = tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options)

        with tf.Session(config = config) as sess:
            # Initialise and load variables
            sess.run(init)
            saver.restore(sess, model_path)

            print("Current directory: " + str(directory))
            result_dir = os.path.join(directory +'_epiStromalSeg')
            create_dir(result_dir)
            filename_list = glob.glob(os.path.join(directory, '*.png')) #the main statement, returns all the files in the directory
            print("Num Files: " + str(len(filename_list)))
            for file in filename_list:
                print(file)
                basename = os.path.basename(file)
                basename = os.path.splitext(basename)[0]
                savename = os.path.join(result_dir, basename + '.png')
                if not os.path.exists(savename) and not ('mask' in file or 'thumbnail' in file):
                    result = batch_processing(file, sess, logits_test, parameters, images_test, keep_prob, mean_image, variance_image, flags)
                    KSimage.imwrite(result, savename)

            print("Total Time: " + str(time.time() - startTime))

#############################################################################################
def get_immediate_subdirectories(a_dir):
    """
    get_immediate_subdirectories returns all of the subdirectories
    in a given directory

    param: a_dir
    return: subdirectory list
    """

    return [name for name in os.listdir(a_dir)
            if os.path.isdir(os.path.join(a_dir, name))]
