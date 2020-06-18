"""
flags_epi_stroma_segmentation.py

This file contains flags which specify the various settings
and hyperparameters used throughout the project. They include
file paths used for training data and testing results, as well
as network input patch size and training batch size.

"""

import os

flags = dict() #initialize flag dictionary


##########directories with all training data and labels##########
flags['annotation_folder'] = os.path.join('/home', 'avellal14','data', 'EpiStromaVGG','annotation_epi_stromal_segmentation')
flags['annotation_images_folder'] = os.path.join(flags['annotation_folder'],'images') #directory for training images
flags['annotation_groundtruths_folder'] = os.path.join(flags['annotation_folder'],'groundtruths') #directory for groundtruth labels
flags['annotation_groups_folder'] = os.path.join(flags['annotation_folder'],'groups') #directory for training data (different types and proliferative categories of BBD)
flags['annotation_weights_folder'] = os.path.join(flags['annotation_folder'],'weights') #directory for pixelwise weights for each training patch (weighted by tissue type)

#dictionary of all directories
flags['dict_path'] = {'image':flags['annotation_images_folder'],
                      'groundtruth': flags['annotation_groundtruths_folder'],
                      'group': flags['annotation_groups_folder'],
                      'weight': flags['annotation_weights_folder']}


##########directory with all testing/validation data##########
flags['experiment_folder'] = os.path.join('/home', 'avellal14','data','EpiStromaVGG','experiment_epi_stromal_segmentation')


##########file extensions##########
flags['image_ext'] = '.jpg'
flags['groundtruth_ext'] = '.png'
flags['group_ext'] = '.csv'
flags['weight_ext'] = '.png'

#dictionary of all extensions
flags['dict_ext'] = {'image':flags['image_ext'],
                     'groundtruth':flags['groundtruth_ext'],
                     'group':flags['group_ext'],
                     'weight':flags['weight_ext']}


##########hyperparameter for library computer vision functions##########
sigma = 5.0


##########data training/validation split method and proportion##########
flags['split_method'] = 'perm' #randomly permutes a training/validation split while balancing the different groups and subcategories
flags['num_split'] = 1 #only do one split since we are not using a cross validation method
flags['test_percentage'] = 0.0 #no test data
flags['val_percentage'] = 25.0 #25% validation data
flags['instance_proportion'] = {0: 0.5, 1: 0.5} #TODO: could be removed if routine.py works properly without it


##########patch generation and data augmentation settings##########

#sliding window of size 128x128 and stride 40 pixels to generate training patches
flags['gen_train_val_method'] = 'sliding_window'
flags['size_input_patch'] = [128,128,3]
flags['size_output_patch'] = [128,128,1]
flags['stride'] = [40,40]

flags['augmentation_keyword'] = 'mod' #keyword used to designate files containing augmented data


##########training process settings##########
flags['num_preprocess_threads'] = 8
flags['gpu'] = '/gpu:0'
flags['min_fraction_of_examples_in_queue'] = 0.01
flags['gpu_memory_fraction'] = 1.0

flags['num_epochs'] = 300 #number of iterations over which the network is trained
flags['num_examples_per_epoch_for_train'] = 10000
flags['num_examples_per_epoch_for_val'] = int(flags['num_examples_per_epoch_for_train']*0.25) #number of training examples processed in each epoch

flags['n_classes'] = 4 #four classes: epithelial, stromal, fat, and background
flags['alpha'] = 1.0 #coefficient for bias term added to sum of pixel based weights

flags['batch_size'] = 8
flags['initial_learning_rate'] = 1e-4


##########testing settings##########
flags['test_image_list'] = os.path.join(flags['experiment_folder'],'perm1','val_image_list.csv')
flags['use_patches'] = True #true for patch based detection, false for whole image based detection with dilated convolutional networks
flags['test_model'] = 0 #use model trained for 300 epochs
flags['stride_test'] = [120,120] #patch size is same as during training, 128 x 128
flags['test_batch_size'] = 8