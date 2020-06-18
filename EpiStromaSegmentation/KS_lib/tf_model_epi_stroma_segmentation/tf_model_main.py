"""
tf_model_main.py

This file contains the code for the high level execution of the network,
for functions including training, testing, and segmenting WSIs.
"""

import glob
import os
import csv

from KS_lib.tf_model_epi_stroma_segmentation import tf_model_test
from KS_lib.tf_model_epi_stroma_segmentation import tf_model_train
from KS_lib.tf_model_epi_stroma_segmentation import utils

from KS_lib.prepare_data import routine
from KS_lib.general import matlab
from KS_lib.general import KScsv
import numpy as np

###################################################
def main(nth_fold,mode,flags,testdir):
    """
    main trains, tests, or executes the model on the provided
    data based on the specified preferences

    param: nth_fold
    param: mode
    param: experiment_folder
    param: image_ext
    param: test_model
    param: test_image_list
    return: saves segmentation results to appropriate file/directory
    """

    # check if cv or perm
    list_dir = os.listdir(os.path.join(flags['experiment_folder']))
    if ('cv' + str(nth_fold) in list_dir) and ('perm' + str(nth_fold) in list_dir):
        raise ValueError('Dangerous! You have both cv and perm on the path.')
    elif 'cv' + str(nth_fold) in list_dir:
        object_folder = os.path.join(flags['experiment_folder'], 'cv' + str(nth_fold))
    elif 'perm' + str(nth_fold) in list_dir:
        object_folder = os.path.join(flags['experiment_folder'], 'perm' + str(nth_fold))
    else:
        raise ValueError('No cv or perm folder!')

    # Train model
    if mode == 'train':
        checkpoint_folder = os.path.join(object_folder, 'checkpoint')
        network_stats_file_path = os.path.join(checkpoint_folder, 'network_stats.mat')

        train_images_folder = os.path.join(object_folder, 'train', 'image')

        if not os.path.isfile(network_stats_file_path):
            list_images = glob.glob(os.path.join(train_images_folder, '*' + flags['image_ext']))
            print('calculating mean and variance image')
            mean_image, variance_image = utils.calculate_mean_variance_image(list_images)
            routine.create_dir(checkpoint_folder)
            matlab.save(network_stats_file_path, {'mean_image': mean_image, 'variance_image': variance_image})

        tf_model_train.train(object_folder, flags)

    # Test model on validation set
    elif mode == 'test_model':
        checkpointlist = glob.glob(os.path.join(object_folder, 'checkpoint', 'model*meta'))
        checkpointlist = [file for file in checkpointlist if 'pretrain' not in file]
        temp = []
        for filepath in checkpointlist:
            basename = os.path.basename(filepath)
            temp.append(int(float(basename.split('-')[-1].split('.')[0])))
        temp = np.sort(temp)

        model_path = os.path.join(object_folder, 'checkpoint', 'model.ckpt-' + str(temp[flags['test_model']]))
        print('use epoch %d : model %s' % (flags['test_model'], 'model.ckpt-' + str(temp[flags['test_model']])))
        test_images_list = flags['test_image_list']
        filename_list = KScsv.read_csv(test_images_list)
        tf_model_test.test(object_folder, model_path, filename_list, flags)

    #Segment WSIs
    elif mode == 'test_WSI':
        checkpointlist = glob.glob(os.path.join(object_folder, 'checkpoint', 'model*meta'))
        checkpointlist = [file for file in checkpointlist if 'pretrain' not in file]
        temp = []
        for filepath in checkpointlist:
            basename = os.path.basename(filepath)
            temp.append(int(float(basename.split('-')[-1].split('.')[0])))
        temp = np.sort(temp)

        model_path = os.path.join(object_folder, 'checkpoint', 'model.ckpt-' + str(temp[flags['test_model']]))
        print('use epoch %d : model %s' % (flags['test_model'], 'model.ckpt-' + str(temp[flags['test_model']])))

        #should iterate over all subdirectories
        paths = get_immediate_subdirectories(testdir)
        list.sort(paths) #sort WSIs into ascending numerical order
        #paths = paths[100:] #TODO: Enable based on which batch this code is running
        print("TEST DIR: " + str(testdir))

        for path in paths:
            print(os.path.join(testdir,path))
            if not os.path.isdir(os.path.join(testdir,path+'epiStromalSeg')): #prevents this from being executed with exsiting directories
               tf_model_test.testWSI(object_folder, model_path, os.path.join(testdir,path), flags)
                
                #TODO: uncomment to process only controls
                #imageCSV = open(os.path.join('/data', 'avellal14', 'WSI_patches', 'BBD_NCC_Covariate_Outcome_KK_JH_modifiedWithPaths.csv'),'rb')
		        #reader = csv.reader(imageCSV)
                #csvList = list(reader)
                #patientId = path[:path.index('_')]
		        #caseControlList =  next(subl for subl in csvList if patientId in subl)
	            #TODO: uncomment to process only cases
                # if(caseControlList[1] == '1'): #only test the WSI if the image is indeed a case(1)
               #        tf_model_test.testWSI(object_folder, model_path, os.path.join(testdir,path), flags)

    #Segment WSIs at patient level using data from CSV
    elif mode == 'test_Case_Control':
        checkpointlist = glob.glob(os.path.join(object_folder, 'checkpoint', 'model*meta'))
        checkpointlist = [file for file in checkpointlist if 'pretrain' not in file]
        temp = []
        for filepath in checkpointlist:
            basename = os.path.basename(filepath)
            temp.append(int(float(basename.split('-')[-1].split('.')[0])))
        temp = np.sort(temp)

        model_path = os.path.join(object_folder, 'checkpoint', 'model.ckpt-' + str(temp[flags['test_model']]))
        print('use epoch %d : model %s' % (flags['test_model'], 'model.ckpt-' + str(temp[flags['test_model']])))

        with open(os.path.join('/home', 'avellal14', 'data', 'Adithya_BBD_NHS', 'NHS_BBD_CODE', 'casesAndMatchedControls224.csv')) as csvFile:
                csvReader = csv.DictReader(csvFile)
                for row in csvReader:
                        if(row['path'] == 'BBD_NCC_extractedat20x' or row['path'] == 'BBD_NCC_extractedat20x_round2'):
                                testdir = os.path.join('/home', 'avellal14', 'data', 'Adithya_BBD_NHS', row['path'])
                                paths = get_subdirectories_by_patient(testdir, row['id'])

                                for path in paths:
                                        print('CURRENT WSI BEING SEGMENTED', os.path.join(testdir,path))
                                        if not os.path.isdir(os.path.join(testdir,path+'_cellSeg')): #prevents this from being executed with exsiting directories
                                                tf_model_test.testWSI(object_folder, model_path, os.path.join(testdir,path), flags)


###################################################
def get_subdirectories_by_patient(a_dir, patientID):
    """
    get_subdirectories_by_patient returns WSI directories
    associated with the current patient that have not been
    segmented yet

    param: a_dir
    param: patientID
    return: list of WSI directories
    """

    return [name for name in os.listdir(a_dir) if os.path.isdir(os.path.join(a_dir, name)) and 'Seg' not in name and patientID in name]

###################################################
def get_immediate_subdirectories(a_dir):
    """
    get_immediate_subdirectories return WSI directories
    that have not been segmented yet

    param: a_dir
    return: list of WSI directories
    """
    return [name for name in os.listdir(a_dir) if os.path.isdir(os.path.join(a_dir, name)) and 'Seg' not in name]
