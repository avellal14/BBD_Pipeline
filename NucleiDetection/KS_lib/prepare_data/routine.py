"""
routine.py

This file generates the experiment folder where
all experiments (dataset generation, training,
testing, etc;) will be conducted.
"""
import os
import glob
import re
import collections
import time
import csv
import numpy as np


from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import StratifiedShuffleSplit
from KS_lib.prepare_data import extract_patches
from KS_lib.general import KScsv
from KS_lib.image import KSimage
from KS_lib.prepare_data import select_instances
from KS_lib.general import matlab

from scipy.ndimage.morphology import binary_erosion
from skimage.morphology import watershed, remove_small_objects
from scipy import ndimage


#################################################################################
class RegexDict(dict):
    """
    RegexDict is a nested class with methods to handle dictionary operations

    param: dict
    param: event
    return: specified keys
    """
    def get_matching(self, event):
        return (self[key] for key in self if re.match(key, event))

    def get_all_matching(self, events):
        return (match for event in events for match in self.get_matching(event))

#################################################################################
def write_train_log(patch_list, flags):
    """
    write_train_log writes names of all training patches to a .csv log

    param: patch_list
    return: void
    """
    object_folder = os.path.join(flags['experiment_folder'], 'perm1')
    train_log_path = os.path.join(object_folder, 'train', 'train_log.csv')
    train_csv = open(train_log_path, 'wb')
    trainWriter = csv.writer(train_csv)

    for i in range(len(patch_list)):
        fname = patch_list[i][1]
        basename = os.path.basename(fname)
        basename = basename[:len(basename) - 4]
        trainWriter.writerow([os.path.join(object_folder, 'train', 'image', basename + '.png'),
                              os.path.join(object_folder, 'train', 'groundtruth', basename + '.png'),
                              os.path.join(object_folder, 'train', 'weight', basename + '.png')])

    print("train_log_path: " + train_log_path)

################################################################################################
def get_pair_list(dict_path, dict_ext):
    """
    get_pair_list organizes and returns list of images with corresponding labels and groups

    param: dict_path
    param: dict_ext
    return: required object list containing original images, labeled images, and image group information
    """
    images_list = glob.glob(os.path.join(dict_path['image'], '*' + dict_ext['image']))
    obj_list = collections.defaultdict(list)

    for image_name in images_list:
        basename = os.path.basename(image_name)
        basename = os.path.splitext(basename)[0]

        dict_name = dict()
        for key in dict_path.keys():
            dict_name[key] = os.path.join(dict_path[key], basename + dict_ext[key])

        if all(os.path.isfile(v) for k, v in dict_name.items()):
            for key in dict_path.keys():
                obj_list[key].append(dict_name[key])

    for key in obj_list.keys():
        if not obj_list[key]:
            print("no data in %s" % (dict_path[key]))
            raise ValueError('terminate!')

    return obj_list

################################################################################################
def create_dir(dir_name):
    """
    create_dir creates a directory if it does not exist

    param: dir_name
    return: void
    """
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

################################################################################################
def split_cv(obj_list, flags):
    """
    split_cv splits data into train, validation, and test stratified by the group label

    param: images_list
    param: labels_list
    param: groups_list
    param: num
    param: val_percentage
    return: void
    """

    num = flags['num_split']
    val_percentage = flags['val_percentage']

    groups_label = list()
    for file in obj_list['group']:
        row = KScsv.read_csv(file)
        groups_label.append(row[0][0])
    groups_label = np.array(groups_label)

    for key in obj_list.keys():
        obj_list[key] = np.array(obj_list[key])

    skf = StratifiedKFold(n_splits=num)
    for i_num, (train_idx, test_idx) in enumerate(skf.split(obj_list['image'], groups_label)):
        cv_folder = os.path.join(flags['experiment_folder'], 'cv' + str(i_num + 1))
        create_dir(cv_folder)

        test_obj_list_dict = dict()
        train_obj_list_dict = dict()
        for key in obj_list.keys():
            test_obj_list_dict[key] = obj_list[key][test_idx]
            train_obj_list_dict[key] = obj_list[key][train_idx]

        train_groups_label = groups_label[train_idx]

        sss = StratifiedShuffleSplit(n_splits=1, test_size=val_percentage / 100.0)
        for train_train_index, train_val_index in sss.split(train_obj_list_dict['image'], train_groups_label):
            train_train_obj_list_dict = dict()
            train_val_obj_list_dict = dict()
            for key in train_obj_list_dict.keys():
                train_train_obj_list_dict[key] = train_obj_list_dict[key][train_train_index]
                train_val_obj_list_dict[key] = train_obj_list_dict[key][train_val_index]

        #################################################################
        # test
        for key in test_obj_list_dict.keys():
            filename = os.path.join(cv_folder, 'test_' + key + '_list.csv')
            if not os.path.isfile(filename):
                row_list = [[item] for item in test_obj_list_dict[key]]
                KScsv.write_csv(row_list, filename)

        #################################################################
        # train
        for key in train_train_obj_list_dict.keys():
            filename = os.path.join(cv_folder, 'train_' + key + '_list.csv')
            if not os.path.isfile(filename):
                row_list = [[item] for item in train_train_obj_list_dict[key]]
                KScsv.write_csv(row_list, filename)

        #################################################################
        # validation
        for key in train_val_obj_list_dict.keys():
            filename = os.path.join(cv_folder, 'val_' + key + '_list.csv')
            if not os.path.isfile(filename):
                row_list = [[item] for item in train_val_obj_list_dict[key]]
                KScsv.write_csv(row_list, filename)


################################################################################################
def split_perm(obj_list, flags):
    """
    split_perm splits data using permutation with stratification based on group label

    param: images_list
    param: labels_list
    param: groups_list
    param: num
    param: test_percentage
    param: val_percentage
    return: void
    """
    num = flags['num_split']
    test_percentage = flags['test_percentage']
    val_percentage = flags['val_percentage']

    groups_label = list()
    for file in obj_list['group']:
        row = KScsv.read_csv(file)
        groups_label.append(row[0][0])
    groups_label = np.array(groups_label)

    for key in obj_list.keys():
        obj_list[key] = np.array(obj_list[key])

    if test_percentage != 0:
        skf = StratifiedShuffleSplit(n_splits=num, test_size=test_percentage / 100.0)
        for i_num, (train_idx, test_idx) in enumerate(skf.split(obj_list['image'], groups_label)):
            cv_folder = os.path.join(flags['experiment_folder'], 'perm' + str(i_num + 1))
            create_dir(cv_folder)

            test_obj_list_dict = dict()
            train_obj_list_dict = dict()
            for key in obj_list.keys():
                test_obj_list_dict[key] = obj_list[key][test_idx]
                train_obj_list_dict[key] = obj_list[key][train_idx]

            train_groups_label = groups_label[train_idx]

            sss = StratifiedShuffleSplit(n_splits=1, test_size=val_percentage / 100.0)
            for train_train_index, train_val_index in sss.split(train_obj_list_dict['image'], train_groups_label):
                train_train_obj_list_dict = dict()
                train_val_obj_list_dict = dict()
                for key in train_obj_list_dict.keys():
                    train_train_obj_list_dict[key] = train_obj_list_dict[key][train_train_index]
                    train_val_obj_list_dict[key] = train_obj_list_dict[key][train_val_index]

            #################################################################
            # test
            for key in test_obj_list_dict.keys():
                filename = os.path.join(cv_folder, 'test_' + key + '_list.csv')
                if not os.path.isfile(filename):
                    row_list = [[item] for item in test_obj_list_dict[key]]
                    KScsv.write_csv(row_list, filename)

            #################################################################
            # train
            dict_path = flags['dict_path']
            dict_ext = flags['dict_ext']

            obj_list_dict = dict()
            for key in dict_path.keys():
                obj_list_dict[key] = glob.glob(os.path.join(dict_path[key], '*' + dict_ext[key]))

            temp_train_train_obj_list_dict = collections.defaultdict(list)

            for name in train_train_obj_list_dict['image']:
                basename = os.path.basename(name)
                basename = os.path.splitext(basename)[0]
                matching = sorted([s for s in obj_list_dict['image'] if basename in s])

                for m in matching:
                    basename = os.path.basename(m)
                    basename = os.path.splitext(basename)[0]

                    basename_dict = dict()
                    for key in train_train_obj_list_dict.keys():
                        basename_dict[key] = os.path.join(dict_path[key], basename + dict_ext[key])

                    if all(basename_dict[k] in obj_list_dict[k] for k in basename_dict.keys()):
                        for key in train_train_obj_list_dict.keys():
                            temp_train_train_obj_list_dict[key].append(basename_dict[key])

            for key in train_train_obj_list_dict.keys():
                train_train_obj_list_dict[key] = np.array(temp_train_train_obj_list_dict[key])

                filename = os.path.join(cv_folder, 'train_' + key + '_list.csv')
                if not os.path.isfile(filename):
                    row_list = [[item] for item in train_train_obj_list_dict[key]]
                    KScsv.write_csv(row_list, filename)

            #################################################################
            # validation
            dict_path = flags['dict_path']
            dict_ext = flags['dict_ext']

            obj_list_dict = dict()
            for key in dict_path.keys():
                obj_list_dict[key] = glob.glob(os.path.join(dict_path[key], '*' + dict_ext[key]))

            temp_train_val_obj_list_dict = collections.defaultdict(list)

            for name in train_val_obj_list_dict['image']:
                basename = os.path.basename(name)
                basename = os.path.splitext(basename)[0]
                matching = sorted([s for s in obj_list_dict['image'] if basename in s])

                for m in matching:
                    basename = os.path.basename(m)
                    basename = os.path.splitext(basename)[0]

                    basename_dict = dict()
                    for key in train_val_obj_list_dict.keys():
                        basename_dict[key] = os.path.join(dict_path[key], basename + dict_ext[key])

                    if all(basename_dict[k] in obj_list_dict[k] for k in basename_dict.keys()):
                        for key in train_val_obj_list_dict.keys():
                            temp_train_val_obj_list_dict[key].append(basename_dict[key])

            for key in train_val_obj_list_dict.keys():
                train_val_obj_list_dict[key] = np.array(temp_train_val_obj_list_dict[key])

                filename = os.path.join(cv_folder, 'val_' + key + '_list.csv')
                if not os.path.isfile(filename):
                    row_list = [[item] for item in train_val_obj_list_dict[key]]
                    KScsv.write_csv(row_list, filename)

    else:
        for i_num in range(num):
            cv_folder = os.path.join(flags['experiment_folder'], 'perm' + str(i_num + 1))
            create_dir(cv_folder)

            train_obj_list_dict = dict()
            for key in obj_list.keys():
                train_obj_list_dict[key] = obj_list[key]
            train_groups_label = groups_label

            sss = StratifiedShuffleSplit(n_splits=1, test_size=val_percentage / 100.0)
            for train_train_index, train_val_index in sss.split(train_obj_list_dict['image'], train_groups_label):
                train_train_obj_list_dict = dict()
                train_val_obj_list_dict = dict()
                for key in train_obj_list_dict.keys():
                    train_train_obj_list_dict[key] = train_obj_list_dict[key][train_train_index]
                    train_val_obj_list_dict[key] = train_obj_list_dict[key][train_val_index]

            #################################################################
            # train
            dict_path = flags['dict_path']
            dict_ext = flags['dict_ext']

            obj_list_dict = dict()
            for key in dict_path.keys():
                obj_list_dict[key] = glob.glob(os.path.join(dict_path[key], '*' + dict_ext[key]))

            temp_train_train_obj_list_dict = collections.defaultdict(list)

            for name in train_train_obj_list_dict['image']:
                basename = os.path.basename(name)
                basename = os.path.splitext(basename)[0]
                matching = sorted([s for s in obj_list_dict['image'] if basename in s])

                for m in matching:
                    basename = os.path.basename(m)
                    basename = os.path.splitext(basename)[0]

                    basename_dict = dict()
                    for key in train_train_obj_list_dict.keys():
                        basename_dict[key] = os.path.join(dict_path[key], basename + dict_ext[key])

                    if all(basename_dict[k] in obj_list_dict[k] for k in basename_dict.keys()):
                        for key in train_train_obj_list_dict.keys():
                            temp_train_train_obj_list_dict[key].append(basename_dict[key])

            for key in train_train_obj_list_dict.keys():
                train_train_obj_list_dict[key] = np.array(temp_train_train_obj_list_dict[key])

                filename = os.path.join(cv_folder, 'train_' + key + '_list.csv')
                if not os.path.isfile(filename):
                    row_list = [[item] for item in train_train_obj_list_dict[key]]
                    KScsv.write_csv(row_list, filename)

            #################################################################
            # validation

            dict_path = flags['dict_path']
            dict_ext = flags['dict_ext']

            obj_list_dict = dict()
            for key in dict_path.keys():
                obj_list_dict[key] = glob.glob(os.path.join(dict_path[key], '*' + dict_ext[key]))

            temp_train_val_obj_list_dict = collections.defaultdict(list)

            for name in train_val_obj_list_dict['image']:
                basename = os.path.basename(name)
                basename = os.path.splitext(basename)[0]
                matching = sorted([s for s in obj_list_dict['image'] if basename in s])

                for m in matching:
                    basename = os.path.basename(m)
                    basename = os.path.splitext(basename)[0]

                    basename_dict = dict()
                    for key in train_val_obj_list_dict.keys():
                        basename_dict[key] = os.path.join(dict_path[key], basename + dict_ext[key])

                    if all(basename_dict[k] in obj_list_dict[k] for k in basename_dict.keys()):
                        for key in train_val_obj_list_dict.keys():
                            temp_train_val_obj_list_dict[key].append(basename_dict[key])

            for key in train_val_obj_list_dict.keys():
                train_val_obj_list_dict[key] = np.array(temp_train_val_obj_list_dict[key])

                filename = os.path.join(cv_folder, 'val_' + key + '_list.csv')
                if not os.path.isfile(filename):
                    row_list = [[item] for item in train_val_obj_list_dict[key]]
                    KScsv.write_csv(row_list, filename)


################################################################################################
def split_data(flags):
    """
    split_data uses preferences specified in flags file to split data into training and testing sets through either perm or cv

    param: dict_path
    param: dict_ext
    param: augmentation_keyword
    param: split_method
    return: void
    """

    obj_list = get_pair_list(flags['dict_path'], flags['dict_ext'])

    for key in obj_list.keys():
        tmp_list = list()
        for name in obj_list[key]:
            if flags['augmentation_keyword'] not in name:
                tmp_list.append(name)
        obj_list[key] = tmp_list

    # cross validation
    if flags['split_method'] == 'cv':
        split_cv(obj_list, flags)

    # permutation
    elif flags['split_method'] == 'perm':
        split_perm(obj_list,flags)

    else:
        raise ValueError('please select cv or perm')

################################################################################################
def gen_train_val_data(nth_fold, flags):
    """
    gen_train_val_data generates training and validation data for training the network. It builds
    directories for train and test and extract patches according to the provided 'method', and it
    maintains a log file containing the contents of all the data splits

    param: nth_fold
    param method: sliding_window
    return: void
    """

    ########## check whether 'cv' or 'perm' exists and which one to use ##########
    list_dir = os.listdir(os.path.join(flags['experiment_folder']))
    if ('cv' + str(nth_fold) in list_dir) and ('perm' + str(nth_fold) in list_dir):
        raise ValueError('Dangerous! You have both cv and perm on the path.')
    elif 'cv' + str(nth_fold) in list_dir:
        object_folder = os.path.join(flags['experiment_folder'], 'cv' + str(nth_fold))
    elif 'perm' + str(nth_fold) in list_dir:
        object_folder = os.path.join(flags['experiment_folder'], 'perm' + str(nth_fold))
    else:
        raise ValueError('No cv or perm folder!')

    ########## create train and val paths ##########
    path_dict = dict()
    path_dict['train_folder'] = os.path.join(object_folder, 'train')
    path_dict['val_folder'] = os.path.join(object_folder, 'val')
    create_dir(path_dict['train_folder'])
    create_dir(path_dict['val_folder'])

    print("Gets to the beginning of an if statement")
    ########## extract patches and put in a designated directory ##########
    if flags['gen_train_val_method'] == 'sliding_window':

        key_list = ['image', 'groundtruth', 'weight']

        for key in key_list:
            path_dict['train_' + key + '_folder'] = os.path.join(path_dict['train_folder'], key)
            create_dir(path_dict['train_' + key + '_folder'])
            path_dict['val_' + key + '_folder'] = os.path.join(path_dict['val_folder'], key)
            create_dir(path_dict['val_' + key + '_folder'])

        list_dict = dict()
        for key in key_list:
            list_dict['train_' + key + '_list'] = KScsv.read_csv(
                os.path.join(object_folder, 'train_' + key + '_list.csv'))
            list_dict['val_' + key + '_list'] = KScsv.read_csv(os.path.join(object_folder, 'val_' + key + '_list.csv'))

        ########## train ##########
        for key in ['train', 'val']:
            if not os.path.isfile(os.path.join(path_dict[key + '_folder'], key + '_log.csv')):
                log_data = list()

                for i_image in range(len(list_dict[key + '_image_list'])):

                    tic = time.time()

                    path_image = list_dict[key + '_image_list'][i_image][0]
                    path_groundtruth = list_dict[key + '_groundtruth_list'][i_image][0]
                    path_weight = list_dict[key + '_weight_list'][i_image][0]

                    # Resize image, groundtruth, and weight from 10x input size to 2.5x (level at which network operates)
                    image = KSimage.imread(path_image)
                    image = KSimage.imresize(image, 0.5)

                    groundtruth = KSimage.imread(path_groundtruth)
                    groundtruth = KSimage.imresize(groundtruth, 0.5)

                    weight = KSimage.imread(path_weight)
                    weight = KSimage.imresize(weight, 0.5)

                    dict_obj = {'image': image,
                                'groundtruth': groundtruth,
                                'weight': weight}

                    extractor = extract_patches.sliding_window(dict_obj, flags['size_input_patch'],
                                                               flags['size_output_patch'], flags['stride'])

                    for j, (out_obj_dict, coord_dict) in enumerate(extractor):
                        images = out_obj_dict['image']
                        groundtruths = out_obj_dict['groundtruth']
                        weights = out_obj_dict['weight']
                        coord_images = coord_dict['image']

                        #############################################################

                        basename = os.path.basename(path_image)
                        basename = os.path.splitext(basename)[0]

                        image_name = os.path.join(path_dict[key + '_image_folder'],
                                                  basename + '_idx' + str(j) + '_row' + str(
                                                      coord_images[0]) + '_col' + str(coord_images[1]) + flags[
                                                      'image_ext'])
                        label_name = os.path.join(path_dict[key + '_groundtruth_folder'],
                                                  basename + '_idx' + str(j) + '_row' + str(
                                                      coord_images[0]) + '_col' + str(coord_images[1]) + flags[
                                                      'groundtruth_ext'])
                        weight_name = os.path.join(path_dict[key + '_weight_folder'],
                                                   basename + '_idx' + str(j) + '_row' + str(
                                                       coord_images[0]) + '_col' + str(coord_images[1]) + flags[
                                                       'weight_ext'])

                        if not os.path.isfile(image_name):
                            KSimage.imwrite(images, image_name)

                        if not os.path.isfile(label_name):
                            KSimage.imwrite(groundtruths, label_name)

                        if not os.path.isfile(weight_name):
                            KSimage.imwrite(weights, weight_name)

                        log_data.append((image_name, label_name, weight_name))

                    print('finish processing %d image from %d images : %.2f' % (
                        i_image + 1, len(list_dict[key + '_image_list']), time.time() - tic))

                KScsv.write_csv(log_data, os.path.join(path_dict[key + '_folder'], key + '_log.csv'))

    ####################################################################################################################
    else:
        print ("ONLY SLIDING WINDOW TRAINING IS SUPPORTED!!!! Training terminated.")
        return



################################################################################################
def select_train_val_instances(nth_fold, method, flags):
    """
    select_train_val_instances is used to balance the class instances
    :param nth_fold:
    :param method:
    :return:
    """
    # check if log files exist
    list_dir = os.listdir(os.path.join(flags['experiment_folder']))
    if ('cv' + str(nth_fold) in list_dir) and ('perm' + str(nth_fold) in list_dir):
        raise ValueError('Dangerous! You have both cv and perm on the path.')
    elif 'cv' + str(nth_fold) in list_dir:
        object_folder = os.path.join(flags['experiment_folder'], 'cv' + str(nth_fold))
    elif 'perm' + str(nth_fold) in list_dir:
        object_folder = os.path.join(flags['experiment_folder'], 'perm' + str(nth_fold))
    else:
        raise ValueError('No cv or perm folder!')

    train_log_file_path = os.path.join(object_folder, 'train', 'train_log.csv')
    val_log_file_path = os.path.join(object_folder, 'val', 'val_log.csv')

    if not os.path.isfile(train_log_file_path):
        raise ValueError('no ' + train_log_file_path)
    if not os.path.isfile(val_log_file_path):
        raise ValueError('no ' + val_log_file_path)

    # read csv
    train_log = KScsv.read_csv(train_log_file_path)
    val_log = KScsv.read_csv(val_log_file_path)

    # count the number
    if method == 'by_numbers':
        train_log = select_instances.by_numbers(train_log)
        val_log = select_instances.by_numbers(val_log)

        KScsv.write_csv(train_log, train_log_file_path)
        KScsv.write_csv(val_log, val_log_file_path)
    else:
        raise ValueError('no method ' + method + ' exists!')

################################################################################################
def retouch_segmentation(file):
    """
    retouch_segmentation uses binary morphological operations (erode and dilate)
    to improve the segmentation result

    param: segmentation result
    return: retouched segmentation result
    """

    matcontent = matlab.load(file)
    mask = matcontent['mask']
    mask = np.squeeze(mask)

    # threshold
    binary_mask = mask > 0.8
    binary_mask_base = mask > 0.5

    # define disk structure
    radius = 5
    [x, y] = np.meshgrid(range(-radius, radius + 1), range(-radius, radius + 1))
    z = np.sqrt(x ** 2 + y ** 2)
    structure = z < radius

    # imerosion
    erode_mask = binary_erosion(binary_mask, structure=structure, border_value=1)
    erode_mask = remove_small_objects(erode_mask, 100)

    # watershed
    distance = ndimage.distance_transform_edt(binary_mask_base)
    markers = ndimage.label(erode_mask)[0]
    labels = watershed(-distance, markers, mask=binary_mask_base)

    return labels


################################################################################################
def post_processing_segmentation(test_image_path):
    """
    post_processing_segmentation iterates through all the test results in
    the given directory and retouches all of them using binary morphological
    operators

    param: test_image_path
    return: void
    """

    file_list = test_image_path

    post_process_folder = os.path.join('postprocess')
    create_dir(post_process_folder)

    for iImage, file in enumerate(file_list):
        tic = time.time()

        path, filename = os.path.split(file)
        basename = os.path.splitext(filename)[0]
        savename = os.path.join(post_process_folder, basename + '.mat')

        if not os.path.isfile(savename):
            labels = retouch_segmentation(file)
            matlab.save(savename, {'mask': labels})

        duration = time.time() - tic
        print('process %d / %d images (%.2f sec)' % (iImage + 1, len(file_list), duration))