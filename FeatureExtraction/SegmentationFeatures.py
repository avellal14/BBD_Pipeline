"""
SegmentationFeatures.py

This file contains the functions for extracting various morphological,
textural, and graph-based features using the original WSI, the epi-stromal segmentation
result, and the nuclei segmentation result.
"""

import numpy as np
import csv
import os
import time
import mahotas as mh
import cv2
import math
from skimage.measure import regionprops
from skimage.morphology import remove_small_objects
from scipy.stats import skew, kurtosis
from scipy.spatial import Voronoi, Delaunay
from numpy.linalg import norm
from StitchPatches import stitch_image, stitch_seg
from SegPostProcessing import remove_blank_blurry_patches, postprocess_segs


############################################################################################
def get_feature_dict():
    """
    get_feature_dict assembles an empty dictionary containing all of the different features to be calculated

    param: none
    return: empty feature dictionary
    """

    #complete feature dictionary
    feature_dict = dict()

    # make a copy of the corresponding features for each tissue type as well as the segmentation result
    feature_dict['epi'] = dict()
    feature_dict['stroma'] = dict()
    #feature_dict['fat'] = dict()
    feature_dict['seg'] = dict()

    return feature_dict




############################################################################################
def label_regions(og_img, epi_stromal_seg, binary_cell_seg):
    """
    label_regions labels all of the individual nuclei, epi, stroma, and fat regions in the segmentation 
    results, and returns them in a dictionary 

    param: og_img, epi_stromal_seg, binary_cell_seg
    return: dictionary with labeled and indexed nuclei, epi, stroma, and fat regions along with their perimeters
    """

    #remove small artifacts from the nuclei segmentation result
    binary_cell_seg = remove_small_objects(binary_cell_seg == 255, 10)
    binary_cell_seg = np.array(binary_cell_seg, np.uint8)

    #create a smoothed out black and white epi mask
    seg_epi = np.copy(epi_stromal_seg)
    seg_epi[seg_epi != 1] = 0
    seg_epi[seg_epi == 1] = 255
    seg_epi = remove_small_objects(seg_epi == 255, 10**2) #remove small artifacts from the epi mask
    seg_epi = (seg_epi * 255).astype(np.uint8)

    #create a smoothed out black and white stroma mask
    seg_stroma = np.copy(epi_stromal_seg)
    seg_stroma[seg_stroma != 2] = 0
    seg_stroma[seg_stroma == 2] = 255
    seg_stroma = remove_small_objects(seg_stroma == 255, 10**2) #remove small artifacts from the stroma mask
    seg_stroma = (seg_stroma * 255).astype(np.uint8)

    #create a smoothed out black and white fat mask
   # seg_fat = np.copy(epi_stromal_seg)
    #seg_fat[seg_fat != 3] = 0
   # seg_fat[seg_fat == 3] = 255
   # seg_fat = remove_small_objects(seg_fat == 255, 10**2) #remove small artifacts from the fat mask
   # seg_fat = (seg_fat * 255).astype(np.uint8)

    #label and index all the individual epi regions and obtain their perimeters as well
    labeled_epi, num_epi = mh.label(seg_epi)
    perim_epi = mh.labeled.bwperim(seg_epi)

    #label and index all the individual stroma regions and obtain their perimeters as well
    labeled_stroma, num_stroma = mh.label(seg_stroma)
    perim_stroma = mh.labeled.bwperim(seg_stroma)

    #label and index all the individual fat regions and obtain their perimeters as well
    #labeled_fat, num_fat = mh.label(seg_fat)
    #perim_fat = mh.labeled.bwperim(seg_fat)

    return {
        'og_img': og_img, #read in image
        'es_seg': epi_stromal_seg, #read in image
        'cell_seg': binary_cell_seg, #read in image

        'labeled_epi': labeled_epi, #regions labeled 0,1,2,3,..,n
        'num_epi': num_epi, #num distinct regions
        'perim_epi': perim_epi, #perimeter mask

        'labeled_stroma': labeled_stroma, #regions labeled 0,1,2,3,..,n
        'num_stroma': num_stroma, #num distinct regions
        'perim_stroma': perim_stroma, #perimeter mask

        #'labeled_fat': labeled_fat, #regions labeled 0,1,2,3,..,n
        #'num_fat': num_fat, #num fat regions
        #'perim_fat': perim_fat, #perimeter mask
    }




############################################################################################
def find_morphological_features_patch(region_label_dict, region_type):
    """
    find_features_patch computes the areas, perimeters, and RGB values of all the individual regions
    belonging to the provided tissue type. It also computes the average nucleus density, number of nuclei,
    nucleus area, nucleus perimeter, and nucleus RGB values using all the nuclei found in the provided type
    of tissue region. All of these results are returned in a dictionary

    param: region_label_dict, region_type
    return: dictionary with morphological features describing both regions of the provided tissue type
            and the nuclei found within them
    """


    area_array = np.ones(region_label_dict['num_' + region_type]) #area of each region in the patch
    perim_array = np.ones(region_label_dict['num_' + region_type]) #perimeter of each region in the patch
    r_array = np.ones(region_label_dict['num_' + region_type]) #average R value of each region in the patch
    g_array = np.ones(region_label_dict['num_' + region_type]) #average G value of each region in the patch
    b_array = np.ones(region_label_dict['num_' + region_type]) #average B value of each region in the patch

    cell_area_array = np.ones(region_label_dict['num_' + region_type]) #average nucleus area of each region in the patch
    cell_perim_array = np.ones(region_label_dict['num_' + region_type]) #average nucleus perimeter of each region in the patch
    cell_r_array = np.ones(region_label_dict['num_' + region_type]) #average nucleus R value of each region in the patch
    cell_g_array = np.ones(region_label_dict['num_' + region_type]) #average nucleus G value of each region in the patch
    cell_b_array = np.ones(region_label_dict['num_' + region_type]) #average nucleus B value of each region in the patch
    num_cells_array = np.ones(region_label_dict['num_' + region_type]) #average number of cell nuclei of each region in the patch
    cell_density_array = np.ones(region_label_dict['num_' + region_type]) #average cell nuclei density of each region in the patch
    total_cells_array = np.empty([0]) #total number of cells in all regions belonging to this tissue type

    #loop through all the regions in the patch belonging to the provided tissue type
    for i in range(region_label_dict['num_' + region_type]):
        region_mask = region_label_dict['labeled_' + region_type] == (i + 1)  #binary mask highlighting the regions of the provided tissue type
        area_array[i] = np.sum(region_mask)  #total pixel area of all regions in the patch

        region_mask = (region_mask * 255).astype(np.uint8)  #make mask black and white to compute perimeters

        #overlay the tissue type mask and the region perimeter mask to compute the total perimeter of all regions belonging to the provided tissue type
        perim_mask = np.array(region_label_dict['perim_' + region_type], dtype=np.uint8)
        perim_mask[perim_mask != 0] = 1
        region_mask[region_mask != 0] = 1
        perim_mask = np.array(perim_mask & region_mask, dtype=np.uint8)
        perim_array[i] = np.sum(perim_mask)


        #take the average of all non-zero RGB pixel values belonging to the tissue regions in the patch
        og_img = region_label_dict['og_img']
        r_channel = og_img[:, :, 0]
        g_channel = og_img[:, :, 1]
        b_channel = og_img[:, :, 2]
        r_vals = r_channel[region_mask != 0]
        g_vals = g_channel[region_mask != 0]
        b_vals = b_channel[region_mask != 0]
        r_array[i] = np.mean(r_vals)
        g_array[i] = np.mean(g_vals)
        b_array[i] = np.mean(b_vals)

        #obtain binary nucleus mask
        cell_mask = region_label_dict['cell_seg']
        cell_mask[cell_mask == 255] = 1

        #overlay the nucleus mask and the tissue region mask
        region_cell_mask = np.array(cell_mask & region_mask, dtype=np.uint8)

        cell_area = np.sum(region_cell_mask) #total area of all nuclei in the patch's tissue regions

        #label and index all the nuclei in the corresponding tissue regions
        region_cell_mask[region_cell_mask == 1] == 255
        labeledCells, num_cells = mh.label(region_cell_mask)


        total_cells_array = np.append(total_cells_array, [int(num_cells)])

        #find the total perimeter of all nuclei in the corresponding tissue regions
        cell_perim = mh.labeled.bwperim(region_cell_mask)

        #calculate the average nucleus area and perimeter for each tissue region
        if (num_cells == 0):
            cell_area_array[i] = 0
            cell_perim_array[i] = 0
        else:
            cell_area_array[i] = cell_area / float(num_cells)
            cell_perim_array[i] = np.sum(cell_perim) / float(num_cells)

        #calculate the average nucleus RGB values for each tissue region
        if (num_cells > 0):
            r_vals = r_channel[region_cell_mask != 0]
            g_vals = g_channel[region_cell_mask != 0]
            b_vals = b_channel[region_cell_mask != 0]
            cell_r_array[i] = np.mean(r_vals)
            cell_b_array[i] = np.mean(g_vals)
            cell_g_array[i] = np.mean(b_vals)
        else:
            cell_r_array[i] = 0
            cell_b_array[i] = 0
            cell_g_array[i] = 0

        #use total number of nuclei to determine nucleus density for each tissue region
        num_cells_array[i] = num_cells
        if (area_array[i] > 0):
            cell_density_array[i] = num_cells_array[i] / float(area_array[i])  # cells per pixel
        else:
            cell_density_array[i] = 0

    return area_array, perim_array, r_array, g_array, b_array, cell_area_array, cell_perim_array, cell_r_array, cell_g_array, cell_b_array, num_cells_array, cell_density_array, total_cells_array




############################################################################################
def find_texture_features_patch(region_label_dict, region_type):
    """
    find_texture_features_patch computes haralick and local binary pattern features for the provided tissue type

    param: region_label_dict, region_type
    return: arrays with haralick and local binary pattern features describing all regions of the provided tissue type
    """

    #obtain a grayscale mask of all regions belonging to the coresponding tissue type
    region_labels = region_label_dict['labeled_' + region_type]
    og_img = region_label_dict['og_img']
    region_mask = np.copy(og_img)
    region_mask[region_labels == 0] = 0
    region_mask = mh.colors.rgb2grey(region_mask)
    region_mask = region_mask.astype(np.uint8)

    #compute haralick texture features using the grayscale tissue region mask
    try:
        return mh.features.haralick(region_mask, ignore_zeros=True), mh.features.lbp(region_mask, 1, 8, ignore_zeros=True)
    
    except:
        return mh.features.haralick(region_mask), mh.features.lbp(region_mask, 1, 8)




############################################################################################
def find_region_props_patch(region_label_dict, region_type):
    """
    find_region_props_patch computes a variety of morphological features for the provided tissue type
    using skimage's regionprops function

    param: region_label_dict, region_type
    return: arrays with numerous morphological features describing all regions of the provided tissue type
    """

    #obtain a grayscale mask of all regions belonging to the coresponding tissue type
    region_labels = region_label_dict['labeled_' + region_type]
    og_img = region_label_dict['og_img']
    region_mask = np.copy(og_img)
    region_mask[region_labels == 0] = 0
    region_mask = mh.colors.rgb2grey(region_mask)
    region_mask = region_mask.astype(np.uint8)

    #use skimage's regionprops function to obtain various morphological features describing the provided regions
    region_props = regionprops(region_mask)

    #initialize feature arrays, where one entry will correspond to the feature value(s) for one region
    equivalent_diameter_array = np.empty([0])
    euler_number_array = np.empty([0])
    convex_area_array = np.empty([0])
    eccentricity_array = np.empty([0])
    extent_array = np.empty([0])
    inertia_tensor_00_array = np.empty([0])
    inertia_tensor_01_array = np.empty([0])
    inertia_tensor_10_array = np.empty([0])
    inertia_tensor_11_array = np.empty([0])
    inertia_tensor_eigvals_0_array = np.empty([0])
    inertia_tensor_eigvals_1_array = np.empty([0])
    major_axis_length_array = np.empty([0])
    orientation_array = np.empty([0])
    solidity_array = np.empty([0])
    moment_00_array = np.empty([0])
    moment_01_array = np.empty([0])
    moment_02_array = np.empty([0])
    moment_10_array = np.empty([0])
    moment_11_array = np.empty([0])
    moment_12_array = np.empty([0])
    moment_20_array = np.empty([0])
    moment_21_array = np.empty([0])
    moment_22_array = np.empty([0])


    #fill the feature arrays with the corresponding feature values from all regions
    for region in region_props:
        equivalent_diameter_array = np.append(equivalent_diameter_array, [region.equivalent_diameter])
        euler_number_array = np.append(euler_number_array, [region.euler_number])
        convex_area_array = np.append(convex_area_array, [region.convex_area])
        eccentricity_array = np.append(eccentricity_array, [region.eccentricity])
        extent_array = np.append(extent_array, [region.extent])
        inertia_tensor_00_array = np.append(inertia_tensor_00_array, [region.inertia_tensor[0,0]])
        inertia_tensor_01_array = np.append(inertia_tensor_01_array, [region.inertia_tensor[0,1]])
        inertia_tensor_10_array = np.append(inertia_tensor_10_array, [region.inertia_tensor[1,0]])
        inertia_tensor_11_array = np.append(inertia_tensor_11_array, [region.inertia_tensor[1,1]])
        inertia_tensor_eigvals_0_array = np.append(inertia_tensor_eigvals_0_array, [region.inertia_tensor_eigvals[0]])
        inertia_tensor_eigvals_1_array = np.append(inertia_tensor_eigvals_1_array, [region.inertia_tensor_eigvals[1]])
        major_axis_length_array = np.append(major_axis_length_array, [region.major_axis_length])
        orientation_array = np.append(orientation_array, [region.orientation])
        solidity_array = np.append(solidity_array, [region.solidity])
        moment_00_array = np.append(moment_00_array, [region.moments[0,0]])
        moment_01_array = np.append(moment_01_array, [region.moments[0,1]])
        moment_02_array = np.append(moment_02_array, [region.moments[0,2]])
        moment_10_array = np.append(moment_10_array, [region.moments[1,0]])
        moment_11_array = np.append(moment_11_array, [region.moments[1,1]])
        moment_12_array = np.append(moment_12_array, [region.moments[1,2]])
        moment_20_array = np.append(moment_20_array, [region.moments[2,0]])
        moment_21_array = np.append(moment_21_array, [region.moments[2,1]])
        moment_22_array = np.append(moment_22_array, [region.moments[2,2]])


    return equivalent_diameter_array, euler_number_array, convex_area_array, eccentricity_array, extent_array, inertia_tensor_00_array, inertia_tensor_01_array, inertia_tensor_10_array, inertia_tensor_11_array, inertia_tensor_eigvals_0_array, inertia_tensor_eigvals_1_array, major_axis_length_array, orientation_array, solidity_array, moment_00_array, moment_01_array, moment_02_array, moment_10_array, moment_11_array, moment_12_array, moment_20_array, moment_21_array, moment_22_array




############################################################################################
def find_region_props_seg(seg_patch):
    """
    find_region_props_seg computes a variety of morphological features on the segmentation result
    using skimage's regionprops function

    param: seg_patch
    return: arrays with numerous morphological features describing the segmentation result
    """

    #use skimage's regionprops function to obtain various morphological features describing the tissue regions in the segmentation result
    region_props = regionprops(seg_patch)

    #initialize feature arrays, where one entry will correspond to the feature value(s) for one region
    equivalent_diameter_array = np.empty([0])
    euler_number_array = np.empty([0])
    convex_area_array = np.empty([0])
    eccentricity_array = np.empty([0])
    extent_array = np.empty([0])
    inertia_tensor_00_array = np.empty([0])
    inertia_tensor_01_array = np.empty([0])
    inertia_tensor_10_array = np.empty([0])
    inertia_tensor_11_array = np.empty([0])
    inertia_tensor_eigvals_0_array = np.empty([0])
    inertia_tensor_eigvals_1_array = np.empty([0])
    major_axis_length_array = np.empty([0])
    orientation_array = np.empty([0])
    solidity_array = np.empty([0])
    moment_00_array = np.empty([0])
    moment_01_array = np.empty([0])
    moment_02_array = np.empty([0])
    moment_10_array = np.empty([0])
    moment_11_array = np.empty([0])
    moment_12_array = np.empty([0])
    moment_20_array = np.empty([0])
    moment_21_array = np.empty([0])
    moment_22_array = np.empty([0])

    # fill the feature arrays with the corresponding feature values from all regions
    for region in region_props:
        equivalent_diameter_array = np.append(equivalent_diameter_array, [region.equivalent_diameter])
        euler_number_array = np.append(euler_number_array, [region.euler_number])
        convex_area_array = np.append(convex_area_array, [region.convex_area])
        eccentricity_array = np.append(eccentricity_array, [region.eccentricity])
        extent_array = np.append(extent_array, [region.extent])
        inertia_tensor_00_array = np.append(inertia_tensor_00_array, [region.inertia_tensor[0,0]])
        inertia_tensor_01_array = np.append(inertia_tensor_01_array, [region.inertia_tensor[0,1]])
        inertia_tensor_10_array = np.append(inertia_tensor_10_array, [region.inertia_tensor[1,0]])
        inertia_tensor_11_array = np.append(inertia_tensor_11_array, [region.inertia_tensor[1,1]])
        inertia_tensor_eigvals_0_array = np.append(inertia_tensor_eigvals_0_array, [region.inertia_tensor_eigvals[0]])
        inertia_tensor_eigvals_1_array = np.append(inertia_tensor_eigvals_1_array, [region.inertia_tensor_eigvals[1]])
        major_axis_length_array = np.append(major_axis_length_array, [region.major_axis_length])
        orientation_array = np.append(orientation_array, [region.orientation])
        solidity_array = np.append(solidity_array, [region.solidity])
        moment_00_array = np.append(moment_00_array, [region.moments[0,0]])
        moment_01_array = np.append(moment_01_array, [region.moments[0,1]])
        moment_02_array = np.append(moment_02_array, [region.moments[0,2]])
        moment_10_array = np.append(moment_10_array, [region.moments[1,0]])
        moment_11_array = np.append(moment_11_array, [region.moments[1,1]])
        moment_12_array = np.append(moment_12_array, [region.moments[1,2]])
        moment_20_array = np.append(moment_20_array, [region.moments[2,0]])
        moment_21_array = np.append(moment_21_array, [region.moments[2,1]])
        moment_22_array = np.append(moment_22_array, [region.moments[2,2]])

    return equivalent_diameter_array, euler_number_array, convex_area_array, eccentricity_array, extent_array, inertia_tensor_00_array, inertia_tensor_01_array, inertia_tensor_10_array, inertia_tensor_11_array, inertia_tensor_eigvals_0_array, inertia_tensor_eigvals_1_array, major_axis_length_array, orientation_array, solidity_array, moment_00_array, moment_01_array, moment_02_array, moment_10_array, moment_11_array, moment_12_array, moment_20_array, moment_21_array, moment_22_array




############################################################################################
def fill_feature_dict_patch(feature_dict, og_img, seg, binary_cell_seg, region_type):
    """
    fill_feature_dict_patch aggregates the features over all patches in the WSI and then averages them

    param: feature_dict, og_img, seg, binary_cell_seg, region_type
    return: dictionary with all the WSI-level morphological and texture features
    """
    
    #initialize arrays containing aggregate feature values
    area_array_total = np.empty([0])
    perim_array_total = np.empty([0])
    r_array_total = np.empty([0])
    g_array_total = np.empty([0])
    b_array_total = np.empty([0])
    cell_area_array_total = np.empty([0])
    cell_perim_array_total = np.empty([0])
    cell_r_array_total = np.empty([0])
    cell_g_array_total = np.empty([0])
    cell_b_array_total = np.empty([0])
    num_cells_array_total = np.empty([0])
    cell_density_array_total = np.empty([0])
    array_haralick_total = np.zeros([4, 13])
    array_lbp_total = np.zeros([36])
    total_num_regions = 0

    equivalent_diameter_array_total = np.empty([0])
    euler_number_array_total = np.empty([0])
    convex_area_array_total = np.empty([0])
    eccentricity_array_total = np.empty([0])
    extent_array_total = np.empty([0])
    inertia_tensor_00_array_total = np.empty([0])
    inertia_tensor_01_array_total = np.empty([0])
    inertia_tensor_10_array_total = np.empty([0])
    inertia_tensor_11_array_total = np.empty([0])
    inertia_tensor_eigvals_0_array_total = np.empty([0])
    inertia_tensor_eigvals_1_array_total = np.empty([0])
    major_axis_length_array_total = np.empty([0])
    orientation_array_total = np.empty([0])
    solidity_array_total = np.empty([0])
    moment_00_array_total = np.empty([0])
    moment_01_array_total = np.empty([0])
    moment_02_array_total = np.empty([0])
    moment_10_array_total = np.empty([0])
    moment_11_array_total = np.empty([0])
    moment_12_array_total = np.empty([0])
    moment_20_array_total = np.empty([0])
    moment_21_array_total = np.empty([0])
    moment_22_array_total = np.empty([0])

    t = time.time()
    num_patches = 0
    
    #loop through all the patches in the WSI
    for i in range(int(og_img.shape[0] / 2048)):
        for j in range(int(og_img.shape[1] / 2048)):
            og_patch = og_img[i * 2048: i * 2048 + 2048, j * 2048: j * 2048 + 2048, :] #patch of WSI
            seg_patch = seg[i * 2048: i * 2048 + 2048, j * 2048: j * 2048 + 2048] #corresponding epi stromal segmentation patch
            cell_seg_patch = binary_cell_seg[i * 2048: i * 2048 + 2048, j * 2048: j * 2048 + 2048] #corresponding nuclei segmentation patch
            
            #label all the tissue regions in the patch
            region_label_dict = label_regions(og_patch, seg_patch, cell_seg_patch)
            total_num_regions = total_num_regions + region_label_dict['num_' + region_type]
            
            #obtain all the morphological and texture features for the patch
            area_array, perim_array, r_array, g_array, b_array, cell_area_array, cell_perim_array, cell_r_array, cell_g_array, cell_b_array, num_cells_array, cell_density_array, total_cells_array = find_morphological_features_patch(region_label_dict, region_type)
            array_haralick, array_lbp = find_texture_features_patch(region_label_dict, region_type)

            #obtain all the region props for the patch
            equivalent_diameter_array, euler_number_array, convex_area_array, eccentricity_array, extent_array, inertia_tensor_00_array,\
            inertia_tensor_01_array, inertia_tensor_10_array, inertia_tensor_11_array, inertia_tensor_eigvals_0_array, inertia_tensor_eigvals_1_array,\
            major_axis_length_array, orientation_array, solidity_array, moment_00_array, moment_01_array, moment_02_array,\
            moment_10_array, moment_11_array, moment_12_array, moment_20_array, moment_21_array, moment_22_array = \
            find_region_props_patch(region_label_dict, region_type)

            #aggregate all the patch-level texture features to obtain the WSI-level texture feature values
            if np.sum(array_haralick_total) == 0:
                array_haralick_total = array_haralick
            else:
                array_haralick_total = np.dstack((array_haralick_total, array_haralick))

            if np.sum(array_lbp_total) == 0:
                array_lbp_total = array_lbp
            else:
                array_lbp_total = np.dstack((array_lbp_total, array_lbp))

            #aggregate all the patch-level region props to obtain the WSI-level region prop feature values
            equivalent_diameter_array_total = np.append(equivalent_diameter_array_total, equivalent_diameter_array)
            euler_number_array_total = np.append(euler_number_array_total, euler_number_array)
            convex_area_array_total = np.append(convex_area_array_total, convex_area_array)
            eccentricity_array_total = np.append(eccentricity_array_total, eccentricity_array)
            extent_array_total = np.append(extent_array_total, extent_array)
            inertia_tensor_00_array_total = np.append(inertia_tensor_00_array_total, inertia_tensor_00_array)
            inertia_tensor_10_array_total  = np.append(inertia_tensor_10_array_total, inertia_tensor_10_array)
            inertia_tensor_01_array_total = np.append(inertia_tensor_01_array_total, inertia_tensor_01_array)
            inertia_tensor_11_array_total = np.append(inertia_tensor_11_array_total, inertia_tensor_11_array)
            inertia_tensor_eigvals_0_array_total = np.append(inertia_tensor_eigvals_0_array_total, inertia_tensor_eigvals_0_array)
            inertia_tensor_eigvals_1_array_total = np.append(inertia_tensor_eigvals_1_array_total, inertia_tensor_eigvals_1_array)
            major_axis_length_array_total = np.append(major_axis_length_array_total, major_axis_length_array)
            orientation_array_total = np.append(orientation_array_total, orientation_array)
            solidity_array_total = np.append(solidity_array_total, solidity_array)
            moment_00_array_total = np.append(moment_00_array_total, moment_00_array)
            moment_01_array_total = np.append(moment_01_array_total, moment_01_array)
            moment_02_array_total = np.append(moment_02_array_total, moment_02_array)
            moment_10_array_total = np.append(moment_10_array_total, moment_10_array)
            moment_11_array_total = np.append(moment_11_array_total, moment_11_array)
            moment_12_array_total = np.append(moment_12_array_total, moment_12_array)
            moment_20_array_total = np.append(moment_20_array_total, moment_20_array)
            moment_21_array_total = np.append(moment_21_array_total, moment_21_array)
            moment_22_array_total = np.append(moment_22_array_total, moment_22_array)


            #aggregate all the patch level morphological features to obtain the WSI-level morphological feature values
            area_array_total = np.append(area_array_total, area_array)
            perim_array_total = np.append(perim_array_total, perim_array)
            r_array_total = np.append(r_array_total, r_array)
            g_array_total = np.append(g_array_total, g_array)
            b_array_total = np.append(b_array_total, b_array)
            cell_area_array_total = np.append(cell_area_array_total, cell_area_array)
            cell_perim_array_total = np.append(cell_perim_array_total, cell_perim_array)
            cell_r_array_total = np.append(cell_r_array_total, cell_r_array)
            cell_g_array_total = np.append(cell_g_array_total, cell_g_array)
            cell_b_array_total = np.append(cell_b_array_total, cell_b_array)
            num_cells_array_total = np.append(num_cells_array_total, num_cells_array)
            cell_density_array_total = np.append(cell_density_array_total, cell_density_array)


            num_patches = num_patches + 1
            t = time.time()


    #reshape the lbp array appropriately if no feature values are successfully computed
    if (array_lbp_total.ndim == 1):
        array_lbp_total = np.reshape(array_lbp_total, (1,36))

    summary_stats = ['mean', 'median', 'std_dev', 'skew', 'kurtosis'] #list of summary statistics used to summarize patch-level features at the WSI-level
    haralick_directions = ['vert', 'horiz', 'LR', 'RL'] #list containing the four directions of the GLCM matrix used to calculate haralick texture features
    haralick_features = ['angular_second_moment', 'contrast', 'correlation', 'variance', 'inverse_difference_moment', 'sum_average', 'sum_variance', 'sum_entropy', 'entropy', 'difference_variance',
                         'difference_entropy', 'info_measure_of_correlation_1', 'info_measure_of_correlation_2'] #list containing all the different haralick texture features

    #compute summary statistics on all the morphological features at the WSI level, then put them in the WSI-level feature dictionary
    feature_dict[region_type]['num_regions'] = region_label_dict['num_' + region_type]
    feature_dict[region_type]['total_cells_array'] = total_cells_array

    compile_summary_stats_to_dict(feature_dict, region_type, summary_stats, 'area_regions', area_array_total, [])
    compile_summary_stats_to_dict(feature_dict, region_type, summary_stats, 'perimeter_regions', perim_array_total, [])
    compile_summary_stats_to_dict(feature_dict, region_type, summary_stats, 'R_val_per_region', r_array_total, [])
    compile_summary_stats_to_dict(feature_dict, region_type, summary_stats, 'G_val_per_region', g_array_total, [])
    compile_summary_stats_to_dict(feature_dict, region_type, summary_stats, 'B_val_per_region', b_array_total, [])
    compile_summary_stats_to_dict(feature_dict, region_type, summary_stats, 'nucleus_area', cell_area_array_total, [])
    compile_summary_stats_to_dict(feature_dict, region_type, summary_stats, 'nucleus_circumference', cell_perim_array_total, [])
    compile_summary_stats_to_dict(feature_dict, region_type, summary_stats, 'nucleus_R_val', cell_r_array_total, [])
    compile_summary_stats_to_dict(feature_dict, region_type, summary_stats, 'nucleus_G_val', cell_g_array_total, [])
    compile_summary_stats_to_dict(feature_dict, region_type, summary_stats, 'nucleus_B_val', cell_b_array_total, [])
    compile_summary_stats_to_dict(feature_dict, region_type, summary_stats, 'nuclei_per_region', num_cells_array_total, [])
    compile_summary_stats_to_dict(feature_dict, region_type, summary_stats, 'density_nuclei', cell_density_array_total, [])

    #compute summary statistics on all the haralick and lbp texture features at the WSI level, then put them in the WSI-level feature dictionary
    for direction_index, d in enumerate(haralick_directions):
        for index, feature in enumerate(haralick_features):
            compile_summary_stats_to_dict(feature_dict, region_type, summary_stats, d + '_' + feature, array_haralick_total[direction_index, index], ['std_dev', 'skew', 'kurtosis', 'min', 'max', 'median']) #only use mean and std deviation to summarize haralick features

    for i in range(len(array_lbp_total)):
        compile_summary_stats_to_dict(feature_dict, region_type, summary_stats, 'lbp_' + str(i + 1), array_lbp_total[0,i], [])

    #compute summary statistics on all the region props features at the WSI level, then put them in the WSI-level feature dictionary
    compile_summary_stats_to_dict(feature_dict, region_type, summary_stats, 'equivalent_diameter', equivalent_diameter_array_total, [])
    compile_summary_stats_to_dict(feature_dict, region_type, summary_stats, 'euler_number', euler_number_array_total, [])
    compile_summary_stats_to_dict(feature_dict, region_type, summary_stats, 'convex_area', convex_area_array_total, [])
    compile_summary_stats_to_dict(feature_dict, region_type, summary_stats, 'eccentricity', eccentricity_array_total, [])
    compile_summary_stats_to_dict(feature_dict, region_type, summary_stats, 'extent', extent_array_total, [])
    compile_summary_stats_to_dict(feature_dict, region_type, summary_stats, 'inertia_tensor_00', inertia_tensor_00_array_total, [])
    compile_summary_stats_to_dict(feature_dict, region_type, summary_stats, 'inertia_tensor_01', inertia_tensor_01_array_total, [])
    compile_summary_stats_to_dict(feature_dict, region_type, summary_stats, 'inertia_tensor_10', inertia_tensor_10_array_total, [])
    compile_summary_stats_to_dict(feature_dict, region_type, summary_stats, 'inertia_tensor_11', inertia_tensor_11_array_total, [])
    compile_summary_stats_to_dict(feature_dict, region_type, summary_stats, 'inertia_tensor_eigvals_0', inertia_tensor_eigvals_0_array_total, [])
    compile_summary_stats_to_dict(feature_dict, region_type, summary_stats, 'inertia_tensor_eigvals_1', inertia_tensor_eigvals_1_array_total, [])
    compile_summary_stats_to_dict(feature_dict, region_type, summary_stats, 'major_axis_length', major_axis_length_array_total, [])
    compile_summary_stats_to_dict(feature_dict, region_type, summary_stats, 'moment_00', moment_00_array_total, [])
    compile_summary_stats_to_dict(feature_dict, region_type, summary_stats, 'moment_01', moment_01_array_total, [])
    compile_summary_stats_to_dict(feature_dict, region_type, summary_stats, 'moment_02', moment_02_array_total, [])
    compile_summary_stats_to_dict(feature_dict, region_type, summary_stats, 'moment_10', moment_10_array_total, [])
    compile_summary_stats_to_dict(feature_dict, region_type, summary_stats, 'moment_11', moment_11_array_total, [])
    compile_summary_stats_to_dict(feature_dict, region_type, summary_stats, 'moment_12', moment_12_array_total, [])
    compile_summary_stats_to_dict(feature_dict, region_type, summary_stats, 'moment_20', moment_20_array_total, [])
    compile_summary_stats_to_dict(feature_dict, region_type, summary_stats, 'moment_21', moment_21_array_total, [])
    compile_summary_stats_to_dict(feature_dict, region_type, summary_stats, 'moment_22', moment_22_array_total, [])
    compile_summary_stats_to_dict(feature_dict, region_type, summary_stats, 'orientation', orientation_array_total, [])
    compile_summary_stats_to_dict(feature_dict, region_type, summary_stats, 'extent', solidity_array_total, [])




############################################################################################
def fill_feature_dict_patch_seg(feature_dict, seg):
    """
    fill_feature_dict_patch_seg aggregates features calculated directly on the segmentation result
    over all patches in the WSI and then averages them

    param: feature_dict, seg
    return: dictionary with all the WSI-level segmentation-based features
    """

    #initialize arrays containing aggregate segmentation texture feature values
    array_haralick_total = np.zeros([4,13])
    array_lbp_total = np.zeros([36])

    # initialize arrays containing aggregate segmentation region prop feature values
    equivalent_diameter_array_total = np.empty([0])
    euler_number_array_total = np.empty([0])
    convex_area_array_total = np.empty([0])
    eccentricity_array_total = np.empty([0])
    extent_array_total = np.empty([0])
    inertia_tensor_00_array_total = np.empty([0])
    inertia_tensor_01_array_total = np.empty([0])
    inertia_tensor_10_array_total = np.empty([0])
    inertia_tensor_11_array_total = np.empty([0])
    inertia_tensor_eigvals_0_array_total = np.empty([0])
    inertia_tensor_eigvals_1_array_total = np.empty([0])
    major_axis_length_array_total = np.empty([0])
    orientation_array_total = np.empty([0])
    solidity_array_total = np.empty([0])
    moment_00_array_total = np.empty([0])
    moment_01_array_total = np.empty([0])
    moment_02_array_total = np.empty([0])
    moment_10_array_total = np.empty([0])
    moment_11_array_total = np.empty([0])
    moment_12_array_total = np.empty([0])
    moment_20_array_total = np.empty([0])
    moment_21_array_total = np.empty([0])
    moment_22_array_total = np.empty([0])

    num_patches = 0

    #loop through all the patches in the WSI
    for i in range(int(seg.shape[0] / float(2048))):
        for j in range(int(seg.shape[1] / float(2048))):
            seg_patch = seg[i * 2048: i * 2048 + 2048, j * 2048: j * 2048 + 2048] #current segmentation patch

            #aggregate all the texture features corresponding to the current segmentation patch
            try:

                if(np.sum(array_haralick_total) == 0):
                    array_haralick_total = mh.features.haralick(seg_patch, ignore_zeros=True)
                else:
                    array_haralick_total = np.dstack((array_haralick_total, mh.features.haralick(seg_patch, ignore_zeros=True))) 

                if(np.sum(array_lbp_total) == 0):
                    array_lbp_total = mh.features.lbp(seg_patch, 1, 8, ignore_zeros=True)
                else:
                    array_lbp_total = np.dstack((array_lbp_total, mh.features.lbp(seg_patch, 1, 8, ignore_zeros=True)))

            except:
                if (np.sum(array_haralick_total) == 0):
                    array_haralick_total = mh.features.haralick(seg_patch)
                else:
                    array_haralick_total = np.dstack((array_haralick_total, mh.features.haralick(seg_patch)))  

                if (np.sum(array_lbp_total) == 0):
                    array_lbp_total = mh.features.lbp(seg_patch, 1, 8)
                else:
                    array_lbp_total = np.dstack((array_lbp_total, mh.features.lbp(seg_patch, 1, 8)))


            #aggregate all the region prop features corresponding to the current segmentation patch
            equivalent_diameter_array, euler_number_array, convex_area_array, eccentricity_array, extent_array, inertia_tensor_00_array, \
            inertia_tensor_01_array, inertia_tensor_10_array, inertia_tensor_11_array, inertia_tensor_eigvals_0_array, inertia_tensor_eigvals_1_array, \
            major_axis_length_array, orientation_array, solidity_array, moment_00_array, moment_01_array, moment_02_array, \
            moment_10_array, moment_11_array, moment_12_array, moment_20_array, moment_21_array, moment_22_array = \
            find_region_props_seg(seg_patch)

            equivalent_diameter_array_total = np.append(equivalent_diameter_array_total, equivalent_diameter_array)
            euler_number_array_total = np.append(euler_number_array_total, euler_number_array)
            convex_area_array_total = np.append(convex_area_array_total, convex_area_array)
            eccentricity_array_total = np.append(eccentricity_array_total, eccentricity_array)
            extent_array_total = np.append(extent_array_total, extent_array)
            inertia_tensor_00_array_total = np.append(inertia_tensor_00_array_total, inertia_tensor_00_array)
            inertia_tensor_10_array_total = np.append(inertia_tensor_10_array_total, inertia_tensor_10_array)
            inertia_tensor_01_array_total = np.append(inertia_tensor_01_array_total, inertia_tensor_01_array)
            inertia_tensor_11_array_total = np.append(inertia_tensor_11_array_total, inertia_tensor_11_array)
            inertia_tensor_eigvals_0_array_total = np.append(inertia_tensor_eigvals_0_array_total, inertia_tensor_eigvals_0_array)
            inertia_tensor_eigvals_1_array_total = np.append(inertia_tensor_eigvals_1_array_total, inertia_tensor_eigvals_1_array)
            major_axis_length_array_total = np.append(major_axis_length_array_total, major_axis_length_array)
            orientation_array_total = np.append(orientation_array_total, orientation_array)
            solidity_array_total = np.append(solidity_array_total, solidity_array)
            moment_00_array_total = np.append(moment_00_array_total, moment_00_array)
            moment_01_array_total = np.append(moment_01_array_total, moment_01_array)
            moment_02_array_total = np.append(moment_02_array_total, moment_02_array)
            moment_10_array_total = np.append(moment_10_array_total, moment_10_array)
            moment_11_array_total = np.append(moment_11_array_total, moment_11_array)
            moment_12_array_total = np.append(moment_12_array_total, moment_12_array)
            moment_20_array_total = np.append(moment_20_array_total, moment_20_array)
            moment_21_array_total = np.append(moment_21_array_total, moment_21_array)
            moment_22_array_total = np.append(moment_22_array_total, moment_22_array)

            num_patches = num_patches + 1

    ##reshape the lbp array appropriately if no feature values are successfully computed
    if (array_lbp_total.ndim == 1):
        array_lbp_total = np.reshape(array_lbp_total, (1,36))

    summary_stats = ['mean', 'median', 'std_dev', 'skew', 'kurtosis'] #list of summary statistics used to summarize patch-level features at the WSI-level
    haralick_directions = ['vert', 'horiz', 'LR', 'RL'] #list containing the four directions of the GLCM matrix used to calculate haralick texture features
    haralick_features = ['angular_second_moment', 'contrast', 'correlation', 'variance', 'inverse_difference_moment', 'sum_average', 'sum_variance', 'sum_entropy', 'entropy', 'difference_variance',
                         'difference_entropy', 'info_measure_of_correlation_1', 'info_measure_of_correlation_2'] #list containing all the different haralick texture features

    #compute summary statistics on all the haralick and lbp texture features at the WSI level, then put them in the WSI-level feature dictionary
    for direction_index, d in enumerate(haralick_directions):
        for index, feature in enumerate(haralick_features):
            compile_summary_stats_to_dict(feature_dict, 'seg', summary_stats, d + '_' + feature, array_haralick_total[direction_index, index], ['std_dev', 'skew', 'kurtosis', 'median']) #only use mean and std deviation to summarize haralick features

    for i in range(len(array_lbp_total)):
        compile_summary_stats_to_dict(feature_dict, 'seg', summary_stats, 'lbp_' + str(i + 1), array_lbp_total[0,i],[])

    #compute all summary statistics except kurtosis on all the region props features at the WSI level, then put them in the WSI-level feature dictionary
    compile_summary_stats_to_dict(feature_dict, 'seg', summary_stats, 'equivalent_diameter', equivalent_diameter_array_total,['kurtosis'])
    compile_summary_stats_to_dict(feature_dict, 'seg', summary_stats, 'euler_number', euler_number_array_total,['kurtosis'])
    compile_summary_stats_to_dict(feature_dict, 'seg', summary_stats, 'convex_area', convex_area_array_total,['kurtosis'])
    compile_summary_stats_to_dict(feature_dict, 'seg', summary_stats, 'eccentricity', eccentricity_array_total,['kurtosis'])
    compile_summary_stats_to_dict(feature_dict, 'seg', summary_stats, 'extent', extent_array_total,['kurtosis'])
    compile_summary_stats_to_dict(feature_dict, 'seg', summary_stats, 'inertia_tensor_00', inertia_tensor_00_array_total,['kurtosis'])
    compile_summary_stats_to_dict(feature_dict, 'seg', summary_stats, 'inertia_tensor_01', inertia_tensor_01_array_total,['kurtosis'])
    compile_summary_stats_to_dict(feature_dict, 'seg', summary_stats, 'inertia_tensor_10', inertia_tensor_10_array_total,['kurtosis'])
    compile_summary_stats_to_dict(feature_dict, 'seg', summary_stats, 'inertia_tensor_11', inertia_tensor_11_array_total,['kurtosis'])
    compile_summary_stats_to_dict(feature_dict, 'seg', summary_stats, 'inertia_tensor_eigvals_0', inertia_tensor_eigvals_0_array_total,['kurtosis'])
    compile_summary_stats_to_dict(feature_dict, 'seg', summary_stats, 'inertia_tensor_eigvals_1', inertia_tensor_eigvals_1_array_total,['kurtosis'])
    compile_summary_stats_to_dict(feature_dict, 'seg', summary_stats, 'major_axis_length', major_axis_length_array_total,['kurtosis'])
    compile_summary_stats_to_dict(feature_dict, 'seg', summary_stats, 'moment_00', moment_00_array_total,['kurtosis'])
    compile_summary_stats_to_dict(feature_dict, 'seg', summary_stats, 'moment_01', moment_01_array_total,['kurtosis'])
    compile_summary_stats_to_dict(feature_dict, 'seg', summary_stats, 'moment_02', moment_02_array_total,['kurtosis'])
    compile_summary_stats_to_dict(feature_dict, 'seg', summary_stats, 'moment_10', moment_10_array_total,['kurtosis'])
    compile_summary_stats_to_dict(feature_dict, 'seg', summary_stats, 'moment_11', moment_11_array_total,['kurtosis'])
    compile_summary_stats_to_dict(feature_dict, 'seg', summary_stats, 'moment_12', moment_12_array_total,['kurtosis'])
    compile_summary_stats_to_dict(feature_dict, 'seg', summary_stats, 'moment_20', moment_20_array_total,['kurtosis'])
    compile_summary_stats_to_dict(feature_dict, 'seg', summary_stats, 'moment_21', moment_21_array_total,['kurtosis'])
    compile_summary_stats_to_dict(feature_dict, 'seg', summary_stats, 'moment_22', moment_22_array_total,['kurtosis'])
    compile_summary_stats_to_dict(feature_dict, 'seg', summary_stats, 'orientation', orientation_array_total,['kurtosis'])
    compile_summary_stats_to_dict(feature_dict, 'seg', summary_stats, 'extent', solidity_array_total,['kurtosis'])





############################################################################################
def compile_summary_stats_to_dict(feature_dict, region_type, summary_stats, feature_name, feature_array, not_included_stats):
    """
    compile_summary_stats_to_dict obtains an array containing the distribution of a feature over all the patches in a WSI,
    and then computes seven different statistics (unless certain statistics are specifically excluded using the parameter
    not_included_stats in order to summarize these distributions at the whole slide level.

    param: feature_dict, region_type, summary_stats, feature_name, feature_array
    return: update dictionary with the features summarized at a whole slide level
    """

    #if the feature array is empty, set the feature value to 0
    if(feature_array.size == 0):
        feature_array = np.zeros([1])

    #fill the corresponding dictionary entries with the summarized feature values
    for s in summary_stats:
        if s not in not_included_stats:
            if s is 'mean':
                feature_dict[region_type][s + '_' + feature_name] = np.mean(feature_array)
            elif s is 'median':
                feature_dict[region_type][s + '_' + feature_name] = np.median(feature_array)
            #elif s is 'min':
             #   feature_dict[region_type][s + '_' + feature_name] = np.min(feature_array)
            #elif s is 'max':
             #   feature_dict[region_type][s + '_' + feature_name] = np.max(feature_array)
            elif s is 'std_dev':
                feature_dict[region_type][s + '_' + feature_name] = np.std(feature_array)
            elif s is 'skew':
                feature_dict[region_type][s + '_' + feature_name] = skew(feature_array)
            else:
                feature_dict[region_type][s + '_' + feature_name] = kurtosis(feature_array)




############################################################################################
def calculate_cell_bhattacharyya_distance(feature_dict):
    """
    calculate_cell_bhattacharyya distance calculates the bhattacharyya distance in order to measure the difference between the epithelial and stromal nuclei distributions throughout the entire WSI

    param: feature_dict
    return: update feature dictionary with bhattacharyya distance measure capturing the difference between epithelial and stromal nuclei counts in the WSI
    """

    #obtain the patch-based epithelial and stromal nuclei counts, then remove them from the WSI-level feature dictionary
    total_cells_epi = feature_dict['epi']['total_cells_array']
    total_cells_stroma = feature_dict['stroma']['total_cells_array']
    del feature_dict['epi']['total_cells_array']
    del feature_dict['stroma']['total_cells_array']
    #del feature_dict['fat']['total_cells_array']

    #create histograms from the epithelial and stromal nuclei count arrays
   
    if(total_cells_epi.size == 0 or total_cells_stroma.size == 0):
	feature_dict['epi_stroma_cell_bhattacharyya'] = 0

    else: 
	maximum = int(max(np.max(total_cells_epi), np.max(total_cells_stroma)))
        maximum = max(maximum, 1)
    	total_cells_epi, bin_edges = np.histogram(total_cells_epi, bins=maximum, range=(0,maximum))
    	total_cells_stroma, bin_edges = np.histogram(total_cells_stroma, bins=maximum, range=(0,maximum))

    	#compute the bhattacharyya distance between the WSI-level epithelial and stromal nuclei counts and add it to the feature dictionary
    	feature_dict['epi_stroma_cell_bhattacharyya'] = math.sqrt( 1 - ( 1 / math.sqrt(np.mean(total_cells_epi)*np.mean(total_cells_stroma)*len(total_cells_epi)**2)) * np.sum(np.sqrt(np.multiply(total_cells_epi,total_cells_stroma))))




############################################################################################
def calculate_spatial_features(feature_dict, seg):
    """
    calculate_spatial_features uses the tissue segmentation result to obtain the centroids of all the epithelial
    regions in the WSI at a 0.625x magnification level. It then constructs a voronoi diagram and delaunay triangulation
    using these centroids, and computes the area of each voronoi region, the area of each delaunay triangle, and the
    length of each edge in teh delaunay triangulation using these diagrams in order to capture the high-level
    spatial attributes of the epithelial regions in the WSI

    param: feature_dict, seg
    return: updates dictionary with the WSI-level spatial features obtained using the voronoi diagram and delaunay triangulation
    """
    try:
    	seg_25 = cv2.resize(seg, (0, 0), fx=0.0625, fy=0.0625)  # resize WSI tissue segmentation from 10x to 0.625x 
   
    except:
        shape = seg.shape
	length = shape[0]
        width = shape[1]

        
        print("image > 2^31 pixels - had to resize using grid method")

        seg_25_11 = cv2.resize(seg[:length/2, :width/2], (0,0), fx=0.0625, fy=0.0625)
        seg_25_12 = cv2.resize(seg[:length/2, width/2:], (0,0), fx=0.0625, fy=0.0625)
        seg_25_21 = cv2.resize(seg[length/2:, :width/2], (0,0), fx=0.0625, fy=0.0625)
        seg_25_22 = cv2.resize(seg[length/2:, width/2:], (0,0), fx=0.0625, fy=0.0625)

        seg_25 = np.zeros([seg_25_11.shape[0]*2, seg_25_11.shape[1]*2])
        
        new_length = seg_25.shape[0]
        new_width = seg_25.shape[1]

        seg_25[:new_length/2, :new_width/2] = seg_25_11
        seg_25[:new_length/2, new_width/2:] = seg_25_12
        seg_25[new_length/2:, :new_width/2] = seg_25_21
        seg_25[new_length/2:, new_width/2:] = seg_25_22
 
    # create a smoothed out black and white epi mask
    epithelial_region_mask = np.copy(seg_25)
    epithelial_region_mask[epithelial_region_mask != 1] = 0
    epithelial_region_mask[epithelial_region_mask == 1] = 255
    epithelial_region_mask = remove_small_objects(epithelial_region_mask == 255, 2 ** 2)  #remove small artifacts from the epi mask
    epithelial_region_mask = (epithelial_region_mask * 255).astype(np.uint8)

    # label and index all the individual epi regions
    labeled_epi, num_epi = mh.label(epithelial_region_mask)

    #use skimage's regionprops method to obtain an n x 2 matrix containing the x and y coordinates of the centroids of all n epithelial regions found in the WSI
    props = regionprops(labeled_epi)
    centroid_list = list()

    for prop in props:
        centroid_list.append(prop.centroid)

    centroid_array = np.zeros([len(centroid_list), 2])

    for c_index, c in enumerate(centroid_list):
        centroid_array[c_index] = [c[0], c[1]]

    #create a voronoi diagram using the matrix of centroids, then calculate the relative areas of all voronoi regions in the diagram
    voronoi_epi = Voronoi(centroid_array)
    voronoi_regions, voronoi_vertices = voronoi_finite_polygons_2d(voronoi_epi)
    voronoi_epi_areas = np.array([polygon_area(voronoi_vertices[i]) for i in voronoi_regions])
    voronoi_epi_areas = voronoi_epi_areas / np.sum(voronoi_epi_areas)

    #create a delaunay triangulation using the matrix of centroids, then calculate the relative areas of all delaunay triangles and the lengths of all edges in the triangulation
    delaunay_epi = Delaunay(centroid_array)
    delaunay_points = delaunay_epi.points
    delaunay_triangles_epi = delaunay_points[delaunay_epi.simplices]
    delaunay_triangles_epi_areas = np.zeros([len(delaunay_triangles_epi)])

    for tri_index, tri in enumerate(delaunay_triangles_epi):
        delaunay_triangles_epi_areas[tri_index] = 0.5 * norm(np.cross(tri[1] - tri[0], tri[2] - tri[0])) #calculate the area of each delaunay triangle

    delaunay_triangles_epi_areas = delaunay_triangles_epi_areas / np.sum(delaunay_triangles_epi_areas)

    delaunay_epi_edge_list = list()

    #aggregate all the edge lengths of each triangle in the triangulation
    for tri in delaunay_triangles_epi:
        delaunay_epi_edge_list.append(norm(tri[1] - tri[0]))
        delaunay_epi_edge_list.append(norm(tri[2] - tri[0]))
        delaunay_epi_edge_list.append(norm(tri[2] - tri[1]))

    #keep only the unique edge lengths to prevent repeats
    delaunay_epi_edge_list = list(set(delaunay_epi_edge_list))
    delaunay_epi_edge_lengths = np.asarray(delaunay_epi_edge_list)

    #compute all summary statistics on voronoi region area, delaunay triangle area, and delaunay edge length at the WSI level, then put them in the WSI-level feature dictionary
    summary_stats = ['mean', 'median', 'min', 'max', 'std_dev', 'skew', 'kurtosis']
    compile_summary_stats_to_dict(feature_dict, 'epi', summary_stats, 'voronoi_area', voronoi_epi_areas,[])
    compile_summary_stats_to_dict(feature_dict, 'epi', summary_stats, 'delaunay_triangle_area', delaunay_triangles_epi_areas,[])
    compile_summary_stats_to_dict(feature_dict, 'epi', summary_stats, 'delaunay_edge_length', delaunay_epi_edge_lengths,[])




############################################################################################
def polygon_area(polygon):
    """
    polygon_area computes the area of a polygon enclosed by the array of vertices in polygon

    param: P
    return: area of polygon enclosed by the vertices in polygon
    """

    #use the polygon area formula to calculate the area using the x and y coordinates of each vertex in polygon
    lines = np.hstack([polygon, np.roll(polygon, -1, axis=0)])
    return 0.5 * abs(sum(x1 * y2 - x2 * y1 for x1, y1, x2, y2 in lines))




############################################################################################
def voronoi_finite_polygons_2d(vor):
    """
    voronoi_finite_polygons_2d takes a voronoi diagram as input and revises it by
    removing any infinite regions. It then returns the indexed vertices of all the revised
    regions in the diagram.


    param: vor
    return: new_regions (indices of all revised vertices), new_vertices (coordinates of all revised vertices)

    @Author: Pauli Virtanen https://gist.github.com/pv/8036995
    """

    new_regions = []
    new_vertices = vor.vertices.tolist()

    center = vor.points.mean(axis=0)
    radius = vor.points.ptp().max() * 2

    # construct a map containing all ridges for a given point
    all_ridges = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        all_ridges.setdefault(p1, []).append((p2, v1, v2))
        all_ridges.setdefault(p2, []).append((p1, v1, v2))

    # reconstruct infinite regions
    for p1, region in enumerate(vor.point_region):
        vertices = vor.regions[region]

        if all(v >= 0 for v in vertices):
            # finite region
            new_regions.append(vertices)
            continue

        # reconstruct a non-finite region
        ridges = all_ridges[p1]
        new_region = [v for v in vertices if v >= 0]

        for p2, v1, v2 in ridges:
            if v2 < 0:
                v1, v2 = v2, v1
            if v1 >= 0:
                # finite ridge: already in the region
                continue

            # compute the missing endpoint of an infinite ridge
            t = vor.points[p2] - vor.points[p1]  # tangent
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])  # normal

            midpoint = vor.points[[p1, p2]].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, n)) * n
            far_point = vor.vertices[v2] + direction * radius

            new_region.append(len(new_vertices))
            new_vertices.append(far_point.tolist())

        # sort regions counterclockwise and add them to the overall list of regions
        vs = np.asarray([new_vertices[v] for v in new_region])
        c = vs.mean(axis=0)
        angles = np.arctan2(vs[:, 1] - c[1], vs[:, 0] - c[0])
        new_region = np.array(new_region)[np.argsort(angles)]

        new_regions.append(new_region.tolist())

    return new_regions, np.asarray(new_vertices)




############################################################################################
def calculate_features_by_patch(og_img, seg, binary_cell_seg):
    """
    calculate_features_by_patch calculates the morphological features, textural, region props, and spatial features
    on all tissue types as well as directly on the segmentation result. The results
    are placed in the WSI-level feature dictionary, and the percent area features for the three
    tissue types are calculated at the WSI-level and subsequently placed in the feature dictionary.

    param: og_img, seg, binary_cell_seg
    return: dictionary with all WSI-level morphological, textural, region props, and spatial features
    """

    #initialize the WSI-level feature dictionary and calculate the features for the three tissue types and the direct segmentation result
    feature_dict = get_feature_dict()
    fill_feature_dict_patch(feature_dict, og_img, seg, binary_cell_seg, 'epi')
    #fill_feature_dict_patch(feature_dict, og_img, seg, binary_cell_seg, 'fat')
    fill_feature_dict_patch(feature_dict, og_img, seg, binary_cell_seg, 'stroma')
    fill_feature_dict_patch_seg(feature_dict, seg)


    #calculate the percent epi, percent stroma, and percent fat features at the WSI level
    #total_area = feature_dict['epi']['mean_area_regions'] * feature_dict['epi']['num_regions'] + feature_dict['stroma']['mean_area_regions'] * feature_dict['stroma']['num_regions'] + feature_dict['fat']['mean_area_regions'] * feature_dict['fat']['num_regions']

    #feature_dict['epi']['percent_area'] = feature_dict['epi']['mean_area_regions'] * feature_dict['epi']['num_regions'] / total_area
    #feature_dict['stroma']['percent_area'] = feature_dict['stroma']['mean_area_regions'] * feature_dict['stroma']['num_regions'] / total_area
    #feature_dict['fat']['percent_area'] = feature_dict['fat']['mean_area_regions'] * feature_dict['fat']['num_regions'] / total_area

    return feature_dict




############################################################################################
def calculate_features_WSI(patch_dir, es_seg_dir, cell_seg_dir):
    """
    calculate_features_WSI removes all blank or blurry patches, performs postprocessing on the
    epi stromal segmentation and nuclei segmentation results, then stitches them all together
    and extracts all morphological and textural features at the WSI level. The results are
    written to a .csv file containing the feature values for the provided WSI.

    param: patch_dir, es_seg_dir, cell_seg_dir
    return: writes WSI-level features to a corresponding .csv file
    """


    start_wsi = time.time()

    start = time.time()
    #remove_blank_blurry_patches(patch_dir) #remove blank and blurry patches from the original WSI and the corresponding epi stromal and nuclei segmentation results
    print("removed blank and blurry patches. Time: ", time.time() - start)

    start = time.time()
    #postprocess_segs(patch_dir, es_seg_dir, True) #postprocess epi stromal segmentation results
    print("postprocessed epi stromal segmentations. Time:", time.time() - start)

    start = time.time()
    #postprocess_segs(patch_dir, cell_seg_dir, False) #postprocess nuclei segmentation results
    print("postprocessed nuclei detections. Time: ", time.time() - start)

    start = time.time()
    og_img_WSI = stitch_image(patch_dir) #stitch together the original WSI
    print("stitched original image. Time: ", time.time() - start)

    start = time.time()
    es_seg_WSI = stitch_seg(es_seg_dir) #stitch together the epi stromal segmentation result
    print("stitched epi stromal segmentations. Time: ", time.time() - start)
    
    start = time.time()
    cell_seg_WSI = stitch_seg(cell_seg_dir) #stitch together the nuclei segmentation result
    print("stitched nuclei detections. Time: ", time.time() - start)

    #if the stitched WSI is blank, don't calculate features
    if (es_seg_WSI.size == 0):
        print("Image size 0, so no features calculated")
        return

    #calculate all the WSI-level morphological, textural, region props, and spatial features, then write them to the corresponding CSV
    
    start = time.time()
    feature_dict = calculate_features_by_patch(og_img_WSI, es_seg_WSI, cell_seg_WSI)
    print("extracted patch features. Time: ", time.time() - start)
    
    start = time.time()
    calculate_cell_bhattacharyya_distance(feature_dict)
    print("calculated bhattacharrya distance. Time: ", time.time() - start)    

    start = time.time()
    calculate_spatial_features(feature_dict, es_seg_WSI)
    print("calculated spatial features. Time: ", time.time() - start)    

    start = time.time()
    write_csv(feature_dict, patch_dir)
    print("wrote to csv. Time: ", time.time() - start)


    end_wsi = time.time()
    print("TOTAL TIME WSI(hours): ", (end_wsi - start_wsi)/3600.0)
    print("")
    print("")

############################################################################################
def write_csv(dict, patch_dir):
    """
    write_csv writes the provided feature dictionary to a .csv file corresponding to the original WSI
    param: dict, patch_dir
    return: writes feature dictionary to a corresponding .csv file
    """


    #determine the name of the .csv file based on the directory of the original WSI
    positions = [pos for pos, char in enumerate(os.path.basename(patch_dir)) if char == '_']
    if (len(positions) == 2):
        name = os.path.basename(patch_dir)[:positions[1]]
    else:
        name = os.path.basename(patch_dir)[:positions[0]]

    csv_name = os.path.join(os.path.dirname(patch_dir), name + "_features.csv")
    print("csv name: " + csv_name)

    #write the feature dictionary to the .csv file
    with open(csv_name, 'wb') as fout:
        csv_out = csv.writer(fout)
        for row in dict['epi']:
            csv_out.writerow(['epi_' + row, dict['epi'][row]])
        for row in dict['stroma']:
            csv_out.writerow(['stroma_' + row, dict['stroma'][row]])
        #for row in dict['fat']:
            #csv_out.writerow(['fat_' + row, dict['fat'][row]])
        for row in dict['seg']:
            csv_out.writerow(['seg_' + row, dict['seg'][row]])

        csv_out.writerow(['epi_stroma_cell_bhattacharyya', dict['epi_stroma_cell_bhattacharyya']])





############################################################################################
def extract_features_from_dir(parent_dir):
    """
    extract_features_from_dir performs feature extraction on all the WSIs in the provided directory

    param: parent_dir
    return: performs feature extraction on all WSIs in parent_dir
    """

    #get the subdirectories of all the WSIs in parent_dir
    sub_dirs = get_immediate_sub_dirs(parent_dir)
    sub_dirs.sort()
    print("WSI length", len(sub_dirs))
	#TODO: Jan, please change the line below based on which desktop you are running it on. So, if we decide batch size is 200 images, first one would be sub_dirs = sub_dirs[0:], then sub_dirs[200:], and so on so forth 
    #sub_dirs = sub_dirs[:200]
    print("WSI length new", len(sub_dirs))
    print("WSI Num Cutoff", sub_dirs[0])


    #loop through all the WSIs in the parent directory
    for patch_dir in sub_dirs:
        positions = [pos for pos, char in enumerate(os.path.basename(patch_dir)) if char == '_']
     
        #determine the name of the corresponding feature .csv to check if it already exists
        if (len(positions) == 2):
            name = os.path.basename(patch_dir)[:positions[1]]
        else:
            name = os.path.basename(patch_dir)[:positions[0]]
        csv_name = os.path.join(os.path.dirname(patch_dir), name + "_features.csv")

        #if the corresponding epi stromal segmentation and nuclei segmentation results exist, continue
        if os.path.isdir(os.path.join(parent_dir, patch_dir + '_cellSeg')) and os.path.isdir(os.path.join(parent_dir, patch_dir + '_epiStromalSeg')):

            #if the feature .csv already exists, skip feature extraction on the current WSI since it has already been performed
            if os.path.isfile(os.path.join(parent_dir, csv_name)):
                print("Skipped because CSV exists: " + str(os.path.join(parent_dir, csv_name)))

            #otherwise, perform feature extraction on the current WSI
            else:
                print("Calculating Features: " + str(patch_dir))
                calculate_features_WSI(os.path.join(parent_dir, patch_dir), os.path.join(parent_dir, patch_dir + '_epiStromalSeg'), os.path.join(parent_dir, patch_dir + '_cellSeg'))




############################################################################################
def get_immediate_sub_dirs(parent_dir):
    """
    get_immediate_sub_dirs obtains a list of all the WSI subdirectories in the provided directory

    param: parent_dir
    return: list of all WSI subdirectories in parent_dir
    """

    return [name for name in os.listdir(parent_dir) if os.path.isdir(os.path.join(parent_dir, name)) and 'Seg' not in name and os.path.isdir(
            os.path.join(parent_dir, name + '_cellSeg')) and os.path.isdir( os.path.join(parent_dir, name + '_epiStromalSeg'))]





start_time = time.time()
#TODO: Jan, please change the line below to point to the directory corresponding to all of the BBD images 
extract_features_from_dir(os.path.join('/media', 'avellal14', 'Heng_Hard_Drive_21', 'BBD_Images')) 
#extract_features_from_dir(os.path.join('R:', os.sep, 'Beck Lab', 'Heng_BBD_Texture_NHS', 'BBD_NCC_Images'))
print("TOTAL TIME ELAPSED:", str(time.time() - start_time))




