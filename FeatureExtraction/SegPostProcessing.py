"""
SegPostProcessing.py

This file contains the functions for postprocessing the segmentation results and removing any blurry patches to prevent added noise in the
feature extraction stage.
"""

import os
import glob
import numpy as np
import cv2
from scipy import stats
from skimage.morphology import remove_small_objects
from skimage.filters import threshold_otsu

############################################################################################
def postprocess_nuclei(og_img_file, seg_file):
    """
    postprocess_nuclei uses the segmentation result and the corresponding original image to remove
    noise from the segmentation result

    param: og_img_file
    param: seg_file
    return: postprocessed nuclei segmentation result
    """

    #read in the original image and the segmentation result
    try:
        og_img = cv2.imread(og_img_file)
        seg = cv2.imread(seg_file)

    except:
        if(os.path.isfile(og_img_file)):
           os.remove(og_img_file)

        if(os.path.isfile(seg_file)):
           os.remove(seg_file)

        return


    hsv_image = cv2.cvtColor(og_img, cv2.COLOR_BGR2HSV)  #convert from BGR to HSV
    h, s, v = cv2.split(hsv_image)  #split the channels
    h_mask = h > threshold_otsu(h)  #perform otsu thresholding on h channel
    s_mask = s > threshold_otsu(s)  #perform otsu thresholding on s channel
    mask = np.logical_and(h_mask, s_mask)  #combining the mask with logical AND

    #make a copy of the current segmentation result to be post-processed
    postprocessed_seg = np.copy(seg)

    #set all the background areas in the otsu thresholding mask to background in the segmentation result
    postprocessed_seg[mask == 0] = 0

    #turn the segmentation result into a 1D boolean mask
    postprocessed_seg[postprocessed_seg != 0] = 1
    postprocessed_seg = postprocessed_seg[:, :, 0]
    postprocessed_seg = postprocessed_seg.astype(np.bool)

    #remove any connected components that are less than 10 pixels
    postprocessed_seg = remove_small_objects(postprocessed_seg, min_size=10)

    #perform a morphological dilation operation to smooth out the mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    postprocessed_seg = cv2.morphologyEx(np.uint8(postprocessed_seg), cv2.MORPH_DILATE, kernel)

    #make the segmentation result black and white
    postprocessed_seg[postprocessed_seg != 0] = 255

    return postprocessed_seg




############################################################################################
def postprocess_epi_stroma(og_img_file, seg_file):
    """
    postprocess_epi uses the segmentation result and the corresponding original image to remove
    noise from the segmentation result

    param: og_img_file
    param: seg_file
    return: postprocessed epi stromal segmentation result
    """

    #read in the original image and the segmentation result
    try:
	og_img = cv2.imread(og_img_file)
   	seg = cv2.imread(seg_file)

    except:
        if(os.path.isfile(og_img_file)):
           os.remove(og_img_file)

        if(os.path.isfile(seg_file)):
	   os.remove(seg_file)
		
        return

    #pad the original image or segmentation result in case the shapes do not match up perfectly
    if (seg.shape[0] <= og_img.shape[0]):
        seg_copy = np.zeros(og_img.shape)
        [length, width, height] = seg.shape
        seg_copy[:length, :width, :height] = seg
        seg = seg_copy

    else:
        og_img_copy = np.zeros(seg.shape)
        [length, width, height] = og_img.shape
        og_img_copy[:length, :width, :height] = og_img
        og_img = og_img_copy


    hsv_image = cv2.cvtColor(og_img, cv2.COLOR_BGR2HSV)  # convert from BGR to HSV
    h, s, v = cv2.split(hsv_image)  # split the channels
    h_mask = h > threshold_otsu(h)  # perform otsu thresholding on h channel
    s_mask = s > threshold_otsu(s)  # perform otsu thresholding on s channel
    mask = np.logical_and(h_mask, s_mask)  # combining the mask with logical AND

    #use otsu thresholding to create a mask of all the background pixels in the segmentation result
    mask = (mask == 0)

    # make a 1D copy of the current segmentation result to be post-processed
    postprocessed_seg = np.copy(seg)
    postprocessed_seg = postprocessed_seg[:, :, 0]

    #create binary masks for each of the 4 tissue classes
    bg_mask = (mask == 0)
    epi_mask = (postprocessed_seg == 1)
    stroma_mask = (postprocessed_seg == 2)
    fat_mask = (postprocessed_seg == 3)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))

    # fix up the background mask with the morphological close operation, then overlay the results with the multi-class segmentation
    bg_mask = cv2.morphologyEx(np.uint8(bg_mask), cv2.MORPH_CLOSE, kernel)
    postprocessed_seg[bg_mask == False] = 0

    # fix up the epithelial mask with the morphological open operation, then overlay the results with the multi-class segmentation
    epi_mask = cv2.morphologyEx(np.uint8(epi_mask), cv2.MORPH_OPEN, kernel)
    postprocessed_seg[epi_mask == True] = 85

    # fix up the stroma mask by identifying areas outside the epithelium which are identified as background by the otsu mask but actually belong to the stroma class, then overlay the results with the multi-class segmentation
    stroma_mask = np.logical_and(stroma_mask, postprocessed_seg == 0, epi_mask == False)
    postprocessed_seg[stroma_mask == True] = 170

    # fix up the fat mask by identifying areas outside the epithelium which are identified as background by the otsu mask but actually belong to the fat class, then overlay the results with the multi-class segmentation
    fat_mask = np.logical_and(fat_mask, postprocessed_seg == 0, epi_mask == False)
    postprocessed_seg[fat_mask == True] = 255

    #smooth a 20 pixel-wide border of the segmentation result by setting each pixel's value to the mode of its corresponding 5-by-1 pixel neighborhood
    [length, width] = postprocessed_seg.shape

    for i in range(width / 5):
        postprocessed_seg[0:20, 5 * i:5 * i + 5] = (stats.mode(postprocessed_seg[20:40, 5 * i:5 * i + 5], axis=0))[0] * np.ones([20, 5])
        postprocessed_seg[length - 20:, 5 * i:5 * i + 5] = (stats.mode(postprocessed_seg[length - 40:length - 20, 5 * i:5 * i + 5], axis=0))[0] * np.ones([20, 5])

    for i in range(length / 5):
        postprocessed_seg[5 * i:5 * i + 5, 0:20] = (stats.mode(postprocessed_seg[5 * i:5 * i + 5, 20:40], axis=0))[0] * np.ones([5, 20])
        postprocessed_seg[5 * i:5 * i + 5, width - 20:] = (stats.mode(postprocessed_seg[5 * i:5 * i + 5, width - 40:width - 20], axis=0))[0] * np.ones([5, 20])

    #transform the pixel values back to the original labels of 1 for epi, 2 for stroma, and 3 for fat (0 is still background)

    postprocessed_seg[postprocessed_seg == 85] = 1
    postprocessed_seg[postprocessed_seg == 170] = 2
    postprocessed_seg[postprocessed_seg == 255] = 3

    return postprocessed_seg




############################################################################################
def postprocess_segs(og_dir, result_dir, is_epi):
    """
    postprocess_segs performs the postprocessing operation on all the segmentation results
    in the provided directory and then saves these postprocessed results in place of the original
    segmentations

    param: og_dir
    param: result_dir
    return: saved postprocessed epi stromal or nuclei segmentation results
    """

    #obtain the names of all the segmentation result files in the directory
    filename_list = glob.glob(os.path.join(result_dir, '*.png'))

    #iterate through all the segmentation results in the directory, postprocess them, and save them in place of the original results
    for file in filename_list:

        if(is_epi):
            seg = postprocess_epi_stroma(os.path.join(og_dir, os.path.basename(file)), file)
        else:
            seg = postprocess_nuclei(os.path.join(og_dir, os.path.basename(file)), file)

        #if problems occur reading files, delete original image and seg result
        if seg is not None:
            cv2.imwrite(file, seg)




############################################################################################
def remove_blank_blurry_patches(patch_dir):
    """
    remove_blank_blurry_patches iterates through all the patches in the provided
    directory and removes any patches(and their corresponding epi stromal and nuclei segmentations) that are either completely blank or blurry.

    param: patches_dir
    """

    #obtain the names of all the patch files in the directory and sort them
    patch_list = os.listdir(patch_dir)
    patch_list = [patch for patch in patch_list if 'patch' in patch]
    patch_list.sort()

    #iterate through all the patches in the directory
    for patch in patch_list:

        
	patch_im = cv2.imread(os.path.join(patch_dir, patch)) #read in the image in BGR format
        
	try:
		grayscale_patch = cv2.cvtColor(patch_im, cv2.COLOR_BGR2GRAY) #read in the image in grayscale

	except:
                #if it cannot be read, remove the patch and the corresponding epi stromal and nuclei segmentation results
		os.remove(os.path.join(patch_dir, patch))
                if os.path.exists(os.path.join(patch_dir + '_epiStromalSeg', patch)):
			os.remove(os.path.join(patch_dir + '_epiStromalSeg', patch))

                if os.path.exists(os.path.join(patch_dir + '_cellSeg', patch)):
			os.remove(os.path.join(patch_dir + '_cellSeg', patch))
             
		return 

        #if the average RGB pixel value is less than 10, the image is blank, and if the laplacian variance of the grayscale is less than 40, the image is blurry
        if(np.sum(patch_im) < 10 * np.size(patch_im) or cv2.Laplacian(grayscale_patch, cv2.CV_64F).var() < 40):
            print("removing patch because it is blank or blurry: ", patch)
            os.remove(os.path.join(patch_dir, patch)) #remove the blurry image

            #remove the corresponding epi stromal segmentation and nuclei segmentation results
            if os.path.exists(os.path.join(patch_dir + '_epiStromalSeg', patch)):
                os.remove(os.path.join(patch_dir + '_epiStromalSeg', patch))

            if os.path.exists(os.path.join(patch_dir + '_cellSeg', patch)):
                os.remove(os.path.join(patch_dir + '_cellSeg', patch))


