"""
StitchPatches.py

This file contains the functions for stitching together both the original image patches and the segmentation
result patches in order to form the original WSI and the WSI-level segmentation results.
"""

import os
import numpy as np
import cv2


############################################################################################
def stitch_seg(dir):
    """
    stitch_seg takes in a directory of segmentation result patches, reads in each patch
    as a grayscale image, then stitches all the patches together and returns the stitched image.

    param: dir
    return: stitched segmentation result
    """

    #obtain a list of all the patch files in the directory
    patch_list = os.listdir(dir)
    patch_list = [patch for patch in patch_list if 'patch' in patch]
    patch_list.sort()
    print("Num Patches: ", len(patch_list))


    #we have to loop through all the patches in the WSI to find the patch with the maximum column value
    max_row = 0
    max_col = 0
    for patch in patch_list:
        indices_ = [pos for pos, char in enumerate(patch) if char == '_']
        indices_period = [pos for pos, char in enumerate(patch) if char == '.']
        current_row = int(patch[indices_[0]+1:indices_[1]])
        current_col = int(patch[indices_[1]+1:indices_period[0]])
       
        if(current_row > max_row): max_row = current_row
	if(current_col > max_col): max_col = current_col


    #use the maxium row and maximum column values to create an array of the appropriate size for the stitched image
    final_img = np.zeros([max_row*128+2048, max_col*128+2048], np.uint8) #patches are indexed in increments of 16, where each increase in 16 corresponds to 2048 more pixels in either the row or column direction
 
    i = 0
    #loop through all the patches in the list and store them in the appropriate location of the stitched image array
    for patch in patch_list:
        #determine the row and column value of the current patch
        indices_ = [pos for pos, char in enumerate(patch) if char == '_']
        indices_period = [pos for pos, char in enumerate(patch) if char == '.']

        row = int(patch[indices_[0]+1:indices_[1]])
        col = int(patch[indices_[1]+1:indices_period[0]])


        seg_patch = cv2.imread(os.path.join(dir, patch_list[i]), 0)
        
	if seg_patch is not None:
		#use the row and column values to place the patch in the appropriate part of the stitched image
        	final_img[row*128: row*128 + 2048, col*128: col*128 + 2048] = seg_patch

        i += 1
    
    return final_img


############################################################################################
def stitch_image(dir):
    """
    stitch_image takes in a directory of original image patches, reads in each patch
    as a BGR image, then stitches all the patches together and returns the stitched image in RGB form.

    param: dir
    return: stitched original WSI
    """

    #obtain a list of all the patch files in the directory
    print("Stitch Seg Dir: " + str(dir))
    patch_list = os.listdir(dir)
    patch_list = [patch for patch in patch_list if 'patch' in patch]
    patch_list.sort()
    print("Num Patches: ", len(patch_list))

    #we know that the last patch in the sorted list will have the maximum row value of all patches in the WSI
    final_patch = patch_list[len(patch_list)-1]
    indices_ = [pos for pos, char in enumerate(final_patch) if char == '_']

    #we have to loop through all the patches in the WSI to find the patch with the maximum column value
    max_row = 0
    max_col = 0
    for patch in patch_list:
        indices_ = [pos for pos, char in enumerate(patch) if char == '_']
        indices_period = [pos for pos, char in enumerate(patch) if char == '.']
        current_row = int(patch[indices_[0]+1:indices_[1]])
        current_col = int(patch[indices_[1]+1:indices_period[0]])
       
        if(current_row > max_row): max_row = current_row
	if(current_col > max_col): max_col = current_col

    #use the maxium row and maximum column values to create an array of the appropriate size for the stitched image
    final_img = np.zeros([max_row*128+2048, max_col*128+2048,3], np.uint8) #patches are indexed in increments of 16, where each increase in 16 corresponds to 2048 more pixels in either the row or column direction

    i = 0
    #loop through all the patches in the list and store them in the appropriate location of the stitched image array
    for patch in patch_list:
        
	#determine the row and column value of the current patch
        indices_ = [pos for pos, char in enumerate(patch) if char == '_']
        indices_period = [pos for pos, char in enumerate(patch) if char == '.']

        row = int(patch[indices_[0]+1:indices_[1]])
        col = int(patch[indices_[1]+1:indices_period[0]])

        #use the row and column values to place the patch in the appropriate part of the stitched image
        bgr = cv2.imread(os.path.join(dir, patch_list[i]))
        if bgr is not None:
		final_img[row*128: row*128 + 2048, col*128: col*128 + 2048, :] = bgr[...,::-1] #invert the patches from BGR to RGB
        
        i += 1

    return final_img
