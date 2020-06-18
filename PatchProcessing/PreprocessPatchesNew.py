import openslide
import os
import cv2
from PIL import Image
import numpy as np
import time
import csv


#Input: WSI to process
#Output: returns a tissue mask, white = tissue, black = non-tissue
def applyMask(filename):

   #open slide using OpenSlide, save the RGB and HLS representations of the thumbnail
   wsiOG = openslide.open_slide(filename)
   wsiThmbnl = wsiOG.read_region((0, 0), 9, wsiOG.level_dimensions[9])
   wsiThmbnl = cv2.cvtColor(np.asarray(wsiThmbnl), cv2.COLOR_RGBA2RGB)
   wsiHLS = cv2.cvtColor(wsiThmbnl, cv2.COLOR_RGB2HLS)

   wsiThmbnl = wsiThmbnl[:,:,1] #keep the Green channel of RGB thumbnail
   hlsMask = wsiHLS[:,:,0] #keep the Hue channel of HLS thumbnail

   #everything with Green < 220 and Hue > 130 is tissue (and everything not fitting these criteria are non-tissue)
   gMask = wsiThmbnl < 220
   hMask = hlsMask > 130
   hlsMask = gMask & hMask

   #turn mask from boolean into black and white
   hlsMask = np.uint8(hlsMask)
   hlsMask[hlsMask == 1] = 255

   #use morphological operations to smooth out the mask and get rid of inconsistencies
   kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (24, 24))
   hlsMask = cv2.morphologyEx(hlsMask, cv2.MORPH_CLOSE, kernel)
   return hlsMask


#Input: patch of WSI
#Output: boolean, whether the patch contains at least 2.5% tissue
def patchContainsTissue(wsiThmbnl):

    wsiHLS = cv2.cvtColor(wsiThmbnl, cv2.COLOR_RGB2HLS)
    wsiThmbnl = wsiThmbnl[:,:,1]  #keep the Green channel of the RGB patch
    hlsMask = wsiHLS[:,:,0]       #keep the Hue channel of the HLS patch

    # everything with Green < 220 and Hue > 130 is tissue (and everything not fitting these criteria are non-tissue)
    gMask = wsiThmbnl < 220
    hMask = hlsMask > 130

    # create binary mask with all "tissue" pixels being those that satisfy both of the above conditions
    hlsMask = gMask & hMask
    hlsMask = np.uint8(hlsMask)

    #if 2.5% or more of the patch contains tissue, then return true
    [length, width] = hlsMask.shape

    if(sum(sum(hlsMask)) * 40 < (length*width)):
        return False
    else:
        return True

#Input: The directory for the WSI image, and the parent directory for the patch outputs
#Output: Saves all the patches in the corresponding patch directory
def preprocessImage(filename, destination):
   print("Beginning processing " + filename)

   #opens the whole slide and creates the patch directory
   wsi = openslide.open_slide(filename)
   os.makedirs(destination)

   #obtains the mask using the helper function
   mask = applyMask(filename)
   maskFile = os.path.join(destination, '_mask.png')
   maskSaved = Image.fromarray(mask)
   maskSaved.save(maskFile)

   #obtains the thumbnail using the helper function
   thumbnailFile = os.path.join(destination, '_thumbnail.png')
   thumbnailSaved = wsi.read_region((0, 0), 9, wsi.level_dimensions[9])
   thumbnailSaved.save(thumbnailFile)

   #loops through 4 x 4 patches of the thumbnail, which correspond to 2048 x 2048 patches at a magnification of 40x
   #each patch that has more than 2.5% of tissue is saved
   [length, width] = mask.shape
   i = 0
   while (i < length):
       j = 0
       while (j < width):
           patch = mask[i: i + 16, j: j + 16] #16 x 16 patch * (2^9, 2^9) = (8192 x 8192) @40x
           patchSum = np.sum(patch)
           #if the patch has some tissue, extract it from the whole slide at 40x (size = 2048 x 2048 pixels)
           if (patchSum > 0):
              #4 x 4 times (512 x 512) --> 2048 x 2048 at 0 zoom out
              #also --> 512 (or 2^[9-0]) is the scale factor btwn thumbnail and actual WSI
              #4 x 4 in thumbnail = 2048 x 2048 @40x --> 16 x 16 thumbnail = 8192 x 8192 @40x = 2048 x 2048 @10x
               currentPatch = wsi.read_region((j * 512, i * 512), 2, (2048, 2048)) #mag levels: 0 --> 40x, 1 --> 20x, 2 --> 10x.
               currentPatch = cv2.cvtColor(np.asarray(currentPatch), cv2.COLOR_RGBA2RGB)

               #if the patch has at least 2.5% tissue, then save the patch as a png image
               if (patchContainsTissue(currentPatch)):
                   currentPatch = Image.fromarray(currentPatch)
                   newFilename = os.path.join(destination, 'patch_' + str(i) + '_' + str(j) + '.png')
                   currentPatch.save(newFilename)

           j = j + 16
       i = i + 16

   print("Finished processing", filename)


#Input: Directory of CSV file with images, Parent directory in which to save all the patches
#Output: All of the patches will be saved in their correspondingly named folders located in the parent directory
def processCSV(csvLocation, destinationDirectory):
   t = time.time()

   #create parent directory if it does not exist
   if not os.path.exists(destinationDirectory):
       os.mkdir(destinationDirectory)

   #read the CSV file, store it internally as a list
   outcomeCSV = open(csvLocation, 'rb')
   reader = csv.reader(outcomeCSV)
   patientList = list()
   for row in reader:
       patientList.append(row)
 
   patientList = patientList[1:] #remove the header row
   print("length patient list", len(patientList))

   letterArray = ['A', 'B', 'C', 'D'] #this array is simply for processing the image names (some WSIs have A,B,C,D instead of 1,2,3,4)

   i = 0 #number of CSV lines processed
   trueCounter = 0 #number of patients that are processed
   totalNumImagesProcessed = 0 #number of WSIs that are processed

   #########################
   #TODO: Jan, please set the start (leave at 0) and end indices as you would like to run batches
   imageStartIndex = 0 #start index (inclusive)
   imageEndIndex =  200 #end index (exclusive)
   #########################

   #TODO: IMPORTANT LINE HERE
   patientList = patientList[imageStartIndex:imageEndIndex]





   i = 0
   while(i < len(patientList)):

       #make sure that the line of the CSV contains any information, otherwise loop again
       try:
           currentPatient = patientList[i]
       except:
           continue

       print("current patient", currentPatient)
       #if a number of images is listed for the patient, process the images. Otherwise, move to the next patient
       if(len(currentPatient[3])>0):
           numImages = int(currentPatient[3])
           mrxsFileList = os.listdir(currentPatient[4])
           mrxsFileList = [f for f in mrxsFileList if os.path.isfile(os.path.join(currentPatient[4], f)) and currentPatient[0] in f]
 #PRO STRATS --> grab first n images with patients name
           if(len(mrxsFileList) > numImages):
                mrxsFileList = mrxsFileList[0:numImages]

           for j in range(len(mrxsFileList)):
               currentFilePath = os.path.join(currentPatient[4], mrxsFileList[j])
               destination = os.path.join(destinationDirectory, mrxsFileList[j][0:len(mrxsFileList[j]) - 5] + '_patches')

               #if the image has not already been processed, then process it
               if os.path.exists(currentFilePath) and not os.path.exists(destination):
                       preprocessImage(currentFilePath, destination)
                   #numImagesProcessed = numImagesProcessed + 1 #increment the number of images processed for the patient


       i = i + 1 #add 1 to the number of CSV lines processed
   print("TOTAL TIME ELAPSED: " + str(time.time() - t)) #print total time elapsed



###################
#TODO: Jan, please input the appropriate information here. First directory should be location of CSV file(should already be correct). Second directory is where you want to save all the patches (I have set it to C:/WSI_Patches on the local machine, but you can change it)
processCSV(os.path.join('R:', os.sep, 'Beck Lab', 'HENG_BBD_Texture_NHS', 'BBD_NCC_modifiedWithPaths_tocreatepatches40x_15Mar18.csv'), os.path.join('R:', os.sep, 'Beck Lab', 'HENG_BBD_Texture_NHS', 'BBD_NCC_10xExtraction_40x'))
###################


#processCSV(os.path.join('C:', os.sep, 'Adithya', 'BIDMC', 'Test_Images_12-21', 'BBD_NCC_modifiedWithPaths_tocreatepatches_part3_40x.csv'), os.path.join('C:', os.sep, 'Adithya', 'BIDMC', 'WSI_Patches')) #'R:\Beck Lab\HENG_BBD_Texture_NHS\BBD_NCC_Covariate_Outcome_KK_JH_modifiedWithPaths.csv')




