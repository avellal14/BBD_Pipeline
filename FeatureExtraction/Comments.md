- Strange small spurious pixels around individual nuclei in nuclear segmentation results. I have included lines in `labelRegions` function to exclude these artefacts, i.e.,

  ```
  cellSeg = remove_small_objects(cellSeg == 255, 10)
  cellSeg = np.array(cellSeg, np.uint8)
  ```

- `labelRegions` function in `SegmentationFeatures.py` doesn't seem to do what it's supposed to do. The values of pixels in `segEpi` variable are [0, 85, 170, 255]. 

  ```
  segEpi[segEpi != 85] = 0
  segEpi[segEpi == 85] = 255
  segEpi = remove_small_objects(segEpi == 255, 10**2)
  segEpi = (segEpi * 255).astype(np.uint8)

  segStroma[segStroma != 170] = 0
  segStroma[segStroma == 170] = 255
  segStroma = remove_small_objects(segStroma == 255, 10**2)
  segStroma = (segStroma * 255).astype(np.uint8)

  segFat[segFat != 255] = 0
  segFat[segFat == 255] = 255
  segFat = remove_small_objects(segFat == 255, 10**2)
  segFat = (segFat * 255).astype(np.uint8)
  ```

- `cv2.imread` read colour images as BGR so in `stitchImage` function (`StitchPatches.py`) . 

  ```
  bgr = cv2.imread(fileList[i])
  finalImg[i * 2048:(i + 1) * 2048, :, :] = bgr[...,::-1]
  ```

- `Eliminatebackground` function has been fixed.

- Not quite sure what sum of Haralick for each region mean in `fillFeatureDictPatch` function in `SegmentationFeatures.py`

  ```
  arrayHaralickF = arrayHaralickF + arrayHaralick
  ```