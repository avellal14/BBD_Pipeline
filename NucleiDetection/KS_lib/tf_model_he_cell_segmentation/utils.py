"""
utils.py

This file contains a utility function for calculating the pixelwise
mean and variance for a set of images.
"""

from KS_lib.image import KSimage
import numpy as np

##############################################################################
def calculate_mean_variance_image(list_images):
    """
    calculate_mean_variance_image simply the mean
    and variance for a list of images at each pixel
    position

    param: list_images
    return: mean_image, variance_image
    """

    image = KSimage.imread(list_images[0])

    if np.random.randint(2, size=1) == 1:
        image = np.flipud(image)
    if np.random.randint(2, size=1) == 1:
        image = np.fliplr(image)
    image = np.float32(image)

    mean_image = image
    variance_image = np.zeros(shape=image.shape, dtype=np.float32)

    for t, image_file in enumerate(list_images[1:]):
        image = KSimage.imread(image_file)


        if np.random.randint(2, size=1) == 1:
            image = np.flipud(image)
        if np.random.randint(2, size=1) == 1:
            image = np.fliplr(image)
        image = np.float32(image)

        mean_image = (np.float32(t + 1) * mean_image + image) / np.float32(t + 2)

        variance_image = np.float32(t + 1) / np.float32(t + 2) * variance_image \
                         + np.float32(1) / np.float32(t + 1) * ((image - mean_image) ** 2)

        print('calculate mean and variance: processing %d out of %d' % (t + 2, len(list_images)))
    return mean_image, variance_image