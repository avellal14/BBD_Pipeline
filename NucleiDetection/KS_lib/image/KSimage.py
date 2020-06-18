import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy import misc
from scipy import ndimage
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.morphology import distance_transform_cdt
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
from skimage import feature
from skimage import transform
from skimage.feature import peak_local_max
from skimage.filters import rank
from skimage.morphology import disk, remove_small_objects
from sklearn.cluster import KMeans

from KS_lib.misc.ELSClique import ESLclique


###########################################
def imread(image_path):
    """
    read an image
    :param image_path:
    :return:
    """
    I = cv2.imread(image_path,-1)
    # I = misc.imread(image_path)
    if I.ndim == 3:
        I = I[..., ::-1]
    if I.ndim == 2:
        I = np.expand_dims(I, 2)
    return I

###########################################
def imwrite(I, save_path):
    """
    write an image to path
    :param save_path:
    :return:
    """
    if I.ndim == 3 and I.shape[2] == 1:
        I = np.squeeze(I, axis=2)

    if I.ndim == 3 and I.shape[2] == 3:
        I = I[...,::-1]

    # misc.imsave(save_path, I)
    cv2.imwrite(save_path, I)

###########################################
def imshow(I):
    plt.imshow(I)
    # plt.show()

###########################################
def distance_transform(binary_image):
    bw = np.logical_not(binary_image)
    D = ndimage.distance_transform_edt(bw)
    return D

###########################################
def distance_transfrom_chessboard(binary_image):
    bw = np.logical_not(binary_image)
    D = distance_transform_cdt(bw)
    return D

###########################################
def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

###########################################
def auto_canny(image, sigma=0.33):
    # preprocessing
    image = ndimage.gaussian_filter(image, 1)

    # compute the median of the single channel pixel intensities
    v = np.median(image)

    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = feature.canny(image, sigma = 1, low_threshold = lower, high_threshold = upper)

    # return the edged image
    return edged

###########################################
def bwlabel(bw):
    labels = ndimage.label(bw)
    return labels[0], labels[1]

###########################################
def bwperim(bw, n=4):
    """
    perim = bwperim(bw, n=4)
    Find the perimeter of objects in binary images.
    A pixel is part of an object perimeter if its value is one and there
    is at least one zero-valued pixel in its neighborhood.
    By default the neighborhood of a pixel is 4 nearest pixels, but
    if `n` is set to 8 the 8 nearest pixels will be considered.
    Parameters
    ----------
      bw : A black-and-white image
      n : Connectivity. Must be 4 or 8 (default: 8)
    Returns
    -------
      perim : A boolean image
    """

    if n not in (4,8):
        raise ValueError('mahotas.bwperim: n must be 4 or 8')
    rows,cols = bw.shape

    # Translate image by one pixel in all directions
    north = np.zeros((rows,cols))
    south = np.zeros((rows,cols))
    west = np.zeros((rows,cols))
    east = np.zeros((rows,cols))

    north[:-1,:] = bw[1:,:]
    south[1:,:]  = bw[:-1,:]
    west[:,:-1]  = bw[:,1:]
    east[:,1:]   = bw[:,:-1]
    idx = (north == bw) & \
          (south == bw) & \
          (west  == bw) & \
          (east  == bw)
    if n == 8:
        north_east = np.zeros((rows, cols))
        north_west = np.zeros((rows, cols))
        south_east = np.zeros((rows, cols))
        south_west = np.zeros((rows, cols))
        north_east[:-1, 1:]   = bw[1:, :-1]
        north_west[:-1, :-1] = bw[1:, 1:]
        south_east[1:, 1:]     = bw[:-1, :-1]
        south_west[1:, :-1]   = bw[:-1, 1:]
        idx &= (north_east == bw) & \
               (south_east == bw) & \
               (south_west == bw) & \
               (north_west == bw)
    return ~idx * bw

####################################################
def shearing(img_file):
    # Load the image as a matrix
    image = imread(img_file)

    # Create Afine transform
    afine_tf = transform.AffineTransform(shear=0.2)

    # Apply transform to image data
    modified = transform.warp(image, afine_tf)

    return modified

#####################################################
def elastic_transform(image, alpha, sigma, random_state=None):
    """Elastic deformation of images as described in [Simard2003]_.
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
       Convolutional Neural Networks applied to Visual Document Analysis", in
       Proc. of the International Conference on Document Analysis and
       Recognition, 2003.
    """

    if random_state is None:
        random_state = np.random.RandomState(None)

    if len(image.shape) == 2:
        shape = image.shape[0:2]
        dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
        dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
        x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij')
        indices = np.reshape(x + dx, (-1, 1)), np.reshape(y + dy, (-1, 1))

        image = map_coordinates(image, indices, order=1).reshape(shape)
    else:

        shape = image.shape[0:2]
        dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
        dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha

        x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij')
        indices = np.reshape(x + dx, (-1, 1)), np.reshape(y + dy, (-1, 1))

        for i in range(image.ndim):
            image[:,:,i] = map_coordinates(image[:,:,i], indices, order=1).reshape(shape)

    return image

#####################################################
def imresize(image,size,mode=None):
    # if image.ndim == 3 and image.shape[2] == 1:
    #     image = np.squeeze(image,2)
    resize_img = list()
    for i in range(image.shape[2]):
        resize_img.append(misc.imresize(image[:,:,i],size,interp='bicubic',mode=mode))

    return np.dstack(resize_img)

#####################################################
def adaptive_histeq(img):
    # create a CLAHE object (Arguments are optional).
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl = list()

    if img.ndim == 2:
        img = np.expand_dims(img, axis=2)

    for i in range(img.shape[2]):
        cl.append(clahe.apply(img[:,:,i]))

    img = np.dstack(cl)

    if img.ndim == 3 and img.shape[img.ndim-1] == 1:
        img = np.squeeze(img, axis = 2)

    return img

#####################################################
def find_local_maxima(img, min_distance, threshold_rel):
    img = np.squeeze(img)
    coordinate = peak_local_max(img, min_distance=min_distance, threshold_rel=threshold_rel)

    nparts = np.int(np.ceil(coordinate.shape[0] / 1000.0))
    kmeans = KMeans(n_clusters=nparts, n_init = 1).fit(coordinate)
    idx = kmeans.labels_

    all_coordinate = list()

    for i in range(nparts):
        temp_coordinate = coordinate[idx == i,:]

        for min_distance in np.linspace(int(min_distance/2.0),min_distance+1,3):

            dist_mat = squareform(pdist(temp_coordinate, 'euclidean'))

            dBinary = dist_mat < min_distance
            dBinary[np.eye(dBinary.shape[0]).astype(np.bool)] = 0

            m = ESLclique(dBinary)
            m1 = m.tocsc()

            nCliques = m1.shape[1]

            Ic = np.zeros((nCliques, 1))
            Jc = np.zeros((nCliques, 1))

            for i in range(nCliques):
                Im = temp_coordinate[np.squeeze(m1[:, i].toarray().astype(np.bool)),0]
                Jm = temp_coordinate[np.squeeze(m1[:, i].toarray().astype(np.bool)),1]

                Ic[i] = np.mean(Im)
                Jc[i] = np.mean(Jm)

            Ic = np.round(Ic)
            Jc = np.round(Jc)

            temp_coordinate = np.column_stack((Ic,Jc))

        all_coordinate.append(temp_coordinate)

    coordinate = np.row_stack(all_coordinate)

    return coordinate.astype(np.int)

#####################################################
def bwareaopen(mask,area_limit):

    mask = mask.astype(np.bool)
    # mask = mask.astype(np.uint8) * 255
    # im, contours, hierarchy = cv2.findContours(mask, 1, 2)
    #
    # temp_mask = np.zeros(mask.shape[:2], dtype='uint8')
    # for cnt in contours:
    #     area = cv2.contourArea(cnt)
    #     if area > area_limit:
    #         cv2.drawContours(temp_mask, [cnt], -1, 255, -1)
    temp_mask = remove_small_objects(mask, area_limit)

    return temp_mask.astype(np.bool)

#####################################################
def imdilate(bw,r):
    selem = disk(r)

    return rank.maximum(bw, selem).astype(np.bool)
    # return cv2.dilate(bw,cv2.getStructuringElement(cv2.MORPH_RECT,(5,5)),iterations = 1)

#####################################################
def imclose(bw, r):
    salem = disk(r)

    return cv2.morphologyEx(bw, cv2.MORPH_CLOSE, salem)

#####################################################
def label2idx(L):
    unq, unq_inv, unq_cnt = np.unique(L, return_inverse=True, return_counts=True)
    return dict(zip(unq,np.split(np.argsort(unq_inv), np.cumsum(unq_cnt[:-1]))))

#####################################################
def rgb2hsv(rgb):
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)

#####################################################
def hsv2rgb(hsv):
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
