import numpy as np
import time
import cv2

import KS_lib.image as KSimage
from skimage.morphology import watershed

############################################################################################
def generate_label_map(dcis_mask, cell_mask, coordinate_cell_detection, prediction):

    cell_mask_bw = np.logical_and(dcis_mask == 255, np.expand_dims(cell_mask[:,:,1] > 0.3*255.0,axis=2))
    cell_mask_bw = KSimage.KSimage.bwareaopen(cell_mask_bw, area_limit = 50.0**2)
    cell_mask_bw = KSimage.KSimage.imdilate(np.squeeze(cell_mask_bw), r = 3)

    canvas = np.zeros((cell_mask_bw.shape[0],cell_mask_bw.shape[1])).astype(np.bool)
    for i in range(coordinate_cell_detection.shape[0]):
        if cell_mask_bw[coordinate_cell_detection[i, 0], coordinate_cell_detection[i, 1]]:
            canvas[coordinate_cell_detection[i, 0], coordinate_cell_detection[i, 1]] = True

    dist = KSimage.KSimage.distance_transfrom_chessboard(canvas)
    bgdist = KSimage.KSimage.distance_transfrom_chessboard(cell_mask[:,:,2] > 128)
    bgdist = np.max(dist[cell_mask_bw]) - bgdist
    dist = dist + bgdist

    canvas, nobj = KSimage.KSimage.bwlabel(canvas)
    x = watershed(dist, canvas, mask=cell_mask_bw)
    x[cell_mask_bw == 0] = 0

    # lab = np.setdiff1d(np.unique(x), 0)
    # count = np.max(lab) + 1
    # count_thresh = count
    # for i in range(len(lab)):
    #     t = time.time()
    #     mask = x == lab[i]
    #     print('time1 %f' % (time.time() - t))
    #     t = time.time()
    #     output = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8, ltype=cv2.CV_32S)
    #     nobj = output[0] - 1
    #     print('time2 %f' % (time.time() - t))
    #
    #     if nobj > 1:
    #         for j in range(1, nobj + 1):
    #             intersection = np.logical_and(obj == j, canvas)
    #             if not np.any(intersection[:]):
    #                 x[obj == j] = count
    #                 count = count + 1
    #     print i

    # lab = np.setdiff1d(np.unique(x), 0)
    # for i in range(len(lab)):
    #     if lab[i] >= count_thresh:
    #         mask = x == lab[i]
    #         dilated_mask = KSimage.imdilate(mask, r = 3)
    #         elements = x[dilated_mask]
    #         member = np.setdiff1d(elements, [lab[i], 0])
    #         if member:
    #             x[mask] = mode(member)
    #         else:
    #             x[mask] = 0

    # lab = np.setdiff1d(np.unique(x), 0)
    # for i in range(len(lab)):
    #     mask = x == lab[i]
    #     mask = KSimage.imclose(mask, r = 2)
    #     x[mask] = lab[i]

    #######################################################################

    # t = time.time()
    # coordinate_cell_detection = coordinate_cell_detection[prediction[:, 1] < 0.5, :]
    # global shared_array
    # shared_array_base = multiprocess.Array(ctypes.c_int, x.flatten())
    # shared_array = np.ctypeslib.as_array(shared_array_base.get_obj())
    # shared_array = shared_array.reshape(x.shape[0],x.shape[1])
    #
    # def remove_cell(coordinate_cell_detection, def_param = shared_array):
    #     lab = shared_array[coordinate_cell_detection[0],coordinate_cell_detection[1]]
    #     if lab:
    #         shared_array[shared_array==lab] = 0
    #
    # pool = Pool(processes=6)
    # pool.map(remove_cell, np.array(coordinate_cell_detection))
    # # pool.close()
    # # pool.join()
    # print('time elapsed : %.2f sec' % (time.time() - t))
    #######################################################################

    # counting the number of red and the number of green
    t = time.time()
    idx = KSimage.KSimage.label2idx(x)
    del idx[0]

    coordinate_cell_detection = coordinate_cell_detection[prediction[:, 1] < 0.5, :]
    for i in range(coordinate_cell_detection.shape[0]):
        lab = x[coordinate_cell_detection[i, 0], coordinate_cell_detection[i, 1]]
        if lab:
            idx_pair = np.unravel_index(idx[lab], x.shape)
            x[idx_pair] = 0
        print('removing non-epithelial cell % d from % d' % (i + 1, coordinate_cell_detection.shape[0]))
    print('time elapsed : %.2f sec' % (time.time() - t))

    shared_array = x


    # # counting the number of red and the number of green
    # t = time.time()
    # coordinate_cell_detection = coordinate_cell_detection[prediction[:, 1] < 0.5, :]
    # for i in range(coordinate_cell_detection.shape[0]):
    #     lab = x[coordinate_cell_detection[i, 0], coordinate_cell_detection[i, 1]]
    #     if lab:
    #         x[x == lab] = 0
    #     print('removing non-epithelial cell % d from % d' % (i + 1, coordinate_cell_detection.shape[0]))
    # print('time elapsed : %.2f sec' % (time.time() - t))
    #
    # shared_array = x

    return shared_array

#######################################################################################
def count_f(j, label_map, canvas, keys):
    mask = label_map == j
    prop = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8, ltype=cv2.CV_32S)
    centroid = prop[3][1]

    count = dict()
    for key in keys:
        signal = np.logical_and(mask, canvas[key])
        prop = cv2.connectedComponents(signal.astype(np.uint8), connectivity=8, ltype=cv2.CV_32S)
        count[key] = prop[0] - 1

    return ( centroid, count )

#######################################################################################
def count_signal(label_map, coordinate_probe_dict):

    canvas = dict()

    for key in coordinate_probe_dict.keys():
        canvas[key] = np.zeros((label_map.shape[0],label_map.shape[1])).astype(np.bool)
        coordinate = coordinate_probe_dict[key]
        for j in range(coordinate.shape[0]):
            canvas[key][coordinate[j,0],coordinate[j,1]] = True

    lab = np.setdiff1d(np.unique(label_map), 0)

    output = dict()
    output['label_map'] = label_map
    output['coordinate'] = list()
    for key in coordinate_probe_dict.keys():
        output[key] = list()

    ############################################################################################
    # t = time.time()
    # for idx, j in enumerate(lab):
    #     mask = label_map == j
    #     prop = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8, ltype=cv2.CV_32S)
    #     output['coordinate'].append(prop[3][1])
    #
    #     for key in coordinate_probe_dict.keys():
    #         signal = np.logical_and(mask,canvas[key])
    #         prop = cv2.connectedComponents(signal.astype(np.uint8), connectivity=8, ltype=cv2.CV_32S)
    #         output[key].append(prop[0] - 1)
    #
    #     print('counting signal from cell %d of %d cells' % (idx, len(lab)))
    # print('time elapsed : %.2f' % (time.time() - t))

    ############################################################################################
    t = time.time()
    idx = KSimage.KSimage.label2idx(label_map)
    del idx[0]

    for i, cell_idx in enumerate(idx.keys()):
        idx_pair = np.unravel_index(idx[cell_idx], label_map.shape)
        output['coordinate'].append(np.mean(idx_pair,axis = 1))

        for key in coordinate_probe_dict.keys():
            signal = canvas[key][idx_pair]
            output[key].append(np.sum(signal))

        print('counting signal from cell %d of %d cells' % (i+1, len(idx)))
    print('time elapsed : %.2f' % (time.time() - t))
    t = time.time()

    # partial_count_f = partial(count_f,
    #                             label_map = label_map,
    #                             canvas = canvas,
    #                             keys = coordinate_probe_dict.keys())
    #
    # p = Pool(processes=6)
    # tmp = p.map(partial_count_f, lab)
    # # p.close()
    # # p.join()
    #
    # for obj in tmp:
    #     output['coordinate'].append(obj[0])
    #     for key in coordinate_probe_dict.keys():
    #         output[key].append(obj[1][key])
    #
    # output['coordinate'] = np.row_stack(output['coordinate'])
    # for key in coordinate_probe_dict.keys():
    #     output[key] = np.row_stack(output[key])
    #
    # print('time elapsed : %.2f' % (time.time() - t))

    return output
