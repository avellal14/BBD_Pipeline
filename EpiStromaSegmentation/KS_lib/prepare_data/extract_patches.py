"""
extract_patches.py

This file contains the implementation of the sliding window patch extraction
method which is used during training and testing.
"""

import numpy as np

def sliding_window(dict_obj, size_input_patch, size_output_patch, stride):
    """
    sliding_window extracts patches from an input image using the specified
    length, width, and stride

    param: dict_obj
    param: size_input_patch
    param: size_output_patch
    param: stride
    return: dictionary of patches
    """


    # padding
    for key in dict_obj.keys():
        padrow = size_input_patch[0]
        padcol = size_input_patch[1]

        if dict_obj[key].ndim == 2:
            dict_obj[key] = np.lib.pad(dict_obj[key], ((padrow, padrow), (padcol, padcol)), 'symmetric')
        else:
            dict_obj[key] = np.lib.pad(dict_obj[key], ((padrow, padrow), (padcol, padcol), (0, 0)), 'symmetric')

    ntimes_row = int(np.floor((dict_obj['image'].shape[0] - size_input_patch[0]) / float(stride[0])) + 1)
    ntimes_col = int(np.floor((dict_obj['image'].shape[1] - size_input_patch[1]) / float(stride[1])) + 1)
    row_range = range(0, ntimes_row * stride[0], stride[0])
    col_range = range(0, ntimes_col * stride[1], stride[1])

    assert size_input_patch[0] >= size_output_patch[0]
    assert size_input_patch[1] >= size_output_patch[1]

    centre_index_row = int(round((size_input_patch[0] - size_output_patch[0]) / 2.0))
    centre_index_col = int(round((size_input_patch[1] - size_output_patch[1]) / 2.0))

    for row in row_range:
        for col in col_range:
            coord_dict = dict()
            out_obj_dict = dict()

            for key in dict_obj.keys():
                if dict_obj[key].ndim == 2:
                    if key == 'image':
                        out_obj_dict[key] = dict_obj[key][
                                            row : row + size_input_patch[0],
                                            col : col + size_input_patch[1]]
                    else:
                        out_obj_dict[key] = dict_obj[key][
                                            row + centre_index_row: row + centre_index_row + size_output_patch[0],
                                            col + centre_index_col: col + centre_index_col + size_output_patch[1]]
                    coord_dict[key] = row, col
                else:
                    if key == 'image':
                        out_obj_dict[key] = dict_obj[key][
                                            row : row + size_input_patch[0],
                                            col : col + size_input_patch[1], :]
                    else:
                        out_obj_dict[key] = dict_obj[key][
                                            row + centre_index_row: row + centre_index_row + size_output_patch[0],
                                            col + centre_index_col: col + centre_index_col + size_output_patch[1], :]
                    coord_dict[key] = row, col

            yield (out_obj_dict, coord_dict)




