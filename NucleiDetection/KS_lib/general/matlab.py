import scipy.io as sio
import hdf5storage

def save(file_path, dictionary):
    """
    save mat file
    :param file_path:
    :param dictionary:
    :return:
    """
    # sio.savemat(file_path, dictionary)

    def dict_keys_to_unicode(d):
        out = dict()
        for k, v in d.items():
            out[k.decode()] = v
        return out

    try:
        tmp_dictionary = dict_keys_to_unicode(dictionary)
        for key in tmp_dictionary.keys():
            if isinstance(tmp_dictionary[key], dict):
                tmp_dictionary[key] = [dict_keys_to_unicode(tmp_dictionary[key])]

        hdf5storage.savemat(file_path, tmp_dictionary, format='7.3', oned_as='column',
                        store_python_metadata=True)
    except:
        sio.savemat(file_path, dictionary)

def load(file_path):
    """
    save mat file
    :param file_path:
    :return: mat_contents
    """
    try:
        mat_contents = hdf5storage.loadmat(file_path)
    except:
        mat_contents = sio.loadmat(file_path)
    return mat_contents