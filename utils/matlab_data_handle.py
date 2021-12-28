import scipy.io
import numpy as np
# import h5py
# import hdf5storage

def save_mat_dict(filename, data):
    # hdf5storage.write(data, '.', 'example.mat', matlab_compatible=True)
    scipy.io.savemat(filename, data)


def load_mat_dict(filename):
    """
    this function should be called instead of direct scipy.io.loadmat
    as it cures the problem of not properly recovering python dictionaries
    from mat files. It calls the function check keys to cure all entries
    which are still mat-objects.
    Code from: https://github.com/scipy/scipy/issues/7895#issuecomment-548060478
    """

    def _check_keys(d):
        """
        checks if entries in dictionary are mat-objects. If yes
        todict is called to change them to nested dictionaries
        """
        for key in d:
            if isinstance(d[key], scipy.io.matlab.mio5_params.mat_struct):
                d[key] = _to_dict(d[key])
        return d

    def _to_dict(mat_object):
        """
        A recursive function which constructs from matobjects nested dictionaries
        """
        d = {}
        for field_name in mat_object._fieldnames:
            elem = mat_object.__dict__[field_name]
            if isinstance(elem, scipy.io.matlab.mio5_params.mat_struct):
                d[field_name] = _to_dict(elem)
            elif isinstance(elem, np.ndarray):
                d[field_name] = _to_list(elem)
            else:
                d[field_name] = elem
        return d

    def _to_list(input_array: np.ndarray):
        """
        A recursive function which constructs lists from cell-arrays
        (which are loaded as numpy ndarrays).
        """
        output_list = []
        for sub_elem in input_array:
            if isinstance(sub_elem, scipy.io.matlab.mio5_params.mat_struct):
                output_list.append(_to_dict(sub_elem))
            elif isinstance(sub_elem, np.ndarray):
                output_list.append(_to_list(sub_elem))
            else:
                output_list.append(sub_elem)
        return output_list

    data = scipy.io.loadmat(filename, struct_as_record=False, squeeze_me=True, mat_dtype=True)
    return _check_keys(data)
