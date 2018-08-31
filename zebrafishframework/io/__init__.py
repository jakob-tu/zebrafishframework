import os.path

import deepdish as dd
import nrrd
import numpy as np
import tables
import SimpleITK as sitk
from . import lif


class UnsupportedFormatException(Exception):
    pass


def get_shape(fn):
    """

    :param fn: image file
    :return: shape of that file
    """
    in_ext = os.path.splitext(fn)[1]

    if in_ext == '.h5':
        """
        f = tables.open_file(fn)
        return f.get_node('/stack').shape
        """
        img = load(fn)
        return img.shape
    elif in_ext == '.nrrd':
        img = load(fn)
        return img.shape
    else:
        raise UnsupportedFormatException('Input format "' + in_ext + '" is not supported.')


def get_frame(fn, t):
    """
    Get a frame from series images
    :param fn:
    :param t: timestamp
    :return: frame as numpy array
    """
    in_ext = os.path.splitext(fn)[1]

    if in_ext == '.h5':
        # Leica format
        sel = (slice(None), slice(t, t + 1))
        img = dd.io.load(fn, '/stack', sel=sel)
        img = img.squeeze()

        # zyx -> xyz
        # img = img.swapaxes(0, 2)
        return img
    else:
        raise UnsupportedFormatException('Only h5 as time series format supported.')


def __sitkread(filename):
    img = sitk.ReadImage(filename)
    return sitk.GetArrayFromImage(img)


def __sitkwrite(filename, data):
    img = sitk.GetImageFromArray(data)
    sitk.WriteImage(img, filename)


def load(fn):
    in_ext = os.path.splitext(fn)[1]

    if in_ext == '.nrrd':
        return __sitkread(fn)
    elif in_ext == '.h5':
        # ITK/ANTs:
        # dd.io.load(fn, '/ITKImage/0/VoxelData')
        #return dd.io.load(fn, '/stack')
        # ITK/ANTs format
        return __sitkread(fn)
    else:
        raise UnsupportedFormatException('Input format "' + in_ext + '" is not supported.')


def save(fn, data):
    out_ext = os.path.splitext(fn)[1]

    if out_ext == '.nrrd':
        __sitkwrite(fn, data)
    elif out_ext == '.h5':
        """
        with tables.open_file(fn, mode='w') as f:
            f.create_array('/', 'stack', data.astype(np.float32))
            f.close()
        """
        __sitkwrite(fn, data)
    else:
        raise UnsupportedFormatException('Output format "' + out_ext + '" is not supported.')


def convert(fn, out_ext):
    in_ext = os.path.splitext(fn)[1]
    comb = in_ext + '/' + out_ext

    # time series
    if in_ext == '.lif':
        if out_ext == '.h5':
            lif.readLifAndSaveAsH5(fn)
        else:
            raise UnsupportedFormatException(comb + ': lif can only be converted to h5.')
    else:

        supported_in = ['.nrrd']

        if in_ext not in supported_in:
            raise UnsupportedFormatException('Input format "' + in_ext + '" is not supported.')
