import bioformats
import deepdish as dd
import h5py
import javabridge
import numpy as np
import os.path
from pyprind import prog_percent
import SimpleITK as sitk
import tables
import time
from xml.etree import ElementTree as ETree


from . import util


SPACING_ZBB = (0.798, 0.798, 2)
SPACING_JAKOB = (0.7188675, 0.7188675, 10)
SPACING_JAKOB_HQ = (0.7188675, 0.7188675, 1)


class UnsupportedFormatException(Exception):
    pass


def lif_get_metas(fn):
    md = bioformats.get_omexml_metadata(fn)  # Load meta data
    mdroot = ETree.fromstring(md)  # Parse XML
    #    meta = mdroot[1][3].attrib # Get relevant meta data
    metas = list(map(lambda e: e.attrib, mdroot.iter('{http://www.openmicroscopy.org/Schemas/OME/2016-06}Pixels')))

    return metas


def lif_open(fn):
    javabridge.start_vm(class_path=bioformats.JARS)
    ir = bioformats.ImageReader(fn)

    return ir


def lif_read_stack(fn):
    ir = lif_open(fn)
    img_i = lif_find_timeseries(fn)
    shape = get_shape(fn, img_i)

    stack = np.empty(shape, dtype=np.uint16)

    # Load the whole stack...
    for t in prog_percent(range(stack.shape[0])):
        for z in range(stack.shape[1]):
            stack[t, z] = ir.read(t=t, z=z, c=0, series=img_i, rescale=False)

    return stack


def readLifAndSaveAsH5(fn):
    '''Read LIF (Leica) file using bioformats and save it compressed as HDF5 file
    :param fn: filename
    :return: filename of HDF5 file'''
    print('Working on ', fn)

    fn_hdf5 = fn.replace('.lif', '.h5')
    stack = lif_read_stack(fn)

    dd.io.save(fn_hdf5, {'stack': stack}, compression='blosc')
    #    save(fn_hdf5, prealloc_stack, spacing)

    print('Data saved [%s]' % util.format_time(time.time() - t))

    return fn_hdf5


def lif_find_timeseries(fn):
    metas = lif_get_metas(fn)

    meta = None
    img_i = 0
    for i, m in enumerate(metas):
        if int(m['SizeT']) > 1:
            meta = m
            img_i = i

    if not meta:
        raise ValueError('lif does not contain an image with sizeT > 1')

    return img_i


def get_shape(fn, index=0):
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
    elif in_ext == '.lif':
        metas = lif_get_metas(fn)
        meta = metas[index]

        shape = (
            int(meta['SizeT']),
            int(meta['SizeZ']),
            int(meta['SizeY']),
            int(meta['SizeX']),
        )
        order = meta['DimensionOrder']
        spacing = tuple([float(meta['PhysicalSize%s' % c]) for c in 'XYZ'])

        return shape

    else:
        raise UnsupportedFormatException('Input format "' + in_ext + '" is not supported.')


def h5_node_exists(fn, name):
    f = tables.open_file(fn)
    try:
        f.get_node(name)
        return True
    except tables.NoSuchNodeError:
        return False


def h5_is_itk(fn):
    return h5_node_exists(fn, '/ITKImage')


def read_h5py(fn):
    f = h5py.File(fn, mode='r')
    dataset = list(f.items())[0][1]
    w = np.empty(dataset.shape, dataset.dtype)
    dataset.read_direct(w)
    return w


def get_frame(fn, t):
    """
    Get a frame from series images
    :param fn:
    :param t: timestamp
    :return: frame as numpy array
    """
    in_ext = os.path.splitext(fn)[1]

    if in_ext == '.h5':
        if h5_is_itk(fn):
            return None # todo
        else:
            # assume file was converted from .lif
            sel = (slice(t, t + 1),)
            img = dd.io.load(fn, '/stack', sel=sel)
            img = img.squeeze()
            return img
    elif in_ext == '.lif':
        ir = lif_open(fn)
        img_i = lif_find_timeseries(fn)
        shape = get_shape(fn, img_i)
        frame = np.zeros(shape[1:])
        for z in range(shape[1]):
            frame[z] = ir.read(t=t, z=z, c=0, series=img_i, rescale=False)
        return frame

    else:
        raise UnsupportedFormatException('h5 and lif as time series format supported.')


def __sitkread(filename):
    img = sitk.ReadImage(filename)
    spacing = img.GetSpacing()
    return sitk.GetArrayFromImage(img), spacing


def __sitkwrite(filename, data, spacing):
    img = sitk.GetImageFromArray(data)
    img.SetSpacing(spacing)
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


def save(fn, data, spacing):
    out_ext = os.path.splitext(fn)[1]

    if out_ext == '.nrrd':
        __sitkwrite(fn, data, spacing)
    elif out_ext == '.h5':
        """
        with tables.open_file(fn, mode='w') as f:
            f.create_array('/', 'stack', data.astype(np.float32))
            f.close()
        """
        __sitkwrite(fn, data, spacing)
    else:
        raise UnsupportedFormatException('Output format "' + out_ext + '" is not supported.')



def convert(fn, out_ext):
    in_ext = os.path.splitext(fn)[1]
    comb = in_ext + '/' + out_ext
    new_fn = fn.replace(in_ext, out_ext)

    # time series
    if in_ext == '.lif':
        if out_ext == '.h5':
            readLifAndSaveAsH5(fn)
        else:
            raise UnsupportedFormatException(comb + ': lif can only be converted to h5.')
    else:

        supported_in = ['.nrrd']

        if in_ext not in supported_in:
            raise UnsupportedFormatException('Input format "' + in_ext + '" is not supported.')

        img, spacing = load(fn)
        save(new_fn, img, spacing)
