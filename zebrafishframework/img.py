import math
import numpy as np
import os
from pyprind import prog_percent

from . import io
from . import ants_cmd


# rotate left 90 degrees and flip z axis
def our_view_to_zbrain_img(img):
    if len(img.shape) != 3:
        raise ValueError('Only implemented for 3d')
    return np.flip(np.rot90(img, axes=(1, 2)), axis=0)


def our_view_to_zbrain_point(p, shape):
    '''

    :param p: point as xyz
    :param shape: shape as zyx
    :return:
    '''
    if p.shape[0] != 3:
        raise ValueError('Only implemented for 3d')
    return np.array([p[1], shape[1]-p[0]-1, shape[0]-p[2]-1])


def our_view_to_zbrain_rois(rois, shape):
    xyzs = rois[:, :3]
    rs = rois[:, 3:]
    xyzs_transformed = np.array(list(map(lambda xyz: our_view_to_zbrain_point(xyz, shape), xyzs)))
    return np.concatenate([xyzs_transformed, rs], axis=1)


def enlarge_image(img, by):
    '''
    Enlarge an image by adding space 'left' and 'right' in each dimension.
    :param img: d-dimensional image
    :param by: array of shape dx2. by[x,0] is the number of elements added at 0, by[x,1] is added after img.shape[x]
    :param fill_value: value to fill
    :return: enlarged image
    '''

    by = np.array(by, dtype=np.int)
    shape = np.array(img.shape) + np.sum(by, axis=1)
    enlarged = np.zeros(shape, dtype=img.dtype)
    enlarged[tuple( [slice(b[0], s+b[0]) for b, s in zip(by, img.shape)] )] = img
    return enlarged


def enlarge_points(rois, by):
    by_rev = np.flip(by)
    return np.array([(x + by_rev[0], y + by_rev[1], z + by_rev[2], r) for x, y, z, r in rois])


def selective_z_downscale(img, z2):
    z1 = img.shape[-1]

    # step is the size, which results in z2-1 steps, because you need z2-1 steps in order to go through z2 points
    step_size = math.floor(z1/(z2 - 1))
    start = math.floor((z1 - (z2-1)*step_size)/2)

    indices = slice(start, z1, step_size)
    return img[:, :, indices]


def cut_series(fn, ts):
    for new_t, t in prog_percent(list(enumerate(ts))):
        frame = io.get_frame(fn, t)
        if new_t == 0:
            new_shape = list(frame.shape)
            new_shape.insert(1, len(ts))
            new = np.zeros(new_shape)
        new[:,new_t,:,:] = frame

    return new


def slice_series(fn, z, ts):
    for new_t, t in prog_percent(list(enumerate(ts))):
        frame = io.get_frame(fn, t)
        if new_t == 0:
            new_shape = list(frame.shape)[1:]
            new_shape = [len(ts)] + new_shape
            new = np.zeros(new_shape)
        new[new_t,:,:] = frame[z]

    return new


def split_series(fn, t, folder):
    frame = io.get_frame(fn, t)
    for z in range(frame.shape[0]):
        io.save(os.path.join(folder, '%04d_%03d.nrrd' % (t, z)), frame[z])


def cmp_images(imgs):
    s = imgs[0].shape
    for img in imgs:
        if img.shape != s:
            raise AttributeError('All images must have the same shape.')
    for i, img in enumerate(imgs):
        print('%d/%d' % (i+1, len(imgs)))
        out[i] = img

    return out


def register_timeseries(fn, ts, params, num_threads):
    frame_folder = 'frames'
    registered_folder = 'registered_frames'

    if not os.path.exists(frame_folder):
        os.mkdir(frame_folder)

    def frame_name(t):
        return os.path.join(frame_folder, 'f%04d.nrrd' % t)

    def registered_name(t):
        return os.path.join(registered_folder, 'f%04d_Warped.nrrd' % t)

    def registered_matrix(t):
        return os.path.join(registered_folder, 'f%04d_0GenericAffine.mat')

    print('Extracting frames')
    for t in ts:
        if not os.path.exists(frame_name(t)):
            frame = io.get_frame(fn, t)
            io.save(frame_name(t), frame)
            print(t)
        else:
            print('Skipping %d' % t)

    print('Registering')
    ref = frame_name(ts[0])
    for i, t in enumerate(ts[1:]):
        if not os.path.exists(registered_name(t)):
            ants_cmd.run_antsreg(frame_name(t), ref, )
            print(t)
        else:
            print('Skipping %d' % t)

    print('Loading registered')
    imgs = []
    for i, t in enumerate(ts):
        name = frame_name(t) if i == 0 else registered_name(t)
        if os.path.exists(name):
            imgs.append(io.load(name))
            print(t)

    print('Merging registered')
    io.save('registered.h5', cmp_images(imgs))

    print('Merging unregistered')
    frames = []
    for t in ts:
        frames.append(io.get_frame(fn, t))
    io.save('unregistered.h5', cmp_images(frames))
