import math
import numpy as np
import os
from pyprind import prog_percent

from . import io
from . import ants


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

    out = np.zeros((len(imgs), ) + s)
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
            ants.run_antsreg(frame_name(t), ref,)
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
