#!/usr/bin/env python

import deepdish as dd
from glob import glob
import numpy as np
import os
from pyprind import prog_percent
import sys
import re

from zebrafishframework import io
from zebrafishframework import segmentation

def main(argv):
    folder = '/Users/koesterlab/registered/control/'
    files = glob(folder + '*_aligned.h5')

    for f in prog_percent(files):
        try:
            r = re.compile(r'^' + folder + '(?P<fn>.*)_aligned.h5')
            m = r.match(f)
            fn = m.group('fn')
            folder_new = folder.replace('registered', 'segmented')
            process_file(folder + fn, folder_new + fn)
        except Exception as e:
            print(e)


def process_file(base, base_out):
    print(base)
    if os.path.exists(base_out + '_rois.npy'):
        print('Found rois. Skipping')
        return

    print('Loading invalid frames...')
    invalid_frames = np.load(base + '_invalid_frames.npy')
    print('Loading stack...')
    stack = dd.io.load(base + '_aligned.h5')
    print('Computing std...')
    valid_frames = segmentation.valid_frames(invalid_frames, length=np.alen(stack))
    std = segmentation.std(stack, valid_frames=valid_frames)
    print('Saving std...')
    io.save(base_out + '_std_dev.h5', std, spacing=io.SPACING_JAKOB)
    print('Finding rois...')
    rois = segmentation.find_rois_blob(std)
    print('Saving rois...')
    np.save(base_out + '_rois.npy', rois)
    print('Getting traces...')
    traces = segmentation.get_traces(stack, rois)
    print('Saving traces...')
    np.save(base_out + '_traces.npy', traces)


if __name__ == '__main__':
    main(sys.argv)
