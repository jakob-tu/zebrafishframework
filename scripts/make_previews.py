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

import fileinput

def main(argv):
    folder = '/Users/koesterlab/registered/control/'
    files = glob(folder + '*_aligned.h5')

    #files = fileinput.input()
    for f in prog_percent(files):
        try:
            r = re.compile(r'^' + folder + '(?P<fn>.*)_aligned.h5')
            m = r.match(f)
            fn = m.group('fn')
            process_file(folder + fn)
        except Exception as e:
            print(e)


def process_file(base):
    #print(base)
    if os.path.exists(base + '_aligned_preview.h5'):
        print('Found preview. Skipping')
        return

    #print('Loading stack...')
    sel = (slice(0, 1800), slice(10, 11))
    stack = dd.io.load(base + '_aligned.h5', sel=sel)
    #print('Saving preview...')
    
    # atm fiji h5 plugin does not support blosc compression
    dd.io.save(base + '_aligned_preview.h5', stack)


if __name__ == '__main__':
    main(sys.argv)
