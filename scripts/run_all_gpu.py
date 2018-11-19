#!/usr/bin/env python
import fileinput
from glob import glob
from pyprind import prog_percent
import shutil
import os

from zebrafishframework import util

#for f in fileinput.input():
files = glob('/Users/koesterlab/calcium/control/fish*.lif')

def make_base(f):
    f_base = f.replace('calcium', 'registered').replace('.lif', '')
    return f_base

def necessary(f):
    f_base = make_base(f)
    return not os.path.exists(f_base + '_aligned.h5')


for f in prog_percent(list(filter(necessary, files))):
    f = f.strip()
    print(f)
    try:
        f_base = make_base(f)
        tmpf = '/Users/koesterlab/tmp/tmp.lif'
        print('Copying %s to %s' % (f, tmpf))
        util.print_time(lambda: shutil.copyfile(f, tmpf))

        cmd = 'python -m zebrafishframework.gpu "%s" "%s"' % (tmpf, f_base)
        os.system(cmd)

        print('removing temp file')
        os.remove(tmpf)

    except Exception as e:
        print(e)
