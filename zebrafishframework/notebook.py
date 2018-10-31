import os
from pyprind import prog_percent

from . import ants_cmd
from . import img
from . import io

from matplotlib import pyplot as plt

def process_scored(scored, result_folder='results'):
    print('Processing')
    if not os.path.exists(result_folder):
        os.mkdir(result_folder)

    # top 3, flop 3
    print('Top 3')
    for i, s in enumerate(scored[:3]):
        print('%f: %s' % (s[0], s[1].arguments.params))
        fn = os.path.join(result_folder, 'top_%d.h5' % i)
        if not os.path.exists(fn):
            io.save(fn, img.cmp_images([io.load(s[1].arguments.reference), io.load(s[1].get_warped())]))

    print()
    print('Flop 3')
    flops = list(scored[-3:])
    flops.reverse()
    for i, s in enumerate(flops):
        print('%f: %s' % (s[0], s[1].arguments.params))
        fn = os.path.join(result_folder, 'flop_%d.h5' % i)
        if not os.path.exists(fn):
            io.save(fn, img.cmp_images([io.load(s[1].arguments.reference), io.load(s[1].get_warped())]))

    # score vs time
    scores = list(map(lambda e: e[0], scored))
    delta_times = list(map(lambda s: s[1].delta_time, scored))

    plt.scatter(delta_times, scores)
    plt.autoscale(tight=True)
    plt.show()

    return scored
