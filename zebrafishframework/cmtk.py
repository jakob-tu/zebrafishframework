import multiprocessing as mp
import os

from . import util


# example call: munger -a -w -r 010203  -X 52 -C 8 -G 80 -R 3
# -A '--accuracy 0.4' -W '--accuracy 1.6' -T 32 -s "refbrain/Elavl3-GCaMP5G.nrrd" images
def gen_cmtk(input_fish, reference, output_folder, num_threads=None):
    if not num_threads:
        num_threads = mp.cpu_count()

    cmd = ''

    cmd += 'cd %s;' % os.path.abspath(output_folder)
    cmd += ' munger -a -w -r 010203'
    cmd += ' -X 52 -C 8 -G 80 -R 3'
    cmd += ' -A \'--accuracy 0.4\''
    cmd += '-W \'--accuracy 1.6\''
    cmd += '-T %d -s %s %s' % (num_threads, os.path.abspath(reference), os.path.abspath(input_fish))

    return cmd


def run_cmtk(input_fish, reference):
    in_name = os.path.splitext(os.path.basename(input_fish))[0]
    ref_name = os.path.splitext(os.path.basename(reference))[0]
    output_folder = in_name + '_CTMK_' + ref_name
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    cmd = gen_cmtk(input_fish, reference, output_folder)
    out = util.call(cmd, print_output=True)
    return out
