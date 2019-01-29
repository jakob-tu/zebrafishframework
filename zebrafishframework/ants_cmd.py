import itertools
import copy
import matplotlib.pyplot as plt
import multiprocessing as mp
import numpy as np
import os
import os.path
import pickle
from pyprind import prog_percent
import SimpleITK as sitk
import time
import tempfile
import re

from . import io
from . import img
from . import util


# placeholder for reference,input in ANTs cmd
REF_IN_KEY = '$REF_IN'

METRIC_MI = 'MI[' + REF_IN_KEY + ',1,32,Regular,0.25]'
METRIC_CC = 'CC[' + REF_IN_KEY + ',1,2]'


class Arguments:
    def __init__(self, input_file, reference, params=None, output_folder=None):
        self.input_file = input_file
        self.reference = reference
        self.input_name = os.path.splitext(os.path.basename(input_file))[0]
        self.reference_name = os.path.splitext(os.path.basename(reference))[0]

        self.dimensions = 3
        self.use_float = 0
        self.use_histogram_matching = 0
        self.interpolation = 'BSpline'
        self.winsorize = '[0.005, 0.995]'

        if not params:
            self.params = get_default_params()
        else:
            self.params = params

        if not output_folder:
            self.output_folder = self.input_name + '_ANTs_' + self.reference_name
        else:
            self.output_folder = output_folder

        self.num_threads = mp.cpu_count()


class Result:
    def __init__(self, arguments, raw):
        self.arguments = arguments
        self.raw = raw

        try:
            self.stages = parse_antsreg(raw)
        except:
            self.stages = None

    def get_warped(self):
        return os.path.join(self.arguments.output_folder, self.arguments.input_name + '_Warped.nrrd')

    def load_warped(self):
        return io.load(self.get_warped())

    def get_generic_affine(self):
        fn, ext = os.path.splitext(os.path.basename(self.arguments.input_file))
        prefix = os.path.join(self.arguments.output_folder, fn + '_')
        return prefix + '0GenericAffine.mat'



def get_default_params():
    """
    Generates sensible default parameters for gen_antsreg
    :return: params as list (stages) of dict (parameters for stage) with keys: \
    transform, metric, convergences, shrink_factors, smoothing_sigmas
    """
    ref_in = REF_IN_KEY

    return [
        {
            'transform': 'Rigid[0.1]',
            'metric': 'MI[' + ref_in + ',1,32,Regular,0.25]',
            'convergences': '[500x500x500x0,1e-8,10]',
            'shrink_factors': '12x8x4x2',
            'smoothing_sigmas': '4x3x2x1'
        },
        {
            'transform': 'Affine[0.1]',
            'metric': 'MI[' + ref_in + ',1,32,Regular,0.25]',
            'convergences': '[200x200x200x0,1e-8,10]',
            'shrink_factors': '12x8x4x2',
            'smoothing_sigmas': '4x3x2x1'
        },
        {
            'transform': 'SyN[0.1,6,0]',
            'metric': 'CC[' + ref_in + ',1,2]',
            'convergences': '[200x200x200x200x0,1e-7,10]',
            'shrink_factors': '12x8x4x2x1',
            'smoothing_sigmas': '4x3x2x1x0'
        }
    ]


def iter_param_space(param_space):

    def merge_dicts(dicts):
        z = {}
        for d in dicts:
            z.update(d)
        return z

    def stage_iterator(stage):
        for dicts in itertools.product(*stage):
            yield merge_dicts(dicts)

    return itertools.product(*map(stage_iterator, param_space))


def measure_similarity(fn_a, fn_b, metric):
    """
    Use ants tool to measure image similarity
    :param fn_a: filename of a
    :param fn_b: filename of b
    :param metric: metric string as defined in ANTs help
    :return: similarity
    """
    fn_a = os.path.abspath(fn_a)
    fn_b = os.path.abspath(fn_b)
    ref_in = fn_a + ',' + fn_b
    metric = metric.replace(REF_IN_KEY, ref_in)
    out = util.call('MeasureImageSimilarity -d 3 -m %s' % metric)
    print(metric)
    print(out)
    return float(out)


def get_param_search_space():
    """
    Generate a parameter space used for optimization. Customize to your liking
    The parameter space has the following structure:
    parameter_space is a list of stages
    a stage is a list of parts, those are eventually merged by cartesian product for that stage
    a part is a list of options
    an option is a dict of individual parameters
    :return: parameter_space
    """
    ref_in = REF_IN_KEY

    def pack(keys, list_of_params):
        return [{k: params[i] for i, k in enumerate(keys)} for params in zip(*list_of_params)]

    return\
        [ # Rigid
            [{'transform': 'Rigid[%0.2f]' % step} for step in np.arange(.1, .3, .1)],
            [{'metric': 'MI[' + ref_in + ',1,32,Regular,0.25]'}],
            pack(['convergences', 'shrink_factors', 'smoothing_sigmas'], [
                ['[500x500x500x0,1e-8,10]', '[500x500x500x0,1e-8,10]', '[500x500x500x0,1e-8,10]'],
                ['12x8x4x2', '12x8x4x2', '12x8x4x2'],
                ['4x3x2x1', '2x1.5x1x.5', '8x6x4x2']
            ])
        ], [ # Affine
            [{'transform': 'Affine[%0.2f]' % step} for step in np.arange(.1, .3, .1)],
            [{'metric': 'MI[' + ref_in + ',1,32,Regular,0.25]'}],
            pack(['convergences', 'shrink_factors', 'smoothing_sigmas'], [
                ['[200x200x200x0,1e-8,10]', '[200x200x200x0,1e-8,10]', '[200x200x200x0,1e-8,10]'],
                ['12x8x4x2', '12x8x4x2', '12x8x4x2'],
                ['4x3x2x1', '2x1.5x1x.5', '8x6x4x2']
            ])
        ], [ # SyN
            [{'transform': 'SyN[0.1]'}],
            [{'metric': 'CC[' + ref_in + ',1,2]'}],
            pack(['convergences', 'shrink_factors', 'smoothing_sigmas'], [
                ['[200x200x200x200x0,1e-7,10]'],
                ['12x8x4x2x1'],
                ['4x3x2x1x0']
            ])
        ]


def gen_antsreg(arguments, param_prefix=None):
    """ ANTs batch registration
    :param arguments: instance of AntsArguments
    :return: cmd
    """
    ref_in = arguments.reference + ',' + arguments.input_file

    fn, ext = os.path.splitext(os.path.basename(arguments.input_file))
    prefix = os.path.join(arguments.output_folder, fn + '_')
    warped = prefix + 'Warped' + ext

    cmd = ''

    if arguments.num_threads:
        cmd += 'export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=%d; ' % arguments.num_threads
    else:
        cmd += 'unset ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS; '

    cmd += 'antsRegistration'
    cmd += ' --verbose 1'
    cmd += ' -d %d --float %d' % (arguments.dimensions, arguments.use_float)
    cmd += ' --interpolation %s' % arguments.interpolation
    cmd += ' -o [' + prefix + ',' + warped + ']'
    cmd += ' --use-histogram-matching %d' % arguments.use_histogram_matching
    if arguments.winsorize:
        cmd += ' --winsorize-image-intensities %s' % arguments.winsorize

    # apply param_prefix before initial-moving-transform center of mass
    # this is so the prescale matrix works properly
    if param_prefix:
        cmd += ' ' + param_prefix.strip()

    cmd += ' -r [' + ref_in + ',1]'

    for stage in arguments.params:
        cmd += ' -t ' + stage['transform']
        cmd += ' -m ' + stage['metric'].replace(REF_IN_KEY, ref_in)
        cmd += ' -c ' + stage['convergences']
        cmd += ' --shrink-factors ' + stage['shrink_factors']
        cmd += ' --smoothing-sigmas ' + stage['smoothing_sigmas']

#    cmd += '|tee ' + prefix + 'log.txt'

    return cmd


def run_antsreg(arguments, print_output=True):
    """
    Run ants registration with sensible defaults
    :param arguments: instance of AntsArguments
    :return: output of ANTs call
    """

    if not os.path.exists(arguments.output_folder):
        os.mkdir(arguments.output_folder)

    param_prefix = None
    cmd = gen_antsreg(arguments, param_prefix)

    start = time.time()
    out = util.call(cmd, print_output=print_output)

    with open(os.path.join(arguments.output_folder, 'log.txt'), 'w') as f:
        f.write(out)

    result = Result(arguments, out)
    result.delta_time = time.time() - start
    result.cmd = cmd

    return result


def apply_transform(inp, ref, out, transforms, print_output=False):
    cmd = 'antsApplyTransforms'
    cmd += ' -d 3 -v 1 --float'
    cmd += ' -n WelchWindowedSinc'
    cmd += ' -i %s' % inp
    cmd += ' -r %s' % ref
    cmd += ' -o %s' % out
    for t in transforms:
        cmd += ' -t %s' % t

    if print_output:
        print(cmd)

    util.call(cmd, print_output=print_output)


def plane_wise(arguments, z_range=None, print_output=False):
    image = io.load(arguments.input_file)
    ref = io.load(arguments.reference)
    r1 = util.randomword(20)
    r2 = util.randomword(20)
    tmpf = tempfile.gettempdir()

    def plane_name(r, z):
        return os.path.join(tmpf, r + '_z%d.nrrd' % z)

    if not z_range:
        z_range = np.arange(0, image.shape[0])

    warped = []
    results = []
    for z in prog_percent(z_range):
        inp_plane = image[z]
        ref_plane = ref[z]
        inp_fn = plane_name(r1, z)
        ref_fn = plane_name(r2, z)
        io.save(inp_fn, inp_plane)
        io.save(ref_fn, ref_plane)
        arguments_z = copy.deepcopy(arguments)
        arguments_z.input_file = inp_fn
        arguments_z.reference = ref_fn
        arguments_z.dimensions = 2
        result = run_antsreg(arguments_z)
        results.append(result)
        warped.append(result.load_warped())

    return img.cmp_images(warped), results


def grid_search(list_input_fish, references, base_folder='gridsearch/', param_space=None):

    if not param_space:
        param_space = get_param_search_space()

    list_input_fish = map(os.path.abspath, list_input_fish)
    references = map(os.path.abspath, references)
    base_folder = os.path.abspath(base_folder)

    in_ref_combinations = itertools.product(list_input_fish, references)
    params = list(iter_param_space(param_space))

    # shuffle configurations for various reasons:
    # - improve ETA calculation
    # - have broader results when cancelling prematurely
    import random
    random.shuffle(params)

    configurations = itertools.product(in_ref_combinations, params)

    def encode_config(config):
        import json
        import hashlib
        m = hashlib.sha256()
        m.update(bytes(json.dumps(config), 'utf-8'))
        return m.hexdigest()

    def run_config(config):
        in_ref, params = config
        inp, ref = in_ref
        encode = encode_config(config)
        arguments = Arguments(inp, ref, params=params)
        arguments.output_folder = os.path.join(base_folder, 'run_' + encode)

        RESFILE = 'result.pickle'

        # output_folder is unique to configuration. if output_resfile exists, this config has already been performed
        # check for resfile to see if command ran completely and return it to save time
        resfn = os.path.join(arguments.output_folder, RESFILE)
        if os.path.exists(resfn):
            return pickle.load(resfn)

        out = run_antsreg(arguments, print_output=False)
        with open(os.path.abspath(os.path.join(arguments.output_folder, 'result.pickle')), 'wb') as resfile:
            pickle.dump(out, resfile)

        return out

    if not os.path.exists(base_folder):
        os.mkdir(base_folder)

    # return function and sequence for map()
    return run_config, list(configurations)


def run_gridsearch(*args, **kwargs):
    f, configs = grid_search(*args, **kwargs)
    results = []
    for i, c in prog_percent(list(enumerate(configs))):
        result = f(c)
        results.append(result)
    return results


def score_results(results, metric='MI[$REF_IN,1,32,Regular,0.25]'):
    def func(res):
        warped = res.get_warped()
        ref = res.arguments.reference
        sim = measure_similarity(ref, warped, metric)
        return sim, res

    print('Scoring')
    scored = list(map(func, prog_percent(results)))
    scored.sort(key=lambda e: e[0])

    return scored


def parse_antsreg(output):
    """
    Prints and plots convergences and durations
    :param output: output of antsRegistrationcommand
    :return: list (stages) of list (levels) of list (iterations) of tuple (metric, abs_time, delta_time)
    """
    lines = output.split('\n')

    # not beautiful, but it works for now
    r_stage = re.compile(r'^Stage \d$')
    r_level = re.compile(r'^X*DIAGNOSTIC,Iteration,metricValue,convergenceValue,ITERATION_TIME_INDEX,SINCE_LAST')
    r_iter = re.compile(r'.*DIAGNOSTIC.*,.*,.*,(?P<conv>.*),.*')
    stages = []
    for l in lines:
        m_stage = r_stage.match(l)
        m_level = r_level.match(l)
        m_iter = r_iter.match(l.replace(' ', ''))
        if m_stage:
            levels = []
            stages.append(levels)
        elif m_level:
            iters = []
            levels.append(iters)
        elif m_iter:
            s = l.split(',')
            m = float(s[2])
            t = float(s[4])
            dt = float(s[5])
            iters.append((m, dt, t))
    return stages


def final_metric(stages):
    return stages[-1][-1][-1][0]


def show_antsreg_plots(stages):
    for i, s in enumerate(stages):
        delta = 0
        for l in s:
            length = len(l)
            x = list(range(delta, delta + length))
            delta += length
            plt.plot(x, list(map(lambda e: e[0], l)))
        plt.yscale('linear')
        plt.title('Stage %d Metric' % i)
        plt.show()

    """
    for i, s in enumerate(stages):
        delta = 0
        for l in s:
            l = l[1:]  # exclude first entry
            length = len(l)
            x = list(range(delta, delta + length))
            delta += length
            plt.plot(x, list(map(lambda e: e[1], l)))
        plt.yscale('linear')
        plt.title('Stage %d Time/iter' % i)
        plt.show()
    """

    def time(s):
        return '%fs (%fh)' % (s, s / 3600)

    for i, s in enumerate(stages):
        time_level = 0
        for j, l in enumerate(s):
            time_level = l[-1][2] - time_level
            print('Level %d: %s' % (j, time(time_level)))
        time_stage = s[-1][-1][2]
        print('Stage %d: %s\n' % (i, time(time_stage)))
    print('Time total: %s' % time(stages[-1][-1][-1][2]))
