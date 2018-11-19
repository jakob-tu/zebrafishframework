import numpy as np
import scipy.stats


def calc_bleaching_constant(ts, mean_F):
    '''

    :param ts:
    :param mean_F:
    :return:
    '''
    (a_s, b_s, r, tt, stderr) = scipy.stats.linregress(ts, np.log(mean_F))
    return a_s


def correct_bleaching(ts, F, constant):
    return np.exp(-constant * ts) * F
