import numpy as np
from pyprind import prog_percent
import random
import scipy

import moviepy.editor as mpy

FFMPEG_BIN = 'ffmpeg'

def orthogonal(rois, traces, color_func, ts, shape):
    activity = np.zeros((np.alen(ts),) + tuple(shape) + (3,), dtype=np.uint8)

    for i, t in prog_percent(list(enumerate((ts)))):
        for roi_id, (roi, trace) in enumerate(zip(rois, traces)):
            x, y, z, _ = roi
            activity[i, y, x] = color_func(trace[t])

    return activity


def contribution_function(dist, exp=1, radius=3):
    c = (1 - dist/radius)**exp
    return c


def green_magenta_dFF_func(dFF):
        final_a = (0, 255, 0)
        final_b = (255, 0, 255)
        alpha = 1
        max_dFF = 1
        c = np.array(final_b if dFF > 0 else final_a, dtype=np.float32)
        dFF = min(abs(dFF), max_dFF) / max_dFF
        return np.array(c * alpha * dFF, dtype=np.uint8)


def downscale_rois(rois, from_shape, to_shape):
    xyzs = rois[:,:3]
    rs = rois[:,3:]

    fact = np.array(to_shape)/np.array(from_shape)
    xyzs = (xyzs.T * fact).T

    return np.concatenate((xyzs, rs), axis=1)


from scipy import spatial
def pixel_map(rois, shape, radius=3):
    '''
    Map pixels in a rendered frame to the rois that are within radius of that pixel. Required for fast average rendering.
    :param rois: rois coordinates, can be xyz or xy
    :param shape: shape of the output frame
    :param radius: radius to include rois
    :return: list of roi ids for each pixel, list of distances of the rois for each pixel, list of pixels where rois within the radius exist
    '''

    rois_flipped = np.flip(rois, axis=1)
    tree = spatial.cKDTree(rois_flipped)

    frame_involved_rois = np.zeros(shape, dtype=object)
    frame_dists = np.zeros(shape, dtype=object)
    pixel_list = []
    #for p_nda in prog_percent(np.ndindex(shape), iterations=np.prod(shape)):
    for p in np.ndindex(shape):
        indices = tree.query_ball_point(p, radius)
        if len(indices) > 0:
            dists = list(np.sqrt(np.sum(np.square(rois_flipped[indices] - p), axis=1)))
            pixel_list.append(p)
            frame_dists[p] = dists
            frame_involved_rois[p] = indices
    return frame_involved_rois, frame_dists, pixel_list


def pix_map_filter(pix_map, N = 3):
    '''
    Clear out pixels where less than N rois are located
    :param pix_map:
    :param N:
    :return:
    '''

    frame_involved, frame_dists, pixel_list = pix_map
    frame_involved = frame_involved.copy()
    frame_dists = frame_dists.copy()
    pixel_list_new = []
    for p in pixel_list:
        if len(frame_involved[p]) < N:
            frame_involved[p] = 0
            frame_dists[p] = 0
        else:
            pixel_list_new.append(p)

    return frame_involved, frame_dists, pixel_list_new


def orthogonal_averaged(pix_map, traces, ts, shape, contribution_f=None, fill_value = 0):
    if not contribution_f:
        contribution_f = contribution_function

    frame_involved, frame_dists, pixel_list = pix_map

    # improves ETA calculation
    random.shuffle(pixel_list)

    contributions = np.zeros(frame_dists.shape, dtype=object)
    for p in pixel_list:
        contributions[p] = [contribution_f(d) for d in frame_dists[p]]
        contributions[p] /= np.sum(contributions[p])

    # frame = np.zeros(shape)
    #dim = len(shape)
    #sel = tuple(np.array(pixel_list).T)

    video = np.full((len(ts),) + shape, fill_value, dtype=np.float32)
    sl = (slice(np.alen(video)),)
    for p in prog_percent(pixel_list):
        # frame[p] = np.sum(traces[frame_involved[p], ts]*contributions[p])
        traces_in_p = traces[frame_involved[p]][:,ts]
        traces_weighed = (traces_in_p.T*contributions[p]).T
        video[sl + p] = np.sum(traces_weighed, axis=0)

        # does not work
#        frame[sel] = np.sum(traces[frame_involved[sel], t]*contributions[sel], axis=dim)
    return video


def colorize(video, color_func, input_range):
    if len(input_range) == 1:
        input_range = 0, input_range[0]

    if len(input_range) == 2:
        input_range = tuple(input_range) + (1,)

    input_range = np.array(input_range)
    if len(input_range) == 3:
        input_range[1] += input_range[2] # add one step to include 'to' value in input_values

    input_values = np.arange(*input_range)
    output_values = np.array([color_func(v) for v in input_values]).astype(np.uint8)

    video_inp = np.copy(video)

    # clamp values
    video_inp = np.minimum(video_inp, input_range[1])
    video_inp = np.maximum(video_inp, input_range[0])

    # convert to indices for output_values
    video_inp -= input_range[0]
    f = float((len(input_values))/(input_range[1] - input_range[0]))
    video_inp *= f
    video_inp = np.floor(video_inp).astype(np.uint32)

    # do all the work
    colorized = output_values[video_inp]
    return colorized


def to_file(filename, video, fps=30):

    def make_frame(t):
        frame = video[int(t*fps)]
        return frame

    clip = mpy.VideoClip(make_frame, duration=np.alen(video)/fps)
    clip.write_videofile(filename, fps=fps)
