import numpy as np
from pyprind import prog_percent

import moviepy.editor as mpy

FFMPEG_BIN = 'ffmpeg'

def orthogonal(rois, traces, color_func, ts, shape):
    activity = np.zeros((np.alen(ts),) + tuple(shape) + (3,), dtype=np.uint8)

    for i, t in prog_percent(list(enumerate((ts)))):
        for roi_id, (roi, trace) in enumerate(zip(rois, traces)):
            x, y, z, _ = roi
            activity[i, x, y] = color_func(trace[t])

    return activity


def to_file(filename, video, fps=30):

    def make_frame(t):
        frame = video[int(t*fps)]
        return frame

    clip = mpy.VideoClip(make_frame, duration=np.alen(video)/fps)
    clip.write_videofile(filename, fps=fps)
