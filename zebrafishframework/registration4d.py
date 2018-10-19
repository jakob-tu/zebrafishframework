import numpy as np
from pyprind import prog_bar
from skimage.feature import register_translation
from skimage.transform import AffineTransform, warp
from time import time


def align_andreas(w, align_to_frame=0):
    # Alignment of planes from time series to t=0 in individual brain
    aligned = np.zeros_like(w)
    shifts = []

    # for plane in prog_bar(range(w.shape[1])):
    for plane in [10]:
        print("Working on plane ", plane)
        t = time()
        # Jede plane wird registriert
        for frame in prog_bar(range(w.shape[0])):
            # Jeden Frame
            shift, err, phase = register_translation(w[frame, plane], w[align_to_frame, plane])
            at = AffineTransform(translation=shift)
            aligned[frame, plane] = warp(w[frame, plane], at, preserve_range=True)

            shifts.append([plane, frame, np.sqrt(np.sum(shift ** 2)), err, phase])

        print("Took {:.2f} seconds".format(time() - t))

    shifts = np.array(shifts)

    return aligned, shifts