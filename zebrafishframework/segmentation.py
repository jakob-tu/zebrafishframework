import numpy as np
from skimage.draw import circle
from skimage.feature import match_template, peak_local_max


def std(trace, anatomy, shifts=None):
    anatomy_std = []

    for plane in range(anatomy.shape[0]):
        #    anatomy_std.append(np.std(trace[displacement[plane]<30, plane], axis=0))
        anatomy_std.append(np.std(trace[:, plane, ...], axis=0))

    anatomy_std = np.array(anatomy_std, dtype=np.float32)
    return anatomy_std


def find_rois_andreas(anatomy_std, template, trace, mask=None):
    rois = []
    roi_id = 0
    radius = 5  # px

    for plane in range(anatomy_std.shape[0]):
        # Iterate over planes
        m = match_template(anatomy_std[plane], template, pad_input=True)
        if mask:
            m *= mask[plane]
        plm = peak_local_max(m, min_distance=2, threshold_rel=.2)  # Another sensitivity point is threshold

        for x, y in plm:
            if anatomy_std[plane, x, y] > 20:  # Change intensity if it is too or not sensitive enough
                stencil = circle(x, y, radius, anatomy_std[plane].shape)
                rois.append(dict(x=x, y=y, z=plane, trace=np.array(trace[:, plane, stencil[0], stencil[1]].mean(1)),
                                    radius=radius))
            roi_id += 1

        print('plane %d: %d' % (plane, plm.shape[0]))

    return rois


def draw_rois(rois, anatomy_std, color_func=None):
    roi_map = np.zeros(anatomy_std.shape + (3,), dtype=np.uint8)

    for plane in range(anatomy_std.shape[0]):
        roi_map[plane, :, :] = (anatomy_std[plane][..., None] / np.max(anatomy_std[plane]) * 255).astype(np.uint8)

    for roi_id, roi in enumerate(rois):
        x = roi['x']
        y = roi['y']
        z = roi['z']
        com = circle(x, y, 1.2, anatomy_std[z].shape)
        if color_func:
            color = color_func(roi_id)
        else:
            color = (255, 0, 255)
        roi_map[z][com] = color

    return roi_map


def dFF(traces, pre_range):
#    traces = np.array([v['trace'] for v in rois])
    pre_mean = traces[:, pre_range].mean(1)
    traces_dFF = ((traces.T - pre_mean) / pre_mean).T
    return traces_dFF