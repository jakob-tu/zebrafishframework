import numpy as np
from skimage.draw import circle
from skimage.feature import match_template, peak_local_max, blob_log
from sklearn.preprocessing import normalize


def valid_frames(invalid_frames, length):
    frames = list(range(length))
    if invalid_frames:
        for f in invalid_frames:
            frames.remove(f)

    return frames


def std(stack, valid_frames=None):
    anatomy_std = []

    if not valid_frames:
        valid_frames = np.arange(np.alen(stack))

    for plane in range(stack.shape[1]):
        #    anatomy_std.append(np.std(trace[displacement[plane]<30, plane], axis=0))
        anatomy_std.append(np.std(stack[valid_frames, plane, ...], axis=0))

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


def get_traces(stack, rois):
    traces = []
    for roi in rois:
        x, y, z, r = list(map(int, roi))
        stencil = circle(x, y, r, stack[0][z].shape)
        trace = np.array(stack[:, z, stencil[0], stencil[1]].mean(1))
        traces.append(trace)
    return np.asarray(traces)


def find_rois_template(img, template, peakMinDistance=2, peakRelThreshold=.2, minIntensity=20):
    m = match_template(img, template, pad_input=True)
    plm = peak_local_max(m, min_distance=peakMinDistance, threshold_rel=peakRelThreshold)
    return np.asarray([(y, x) for x, y in plm if (img[x, y] > minIntensity)])


def find_rois_blob(nda, maxRadius=3, sizeIters=30, threshold=1.5, overlap=0, minIntensity=20):
    rois = []
    for z, img in enumerate(nda):
        blobs = blob_log(normalize(img), max_sigma=maxRadius*0.70710678118, # sigma = radius/sqrt(2)
                         num_sigma=sizeIters, threshold=threshold/np.max(img), overlap=overlap)
        blobs = np.asarray([(y, x, z, r) for x, y, r in blobs if (img[int(x), int(y)] > minIntensity)])
        rois.extend(blobs)
    return rois


def draw_rois(rois, anatomy_std, color_func=None):
    roi_map = np.zeros(anatomy_std.shape + (3,), dtype=np.uint8)

    for plane in range(anatomy_std.shape[0]):
        roi_map[plane, :, :] = (anatomy_std[plane][..., None] / np.max(anatomy_std[plane]) * 255).astype(np.uint8)

    for roi_id, roi in enumerate(rois):
        x, y, z, r, _ = roi.astype(np.int32)
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