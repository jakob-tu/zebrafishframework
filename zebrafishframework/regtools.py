import ants
import numpy as np
from pyprind import prog_percent
from skimage.draw import circle


def points_to_image(nda, radius=20, shape=(21, 1024, 1024)):
    img = np.zeros(shape, np.uint8)
    for p in nda:
        x, y, z, _ = p
        c = circle(y, x, radius, shape[1:])
        img[int(z)][c] = 255
    return img


# antspy is xyz, we are zyx
def to_ants(img):
    if type(img) == ants.core.ants_image.ANTsImage:
        return img
    if type(img) == np.ndarray:
        if len(img.shape) == 3:
            return ants.from_numpy(img.swapaxes(0, 2).astype(np.uint32))
        elif len(img.shape) == 2:
            return ants.from_numpy(img.swapaxes(0, 1).astype(np.uint32))
    raise ValueError('Cannot convert img')


def to_numpy(img):
    if type(img) == np.ndarray:
        return img
    if type(img) == ants.core.ants_image.ANTsImage:
        if len(img.shape) == 3:
            return img.numpy().swapaxes(0, 2)
        elif len(img.shape) == 2:
            return img.numpy().swapaxes(0, 1)
    raise ValueError('Cannot convert img')


def get_zshift(fixed, moving):
    fixed_ants = to_ants(fixed)
    moving_ants = to_ants(moving)

    res = ants.registration(fixed_ants, moving_ants,
                            type_of_transform='Translation',
                            grad_step=.2)
    t = ants.read_transform(res['fwdtransforms'][0])
    zshift_float = t.parameters[-1]
    zshift = int(round(zshift_float))
    return zshift


def planewise_affine(fixed, moving, return_transforms=False):
    zshift = get_zshift(fixed, moving)

    fixed = to_numpy(fixed)
    moving = to_numpy(moving)

    size_z = fixed.shape[0]

    warped = np.zeros_like(fixed)
    transforms = [None]*size_z
    for z in prog_percent(list(range(max((0, -zshift)), min((size_z, -zshift + size_z))))):
        mov = ants.from_numpy(moving[z + zshift].swapaxes(0, 1))
        fix = ants.from_numpy(fixed[z].swapaxes(0, 1))
        res = ants.registration(mov, fix,
                                type_of_transform='Affine',
                                reg_iterations=[500, 500, 500],
                                grad_step=.1,
                                verbose=True)
        t = ants.read_transform(res['fwdtransforms'][0])
        transforms[z] = t
        trans = ants.apply_ants_transform_to_image(t, mov, fix)
        warped[z] = trans.numpy().swapaxes(0, 1)

    if return_transforms:
        return warped, (transforms, zshift)

    return warped


def transform_planewise_points(nda, transforms_zshift):
    transforms, zshift = transforms_zshift

    points = []
    size_z = len(transforms)
    for p in nda:
        x, y, z, _ = p
        fixed_z = int(round(z - zshift))
        if fixed_z < 0 or fixed_z >= size_z:
            continue
        t = transforms[fixed_z]

        if t:
            transformed = ants.apply_ants_transform_to_point(t, p)
            p_new = int(round(transformed[0])), int(round(transformed[1])), fixed_z
            points.append(p_new)

    return np.array(points)

