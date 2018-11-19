import numpy as np
import deepdish as dd
import os
from pyprind import prog_percent
import sys
from threading import Lock
import torch as th
from torch.autograd import Variable
from torch.nn.functional import affine_grid, grid_sample

from zebrafishframework import io

def toGPU(npArray):
    return Variable(th.from_numpy(npArray.astype(np.float32)).cuda(), requires_grad=False)

def toCPU(tensor):
    return tensor.cpu().numpy()

def tensor3D(shape):
    return th.Tensor(shape[0], shape[1], shape[2]).cuda()

class Alignment:
    def __init__(self, lif_filename, align_to=0):
        self.lif_filename = lif_filename
        self.align_to = align_to

        self.sumMutexGPU = Lock()
        self.sumMutexCPU = Lock()

        self.ir = io.lif_open(lif_filename)
        self.lif_idx = io.lif_find_timeseries(lif_filename)
        # TZYX -> ZXY
        self.lif_shape = io.get_shape(lif_filename, self.lif_idx)
        nt, nz, ny, nx = self.lif_shape
        self.shape = nz, nx, ny
        self.nf = nt
        self.alignmentFrame = io.get_frame(lif_filename, self.align_to, img_i=self.lif_idx)

        self.prepared = False

    def prepare(self):
        self.imgShape = np.zeros(0)
        self.imgDimCount = int()
        self.imgMidpoints = np.zeros(0)
        self.alignmentFrameTensor = th.Tensor()
        self.alignmentFreqs = th.Tensor()
        self.affineGridSize = th.Size()
        self.invalidFrames = np.empty(0)
        self.sumStdDev = np.empty(0)
        self.sumSqStdDev = np.empty(0)
        self.sumTensor = th.Tensor()
        self.sumSqTensor = th.Tensor()

        # sum tensors and invalid frames
        self.sumStdDev   = np.empty(self.shape)
        self.sumSqStdDev = np.empty(self.shape)
        self.sumTensor   = toGPU(self.sumStdDev)
        self.sumSqTensor = toGPU(self.sumSqStdDev)
        self.shifts = np.empty((self.nf, self.shape[0], 2))

        # alignmentFrame
        self.imgDimCount = self.alignmentFrame[0].ndim
        self.imgShape = self.alignmentFrame[0].shape
        self.imgMidpoints = np.array([np.fix(axis_size / 2) for axis_size in self.imgShape])
        self.affineGridSize = th.Size([1, 1, self.imgShape[0], self.imgShape[1]])
        self.alignmentFrameTensor = toGPU(self.alignmentFrame)
        self.alignmentFreqs = th.stack([th.rfft(plane, self.imgDimCount, onesided=True) for plane in self.alignmentFrameTensor])
        self.alignmentFreqs[:,:,:,1] *= -1       # complex conjugate for crosscorrelation

        self.prepared = True

    def run(self):
        if not self.prepared:
            self.prepare()

        frame_stack = np.zeros(self.lif_shape, np.uint16)
        ts = list(range(self.nf))

        for f in prog_percent(ts):
            for z in range(self.lif_shape[1]):
                plane = self.ir.read(z=z, t=f, c=0, series=self.lif_idx, rescale=False)
                frame_stack[f, z] = toCPU(self.alignPlaneGPU(toGPU(plane), z, f))

        """
        self.frameCount -= len(self.invalidFrames)
        self.stdDeviation = toCPU(th.sqrt((self.sumSqTensor[z] - self.sumTensor[z]**2/self.frameCount)/(self.frameCount-1)))
        """

        #stdDevDiffTensor = toGPU(sumSqStdDev[z]) + sumSqTensor[z] - (toGPU(sumStdDev[z]) + sumTensor[z])**2/frameCount
        #stdDeviation = toCPU(th.sqrt(stdDevDiffTensor**2/(frameCount-1)))

        return dict(aligned=frame_stack, shifts=self.shifts)

    def alignPlaneGPU(self, planeTensor, z, f):
        global sumTensor, sumSqTensor, invalidFrames

        shift = self.registerTranslationGPU(planeTensor, z)
        """
        if np.sqrt(np.sum(shift**2)) > self.MAX_DISPLACEMENT:
            np.append(self.invalidFrames, f)
            planeTensor[:,:] = 0
            return planeTensor
        """
        self.shifts[f, z] = shift
        planeTensor = self.warpGPU(planeTensor, shift)
        with self.sumMutexGPU:
            self.sumTensor[z]   += planeTensor
            self.sumSqTensor[z] += planeTensor**2
        return planeTensor

    def registerTranslationGPU(self, imgTensor, alignmentPlaneIdx):
        # Whole-pixel shift - Compute cross-correlation by an IFFT
        imageProduct = th.rfft(imgTensor, self.imgDimCount, onesided=True) * self.alignmentFreqs[alignmentPlaneIdx]
        crossCorrelation = th.irfft(imageProduct, self.imgDimCount)
        # Locate maximum
        maxima = th.argmax(th.abs(crossCorrelation)).cpu().numpy()
        maxima = np.unravel_index(maxima, crossCorrelation.size())
        shifts = np.array(maxima, dtype=np.float64)
        shifts[shifts > self.imgMidpoints] -= np.array(crossCorrelation.shape)[shifts > self.imgMidpoints]
        return shifts

    def warpGPU(self, src, shift):
        # warp src image
        theta = th.Tensor([[[1, 0, shift[0]/self.imgShape[0]],
                            [0, 1, shift[1]/self.imgShape[1]]]]).cuda()
        grid = affine_grid(theta, self.affineGridSize)                      # (N × H × W × 2)
        proj = grid_sample(src.unsqueeze(0).unsqueeze(0), grid)
        return proj.squeeze(0).squeeze(0)

    """
    def alignFrameCPU(self, frame, frameIdx):
        global sumStdDev, sumSqStdDev, invalidFrames
        nz = frame.shape[0]
        shifts = np.empty([nz, 2])
        for z in range(nz):
            shift, _, _ = register_translation(frame[z], alignmentFrame[z])
            if np.sqrt(np.sum(shift**2)) > MAX_DISPLACEMENT:
                np.append(invalidFrames, frameIdx)
                return 0
            shifts[z] = shift
        aligFrame = frame.astype(np.float32) #np.empty_like(tempFrame, dtype=np.float32)
        for z in range(nz):
            aligFrame[z] = warp(aligFrame[z], AffineTransform(translation=shifts[z]), preserve_range=True)
        with sumMutexCPU:
            sumStdDev    += aligFrame
            sumSqStdDev  += aligFrame**2
        return aligFrame

    def alignPlaneCPU(plane, z, f):
        global sumStdDev, sumSqStdDev, invalidFrames

        shift, _, _ = register_translation(plane, alignmentFrame[z])
        if np.sqrt(np.sum(shift**2)) > MAX_DISPLACEMENT:
            np.append(invalidFrames, f)
            plane[:,:] = 0
            return plane
        plane = warp(plane, AffineTransform(translation=shift), preserve_range=True)
        with sumMutexCPU:
            sumStdDev[z]   += plane
            sumSqStdDev[z] += plane**2
        return plane

    """


if __name__ == '__main__':
    print(sys.argv)
    if len(sys.argv) < 3:
        print('Usage: %s input.lif output_base' % sys.argv[0])
        exit(1)
    in_lif, base = sys.argv[1:3]

    try:
        alignment = Alignment(in_lif)
        res = alignment.run()

        aligned = res['aligned']
        shifts = res['shifts']

        print('Saving shifts...')
        np.save(base + '_shifts.npy', shifts)
        print('Saving aligned stack...')
        dd.io.save(base + '_aligned.h5', aligned, 'blosc')
        print('Saving layer 10...')
        # without blosc compression, the fiji hdf5 plugin does not support this atm
        dd.io.save(base+ '_aligned_z10.h5', aligned[:,10,...])
        print('done')
    except Exception as e:
        print(e)

    io.kill_jvm()
    os._exit(0)
