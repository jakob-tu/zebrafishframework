import sys
import os
from xml import etree as et
from queue import Queue
from threading import Thread, Lock
import matplotlib.pyplot as plt
import matplotlib.cm as mplColorMap
from skimage.feature import register_translation, peak_local_max, match_template, blob_log
from skimage.draw import circle
from skimage.transform import AffineTransform, warp
from sklearn.preprocessing import normalize
import torch as th
from torch.autograd import Variable
from torch.nn.functional import grid_sample, affine_grid, pad, conv2d
from torch.nn import MaxPool2d
import torchvision.transforms as tvTransforms
import numpy as np
import javabridge as jv
import bioformats as bf
import time
import re
import tables
import h5py
#import ipyvolume as ipv
import ipywidgets as widgets

EPSILON = np.finfo(np.float64).eps
MAX_DISPLACEMENT = 0
sumMutexGPU = Lock()
sumMutexCPU = Lock()

#get_ipython().magic(u'matplotlib inline')



# lif reader
    
tempFrame = np.empty(0)
tempChunk = np.empty(0)
tempPlaneStack = np.empty(0)
tempPlaneChunk = np.empty(0)
framesPerChunk = 0
def startLifReader(filePath, framesPChunk):
    global tempFrame, tempChunk, framesPerChunk, tempPlaneStack, tempPlaneChunk, jvmRunning
    framesPerChunk = framesPChunk
    jv.start_vm(class_path=bf.JARS)
    ir = bf.ImageReader(filePath, perform_init=True)
    mdroot = et.ElementTree.fromstring(bf.get_omexml_metadata(filePath))
    mds = list(map(lambda e: e.attrib, mdroot.iter('{http://www.openmicroscopy.org/Schemas/OME/2016-06}Pixels')))

    # lif can contain multiple images, select one that is likely to be the timeseries
    metaData = None
    for idx, md in enumerate(mds): 
        if int(md['SizeT']) > 1: 
            lifImgStackIdx = idx
            metaData       = md
    if not metaData: raise ValueError('lif does not contain an image with sizeT > 1')
    nf = int(metaData['SizeT'])
    nz = int(metaData['SizeZ'])
    nx = int(metaData['SizeX'])
    ny = int(metaData['SizeY'])
    
    tempFrame = np.empty((nz,nx,ny), dtype=np.uint16)
    tempChunk = np.empty(((framesPerChunk,) + tempFrame.shape), dtype=np.uint16)
    tempPlaneStack = np.empty((nf,nx,ny), dtype=np.uint16)
    tempPlaneChunk = np.empty((framesPChunk,nx,ny), dtype=np.uint16)
    metaData = {
        "order": metaData['DimensionOrder'],
        "spacing": tuple([float(metaData['PhysicalSize%s' % c]) for c in 'XYZ'])
    }
    return ir, (nf,nz,nx,ny), metaData
    
def closeLifReader():
    # jv.kill_vm() buggy
    return


def readFrame(lifReader, f, planeCount, lifImgStackIdx):
    frame = np.empty_like(tempFrame)
    for z in range(planeCount):
        frame[z] = lifReader.read(t=f, z=z, c=0, series=lifImgStackIdx, rescale=False)
    return frame

def readPlane(lifReader, z, f, lifImgStackIdx):
    return lifReader.read(t=f, z=z, c=0, series=lifImgStackIdx, rescale=False)




# IO and System Functions

def toGPU(npArray):
    return Variable(th.from_numpy(npArray.astype(np.float32)).cuda(), requires_grad=False)

def toCPU(tensor):
    return tensor.cpu().numpy()

def tensor3D(shape):
    return th.Tensor(shape[0], shape[1], shape[2]).cuda()

def getAvailableThreadCount():
    if sys.platform == 'win32':
        return (int)(os.environ['NUMBER_OF_PROCESSORS'])
    return (int)(os.popen('grep -c cores /proc/cpuinfo').read())

def removeIfExists(filename):
    try:
        os.remove(filename)
    except OSError:
        pass

def createQueue(function, maxSize=0, cpuThreads=1, gpuThreads=0):
    q = Queue(maxsize=maxSize)
    for i in range(cpuThreads):
        t = Thread(target=function, args=(q,False))
        t.setDaemon(True)
        t.start()
    for i in range(gpuThreads):
        t = Thread(target=function, args=(q,True))
        t.setDaemon(True)
        t.start()
    return q


# MatPlotLib

def showInlineImage(img, size=(5,5), vmin=None, vmax=None):
    plt.figure(figsize=size)
    plt.imshow(img, vmin=vmin, vmax=vmax)
    plt.colorbar()
    #plt.savefig("plot.png")
    plt.show()
    
def saveImage(img, name, size=(5,5), dpi=80, vmax=None):
    fig = plt.figure()
    fig.set_size_inches(size)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(img, aspect='equal', vmax=vmax)
    plt.savefig(name, dpi=dpi)

def showInlineScatter(points, size=(5,5)):
    if isinstance(size, tuple): plt.figure(figsize=size)
    else: plt.figure(figsize=(size, size))
    plt.scatter(points[:,0], points[:,1])
    plt.show()
    
def showInlineScatterColored(points, size=(5,5), colors=[]):
    if isinstance(size, tuple): plt.figure(figsize=size)
    else: plt.figure(figsize=(size, size))
    plt.scatter(points[:,0], points[:,1], c=colors)
    plt.colorbar()

def showInlineScatterAndImage(img, points, size=(5,5), facecolors=None, edgecolors='white', vmin=None, vmax=None, s=None):
    plt.figure(figsize=size)
    plt.imshow(img, zorder=0, vmin=vmin, vmax=vmax)
    plt.scatter(points[:,0], points[:,1], s=s, edgecolors=edgecolors, facecolors=facecolors)
    plt.show()
    
    

# bokeh

y = ''  # y is filled in javascipt "selectCallback" of Bokeh
def removeCellsByHand(z, stdDevs, cellPoss, cellSizes, cellActs, size=(600, 600), vmax=None):
    from bokeh.models import LassoSelectTool, BoxSelectTool, ResetTool, WheelZoomTool, BoxZoomTool
    from bokeh.plotting import figure, show, output_notebook
    from bokeh.models import CustomJS, ColumnDataSource, Circle, Toolbar, ToolbarPanel
    from bokeh.io import push_notebook, reset_output
    from bokeh.layouts import row
    output_notebook(hide_banner=True)
    
    stdDev = stdDevs[z]
    if vmax != None: stdDev = np.clip(stdDev, 0, vmax)
    nx, ny = stdDev.shape
    cellPos = cellPoss[z]
    cellAct = cellActs[z]
    
    def save(b):
        global y
        if y == '': return
        remIdxs = np.asarray([idx for idx, y in enumerate(map(float,y.split(','))) if y < 0])
        y = ''
        cellPoss[z] = np.delete(cellPoss[z], remIdxs, axis=0)
        cellSizes[z] = np.delete(cellSizes[z], remIdxs, axis=0)
        cellActs[z] = np.delete(cellActs[z], remIdxs, axis=0)
    button = widgets.Button(description="Save", button_style='success', icon='check')
    button.on_click(save)
    display(button)
    
    source = ColumnDataSource(data=dict(x=[],y=[]))
    selectCallback = CustomJS(args=dict(s=source), code="""
        var idxs = s.selected.indices;
        if (idxs.length == 0) return;
        var d = s.data;
        for (var i = 0; i < idxs.length; i++) d['y'][idxs[i]] = -1000;
        s.change.emit();
        cmd = "h.y = '" + d['y'] + "'";
        IPython.notebook.kernel.execute(cmd, {}, {});
        """)
    lassoTool = LassoSelectTool(callback=selectCallback, select_every_mousemove=False)
    boxTool   = BoxSelectTool(callback=selectCallback, select_every_mousemove=False)
    wheelTool = WheelZoomTool(zoom_on_axis=False)
    tools = [WheelZoomTool(zoom_on_axis=False), ResetTool(), BoxZoomTool(), lassoTool, boxTool]
    
    p = figure(title="draw to remove:", output_backend="webgl", toolbar_location="above", tools=tools,
               x_range=(0, nx), y_range=(ny,0), plot_width=size[0], plot_height=size[1])
    p.xgrid.visible = p.ygrid.visible = False
    p.toolbar_location = None
    p.add_layout(ToolbarPanel(toolbar=Toolbar(tools=tools)), "right")
    
    lassoOverlay = p.select_one(LassoSelectTool).overlay
    lassoOverlay.fill_color = "firebrick"
    lassoOverlay.line_color = "white"
    
    h = show(row(p), notebook_handle=True)
    
    p.image(image=[stdDev[::-1]], x=0, y=ny, dw=nx, dh=ny, palette="Viridis256", level="image")
    scat = p.circle('x', 'y', color='red', size=5, source=source, fill_color='red', alpha=1)
    scat.data_source.data = dict(x=cellPos[:,0], y=cellPos[:,1])
    scat.nonselection_glyph = Circle(line_color='red', fill_color='red')
    
    push_notebook(handle=h)
    
    
# image alignment

imgShape = np.zeros(0)
imgDimCount = int()
imgMidpoints = np.zeros(0)
alignmentFrame = np.zeros(0)
alignmentFrameTensor = th.Tensor()
alignmentFreqs = th.Tensor()
affineGridSize = th.Size()
def prepareAlignmentFrame(frame, maxDisplacement):
    global imgShape, imgDimCount, imgMidpoints, alignmentFrame, affineGridSize, MAX_DISPLACEMENT
    global alignmentFreqs, alignmentFrameTensor
    MAX_DISPLACEMENT = maxDisplacement
    imgDimCount = frame[0].ndim
    imgShape = frame[0].shape
    imgMidpoints = np.array([np.fix(axis_size / 2) for axis_size in imgShape])
    affineGridSize = th.Size([1, 1, imgShape[0], imgShape[1]])
    alignmentFrame = frame
    alignmentFrameTensor = toGPU(frame)
    alignmentFreqs = th.stack([th.rfft(plane, imgDimCount, onesided=True) for plane in alignmentFrameTensor])
    alignmentFreqs[:,:,:,1] *= -1       # complex conjugate for crosscorrelation
        
invalidFrames = np.empty(0)
sumStdDev = np.empty(0)
sumSqStdDev = np.empty(0)
sumTensor = th.Tensor()
sumSqTensor = th.Tensor()
def prepareSumTensorsAndInvalidFrames(shape):
    global sumStdDev, sumSqStdDev, sumTensor, sumSqTensor, invalidFrames
    sumStdDev   = np.empty(shape)
    sumSqStdDev = np.empty(shape)
    sumTensor   = toGPU(sumStdDev)
    sumSqTensor = toGPU(sumSqStdDev)
    invalidFrames = np.empty(0)

def alignFrameWorker(queue, useGpu):
    
    def alignFrameGPU(frameTensor, frameIdx):
        global sumTensor, sumSqTensor, invalidFrames

        nz = frameTensor.shape[0]
        shifts = np.empty([nz, 2])
        for z in range(nz):
            shift = registerTranslationGPU(frameTensor[z], z)
            if np.sqrt(np.sum(shift**2)) > MAX_DISPLACEMENT:
                np.append(invalidFrames, frameIdx)
                frameTensor[:,:,:] = 0
                return frameTensor
            shifts[z] = shift
        aligFrameTensor = th.stack([warpGPU(frameTensor[z], shifts[z]) for z in range(nz)])
        with sumMutexGPU:
            sumTensor    += aligFrameTensor
            sumSqTensor  += aligFrameTensor**2
        return aligFrameTensor

    if useGpu:
        while(True):
            frame, frameIdx, dstFrameIdx, dst = queue.get()
            dst[dstFrameIdx] = toCPU(alignFrameGPU(toGPU(frame), frameIdx))
            queue.task_done()
    else:
        while(True):
            frame, frameIdx, dstFrameIdx, dst = queue.get()
            dst[dstFrameIdx] = alignFrameCPU(frame, frameIdx)
            queue.task_done()
    return
        
def registerTranslationGPU(imgTensor, alignmentPlaneIdx):
    # Whole-pixel shift - Compute cross-correlation by an IFFT
    imageProduct = th.rfft(imgTensor, imgDimCount, onesided=True) * alignmentFreqs[alignmentPlaneIdx]
    crossCorrelation = th.irfft(imageProduct, imgDimCount)
    # Locate maximum
    maxima = th.argmax(th.abs(crossCorrelation)).cpu().numpy()
    maxima = np.unravel_index(maxima, crossCorrelation.size())
    shifts = np.array(maxima, dtype=np.float64)
    shifts[shifts > imgMidpoints] -= np.array(crossCorrelation.shape)[shifts > imgMidpoints]
    return shifts

def warpGPU(src, shift):
    # warp src image
    theta = th.Tensor([[[1, 0, shift[0]/imgShape[0]],
                        [0, 1, shift[1]/imgShape[1]]]]).cuda()
    grid = affine_grid(theta, affineGridSize)                      # (N × H × W × 2)
    proj = grid_sample(src.unsqueeze(0).unsqueeze(0), grid)
    return proj.squeeze(0).squeeze(0)

def alignFrameGPU(frame, frameIdx):
    global sumTensor, sumSqTensor, invalidFrames
    frameTensor = toGPU(frame)
    nz = frameTensor.shape[0]
    shifts = np.empty([nz, 2])
    for z in range(nz):
        shift = registerTranslationGPU(frameTensor[z], z)
        if np.sqrt(np.sum(shift**2)) > MAX_DISPLACEMENT:
            np.append(invalidFrames, frameIdx)
            frameTensor[:,:,:] = 0
            return frameTensor
        shifts[z] = shift
    aligFrameTensor = th.stack([warpGPU(frameTensor[z], shifts[z]) for z in range(nz)])
    with sumMutexGPU:
        sumTensor    += aligFrameTensor
        sumSqTensor  += aligFrameTensor**2
    return toCPU(aligFrameTensor)

def getInvalidFramesStdDeviationError(frameCount, gpuWasUsed):
    frameCount -= len(invalidFrames)
    
    def getStdDev():
        if not gpuWasUsed:
            return np.sqrt((sumSqStdDev - sumStdDev**2/frameCount)/(frameCount-1))
        return toCPU(th.sqrt((sumSqTensor - sumTensor**2/frameCount)/(frameCount-1)))
        
    stdDeviation = getStdDev()
    return sorted(invalidFrames), stdDeviation, 0
        
def alignFrameCPU(frame, frameIdx):
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

def alignPlaneGPU(planeTensor, z, f):
    global sumTensor, sumSqTensor, invalidFrames
    
    shift = registerTranslationGPU(planeTensor, z)
    if np.sqrt(np.sum(shift**2)) > MAX_DISPLACEMENT:
        np.append(invalidFrames, f)
        planeTensor[:,:] = 0
        return planeTensor
    planeTensor = warpGPU(planeTensor, shift)
    with sumMutexGPU:
        sumTensor[z]   += planeTensor
        sumSqTensor[z] += planeTensor**2
    return planeTensor

def alignPlaneWorker(queue, useGpu):
    if useGpu:
        while(True):
            plane, z, planeStack, f = queue.get()
            planeStack[f] = toCPU(alignPlaneGPU(toGPU(plane), z, f))
            queue.task_done()
    else:
        while(True):
            plane, z, planeStack, f = queue.get()
            planeStack[f] = alignPlaneCPU(plane, z, f)
            queue.task_done()
    return

def getPlaneStdDeviation(z, frameCount):
    global invalidFrames
    invalidFrames = np.unique(np.sort(invalidFrames))
    frameCount -= len(invalidFrames)
    
    stdDeviation = toCPU(th.sqrt((sumSqTensor[z] - sumTensor[z]**2/frameCount)/(frameCount-1)))
    
    #stdDevDiffTensor = toGPU(sumSqStdDev[z]) + sumSqTensor[z] - (toGPU(sumStdDev[z]) + sumTensor[z])**2/frameCount
    #stdDeviation = toCPU(th.sqrt(stdDevDiffTensor**2/(frameCount-1)))
    return stdDeviation

def getInvalidFrames():
    return np.unique(np.sort(invalidFrames))


# ROI Detection

def findCellPositions(img, template, peakMinDistance=2, peakRelThreshold=.2, minIntensity=20):
    m = match_template(img, template, pad_input=True)
    plm = peak_local_max(m, min_distance=peakMinDistance, threshold_rel=peakRelThreshold)
    return np.asarray([(y, x) for x, y in plm if (img[x, y] > minIntensity)])

def findCellPositionsBlob(img, maxRadius=3, sizeIters=30, threshold=1.5, overlap=0, minIntensity=20):
    blobs = blob_log(normalize(img), max_sigma=maxRadius*0.70710678118, # sigma = radius/sqrt(2)
                     num_sigma=sizeIters, threshold=threshold/np.max(img), overlap=overlap) 
    blobs = np.asarray([(y, x, r) for x, y, r in blobs if (img[int(x), int(y)] > minIntensity)])
    return blobs[:,(0,1)], blobs[:,2]
    
def getActivityOverTime(cellPositions, cellSizes, planeStack):
    nf, nx, ny = planeStack.shape
    cellActivitys = np.empty((cellPositions.shape[0], nf))
    for i, pos in enumerate(cellPositions):
        stencil = circle(pos[0], pos[1], cellSizes[i], (nx, ny))
        cellActivitys[i] = planeStack[:, stencil[0], stencil[1]].mean(1)
    return cellActivitys



# iPyVolume Visualization

def getDFF(cellActivitys, controlStart, controlEnd):
    dff = np.asarray([np.empty(a.shape) for a in cellActivitys])
    for z in range(len(cellActivitys)):
        normalCellIntensities = cellActivitys[z][:,controlStart:controlEnd].mean(1)
        dff[z] = ((cellActivitys[z].T - normalCellIntensities)/(normalCellIntensities)).T
    return dff

def getCellPositionsColorsSizes(cellPos, vals, cMin, cMid, cMax, minVal, midVal, maxVal, minSize=0.1, maxSize=1):
    def hexToRGB(hexString):
        return np.asarray([int(hexString.lstrip('#')[i:i+2], 16) for i in (0, 2 ,4)])
    cMin, cMid, cMax = (hexToRGB(cMin)/255, hexToRGB(cMid)/255, hexToRGB(cMax)/255)
    vals = np.concatenate(vals)
    vals = np.clip(np.where(vals < midVal, vals/(-minVal), vals/maxVal), -1, 1)
    cellX = np.concatenate([cps[:,0].astype(np.float32) for cps in cellPos])
    cellY = np.concatenate([cps[:,1].astype(np.float32) for cps in cellPos])
    cellZ = np.concatenate([np.full(cps.shape[0], z, dtype=float) for z, cps in enumerate(cellPos)])
    cellC = np.empty((vals.shape[0], vals.shape[1], 3))
    cellS = np.empty(vals.shape)
    for i, v in enumerate(vals):
        cellC[i] = np.where(v < midVal, np.multiply.outer(cMin, 0-v) + np.multiply.outer(cMid, 1+v), 
                                        np.multiply.outer(cMid, 1-v) + np.multiply.outer(cMax, 0+v)).T
        cellS[i] = minSize + (maxSize-minSize)*np.absolute(v)
    return cellX, cellY, cellZ, cellC.swapaxes(0,1), cellS.swapaxes(0,1)

def drawInline3DScatter(size, xyzLimits, xData, yData, zData, cData, sData, volume=None, volMaxVal=30, speed=0):
    ipv.figure(width=size[0], height=size[1])
    ipv.xlim(0, xyzLimits[0])
    ipv.ylim(0, xyzLimits[1])
    ipv.zlim(0-xyzLimits[2]//1.5, xyzLimits[2]+xyzLimits[2]//1.5)
    ipv.pylab.style.axes_off()
    ipv.pylab.style.set_style_dark()
    if type(volume) is np.ndarray:
        ipv.volshow(np.where(volume<volMaxVal, 0, 1), level=(1,1,1), opacity=(0.0075, 0.0075,0.0075))
    scatterPlot = ipv.scatter(xData, yData, zData, cData,
                              marker='circle_2d', size=sData)
    ipv.animation_control(scatterPlot, interval=1000/speed, sequence_length=cData.shape[0])
    ipv.show()

    
 
