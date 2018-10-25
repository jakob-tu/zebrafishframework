
# coding: utf-8

# In[ ]:


import numpy as np
import torch as th
from time import time as timer
import h5py
from tqdm import tqdm_notebook
from skimage.external.tifffile import imread, imsave
from os import path as op
from sys import path as sp
module_path = op.abspath(op.join('..'))
if module_path not in sp:  # add local path to import helpers
    sp.append(module_path)
import helpers as h
from importlib import reload
import ipywidgets as widgets
h = reload(h)


# # Set Parameters

# In[ ]:


CPU_THREADS         = 0 #h.getAvailableThreadCount()-2
GPU_THREADS         = 8 #2

FRAMES_PER_CHUNK    = 50    # needs VRAM
ALIGNMENT_FRAME_IDX = 0     # Frame idx to which others will be aligned (starting at 0) 
MAX_DISPLACEMENT    = 30    # if displacement of frame is higher it will be discarded

FILE_NAME          = "./data/fish4_stimulus.lif"
NEURON_TEMPLATE    = imread("./data/cell.tif")
NEURON_RADIUS      = 5      # px

STD_DEV_FILE_NAME = FILE_NAME.replace(".lif", "_std_div.tif")
CELL_POS_FILE_NAME = FILE_NAME.replace(".lif", "_cell_positions.npy")
CELL_SIZE_FILE_NAME = FILE_NAME.replace(".lif", "_cell_sizes.npy")
CELL_ACT_FILE_NAME = FILE_NAME.replace(".lif", "_cell_activities.npy")
INVALID_FRAMES_FILE_NAME = FILE_NAME.replace(".lif", "_invalid_frames.npy")


# # Read stack and align images (Takes ~13,5 minutes for 80GB)

# In[ ]:


# prepare reader
ir, shape, metaData = h.startLifReader(FILE_NAME, FRAMES_PER_CHUNK)
nf, nz, nx, ny = shape
print("Found image stack of shape  {}".format(shape))

# read and upload Alignment Frame to GPU
h.prepareAlignmentFrame(h.readFrame(ir, ALIGNMENT_FRAME_IDX, nz), MAX_DISPLACEMENT)
h.prepareSumTensorsAndInvalidFrames((nz, nx, ny))

# create Threads and Queue for image alignment
registerQueue = h.createQueue(h.alignPlaneWorker, 0, CPU_THREADS, GPU_THREADS)

stdDeviations = np.empty((nz,nx,ny))
cellPositions = [None] * nz
cellSizes     = [None] * nz
cellActivitys = [None] * nz

planeStack = np.empty((nf, nx, ny), dtype=np.uint16)
for z in tqdm_notebook(range(nz), desc='z Plane'):

    # register planes
    for f in tqdm_notebook(range(nf)): registerQueue.put((h.readPlane(ir,z,f), z, planeStack, f))
    #for f in tqdm_notebook(range(nf)): planeStack[f] = h.toCPU(h.alignPlaneGPU(h.toGPU(h.readPlane(ir,z,f)),z,f))
    #for f in tqdm_notebook(range(nf)): planeStack[f] = h.alignPlaneCPU(h.readPlane(ir,z,f),z,f)
    registerQueue.join()

    # Find Cell Positions
    stdDeviations[z] = h.getPlaneStdDeviation(z, nf)
    #cellPositions[z] = h.findCellPositions(stdDeviations[z], NEURON_TEMPLATE, 2, .2, 20)
    cellPositions[z], cellSizes[z] = h.findCellPositionsBlob(stdDeviations[z])
    print("Found {} Cells".format(cellPositions[z].shape[0]))
    h.showInlineScatterAndImage(stdDeviations[z], cellPositions[z], (10,10), "None", "white", s=cellSizes[z]*10, vmax=100)

    # Read Cell Activity over frames
    cellActivitys[z] = h.getActivityOverTime(cellPositions[z], cellSizes[z], planeStack)

# calc standard deviation and save everything
imsave(STD_DEV_FILE_NAME,   stdDeviations)
np.save(CELL_POS_FILE_NAME, cellPositions)
np.save(CELL_SIZE_FILE_NAME, cellSizes)
np.save(CELL_ACT_FILE_NAME, cellActivitys)
invalidFrames = h.getInvalidFrames()
np.save(INVALID_FRAMES_FILE_NAME, invalidFrames)
print("Total cell count: {}".format(sum([cp.shape[0] for cp in cellPositions])))


# ## Alternative: load traces if already calculated

# In[ ]:


stdDeviations = imread(STD_DEV_FILE_NAME)
cellPositions = np.load(CELL_POS_FILE_NAME)
cellSizes     = np.load(CELL_SIZE_FILE_NAME)
cellActivitys = np.load(CELL_ACT_FILE_NAME)
invalidFrames = np.load(INVALID_FRAMES_FILE_NAME)


# # Remove invalid Cells manually

# In[ ]:


def f(z=0): h.removeCellsByHand(z, stdDeviations, cellPositions, cellSizes, cellActivitys, (900,900), vmax=100)
widgets.interact(f, z=range(stdDeviations.shape[0]))


# ### Save Cell Positions and Activity

# In[ ]:


np.save(CELL_POS_FILE_NAME, cellPositions)
np.save(CELL_SIZE_FILE_NAME, cellSizes)
np.save(CELL_ACT_FILE_NAME, cellActivitys)


# ## calculate ΔF/F

# In[ ]:


CONTROL_FRAMES_START = 160
CONTROL_FRAMES_END   = 180 #invalidFrames[0]
cellActDFFs = h.getDFF(cellActivitys, CONTROL_FRAMES_START, CONTROL_FRAMES_END)


# In[ ]:


def f(z=0): h.showInlineImage(cellActivitys[z], (20,10)) 
widgets.interact(f, z=range(len(cellActivitys)))


# In[ ]:


def f(z=0): h.showInlineImage(cellActDFFs[z], (20,10), vmin=-1, vmax=2)   
widgets.interact(f, z=range(cellActDFFs.shape[0]))


# # Visualize in 3D

# In[ ]:


maxColWgt = widgets.ColorPicker(value='#00ff00', description='more Activity')
midColWgt = widgets.ColorPicker(value='#3333cc', description='same Activity')
minColWgt = widgets.ColorPicker(value='#ff0000', description='less Activity')
display(widgets.VBox(children=[maxColWgt, midColWgt, minColWgt]))


# In[ ]:


MIN_VAL = -1.0
MID_VAL = 0
MAX_VAL = 2.0

SIZE_SMALL = 0.1
SIZE_LARGE = 1.0

cellX, cellY, cellZ, cellC, cellS = h.getCellPositionsColorsSizes(cellPositions, cellActDFFs,
                                        minColWgt.value, midColWgt.value, maxColWgt.value,
                                        MIN_VAL, MID_VAL, MAX_VAL, SIZE_SMALL, SIZE_LARGE)


# ### Takes time to transfer data to PC for visualization 

# In[ ]:


FRAME_RANGE = range(1700, 1800) #range(invalidFrames[-1], nf-1)
SPEED = 10   # in frames per second

nz, nx, ny = stdDeviations.shape
h.drawInline3DScatter((800,800), (nx,ny,nz), cellX, cellY, cellZ, cellC[FRAME_RANGE], cellS[FRAME_RANGE],
                      stdDeviations, 25, SPEED)


# In[ ]:


sum([cps.shape[0] for cps in cellPositions])

