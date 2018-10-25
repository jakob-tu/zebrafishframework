
# coding: utf-8

# In[ ]:


import numpy as np
import torch as th
import h5py
from tqdm import tqdm_notebook
from skimage.external.tifffile import imread, imsave
from skimage.draw import circle
from IPython.display import clear_output
from os import path as op
from sys import path as sp
module_path = op.abspath(op.join('..'))
if module_path not in sp:  # add local path to import helpers
    sp.append(module_path)
import helpers as h
from importlib import reload
import ipywidgets as widgets
import deepdish as dd
from time import time as timer
h = reload(h)


# # Set Parameters

# In[ ]:


CPU_THREADS         = h.getAvailableThreadCount()-2
GPU_THREADS         = 2

ALIGNMENT_FRAME_IDX = 0     # Frame idx to which others will be aligned (starting at 0) 
MAX_DISPLACEMENT    = 30    # if displacement of frame is higher it will be discarded

FILE_NAME          = "./data/fish4_stimulus.lif"
TEMPLATE_FILE_NAME = "./data/cell.tif"
MASK_FILE_NAME     = "./data/fish1_psy_anatomy_mask.tif"

H5_FILE_NAME = FILE_NAME.replace(".lif", ".hdf5")
STD_DEV_FILE_NAME = FILE_NAME.replace(".lif", "_std_div.tif")
INVALID_FRAMES_FILE_NAME = FILE_NAME.replace(".lif", "_invalid_frames.npy")
TRACES_FILE_NAME = FILE_NAME.replace(".lif", "_cell_traces.npy")
CELL_POS_FILE_NAME = FILE_NAME.replace(".lif", "_cell_positions.npy")


# # Read stack and align images (Takes ~13,5 minutes for 80GB)

# In[ ]:


# prepare reader
ir, shape, metaData = h.startLifReader(FILE_NAME, 0)
nf, nz, nx, ny = shape
print("Found image stack of shape  {}".format(shape))

# read and upload Alignment Frame to GPU
h.prepareAlignmentFrame(h.readFrame(ir, ALIGNMENT_FRAME_IDX, nz), MAX_DISPLACEMENT)

# prepare queues for multithreading
frameQueue = h.createQueue(h.alignFrameWorker, (CPU_THREADS+GPU_THREADS)*3, CPU_THREADS, GPU_THREADS)
h.prepareSumTensorsAndInvalidFrames((nz, nx, ny))
imgStack = np.empty((nf, nz, nx, ny), dtype=np.uint16)
t0 = timer()
#for f in tqdm_notebook(range(nf)): frameQueue.put([h.readFrame(ir, f, nz), f, f, imgStack])
#frameQueue.join()
for f in tqdm_notebook(range(nf)): imgStack[f] = h.alignFrameCPU(h.readFrame(ir, f, nz), f)
print(timer() - t0)
np.save("regTime.npy", timer() - t0)
   

# calc standard deviation and save invalid Frame indeces
invalidFrames, stdDeviation, regError = h.getInvalidFramesStdDeviationError(nf, GPU_THREADS>0)
np.save(INVALID_FRAMES_FILE_NAME, sorted(h.invalidFrames))
imsave(STD_DEV_FILE_NAME, stdDeviation)
print("Registrationerror: {}".format(regError))


# In[ ]:


def f(z=0): h.showInlineImage(stdDeviation[z], (8,8))
widgets.interact(f, z=(0, nz-1, 1))


# ## save to HDF5 (Takes long)

# In[ ]:


dd.io.save(H5_FILE_NAME+"noComp", {'stack': [imgStack[:,z] for z in range(nz)]}, compression=('blosc',0))


# In[ ]:


# d = {'t0':{'channel0':chunk[0,0], 'shape':shape, 'element_size_um': metaData['spacing']}}
# dd.io.save(H5_FILE_NAME, d, compression='blosc')


# ## Alternative: read existing registered Stack (Takes time)

# In[ ]:


imgStack = dd.io.load(H5_FILE_NAME, '/stack')


# ### or just read existing Images

# In[ ]:


stdDeviation = imread(STD_DEV_FILE_NAME)
invalidFrames = np.load(INVALID_FRAMES_FILE_NAME)
nz, nx, ny = stdDeviation.shape


# # Find ROIs
# Neuron cells will have changed the most so they should be visible as peaks. We look for peaks to find their positions. 

# In[ ]:



neuronTemplate = imread(TEMPLATE_FILE_NAME)
masks = imread(MASK_FILE_NAME) > 0
cellPositions = [None] * nz
stencils = [None] * nz
for z in tqdm_notebook(range(nz)): 
    m = h.matchTemplateCPU(stdDeviation[z], neuronTemplate)
    plm = h.peakLocalMaxCPU(m, min_distance=2, threshold_rel=.2)
    plm = np.asarray([(y, x) for x, y in plm if (stdDeviation[z, x, y] > 20)])
    cellPositions[z] = plm
cellCount = sum([cp.shape[0] for cp in cellPositions])
np.save(CELL_POS_FILE_NAME, cellPositions)
print("Total cell count: {}".format(cellCount))


# ## Remove invalid Cells by Hand

# In[ ]:


#widgets.interact(f, z=(0, nz-1, 1))
button = widgets.Button(description="Save & Next", button_style='success', icon='check')
display(button)
plot = h.BokehPlot("draw to remove:", stdDeviation[0].shape, (600, 600))
z, x, y = 0, '', '' # x and y are filled in javascipt "selectCallback" of Bokeh
plot.fillPlot(stdDeviation[z], cellPositions[z])
print("Plain {}/{}".format(z+1,nz))
def saveAndNext(b):
    global z, cellPositions
    if x != '':
        cellPositions[z] = np.asarray([(x,y) for x,y in zip(map(int,x.split(',')),map(int,y.split(','))) if y>0])
        np.save(CELL_POS_FILE_NAME, cellPositions)
        cellCount = sum([cp.shape[0] for cp in cellPositions])
    if z+1 == nz: 
        clear_output()
        return
    z += 1
    print("Plain {}/{}".format(z+1,nz))
    plot.fillPlot(stdDeviation[z], cellPositions[z])
button.on_click(saveAndNext)


# In[ ]:


print("Total cell count: {}".format(cellCount))


# ## Optional: See how well found cells fit

# In[ ]:


def f(z=0): h.showInlineScatterAndImage(stdDeviation[z], cellPositions[z], (15,15), "None", "white")
widgets.interact(f, z=(0, nz-1, 1))


# # get Cell Traces through frames (Takes a little)
# Now that we have the cell positions and sizes (as stencils) we can look at those positions in all frames to get the neuron activity in that frame

# In[ ]:


RADIUS = 5 # px
traces = np.asarray([np.empty((len(cps), nf)) for cps in cellPositions])
for z in tqdm_notebook(range(nz)):
    for i, pos in enumerate(cellPositions[z]):
        stencil = circle(pos[0], pos[1], RADIUS, (nx, ny))
        traces[z][i] = imgStack[z][:, stencil[0], stencil[1]].mean(1)
np.save(TRACES_FILE_NAME, traces)


# ## Alternative: load traces if already calculated

# In[ ]:


traces = np.load(TRACES_FILE_NAME)
stdDeviation = imread(STD_DEV_FILE_NAME)
invalidFrames = np.load(INVALID_FRAMES_FILE_NAME)
cellPositions = np.load(CELL_POS_FILE_NAME)
cellCount = sum([cp.shape[0] for cp in cellPositions])
nz, nx, ny = stdDeviation.shape
nz = len(traces)
nf = traces[0].shape[1]


# ## free memory

# In[ ]:


stencils = None
imgStack = None 


# ## calculate ΔF/F

# In[ ]:


CONTROL_FRAMES_START = 1
CONTROL_FRAMES_END   = invalidFrames[0]
cell_dFFs = np.asarray([np.empty((len(trace), nf)) for trace in traces])
for z in range(nz):
    normalCellIntensities = traces[z][:,CONTROL_FRAMES_START:CONTROL_FRAMES_END].mean(1)
    cell_dFFs[z] = ((traces[z].T - normalCellIntensities)/(normalCellIntensities)).T


# In[ ]:


def f(z=0): h.showInlineImage(traces[z], (20,10), vmin=-250, vmax=500)   
widgets.interact(f, z=(0, nz-1, 1))


# In[ ]:


def f(z=0): h.showInlineImage(cell_dFFs[z], (20,10), vmin=-1, vmax=1)   
widgets.interact(f, z=(0, nz-1, 1))


# # Visualize in 3D

# In[ ]:


maxColWgt = widgets.ColorPicker(value='#00ff00', description='more Activity')
midColWgt = widgets.ColorPicker(value='#3333cc', description='same Activity')
minColWgt = widgets.ColorPicker(value='#ff0000', description='less Activity')
display(widgets.VBox(children=[maxColWgt, midColWgt, minColWgt]))


# In[ ]:


MIN_DFF = -1.0
MAX_DFF = 2.0
SIZE_SMALL = 0.1
SIZE_LARGE = 1.0

cellX, cellY, cellZ, cellC, cellS = h.getCellPositionsColorsSizes(cellPositions, cell_dFFs,
                                        maxColWgt.value, midColWgt.value, minColWgt.value,
                                        MIN_DFF, MAX_DFF, SIZE_SMALL, SIZE_LARGE)


# ### Takes time to transfer data to PC for visualization 

# In[ ]:


VIS_FRAMES_START = 1500 #invalidFrames[-1]
VIS_FRAMES_END   = 1600 #nf-1
SPEED = 10   # in frames per second

h.drawInline3DScatter((800,800), (nx,ny,nz), cellX, cellY, cellZ, cellC, cellS, 
                      VIS_FRAMES_START, VIS_FRAMES_END, SPEED)

