
# coding: utf-8

# In[ ]:


import numpy as np
import torch as th
import h5py
from tqdm import tqdm_notebook
from time import time as timer
from skimage.external.tifffile import imsave
from IPython.display import clear_output
from os import path as op
from sys import path as sp
module_path = op.abspath(op.join('..'))
if module_path not in sp:  # add local path to import helpers
    sp.append(module_path)
import helpers as h
from importlib import reload
from IPython.display import display_html
import deepdish as dd
from types import SimpleNamespace as NS
h = reload(h)


# In[ ]:


CPU_THREADS         = 0 #h.getAvailableThreadCount()-2
GPU_THREADS         = 8 #2

FRAMES_PER_CHUNK    = 16    # more Frames will need more RAM but speed up process (max: 95)
COMPRESSION         = None  # http://docs.h5py.org/en/latest/high/dataset.html#lossless-compression-filters
ALIGNMENT_FRAME_IDX = 0     # Frame idx to which others will be aligned (starting at 0) 
MAX_DISPLACEMENT    = 30    # if displacement of frame is higher it will be discarded
DELETE_LAST_H5_FILE = True  # if h5-File already exists it will be deleted

FILE_NAME = "./data/fish4_stimulus.lif"
TEMPLATE_FILE_NAME = "./data/cell.tif"

H5_FILE_NAME = FILE_NAME.replace(".lif", ".hdf5")
STD_DEV_FILE_NAME = FILE_NAME.replace(".lif", "_std_div.tif")
INVALID_FRAMES_FILE_NAME = FILE_NAME.replace(".lif", "_invalid_frames.npy")
DATA_SET_NAME = "ImageStack"


# # Read stack, align images and save to hdf5

# In[ ]:


MAX_THREADS = 1
# prepare reader
ir, shape, metaData = h.startLifReader(FILE_NAME, FRAMES_PER_CHUNK)
nf, nz, nx, ny = shape
print("Found image stack of shape  {}".format(shape))

# prepare chunks
chunkShape = (FRAMES_PER_CHUNK, nz, nx, ny)
chunkCount = nf // FRAMES_PER_CHUNK
chunkPair = (np.empty(chunkShape, dtype=np.uint16), np.empty(chunkShape, dtype=np.uint16))

if (DELETE_LAST_H5_FILE): h.removeIfExists(H5_FILE_NAME)
    
# create Threads and Queues for image alignment
frameQueuePair = (h.createQueue(h.alignFrameWorker, FRAMES_PER_CHUNK, CPU_THREADS, GPU_THREADS),
                  h.createQueue(h.alignFrameWorker, FRAMES_PER_CHUNK, CPU_THREADS, GPU_THREADS))

h.prepareAlignmentFrame(h.readFrame(ir, ALIGNMENT_FRAME_IDX, nz), MAX_DISPLACEMENT)
h.prepareSumTensorsAndInvalidFrames((nz, nx, ny))
t0 = timer()
for chunkIdx in tqdm_notebook(range(chunkCount)):

    # alternate chunks and queues to read current chunk and write last results into last chunk simultaneously
    a, b = (0,1) if chunkIdx % 2 else (1,0)
    thisChunk,      lastChunk      = (chunkPair[a],      chunkPair[b]) 
    thisFrameQueue, lastFrameQueue = (frameQueuePair[a], frameQueuePair[b])
    
    # read Frames and put them into current queue to be processed
    chunkStart = chunkIdx*FRAMES_PER_CHUNK
    for f in range(FRAMES_PER_CHUNK if chunkIdx < chunkCount-1 else nf % FRAMES_PER_CHUNK):
        thisFrameQueue.put([h.readFrame(ir, chunkStart+f, nz), chunkStart+f, f, thisChunk])
    
    # save last chunk while processing this one (except at first chunk)
    if chunkIdx > 0:
        lastFrameQueue.join()
        lastChunkStart = chunkStart - FRAMES_PER_CHUNK
        dSetName = "chunk" + str(chunkIdx-1)
        dd.io.save("test.h5", {dSetName: lastChunk}, compression=('blosc',6))
        #h5File[DATA_SET_NAME][:, lastChunkStart:chunkStart] = lastChunk.swapaxes(0,1)
        
    # at last chunk: save it because there is no next
    if chunkIdx == chunkCount-1 or True: 
        thisFrameQueue.join()
        chunkEnd = chunkStart + FRAMES_PER_CHUNK
        dSetName = "chunk" + str(chunkIdx)
        dd.io.save("test.h5", {dSetName: thisChunk}, compression=('blosc',6))
        #h5File[DATA_SET_NAME][:, chunkStart:chunkEnd] = thisChunk.swapaxes(0,1)
np.save("regTime1.npy", timer() - t0)
print(timer() - t0)


# In[ ]:


MAX_THREADS = 2
# prepare reader
ir, shape, metaData = h.startLifReader(FILE_NAME, FRAMES_PER_CHUNK)
nf, nz, nx, ny = shape
print("Found image stack of shape  {}".format(shape))

# prepare chunks
chunkShape = (FRAMES_PER_CHUNK, nz, nx, ny)
chunkCount = nf // FRAMES_PER_CHUNK
chunkPair = (np.empty(chunkShape, dtype=np.uint16), np.empty(chunkShape, dtype=np.uint16))

if (DELETE_LAST_H5_FILE): h.removeIfExists(H5_FILE_NAME)
    
# create Threads and Queues for image alignment
frameQueuePair = (h.createQueue(h.alignFrameWorker, FRAMES_PER_CHUNK, MAX_THREADS, USE_GPU),
                  h.createQueue(h.alignFrameWorker, FRAMES_PER_CHUNK, MAX_THREADS, USE_GPU))

h.prepareAlignmentFrame(h.readFrame(ir, ALIGNMENT_FRAME_IDX, nz), MAX_DISPLACEMENT)
h.prepareSumTensorsAndInvalidFrames((nz, nx, ny))
t0 = timer()
for chunkIdx in tqdm_notebook(range(chunkCount)):

    # alternate chunks and queues to read current chunk and write last results into last chunk simultaneously
    a, b = (0,1) if chunkIdx % 2 else (1,0)
    thisChunk,      lastChunk      = (chunkPair[a],      chunkPair[b]) 
    thisFrameQueue, lastFrameQueue = (frameQueuePair[a], frameQueuePair[b])
    
    # read Frames and put them into current queue to be processed
    chunkStart = chunkIdx*FRAMES_PER_CHUNK
    for f in range(FRAMES_PER_CHUNK if chunkIdx < chunkCount-1 else nf % FRAMES_PER_CHUNK):
        thisFrameQueue.put([h.readFrame(ir, chunkStart+f, nz), chunkStart+f, f, thisChunk])
    
    # save last chunk while processing this one (except at first chunk)
    if chunkIdx > 0:
        lastFrameQueue.join()
    #    lastChunkStart = chunkStart - FRAMES_PER_CHUNK
    #    h5File[DATA_SET_NAME][:, lastChunkStart:chunkStart] = lastChunk.swapaxes(0,1)
        
    # at last chunk: save it because there is no next
    if chunkIdx == chunkCount-1 or True: 
        thisFrameQueue.join()
    #    chunkEnd = chunkStart + FRAMES_PER_CHUNK
    #    h5File[DATA_SET_NAME][:, chunkStart:chunkEnd] = thisChunk.swapaxes(0,1)
np.save("regTime2.npy", timer() - t0)


# In[ ]:


MAX_THREADS = 4
# prepare reader
ir, shape, metaData = h.startLifReader(FILE_NAME, FRAMES_PER_CHUNK)
nf, nz, nx, ny = shape
print("Found image stack of shape  {}".format(shape))

# prepare chunks
chunkShape = (FRAMES_PER_CHUNK, nz, nx, ny)
chunkCount = nf // FRAMES_PER_CHUNK
chunkPair = (np.empty(chunkShape, dtype=np.uint16), np.empty(chunkShape, dtype=np.uint16))

if (DELETE_LAST_H5_FILE): h.removeIfExists(H5_FILE_NAME)
    
# create Threads and Queues for image alignment
frameQueuePair = (h.createQueue(h.alignFrameWorker, FRAMES_PER_CHUNK, MAX_THREADS, USE_GPU),
                  h.createQueue(h.alignFrameWorker, FRAMES_PER_CHUNK, MAX_THREADS, USE_GPU))

h.prepareAlignmentFrame(h.readFrame(ir, ALIGNMENT_FRAME_IDX, nz), MAX_DISPLACEMENT)
h.prepareSumTensorsAndInvalidFrames((nz, nx, ny))
t0 = timer()
for chunkIdx in tqdm_notebook(range(chunkCount)):

    # alternate chunks and queues to read current chunk and write last results into last chunk simultaneously
    a, b = (0,1) if chunkIdx % 2 else (1,0)
    thisChunk,      lastChunk      = (chunkPair[a],      chunkPair[b]) 
    thisFrameQueue, lastFrameQueue = (frameQueuePair[a], frameQueuePair[b])
    
    # read Frames and put them into current queue to be processed
    chunkStart = chunkIdx*FRAMES_PER_CHUNK
    for f in range(FRAMES_PER_CHUNK if chunkIdx < chunkCount-1 else nf % FRAMES_PER_CHUNK):
        thisFrameQueue.put([h.readFrame(ir, chunkStart+f, nz), chunkStart+f, f, thisChunk])
    
    # save last chunk while processing this one (except at first chunk)
    if chunkIdx > 0:
        lastFrameQueue.join()
    #    lastChunkStart = chunkStart - FRAMES_PER_CHUNK
    #    h5File[DATA_SET_NAME][:, lastChunkStart:chunkStart] = lastChunk.swapaxes(0,1)
        
    # at last chunk: save it because there is no next
    if chunkIdx == chunkCount-1 or True: 
        thisFrameQueue.join()
    #    chunkEnd = chunkStart + FRAMES_PER_CHUNK
    #    h5File[DATA_SET_NAME][:, chunkStart:chunkEnd] = thisChunk.swapaxes(0,1)
np.save("regTime4.npy", timer() - t0)


# In[ ]:


MAX_THREADS = 8
# prepare reader
ir, shape, metaData = h.startLifReader(FILE_NAME, FRAMES_PER_CHUNK)
nf, nz, nx, ny = shape
print("Found image stack of shape  {}".format(shape))

# prepare chunks
chunkShape = (FRAMES_PER_CHUNK, nz, nx, ny)
chunkCount = nf // FRAMES_PER_CHUNK
chunkPair = (np.empty(chunkShape, dtype=np.uint16), np.empty(chunkShape, dtype=np.uint16))

if (DELETE_LAST_H5_FILE): h.removeIfExists(H5_FILE_NAME)
    
# create Threads and Queues for image alignment
frameQueuePair = (h.createQueue(h.alignFrameWorker, FRAMES_PER_CHUNK, MAX_THREADS, USE_GPU),
                  h.createQueue(h.alignFrameWorker, FRAMES_PER_CHUNK, MAX_THREADS, USE_GPU))

h.prepareAlignmentFrame(h.readFrame(ir, ALIGNMENT_FRAME_IDX, nz), MAX_DISPLACEMENT)
h.prepareSumTensorsAndInvalidFrames((nz, nx, ny))
t0 = timer()
for chunkIdx in tqdm_notebook(range(chunkCount)):

    # alternate chunks and queues to read current chunk and write last results into last chunk simultaneously
    a, b = (0,1) if chunkIdx % 2 else (1,0)
    thisChunk,      lastChunk      = (chunkPair[a],      chunkPair[b]) 
    thisFrameQueue, lastFrameQueue = (frameQueuePair[a], frameQueuePair[b])
    
    # read Frames and put them into current queue to be processed
    chunkStart = chunkIdx*FRAMES_PER_CHUNK
    for f in range(FRAMES_PER_CHUNK if chunkIdx < chunkCount-1 else nf % FRAMES_PER_CHUNK):
        thisFrameQueue.put([h.readFrame(ir, chunkStart+f, nz), chunkStart+f, f, thisChunk])
    
    # save last chunk while processing this one (except at first chunk)
    if chunkIdx > 0:
        lastFrameQueue.join()
    #    lastChunkStart = chunkStart - FRAMES_PER_CHUNK
    #    h5File[DATA_SET_NAME][:, lastChunkStart:chunkStart] = lastChunk.swapaxes(0,1)
        
    # at last chunk: save it because there is no next
    if chunkIdx == chunkCount-1 or True: 
        thisFrameQueue.join()
    #    chunkEnd = chunkStart + FRAMES_PER_CHUNK
    #    h5File[DATA_SET_NAME][:, chunkStart:chunkEnd] = thisChunk.swapaxes(0,1)
np.save("regTime8.npy", timer() - t0)


# In[ ]:


np.load("regTime1.npy")


# In[ ]:


#nf=20

plainQueue = Queue(maxsize=2)
for i in range(MAX_THREADS): h.createAndStartThreadWithQueue(h.alignPlainWorker, plainQueue)
   
h.prepareSumTensorsAndInvalidFrames((nz, nx, ny))
for z in tqdm_notebook(range(nz)):
    plainStack = h.readPlainStack(ir, z, nf)
    
    # push chunkwise to GPU because of limited Memory
    for chunkf in range(0, nf, FRAMES_PER_CHUNK):
        fRange = range(chunkf, chunkf+FRAMES_PER_CHUNK)
        plainChunkTensor = h.toGPU(plainStack[fRange])
        for f in range(FRAMES_PER_CHUNK):
            plainQueue.put([plainChunkTensor, f, z, chunkf+f])
        plainQueue.join()
        plainStack[fRange] = h.toCPU(plainChunkTensor)
        
    name = "plain{}".format(z)
    dd.io.save(H5_FILE_NAME, {"stacks":{name:plainStack}}, compression='blosc')
    
h.closeLifReader()

# calc standard deviation and save invalid Frame indeces
frameCount = nf - len(h.invalidFrames)
stdDeviation = h.toCPU(th.sqrt((h.sumSqTensor - h.sumTensor**2/frameCount)/(frameCount-1)))
imsave(STD_DEV_FILE_NAME, stdDeviation)
np.save(INVALID_FRAMES_FILE_NAME, sorted(h.invalidFrames))

#display_html("<script>Jupyter.notebook.kernel.restart()</script>",raw=True)  # kill kernel

