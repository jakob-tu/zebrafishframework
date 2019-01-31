#!/usr/bin/env python3

# https://stackoverflow.com/questions/40845304/runtimewarning-numpy-dtype-size-changed-may-indicate-binary-incompatibility
import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")
import argparse
import numpy as np
import javabridge as jv
import bioformats as bf
from pyprind import prog_percent
from xml import etree as et
from queue import Queue
from threading import Thread, Lock
from enum import Enum
import torch as th
from torch.autograd import Variable
from torch.nn.functional import grid_sample
from tqdm import tqdm_notebook
import h5py
import os
from skimage.feature import blob_log
from skimage.draw import circle
from skimage.transform import AffineTransform, warp
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
import matplotlib.cm as mplColorMap
import ipywidgets as widgets
import deepdish as dd
#import ipyvolume as ipv
from scipy.interpolate import RegularGridInterpolator

class Pyfish:
        
    def __init__(self, lif_file_path, save_path="", align_to_frame=0, use_gpu=True, max_displacement=300, thread_count=4):
        self.lif_file_path = lif_file_path
        self.align_to_frame = align_to_frame #un frami ke nesbat be un baghie axaro align mikonim
        self.use_gpu = use_gpu
        self.max_displacement = max_displacement
        self.thread_count = thread_count
        
        self._start_lif_reader()
        self._set_shapes()
        self._prepare_alignment_frame()
        
    
    @staticmethod
    def _to_gpu(numpy_array):
        return Variable(th.from_numpy(numpy_array.astype(np.float32)).cuda(), requires_grad=False)

    @staticmethod
    def _to_cpu(pytorch_tensor):
        return pytorch_tensor.cpu().numpy()
    
    
    # start lif reader 
    # set image stack shape
    # lif files can contain multiple stacks so we pick the index of one with more than one frames in lif_stack_idx
    def _start_lif_reader(self):
        jv.start_vm(class_path=bf.JARS)

        log_level = 'ERROR'
	# reduce log level
        rootLoggerName = jv.get_static_field("org/slf4j/Logger", "ROOT_LOGGER_NAME", "Ljava/lang/String;")
        rootLogger = jv.static_call("org/slf4j/LoggerFactory", "getLogger", "(Ljava/lang/String;)Lorg/slf4j/Logger;", rootLoggerName)
        logLevel = jv.get_static_field("ch/qos/logback/classic/Level", log_level, "Lch/qos/logback/classic/Level;")
        jv.call(rootLogger, "setLevel", "(Lch/qos/logback/classic/Level;)V", logLevel)

        self.ir = bf.ImageReader(self.lif_file_path, perform_init=True)
        mdroot = et.ElementTree.fromstring(bf.get_omexml_metadata(self.lif_file_path))
        mds = list(map(lambda e: e.attrib, mdroot.iter('{http://www.openmicroscopy.org/Schemas/OME/2016-06}Pixels')))

        # lif can contain multiple images, select one that is likely to be the timeseries
        self.metadata = None
        self.lif_stack_idx = 0
        for idx, md in enumerate(mds): 
            if int(md['SizeT']) > 1: 
                self.lif_stack_idx = idx
                self.metadata      = md
        if not self.metadata: raise ValueError('lif does not contain an image with sizeT > 1')
    
    def read_frame(self, t):
        frame = np.empty(self.frame_shape, dtype=np.uint16)
        for z in range(self.nz):
            frame[z] = self._read_plane(t, z)
        return frame

    def _read_plane(self, t, z):
        return self.ir.read(t=t, z=z, c=0, series=self.lif_stack_idx, rescale=False)
    
    def _set_shapes(self):
        self.nt = int(self.metadata['SizeT'])
        self.nz = int(self.metadata['SizeZ'])
        self.nx = int(self.metadata['SizeX'])
        self.ny = int(self.metadata['SizeY'])
        
        self.stack_shape = (self.nt, self.nz, self.nx, self.ny)
        self.frame_shape = (self.nz, self.nx, self.ny)
        self.plane_shape = (self.nx, self.ny)
        self.frame_dims = 3
        self.plane_dims = 2
    
    def _prepare_std_deviation_and_invalid_frames_and_result(self):
        if self.use_gpu:
            self.std_deviation_sum    = self._to_gpu(np.empty(self.frame_shape))
            self.std_deviation_sum_sq = self._to_gpu(np.empty(self.frame_shape))
        else:
            self.std_deviation_sum    = np.empty(self.frame_shape)
            self.std_deviation_sum_sq = np.empty(self.frame_shape)
        self.std_deviation_sum_mutex = Lock()
      
    def _prepare_invalid_frames(self):
        self.invalid_frames = np.empty(0)
        #print ('atfirst',self.invalid_frames)
        
    def _get_hdf5_file(self, save_path):
        save_path = save_path if save_path != "" else self.lif_file_path.replace(".lif", ".hdf5")
        try:
            os.remove(save_path)
        except OSError:
            pass
        h5_file = h5py.File(save_path, "w", libver='latest')
        h5_file.create_dataset("stack", (self.nz,self.nt,self.nx,self.ny), chunks=(1,1,self.nx,self.ny), dtype=np.uint16)
        #h5_file.create_dataset("stack", (self.nz,self.nt,self.nx,self.ny), dtype=np.uint16)
        return h5_file
    
    def _prepare_alignment_frame(self):
        
        # read alignment frame
        self.alignment_frame = self.read_frame(self.align_to_frame)
        
        # upload to GPU and prepare for cross correlation
        if self.use_gpu:
            self.frame_midpoints = np.array([np.fix(axis_size / 2) for axis_size in self.frame_shape])
            self.affine_grid_size = th.Size([1, 1, self.frame_shape[0], self.frame_shape[1], self.frame_shape[2]])
            #print (self.alignment_frame)
            frame_tensor = self._to_gpu(self.alignment_frame) #frame_tensor majmue plane ha dar z haye mokhtalef baraye t=0 (alignment frame) hast.
            self.alignment_frame_fourier = th.rfft(frame_tensor, self.frame_dims, onesided=False) 
                                                     #for plane in frame_tensor])
            self.alignment_frame_fourier[:,:,:,1] *= -1        # complex conjugate for crosscorrelation
    
    
    ### image registration functions
        
    def _create_queue(self, function, maxsize=0):
        q = Queue(maxsize=maxsize)
        for i in range(self.thread_count):
            t = Thread(target=function, args=(q,))
            t.setDaemon(True)
            t.start()
        return q
    
    def _align_frame_worker(self, queue):
        while(True):
            #frame, t, dst, dst_idx = queue.get() 
            frame, t, dst, dst_idx = queue.get() 
            #print ('dst.idx',dst_idx)
            #dst[dst_idx] = self._align_frame(frame, t)
            dst = self._align_frame(frame, t)
            #self.registered_stack[t] = dst
            queue.task_done()
        return
    
    def _align_plane_worker(self, queue):
        while(True):
            plane, t, z, dst = queue.get()
            dst[t] = self._align_plane(plane, t, z)
            queue.task_done()
        return
    

    def _register_translation_gpu(self,frame,upsample_factor=20):
        # Whole-pixel shift - Compute cross-correlation by an IFFT
        img_fourier = th.rfft(frame, self.frame_dims, onesided=False)
        image_product = th.zeros(img_fourier.size())
        image_product[:,:,:,0] = img_fourier[:,:,:,0]* self.alignment_frame_fourier[:,:,:,0]- \
        img_fourier[:,:,:,1]* self.alignment_frame_fourier[:,:,:,1]
        image_product[:,:,:,1] = img_fourier[:,:,:,0]* self.alignment_frame_fourier[:,:,:,1]+ \
        img_fourier[:,:,:,1]* self.alignment_frame_fourier[:,:,:,0]                                        
        cross_correlation = th.irfft(image_product, self.frame_dims, onesided=False, signal_sizes = frame.shape)
        # Locate maximum
        maxima = self._to_cpu(th.argmax(cross_correlation))        
        maxima = np.unravel_index(maxima, cross_correlation.size(), order='C')
        maxima = np.asarray(maxima)
        shifts = np.array(maxima, dtype=np.float64)
        shifts[shifts > self.frame_midpoints] -= np.array(cross_correlation.shape)[shifts > self.frame_midpoints] # in bara chie??
        shifts = np.round(shifts * upsample_factor) / upsample_factor #aya round numpy ba torch yejur amal mikone?bale
        upsampled_region_size = np.ceil(upsample_factor * 1.5)        
        dftshift = np.fix(upsampled_region_size / 2.0)
        upsample_factor = np.array(upsample_factor, dtype=np.float64)
        normalization = (self._to_cpu(img_fourier).size * upsample_factor ** 2)
        sample_region_offset = dftshift - shifts*upsample_factor
        image_product = self._to_cpu(image_product)
        imag_part = 1j*image_product[:,:,:,1]
        img_product_cpu = image_product[:,:,:,1]+imag_part         
        cross_correlation = self._upsampled_dft_cpu(img_product_cpu.conj(), upsampled_region_size, upsample_factor, sample_region_offset).conj()
        
        cross_correlation /= normalization
        # Locate maximum and map back to original pixel grid
        maxima = np.array(np.unravel_index(np.argmax(np.abs(cross_correlation)),cross_correlation.shape,order='C'), dtype=np.float64)
        maxima -= dftshift        
        shifts = shifts + maxima / upsample_factor
        return shifts


    def _upsampled_dft_cpu(self,data, upsampled_region_size, upsample_factor=20, axis_offsets=None):
        #print ('data', data.shape)
        if not hasattr(upsampled_region_size, "__iter__"):
            upsampled_region_size = [upsampled_region_size, ] * data.ndim
        else:
            if len(upsampled_region_size) != data.ndim:
                raise ValueError("shape of upsampled region sizes must be equal to input data's number of dimensions.")
        if axis_offsets is None:
            axis_offsets = [0, ] * data.ndim
        else:
            if len(axis_offsets) != data.ndim:
                raise ValueError("number of axis offsets must be equal to input data's number of dimensions.")
        if data.ndim == 3:
            im2pi = 1j * 2 * np.pi
            dim_kernels = []
            #print ('upsampled_region_size',upsampled_region_size)
            #print('axis_offsets',axis_offsets)
            for (n_items, ups_size, ax_offset) in zip(data.shape,upsampled_region_size, axis_offsets):
                dim_kernels.append(
                     np.exp(np.dot(
                    (-im2pi / (n_items * upsample_factor)) *
                    (np.arange(upsampled_region_size[0])[:, None] - ax_offset),
                    (np.fft.ifftshift(np.arange(n_items))[None, :]
                     - n_items // 2))))
            #print('dim_kernels',len(dim_kernels))
            # To compute the upsampled DFT across all spatial dimensions, a tensor product is computed with einsum
            try:
                return np.einsum('ijk, li, mj, nk -> lmn', data, *dim_kernels, optimize=True)
            except TypeError:
           # warnings.warn("Subpixel registration of 3D images will be very slow if your numpy version is earlier than 1.12")
                return np.einsum('ijk, li, mj, nk -> lmn', data, *dim_kernels)

       
    def _align_frame(self, frame, t):

        shifts = np.zeros([1, 3])
        if self.use_gpu:
            frame_tensor = self._to_gpu(frame)
            
            ## 3D translation and warping here
            
            shift = self._register_translation_gpu(frame_tensor)
            if np.sqrt(np.sum(shift**2)) > self.max_displacement:
                 self.invalid_frames = np.append(self.invalid_frames, t)
                 frame_tensor[:,:,:] = 0
                 #self.registered_stack[t] = frame_tensor
                 self.displacement[t] = shifts
                 return frame_tensor

            shifts = shift
            self.displacement[t] = shifts
            #if np.absolute (shifts[0]) > 1:
             #  shifts[0] = np.where(shifts[0] > 0, [0.99],[-.99])
            aligned_frame_tensor = self._warp_gpu(frame_tensor, shifts)
            with self.std_deviation_sum_mutex:
                self.std_deviation_sum    += aligned_frame_tensor
                self.std_deviation_sum_sq += aligned_frame_tensor**2
            aligned_frame_tensor = self._to_cpu(aligned_frame_tensor)
            #self.registered_stack[t] = aligned_frame_tensor
            return aligned_frame_tensor
            
        else:
            #for z in range(self.nz):
                #shift, _, _ = register_translation(frame[z], alignment_frame[z])
            shift = self._register_translation_cpu(frame, self.alignment_frame, upsample_factor=20)
            if np.sqrt(np.sum(shift**2)) > self.max_displacement:
                self.invalid_frames = np.append(self.invalid_frames, t)
                frame_tensor[:,:,:] = 0
                self.registered_stack[t] = frame_tensor
                return frame_tensor
                #shifts = shift
            #self.displacement[t] = shift
            print (shift)
            shift = self._to_gpu(shift)
            #frame_tensor = self._to_gpu(frame)
            #return shift
            frame_tensor = self._to_gpu(frame)
            aligned_frame_tensor = self._warp_gpu(frame_tensor, shift)
            self.registered_stack[t] = frame_tensor
            #aligned_frame_tensor = self._warp_gpu(frame, shift)
           # for z in range(self.nz): 
              #  aligned_frame[z] = warp(aligned_frame[z], AffineTransform(translation=shifts[z]), preserve_range=True)
            with self.std_deviation_sum_mutex:
                self.std_deviation_sum    += aligned_frame_tensor
                self.std_deviation_sum_sq += aligned_frame_tensor**2
            return self._to_cpu(aligned_frame_tensor)

    
    def _align_plane(self, plane, t, z):
        if self.use_gpu:
            plane_tensor = self._to_gpu(plane)
            shift = self._register_translation_gpu(plane_tensor, z)
            if np.sqrt(np.sum(shift**2)) > self.max_displacement:
                np.append(self.invalid_frames, t)
                plane_tensor[:,:] = 0
                return plane_tensor
            plane_tensor = self._warp_gpu(plane_tensor, shift)
            with self.std_deviation_sum_mutex:
                self.std_deviation_sum[z]    += plane_tensor
                self.std_deviation_sum_sq[z] += plane_tensor**2
            return plane_tensor
        else:
            shift, _, _ = register_translation(plane, self.alignment_frame[z])
            if np.sqrt(np.sum(shift**2)) > self.max_displacement:
                np.append(self.invalid_frames, t)
                plane[:,:] = 0
                return plane
            plane = warp(plane, AffineTransform(translation=shift), preserve_range=True)
            with self.std_deviation_sum_mutex:
                self.std_deviation_sum[z]    += plane
                self.std_deviation_sum_sq[z] += plane**2
            return plane
     


    def _warp_gpu(self, src, shift):

        theta = th.Tensor([[[1, 0, 0, shift[2]*2/self.frame_shape[1]],
                            [0, 1, 0, shift[1]*2/self.frame_shape[2]],
                            [0, 0, 1, shift[0]*2/self.frame_shape[0]]]]).cuda()

        self.affine_grid_size = th.Size([1, 1, self.frame_shape[0], self.frame_shape[2], self.frame_shape[1]])
        grid = self._affine_grid(theta, self.affine_grid_size) 
        proj = grid_sample(src.unsqueeze(0).unsqueeze(0), grid)
        return proj.squeeze(0).squeeze(0)

    @staticmethod
    def _affine_grid( theta, size):
        #assert type(size) == torch.Size
        #ctx.size = size
        #ctx.is_cuda = theta.is_cuda
        if len(size) == 5:
            N, C, D, H, W = size
            base_grid = theta.new(N, D, H, W, 4)

            base_grid[:, :, :, :, 0] = (th.linspace(-1, 1, W) if W > 1 else th.Tensor([-1]))
            base_grid[:, :, :, :, 1] = (th.linspace(-1, 1, H) if H > 1 else th.Tensor([-1]))\
                .unsqueeze(-1)
            base_grid[:, :, :, :, 2] = (th.linspace(-1, 1, D) if D > 1 else th.Tensor([-1]))\
                .unsqueeze(-1).unsqueeze(-1)
            base_grid[:, :, :, :, 3] = 1

            grid = th.bmm(base_grid.view(N, D * H * W, 4), theta.transpose(1, 2))
            grid = grid.view(N, D, H, W, 3)
            #ctx.base_grid = base_grid

        return grid

    # ROI (Region Of Interest) Extrection
    
    def _get_plane_std_deviation(self, z):
        self.invalid_frames = np.unique(np.sort(self.invalid_frames))
        frame_count = self.nt - len(self.invalid_frames)
        std_dev_plane = (self.std_deviation_sum_sq[z] - self.std_deviation_sum[z]**2/frame_count)/(frame_count-1)
        return self._to_cpu(th.sqrt(std_dev_plane)) if self.use_gpu else np.sqrt(std_dev_plane)
    
    @staticmethod
    def get_cells(imgs, max_radius=5, size_iterations=5, rel_threshold=0.3, min_intensity=5, overlap=0):
        nz = stdDeviations.shape[0]
        positions = [None] * nz
        sizes     = [None] * nz
        blob_detect_info = (max_radius, size_iterations, rel_threshold, overlap, min_intensity)
        for z in range(nz): 
            positions[z], sizes[z] = _get_plane_cells(imgs[z], blob_detect_info)
        return positions, sizes
    
    @staticmethod
    def get_cell_activity(positions, sizes, frame_stack):
        nz = frame_stack.shape[1]
        activities = [None] * nz
        for z in range(nz): 
            activities[z] = _get_plane_cell_activity(z, positions[z], sizes[z], plane_stack[:,z])
        return activities
        
    
    @staticmethod
    def _get_plane_cell_activity(z, positions, sizes, plane_stack):
        nt, nx, ny = plane_stack.shape
        activity = np.empty((positions.shape[0], nt)) 
        for i, pos in enumerate(positions):
            stencil = circle(pos[0], pos[1], sizes[i], (nx, ny))
            activity[i] = plane_stack[:, stencil[0], stencil[1]].mean(1) 
        return activity


    @staticmethod
    def _get_plane_cells(img, blob_detect_info):
        max_radius, size_iterations, rel_threshold, overlap, min_intensity = blob_detect_info
        blobs = blob_log(normalize(img),min_sigma = 1, max_sigma=max_radius*0.70710678118, # sigma = radius/sqrt(2)
                         num_sigma=size_iterations, threshold=rel_threshold/np.max(img), overlap=overlap)
        blobs = np.asarray([(y, x, r) for x, y, r in blobs if (img[int(x), int(y)] > min_intensity)]) #chera y ro miare aval bad x ro?
        return blobs[:,(0,1)], blobs[:,2]
    
    # register whole stack without chunking. Requires RAM bigger than lif-file
    def register_whole_stack(self, save_path="", max_cell_radius=5, cell_size_iterations=5, cell_rel_threshold=0.4, min_cell_intensity=20, 
                            cell_overlap=0):

        #disp = []
        #self.aligned_frame = [None] * self.nt
        self.displacement = np.empty([self.nt, 3])
        self.registered_stack = np.empty((self.nt, self.nz, self.nx, self.ny),dtype=np.uint16)        
        self._prepare_std_deviation_and_invalid_frames_and_result()
        self._prepare_invalid_frames()
        
        register_queue = self._create_queue(self._align_frame_worker, self.thread_count * 2)

        for t in prog_percent(range(self.nt)):
            register_queue.put([self.read_frame(t), t, self.registered_stack[t], t])
        register_queue.join()

        return self.displacement, self.registered_stack

    
    # register whole stack without chunking. Requires RAM bigger than lif-file
    def register_and_segmentation_whole_stack(self, save_path="", max_cell_radius=5, cell_size_iterations=5, cell_rel_threshold=0.4, min_cell_intensity=20, 
                            cell_overlap=0):

        #disp = []
        #self.aligned_frame = [None] * self.nt
        self.displacement = np.empty([self.nt, 3])
        self.registered_stack = np.empty((self.nt, self.nz, self.nx, self.ny),dtype=np.uint16)        
        self._prepare_std_deviation_and_invalid_frames_and_result()
        self._prepare_invalid_frames()
        
        register_queue = self._create_queue(self._align_frame_worker, self.thread_count * 2)

        for t in tqdm_notebook(range(self.nt)):
            register_queue.put([self.read_frame(t), t, self.registered_stack[t], t])
        register_queue.join()

        std_deviation   = np.empty((self.nz, self.nx, self.ny))
        cell_positions  = [None] * self.nz
        cell_sizes      = [None] * self.nz
        cell_activities = [None] * self.nz

        for z in tqdm_notebook(range(self.nz)):
                std_deviation[z] = self._get_plane_std_deviation(z)
                np.set_printoptions(suppress=True,precision=2,threshold = np.nan)
                blob_detect_info = (max_cell_radius, cell_size_iterations, cell_rel_threshold, cell_overlap, min_cell_intensity)
                cell_positions[z], cell_sizes[z] = self._get_plane_cells(std_deviation[z], blob_detect_info)
                cell_activities[z] = self._get_plane_cell_activity(z, cell_positions[z], cell_sizes[z], self.registered_stack[:,z])
        return self.displacement, std_deviation, cell_positions, cell_sizes, cell_activities
        


# Plotting

def show_inline_scatter_and_image(img, points, size=(10,10), facecolors="None", edgecolors="red", 
                                  vmin=None, vmax=None, s=None):
    plt.figure(figsize=size)
    plt.imshow(img, zorder=0, vmin=vmin, vmax=vmax)
    plt.scatter(points[:,0], points[:,1], s=s*min(size), edgecolors=edgecolors, facecolors=facecolors)
    plt.show()



# cell filtering with bokeh

y = ''  # y is filled in javascipt "selectCallback" of Bokeh
def remove_cells_by_hand(z, imgs, cell_poss, cell_sizes, cell_acts, size=(600, 600), vmax=None, remove_border=0):
    from bokeh.models import LassoSelectTool, BoxSelectTool, ResetTool, WheelZoomTool, BoxZoomTool
    from bokeh.plotting import figure, show, output_notebook
    from bokeh.models import CustomJS, ColumnDataSource, Circle, Toolbar, ToolbarPanel
    from bokeh.io import push_notebook, reset_output
    from bokeh.layouts import row
    output_notebook(hide_banner=True)

    img = imgs[z]
    if vmax != None: img = np.clip(img, 0, vmax)
    nx, ny = img.shape

    # delete border Cells
    border_idxs = np.asarray([idx for idx, pos in enumerate(cell_poss[z]) if pos[0] <= remove_border 
                                                                          or pos[0] >= nx - remove_border
                                                                          or pos[1] <= remove_border
                                                                          or pos[1] >= ny - remove_border])
    cell_poss[z] = np.delete(cell_poss[z],   border_idxs, axis=0)
    cell_sizes[z] = np.delete(cell_sizes[z], border_idxs, axis=0)
    cell_acts[z] = np.delete(cell_acts[z],   border_idxs, axis=0)
    cell_pos = cell_poss[z]
    cell_size = cell_sizes[z]
    cell_act = cell_acts[z]
    
    def save(b):
        global y
        print(y)
        if y == '': return
        rem_idxs = np.asarray([idx for idx, y in enumerate(map(float,y.split(','))) if y < 0])
        y = ''
        print(cell_poss[z].shape)
        cell_poss[z]  = np.delete(cell_poss[z],  rem_idxs, axis=0)
        cell_sizes[z] = np.delete(cell_sizes[z], rem_idxs, axis=0)
        cell_acts[z]  = np.delete(cell_acts[z],  rem_idxs, axis=0)
        print(cell_poss[z].shape)
    button = widgets.Button(description="Save", button_style='success', icon='save')
    button.on_click(save)
    display(button)

    source = ColumnDataSource(data=dict(x=[],y=[],s=[]))
    select_callback = CustomJS(args=dict(s=source), code="""
        var idxs = s.selected.indices;
        if (idxs.length == 0) return;
        var d = s.data;
        for (var i = 0; i < idxs.length; i++) d['y'][idxs[i]] = -1000;
        s.change.emit();
        cmd = "pf.y = '" + d['y'] + "'";
        IPython.notebook.kernel.execute(cmd, {}, {});
        """)
    lasso_tool = LassoSelectTool(callback=select_callback, select_every_mousemove=False)
    box_tool   = BoxSelectTool(callback=select_callback, select_every_mousemove=False)
    wheel_tool = WheelZoomTool(zoom_on_axis=False)
    tools = [ResetTool(), BoxZoomTool(), lasso_tool, box_tool, wheel_tool]

    p = figure(title="draw to remove:", output_backend="webgl", toolbar_location="above", tools=tools,
               x_range=(0, nx), y_range=(ny,0), plot_width=size[0], plot_height=size[1])
    p.xgrid.visible = p.ygrid.visible = False
    p.toolbar_location = None
    p.add_layout(ToolbarPanel(toolbar=Toolbar(tools=tools)), "right")

    lasso_overlay = p.select_one(LassoSelectTool).overlay
    lasso_overlay.fill_color = "firebrick"
    lasso_overlay.line_color = "white"

    h = show(row(p), notebook_handle=True)
    
    p.image(image=[img[::-1]], x=0, y=ny, dw=nx, dh=ny, palette="Viridis256", level="image")
    scat = p.circle('x', 'y', color='red', size='s', source=source, fill_color='red', alpha=1)
    scat.data_source.data = dict(x=cell_pos[:,0], y=cell_pos[:,1], s=cell_size*2)
    scat.nonselection_glyph = Circle(line_color='red', fill_color='red')

    push_notebook(handle=h)

    
    
def get_dff(activities, control_frame_count=100): #farz mishe dar t=100 made stimulation tazrigh shode
    t_end = min([min([t for t in range(act.shape[1]) if act[0,t] == 0]) for act in activities]) #if bekhatere ine ke tu range t unayi ke sharte act[0,t] == 0 ro nadaran, tuye min darnazar nagirateshun. 
    t_start = t_end - control_frame_count
    dff = np.asarray([np.empty(act.shape) for act in activities])
    for z in range(len(activities)):
        control_intensities = activities[z][:,t_start:t_end].mean(1)
        dff[z] = ((activities[z].T - control_intensities)/(control_intensities)).T
    return dff

def show_inline_image(img, size=(5,5), vmin=None, vmax=None):
    plt.figure(figsize=size)
    plt.imshow(img, vmin=vmin, vmax=vmax)
    plt.colorbar()
    plt.show()


# 3D Plotting with ipyvolume

def intensity_to_color_and_size(cell_pos, vals, col_min, col_mid, col_max, min_val=-1.0, mid_val=0.0, max_val=2.0, 
                                        min_size=0.1, max_size=1):
    def hex_to_rgb(hex_string):
        return np.asarray([int(hex_string.lstrip('#')[i:i+2], 16) for i in (0, 2 ,4)])

    col_min, col_mid, col_max = (hex_to_rgb(col_min)/255, hex_to_rgb(col_mid)/255, hex_to_rgb(col_max)/255)
    vals = np.concatenate(vals)
    vals = np.clip(np.where(vals < mid_val, vals/(-min_val), vals/max_val), -1, 1)
    cell_x = np.concatenate([cps[:,0].astype(np.float32) for cps in cell_pos])
    cell_y = np.concatenate([cps[:,1].astype(np.float32) for cps in cell_pos])
    cell_z = np.concatenate([np.full(cps.shape[0], z, dtype=float) for z, cps in enumerate(cell_pos)])
    cell_c = np.empty((vals.shape[0], vals.shape[1], 3))
    cell_s = np.empty(vals.shape)
    print ('vals.shape',vals.shape) 
    for i, v in enumerate(vals):
        cell_c[i] = np.where(v < mid_val, np.multiply.outer(col_min, 0-v) + np.multiply.outer(col_mid, 1+v), 
                                          np.multiply.outer(col_mid, 1-v) + np.multiply.outer(col_max, 0+v)).T
        cell_s[i] = min_size + (max_size-min_size)*np.absolute(v)
    return (cell_x, cell_y, cell_z, cell_c.swapaxes(0,1), cell_s.swapaxes(0,1))

def show_inline_3D_scatter(cell_info, volume, size=(720,720), vol_max_val=30, speed=5, t_range=None):
    x, y, z, c, s = cell_info
    if t_range:
       c, s = c[t_range], s[t_range]
    nz, nx, ny = volume.shape
    ipv.figure(width=size[0], height=size[1])
    ipv.xlim(0, nx)
    ipv.ylim(0, ny)
    ipv.zlim(0-nz//1.5, nz+nz//1.5)
    ipv.pylab.style.axes_off()
    ipv.pylab.style.set_style_dark()
    if type(volume) is np.ndarray:
        ipv.volshow(np.where(volume<vol_max_val, 0, 1), level=(1,1,1), opacity=(0.0075, 0.0075,0.0075))
    scatter_plot = ipv.scatter(x, y, z, c, marker='circle_2d', size=s)
    ipv.animation_control(scatter_plot, interval=1000/speed, sequence_length=c.shape[0])
    ipv.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('liffile')
    parser.add_argument('output_base')

    args = parser.parse_args()
    pyfish = Pyfish(args.liffile, use_gpu = True, thread_count=2)
    displacements, aligned = pyfish.register_whole_stack()

    print('Saving shifts...')
    np.save(args.output_base + '_shifts.npy', displacements)

    print('Saving aligned stack...')
    dd.io.save(args.output_base + '_aligned.h5', aligned, 'blosc')


