import deepdish as dd
import numpy as np

fish = 2
base = '/Users/koesterlab/registered/control/fish%02d_6dpf_medium' % fish
aligned_fn = base + '_aligned.h5'

base2 = base.replace('registered', 'segmented')
rois_fn = base2 + '_rois.npy'
traces_fn = base2 + '_traces.npy'

rois = np.load(rois_fn)
traces = np.load(traces_fn)
#rois = np.zeros((1,))
#traces = np.zeros((1,))


def load(t, z):
    return dd.io.load(aligned_fn, sel=(slice(t, t + 1), slice(z, z + 1))).squeeze()


from bokeh.layouts import column, row, layout
from bokeh.models import ColumnDataSource, Range1d
from bokeh.models.widgets import Slider
from bokeh.plotting import curdoc, figure

view_figure = figure(plot_width=950, plot_height=950, x_range=Range1d(0, 1024, bounds='auto'),
           y_range=Range1d(1024, 0, bounds='auto'))
img_src = ColumnDataSource({})

rois_data = []
for z in range(21):
    x = [r[0] for r in rois if r[2] == z]
    y = [r[1] for r in rois if r[2] == z]
    rois_data.append((x, y))

img = view_figure.image('value', x=0, y=1024, dw=1024, dh=1024, source=img_src, palette='Greys256')

rois_src = ColumnDataSource(dict(x=[], y=[]))
rois_glyph = view_figure.circle(x='x', y='y', size=2, color='red', source=rois_src)

hist_figure = figure(plot_width=800, plot_height=250)
hist_src = ColumnDataSource(dict(x=[], top=[]))
hist = hist_figure.vbar(x='x', top='top', source=hist_src, width=1)

minv_slider = Slider(start=0, end=1000, value=0, step=1, title='Min', width=900)
maxv_slider = Slider(start=0, end=1000, value=1000, step=1, title='Max', width=900)

z = 10
t = 0

def update_rois():
    global z
    x, y = rois_data[z]
    rois_src.data = dict(x=x, y=y)

def update_img():
    global z, t
    new_data = load(t, z)
    displayed = display_image(new_data, 0, 800)
    img_src.data = {'value': [displayed]}
    hist_data = np.bincount(new_data.flatten())
    hist_src.data = dict(x=np.arange(np.alen(hist_data)), top=hist_data)

def display_image(img, min_v, max_v):
    f = 255.0/(max_v-min_v)
    return np.flip((np.array(np.maximum(np.minimum(img, max_v), min_v)-min_v)*f).astype(np.float32), axis=0)

def t_select_handler(attr, old, new):
    global t
    t = new
    update_img()

def z_select_handler(attr, old, new):
    global z
    z = new
    update_img()
    update_rois()


update_img()
update_rois()

t_select = Slider(start=0, end=1799, value=0, step=1, title='Time', width=950)
t_select.on_change('value', t_select_handler)

z_select = Slider(start=0, end=20, value=10, step=1, title='Z', width=50, height=900, orientation='vertical')
z_select.on_change('value', z_select_handler)

l = layout([
    [view_figure, z_select, [hist_figure, minv_slider, maxv_slider]],
    [t_select]
])

curdoc().add_root(l)
