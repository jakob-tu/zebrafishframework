from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph as pg
import numpy as np
import os
import sys

from zebrafishframework import io

imageData = np.zeros((1024, 1024))

if __name__ == '__main__':

    fn = sys.argv[1]
    if not os.path.exists(fn):
        print('File does not exist.')
        sys.exit(1)

    print('Loading ' + fn + '...')
#    w = io.read_h5py(fn)
    frame = io.load('/Users/koesterlab/immanuel_code/data/frame.nrrd')[0]
    frame = frame.swapaxes(1, 2)

    t = 10
#    frame = w[t]
#    frame = np.zeros((21, 1024, 1024))

    app = QtGui.QApplication([])

    win = QtGui.QMainWindow()
    win.resize(1024, 1024)
    imv = pg.ImageView()
    win.setCentralWidget(imv)
    win.show()
    win.setWindowTitle('H5 Viewer')

    imv.setImage(frame, xvals=np.arange(0, frame.shape[0]))

    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()
