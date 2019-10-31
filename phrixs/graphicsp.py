import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui
import numpy as np
import config as cg
class graph(object):
    """docstring for plot."""
    def __init__(self,app,win,plot,nruns=1,file=cg.temp_rixs_full_file):
        super(graph, self).__init__()
        self.nruns=str(nruns)
        self.app=app
        self.win=win
        self.x,self.y=np.load(file+'_run_'+self.nruns+cg.extension_final)
        self.y_norm=self.y#/max(self.y)
        pg.setConfigOptions(antialias=True)
        self.p=plot
    def simple(self):
        self.p.plot(self.x,self.y_norm,\
                            pen=(int(self.nruns),4),linewidth=100)
        # lr = pg.LinearRegionItem([0.1,0.5])
        # lr.setZValue(-10)
        # self.p.addItem(lr)
        self.p.setLabel('left', "RIXS Intensity", units='arb. units')
        self.p.setLabel('bottom', "Energy Loss", units='eV')
    def nonsimple(self):
        self.p.plot(self.x,self.y_norm, pen=(0,0,255,200),linewidth=100)
        lr = pg.LinearRegionItem([0.1,0.5])
        lr.setZValue(-10)
        self.p.addItem(lr)
        self.p.setLabel('left', "RIXS Intensity", units='arb. units')
        self.p.setLabel('bottom', "Energy Loss", units='eV')
