import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui
import numpy as np
import config as cg

class graph(object):
    """docstring for plot."""
    def __init__(self,plot=[],nruns=1,file=cg.temp_rixs_full_file):
        super(graph, self).__init__()
        self.nruns=str(nruns)
        try:
            self.x,self.y=np.load(file+'_run_'+self.nruns+cg.extension_final)
            self.max_=max(self.y)
            self.y_norm=self.y/max(self.y)
        except:
            pass
        self.p=plot
        pg.setConfigOptions(antialias=True)
    def simple(self,scale=1):
        self.color_list=['r','b','g']
        self.p.plot(self.x,self.y*scale,\
                            pen=pg.mkPen(self.color_list[int(self.nruns)-1],width=2),linewidth=100)
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
    def fill_between_2d(self,win,p,scale=1):
        scale_fig=100
        self.x_1,self.y_1=np.load('temp_2_run_1.npy')
        self.x_2,self.y_2=np.load('temp_2_run_2.npy')
        self.x_3,self.y_3=np.load('temp_2_run_3.npy')
        self.y_1=self.y_1/scale_fig
        self.y_2=self.y_2/scale_fig
        self.y_3=self.y_3/scale_fig
        self.max_=max(self.y_1+self.y_2)

        # curve1b=p.plot(x=self.x_2,y=self.y_2) *self.max_/max(self.y_3)
        curve2=p.plot(x=self.x_3,y=self.y_3,pen=pg.mkPen((0,154,0),width=4),name='two coupled modes')
        curve1=p.plot(x=self.x_1,y=self.y_1+self.y_2,pen=pg.mkPen(0.4,width=3),name='sum 1 and 2')
        mode1=p.plot(self.x_1,self.y_1,\
                            pen=pg.mkPen('r',width=2)\
                            ,name='single mode 1',linewidth=100)
        mode2=p.plot(self.x_2,self.y_2,\
                            pen=pg.mkPen('b',width=2)\
                            ,name='single mode 2',linewidth=100)


        fill=pg.FillBetweenItem(curve1=curve1, curve2=curve2, brush=(0.6))#(0,120,0,220))
        p.addItem(fill)
        font=QtGui.QFont()
        font.setPixelSize(15)
        label_style = {'font-size': '20pt'}
        self.p.setLabel('left', "RIXS Intensity", units='arb. units',**label_style)
        self.p.setLabel('bottom', "Energy Loss", units='eV',**label_style)
        self.p.showAxis("right")
        self.p.showAxis("top")
        self.p.getAxis("top").setTicks([])
        self.p.getAxis("right").setTicks([])
        self.p.getAxis("bottom").tickFont = font
        self.p.getAxis("bottom").setStyle(tickTextOffset = 10)
        self.p.getAxis("left").tickFont = font
        self.p.getAxis("left").setStyle(tickTextOffset = 10)
    def plot_omega_q(self,scale=1):
        q,omegaq=np.loadtxt('phonon_energy_vs_q')
        self.color_list=['r','b','g']
        self.p.plot(q,omegaq,\
                            pen=pg.mkPen('b',width=2))
        self.p.setLabel('left', "Energy", units='eV')
        self.p.setLabel('bottom', "q", units='pi/a')
    def plot_g_q(self,scale=1):
        q,omegaq=np.loadtxt('eph_coupling_vs_q')
        self.color_list=['r','b','g']
        self.p.plot(q,omegaq,\
                            pen=pg.mkPen('b',width=2))
        self.p.setLabel('left', "coupling", units='eV')
        self.p.setLabel('bottom', "q", units='pi/a')
