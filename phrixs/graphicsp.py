import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui
import numpy as np
import config as cg
class graph(object):
    """docstring for plot."""
    def __init__(self,plot,nruns=1,file=cg.temp_rixs_full_file):
        super(graph, self).__init__()
        self.nruns=str(nruns)
        self.x,self.y=np.load(file+'_run_'+self.nruns+cg.extension_final)
        self.y_norm=self.y/max(self.y)
        pg.setConfigOptions(antialias=True)
        self.p=plot
        self.max_=max(self.y)
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
        
    def fill_between(self,win,p,scale=1):

        self.x_1,self.y_1=np.load('temp_2_run_1.npy')
        self.x_2,self.y_2=np.load('temp_2_run_2.npy')
        self.x_3,self.y_3=np.load('temp_2_run_3.npy')
        self.max_=max(self.y_1+self.y_2)
        curve1a=p.plot(x=self.x_1,y=self.y_1+self.y_2,pen=pg.mkPen(0.5,width=2),name='sum 1 and 2')
        # curve1b=p.plot(x=self.x_2,y=self.y_2)
        curve2=p.plot(x=self.x_3,y=self.y_3*self.max_/max(self.y_3))

        mode1=p.plot(self.x_1,self.y_1,\
                            pen=pg.mkPen('r',width=3)\
                            ,name='single mode 1',linewidth=100)
        mode2=p.plot(self.x_2,self.y_2,\
                            pen=pg.mkPen('b',width=3)\
                            ,name='single mode 2',linewidth=100)

        fill=pg.FillBetweenItem(curve1=curve1a, curve2=curve2, brush=(0.9))
        p.addItem(fill)
    def figure_dd(self,scale=1):

        self.color_list=[(0,154,0),'r','b']
        self.name_list=['B = 1','B =0.8','B=1.2']
        self.p.plot(self.x/0.1,self.y,\
                            pen=pg.mkPen(self.color_list[int(self.nruns)-1],width=3)\
                            ,name=self.name_list[int(self.nruns)-1],linewidth=100)
        font=QtGui.QFont()
        font.setPixelSize(15)
        label_style = {'font-size': '20pt'}
        self.p.setLabel('left', "RIXS Intensity", units='arb. units',**label_style)
        self.p.setLabel('bottom', "Energy Loss/Phonon Energy", units='',**label_style)
        self.p.showAxis("right")
        self.p.showAxis("top")
        self.p.getAxis("top").setTicks([])
        self.p.getAxis("right").setTicks([])
        self.p.getAxis("bottom").tickFont = font
        self.p.getAxis("bottom").setStyle(tickTextOffset = 10)
        self.p.getAxis("left").tickFont = font
        self.p.getAxis("left").setStyle(tickTextOffset = 10)
    def figure_2d(self,scale=1):

        self.color_list=['r','b','g']
        self.name_list=['single mode 1','single mode 2',' two coupled modes']
        self.p.plot(self.x,self.y_norm*scale/max(self.y_norm),\
                            pen=pg.mkPen((0,154,0),width=3)\
                            ,name=self.name_list[int(self.nruns)-1],linewidth=100)
        # lr = pg.LinearRegionItem([0.1,0.5])
        # lr.setZValue(-10)
        # self.p.addItem(lr)
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
