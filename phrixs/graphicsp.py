import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui
import numpy as np
import config as cg
class graph(object):
    """docstring for plot."""
    def __init__(self,plot=[],nruns=1,file=cg.temp_rixs_full_file):
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
        scale_fig=100
        self.x_1,self.y_1=np.load('temp_2_run_1.npy')
        self.x_2,self.y_2=np.load('temp_2_run_2.npy')
        self.x_3,self.y_3=np.load('temp_2_run_3.npy')
        self.y_1=self.y_1/scale_fig
        self.y_2=self.y_2/scale_fig
        self.y_3=self.y_3/scale_fig
        self.max_=max(self.y_1+self.y_2)

        # curve1b=p.plot(x=self.x_2,y=self.y_2)
        curve2=p.plot(x=self.x_3,y=self.y_3*self.max_/max(self.y_3),pen=pg.mkPen((0,154,0),width=2),name='two coupled modes')
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
    def figure_dd(self,win,scale=1):
        #self.p.addLegend(offset=(300,50))
        self.x_1,self.y_1=np.load('temp_2_run_2.npy')
        self.x_2,self.y_2=np.load('temp_2_run_1.npy')
        self.x_3,self.y_3=np.load('temp_2_run_3.npy')
        self.color_list=['r',(0,154,0),'b']
        # self.color_list=[0.5,0.5,0.5]
        self.name_list=['B = 1','B =0.7','B=1.2']
        self.win=win
        font=QtGui.QFont()
        font.setPixelSize(15)
        label_style = {'font-size': '10pt'}
        l2 = self.win.addLayout(colspan=3)
        l2.setContentsMargins(0, 0, 0, 0)
        norm=0.05
        l2.addLabel("RIXS Intensity ( arb. units )", angle=-90, rowspan=3)
        p1=l2.addPlot()
        p1.plot(self.x_1/norm,self.y_1/max(self.y_1),\
                            pen=pg.mkPen(self.color_list[0],width=2))
        l2.nextRow()
        p2=l2.addPlot()
        p2.plot(self.x_2/norm,self.y_2/max(self.y_2),\
                            pen=pg.mkPen(self.color_list[1],width=2))
        # p2.setLabel('left', "RIXS Intensity", units='arb. units',**label_style)
        # p2.getAxis("left").setStyle(tickTextOffset = 20)
        l2.nextRow()
        p3=l2.addPlot()
        p3.plot(self.x_3/norm,self.y_3/max(self.y_3),\
                            pen=pg.mkPen(self.color_list[2],width=2))
        for p in [p1,p2,p3]:
            p.getAxis("bottom").tickFont = font
            p.getAxis("bottom").setStyle(tickTextOffset = 10)
            p.setXRange(-0.5,8.5)

            p.showAxis("right")
            p.getAxis("right").setTicks([])
        # p1.showAxis("top")
        # p1.getAxis("top").setTicks([])
        # p3.setLabel('left', "  ", units=' ',**label_style)
        l2.nextRow()
        l2.addLabel( "Energy Loss ( eV )", col=1, colspan=1,**label_style)

    def plot_pes(self,win,beta):
        win.resize(500,500)
        font=QtGui.QFont()
        font.setPixelSize(15)
        p=win.addPlot()
        x0=np.linspace(-1,1,100)
        x1=np.linspace(-1,2.5,100)
        omega=2.
        offset=2.2
        y0=omega*(x0)**2
        y1=omega*(x1-1)**2+offset
        y2=(omega*beta**2)*(x1-1)**2+offset
        curve0=p.plot(x=x0,y=y0,pen=pg.mkPen((0,154,0),width=2))
        curve2=p.plot(x=x1,y=y1,pen=pg.mkPen((0,154,0),width=2))
        curve1=p.plot(x=x1,y=y2,pen=pg.mkPen((255,184,3),width=3))
        fill=pg.FillBetweenItem(curve1=curve1, curve2=curve2, brush=(0.7))#(0,120,0,220))
        p.addItem(fill)
        p.addLine(x=0)
        p.addLine(x=1)
        p.setYRange(0,4)
        p.getAxis("bottom").setTicks([])
        p.getAxis("left").setTicks([])
        label_style = {'font-size': '20pt'}
        p.setLabel('left', "PES", units='arb. units',**label_style)
        p.setLabel('bottom', "R", units='arb. units',**label_style)

    def figure_2d(self,scale=1):

        self.color_list=['r','b','g']
        self.name_list=['single mode 1','single mode 2',' two coupled modes']
        self.p.plot(self.x,self.y*scale/max(self.y),\
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
