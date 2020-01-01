import numpy as np
import config as cg
import matplotlib.pyplot as plt
try:
    import pyqtgraph as pg
    from pyqtgraph.Qt import QtGui
    lib_pyqgrapth=True
except ImportError:
    lib_pyqgrapth=False
class graph(object):
    """docstring for plot."""
    def __init__(self,nruns=1,file=cg.temp_rixs_full_file,dict_total={}):
        super(graph, self).__init__()
        self.nruns=nruns
        self.dict_total=dict_total
        self.dict_x,self.dict_y={},{}
        self.color_list=['r','b','g']
        try:
            for irun in range(self.nruns):
                self.dict_x[irun],self.dict_y[irun]=\
                        np.load(file+'_run_'+str(irun+1)+cg.extension_final)
        except:
            pass
    def simple_qt(self,scale=1):
        pg.mkQApp()
        pg.setConfigOptions(antialias=True)
        self.win=pg.GraphicsWindow()
        self.app = QtGui.QApplication([])
        self.win.resize(800,400)
        self.p=self.win.addPlot()
        self.color_list=['r','b','g']
        for irun in range(self.nruns):
            if scale==0:  scale=self.dict_y[irun]/max(self.dict_y[irun])
            self.p.plot(self.dict_x[irun],self.dict_y[irun]*scale,\
                pen=pg.mkPen(self.color_list[irun],width=2))

        self.p.setLabel('left', "RIXS Intensity", units='arb. units')
        self.p.setLabel('bottom', "Energy Loss", units='eV')
        self.p.setXRange(0.,0.35)
        self.win.show()
        self.app.exec_()
    def simple_matplot(self,scale=1):
        self.fig = plt.figure()
        self.p = self.fig.add_subplot(1, 1, 1)
        print('nruns:',self.nruns)
        for irun in range(self.nruns):
            if scale==0:  scale=self.dict_y[irun]/max(self.dict_y[irun])
            self.p.plot(self.dict_x[irun],self.dict_y[irun]*scale,\
                    linewidth=2,color=self.color_list[irun])
        self.p.set_xlabel("$\mathrm{Energy\ Loss, \ eV}$",fontsize=15)
        self.p.set_ylabel("$\mathrm{RIXS\ Intensity, \ arb.\ units}$",fontsize=15)
        # self.p.set_xlim([0.,0.36])
        plt.show()
    def simple(self,scale=1):
        if lib_pyqgrapth:
            self.simple_qt(scale)
        else:
            self.simple_matplot(scale)
    def plot_2do_qt(self,scale=1):
        pg.mkQApp()
        pg.setConfigOption('background', 'w')
        pg.setConfigOption('foreground', 'k')
        self.win = pg.GraphicsWindow()
        self.win.resize(600,400)
        self.app = QtGui.QApplication([])
        self.p = self.win.addPlot()
        self.p.addLegend(offset=(300,10))

        scale_fig=100
        self.x_1,self.y_1=np.load('temp_2_run_1.npy')
        self.x_2,self.y_2=np.load('temp_2_run_2.npy')
        self.x_3,self.y_3=np.load('temp_2_run_3.npy')
        self.y_1=self.y_1/scale_fig
        self.y_2=self.y_2/scale_fig
        self.y_3=self.y_3/scale_fig
        self.max_=max(self.y_1+self.y_2)

        curve2=self.p.plot(x=self.x_3,y=self.y_3,pen=pg.mkPen((0,154,0),width=4),name='two coupled modes')
        curve1=self.p.plot(x=self.x_1,y=self.y_1+self.y_2,pen=pg.mkPen(0.4,width=3),name='sum 1 and 2')
        mode1=self.p.plot(self.x_1,self.y_1,\
                            pen=pg.mkPen('r',width=2)\
                            ,name='single mode 1',linewidth=100)
        mode2=self.p.plot(self.x_2,self.y_2,\
                            pen=pg.mkPen('b',width=2)\
                            ,name='single mode 2',linewidth=100)


        fill=pg.FillBetweenItem(curve1=curve1, curve2=curve2, brush=(0.6))#(0,120,0,220))
        self.p.addItem(fill)
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
        self.p.setXRange(-0.025,0.360)
        self.p.setYRange(-0.1,7.5)
        self.win.show()
        self.app.exec_()
    def plot_2do_matplot(self,scale=1):
        self.fig = plt.figure()
        self.p = self.fig.add_subplot(1, 1, 1)
        scale_fig=100
        self.x_1,self.y_1=np.load('temp_2_run_1.npy')
        self.x_2,self.y_2=np.load('temp_2_run_2.npy')
        self.x_3,self.y_3=np.load('temp_2_run_3.npy')
        self.y_1=self.y_1/scale_fig
        self.y_2=self.y_2/scale_fig
        self.y_3=self.y_3/scale_fig
        self.max_=max(self.y_1+self.y_2)

        self.p.plot(self.x_3,self.y_3,color='g',linewidth=2,label='two coupled modes')
        self.p.plot(self.x_1,self.y_1+self.y_2,color='grey',linewidth=2,label='sum 1 and 2')

        self.p.plot(self.x_1,self.y_1,\
                            color='r',linewidth=2\
                            ,label='single mode 1')
        self.p.plot(self.x_2,self.y_2,\
                            color='b',linewidth=2\
                            ,label='single mode 2')
        plt.legend()
        self.p.set_xlabel("$\mathrm{Energy\ Loss, \ eV}$",fontsize=15)
        self.p.set_ylabel("$\mathrm{RIXS\ Intensity, \ arb.\ units}$",fontsize=15)
        self.p.set_xlim([-0.025,0.360])
        self.p.set_ylim([-0.1,7.5])
        plt.show()
    def plot_2do(self):
        if lib_pyqgrapth:
            self.plot_2do_qt()
        else:
            self.plot_2do_matplot()
    def plot_rixsq_matplot(self,scale=1):
        self.fig = plt.figure(figsize=(9,4))
        self.plot_oq = self.fig.add_subplot(1, 3, 1)
        q,omegaq=np.loadtxt('phonon_energy_vs_q')
        self.plot_oq.plot(q,omegaq,color='b',linewidth=2)
        self.plot_oq.set_ylabel("$\mathrm{Energy,\ eV}$",fontsize=15)
        self.plot_oq.set_xlabel("$q\ \pi/a$",fontsize=15)
        self.plot_oq.axvline(x=self.dict_total['input']['qx'],color='grey',linewidth=2)

        self.plot_gq = self.fig.add_subplot(1, 3, 2)
        q,omegaq=np.loadtxt('eph_coupling_vs_q')
        self.plot_gq.plot(q,omegaq,color='b',linewidth=2)
        self.plot_gq.set_ylabel("$\mathrm{Coupling,\ eV}$",fontsize=15)
        self.plot_gq.set_xlabel("$q\ \pi/a$",fontsize=15)
        self.plot_gq.axvline(x=self.dict_total['input']['qx'],color='grey',linewidth=2)

        self.plot_rixs = self.fig.add_subplot(1, 3, 3)

        for irun in range(self.nruns):
            if scale==0:  scale=self.dict_y[irun]/max(self.dict_y[irun])
            self.plot_rixs.plot(self.dict_x[irun],self.dict_y[irun]*scale,\
                    linewidth=2,color=self.color_list[self.nruns-1],label='$q=$'\
                            +str(self.dict_total['input']['qx']))

        self.plot_rixs.set_xlabel("$\mathrm{Energy\ Loss, \ eV}$",fontsize=15)
        self.plot_rixs.set_ylabel("$\mathrm{RIXS\ Intensity, \ arb.\ units}$",fontsize=15)
        # self.plot.setXRange(-0.025,0.360)
        self.plot_oq.set_xlim([-1,1])
        self.plot_gq.set_xlim([-1,1])
        self.plot_rixs.set_yticks([])
        self.plot_rixs.legend()
        plt.tight_layout()
        plt.show()
    def plot_rixsq(self,scale=1):
        if lib_pyqgrapth:
            self.plot_rixsq_qt(scale=scale)
        else:
            self.plot_rixsq_matplot(scale=scale)
    def plot_rixsq_qt(self,scale=1):
        pg.mkQApp()
        # pg.setConfigOption('background', 'w')
        # pg.setConfigOption('foreground', 'k')
        self.win = pg.GraphicsWindow()
        self.win.resize(900,400)
        self.app = QtGui.QApplication([])
        self.plot_oq = self.win.addPlot()
        self.plot_oq.addLegend(offset=(300,10))
        self.plot_omega_q(plot=self.plot_oq)
        self.plot_oq.addLine(x=self.dict_total['input']['qx'])
        self.plot_gq = self.win.addPlot()
        self.plot_gq.addLegend(offset=(300,10))
        self.plot_g_q(plot=self.plot_gq)
        self.plot_gq.addLine(x=self.dict_total['input']['qx'])
        self.plot = self.win.addPlot()
        self.plot.addLegend(offset=(300,10))
        self.simple_add_plot(plot=self.plot,scale=scale)
        self.plot.setXRange(-0.025,0.360)
        self.plot_oq.setXRange(-1,1)
        self.plot_gq.setXRange(-1,1)
        self.win.show()
        self.app.exec_()
    def simple_add_plot(self,plot=1,scale=1):
        for irun in range(self.nruns):
            if scale==0:  scale=self.dict_y[irun]/max(self.dict_y[irun])
            plot.plot(self.dict_x[irun],self.dict_y[irun]*scale,\
                pen=pg.mkPen(self.color_list[int(self.nruns)-1],width=2),linewidth=100)

        plot.setLabel('left', "RIXS Intensity", units='arb. units')
        plot.setLabel('bottom', "Energy Loss", units='eV')
        plot.setXRange(0.,0.35)
    def plot_omega_q(self,plot=[],scale=1):
        q,omegaq=np.loadtxt('phonon_energy_vs_q')
        plot.plot(q,omegaq,\
                            pen=pg.mkPen('b',width=2))
        plot.setLabel('left', "Energy", units='eV')
        plot.setLabel('bottom', "q", units='pi/a')
    def plot_g_q(self,plot=[],scale=1):
        q,omegaq=np.loadtxt('eph_coupling_vs_q')

        plot.plot(q,omegaq,\
                            pen=pg.mkPen('b',width=2))
        plot.setLabel('left', "coupling", units='eV')
        plot.setLabel('bottom', "q", units='pi/a')
