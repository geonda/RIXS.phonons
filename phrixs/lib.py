
from initp import *
from inputp import *
from kernelsp import *
from graphicsp import *
from specp import *
from qtp import *
import os
import json
from tqdm import tqdm
from timeit import default_timer as time
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui


class workspace(object):
    """docstring for workspace."""
    def __init__(self):
        super(workspace, self).__init__()
        self.ninputs=0
        self.nproblems=0
        self.nruns=0
        self.en,self.st=0.,0.
        self.max=[]
    def initp(self,type_problem='rixs',method='fc',type_calc='model',\
                                el_space=1, vib_space=1):
        self.nproblems+=1
        ip=init_problem(type_problem,method,type_calc,\
                                el_space, vib_space)
        self.dict_problem=ip.dict_current
    def inputp(self, type='ask'):
        self.ninputs+=1
        if type=='ask' :
            self.dict_input=inputp(self.ninputs,self.nproblems).create_input()
            # self.win_input,self._app_input=app(dict)
        elif type=='skip':
            with open(cg.dict_input_file+'_'+str(self.nproblems)+'_'+str(self.ninputs)+'.json') as f:
                self.dict_input=json.load(f)
                # ws2=app(dict)
            # self.win_input,self.app_input=ws2.win,ws2.app
        else : print('type input error')
        self.dict_total={'problem':self.dict_problem,'input':self.dict_input}
    def scan(self):
        self.dict_scan=inputp().scan_input()
        self.dict_total['scan']=self.dict_scan
    def run_scan(self):
        [self.runp() for _ in tqdm(range(self.dict_scan['nruns']))]
    def runp(self):
        self.dict_total=inputp(self.ninputs,self.nproblems).update_total(self.dict_total)
        self.nruns+=1
        rixs_model(self.dict_total,nruns=self.nruns).cross_section()
        spec(self.dict_total,nruns=self.nruns).run_broad()
    def figure_2d(self):
        pg.mkQApp()

        pg.setConfigOption('background', 'w')
        pg.setConfigOption('foreground', 'k')

        self.win = pg.GraphicsWindow()
        self.win.resize(600,400)
        self.app = QtGui.QApplication([])

        self.plot = self.win.addPlot()
        self.plot.addLegend(offset=(300,50))

        gs=graph(self.plot,nruns=self.nruns,file=cg.temp_rixs_noel_file)
        gs.fill_between(self.win,self.plot)
        self.max.append(gs.max_)
        gs.figure_2d(scale=float(gs.max_))

        self.plot.setXRange(0.,0.25)
        self.win.show()
        self.app.exec_()

    def figure_dd(self):
        pg.mkQApp()
        pg.setConfigOption('background', 'w')
        pg.setConfigOption('foreground', 'k')
        self.win=pg.GraphicsWindow()
        self.app = QtGui.QApplication([])
        self.win.resize(600,400)
        self.plot=self.win.addPlot()
        self.plot.addLegend(offset=(300,50))
        [graph(self.plot,nruns=runs+1,file=cg.temp_rixs_noel_file).figure_dd()
                                                for runs in range(self.nruns)]
        self.plot.setXRange(0.,5)
        self.win.show()
        self.app.exec_()

    def plotp(self):
        pg.mkQApp()
        self.win=pg.GraphicsWindow()
        self.app = QtGui.QApplication([])
        self.win.resize(800,400)
        self.plot=self.win.addPlot()
        [graph(self.plot,nruns=runs+1,file=cg.temp_rixs_noel_file).simple() \
                                                for runs in range(self.nruns)]
        self.plot.setXRange(0.,0.3)
        self.win.show()
        self.app.exec_()

    def plotp_app(self,plot):
        [graph(plot,nruns=runs+1,file=cg.temp_rixs_noel_file).simple()\
                                                for runs in range(self.nruns)]
        plot.setXRange(0.,0.3)
    def clear(self):
        os.system('rm ./temp*')
        os.system('rm ./scan.json')
    def timer_start(self):
        self.st=time()
    def timer_round(self,label):
        if self.en==0: start_time=self.st
        else: start_time=self.en
        current_time=time()
        print(label,np.round(current_time-start_time, 4))
        self.en=time()
    def timer_total(self,label):
        print(label,np.round(time()-self.st,4))
