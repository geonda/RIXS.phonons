import numpy as np
import matplotlib.pyplot as plt
plt.style.use('dark_background')
try:
    import pyqtgraph as pg
    from pyqtgraph.Qt import QtGui
    lib_pyqgrapth=True
except ImportError:
    pass
##### ad hoc fix to use matplotlib only
lib_pyqgrapth=False
#######################################

class plot(object):
    """docstring for plot."""
    def __init__(self,model_list=[],exp=[]):
        super(plot, self).__init__()
        self.model_list = model_list
        self.exp = exp

    def show(self,scale=1):
        if scale==0:
            for model in self.model_list:
                plt.plot(model.x,model.y_norm, color=model.color,linewidth = 2,
                    label = 'model #{nm} : {name}'.format(nm = model.nmodel,name=model.name))
            plt.scatter(self.exp.x,self.exp.y_norm, facecolor = 'None', edgecolor = 'w',\
                    label = self.exp.name)
        else:
            for model in self.model_list:
                plt.plot(model.x,model.y*scale, color=model.color,linewidth = 2,
                    label = 'model #{nm} : {name}'.format(nm = model.nmodel,name=model.name))
            plt.scatter(self.exp.x,self.exp.y, facecolor = 'None', edgecolor = 'w',\
                    label = self.exp.name)

        plt.xlabel("$\mathrm{Energy\ Loss, \ eV}$",fontsize=15)
        plt.ylabel("$\mathrm{RIXS\ Intensity, \ arb.\ units}$",fontsize=15)
        # self.p.set_xlim([0.,0.36])
        plt.legend()
        plt.show()
