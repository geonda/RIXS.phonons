import numpy as np
import matplotlib.pyplot as plt
plt.style.use('dark_background')

#######################################

class plot(object):
    """

    Visualization.

    Args:
        model_list: list
            list of models.
        exp: object
            experiment.

    Attributes:
        if_exp: boolen
            returns True if experiment (object) is specified.

    """
    def __init__(self,model_list=[],exp=[]):
        super(plot, self).__init__()
        self.model_list = model_list
        self.scatter_ec = 'w'
        try :
            self.xexp = exp.x
            self.exp = exp
            self.if_exp = True
        except:
            print('no experiment in visual')
            self.if_exp = False
            pass

    def show_xas(self):
        plt.figure()
        for model in self.model_list:
            try:
                plt.plot(model.xas_freq - model.input['energy_ex'], model.xas_int,linewidth = 2, color=model.color,
                    label = 'model #{nm} : {name}'.format(nm = model.nmodel,name=model.name))
                plt.plot(model.xas_freq_no_q - model.input['energy_ex'], model.xas_int_no_q,linewidth = 2, color='w',
                    label = 'single q')
            except:
                pass
        plt.xlabel(r"$\mathrm{\omega-E_{ex}, \ eV}$",fontsize=15)
        plt.ylabel("$\mathrm{XAS\ Intensity, \ arb.\ units}$",fontsize=15)
        plt.axvline(x=0,linestyle='--',color='w')
        plt.yticks([])
        plt.xlim([-1,1.5])
        plt.legend()
        plt.show()

    def show(self,scale=1):
        if scale==0:
            for model in self.model_list:
                if not model.color :
                    plt.plot(model.x,model.y_norm, linewidth = 2,
                        label = 'model #{nm} : {name}'.format(nm = model.nmodel,name=model.name))
                else:
                    plt.plot(model.x,model.y_norm, linewidth = 2, color=model.color,
                        label = 'model #{nm} : {name}'.format(nm = model.nmodel,name=model.name))
            if self.if_exp :
                plt.scatter(self.exp.x,self.exp.y_norm, facecolor = 'None', edgecolor = self.scatter_ec,\
                    label = self.exp.name)
        else:
            for model in self.model_list:
                if not model.color :
                    plt.plot(model.x,model.y*scale,linewidth = 2,
                        label = 'model #{nm} : {name}'.format(nm = model.nmodel,name=model.name))
                else:
                    plt.plot(model.x,model.y*scale, color=model.color,linewidth = 2,
                        label = 'model #{nm} : {name}'.format(nm = model.nmodel,name=model.name))
            if self.if_exp :
                plt.scatter(self.exp.x,self.exp.y, facecolor = 'None', edgecolor = self.scatter_ec,\
                    label = self.exp.name)

        plt.xlabel("$\mathrm{Energy\ Loss, \ eV}$",fontsize=15)
        plt.ylabel("$\mathrm{RIXS\ Intensity, \ arb.\ units}$",fontsize=15)
        plt.legend()
        plt.show()
