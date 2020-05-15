from phlab import model
from phlab import experiment
from phlab import visual
import json
import os

class rixs(object):
    """docstring for create_workspace."""

    def __init__(self,problem_name = 'rixs',\
                    out_dir = './_output/',\
                    inp_dir = './_input/'):
        super(rixs, self).__init__()
        self.nws=1
        self.nmodel=0
        self.nexp=0
        self.inp_dir=inp_dir
        self.out_dir=out_dir
        try:
            os.mkdir(self.inp_dir)
        except:
            pass

    def model_single_osc(self, name = ''):
        self.nmodel += 1
        self.name = name

        print('creating model')

        try:
            os.mkdir(out_dir+'/model_{nm}'.format(nm = self.nmodel))
        except:
            pass

        return model.single_osc(inp_dir = self.inp_dir,
                                out_dir = self.out_dir,
                                nmodel = self.nmodel,
                                name = name)
                                
    def model_double_osc(self, name = ''):
        self.nmodel += 1
        self.name = name

        print('creating model')
        try:
            os.mkdir(self.out_dir+'/model_{nm}'.format(nm = self.nmodel))
        except:
            pass


        return model.double_osc(inp_dir = self.inp_dir,
                                out_dir = self.out_dir,
                                nmodel = self.nmodel,
                                name = name)
    def model_dist_disp_osc(self, name = ''):
        self.nmodel += 1
        self.name = name

        print('creating model'+str(self.nmodel))
        try:
            os.mkdir(self.out_dir+'/model_{nm}'.format(nm = self.nmodel))
        except:
            pass


        return model.dist_disp_osc(inp_dir = self.inp_dir,
                                out_dir = self.out_dir,
                                nmodel = self.nmodel,
                                name = name)

    def experiment(self,file='', col=[0,1],name = ''):
        self.nexp+=1
        return experiment.experiment(expfile = file, \
                                    columns= col, nexp = self.nexp, name = name)


    def visual(self,model_list=[],exp=[]):
        return visual.plot(model_list=model_list,exp=exp)
