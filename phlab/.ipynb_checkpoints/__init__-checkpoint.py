# from phpack.workspace.lib import *
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
        self.input_name='input_model_{nm}.json'
        try:
            os.mkdir(self.inp_dir)
        except:
            pass

    def create_model(self):

        self.nmodel += 1
        try:
            temp = self.read_input(file = self.input_name.format(nm = self.nmodel))
            print('done parsing input')
        except:
            print('no input found')
            print('creating new input')
            print('waring : please check new input')
            temp = self.create_default_input(file = self.input_name.format(nm = self.nmodel))

        return model.model(input = temp,
                            inp_dir = self.inp_dir,
                            out_dir = self.out_dir,
                            nmodel = self.nmodel)

    def create_experiment(self,file='', col=[0,1],name = ''):
        self.nexp+=1
        return experiment.experiment(expfile = file, \
                                    columns= col, nexp = self.nexp, name = name)

    def read_input(self,file=''):
        with open(self.inp_dir+file) as f:
            temp=json.load(f)
        return temp

    def create_default_input(self,file=''):
        temp = {
        'problem_type': 'rixs',
        'type_calc': '1d',
        'method': 'fc',
        'vib_space': 1,
         "coupling0": 0.1,
         "omega_ph0": 0.1,
         "nf": 10.0,
         "nm": 100.0,
         "energy_ex": 10.0,
         "omega_in": 10.0,
         "gamma": 0.1,
         "gamma_ph": 0.01,
         "alpha_exp": 0.01,
        }
        with open(self.inp_dir+file, 'w') as fp:
            json.dump(temp,fp,indent=1)
        return temp

    def connect_visual(self,model_list=[],exp=[]):
        return visual.graph(model_list=model_list,exp=exp)
