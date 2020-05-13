from phlab.model import *
from phlab.model import kernelsp
from phlab.model import specp
import numpy as np
import json
import os
class model(object):
    """docstring for model."""

    def __init__(self,
                    input = {},
                    inp_dir = './_input/',
                    out_dir = './_output/',
                    nmodel = 0
                    ):
        super(model, self).__init__()
        print('creating model')
        try:
            os.mkdir(out_dir+'/model_{nm}'.format(nm = nmodel))
        except:
            pass
        self.inp_dir = inp_dir
        self.out_dir = out_dir
        self.nmodel = nmodel
        self.input=input
        self.nruns=0
        self.color='r'
        print('number of models : {nm}'.format(nm = self.nmodel))

    def run(self):
        self.input_update()
        self.nruns+=1
        if self.input['problem_type']=='rixs_q':
            kernelsp.rixs_model_q(self.input, nruns = self.nruns,nmodel = self.nmodel).cross_section()
        else:
            kernelsp.rixs_model(self.input, nruns = self.nruns, nmodel = self.nmodel).cross_section()
        specp.spec(self.input, nruns = self.nruns,nmodel = self.nmodel).run_broad()
        self.x,self.y=np.transpose(np.loadtxt(self.out_dir\
                                +'/model_{nm}/{nr}_rixs_phonons.csv'.format(\
                                                nm = self.nmodel, nr = self.nruns)))
        self.y_norm=self.y/max(self.y)

    def input_update(self):
        input_name = 'input_model_{nm}.json'.format(nm = self.nmodel)
        with open(self.inp_dir+input_name, 'w') as f:
            json.dump(self.input,f,indent=1)
