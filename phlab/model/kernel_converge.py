from phlab.model import input_handler
from phlab.model import spec

import numpy as np
import json
import os

class execute(object):
    """docstring for conv"""

    def __init__(self, kernel = '', local = ''):
        super(execute, self).__init__()
        self.local = local
        parameter = self.local.param['parameter']
        pmin = self.local.param['pmin']
        pmax = self.local.param['pmax']
        steps = self.local.param['steps']

        self.kernel = kernel
        self.local.conv_out_dir\
                = self.local.out_dir+'/converge_{np}/'.format(np=parameter,nm=self.local.nmodel)

        if not os.path.isdir(self.local.conv_out_dir):
            os.mkdir(self.local.conv_out_dir)

        y_temp_0 = np.zeros(int(self.local.input['nf']))
        param_space = np.linspace(pmin,pmax,steps)
        conv_arr = []
        param_arr = []
        for iter, param_i in enumerate(param_space) :
            self.local.nruns+=1
            self.local.input[parameter] = param_i
            self.local.input_class.input_update(self.local.input)
            conv_=self.kernel.kernel(self.local.input,
                                    nruns = iter,
                                    nmodel = self.local.nmodel,
                                    out_dir=self.local.conv_out_dir,
                                    temp_rixs_file = '{nruns}_rixs_raw.csv')
            conv_.cross_section()
            y_temp_1 = np.array(conv_.y_raw)
            if self.local.nruns != 1:
                dif = np.sum(abs(y_temp_1-y_temp_0))#/len(y_temp_1)
                conv_arr.append(dif)
                param_arr.append(int(param_i))
            y_temp_0 = y_temp_1
        param_space=param_space[1:]
        self.conv_arr = conv_arr
        self.param_space = param_arr
        np.savetxt(self.local.conv_out_dir+'/conv_arr.csv',\
                                    np.column_stack([param_arr,conv_arr]))
