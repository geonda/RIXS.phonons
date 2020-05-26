import numpy as np
import json
import os

class input_handler(object):

    """

    Contains methods to read and update input.

    Args:
        input_default: dict
            dictionary with input parameters
        inp_dir: str
            name of the input directory
        nmodel: int
            id number of the model
        model_name: str
            name of the model


    Attributes:
        input: dict
            dictionary with input parameters
    """

    def __init__(self,
                input_default = {},
                inp_dir = './_input/',
                nmodel = 1,
                inp_name = 'input_model_{nm}.json',
                model_name='1d'):
        super(input_handler, self).__init__()
        self.inp_dir = inp_dir
        self.input_name = inp_name
        self.input_default = input_default
        self.nmodel = nmodel
        self.input = self.parsing_input()

    def input_update(self,input_temp):
        input_name = 'input_model_{nm}.json'.format(nm = self.nmodel)
        with open(self.inp_dir+input_name, 'w') as f:
            json.dump(input_temp,f,indent=1)

    def parsing_input(self):
        try:
            temp = self.read_input(file = self.input_name.format(nm = self.nmodel))
            print('done parsing input')
        except:
            print('no input found')
            print('creating new input')
            print('warning: please check new input')
            temp = self.create_default_input(file = self.input_name.format(nm = self.nmodel),
                                            temp_input = self.input_default)
        return temp

    def read_input(self,file=''):
        with open(self.inp_dir+file) as f:
            temp=json.load(f)
        if temp['model'] != model_name :
            print('overwriting input file of another model')
            self.create_default_input(file = self.input_name.format(nm = self.nmodel),
                                            temp_input = self.input_default)

        return temp

    def create_default_input(self,file='',temp_input={}):
        with open(self.inp_dir+file, 'w') as fp:
            json.dump(temp_input,fp,indent=1)
        return temp_input
