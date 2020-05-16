from phlab import model
from phlab import experiment
from phlab import visual
import json
import os

class rixs(object):
    """docstring for create_workspace."""

    def __init__(self,problem_name = 'rixs',\
                    out_dir = '/_output/',\
                    inp_dir = '/_input/'):
        super(rixs, self).__init__()
        self.nws = 1
        self.nmodel = 0
        self.nexp = 0
        self.inp_dir = inp_dir
        self.out_dir = out_dir
        self.abs_path = os.path.abspath('.')


    def model_single_osc(self, name = ''):
        self.nmodel += 1
        self.name = name
        if name !='':
            base = '{name_}'.format(name_ = self.name)
        else:
            base = 'model_{nmodel_}'.format(nmodel_ = self.nmodel)

        base = os.path.join(self.abs_path,base)

        if not os.path.isdir(base) :
             os.mkdir(base)

        inp_dir = base+self.inp_dir
        out_dir = base+self.out_dir
        print('creating model : {dir}'.format(dir = base))
        print(inp_dir)
        if not os.path.isdir(inp_dir):
            os.mkdir(inp_dir)
        if not os.path.isdir(out_dir):
            os.mkdir(out_dir)

        return model.single_osc(inp_dir = inp_dir,
                                out_dir = out_dir,
                                nmodel = self.nmodel,
                                name = name)

    def model_double_osc(self, name = ''):
        self.nmodel += 1
        self.name = name
        if name !='':
            base = '{name_}'.format(name_ = self.name)
        else:
            base = 'model_{nm}'.format(nm = self.nmodel)

        base = os.path.join(self.abs_path,base)

        if not os.path.isdir(base) :
             os.mkdir(base)
        inp_dir = base+self.inp_dir
        out_dir = base+self.out_dir
        print('creating model : {dir}'.format(dir = base))

        if not os.path.isdir(inp_dir):
            os.mkdir(inp_dir)
        if not os.path.isdir(out_dir):
            os.mkdir(out_dir)




        return model.double_osc(inp_dir = inp_dir,
                                out_dir = out_dir,
                                nmodel = self.nmodel,
                                name = name)

    def model_dist_disp_osc(self, name = ''):
        self.nmodel += 1
        self.name = name

        if name !='':
            base = '{name_}'.format(name_ = self.name)
        else:
            base = 'model_{nm}'.format(nm = self.nmodel)

        base = os.path.join(self.abs_path,base)

        if not os.path.isdir(base) :
             os.mkdir(base)
        inp_dir = base+self.inp_dir
        out_dir = base+self.inp_dir
        print('creating model : {dir}'.format(dir = base))

        if not os.path.isdir(inp_dir):
            os.mkdir(inp_dir)
        if not os.path.isdir(out_dir):
            os.mkdir(out_dir)



        return model.dist_disp_osc(inp_dir = inp_dir,
                                out_dir = out_dir,
                                nmodel = self.nmodel,
                                name = name)

    def experiment(self,file='', col=[0,1],name = ''):
        self.nexp+=1
        return experiment.experiment(expfile = file, \
                                    columns= col, nexp = self.nexp, name = name)


    def visual(self,model_list=[],exp=[]):
        return visual.plot(model_list=model_list,exp=exp)
