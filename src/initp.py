from __future__ import print_function
import json
class init_problem(object):
    """docstring for init_problem."""
    def __init__(self,type_problem='rixs',method='fc',type_calc='model',\
                    el_space=1, vib_space=1):
        super(init_problem, self).__init__()
        self.type_problem=type_problem
        self.method=method
        self.type_calc=type_calc
        self.el_space=el_space
        self.vib_space=vib_space
        self.error=False
        dict_current={'type_problem':type_problem,'method':method,\
                    'type_calc':type_calc,'el_space':str(el_space),\
                    'vib_space':str(vib_space)}
        dict_full={'type_problem':['rixs','xas','rixs_q'],'method':['gf','fc'],\
                    'type_calc':['model','fit','1d','2d','dd'],'el_space':'1',\
                    'vib_space':['1','2','3']}
        for key in dict_current:
                if not (dict_current[key] in dict_full[key]) :
                    self.error=True
                    print('error in arguments')
        if self.error==False:
            with open('problem.json', 'w') as fp:
                json.dump(dict_current,fp,indent=1)
        self.dict_current=dict_current
    def help(self):
        print('type_problem: rixs (default), xas, rixs_q ')
        print('method: gf, fc (default) ')
        print('type_clac: model (default), fit ')
        print('el_space: 1 (default) ')
        print('vib_space: 1 (default), 2, 3 ')
