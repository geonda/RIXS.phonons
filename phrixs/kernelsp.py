import json
import numpy as np
from math import factorial
import functools
import config as cg
class rixs_model(object):
    """docstring for calculations."""
    def __init__(self,nruns=1,dict_input=cg.dict_input_file,dict_scan=cg.dict_scan_file):
        super(rixs_model, self).__init__()
        self.nruns=str(nruns)
        self.scan=True
        self.auto_save=cg.temp_rixs_file\
                            +'_run_'+self.nruns+cg.extension_final
        try:
            with open(dict_scan) as fp:
                dict_scan=json.load(fp)
        except: self.scan=False
        with open(dict_input) as fp:
            dict=json.load(fp)
        self.omega=dict['omega_ph']
        # self.m_coupling=dict['coupling']
        self.det=1.j*dict['gamma']+dict['energy_ex']-dict['omega_in']
        self.m=int(dict['nm'])
        self.f=int(dict['nf'])
        self.i=int(0)
        if self.scan:
            self.m_coupling=dict_scan['coupling'][nruns-1]
        else: self.m_coupling=dict['coupling']
        self.g=(self.m_coupling/self.omega)**2

    def amplitude(self,f,i):
        def func_temp(m):
            return self.franck_condon_factors(f,m)\
                    *self.franck_condon_factors(m,i)\
                        /(self.omega*(m-self.g)- self.det)
        return np.sum(list(map(functools.partial(func_temp),range(self.m))))
    def cross_section(self):
        def func_temp(f):
            return abs(self.amplitude(f,self.i)*np.conj(self.amplitude(f,self.i)))**2
        x,y=np.array(range(self.f))*self.omega, list(map(func_temp,range(self.f)))
        np.save(self.auto_save,np.vstack((x,y)))
        return x,y
    def franck_condon_factors(self,n,m):
        mi,no=max(m,n),min(m,n)
        part1=((-1)**mi)*np.sqrt(np.exp(-self.g)*factorial(mi)*factorial(no))
        func=lambda l: ((-self.g)**l)*(np.sqrt(self.g))**(mi-no)\
                /factorial(no-l)/factorial(mi-no+l)/factorial(l)
        part2=list(map(functools.partial(func), range(no+1)))
        return part1*np.sum(part2)
