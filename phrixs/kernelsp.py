import json
import numpy as np
from math import factorial
import functools
import config as cg
import multiprocessing as multi
from pathos.multiprocessing import ProcessingPool as pool
class rixs_model(object):
    """docstring for calculations."""
    def __init__(self,dict,nruns=1,dict_input=cg.dict_input_file,dict_scan=cg.dict_scan_file):
        super(rixs_model, self).__init__()
        print('nruns:'+str(nruns)+' coupling:'+str(dict['input']['coupling0'])+\
        ' omega:'+str(dict['input']['omega_ph0']))
        self.nruns=str(nruns)
        self.dict=dict
        self.nmodes=int(dict['problem']['vib_space'])
        self.det=1.j*self.dict['input']['gamma']+self.dict['input']['energy_ex']-self.dict['input']['omega_in']
        # self.scan=T
        self.nproc=int(dict['input']['nf'])
        self.auto_save=cg.temp_rixs_file\
                            +'_run_'+self.nruns+cg.extension_final
        # try:
        #     with open(dict_scan) as fp:
        #         dict_scan=json.load(fp)
        # except: self.scan=False
        # with open(dict_input) as fp:
        #     dict=json.load(fp)
        # self.omega=dict['omega_ph']
        # # self.m_coupling=dict['coupling']
        # self.det=1.j*dict['gamma']+dict['energy_ex']-dict['omega_in']
        self.m=int(dict['input']['nm'])
        self.f=int(dict['input']['nf'])
        self.i=int(0)
        # if self.scan:
        #     self.m_coupling=dict_scan['coupling'][nruns-1]
        # else: self.m_coupling=dict['coupling']
        # self.g=(self.m_coupling/self.omega)**2

    def amplitude(self,f,i):
        def func_temp(m):
            return self.franck_condon_factors(f,m,self.dict['input']['g0'])\
                    *self.franck_condon_factors(m,i,self.dict['input']['g0'])\
                        /(self.dict['input']['omega_ph0']*(m-self.dict['input']['g0'])- self.det)
        return np.sum(list(map(functools.partial(func_temp),range(self.m))))

    def amplitude_2_(self,f,i):
        ws=np.array([(m1,m2) for m1 in range(self.m) for m2 in range(self.m)])
        def func_temp(m):
            return self.franck_condon_factors(f[0],m[0],self.dict['input']['g0'])\
                    *self.franck_condon_factors(m[0],i,self.dict['input']['g0'])\
                    *self.franck_condon_factors(f[1],m[1],self.dict['input']['g1'])\
                    *self.franck_condon_factors(m[1],i,self.dict['input']['g1'])\
                        /(self.dict['input']['omega_ph0']*(m[0]-self.dict['input']['g0'])+self.dict['input']['omega_ph1']*(m[1]-self.dict['input']['g1'])- self.det)
        return np.sum(list(map(functools.partial(func_temp),ws)))

    def cross_section(self):
        if self.nmodes==1:
            def func_temp(f):
                return abs(self.amplitude(f,self.i)*np.conj(self.amplitude(f,self.i)))**2
            x,y=np.array(range(self.f))*self.dict['input']['omega_ph0'], list(map(func_temp,range(self.f)))
        elif self.nmodes==2:
            ws=[[f1,f2] for f1 in range(self.f) for f2 in range(self.f)]
            def func_temp(f):
                return abs(self.amplitude_2_(f,self.i)*np.conj(self.amplitude_2_(f,self.i)))**2
            fr=np.array(ws).T[0]*self.dict['input']['omega_ph0']+np.array(ws).T[1]*self.dict['input']['omega_ph1']
            ws=np.array(ws)
            x,y=np.array(fr), list(pool(processes=self.nproc).map(func_temp,ws))
        np.save(self.auto_save,np.vstack((x,y)))
        return x,y
    def franck_condon_factors(self,n,m,g):
        mi,no=max(m,n),min(m,n)
        part1=((-1)**mi)*np.sqrt(np.exp(-g)*factorial(mi)*factorial(no))
        func=lambda l: ((-g)**l)*(np.sqrt(g))**(mi-no)\
                /factorial(no-l)/factorial(mi-no+l)/factorial(l)
        part2=list(map(functools.partial(func), range(no+1)))
        return part1*np.sum(part2)
