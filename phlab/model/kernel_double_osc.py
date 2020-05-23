import json
import numpy as np
from math import factorial
from scipy.misc import *
from scipy.special import factorial2
import math
from scipy.special import eval_hermite as H
from scipy.special import binom
import functools
# import phonon_info
from tqdm import tqdm
from scipy.special import wofz
from pathos.multiprocessing import ProcessingPool as pool

class kernel(object):

    """docstring for calculations."""
    def __init__(self,dict,
                    nruns=1,
                    nmodel=1,
                    out_dir='./_output/',
                    temp_rixs_file = '/{nruns}_rixs_raw.csv'):
        super(kernel, self).__init__()

        # print('nruns:'+str(nruns)+' coupling:'+str(dict['coupling0'])+\
        # ' omega:'+str(dict['omega_ph0']))


        self.nmodel = nmodel
        self.nruns=nruns
        self.dict=dict

        self.nmodes=2
        self.det=1.j*self.dict['gamma']+self.dict['energy_ex']-self.dict['omega_in']
        self.nproc=int(dict['nf'])
        self.auto_save = out_dir+temp_rixs_file.format(nruns = self.nruns)
        try:
            self.m=int(dict['nm'])
        except:
            pass
        self.f=int(dict['nf'])
        self.i=int(0)
        self.dict['g0'] = (self.dict['coupling0']/self.dict['omega_ph0'])**2
        self.dict['g1'] = (self.dict['coupling1']/self.dict['omega_ph1'])**2
        # print(self.auto_save)


    def amplitude_2_(self,f,i):
        ws=np.array([(m1,m2) for m1 in range(self.m) for m2 in range(self.m)])
        def func_temp(m):
            return self.franck_condon_factors(f[0],m[0],self.dict['g0'])\
                    *self.franck_condon_factors(m[0],i,self.dict['g0'])\
                    *self.franck_condon_factors(f[1],m[1],self.dict['g1'])\
                    *self.franck_condon_factors(m[1],i,self.dict['g1'])\
                        /(self.dict['omega_ph0']*(m[0]-self.dict['g0'])+self.dict['omega_ph1']*(m[1]-self.dict['g1'])- self.det)
        return np.sum(list(map(functools.partial(func_temp),ws)))

    def cross_section(self):
        ws=[[f1,f2] for f1 in range(self.f) for f2 in range(self.f)]
        def func_temp(f):
            return abs(self.amplitude_2_(f,self.i))**2#*np.conj(self.amplitude_2_(f,self.i)))
        fr=np.array(ws).T[0]*self.dict['omega_ph0']+np.array(ws).T[1]*self.dict['omega_ph1']
        ws=np.array(ws)
        x,y=np.array(fr), list(tqdm(pool(processes=self.nproc).map(func_temp,tqdm(ws))))
        np.savetxt(self.auto_save,np.column_stack((x,y)))
        self.x_raw = x
        self.y_raw = y
        return x,y



    def franck_condon_factors(self,n,m,g):
        mi,no=max(m,n),min(m,n)
        part1=((-1)**mi)*np.sqrt(np.exp(-g)*factorial(mi)*factorial(no))
        func=lambda l: ((-g)**l)*((np.sqrt(g))**(mi-no))\
                /factorial(no-l)/factorial(mi-no+l)/factorial(l)
        part2=list(map(functools.partial(func), range(no+1)))
        return part1*np.sum(part2)

    def goertzel(self,samples, sample_rate, freqs):
        window_size = len(samples)
        f_step = sample_rate / float(window_size)
        f_step_normalized = 1./ window_size
        kx=int(math.floor(freqs / f_step))
        n_range = range(0, window_size)
        freq = [];power=[]
        f = kx * f_step_normalized
        w_real = math.cos(2.0 * math.pi * f)
        w_imag = math.sin(2.0 * math.pi * f)
        dr1, dr2 = 0.0, 0.0
        di1,di2 = 0.0, 0.0
        for n in n_range:
            yrt  = samples[n].real + 2*w_real*dr1-dr2
            dr2, dr1 = dr1, yrt
            yit  = samples[n].imag + 2*w_real * di1 - di2
            di2, di1 = di1, yit
        yr=dr1*w_real-dr2+1.j*w_imag*dr1
        yi=di1*w_real-di2+1.j*w_imag*di1
        y=(1.j*yr+yi)/(window_size)
        power.append(np.abs(y))
        freq.append(freqs)
        return np.array(freqs), np.array(power)



    def amplitude_gf(self,nf):
        self.maxt=self.dict['maxt']
        self.nstep=self.dict['nstep']
        step=self.maxt/self.nstep
        t=np.linspace(0.,self.maxt,self.nstep)
        G0x=-1.j*np.exp(-1.j*(np.pi*2.*self.dict['energy_ex'])*t)
        G=G0x
        om=self.dict['omega_ph0']*2*np.pi
        gx=self.dict['g0']
        Ck=gx*(np.exp(-1.j*om*t)+1.j*om*t-1)
        Dk=(np.sqrt(gx))*(np.exp(-1.j*om*t)-1)
        G=G*(Dk**nf)/np.sqrt(factorial(nf))
        Fx=np.exp(Ck)
        G=G*Fx*np.exp(-2*np.pi*self.dict['gamma']*t)
        omx,intx=self.goertzel(G,self.nstep/self.maxt,self.dict['energy_ex'])
        return float((intx)[0])**2
