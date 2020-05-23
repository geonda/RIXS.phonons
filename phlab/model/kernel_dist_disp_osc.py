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

        # # print('nruns:'+str(nruns)+' coupling:'+str(dict['coupling0'])+\
        # ' omega:'+str(dict['omega_ph']))


        self.nmodel = nmodel
        self.nruns=nruns
        self.dict=dict

        self.nmodes=1
        self.det=1.j*self.dict['gamma']+self.dict['energy_ex']-self.dict['omega_in']
        self.nproc=int(dict['nf'])
        self.auto_save = out_dir+temp_rixs_file.format(nruns = self.nruns)

        self.beta=np.sqrt(float(dict['omega_ph_ex'])/float(dict['omega_ph']))
        # print('beta=',self.beta)
        self.omega_ex=dict['omega_ph_ex']

        self.m=int(dict['nm'])
        self.f=int(dict['nf'])
        self.i=int(0)
        self.dict['g0'] = (self.dict['coupling']/self.dict['omega_ph'])**2
        # print(self.auto_save)


    def cross_section(self):
        if self.nmodes==1:
            def func_temp(f):
                return abs(self.amplitude_dd(f,self.i))**2
            x,y=np.array(range(self.f))*self.dict['omega_ph'], list(map(func_temp,range(self.f)))
        np.savetxt(self.auto_save,np.column_stack((x,y)))
        self.x_raw = x
        self.y_raw = y
        return x,y

    def X(self,l,k,beta):
        b1=np.sqrt(2.*beta/(1.+(beta**2.)))
        # print((1-pow(beta,2.))/(2.*(1.+pow(beta,2.))))
        # b2=pow((1-pow(beta,2.))/(2.*(1.+pow(beta,2.))),((l+k)/2))
        b2=((1-beta**2.)/(2.*(1.+beta**2.)))**int(((l+k)/2))
        # print((1-beta**2.)/(2.*(1.+beta**2.)),(l+k)/2,b2)
        # print((l+k)/2)

        b3=np.sqrt(float(factorial(k)*factorial(l)))
        def fun(j):
            s1=(4.*beta/(1.-beta**2.))**j
            s2=(-1.j)**(l-j)
            s3=factorial(k-j)
            s4=factorial(l-j)
            s5=factorial(j)
            return s1*s2*H(l-j,0)*H(k-j,0)/(s3*s4*s5)
        out=list(map(functools.partial(fun), (range(min(l,k)+1))))
        return b1*b2*b3*np.sum(out)

    def X_chang(self,n,n_p,beta):
        alpha=self.dict['omega_ph']
        alphap=self.dict['omega_ph_ex']
        A=2*np.sqrt(float(alpha*alphap))/(alpha+alphap)
        F=A/float((factorial(n)*factorial(n_p)*(2.**(n+n_p))))
        ws=np.array([(k,kp) for k in range(n+1) for kp in range(n_p+1)])
        def func_temp(ws):
            k,kp=ws[0],ws[1]
            O1=binom(n_p, kp)*binom(n,k)
            O2=H(n_p-kp,0)*H(n-k,0)
            O3=((2*np.sqrt(alphap))**kp)*((2*np.sqrt(alpha))**k)
            if (k+kp) % 2 != 0:
                return 0.
            else:
                return O1*O2*O3*factorial2(int(k+kp-1))\
                                /np.sqrt((alpha+alphap)**(k+kp))
        out=np.sum(list(map(functools.partial(func_temp),ws)))
        return np.sqrt(float(F))*out

    def amplitude_dd(self,f,i):
        def func_temp(ws):
            m,l,k=ws[0],ws[1],ws[2]
            return  np.conj(self.X_chang(i,k,self.beta))*\
                        self.X_chang(l,f,self.beta)*\
                self.franck_condon_factors(k,m,float(self.dict['g0']))*self.franck_condon_factors(m,l,float(self.dict['g0']))\
                        / (self.dict['omega_ph']*(m-self.dict['g0'])-self.det)
        workspace=np.array([(m,l,k) for m in range(self.m) for l in range(self.m) for k in range(self.m)])
        return np.sum(list(map(functools.partial(func_temp),workspace)))

    def franck_condon_factors(self,n,m,g):
        mi,no=max(m,n),min(m,n)
        part1=((-1)**mi)*np.sqrt(np.exp(-g)*factorial(mi)*factorial(no))
        func=lambda l: ((-g)**l)*((np.sqrt(g))**(mi-no))\
                /factorial(no-l)/factorial(mi-no+l)/factorial(l)
        part2=list(map(functools.partial(func), range(no+1)))
        return part1*np.sum(part2)
