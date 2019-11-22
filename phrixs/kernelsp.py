import json
import numpy as np
import math
from math import factorial
from scipy.special import eval_hermite as H
import functools
import config as cg
# import multiprocessing as multi
from tqdm import tqdm
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
        if self.dict['problem']['type_calc']=='dd':
            self.beta=np.sqrt(float(dict['input']['omega_ph_ex'])/float(dict['input']['omega_ph0']))
            print(self.beta)
            self.omega_ex=dict['input']['omega_ph_ex']
        try:
            self.m=int(dict['input']['nm'])
        except:
            pass
        self.f=int(dict['input']['nf'])
        self.i=int(0)

    def amplitude(self,f,i):
        def func_temp(m):
            return self.franck_condon_factors(f,m,float(self.dict['input']['g0']))\
                            *self.franck_condon_factors(m,i,float(self.dict['input']['g0']))\
                                        /(float(self.dict['input']['omega_ph0'])*(m-float(self.dict['input']['g0']))- self.det)
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
                if self.dict['problem']['method']=='fc':
                    if self.dict['problem']['type_calc']=='model':
                        return abs(self.amplitude(f,self.i))**2
                    elif self.dict['problem']['type_calc']=='dd':
                        return abs(self.amplitude_dd(f,self.i))**2
                    else:
                        print('error in kernelsp')
                else:
                    if self.dict['problem']['type_problem']=='rixs':
                        return abs(self.amplitude_gf(f))
                    elif self.dict['problem']['type_problem']=='rixs_q':
                        return abs(self.amplitude_gf_q(f,self.i))**2
                    else:
                        print('error in kernelsp')

            x,y=np.array(range(self.f))*self.dict['input']['omega_ph0'], list(map(func_temp,range(self.f)))
        elif self.nmodes==2:
            ws=[[f1,f2] for f1 in range(self.f) for f2 in range(self.f)]
            def func_temp(f):
                return abs(self.amplitude_2_(f,self.i))**2#*np.conj(self.amplitude_2_(f,self.i)))
            fr=np.array(ws).T[0]*self.dict['input']['omega_ph0']+np.array(ws).T[1]*self.dict['input']['omega_ph1']
            ws=np.array(ws)
            x,y=np.array(fr), list(tqdm(pool(processes=self.nproc).map(func_temp,tqdm(ws))))
        np.save(self.auto_save,np.vstack((x,y)))
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

    def amplitude_dd(self,f,i):
        def func_temp(ws):
            m,l,k=ws[0],ws[1],ws[2]
            return np.conj(self.X(i,k,self.beta))*\
            self.X(f,l,self.beta)*\
            self.franck_condon_factors(l,m,self.dict['input']['coupling0'])*\
            self.franck_condon_factors(m,i,self.dict['input']['coupling0'])\
            /(self.omega_ex*(m-self.dict['input']['coupling0'])-self.det)
        workspace=np.array([(m,l,k) for m in range(self.m) for l in range(self.m) for k in range(self.m)])
        return np.sum(list(map(functools.partial(func_temp),workspace)))

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
    def greens_func_cumulant(self):
        self.maxt=self.dict['input']['maxt']
        self.nstep=self.dict['input']['nstep']
        step=self.maxt/self.nstep
        t=np.linspace(0.,self.maxt,self.nstep)
        omegaq=self.dict['input']['omega_ph0']*2*np.pi
        G0x=-1.j*np.exp(1.j*(np.pi*2.*self.dict['input']['omega_in'])*t)
        Fx=np.exp(self.dict['input']['g0']*(np.exp(1.j*omegaq*t)-1.j*omegaq*t-1))
        G=G0x*Fx*np.exp(-2*np.pi*self.dict['input']['gamma']*t)
        GW=np.fft.fft(G)
        w=np.fft.fftfreq(self.nstep,step)
        x=w[0:len(w)/2]
        y=abs(GW[0:len(GW)/2].imag)
        y=y/(sum(y)*(x[1]-x[0]))
        return x,y
    def amplitude_gf(self,nf):
        self.maxt=self.dict['input']['maxt']
        self.nstep=self.dict['input']['nstep']
        step=self.maxt/self.nstep
        t=np.linspace(0.,self.maxt,self.nstep)
        G0x=-1.j*np.exp(-1.j*(np.pi*2.*self.dict['input']['energy_ex'])*t)
        G=G0x
        om=self.dict['input']['omega_ph0']*2*np.pi
        gx=self.dict['input']['g0']
        Ck=gx*(np.exp(-1.j*om*t)+1.j*om*t-1)
        Dk=(np.sqrt(gx))*(np.exp(-1.j*om*t)-1)
        G=G*(Dk**nf)/np.sqrt(factorial(nf))
        Fx=np.exp(Ck)
        G=G*Fx*np.exp(-2*np.pi*self.dict['input']['gamma']*t)
        omx,intx=self.goertzel(G,self.nstep/self.maxt,self.dict['input']['energy_ex'])
        return float((intx)[0])**2
