import json
import numpy as np
from math import factorial
from scipy.misc import *
from scipy.special import factorial2
import math
from scipy.special import eval_hermite as H
from scipy.special import binom
import functools
import config as cg
import phonon_info
from tqdm import tqdm
from scipy.special import wofz
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
            print('beta=',self.beta)
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
                    elif self.dict['problem']['type_calc']=='2d':
                        return abs(self.amplitude(f,self.i))**2
                    else:
                        print('error in kernelsp')
                else:
                    if self.dict['problem']['type_problem']=='rixs':
                        return abs(self.amplitude_gf(f))
                    elif self.dict['problem']['type_problem']=='rixs_q':
                        print('better use kernelsp.rixs_model_q()')
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

    def X_chang(self,n,n_p,beta):
        alpha=self.dict['input']['omega_ph0']
        alphap=self.dict['input']['omega_ph_ex']
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
                self.franck_condon_factors(k,m,float(self.dict['input']['g0']))*self.franck_condon_factors(m,l,float(self.dict['input']['g0']))\
                        / (self.dict['input']['omega_ph0']*(m-self.dict['input']['g0'])-self.det)
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

class rixs_model_q(object):
    """docstring for rixs_model_q."""
    def __init__(self,dict,nruns=1,dict_input=cg.dict_input_file):
        super(rixs_model_q, self).__init__()
        self.dict=dict
        self.nruns=str(nruns)
        self.auto_save=cg.temp_rixs_file\
                            +'_run_'+self.nruns+cg.extension_final
        self.q=np.linspace(-1,1,self.dict['input']['nq'])
        self.omegaq=phonon_info.energy(self.dict).omegaq
        self.fitomega=np.polyfit(self.q,self.omegaq,3)
        func_omega=np.poly1d(self.fitomega)
        self.omegax=func_omega
        self.gkq=phonon_info.coupling(self.omegaq,self.dict).gkq
        fitg=np.polyfit(self.q,self.gkq,3)
        func_g=np.poly1d(fitg)
        self.gx=func_g
        self.cumulant_kq=self.fkq()
        self.gkqx=self.gx(self.dict['input']['qx'])
        self.omegaqx=self.omegax(self.dict['input']['qx'])
        self.qx=self.dict['input']['qx']
        self.maxt=self.dict['input']['maxt']
        self.nstep=self.dict['input']['nstep']
        self.t=np.linspace(0.,self.maxt,self.nstep)

    def fkq(self):
        step=self.dict['input']['maxt']/self.dict['input']['nstep']
        t=np.linspace(0.,self.dict['input']['maxt'],self.dict['input']['nstep'])
        self.omegaq_cumulant=self.omegaq*2*np.pi
        Fkq=1.
        for gkqi,omegaqi in zip(self.gkq,self.omegaq_cumulant):
            Fkq=Fkq*np.exp(gkqi*(np.exp(-1.j*omegaqi*t)+1.j*omegaqi*t-1)/self.dict['input']['nq'])
        return Fkq

    def single_phonon(self,nph):
        G=-1.j*np.exp(-1.j*(np.pi*2.*self.dict['input']['energy_ex'])*self.t)
        self.frx=self.omegaqx*2*np.pi
        Ck=self.gkqx*(np.exp(-1.j*self.frx*self.t)+1.j*self.frx*self.t-1)
        Dk=(np.sqrt(self.gkqx))*(np.exp(-1.j*self.frx*self.t)-1)
        G=G*(Dk**nph)/np.sqrt(factorial(nph))
        G=G*self.cumulant_kq*np.exp(-2*np.pi*self.dict['input']['gamma']*self.t)
        omx,intx=self.goertzel(G,self.nstep/self.maxt,self.dict['input']['energy_ex'])
        return float(intx[0])**2

    def multi_phonon(self,qmap,nph):
        G=-1.j*np.exp(-1.j*(np.pi*2.*self.dict['input']['energy_ex'])*self.t)
        D=1.;
        for n in range(nph):
            Dk=(np.sqrt(self.gkq[qmap[n]]))*(np.exp(-1.j*self.omegaq[qmap[n]]*2.*np.pi*self.t)-1.)
            D=D*Dk
        G=G*(D)/np.sqrt(factorial(nph))
        G=G*self.cumulant_kq*np.exp(-2*np.pi*self.dict['input']['gamma']*self.t)
        omx,intx=self.goertzel(G,self.nstep/self.maxt,self.dict['input']['energy_ex'])
        return float(intx[0])**2

    def cross_section(self):
        loss=[];r=[]
        for nph in tqdm(range(int(self.dict['input']['nf']))):
            if nph==0:
                loss_temp,r_temp=[self.omegaqx*nph],[self.single_phonon(nph)]
            elif nph==1:
                loss_temp,r_temp=[self.omegaqx*nph],[self.single_phonon(nph)]
            elif nph==2:
                qmap=init_map(self.dict).phonon_sec_q()
                r_temp=list(map(lambda x: self.multi_phonon(x,2), qmap))
                r_temp=np.array(r_temp)/len(qmap)
                loss_temp=list(map(lambda x: self.omegaq[x[0]]+self.omegaq[x[1]], qmap))
            elif nph==3:
                qmap=init_map(self.dict).phonon_thr()
                r_temp=list(map(lambda x: self.multi_phonon(x,3), qmap))
                r_temp=np.array(r_temp)/len(qmap)
                loss_temp=list(map(lambda x: self.omegaq[x[0]]+self.omegaq[x[1]]+self.omegaq[x[2]], qmap))
            elif nph==4:
                qmap=init_map(self.dict).phonon_fou()
                r_temp=list(map(lambda x: self.multi_phonon(x,4), qmap))
                r_temp=np.array(r_temp)/len(qmap)
                loss_temp=list(map(lambda x: self.omegaq[x[0]]+self.omegaq[x[1]]\
                            +self.omegaq[x[2]]+self.omegaq[x[3]], qmap))
            loss.extend(loss_temp)
            r.extend(r_temp)
        # print(len(r),len(loss))
        np.save(self.auto_save,np.vstack((loss,r)))

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

class init_map(object):
    """docstring for _init_map."""
    def __init__(self,dict):
        super(init_map, self).__init__()
        self.dict=dict
        self.Nq=int(self.dict['input']['nq'])
        self.q=np.linspace(-1.,1.,self.Nq)
        self.qx=self.dict['input']['qx']
    def phonon_sec(self):
        qmap=[]
        for i in range(self.Nq):
            for j in range(self.Nq):
                if np.round(self.q[i]+self.q[j]-self.qx,5) == 0.:
                    qmap.append([i,j])
        return (qmap)
    def phonon_sec_q(self):
        qmap=[]
        for i in range(self.Nq):
            for j in range(self.Nq):
                if np.round(self.q[i]+self.q[j]-self.qx,5) == 0.:
                    qmap.append([i,j])
        return (qmap)
    def phonon_thr(self):
        qmap=[]
        for i in range(self.Nq):
            for j in range(self.Nq):
                for k in range(self.Nq):
                    if np.round(self.q[i]+self.q[j]+self.q[k]-self.qx,5)==0.:
                        qmap.append([i,j,k])
        return np.array(qmap)
    def phonon_fou(self):
        qmap=[]
        for i in range(self.Nq):
            for j in range(self.Nq):
                for k in range(self.Nq):
                    for l in range(self.Nq):
                        if np.round(self.q[i]+self.q[j]+self.q[k]+self.q[l]-self.qx,5)==0.:
                            qmap.append([i,j,k,l])
        return np.array(qmap)
