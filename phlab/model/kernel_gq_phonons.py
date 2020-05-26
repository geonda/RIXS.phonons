import json
import numpy as np
from math import factorial
from scipy.misc import *
from scipy.special import factorial2
import math
from scipy.special import eval_hermite as H
from scipy.special import binom
import functools
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

        self.dict=dict
        self.nmodel = nmodel
        self.nruns=str(nruns)
        self.auto_save = out_dir+temp_rixs_file.format(nm = self.nmodel,\
                                                            nruns = self.nruns)
        self.dict['extra']='x'


        self.phonon_info_qx = [0.,1.]
        self.phonon_info_qy = [0.,0.]
        self.phonon_info_phonon_energy = [0.195,0.16]
        self.phonon_info_coupling = [0.75,0.45]
        self.phonon_info_wi =  [1./3.,2./3]

        self.phonon_info_coupling_strength =  [(self.phonon_info_coupling[0]/self.phonon_info_phonon_energy[0])**2,
                                        (self.phonon_info_coupling[1]/self.phonon_info_phonon_energy[1])**2]


        self.qx=np.hstack((-np.array(self.phonon_info_qx),np.array(self.phonon_info_qx)))
        self.qy=np.hstack((-np.array(self.phonon_info_qy),\
                                        np.array(self.phonon_info_qy)))
        self.phonon_energy=np.hstack((np.array(self.phonon_info_phonon_energy),\
                                        np.array(self.phonon_info_phonon_energy)))
        self.coupling_const=\
                np.hstack((np.array(self.phonon_info_coupling),\
                            np.array(self.phonon_info_coupling)))

        self.coupling_strength=\
                np.hstack((np.array(self.phonon_info_coupling_strength),\
                            np.array(self.phonon_info_coupling_strength)))
        self.w=\
                np.hstack((np.array(self.phonon_info_wi),\
                            np.array(self.phonon_info_wi)))

        self.qx=np.delete(self.qx,0)
        self.qy=np.delete(self.qy,0)
        self.w=np.delete(self.w,0)

        self.coupling_strength=np.delete(self.coupling_strength,0)
        self.coupling_const=np.delete(self.coupling_const,0)
        self.phonon_energy=np.delete(self.phonon_energy,0)


        print('w grid', self.w )
        print('qx grid', self.qx )
        print('qy grid', self.qy )

        print('Eph', self.phonon_energy )
        print('M', self.coupling_const )
        print('g', self.coupling_strength)

        self.cumulant_kq=self.fkq()

        self.maxt=self.dict['maxt']
        self.nstep=int(self.dict['nstep'])
        self.t=np.linspace(0.,self.maxt,self.nstep)

    def fkq(self):
        step=self.dict['maxt']/self.dict['nstep']
        t=np.linspace(0.,self.dict['maxt'],int(self.dict['nstep']))
        self.omegaq_cumulant=self.phonon_energy*2*np.pi
        Fkq=1.
        for gkqi,omegaqi,wi in zip(self.coupling_strength,self.omegaq_cumulant,self.w):
            Fkq=wi*Fkq*np.exp(gkqi*(np.exp(-1.j*omegaqi*t)+1.j*omegaqi*t-1)/len(self.omegaq_cumulant))
        return Fkq

    def single_phonon(self,qmap,nph):
        G=-1.j*np.exp(1.j*(-np.pi*2.*self.dict['energy_ex'])*self.t)
        print(qmap)
        self.frx=self.phonon_energy[qmap[0]]*2*np.pi
        self.gkqx=self.coupling_strength[qmap[0]]
        ###log(self.gkqx)
        Dk=(np.sqrt(self.gkqx))*(np.exp(-1.j*self.frx*self.t)-1)
        G=G*(Dk**nph)/np.sqrt(factorial(nph))
        G=G*self.cumulant_kq*np.exp(-2*np.pi*self.dict['gamma']*self.t)
        omx,intx=self.goertzel(G,self.nstep/self.maxt,self.dict['omega_in'])
        return float(intx[0])**2

    def one_phonon(self,qmap,nph):
        G=-1.j*np.exp(1.j*(-np.pi*2.*self.dict['energy_ex'])*self.t)
        print(qmap)
        self.frx=self.phonon_energy[qmap[0]]*2*np.pi
        self.gkqx=self.coupling_strength[qmap[0]]
        ###log(self.gkqx)
        Dk=(np.sqrt(self.gkqx))*(np.exp(-1.j*self.frx*self.t)-1)
        G=G*(Dk**nph)/np.sqrt(factorial(nph))
        G=G*self.cumulant_kq*np.exp(-2*np.pi*self.dict['gamma']*self.t)
        omx,intx=self.goertzel(G,self.nstep/self.maxt,self.dict['omega_in'])
        det=np.linspace(-1,1,200)
        def temp(d):
            omx,intx=self.goertzel(G,self.nstep/self.maxt,self.dict['omega_in']+d)
            return abs(intx[0])**2
        np.savetxt('./_out/1ph_profile_model',np.vstack((det,list(map(temp,det)))))
        return float(intx[0])**2

    def two_phonon(self,qmap,nph):
        def temp(d):
            def nfunc(qmap):
                G=-1.j*np.exp(-1.j*(np.pi*2.*self.dict['energy_ex'])*self.t)
                D=1.;
                for n in range(nph):
                    Dk=(np.sqrt(self.coupling_strength[qmap[n]]))*(np.exp(-1.j*self.phonon_energy[qmap[n]]*2.*np.pi*self.t)-1.)
                    D=D*Dk
                G=G*(D)/np.sqrt(factorial(nph))
                G=G*self.cumulant_kq*np.exp(-2*np.pi*self.dict['gamma']*self.t)
                omx,intx=self.goertzel(G,self.nstep/self.maxt,self.dict['omega_in']+d)
                return abs(intx[0])**2
            r_temp=list(map(lambda x: nfunc(x), qmap))
            r_temp=np.array(r_temp)/len(qmap)
            return sum(r_temp)

        det=np.linspace(-1,1,200)

        np.savetxt('./_out/2ph_profile_model',np.vstack((det,list(map(temp,det)))))


    def greens_func_cumulant(self):
        self.maxt=self.dict['maxt']
        self.nstep=self.dict['nstep']
        step=self.maxt/self.nstep
        t=np.linspace(0.,self.maxt,int(self.nstep))
        G0x=-1.j*np.exp(-1.j*(np.pi*2.*self.dict['energy_ex'])*t)
        G=G0x*self.cumulant_kq*np.exp(-2*np.pi*self.dict['gamma']*t)
        GW=np.fft.ifft(G)
        w=np.fft.fftfreq(int(self.nstep),step)
        x=w[0:int(len(w)/2)]
        y=abs(GW[0:int(len(GW)/2)].imag)
        y=y/(sum(y)*(x[1]-x[0]))
        return x,y

    def greens_func_cumulant_gamma(self):
        qmap = init_map_2d(self.qx,self.qy,0,0).phonon_fir()

        self.maxt=self.dict['maxt']
        self.nstep=self.dict['nstep']
        step=self.maxt/self.nstep
        t=np.linspace(0.,self.maxt,int(self.nstep))

        G0x=-1.j*np.exp(1.j*(np.pi*2.*self.dict['omega_in'])*t)


        self.frx=self.phonon_energy[qmap[0]]*2*np.pi
        self.gkqx=self.coupling_strength[qmap[0]]

        ###log(self.gkqx)

        Ck=self.gkqx*(np.exp(1.j*self.frx*self.t)-1.j*self.frx*self.t-1)
        G=G0x*np.exp(Ck)*np.exp(-2*np.pi*self.dict['gamma']*t)

        GW=np.fft.fft(G)
        w=np.fft.fftfreq(int(self.nstep),step)
        x=w[0:int(len(w)/2)]
        y=abs(GW[0:int(len(GW)/2)].imag)

        y=y/(sum(y)*(x[1]-x[0]))
        return x,y

    def multi_phonon(self,qmap,nph):
        G=-1.j*np.exp(-1.j*(np.pi*2.*self.dict['energy_ex'])*self.t)
        D=1.;
        for n in range(nph):
            Dk=(np.sqrt(self.coupling_strength[qmap[n]]))*(np.exp(-1.j*self.phonon_energy[qmap[n]]*2.*np.pi*self.t)-1.)
            D=D*Dk
        G=G*(D)/np.sqrt(factorial(nph))
        G=G*self.cumulant_kq*np.exp(-2*np.pi*self.dict['gamma']*self.t)

        omx,intx=self.goertzel(G,self.nstep/self.maxt,self.dict['omega_in'])

        # print(qmap,'e:',self.phonon_energy[qmap[0]],self.phonon_energy[qmap[1]],\
            # 'g:',self.coupling_strength[qmap[0]],self.coupling_strength[qmap[1]],
            # 'I:',float(intx[0])**2)

        return abs(float(intx[0]))**2

    def cross_section(self):
        loss=[];r=[]
        # try:
        if self.dict['extra']=='xas':
            x,y=self.greens_func_cumulant()
            np.save('./_out/xas.npy',np.vstack((x,y)))
            x,y=self.greens_func_cumulant_gamma()
            np.save('./_out/xas_gamma.npy',np.vstack((x,y)))
            qmap=init_map_2d(self.qx,self.qy,0,0).phonon_fir()
            self.one_phonon(qmap,1)
            qmap=init_map_2d(self.qx,self.qy,0,0).phonon_sec()

            self.two_phonon(qmap,2)
            print('xas done')
        print(max(self.qx),max(self.qy))
        if self.dict['extra']=='q':
            qx=0.13333333#0.06666 #0.19047619
        else:
            qx=0
        # except:
            # pass
            # print('no xas')
        for nph in tqdm(range(int(self.dict['nf']))):

            if nph==0:
                print('0 phonon : ')
                qmap=init_map_2d(self.qx,self.qy,qx,0).phonon_fir()
                loss_temp,r_temp=[nph],[self.single_phonon(qmap,nph)]
            elif nph==1:
                print('1 phonon : ')
                qmap=init_map_2d(self.qx,self.qy,qx,0).phonon_fir()
                ##log(len(qmap))
                loss_temp,r_temp=[self.phonon_energy[qmap[0]]*nph],[self.w[1]*self.single_phonon(qmap,1)]
                ###log(r_temp)
            elif nph==2:
                print('2 phonon : ')
                # qmap=init_map_2d(self.qx,self.qy,0,0).phonon_fir()
                # ##log(len(qmap))
                # loss_temp,r_temp=[self.phonon_energy[qmap[0]]*nph],[self.single_phonon(qmap,2)]
                ###log(r_temp)
                qmap=init_map_2d(self.qx,self.qy,qx,0).phonon_sec()
                # print(qmap)
                ##log(len(qmap))
                r_temp=list(map(lambda x: self.w[x[0]]*self.w[x[1]]*self.multi_phonon(x,2), qmap))
                # print('nph=2')
                # print(r_temp)
                # print(min(r_temp),max(r_temp))
                r_temp=np.array(r_temp)/len(qmap)
                # ##log(r_temp)
                loss_temp=list(map(lambda x: self.phonon_energy[x[0]]+self.phonon_energy[x[1]], qmap))
                #print(loss_temp)
            elif nph==3:
                # qmap=init_map_2d(self.qx,self.qy,0,0).phonon_fir()
                # # ##log(len(qmap))
                # loss_temp,r_temp=[self.phonon_energy[qmap[0]]*nph],[self.single_phonon(qmap,3)]
                # ###log(r_temp)
                qmap=init_map_2d(self.qx,self.qy,qx,0).phonon_thr()
                ##log(len(qmap))
                r_temp=list(map(lambda x: self.w[x[0]]*self.w[x[1]]*self.w[x[2]]*self.multi_phonon(x,nph), qmap))
                r_temp=np.array(r_temp)/len(qmap)
                # print("ici")
                # print(min(r_temp),max(r_temp))
                loss_temp=list(map(lambda x:  self.phonon_energy[x[0]]\
                                +self.phonon_energy[x[1]]+self.phonon_energy[x[2]], qmap))
            elif nph==4:
                qmap=init_map_2d(self.qx,self.qy,qx,0).phonon_fou()
                ##log(len(qmap))
                r_temp=list(map(lambda x: self.w[x[0]]*self.w[x[1]]*self.w[x[2]]*self.w[x[3]]*self.multi_phonon(x,4), qmap))
                r_temp=np.array(r_temp)/len(qmap)
                loss_temp=list(map(lambda x: self.phonon_energy[x[0]]+self.phonon_energy[x[1]]\
                            +self.phonon_energy[x[2]]+self.phonon_energy[x[3]], qmap))

            elif nph==5:
                qmap=init_map_2d(self.qx,self.qy,qx,0).phonon_fiv()
                ##log(len(qmap))
                r_temp=list(map(lambda x: self.multi_phonon(x,5), qmap))
                r_temp=np.array(r_temp)/len(qmap)
                loss_temp=list(map(lambda x: self.phonon_energy[x[0]]+self.phonon_energy[x[1]]\
                            +self.phonon_energy[x[2]]+self.phonon_energy[x[3]]+self.phonon_energy[x[4]], qmap))
        # print(len(r),len(loss))
        # print(loss_temp,r_temp)
            loss.extend(loss_temp)
            r.extend(r_temp)
        # np.save(self.auto_save,np.vstack((loss,r)))
        np.savetxt(self.auto_save,np.column_stack((loss,r)))
        self.x_raw = loss
        self.y_raw = r

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
        power.append(abs(y))
        freq.append(freqs)
        return np.array(freqs), np.array(power)

class init_map(object):
    """docstring for _init_map."""
    def __init__(self,dict):
        super(init_map, self).__init__()
        self.dict=dict
        self.Nq=int(self.dict['nq'])
        self.q=np.linspace(-1.,1.,self.Nq)
        self.qx=self.dict['qx']
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

class init_map_2d(object):
    """docstring for _init_map."""
    def __init__(self,qx,qy,qtotx,qtoty):
        super(init_map_2d, self).__init__()
        self.qx,self.qy,self.qtotx,self.qtoty=qx,qy,qtotx,qtoty
        self.size=len(self.qx)

    def phonon_fir(self):
        qmap=[]
        for i in range(self.size):
            if np.round(self.qx[i]-self.qtotx,3) == 0. and \
                    np.round(self.qy[i]-self.qtoty,3) == 0.:
                # print([i],'x :',self.qx[i],'y :' ,self.qy[i])
                qmap.append(i)
        return qmap

    def phonon_sec(self):
        qmap=[]
        for i in range(self.size):
            for j in range(self.size):
                if np.round(self.qx[i]+self.qx[j]-self.qtotx,3) == 0. and \
                            np.round(self.qy[i]+self.qy[j]-self.qtoty,3) == 0:
                    # print([i,j],'x :',self.qx[i],self.qx[j],'y :' ,self.qy[i],self.qy[j])
                    qmap.append([i,j])
        # print(qmap)
        return qmap

    def phonon_thr(self):
        qmap=[]
        for i in range(self.size):
            for j in range(self.size):
                for k in range(self.size):
                    qxi=self.qx[i]+self.qx[j]+self.qx[k]
                    qyi=self.qy[i]+self.qy[j]+self.qy[k]
                    if np.round(qxi-self.qtotx,3) == 0. and np.round(qyi-self.qtoty,3) == 0.:
                        qmap.append([i,j,k])
        # print(qmap)
        return np.array(qmap)

    def phonon_fou(self):
        qmap=[]
        for i in range(self.size):
            for j in range(self.size):
                for k in range(self.size):
                    for l in range(self.size):
                        qxi=self.qx[i]+self.qx[j]+self.qx[k]+self.qx[l]
                        qyi=self.qy[i]+self.qy[j]+self.qy[k]+self.qy[l]
                        if np.round(qxi-self.qtotx,3) == 0. and np.round(qyi-self.qtoty,3) == 0.:
                            qmap.append([i,j,k,l])
        return np.array(qmap)

    def phonon_fiv(self):
        qmap=[]
        for i in range(self.size):
            for j in range(self.size):
                for k in range(self.size):
                    for l in range(self.size):
                        for m in range(self.size):
                            qxi=self.qx[i]+self.qx[j]+self.qx[k]+self.qx[l]+self.qx[m]
                            qyi=self.qy[i]+self.qy[j]+self.qy[k]+self.qy[l]+self.qy[m]
                            if np.round(qxi-self.qtotx,3) == 0. and np.round(qyi-self.qtoty,3) == 0.:
                                qmap.append([i,j,k,l,m])
        return np.array(qmap)
