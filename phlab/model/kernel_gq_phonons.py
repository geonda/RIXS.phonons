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
import dask.array as da

##

# import dask.dataframe as dd
# import dask.array as dp
# from dask.distributed import Client, progress
# client = Client(processes=False, threads_per_worker=4,
#                 n_workers=1, memory_limit='2GB')
# print(client)

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
        self.maxt=self.dict['maxt']
        self.nstep=int(self.dict['nstep'])

        self.t=(np.linspace(0.,self.maxt,self.nstep))


        # read dataframe

        self.q_points = np.loadtxt('q_sampling.csv')
        self.qx = self.q_points.T[0]
        self.qy = self.q_points.T[1]
        print(self.qx)
        self.q_weights = np.loadtxt('q_weights.csv')
        self.q_coupling = np.loadtxt('q_coupling.csv')
        self.q_phonon = np.loadtxt('q_phonon.csv')
        self.q_strength = (self.q_coupling/self.q_phonon)**2

        print('Eph', self.q_phonon )

        self.cumulant_kq = self.fkq()


    def fkq(self):
        Fkq=1.
        for gkqi,omegaqi,wi in zip(self.q_strength,\
                            self.q_phonon*2*np.pi,self.q_weights):
            cumulant = \
                (gkqi*(np.exp(-1.j*omegaqi*self.t)+1.j*omegaqi*self.t-1)*wi/len(self.q_points))
            Fkq=Fkq*np.exp(cumulant)
        return Fkq


    def multi_phonon(self,qmap,nph):
        G=-1.j*np.exp(-1.j*(np.pi*2.*self.dict['energy_ex'])*self.t)
        D=1.

        for n in range(nph):
            Dk=(np.sqrt(self.q_strength[qmap[n]]))*(np.exp(-1.j*self.q_phonon[qmap[n]]*2.*np.pi*self.t)-1.)
            D=D*Dk

        G=G*(D)/np.sqrt(factorial(nph))
        G=G*self.cumulant_kq*np.exp(-2*np.pi*self.dict['gamma']*self.t)
        # das
        G = np.array(G)

        intx,_=(self.goertzel(G,self.nstep/self.maxt,self.dict['omega_in']))
        return abs(float(intx[0]))**2

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
        return  np.array(power), np.array(freqs)

    def cross_section(self):
        loss=[];r=[]
        qx =0
        for nph in tqdm(range(self.dict['nf'])):
            if nph==0:
                print('0 phonon : ')
                qmap=(init_map_2d(self.qx,self.qy,qx,0).phonon_fir())
                loss_temp,r_temp=[nph],[(self.multi_phonon(qmap,nph))]

            elif nph==1:
                print('1 phonon : ')
                qmap=(init_map_2d(self.qx,self.qy,qx,0).phonon_fir())
                ##log(len(qmap))
                loss_temp,r_temp=[self.q_phonon[qmap[0]]*nph],[(self.q_weights[1]*self.multi_phonon(qmap,1))]
                ###log(r_temp)
            elif nph==2:
                print('2 phonon : ')
                # qmap=init_map_2d(self.qx,self.qy,0,0).phonon_fir()
                # ##log(len(qmap))
                # loss_temp,r_temp=[self.q_phonon[qmap[0]]*nph],[self.single_phonon(qmap,2)]
                ###log(r_temp)
                qmap=init_map_2d(self.qx,self.qy,qx,0).phonon_sec()
                # print(qmap)
                ##log(len(qmap))
                r_temp=list(map(lambda x: self.q_weights[x[0]]*self.q_weights[x[1]]*self.multi_phonon(x,2), qmap))
                # print('nph=2')
                # print(r_temp)
                # print(min(r_temp),max(r_temp))
                r_temp=np.array(r_temp)/len(qmap)
                # ##log(r_temp)
                loss_temp=list(map(lambda x: self.q_phonon[x[0]]+self.q_phonon[x[1]], qmap))
                #print(loss_temp)
            elif nph==3:
                # qmap=init_map_2d(self.qx,self.qy,0,0).phonon_fir()
                # # ##log(len(qmap))
                # loss_temp,r_temp=[self.q_phonon[qmap[0]]*nph],[self.single_phonon(qmap,3)]
                # ###log(r_temp)
                qmap=init_map_2d(self.qx,self.qy,qx,0).phonon_thr()
                ##log(len(qmap))
                r_temp=list(map(lambda x: self.q_weights[x[0]]*self.q_weights[x[1]]*self.q_weights[x[2]]*self.multi_phonon(x,nph), qmap))
                r_temp=np.array(r_temp)/len(qmap)
                # print("ici")
                # print(min(r_temp),max(r_temp))
                loss_temp=list(map(lambda x:  self.q_phonon[x[0]]\
                                +self.q_phonon[x[1]]+self.q_phonon[x[2]], qmap))
            elif nph==4:
                qmap=init_map_2d(self.qx,self.qy,qx,0).phonon_fou()
                ##log(len(qmap))
                r_temp=list(map(lambda x: self.q_weights[x[0]]*self.q_weights[x[1]]*self.q_weights[x[2]]*self.q_weights[x[3]]*self.multi_phonon(x,4), qmap))
                r_temp=np.array(r_temp)/len(qmap)
                loss_temp=list(map(lambda x: self.q_phonon[x[0]]+self.q_phonon[x[1]]\
                            +self.q_phonon[x[2]]+self.q_phonon[x[3]], qmap))

            elif nph==5:
                qmap=init_map_2d(self.qx,self.qy,qx,0).phonon_fiv()
                ##log(len(qmap))
                r_temp=list(map(lambda x: self.multi_phonon(x,5), qmap))
                r_temp=np.array(r_temp)/len(qmap)
                loss_temp=list(map(lambda x: self.q_phonon[x[0]]+self.q_phonon[x[1]]\
                            +self.q_phonon[x[2]]+self.q_phonon[x[3]]+self.q_phonon[x[4]], qmap))
        # print(len(r),len(loss))
        # print(loss_temp,r_temp)
            loss.extend(loss_temp)
            r.extend(r_temp)
        # np.save(self.auto_save,np.vstack((loss,r)))
        np.savetxt(self.auto_save,np.column_stack((loss,r)))
        self.x_raw = loss
        self.y_raw = r



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
