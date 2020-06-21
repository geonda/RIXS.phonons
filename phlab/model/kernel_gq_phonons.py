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
import time
from scipy.special import wofz
from pathos.multiprocessing import ProcessingPool as pool
from ph_info import pre_process as pre
import dask.array as da
import pandas as pd
##

# import dask.dataframe as dd
# import dask.array as dp
from dask.distributed import Client, progress
import dask
dask=False
absorption=True

def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        if 'log_time' in kw:
            name = kw.get('log_name', method.__name__.upper())
            kw['log_time'][name] = int((te - ts) * 1000)
        else:
            print('%r  %2.2f ms' % \
                  (method.__name__, (te - ts) * 1000))
        return result
    return timed

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
        self.auto_save_xas = out_dir+'/{nruns}_xas.csv'.format(nm = self.nmodel,\
                                                            nruns = self.nruns)
        self.auto_save_xas_no_q = out_dir+'/{nruns}_xas_no_q.csv'.format(nm = self.nmodel,\
                                                    nruns = self.nruns)
        self.maxt=self.dict['maxt']
        self.nstep=int(self.dict['nstep'])

        self.t=(np.linspace(0.,self.maxt,self.nstep))


        # read dataframe


        pre(nq=self.dict['nq'],m_gamma=self.dict['m_gamma'],\
                                m_k=self.dict['m_k'],
                                    r=self.dict['r'])

        df = pd.read_csv('df_ph_info.csv')

        self.q_points = np.loadtxt('q_sampling.csv')
        self.qx = df['qx'].to_numpy()
        self.qy = df['qy'].to_numpy()

        self.q_weights = df['w'].to_numpy()/sum(df['w'].to_numpy())
        self.q_coupling = df['mq'].to_numpy()
        self.q_phonon = df['ph'].to_numpy()
        self.q_strength = df['gq'].to_numpy()


        self.cumulant_kq = self.fkq()
        if absorption:
            self.absorption()
            self.absorption_no_q()

    # @timeit
    def fkq(self):
        Fkq=1.
        for gkqi,omegaqi,wi in zip(self.q_strength,\
                            self.q_phonon*2*np.pi,self.q_weights):
            cumulant = \
                (gkqi*(np.exp(-1.j*omegaqi*self.t)+1.j*omegaqi*self.t-1)*wi)
            Fkq=Fkq*np.exp(cumulant)
        return Fkq

    def absorption(self):

        gamma = self.dict['gamma']
        ex=self.dict['energy_ex']
        g0=-1.j*np.exp(-1.j*(np.pi*2.*ex)*self.t)
        g=g0*self.cumulant_kq*np.exp(-2*np.pi*gamma*self.t)

        gw = np.fft.ifft(g)
        w = np.fft.fftfreq(self.nstep,self.t[1]-self.t[0])

        gw=gw[0:int(len(gw)/2)]
        w=w[0:int(len(w)/2)]

        np.savetxt(self.auto_save_xas,np.column_stack((w,abs(gw.imag))))

        self.xas_freq =  w
        self.xas_int = abs(gw.imag)

    def absorption_no_q(self):

        gamma = self.dict['gamma']
        ex=self.dict['energy_ex']
        g0=-1.j*np.exp(-1.j*(np.pi*2.*ex)*self.t)

        gkqi=self.q_strength[0]
        omegaqi=self.q_phonon[0]*2*np.pi

        cumulant = np.exp(gkqi*(np.exp(-1.j*omegaqi*self.t)+1.j*omegaqi*self.t-1))

        g=g0*cumulant*np.exp(-2*np.pi*gamma*self.t)

        gw = np.fft.ifft(g)
        w = np.fft.fftfreq(self.nstep,self.t[1]-self.t[0])

        gw=gw[0:int(len(gw)/2)]
        w=w[0:int(len(w)/2)]

        np.savetxt(self.auto_save_xas_no_q,np.column_stack((w,abs(gw.imag))))





    # @timeit
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



    @timeit
    def cross_section(self):

        loss=[];r=[]
        r_=[]
        r_temp=[]
        loss_temp=[]
        lenq=[]
        qx =0
        if dask:
            client = Client(processes=False, threads_per_worker=4,
                                n_workers=4, memory_limit='2GB')
        # start_time = time.time()

        for nph in tqdm(range(self.dict['nf'])):
            if nph==0:
                # print('0 phonon : 1')
                qmap=(init_map_2d(self.qx,self.qy,qx,0).phonon_fir())
                lenq.append(len(qmap))
                loss_temp.append([nph])
                r_temp.append([self.q_weights[0]*self.multi_phonon(qmap,nph)])

            elif nph==1:

                qmap=init_map_2d(self.qx,self.qy,qx,0).phonon_fir()
                lenq.append(len(qmap))
                # print('1 phonon : {lq}'.format(lq=len(qmap))) self.q_weights[qmap[0]]*

                loss_temp.append([self.q_phonon[qmap[0]]*nph])
                r_temp.append([self.multi_phonon(qmap,1)])

            elif nph==2:

                qmap,w=init_map_2d(self.qx,self.qy,qx,0).phonon_2nd()

                qmap=qmap.tolist()
                lenq.append(len(qmap))

                # # print('2 phonon : {lq}'.format(lq=len(qmap)))w[qmap.index(x)]\
                #             #    *
                #
                # norm = sum(self.q_weights)
                # print(qmap.index(qmap[0]))
                # print(qmap.index(qmap[1]))
                # print(len(qmap))
                # print(sum(w))
                # ctot=[]
                # for x in qmap:
                #     print('w : ',w[qmap.index(x)])
                #     print('qw : ',self.q_weights[x[0]]*self.q_weights[x[1]])
                #     print('cq : ',(self.q_weights[x[0]]*self.q_weights[x[1]])/norm)
                #     ctot.append((self.q_weights[x[0]]*self.q_weights[x[1]])/norm)
                # print('tot',sum(np.array(ctot)))

                coef=list(map(lambda x : (self.q_weights[x[0]]*self.q_weights[x[1]]), qmap))
                norm = sum(np.array(coef))

                print(qmap)

                print(self.q_coupling)
                print(self.q_strength)

                print(list(map(lambda x : coef[qmap.index(x)]\
                                    *self.multi_phonon(x,2)/norm, qmap)))
                print(sum(np.array(list(map(lambda x : coef[qmap.index(x)]\
                                    *self.multi_phonon(x,2)/norm, qmap)))))
                print(list(map(lambda x : self.multi_phonon(x,2), qmap)))
                print('########',sum(coef/norm))

                if dask:
                    r_.append(client.map(lambda x :(coef[qmap.index(x)])\
                                            *self.multi_phonon(x,2)/norm, qmap))
                else:
                    r_temp.append(list(map(lambda x : coef[qmap.index(x)]\
                                        *self.multi_phonon(x,2)/norm, qmap)))



                loss_temp.append(list(map(lambda x: self.q_phonon[x[0]]+self.q_phonon[x[1]], qmap)))

            elif nph==3:

                qmap,w=init_map_2d(self.qx,self.qy,qx,0).phonon_3rd()

                qmap=qmap.tolist()

                lenq.append(len(qmap))
                # print('3 phonon : {lq}'.format(lq=len(qmap)))


                coef=list(map(lambda x : (self.q_weights[x[0]]*self.q_weights[x[1]]*self.q_weights[x[2]]), qmap))
                norm = sum(np.array(coef))

                if dask:
                    r_.append(client.map(lambda x: (coef[qmap.index(x)])\
                                        *self.multi_phonon(x,nph)/norm, qmap))
                else:
                    r_temp.append(list(map(lambda x: (coef[qmap.index(x)])\
                                        *self.multi_phonon(x,nph)/norm, qmap)))


                loss_temp.append(list(map(lambda x:  self.q_phonon[x[0]]\
                                    +self.q_phonon[x[1]]+self.q_phonon[x[2]], qmap)))
                # print(len(qmap))
                # print(qmap,w)

            elif nph==4:

                qmap,w=init_map_2d(self.qx,self.qy,qx,0).phonon_4th()

                qmap=qmap.tolist()
                lenq.append(len(qmap))
                # print('4 phonon : {lq}'.format(lq=len(qmap)))

                coef=list(map(lambda x : (self.q_weights[x[0]]*self.q_weights[x[1]]\
                        *self.q_weights[x[2]]*self.q_weights[x[2]]*self.q_weights[x[3]]), qmap))
                norm = sum(np.array(coef))

                if dask:
                    r_.append(client.map(lambda x: (coef[qmap.index(x)])*self.multi_phonon(x,nph)/norm, qmap))
                else:
                    r_temp.append(list(map(lambda x: (coef[qmap.index(x)])*self.multi_phonon(x,nph)/norm, qmap)))


                loss_temp.append(list(map(lambda x:  self.q_phonon[x[0]]+self.q_phonon[x[1]]\
                                    +self.q_phonon[x[2]]+self.q_phonon[x[3]], qmap)))
                # print(len(qmap))
                # print(qmap,norm)

            elif nph==5:

                qmap,w=init_map_2d(self.qx,self.qy,qx,0).phonon_5th()

                qmap=qmap.tolist()
                lenq.append(len(qmap))
                # print('5 phonon : {lq}'.format(lq=len(qmap)))

                coef=list(map(lambda x : (self.q_weights[x[0]]*self.q_weights[x[1]]\
                        *self.q_weights[x[2]]*self.q_weights[x[2]]*self.q_weights[x[3]]\
                        *self.q_weights[x[4]]), qmap))
                norm = sum(np.array(coef))
                if dask:
                    r_.append(client.map(lambda x: (coef[qmap.index(x)])*self.multi_phonon(x,nph)/norm, qmap))
                else:
                    r_temp.append(list(map(lambda x: (coef[qmap.index(x)])*self.multi_phonon(x,nph)/norm, qmap)))


                loss_temp.append(list(map(lambda x:  self.q_phonon[x[0]]+self.q_phonon[x[1]]\
                                    +self.q_phonon[x[2]]+self.q_phonon[x[3]]+self.q_phonon[x[4]], qmap)))


        if dask:
            for i in range(self.dict['nf']-2):
                r_temp.append(client.gather(r_[i]))
            client.close()

        for j in range(self.dict['nf']):
            r.extend(r_temp[j])
            loss.extend(loss_temp[j])
            print('{nph} phonon: {lq}'.format(nph=j,lq=lenq[j]))


        # print('time : {t}'.format(t=time.time-strat_time))
        np.savetxt(self.auto_save,np.column_stack((loss,r)))
        self.x_raw = loss
        self.y_raw = r



class init_map_2d(object):
    """docstring for _init_map."""
    def __init__(self,qx,qy,qtotx,qtoty):
        super(init_map_2d, self).__init__()
        self.qx,self.qy,self.qtotx,self.qtoty=qx,qy,qtotx,qtoty
        self.qx_extend = np.hstack((self.qx,-self.qx))
        self.qy_extend = np.hstack((self.qy,-self.qy))

        self.qx_extend=np.delete(self.qx_extend,0)
        self.qy_extend=np.delete(self.qy_extend,0)

        self.size=len(self.qx_extend)

    def phonon_fir(self):
        qmap=[]
        for i in range(self.size):
            if np.round(self.qx_extend[i]-self.qtotx,3) == 0. and \
                    np.round(self.qy_extend[i]-self.qtoty,3) == 0.:
                # print([i],'x :',self.qx[i],'y :' ,self.qy[i])
                qmap.append(i)
        return qmap

    def find_index(self,qx,qy):
        temp = np.vstack((self.qx,self.qy)).transpose().tolist()
        # print(temp)
        return temp.index([qx,qy])


    def phonon_2nd(self):
        qmap=[]
        qvmap=[]
        for i in range(self.size):
            for j in range(self.size):
                qxi=self.qx_extend[i]+self.qx_extend[j]
                qyi=self.qy_extend[i]+self.qy_extend[j]
                if np.round(qxi-self.qtotx,3) == 0. and np.round(qyi-self.qtoty,3) == 0.:
                    qmap.append([i,j])
                    qvmap.append([abs(self.qx_extend[i]),abs(self.qy_extend[i]),abs(self.qx_extend[j]),abs(self.qy_extend[j])])
        qmap=np.array(qmap)
        df_temp = pd.DataFrame(qvmap,columns=['q1','q2','q3','q4'])
        df_temp=df_temp.groupby(df_temp.columns.tolist()).size().reset_index().\
            rename(columns={0:'w'})

        df_temp['index_1']=df_temp.apply(lambda x : self.find_index(x['q1'],x['q2']),axis=1)
        df_temp['index_2']=df_temp.apply(lambda x : self.find_index(x['q3'],x['q4']),axis=1)
        # print(df_temp.head())

        return df_temp[['index_1', 'index_2']].to_numpy(),df_temp['w'].to_numpy()

    def phonon_3rd(self):
        qmap=[]
        qvmap=[]
        for i in range(self.size):
            for j in range(self.size):
                for k in range(self.size):
                    qxi=self.qx_extend[i]+self.qx_extend[j]+self.qx_extend[k]
                    qyi=self.qy_extend[i]+self.qy_extend[j]+self.qy_extend[k]
                    if np.round(qxi-self.qtotx,3) == 0. and np.round(qyi-self.qtoty,3) == 0.:
                        qmap.append([i,j,k])
                        qvmap.append([abs(self.qx_extend[i]),abs(self.qy_extend[i]),\
                                    abs(self.qx_extend[j]),abs(self.qy_extend[j]),\
                                        abs(self.qx_extend[k]),abs(self.qy_extend[k])])
        qmap=np.array(qmap)
        df_temp = pd.DataFrame(qvmap,columns=['qx1','qy1','qx2','qy2','qx3','qy3'])
        df_temp=df_temp.groupby(df_temp.columns.tolist()).size().reset_index().\
            rename(columns={0:'w'})

        df_temp['index_1']=df_temp.apply(lambda x : self.find_index(x['qx1'],x['qy1']),axis=1)
        df_temp['index_2']=df_temp.apply(lambda x : self.find_index(x['qx2'],x['qy2']),axis=1)
        df_temp['index_3']=df_temp.apply(lambda x : self.find_index(x['qx3'],x['qy3']),axis=1)
        # print(df_temp.head())

        return df_temp[['index_1', 'index_2','index_3']].to_numpy(),df_temp['w'].to_numpy()

    def phonon_4th(self):
        qmap=[]
        qvmap=[]
        for i in range(self.size):
            for j in range(self.size):
                for k in range(self.size):
                    for l in range(self.size):
                        qxi=self.qx_extend[i]+self.qx_extend[j]+self.qx_extend[k]+self.qx_extend[l]
                        qyi=self.qy_extend[i]+self.qy_extend[j]+self.qy_extend[k]+self.qy_extend[l]
                        if np.round(qxi-self.qtotx,3) == 0. and np.round(qyi-self.qtoty,3) == 0.:
                            qmap.append([i,j,k,l])
                            qvmap.append([abs(self.qx_extend[i]),abs(self.qy_extend[i]),\
                                    abs(self.qx_extend[j]),abs(self.qy_extend[j]),\
                                        abs(self.qx_extend[k]),abs(self.qy_extend[k]),\
                                        abs(self.qx_extend[l]),abs(self.qy_extend[l])])
        qmap=np.array(qmap)
        df_temp = pd.DataFrame(qvmap,columns=['qx1','qy1','qx2','qy2','qx3','qy3','qx4','qy4'])
        df_temp=df_temp.groupby(df_temp.columns.tolist()).size().reset_index().\
            rename(columns={0:'w'})

        df_temp['index_1']=df_temp.apply(lambda x : self.find_index(x['qx1'],x['qy1']),axis=1)
        df_temp['index_2']=df_temp.apply(lambda x : self.find_index(x['qx2'],x['qy2']),axis=1)
        df_temp['index_3']=df_temp.apply(lambda x : self.find_index(x['qx3'],x['qy3']),axis=1)
        df_temp['index_4']=df_temp.apply(lambda x : self.find_index(x['qx4'],x['qy4']),axis=1)
        # print(df_temp.head())

        return df_temp[['index_1', 'index_2','index_3','index_4']].to_numpy(),df_temp['w'].to_numpy()


    def phonon_5th(self):
        qmap=[]
        qvmap=[]
        for i in range(self.size):
            for j in range(self.size):
                for k in range(self.size):
                    for l in range(self.size):
                        for m in range(self.size):
                            qxi=self.qx_extend[i]+self.qx_extend[j]+self.qx_extend[k]+self.qx_extend[l]\
                            +self.qx_extend[m]
                            qyi=self.qy_extend[i]+self.qy_extend[j]+self.qy_extend[k]+self.qy_extend[l]\
                            +self.qy_extend[m]
                            if np.round(qxi-self.qtotx,3) == 0. and np.round(qyi-self.qtoty,3) == 0.:
                                qmap.append([i,j,k,l,m])
                                qvmap.append([abs(self.qx_extend[i]),abs(self.qy_extend[i]),\
                                    abs(self.qx_extend[j]),abs(self.qy_extend[j]),\
                                        abs(self.qx_extend[k]),abs(self.qy_extend[k]),\
                                        abs(self.qx_extend[l]),abs(self.qy_extend[l]),\
                                        abs(self.qx_extend[m]),abs(self.qy_extend[m])])
        qmap=np.array(qmap)
        df_temp = pd.DataFrame(qvmap,columns=['qx1','qy1','qx2','qy2','qx3','qy3','qx4','qy4','qx5','qy5'])
        df_temp=df_temp.groupby(df_temp.columns.tolist()).size().reset_index().\
            rename(columns={0:'w'})

        df_temp['index_1']=df_temp.apply(lambda x : self.find_index(x['qx1'],x['qy1']),axis=1)
        df_temp['index_2']=df_temp.apply(lambda x : self.find_index(x['qx2'],x['qy2']),axis=1)
        df_temp['index_3']=df_temp.apply(lambda x : self.find_index(x['qx3'],x['qy3']),axis=1)
        df_temp['index_4']=df_temp.apply(lambda x : self.find_index(x['qx4'],x['qy4']),axis=1)
        df_temp['index_5']=df_temp.apply(lambda x : self.find_index(x['qx5'],x['qy5']),axis=1)
        # print(df_temp.head())

        return df_temp[['index_1', 'index_2','index_3','index_4','index_5']].to_numpy(),df_temp['w'].to_numpy()
