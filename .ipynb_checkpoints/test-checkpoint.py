
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
import dask
from dask import delayed
import dask.dataframe as dd
import dask.array as dp
from dask.distributed import Client, progress


# client = Client(processes=False, threads_per_worker=4,
#                  n_workers=1, memory_limit='2GB')
dask.config.set(scheduler='threads')
dict = { 'problem_type': 'rixs',
        'method': 'gf',
        'maxt' : 200,
        'nstep': 1000,
         "nf": 10.0,
         "energy_ex": 10.0,
         "omega_in": 10.0,
         "gamma": 0.105,
         "gamma_ph": 0.05,
         "alpha_exp": 0.01 }

t=(np.linspace(0.,dict['maxt'],int(dict['nstep'])))


# read dataframe
q_points = np.loadtxt('q_sampling.csv')
q_weights = np.loadtxt('q_weights.csv')
q_coupling = np.loadtxt('q_coupling.csv')
q_phonon = np.loadtxt('q_phonon.csv')
q_strength = (q_coupling/q_phonon)**2

#
q_points = dp.asarray(q_points)
q_weights = dp.asarray(q_weights)
q_coupling = dp.asarray(q_coupling)
q_phonon = dp.asarray(q_phonon)
q_strength = dp.asarray(q_strength)
t = dp.asarray(t)


def fkq():
    Fkq=1.
    for gkqi,omegaqi,wi in zip(q_strength,\
                        q_phonon*2*np.pi,q_weights):
        cumulant = \
            gkqi*(np.exp(-1.j*omegaqi*t)+1.j*omegaqi*t-1)*wi/len(q_points)
        Fkq=Fkq*np.exp(cumulant)
    return Fkq

print('Eph', q_phonon )
cumulant_kq = fkq()




# @dask.delayed


# @dask.delayed
def multi_phonon(qmap,nph):
    G=-1.j*np.exp(-1.j*(np.pi*2.*dict['energy_ex'])*t)
    D=1.
    print('here')
    for n in range(nph):
        Dk=(np.sqrt(q_strength[qmap[n]]))*(np.exp(-1.j*q_phonon[qmap[n]]*2.*np.pi*t)-1.)
        D=D*Dk

    G=G*(D)/np.sqrt(factorial(nph))
    G=G*cumulant_kq*np.exp(-2*np.pi*dict['gamma']*t)
    # das
    # G = dp.asarray(G)

    intx=goertzel(G,dict['nstep']/dict['maxt'],dict['omega_in'])

    return abs(intx[0])**2

#@dask.delayed
def goertzel(samples, sample_rate, freqs):
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
    return  (power)



loss=[];r=[]


r = multi_phonon([0],1)
# r.vizualize()

# for nph in tqdm(range(int(dict['nf']))):
#     if nph==0:
#         print('0 phonon : ')
#         qmap=[0]
#         loss_temp,r_temp=[nph],[multi_phonon(qmap,nph)]
#     elif nph==1:
#         print('1 phonon : ')
#         qmap=[0]
#         loss_temp,r_temp=[q_phonon[qmap[0]]*nph],[q_weights[0]*multi_phonon(qmap,1)]
#     loss.extend(loss_temp)
#     r.extend(r_temp)
print('############## ouput ####')
print(r.compute())
client.close()
