import json
import numpy as np
from scipy.special import wofz

class spec(object):
    """docstring for spectra."""
    def __init__(self,dict,nruns=1,nmodel=1,out_dir='./_output/',
                npoints=1000,
                spec_max = -0.1,
                spec_min = 1.):
        super(spec, self).__init__()
        self.nruns=str(nruns)
        self.out_dir=out_dir
        self.npoints=npoints
        self.xmax=spec_max
        self.xmin=spec_min
        self.spec_raw = out_dir+'/{nr}_rixs_raw.csv'.format(nm=nmodel,
                                                                        nr=nruns)

        self.x,self.y=np.transpose(np.loadtxt(self.spec_raw))

        # if int(dict['vib_space'])==1:
        #     # self.xmin=-dict['omega_ph0']
        #     # self.xmax=dict['omega_ph0']*dict['nf']
        #     self.xmax=spec_max
        #     self.xmin=spec_min
        #
        # elif int(dict['vib_space'])==2:
        #     max_=max(dict['omega_ph0'],dict['omega_ph1'])
        #     self.xmin=-max_
        #     self.xmax=max_*dict['nf']
        # else:
        #     print('something went wrong')
        #     print(dict['vib_space'])

        self.gamma_ph=dict['gamma_ph']
        self.alpha_exp=dict['alpha_exp']
        self.save_noel=\
            self.out_dir+'/{nr}_rixs_phonons.csv'.format(nm=nmodel,
                                                                    nr=nruns)
        self.save_full=\
            self.out_dir+'/{nr}_rixs_full.csv'.format(nm=nmodel,
                                                                nr=nruns)

    def run_broad(self):
        x=np.linspace(self.xmin,self.xmax,(self.npoints))
        full, noelastic=np.zeros_like(x),np.zeros_like(x)
        for en,inten in zip(self.x, self.y):
            shape=self.voigt(x-en,self.alpha_exp,self.gamma_ph)
            norm=np.sum(shape)*abs(x[0]-x[1])
            y=(shape)*inten/norm
            if en!=0: noelastic=noelastic+y
            full=full+y
        np.savetxt(self.save_full,np.column_stack([x,full]))
        np.savetxt(self.save_noel,np.column_stack([x,noelastic]))

    def run_broad_fit(self,x=[]):
        full, noelastic=np.zeros_like(x),np.zeros_like(x)
        for en,int in zip(self.x, self.y):
            shape=self.voigt(x-en,self.alpha_exp,self.gamma_ph)
            norm=np.sum(shape)*abs(x[0]-x[1])
            y=(shape)*int/norm
            if en!=0: noelastic=noelastic+y
            full=full+y
        return x,noelastic,full

    def voigt(self,x, alpha, gamma):
	    sigma = alpha / np.sqrt(2 * np.log(2))
	    return np.real(wofz((x + 1j*gamma)/sigma/np.sqrt(2))) / sigma\
	                                                           /np.sqrt(2*np.pi)
