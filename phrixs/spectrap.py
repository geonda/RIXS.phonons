import json
import numpy as np
from scipy.special import wofz
import config as cg
class spectra(object):
    """docstring for spectra."""
    def __init__(self,nruns=1):
        super(spectra, self).__init__()
        self.nruns=str(nruns)
        self.auto_save=cg.temp_rixs_file+\
                        '_run_'+self.nruns+cg.extension_final
        self.dict_input=cg.dict_input_file
        with open(self.dict_input) as fp:
            dict=json.load(fp)
        self.x,self.y=np.load(self.auto_save)
        self.xmin=-dict['omega_ph']
        self.xmax=dict['omega_ph']*dict['nf']
        self.nspectra=1000
        self.gamma_ph=dict['gamma_ph']
        self.alpha_exp=dict['alpha_exp']
        self.save_noel=cg.temp_rixs_noel_file+'_run_'+self.nruns+cg.extension_final
        self.save_full=cg.temp_rixs_full_file+'_run_'+self.nruns+cg.extension_final

    def run_broad(self):
        x=np.linspace(self.xmin,self.xmax,self.nspectra)
        full, noelastic=np.zeros_like(x),np.zeros_like(x)
        for en,int in zip(self.x, self.y):
            shape=self.voigt(x-en,self.alpha_exp,self.gamma_ph)
            norm=np.sum(shape)*abs(x[0]-x[1])
            y=(shape)*int/norm
            if en!=0: noelastic=noelastic+y
            full=full+y
        np.save(self.save_full,np.vstack((x,full)))
        np.save(self.save_noel,np.vstack((x,noelastic)))

    def voigt(self,x, alpha, gamma):
	    sigma = alpha / np.sqrt(2 * np.log(2))
	    return np.real(wofz((x + 1j*gamma)/sigma/np.sqrt(2))) / sigma\
	                                                           /np.sqrt(2*np.pi)
