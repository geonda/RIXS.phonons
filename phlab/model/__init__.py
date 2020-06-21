from phlab.model import spec
from phlab.model import kernel_single_osc
from phlab.model import kernel_double_osc
from phlab.model import kernel_dist_disp_osc
from phlab.model import kernel_gq_phonons
from phlab.model.input_handler import input_handler
from phlab.model import kernel_converge
from phlab.model import kernel_fit

from lmfit import Minimizer, Parameters, report_fit,Model
import numpy as np
import json
import os



class single_osc(object):
    """

    Creates object for 1D harmonic oscillator model.

    Args:
        inp_dir: str
            name of the input directory.
        out_dir: str
            name of the output directory.
        nmodel: int
            id number of the model.
        name: str
            name of the model.


    Attributes:
        input_default: dict
            dictionary with default input parameters.
        input: dict
            dictionary with current input parameters.
        npoints: int
            number of points in the spectrum.
        spec_max: float
            max limt of enrgy loss.
        spec_min: float
            min limt of enrgy loss.
        param2fit: object
            parameters to fit.
        nruns: int
            number of runs.
        color: str
            color of the line
        input_class: object
            returns input_handler for this model.
        x: float
            energy loss in eV for the phonon contribution.
        y: float
            rixs intensity (arb. units) for the phonon contribution.
        y_norm: float
            normalized rixs intensity (arb. units) for the phonon contribution.

    """

    def __init__(self,
                    inp_dir = './_input/',
                    out_dir = './_output/',
                    nmodel = 0,
                    name = ''
                    ):
        super(single_osc, self).__init__()

        self.input_default = {
                'problem_type': 'rixs',
                'method': 'fc',
                 "coupling": 0.1,
                 "omega_ph": 0.195,
                 "nf": 10.0,
                 "nm": 100.0,
                 "energy_ex": 5.0,
                 "omega_in": 5.0,
                 "gamma": 0.105,
                 "gamma_ph": 0.05,
                 "alpha_exp": 0.01
                }

        self.npoints = 1000
        self.spec_max = 1.
        self.spec_min= -0.1
        self.inp_dir = inp_dir
        self.out_dir = out_dir
        self.nmodel = nmodel
        self.name = name
        self.param2fit = parameters2fit()
        self.nruns=0
        self.color=''
        self.input_class=input_handler(input_default = self.input_default,
                inp_dir = self.inp_dir,
                nmodel = self.nmodel)
        self.input = self.input_class.input
        print('number of models : {nm}'.format(nm = self.nmodel))

    def run(self):
        self.input_class.input_update(self.input)
        self.nruns+=1
        results=execute(kernel = kernel_single_osc,
                        local = self)

        self.x,self.y=np.transpose(np.loadtxt(self.out_dir\
                                +'/{nr}_rixs_phonons.csv'.format(\
                                                nm = self.nmodel, nr = self.nruns)))
        self.y_norm=self.y/max(self.y)


    def converge(self,parameter = 'nm', pmin =0 , pmax= 1,steps = 100):
        self.input_class.input_update(self.input)
        self.param = {
                    'parameter' : parameter,
                    'pmin': pmin,
                    'pmax': pmax,
                    'steps' : steps}
        res = kernel_converge.execute(kernel = kernel_single_osc, local = self )
        print(res.kernel)

        self.conv_arr = res.conv_arr
        self.param_space = res.param_space


    def fit(self, experiment={}, method = 'brute' ,verbose = True):

        self.param = {
                    'experiment' : experiment,
                    'method': method,
                    'verbose': verbose}

        results = kernel_fit.execute(kernel = kernel_single_osc, local =self)
        self.fit_report=results.fit_report



class gq_phonons_2d(object):

    """

    Creates object for RIXS model
    describing interaction of q dependent phonons and a single elctornic  level.

    Args:
        inp_dir: str
            name of the input directory.
        out_dir: str
            name of the output directory.
        nmodel: int
            id number of the model.
        name: str
            name of the model.


    Attributes:
        input_default: dict
            dictionary with default input parameters.
        input: dict
            dictionary with current input parameters.
        npoints: int
            number of points in the spectrum.
        spec_max: float
            max limt of enrgy loss.
        spec_min: float
            min limt of enrgy loss.
        param2fit: object
            parameters to fit.
        nruns: int
            number of runs.
        color: str
            color of the line
        input_class: object
            returns input_handler for this model.
        x: float
            energy loss in eV for the phonon contribution.
        y: float
            rixs intensity (arb. units) for the phonon contribution.
        y_norm: float
            normalized rixs intensity (arb. units) for the phonon contribution.

    """

    def __init__(self,
                    inp_dir = './_input/',
                    out_dir = './_output/',
                    nmodel = 0,
                    name = ''
                    ):
        super(gq_phonons_2d, self).__init__()

        self.input_default = {
                'problem_type': 'rixs',
                'method': 'gf',
                'maxt' : 200,
                'nstep': 1000,
                 "nf": 10.0,
                 "nq": 3,
                 "m_gamma": 0.1,
                 "m_k": 0.1,
                 "r": 0.1,
                 "energy_ex": 5.0,
                 "omega_in": 5.0,
                 "gamma": 0.105,
                 "gamma_ph": 0.05,
                 "alpha_exp": 0.01
                }

        self.npoints = 1000
        self.spec_max = 1.
        self.spec_min= -0.1
        self.inp_dir = inp_dir
        self.out_dir = out_dir
        self.nmodel = nmodel
        self.name = name
        self.nruns=0
        self.color=''

        self.param2fit = parameters2fit()
        self.input_class=input_handler(input_default = self.input_default,
                inp_dir = self.inp_dir,
                nmodel = self.nmodel)
        self.input = self.input_class.input

        print('number of models : {nm}'.format(nm = self.nmodel))



    def run(self):

        self.input_class.input_update(self.input)
        self.nruns+=1
        results=execute(kernel = kernel_gq_phonons,
                        local = self)

        self.x,self.y=np.transpose(np.loadtxt(self.out_dir\
                                +'/{nr}_rixs_phonons.csv'.format(\
                                                nm = self.nmodel, nr = self.nruns)))
        self.y_norm=self.y/max(self.y)

        self.xas_freq,self.xas_int = np.transpose(np.loadtxt(self.out_dir\
                                +'/{nr}_xas.csv'.format(\
                                                nm = self.nmodel, nr = self.nruns)))
        self.xas_freq_no_q,self.xas_int_no_q = np.transpose(np.loadtxt(self.out_dir\
                                +'/{nr}_xas_no_q.csv'.format(\
                                                nm = self.nmodel, nr = self.nruns)))


    def converge(self,parameter = 'nm', pmin =0 , pmax= 1,steps = 100):

        self.input_class.input_update(self.input)
        self.param = {
                    'parameter' : parameter,
                    'pmin': pmin,
                    'pmax': pmax,
                    'steps' : steps}
        res = kernel_converge.execute(kernel = kernel_gq_phonons, local = self )
        print(res.kernel)

        self.conv_arr = res.conv_arr
        self.param_space = res.param_space


    def fit(self, experiment={}, method = 'brute' ,verbose = True):

        self.param = {
                    'experiment' : experiment,
                    'method': method,
                    'verbose': verbose}

        results = kernel_fit.execute(kernel = kernel_gq_phonons, local =self)
        self.fit_report=results.fit_report



class double_osc(object):
    """

    Creates object for 2D harmonic oscillator model.

    Args:
        inp_dir: str
            name of the input directory.
        out_dir: str
            name of the output directory.
        nmodel: int
            id number of the model.
        name: str
            name of the model.


    Attributes:
        input_default: dict
            dictionary with default input parameters.
        input: dict
            dictionary with current input parameters.
        npoints: int
            number of points in the spectrum.
        spec_max: float
            max limt of enrgy loss.
        spec_min: float
            min limt of enrgy loss.
        param2fit: object
            parameters to fit.
        nruns: int
            number of runs.
        color: str
            color of the line.
        input_class: object
            returns input_handler for this model.
        x: float
            energy loss in eV for the phonon contribution.
        y: float
            rixs intensities (arb. units) for the phonon contribution.
        y_norm: float
            normalized rixs intensities (arb. units) for the phonon contribution.

    """

    def __init__(self,
                    inp_dir = './_input/',
                    out_dir = './_output/',
                    nmodel = 0,
                    name = ''
                    ):
        super(double_osc, self).__init__()

        self.input_default = {
                'problem_type': 'rixs',
                'method': 'fc',
                 "coupling0": 0.1,
                 "omega_ph0": 0.1,
                 "coupling1": 0.03,
                 "omega_ph1": 0.03,
                 "nf": 10.0,
                 "nm": 3.0,
                 "energy_ex": 10.0,
                 "omega_in": 10.0,
                 "gamma": 0.1,
                 "gamma_ph": 0.01,
                 "alpha_exp": 0.01,
                }

        self.npoints = 1000
        self.spec_max = 1.
        self.spec_min= -0.1
        self.inp_dir = inp_dir
        self.out_dir = out_dir
        self.nmodel = nmodel
        self.nruns=0
        self.color='r'
        self.name = name
        self.param2fit = parameters2fit()
        self.input_class=input_handler(input_default = self.input_default,
                inp_dir = self.inp_dir,
                nmodel = self.nmodel,
                model_name = '2d_osc')

        self.input = self.input_class.input

        print('number of models : {nm}'.format(nm = self.nmodel))

    def run(self):
        self.input_class.input_update(self.input)
        self.nruns+=1
        results=execute(kernel = kernel_double_osc,
                        local = self)

        self.x,self.y=np.transpose(np.loadtxt(self.out_dir\
                                +'/{nr}_rixs_phonons.csv'.format(\
                                                nm = self.nmodel, nr = self.nruns)))
        self.y_norm=self.y/max(self.y)

    def converge(self,parameter = 'nm', pmin =0 , pmax= 1,steps = 100):
        self.input_class.input_update(self.input)
        self.param = {
                    'parameter' : parameter,
                    'pmin': pmin,
                    'pmax': pmax,
                    'steps' : steps}
        res = kernel_converge.execute(kernel = kernel_double_osc, local = self )
        print(res.kernel)

        self.conv_arr = res.conv_arr
        self.param_space = res.param_space


    def fit(self, experiment={}, method = 'brute' ,verbose = True):

        self.param = {
                    'experiment' : experiment,
                    'method': method,
                    'verbose': verbose}

        results = kernel_fit.execute(kernel = kernel_double_osc, local =self)
        self.fit_report=results.fit_report


class dist_disp_osc(object):
    """

    Creates object for distorted and displaced harmonic oscillator model.

    Args:
        inp_dir: str
            name of the input directory.
        out_dir: str
            name of the output directory.
        nmodel: int
            serial number of the model.
        name: str
            name of the model.

    Attributes:
        input_default: dict
            dictionary with default input parameters.
        input: dict
            dictionary with current input parameters.
        npoints: int
            number of points in the spectrum.
        spec_max: float
            max limt of enrgy loss.
        spec_min: float
            min limt of enrgy loss.
        param2fit: object
            parameters to fit.
        nruns: int
            number of runs.
        color: str
            color of the line.
        input_class: object
            returns input_handler for this model.
        x: float
            energy loss in eV for the phonon contribution.
        y: float
            rixs intensities (arb. units) for the phonon contribution.
        y_norm: float
            normalized rixs intensities (arb. units) for the phonon contribution.

    """

    def __init__(self,
                    inp_dir = './_input/',
                    out_dir = './_output/',
                    nmodel = 0,
                    name = ''
                    ):
        super(dist_disp_osc, self).__init__()

        self.input_default = {
                'problem_type': 'rixs',
                'method': 'fc',
                 "coupling": 0.1,
                 "omega_ph": 0.195,
                 'omega_ph_ex':0.1,
                 "nf": 10.0,
                 "nm": 3.0,
                 "energy_ex": 10.0,
                 "omega_in": 10.0,
                 "gamma": 0.1,
                 "gamma_ph": 0.05,
                 "alpha_exp": 0.01,
                }
        self.npoints = 1000
        self.spec_max = 1.
        self.spec_min= -0.1
        self.inp_dir = inp_dir
        self.out_dir = out_dir
        self.nmodel = nmodel
        self.name = name
        self.nruns=0
        self.color='r'
        self.param2fit = parameters2fit()
        self.input_class=input_handler(input_default = self.input_default,
                inp_dir = self.inp_dir,
                nmodel = self.nmodel,
                model_name = 'dd_osc')
        self.input = self.input_class.input
        print('number of models : {nm}'.format(nm = self.nmodel))

    def run(self):
        self.input_class.input_update(self.input)
        self.nruns+=1
        results=execute(kernel = kernel_dist_disp_osc,
                        local = self)

        self.x,self.y=np.transpose(np.loadtxt(self.out_dir\
                                +'/{nr}_rixs_phonons.csv'.format(\
                                                nm = self.nmodel, nr = self.nruns)))
        self.y_norm=self.y/max(self.y)

    def converge(self,parameter = 'nm', pmin =0 , pmax= 1, steps = 100):
        self.input_class.input_update(self.input)
        self.param = {
                    'parameter' : parameter,
                    'pmin': pmin,
                    'pmax': pmax,
                    'steps' : steps}
        res = kernel_converge.execute(kernel = kernel_dist_disp_osc, local = self )
        self.conv_arr = res.conv_arr
        self.param_space = res.param_space


    def fit(self, experiment={}, method = 'brute' ,verbose = True):

        self.param = {
                    'experiment' : experiment,
                    'method': method,
                    'verbose': verbose}

        results = kernel_fit.execute(kernel = kernel_dist_disp_osc, local =self)
        self.fit_report=results.fit_report

# support classes

class parameters2fit(object):
    """

    Defines paramters to fit.

    Attributes:
        dict: dict
            dictionary with parameters to fit

    """

    def __init__(self):
        super(parameters2fit, self).__init__()
        self.dict = {}

    def add(self, name = '', ivalue = 0, range=[0,1]):
        self.dict[name]={'value':ivalue,'range':range}


class execute(object):
    """docstring for execute."""

    def __init__(self, kernel = '', local = ''):
        super(execute, self).__init__()
        self.local = local
        self.kernel = kernel
        self.kernel.kernel(self.local.input,
                                nruns = self.local.nruns,
                                nmodel = self.local.nmodel,
                                out_dir = self.local.out_dir).cross_section()

        spec.spec(self.local.input,
                nruns = self.local.nruns,
                nmodel = self.local.nmodel,
                out_dir = self.local.out_dir,
                npoints = self.local.npoints,
                spec_max = self.local.spec_max,
                spec_min= self.local.spec_min).run_broad()
