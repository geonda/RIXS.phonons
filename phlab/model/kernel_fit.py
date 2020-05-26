from phlab.model import input_handler
from phlab.model import spec
from lmfit import Minimizer, Parameters, report_fit,Model
import numpy as np
import json
import os

class execute(object):
    
    """docstring for converge."""

    def __init__(self, kernel, local):
        super(execute, self).__init__()

        self.local = local
        self.kernel = kernel
        self.experiment = self.local.param['experiment']
        self.verbose = self.local.param['verbose']
        self.method = self.local.param['method']
        xexp,yexp=self.experiment.filter(self.experiment.xmin,self.experiment.xmax)

        def fcn2min(params,xexp,yexp):
            self.param2input(params)
            if self.verbose :
                print(params)
            self.local.input_class.input_update(self.local.input)
            self.kernel.kernel(self.local.input,
                                    nruns = self.local.nruns,
                                    nmodel = self.local.nmodel,
                                    out_dir = self.local.out_dir).cross_section()
            x,yfit,_=spec.spec(self.local.input,
                                nruns = self.local.nruns,
                                nmodel = self.local.nmodel,
                                out_dir = self.local.out_dir).run_broad_fit(x=xexp)
            resid = abs(yfit/max(yfit)-yexp/max(yexp))
            return resid


        params = Parameters()
        for item in self.local.param2fit.dict:
            params.add(item,
                    self.local.param2fit.dict[item]['value'],
                    min = self.local.param2fit.dict[item]['range'][0],
                    max= self.local.param2fit.dict[item]['range'][1], vary=True)

        minner = Minimizer(fcn2min, params,fcn_args=(xexp,yexp))
        result = minner.minimize(method=self.method)
        self.fit_report=report_fit(result)
        self.local.input_class.input_update(self.local.input)
    def param2input(self,param):
        for item in param:
            self.local.input[item]=param[item].value
