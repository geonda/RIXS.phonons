import numpy as np
import json
import os


class experiment(object):
    """

    Experiment.

    Args:
        file: str
            path to the file with the exp data.
        col: list
            [column x ; column y] defines which columns to read from the file.
        name: str
            name of the experiment.
        nexp: int
            id number of the given experiment in the given project.


    Attributes:
        x: float
            energy loss readings from exp file.
        y: float
            rixs intensity readings from  exp file.
        max: float
            max value of y.
        y_norm: float
            normalized y.
        name: str
            name of the experiment.
        xmin: float
            min value of x.
        xmax: float
            max value of x.

    """

    def __init__(self, expfile='', columns=[0,1], nexp = 1, name = ''):
        super(experiment, self).__init__()
        self.expfile = expfile
        data = np.transpose(np.loadtxt(self.expfile))
        self.x = data[columns[0]]
        self.y = data[columns[1]]
        self.max = max(self.y)
        self.y_norm = self.y/self.max
        self.name = name
        self.xmin=min(self.x)
        self.xmax=max(self.x)

    def filter(self,xmin,xmax):
        xnew,ynew = [],[]
        for xi,yi in zip(self.x,self.y):
            if (xi>=xmin) and (xi<=xmax):
                xnew.append(xi)
                ynew.append(yi)
        return np.array(xnew),np.array(ynew)
