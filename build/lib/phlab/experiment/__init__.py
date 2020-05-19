import numpy as np
import json
import os


class experiment(object):
    """docstring for experiment."""

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
