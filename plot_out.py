import numpy as np
from matplotlib import pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import RegularPolygon as rp
from scipy.interpolate import griddata
import matplotlib.ticker as ticker
import math


class plot(object):
    """docstring for plot."""

    def __init__(self,dname):
        super(plot, self).__init__()
        self.q_path={}
        self.coupling_qpath={}
        self.path='../../../storage/'+dname

    def get_coupling(self,path=''):
        self.sym_directions=['gamma-k','k-m','gamma-m']
        transform={\
        'gamma-k':{'x':1,'y':0},\
        'k-m':{'x':np.cos(np.pi/3.),'y':np.sin(np.pi/3.)},\
        'gamma-m':{'x':np.sin(np.pi/3.),'y':np.cos(np.pi/3.)}}
        self.q_bz=[4./3.,2./3.,2./np.sqrt(3.)]
    	for i,names in enumerate(self.sym_directions):
    			# print(i,names)
    		self.q_path[names],self.coupling_qpath[names]\
    					=np.loadtxt(self.path+name+'_coupling_'+names+'.csv')
    		self.q_path[names]=self.q_path[names] * self.q_bz[i]

    def plot_dispersion(self):
        self.get_coupling()

        ARPES_coupling_gamma=0.15
        ARPES_coupling_k=0.2

        LBNL_coupling_gamma=0.42
        LBNL_coupling_k=0.21

        BSE_coupling_gamma=0.75
        BSE_coupling_k=0.475

        fig = plt.figure(figsize=(10,5))
        ax = fig.add_subplot(131)

        plt.scatter(0,ARPES_coupling_gamma,marker='>',s=50,color='r')
        plt.scatter(max(self.q_path['gamma-k']*self.q_bz[0]),ARPES_coupling_k,marker='>',s=50,color='r',label='ARPES')

        plt.scatter(0,LBNL_coupling_gamma,marker='*',s=50,color='b')
        plt.scatter(max(self.q_path['gamma-k']*self.q_bz[0]),LBNL_coupling_k,marker='*',s=50,color='b',label='LBNL')

        plt.scatter(0,BSE_coupling_gamma,marker='s',s=50,color='g')
        plt.scatter(max(self.q_path['gamma-k']*self.q_bz[0]),BSE_coupling_k,marker='s',s=50,color='g',label='BSE')

        ax.set_ylabel(r'Coupling Constant, eV',fontsize=15)
        plt.legend()
        ax.plot(self.q_path['gamma-k']*self.q_bz[0],self.coupling_qpath['gamma-k'],'-',color='grey')
        plt.xticks([min(self.q_path['gamma-k']*self.q_bz[0]),max(self.q_path['gamma-k']*self.q_bz[0])],(r'$\Gamma$',r'$K$'),fontsize=20)
        # ax.axvline(0.08,color='b')
        # ax.axvline(0.011,color='r')
        ax.set_ylim([0,BSE_coupling_gamma*1.2])
        # ax.set_ylim([0.14,max(self.coupling_qpath['gamma-k'])*1.2])
        ax = fig.add_subplot(132)
        ax.plot(self.q_path['k-m']*self.q_bz[1],self.coupling_qpath['k-m'],'-',color='grey',label='model')
        plt.xticks([min(self.q_path['k-m']*self.q_bz[1]),max(self.q_path['k-m']*self.q_bz[1])],(r'$K$',r'$M$'),fontsize=20)
        ax.set_yticks([])
        ax.set_ylim([0,BSE_coupling_gamma*1.2])
        # ax.set_ylim([0,max(self.coupling_qpath['gamma-k'])*1.2])
        # ax.set_ylim([0.14,max(self.coupling_qpath['gamma-k'])*1.2])
        plt.legend()
        ax.set_xlabel(r'$q \ path$',fontsize=15)
        ax = fig.add_subplot(133)
        ax.set_yticks([])
        ax.set_ylim([0,BSE_coupling_gamma*1.2])
        # ax.set_ylim([0,max(self.coupling_qpath['gamma-k'])*1.2])
        ax.plot(self.q_path['gamma-m']*self.q_bz[2],self.coupling_qpath['gamma-m'],'-',color='grey',label='exp')

        plt.xticks([min(self.q_path['gamma-m']*self.q_bz[2]),max(self.q_path['gamma-m']*self.q_bz[2])],(r'$\Gamma$',r'$M$'),fontsize=20)

        plt.gca().invert_xaxis()
        plt.tight_layout()
        plt.show(block=False)
        
ws=plot('model')
ws.plot_dispersion()
