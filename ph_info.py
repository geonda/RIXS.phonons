import numpy as np
from matplotlib import pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import RegularPolygon as rp
from scipy.interpolate import griddata
from scipy import interpolate
from scipy.interpolate import Rbf
import matplotlib.ticker as ticker
from matplotlib import cm
import math
import pandas as pd

class pre_process(object):

	""" """

	def __init__(self, nq= 3,m_gamma=0.1,m_k=0.1,r=0.1):
		super(pre_process, self).__init__()

		self.path_to_file = './temp_exp/TO/'

		self.r,self.m_gamma,self.m_k=r,m_gamma,m_k
		self.df = pd.DataFrame()
		self.convert_qpoints(nq = nq)

		en = self.get_phonon()

		self.df['ph'] =0.195*np.ones(nq)

		self.df['mq'] = self.df.apply(lambda x : self.function_coupling(x['qx'],x['qy']), axis=1)
		self.df['gq'] = self.df.apply(lambda x : (x['mq']/x['ph'])**2, axis=1)
		print(self.df.head(10))
		self.df=self.df[self.df['mq']!=0]
		self.df.to_csv('df_ph_info.csv')
		# cmap = cm.get_cmap('Spectral') # Colour map (there are many others)
		# fig, ax = plt.subplots(1)
		# ax.scatter(self.df['qx'], self.df['qy'], c=self.df['mq'], s=120, cmap = 'plasma', edgecolor='None')
		# fig, ax = plt.subplots(1)
		# ax.scatter(self.df['qx'], self.df['qy'], c=self.df['ph'], s=120, cmap = 'plasma', edgecolor='None')
		# # plt.colorbar()
		# # cbar.set_label('Color Intensity')
		# plt.show()


	def get_phonon(self):
		self.q_path={}
		self.phonon_energy={}
		self.e2d=[]
		self.q2d={'x':[],'y':[]}

		sym_directions=['gamma-k','k-m','gamma-m']
		# transform={\
		# 'gamma-k':{'x':1,'y':0},\
		# 'k-m':{'x' : np.cos(np.pi/3.),'y' : np.sin(np.pi/3.)},\
		# 'gamma-m':{'x' : np.sin(np.pi/3.),'y' : np.cos(np.pi/3.)}}
		# self.q_bz=[4./3.,2./3.,2./np.sqrt(3.)]
		transform={\
		'gamma-k':{'x' : 0.666667,'y' : 1.154701},\
		'k-m':{'x' : np.cos(np.pi/3.),'y' : np.sin(np.pi/3.)},\
		'gamma-m':{'x':0,'y':0.769800}}
		self.q_bz=[4./3.,2./3.,2./np.sqrt(3.)]
		for i,names in enumerate(sym_directions):
			# print(i,names)
			self.q_path[names],self.phonon_energy[names]=\
						np.loadtxt(self.path_to_file+names+'.csv').transpose()
			self.q_path[names]=self.q_path[names] * self.q_bz[i]
			if names=='k-m':
				self.q2d['x'].extend(self.q_bz[0]-self.q_path[names][:] * transform[names]['x'])
			else:
				self.q2d['x'].extend(self.q_path[names][:] * transform[names]['x'])
			self.q2d['y'].extend(self.q_path[names][:] *transform[names]['y'])
			self.e2d.extend(self.phonon_energy[names])
		#print(self.e2d)
		# print(self.q2d['x'])
		# print(self.q2d['y'])
		# print()

		# rbi = np.array(self.q2d['x']), np.array(self.q2d['y']), np.array(self.e2d)
		# print(np.array(self.e2d))

		rbfi = Rbf( np.array(self.q2d['x']), np.array(self.q2d['y']), np.array(self.e2d))
		qx= self.df['qx']
		qy = self.df['qy']
		en = rbfi(qx, qy)
		#
		# print(en)
		return en


	def convert_qpoints(self, nq=3):

		qx,qy,w = [],[],[]
		with open('kpts.{nq}'.format(nq=nq),'r') as f:
			lines = f.readlines()
			for i in range(nq):
				qx.append(abs(float(lines[2+i].split()[4])*2.))
				qy.append(abs(float(lines[2+i].split()[5])*2.))
				w.append(float(lines[2+i].split()[10]))

		self.df['qx'],self.df['qy'],self.df['w'] = [qx,qy,w]


	def function_coupling(self,qx,qy):
		ro  = np.sqrt((qx**2+qy**2))
		ro_k = np.sqrt((1.154701-qy)**2+(0.666667-qx)**2)

		if ro <= self.r :
			return self.m_gamma
		elif ro_k <= self.r :
			return self.m_k
		else:
			return 0.
