import numpy as np
from matplotlib import pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import RegularPolygon as rp
from scipy.interpolate import griddata
import matplotlib.ticker as ticker
import math

class energy(object):

	"""docstring for model_dispertion."""

	def __init__(self):
		super(energy, self).__init__()
		self.path_to_file='./_exp_files/TO/'
		self.q_path={}
		self.phonon_energy={}
		self.e2d=[]
		self.q2d={'x':[],'y':[]}
		self.dict = dict
		self.nq=3j

	def get_phonon(self):
		sym_directions=['gamma-k','k-m','gamma-m']
		transform={\
		'gamma-k':{'x':1,'y':0},\
		'k-m':{'x' : np.cos(np.pi/3.),'y' : np.sin(np.pi/3.)},\
		'gamma-m':{'x' : np.sin(np.pi/3.),'y' : np.cos(np.pi/3.)}}
		self.q_bz=[4./3.,2./3.,2./np.sqrt(3.)]
		for i,names in enumerate(sym_directions):
			# print(i,names)
			self.q_path[names],self.phonon_energy[names]=\
						np.loadtxt(self.path_to_file+names+'.csv').T
			self.q_path[names]=self.q_path[names] * self.q_bz[i]
			if names=='k-m':
				self.q2d['x'].extend(self.q_bz[0]-self.q_path[names][:] * transform[names]['x'])
			else:
				self.q2d['x'].extend(self.q_path[names][:] * transform[names]['x'])
			self.q2d['y'].extend(self.q_path[names][:] *transform[names]['y'])
			self.e2d.extend(self.phonon_energy[names])
		# print(self.phonon_energy['gamma-k'])
		self.interpolate(self.q2d['x'],self.q2d['y'],self.e2d)
		self.reshape_local()


		# plt.show()

	def interpolate(self,x,y,z):
		points=np.vstack((x,y))
		# print(max(x),max(y))
		self.grid_rect_x, self.grid_rect_y = np.mgrid[-max(x):max(x):self.nq, -max(y):max(y):self.nq]
		print(self.grid_rect_x.size())
		self.grid_rect_z = griddata(points.T,z,(self.grid_rect_x, self.grid_rect_y), method='cubic')
		# print('phonon length',len(self.grid_rect_z))

	def reshape_local(self):
		x,y,z=[],[],[]
		for i in range(len(self.grid_rect_z)):
			for j in range(len(self.grid_rect_z)):
				if math.isnan(self.grid_rect_z[i][j]):
					# print(self.grid_rect_x[i][j],self.grid_rect_y[i][j],'NAN')
					pass
				else:
					z.append(self.grid_rect_z[i][j])
					x.append(self.grid_rect_x[i][j])
					y.append(self.grid_rect_y[i][j])
					# print('q=',self.grid_rect_x[i][j],self.grid_rect_y[i][j],'omega=',self.grid_rect_z[i][j])
		self.x,self.y,self.z=x,y,z
