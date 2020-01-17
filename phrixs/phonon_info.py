import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class energy(object):
	"""docstring for model_dispertion."""
	def __init__(self,dict):
		super(energy, self).__init__()
		self.q_path={}
		self.phonon_energy={}
		self.e2d=[]
		self.q2d={'x':[],'y':[]}
		self.dict = dict

		self.omega_func = (lambda x: self.dict['input']['omega_ph0']*(0.93+0.07*(np.cos(np.pi*x))))

		self.q = np.linspace(-1,1,self.dict['input']['nq'])

		self.omegaq = self.omega_func(self.q)

		np.savetxt('phonon_energy_vs_q',np.vstack((self.q,self.omegaq)))

	def plot(self):
		plt.plot(self.q,self.omegaq,linewidth=2)
		plt.axvline(0,linestyle='--',color='grey')
		plt.xlabel('$\mathrm{q_x, \ (\pi/a)} $',fontsize=20)
		plt.ylabel('$\mathrm{\omega} $',fontsize=20)
		plt.ylim([min(self.omegaq)*0.8,max(self.omegaq)*(1.2)])
		plt.xlim([max(self.q),min(self.q)])
		plt.show()

	def get_phonon(self):
		sym_directions=['gamma-k','k-m','gamma-m']
		transform={\
		'gamma-k':{'x':1,'y':0},\
		'k-m':{'x':np.cos(np.pi/3.),'y':np.sin(np.pi/3.)},\
		'gamma-m':{'x':np.sin(np.pi/3.),'y':np.cos(np.pi/3.)}}
		q_bz=[4./3.,2./3.,2./np.sqrt(3.)]
		for i,names in enumerate(sym_directions):
			print(i,names)
			self.q_path[names],self.phonon_energy[names]=\
						np.loadtxt('../../storage/'+names+'.csv').T
			self.q_path[names]=self.q_path[names] * q_bz[i]
			if names=='k-m':
				self.q2d['x'].extend(q_bz[0]-self.q_path[names][:] * transform[names]['x'])
			else:
				self.q2d['x'].extend(self.q_path[names][:] * transform[names]['x'])
			self.q2d['y'].extend(self.q_path[names][:] *transform[names]['y'])
			self.e2d.extend(self.phonon_energy[names])
		print(self.phonon_energy['gamma-k'])
		fig = plt.figure()
		ax = fig.add_subplot(111, projection='3d')
		ax.plot_trisurf(self.q2d['x'],self.q2d['y'],self.e2d)
		# ax.scatter(self.q2d['x'],self.q2d['y'],self.e2d)
		ax.set_xlabel('X Label')
		ax.set_ylabel('Y Label')
		ax.set_zlabel('eph Label')
		plt.show()



class coupling(object):
	"""docstring for coupling_dispersion."""
	def __init__(self,omegaq,dict):
		super(coupling, self).__init__()

		self.omegaq = omegaq

		self.dict = dict
		# cos up dispersion
		self.func = lambda x: 0.2*(1+0.4*(abs(x*x)))#*(1.5-0.5*abs(np.cos(x*np.pi/2)))

		self.q = np.linspace(-1.,1.,self.dict['input']['nq'])

		self.mkq = self.func(self.q)

		self.gkq = list(map(lambda x,y: x*x/y/y, self.mkq,self.omegaq))

		np.savetxt('eph_coupling_vs_q',np.vstack((self.q,self.mkq)))

		np.savetxt('eph_strength_vs_q',np.vstack((self.q,self.gkq)))


	def plot(self):
		plt.plot(self.q,self.mkq,linewidth=2,color='r',label='model2')
		plt.axvline(0,linestyle='--',color='grey')
		plt.xlabel('$\mathrm{q_x, \ (\pi/a)} $',fontsize=20)
		plt.ylabel('$\mathrm{M_k^q, \ (eV)} $',fontsize=20)
		plt.ylim([min(self.mkq)*0.8,max(self.mkq)*(1.3)])
		plt.xlim([max(self.q),min(self.q)])
		plt.show()

	def plot_strength(self):
		plot(self.q,self.gkq,linewidth=2,color='r')
		plt.axvline(0,linestyle='--',color='grey')
		plt.xlabel('$\mathrm{q_x, \ (\pi/a)} $',fontsize=20)
		plt.ylabel('$\mathrm{g_k^q} $',fontsize=20)
		plt.ylim([min(self.gkq)*0.8,max(self.gkq)*(1.3)])
		plt.xlim([max(self.q),min(self.q)])
		plt.show()
