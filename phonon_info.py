import numpy as np
from matplotlib import pyplot as plt

class energy(object):
	"""docstring for model_dispertion."""
	def __init__(self,dict):
		super(energy, self).__init__()
		self.dict = dict
		# flat dispersion
		self.omega_func = (lambda x: self.dict['input']['omega_ph0']*(0.93+0.07*np.cos(np.pi*x)))

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

class coupling(object):
	"""docstring for coupling_dispersion."""
	def __init__(self,omegaq,dict):
		super(coupling, self).__init__()

		self.omegaq = omegaq

		self.dict = dict
		# cos up dispersion
		self.func = lambda x: 0.2*(1+0.4*x*x*x*x)#*(1.5-0.5*abs(np.cos(x*np.pi/2)))

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
