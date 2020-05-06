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
		self.nq=11j

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

	def plot_dispersion(self):

		self.get_phonon()
		fig = plt.figure(figsize=(8,4))

		ax = fig.add_subplot(131)
		ax.set_ylabel(r'Phonon Energy, eV',fontsize=15)
		ax.plot(self.q_path['gamma-k']*self.q_bz[0],self.phonon_energy['gamma-k'],'-o',color='grey')
		plt.xticks([min(self.q_path['gamma-k']*self.q_bz[0]),max(self.q_path['gamma-k']*self.q_bz[0])],(r'$\Gamma$',r'$K$'),fontsize=20)
		ax.set_ylim([0.14,0.2])

		ax = fig.add_subplot(132)
		ax.plot(self.q_path['k-m']*self.q_bz[1],self.phonon_energy['k-m'],'-o',color='grey',label='exp')
		plt.xticks([min(self.q_path['k-m']*self.q_bz[1]),max(self.q_path['k-m']*self.q_bz[1])],(r'$K$',r'$M$'),fontsize=20)
		ax.set_yticks([])
		ax.set_ylim([0.14,0.2])
		plt.legend()
		ax.set_xlabel(r'$q \ path$',fontsize=15)

		ax = fig.add_subplot(133)
		ax.set_yticks([])
		ax.set_ylim([0.14,0.2])
		ax.plot(self.q_path['gamma-m']*self.q_bz[2],self.phonon_energy['gamma-m'],'-o',color='grey',label='exp')
		plt.xticks([min(self.q_path['gamma-m']*self.q_bz[2]),max(self.q_path['gamma-m']*self.q_bz[2])],(r'$\Gamma$',r'$M$'),fontsize=20)
		plt.gca().invert_xaxis()
		plt.tight_layout()
		# plt.show()

	def plot_colormap(self):
		self.get_phonon()
		fig = plt.figure()
		ax = fig.add_subplot(111)
		ims=ax.imshow(self.grid_rect_z.T, extent=(-max(self.q2d['x']),max(self.q2d['x']),-max(self.q2d['y']),max(self.q2d['y'])),origin='lower',cmap="plasma")
		plt.ylim(-1.5,1.5)
		plt.title('Graphite 2D')
		plt.text(-0.2,-0.2,r'$\Gamma$',fontsize=15)
		plt.text(1.3,-0.2,r'$K$',fontsize=15)
		plt.text(1.,0.65,r'$M$',fontsize=15)
		cbar = plt.colorbar(ims)
		cbar.set_label(r'Phonon Energy, eV')
		ax.scatter(self.q2d['x'],self.q2d['y'],label='exp',edgecolor='k',facecolor='none')
		# ax.scatter(self.x,self.y,label='exp',edgecolor='k',facecolor='none')
		plt.legend()
		hx=rp((0,0), 6, radius=4./3., orientation=np.pi/2.,facecolor="none",edgecolor='grey')
		ax.add_patch(hx)
		ax.set_xlabel(r'$q_x,\ \frac{\pi}{a}$',fontsize=15)
		ax.set_ylabel(r'$q_x,\ \frac{\pi}{a}$',fontsize=15)
		# plt.show()

	def interpolate(self,x,y,z):
		points=np.vstack((x,y))
		# print(max(x),max(y))
		self.grid_rect_x, self.grid_rect_y = np.mgrid[-max(x):max(x):self.nq, -max(y):max(y):self.nq]
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


class full_data(object):
	"""docstring for coupling_dispersion."""

	def __init__(self):
		super(full_data, self).__init__()
		self.q_path={}
		self.coupling_qpath={}
		self.e2d=[]
		self.q2d={'x':[],'y':[]}
		self.phonon=energy()
		self.phonon.get_phonon()

		# self.phonon.plot_dispersion()

		# self.omegaq = omegaq
		#
		# self.dict = dict
		# # cos up dispersion
		# self.func = lambda x: 0.2*(1+0.4*(abs(x*x)))#*(1.5-0.5*abs(np.cos(x*np.pi/2)))
		#
		# self.q = np.linspace(-1.,1.,self.dict['input']['nq'])
		#
		# self.mkq = self.func(self.q)
		#
		# self.gkq = list(map(lambda x,y: x*x/y/y, self.mkq,self.omegaq))
		#
		# np.savetxt('eph_coupling_vs_q',np.vstack((self.q,self.mkq)))
		#
		# np.savetxt('eph_strength_vs_q',np.vstack((self.q,self.gkq)))

		self.sym_directions=['gamma-k','k-m','gamma-m']
		self.create_coupling()
		self.get_coupling()
		# print(len(self.x),len(self.y),len(self.z),len(self.phonon.z),len(self.phonon.x))
		self.qx,self.qy = self.x,self.y
		# print(len(self.qx))
		self.phonon_energy = self.phonon.z
		self.coupling_constant = self.z
		# print(len(self.z),len(self.phonon.z))
		self.coupling_strength = (np.array(self.z)/np.array(self.phonon.z))**2

		for i in self.coupling_constant:
			if math.isnan(i):
				print('NAN')

		self.qx, self.qy, self.phonon_energy, self.coupling_constant,self.coupling_strength \
		 = self.reduce()
		np.savetxt('./_out/_temp_phonon_energy_vs_q_2d',np.vstack((self.qx,self.qy,self.phonon_energy)).T)
		np.savetxt('./_out/_temp_eph_coupling_vs_q_2d',np.vstack((self.qx,self.qy,self.coupling_constant)).T)
		np.savetxt('./_out/_temp_eph_strength_vs_q_2d',np.vstack((self.qx,self.qy,self.coupling_strength)).T)

		# print(self.coupling_strength)
		# fig = plt.figure()
		# ax = fig.add_subplot(121, projection='3d')
		# ax.plot_trisurf(self.qx, self.qy, self.coupling_constant, color='white', edgecolors='grey', alpha=0.5)
		# ax.scatter(self.qx, self.qy, self.coupling_constant, c='red')
		# ax.set_xlabel('$q_x$')
		# ax.set_ylabel('$q_y$')
		# ax.set_zlabel('$M,\ eV $')
		# # fig = plt.figure()
		# # ax = fig.add_subplot(121, projection='3d')
		# # ax.plot_trisurf(self.qx, self.qy, self.coupling_strength, color='white', edgecolors='grey', alpha=0.5)
		# # ax.scatter(self.qx, self.qy, self.coupling_strength, c='red')
		# ax = fig.add_subplot(122, projection='3d')
		# ax.plot_trisurf(self.qx, self.qy, self.phonon_energy, color='white', edgecolors='grey', alpha=0.5)
		# ax.scatter(self.qx, self.qy, self.phonon_energy, c='b')
		# ax.set_xlabel('$q_x$')
		# ax.set_ylabel('$q_y$')
		# ax.set_zlabel(r'$\omega_{ph},\ eV $')
		# plt.show()

	def reduce(self):
		print('initial size : ',len(self.qx))
		threshold=0.1
		qx_new, qy_new, energy_ph_new, coupling_ph_new, strength_ph_new \
					= [],[],[],[],[]
		for x, y, ep, cp, sp in zip(self.qx, self.qy, self.phonon_energy, self.coupling_constant,\
							self.coupling_strength):
			if cp >= threshold:
				qx_new.append(x)
				qy_new.append(y)
				energy_ph_new.append(ep)
				coupling_ph_new.append(cp)
				strength_ph_new.append(sp)
		print('final size : ',len(qx_new))
		return qx_new, qy_new, energy_ph_new, coupling_ph_new, strength_ph_new

	def create_coupling(self,name=''):

		self.func_obj={}

		import json
		with open('./inputs/input_phonon_info.json') as fp:
			ti=json.load(fp)

		self.ak = ti['ak']
		self.ag = ti['ag']
		self.am = ti['am']


		self.rk = ti['rk']

		self.rg = ti['rg']

		self.rm = ti['rm']


		self.func_obj['gamma-k'] = lambda x: np.heaviside(self.rg-x,0.5)*self.ag+np.heaviside(x-1+self.rk,0.5)*self.ak

		self.func_obj['k-m'] = lambda  x: np.heaviside(self.rk-x,0.5)*self.ak + np.heaviside(x-1+self.rm,0.5)*self.am

		self.func_obj['gamma-m'] = lambda x: np.heaviside(self.rg-x,0.5)*self.ag + np.heaviside(x-1+self.rm,0.5)*self.am

		#ag*np.exp(-x**2/sigmag)/max(np.exp(-x**2/sigmag))
		# self.func_obj['gamma-k'] = lambda x: ag*np.imag(1/(x-1.j*gammag))/max(np.imag(1/(x-1.j*gammag)))+ak*np.imag(1/(x-1+off-1.j*gammak))/max(np.imag(1/(x-1+off-1.j*gammak)))
		#
		# self.func_obj['k-m'] = lambda  x: ak*np.imag(1/(x-off-1.j*gammak))/max(np.imag(1/(x-off-1.j*gammak)))
		#
		# self.func_obj['gamma-m'] = lambda x: ag*np.imag(1/(x-1.j*gammag))/max(np.imag(1/(x-1.j*gammag)))

		for i,direction in enumerate(self.sym_directions):
			q=np.linspace(0,1.01,101)
			# print(direction,self.func_obj[direction](q))
			with open('./_exp_files/TO/'+name+'_coupling_'+str(direction)+'.csv','w') as f:
				data_temp=np.vstack((q,self.func_obj[direction](q)))
				np.savetxt(f,data_temp)

	def get_coupling(self,name=''):
		self.create_coupling()
		transform={\
		'gamma-k':{'x':1,'y':0},\
		'k-m':{'x':np.cos(np.pi/3.),'y':np.sin(np.pi/3.)},\
		'gamma-m':{'x':np.sin(np.pi/3.),'y':np.cos(np.pi/3.)}}
		self.q_bz=[4./3.,2./3.,2./np.sqrt(3.)]
		for i,names in enumerate(self.sym_directions):
			# print(i,names)
			self.q_path[names],self.coupling_qpath[names]\
					=np.loadtxt('./_exp_files/TO/'+name+'_coupling_'+names+'.csv')
			self.q_path[names]=self.q_path[names] * self.q_bz[i]
			if names=='k-m':
				self.q2d['x'].extend(self.q_bz[0]-self.q_path[names][:] * transform[names]['x'])
			else:
				self.q2d['x'].extend(self.q_path[names][:] * transform[names]['x'])
			self.q2d['y'].extend(self.q_path[names][:] * transform[names]['y'])
			self.e2d.extend(self.coupling_qpath[names])
			for i in self.e2d:
				if math.isnan(i):
					print('e2d:',i)
		self.interpolate(self.q2d['x'],self.q2d['y'],self.e2d)
		# self.reshape_local()

	def plot_dispersion_strength(self):
		self.get_coupling()
		fig = plt.figure(figsize=(10,5))
		self.coupling_qpath['gamma-k']/self.phonon.phonon_energy['gamma-k']

		plt.tight_layout()
		plt.show(block=False)

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

	def plot_colormap(self):

		self.get_coupling()

		self.interpolate_plot(self.q2d['x'],self.q2d['y'],self.e2d)

		# self.reshape_local()
		# import matplotlib.pyplot as plt
		# from mpl_toolkits.mplot3d import Axes3D
		# fig = plt.figure()
		# ax = fig.add_subplot(111, projection='3d')
		# print(self.x[0],self.z)
		# ax.plot(self.x,self.y,self.z)
		# plt.show()
		#print(self.grid_rect_z.shape)
		fig = plt.figure()
		ax = fig.add_subplot(111)
		ims=ax.imshow(self.grid_rect_z.T, extent=(-max(self.q2d['x']),max(self.q2d['x']),-max(self.q2d['y']),max(self.q2d['y'])),origin='lower',cmap="plasma")
		plt.ylim(-1.5,1.5)
		plt.title('Graphite 2D')
		plt.text(-0.2,-0.2,r'$\Gamma$',fontsize=15)
		plt.text(1.3,-0.2,r'$K$',fontsize=15)
		plt.text(1.,0.65,r'$M$',fontsize=15)
		cbar = plt.colorbar(ims)
		cbar.set_label(r'Coupling Constant, eV')
		# ax.scatter(self.q2d['x'],self.q2d['y'])
		# ax.scatter(self.x,self.y)
		hx=rp((0,0), 6, radius=4./3., orientation=np.pi/2.,facecolor="none",edgecolor='grey')
		ax.add_patch(hx)
		ax.set_xlabel(r'$q_x,\ \frac{\pi}{a}$',fontsize=15)
		ax.set_ylabel(r'$q_x,\ \frac{\pi}{a}$',fontsize=15)
		plt.show()

	def interpolate_plot(self,x,y,z):

		points=np.vstack((x,y))
		# print(max(x),max(y))

		self.grid_rect_x, self.grid_rect_y = np.mgrid[-max(x):max(x):11j, -max(y):max(y):11j]


		# print('x:',self.grid_rect_x.shape)
		# print('x ph:',self.phonon.x.shape)
		# self.grid_rect_x, self.grid_rect_y = self.phonon.x,self.phonon.y
		self.grid_rect_z = griddata(points.T,z,(self.grid_rect_x, self.grid_rect_y), method='cubic')
		# print(self.grid_rect_z.shape)
		# print('coupling length',len(self.grid_rect_z))
		# self.x,self.y,self.z=self.grid_rect_x,self.grid_rect_y,self.grid_rect_z

	def interpolate(self,x,y,z):
		points=np.vstack((x,y))
		# print(max(x),max(y))
		# self.grid_rect_x, self.grid_rect_y = np.mgrid[-max(x):max(x):11j, -max(y):max(y):11j]
		self.grid_rect_x, self.grid_rect_y = self.phonon.x,self.phonon.y

		self.grid_rect_z=[]
		for xi,yi in zip(self.grid_rect_x,self.grid_rect_y):
			if (xi**2+yi**2) <= self.rg:
				self.grid_rect_z.append(self.ag)
			elif ((xi-max(x))**2+yi**2) <= self.rk:
				self.grid_rect_z.append(self.ak)
			else:
				self.grid_rect_z.append(0)
		self.grid_rect_z=np.array(self.grid_rect_z)
		# self.grid_rect_z = griddata(points.T,z,(self.grid_rect_x, self.grid_rect_y), method='cubic')
		# print(self.grid_rect_z.shape)
		# print('coupling length',len(self.grid_rect_z))
		self.x,self.y,self.z=self.grid_rect_x,self.grid_rect_y,self.grid_rect_z

	def reshape_local(self):
		x,y,z=[],[],[]
		for i in range(len(self.grid_rect_z)):
			for j in range(len(self.grid_rect_z)):
				if math.isnan(self.grid_rect_z[i][j]):
					pass
				else:
					# print(self.grid_rect_z[i][j])
					z.append(self.grid_rect_z[i][j])
					x.append(self.grid_rect_x[i][j])
					y.append(self.grid_rect_y[i][j])
		self.x,self.y,self.z=x,y,z
		# print(self.z)
