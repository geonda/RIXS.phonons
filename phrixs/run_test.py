from phonon_info import *
dict_trial={'input':{'omega_ph0':0.1,'nq':10}}
# energy(dict_trial).plot_colormap()
# energy(dict_trial).plot_dispersion()
# plt.show()
omq_trial=np.linspace(1,1.1,10)
coupling(omq_trial,dict_trial)
