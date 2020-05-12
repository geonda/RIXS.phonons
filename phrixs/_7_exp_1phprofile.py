import seaborn as sns
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import find_peaks

# main axes
off=291.48
x,y=np.loadtxt('_out/1ph_profile_model')
xe,ye=np.loadtxt('temp_1phprofile')
fig=plt.figure(figsize=(10,5))
axes1 = fig.add_axes([0.1, 0.1, 0.8, 0.8])
axes1.plot(x+off,y/max(y),label='model')
axes1.scatter(xe,ye/max(ye),facecolor='None', edgecolor='k',label='exp')
axes1.set_xlabel('Energy, eV')
axes1.set_ylabel('1 ph profile intensity')
plt.legend()
plt.show()
# np.savetxt('temp_1phprofile',np.vstack((x_temp,y_temp)))
