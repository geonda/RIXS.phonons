import seaborn as sns
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import find_peaks

# main axes


phprofile=[]
x_temp = np.arange(291.0,293.0,0.2)
for name in x_temp:
    name=np.round(name,1)
    x,y,_=np.loadtxt('./_exp_files/export/profile/{N}.txt'.format(N=name)).T
    peaks,_=find_peaks(y, prominence=(max(y)/7, None))
    plt.plot(x,y)
    plt.plot(x[peaks][1],y[peaks][1],'x')
    plt.show()
    phprofile.append(y[peaks][1])



y_temp = np.array(phprofile)
# print(np.arange(291.0,292.0,0.2))
# print(phprofile)

fig=plt.figure(figsize=(10,5))
axes1 = fig.add_axes([0.1, 0.1, 0.8, 0.8])
axes1.scatter(x_temp,y_temp,facecolor=None, color='k')
axes1.set_xlabel('Energy, eV')
axes1.set_ylabel('1 ph profile intensity')
plt.legend()
plt.show()
np.savetxt('temp_1phprofile',np.vstack((x_temp,y_temp)))
