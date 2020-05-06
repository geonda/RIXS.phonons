import seaborn as sns
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
# df=pd.read_csv('../../storage/export/XAS.txt')
# # fig=sns.pairplot(df)
# # fig.show()
# # df.plot()

x,y=np.loadtxt('../../../storage/export/XAS.txt').T



fig=plt.figure(figsize=(10,5))

axes1 = fig.add_axes([0.1, 0.1, 0.8, 0.8]) # main axes
axes2 = fig.add_axes([0.2, 0.5, 0.3, 0.3]) # inset axes

# Larger Figure Axes 1
axes1.plot(x, y, 'grey')
axes1.set_xlabel('Energy, eV')
axes1.set_ylabel('XAS intensity')
axes1.set_title('C K-edge')
axes1.set_xlim([285,295])

# Insert Figure Axes 2
dict={}
for name,color in zip(['291.8','292'],['r','b']):
    axes1.axvline(x=float(name),color=color,label=name)
    dict[name]={'x':[],'y':[]}
    dict[name]['x'],dict[name]['y'],dict[name]['z']=np.loadtxt('../../../storage/export/{N}.txt'.format(N=name)).T
    axes2.plot(dict[name]['x'],dict[name]['y'],label=name,color=color)
    axes2.set_xlabel('Energy loss, eV')
    axes2.set_ylabel('RIXS intensity')
    axes2.set_xlim([-0.1,1.])
plt.legend()
plt.show()
