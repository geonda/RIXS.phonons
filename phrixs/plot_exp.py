from os import listdir
from os.path import isfile, join
import matplotlib.pyplot as plt
import numpy as np
import re
mypath='../../storage/detuning/'

detfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]

exp={}

for i,files in enumerate(detfiles):
    temp=re.split("[=]", files)[-2]
    name=re.split("[e]", temp)[-2]
    exp[name]={}
    exp[name]['x']=np.loadtxt(mypath+files).T[0]
    exp[name]['y']=np.loadtxt(mypath+files).T[1]

[print(labels) for labels in exp]
name_list=['291.8','292.2']
for name in name_list:
    plt.plot(exp[name]['x'],exp[name]['y']/max(exp[name]['y']),label=name)
    plt.xlim([-0.1,0.8])
# # for i in range(len(detfiles)):
# #
# #     plt.plot(exp[i][0],i+exp[i][1]/max(exp[i][1]),label=i)
#
plt.legend()

plt.show()
