from os import listdir
from os.path import isfile, join
import matplotlib.pyplot as plt
import numpy as np
import re
from scipy.signal import *
mypath='../../storage/qdep/'

detfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]

exp={}

for i,files in enumerate(detfiles):
    # temp=re.split("[q]", files)[-1]
    # print(temp)
    # name=re.split("[e]", temp)[-2]
    exp[files]={}
    exp[files]['x']=np.loadtxt(mypath+files).T[0]
    exp[files]['y']=np.loadtxt(mypath+files).T[1]
    name=files
    plt.plot(exp[name]['x'],exp[name]['y']/max(exp[name]['y']),label=name)#/max(exp[name]['y'])

[print(labels) for labels in exp]

# name_list=['291.4','291.8','292','292.2']
# for name in name_list:
#     peaks, _ = find_peaks(exp[name]['y'], threshold=0.1)
#
#     plt.plot(exp[name]['x'],exp[name]['y'],label=name)#/max(exp[name]['y'])
#     plt.xlim([-0.1,0.8])
# # # for i in range(len(detfiles)):
# # #
# # #     plt.plot(exp[i][0],i+exp[i][1]/max(exp[i][1]),label=i)
# #
plt.legend()

plt.show()
