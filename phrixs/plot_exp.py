from os import listdir
from os.path import isfile, join
import matplotlib.pyplot as plt
import numpy as np
import re
from scipy.signal import *
mypath='../../storage/detuning/'

detfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]

exp={}

for i,files in enumerate(detfiles):
    temp=re.split("[=]", files)[-2]
    name=re.split("[e]", temp)[-2]
    exp[name]={}
    exp[name]['x']=np.loadtxt(mypath+files).T[0]
    exp[name]['y']=np.loadtxt(mypath+files).T[1]/10000

# [print(labels) for labels in exp]
#
name_list=['291.8','292','292.2']
for name in name_list:
    peaks, _ = find_peaks(exp[name]['y'], height=2.)
    print(exp[name]['y'][peaks[1]])
    # plt.axhline(1)
    plt.plot(exp[name]['x'],exp[name]['y'],label='$\omega_{in}$='+name+' eV')#/max(exp[name]['y'])
    # np.savetxt(name,np.vstack((exp[name]['x'],exp[name]['y']/exp[name]['y'][peaks[2]])))
    plt.xlim([-0.1,0.8])
    plt.xlabel('Energy loss, eV',fontsize=15)
    plt.ylabel('RIXS Intensity, arb. units',fontsize=15)
# # for i in range(len(detfiles)):
# #
# #     plt.plot(exp[i][0],i+exp[i][1]/max(exp[i][1]),label=i)
#
plt.legend()

plt.show()
# name_list=['291.8','292','292.2']
# for i,name in enumerate(name_list):
#     data=np.loadtxt(name)
#     plt.plot(data[0],data[1]+i,label=name)#/max(exp[name]['y'])
#     plt.xlim([-0.1,0.8])
# # # for i in range(len(detfiles)):
# # #
# # #     plt.plot(exp[i][0],i+exp[i][1]/max(exp[i][1]),label=i)
# #
# plt.legend()
#
# plt.show()