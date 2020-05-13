from lib import *
from tqdm import tqdm
import matplotlib.pyplot as plt
###
##############################
# two curves for beta 0.7 and 1.3
#############################

# intesity_of_first_peak={}
# intesity_of_elastic_peak={}
#
# beta_list=[0.5]
#
# al=[1]
#
# detuning=np.linspace(-0.6,0.6,50) # for beta 1 detuning step can be changed
#
# ws=workspace()
#
# ws.timer_start()
#
# ws.initp(type_calc='dd')
#
# ws.inputp('skip')
# plt.subplot(131)
#
# for i,beta in enumerate(beta_list):
#
#     ws.dict_total['input']['omega_ph0']=0.1
#
#     ws.dict_total['input']['gamma']=0.2
#
#     ws.dict_total['input']['energy_ex']=4
#
#     ws.dict_total['input']['omega_ph_ex']=ws.dict_total['input']['omega_ph0']*beta*beta
#
#     ws.dict_total['input']['nm']=20
#
#     ws.dict_total['input']['nf']=2                                              # only first peak is calculted to save time
#
#     ws.dict_total['input']['coupling0']=0.125
#
#     intesity_of_first_peak[beta]=[]
#     intesity_of_elastic_peak[beta]=[]
#
#     for dE in detuning:
#
#         ws.dict_total['input']['omega_in']\
#                 =ws.dict_total['input']['energy_ex']+dE
#         ws.runp()                                                               # exectute
#
#         file='temp_0_run_'+str(ws.nruns)+'.npy'
#
#         data_temp=np.load(file)                                                 # file with peak positions vs intensities (without broadening)
#
#         intesity_of_first_peak[beta].append(data_temp[1][1])
#         intesity_of_elastic_peak[beta].append(data_temp[1][0])                    # first phonon peak intensity
#     plt.plot(detuning,intesity_of_elastic_peak[beta]/max(intesity_of_elastic_peak[beta]),
#                             '-',label='0 ph vs det',color='b',alpha=al[i])
#     plt.plot(detuning,intesity_of_first_peak[beta]/max(intesity_of_first_peak[beta]),
#                         '-',label='1 ph vs det',color='r',alpha=al[i])
#
# plt.axvline(0,color='grey',linestyle='--')
# plt.legend()
# plt.title(r'$\beta=0.5,\ \gamma=0.2\ eV$')
#     # np.savetxt('0_temp_det_dd_beta_'+str(beta),np.vstack((detuning,intesity_of_first_peak[beta]/intesity_of_first_peak[beta][0])).T)
# ws.timer_total('total [s]: ')
# #
# plt.xlabel('Detuning, (eV)', fontsize=15)
#
#
# plt.ylabel(r'$I_1(Z)/I_1(Z=0)$', fontsize=15)
# # ##############################
# # # different couplings for beta = 1
# # #############################
#


plt.subplot(1,2,1)
detuning_beta_1=np.linspace(-0.6,0.6,200)
plt.title(r'$\beta=1,\ \gamma=0.2\ eV$')
ws=workspace()
# # #
ws.initp(type_calc='model')                                                     #ws.initp(vib_space=2)

ws.inputp('skip')
                                             # use ws.inputp('ask') to modify
ws.dict_total['input']['coupling0']=0.125

ws.dict_total['input']['omega_ph0']=0.1

ws.dict_total['input']['gamma']=0.2

ws.dict_total['input']['nm']=40

ws.dict_total['input']['nf']=2

intesity_of_first_peak=[]

intesity_of_elastic_peak=[]


for dE in detuning_beta_1:

    ws.dict_total['input']['omega_in']=ws.dict_total['input']['energy_ex']+dE

    ws.runp()
                                                                                # exectute
    file='temp_0_run_'+str(ws.nruns)+'.npy'
                                                                                # file with peak positions vs intensities (without broadening)
    data_temp=np.load(file)

    intesity_of_first_peak.append(data_temp[1][1])

    intesity_of_elastic_peak.append(data_temp[1][0])                    # first phonon peak intensity

plt.plot(detuning_beta_1,intesity_of_first_peak/max(intesity_of_first_peak),
                '-',label='1 ph vs det',color='r')
plt.plot(detuning_beta_1,intesity_of_elastic_peak/max(intesity_of_elastic_peak),
                '-',label='0 ph vs det',color='b')
plt.axvline(0,color='grey',linestyle='--')
plt.xlabel('Detuning, (eV)', fontsize=15)

arr=np.array(intesity_of_elastic_peak)
result = np.where(arr == np.amax(arr))
plt.axvline(detuning_beta_1[result[0]],color='green',linestyle='-')
plt.ylabel(r'$I(Z)$', fontsize=15)
plt.legend()
plt.subplot(122)
plt.title(r'$\beta=1,\ \gamma=0.05\ eV$')
# detuning_beta_1=np.linspace(-0.8,0.8,80)
#
# coupling_list=[0.095,0.105,0.115,0.125,0.135,0.145,0.155]
#
# x_coupling=0.125
#
# alpha_list=np.linspace(0,1,len(coupling_list))
#

#
intesity_of_first_peak=[]

intesity_of_elastic_peak=[]
ws.initp(type_calc='model')                                                     #ws.initp(vib_space=2)

ws.inputp('skip')                                                               # use ws.inputp('ask') to modify
ws.dict_total['input']['coupling0']=0.125
ws.dict_total['input']['omega_ph0']=0.1

ws.dict_total['input']['gamma']=0.05

ws.dict_total['input']['nm']=40

ws.dict_total['input']['nf']=2

ws.dict_total['input']['energy_ex']=4

intesity_of_first_peak=[]

intesity_of_elastic_peak=[]


for dE in detuning_beta_1:

    ws.dict_total['input']['omega_in']=ws.dict_total['input']['energy_ex']+dE

    ws.runp()
                                                                                # exectute
    file='temp_0_run_'+str(ws.nruns)+'.npy'
                                                                                # file with peak positions vs intensities (without broadening)
    data_temp=np.load(file)

    intesity_of_first_peak.append(data_temp[1][1])

    intesity_of_elastic_peak.append(data_temp[1][0])                    # first phonon peak intensity

plt.plot(detuning_beta_1,intesity_of_first_peak/max(intesity_of_first_peak),
                '-',label='1 ph vs det',color='r')
plt.plot(detuning_beta_1,intesity_of_elastic_peak/max(intesity_of_elastic_peak),
                '-',label='0 ph vs det',color='b')

plt.xlabel('Detuning, (eV)', fontsize=15)
plt.axvline(0,color='grey',linestyle='--')
arr=np.array(intesity_of_elastic_peak)
result = np.where(arr == np.amax(arr))
plt.axvline(detuning_beta_1[result[0]],color='green',linestyle='-')

plt.ylabel(r'$I(Z)$', fontsize=15)

plt.legend()
plt.tight_layout()
plt.show()

# ws.plotp()
ws.clear()
