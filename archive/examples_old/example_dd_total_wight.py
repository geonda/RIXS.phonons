from lib import *

ws = workspace()
ws.timer_start()

#displaced osc
ws.initp(type_calc='model')

ws.inputp('skip') # use ws.inputp('ask') to modify input
ws.dict_total['input']['nf']=10
ws.dict_total['input']['nm']=10
ws.dict_total['input']['omega_ph0']=0.05

total_wieght_beta_1=[]
total_wieght_beta_small=[]
total_wieght_beta_big=[]
list_coupling=[0.,0.01,0.02,0.03,0.05]
for coupling in list_coupling:
    ws.dict_total['input']['coupling0']=coupling
    ws.runp()
    file='temp_0_run_'+str(ws.nruns)+'.npy' # file with peak positions vs intensities (without broadening)
    data_temp=np.load(file)
    total_wieght_beta_1.append(1-data_temp[1][0]/sum(data_temp[1]))



# plt.show()
ws.initp(type_calc='dd')
ws.inputp('skip') # use ws.inputp('ask') to modify input or call ws.dict_total['input']['parameter_name']
beta_list=[0.9,1.1]

for coupling in list_coupling:
    beta=beta_list[0]
    ws.dict_total['input']['omega_ph0']=0.05
    ws.dict_total['input']['omega_ph_ex']=ws.dict_total['input']['omega_ph0']*beta*beta
    ws.dict_total['input']['nm']=7
    ws.dict_total['input']['nf']=7 # i have suspicion that nm and nf should be equal
    ws.dict_total['input']['coupling0']=coupling
    ws.runp()
    file='temp_0_run_'+str(ws.nruns)+'.npy' # file with peak positions vs intensities (without broadening)
    data_temp=np.load(file)
    total_wieght_beta_small.append(1-data_temp[1][0]/sum(data_temp[1]))

for coupling in list_coupling:
    beta=beta_list[1]
    ws.dict_total['input']['omega_ph0']=0.05
    ws.dict_total['input']['omega_ph_ex']=ws.dict_total['input']['omega_ph0']*beta*beta
    ws.dict_total['input']['nm']=7
    ws.dict_total['input']['nf']=7 # i have suspicion that nm and nf should be equal
    ws.dict_total['input']['coupling0']=coupling
    ws.runp()
    file='temp_0_run_'+str(ws.nruns)+'.npy' # file with peak positions vs intensities (without broadening)
    data_temp=np.load(file)
    total_wieght_beta_big.append(1-data_temp[1][0]/sum(data_temp[1]))

import matplotlib.pyplot as plt
plt.plot(list_coupling,total_wieght_beta_big,'-o',label=r'$\beta=1.1$',color='r')
plt.plot(list_coupling,total_wieght_beta_1,'-o',label=r'$\beta=1$',color='b')
plt.plot(list_coupling,total_wieght_beta_small,'-o',label=r'$\beta=0.9$',color='g')
plt.xlabel('g',fontsize=20)
plt.ylabel('W_ph, eV',fontsize=20)
plt.show()
# ws.timer_total('total [s]: ')
# ws.plotp(scale=0)
# ws.clear()
