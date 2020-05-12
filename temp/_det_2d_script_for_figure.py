from lib import *
from tqdm import tqdm
import matplotlib.pyplot as plt
##############################
# different couplings for beta = 1
#############################
detuning=np.linspace(0.0,0.5,5)

ws=workspace()

ws.initp(type_calc='2d',vib_space=2)                                                     #ws.initp(vib_space=2)

ws.inputp('skip')                                                               # use ws.inputp('ask') to modify

ws.dict_total['input']['omega_ph0']=0.03

ws.dict_total['input']['coupling0']=0.1

ws.dict_total['input']['omega_ph1']=0.1

ws.dict_total['input']['coupling1']=0.125

ws.dict_total['input']['gamma']=0.2

ws.dict_total['input']['nm']=30

ws.dict_total['input']['nf']=6

intesity_of_first_peak={}

intesity_of_first_peak['mode0']=[]

intesity_of_first_peak['mode1']=[]

for dE in detuning:

    ws.dict_total['input']['omega_in']=ws.dict_total['input']['energy_ex']+dE

    ws.runp()
                                                                                # exectute
    file='temp_0_run_'+str(ws.nruns)+'.npy'
                                                                                # file with peak positions vs intensities (without broadening)
    data_temp=np.load(file)

    for i,energy in enumerate(data_temp[0]):

        if energy==ws.dict_total['input']['omega_ph0']:

            intesity_of_first_peak['mode0'].append(data_temp[1][i])

        elif energy==ws.dict_total['input']['omega_ph1']:

                intesity_of_first_peak['mode1'].append(data_temp[1][i])


plt.plot(detuning,intesity_of_first_peak['mode0']/\
                max(intesity_of_first_peak['mode0']),color='b',label='mode0')

plt.plot(detuning,intesity_of_first_peak['mode1']/\
                max(intesity_of_first_peak['mode1']),color='r',label='mode1')

np.savetxt('3_temp_det_2d_mode_0',\
                np.vstack((detuning,intesity_of_first_peak['mode0']\
                            /max(intesity_of_first_peak['mode0']))).T)
np.savetxt('3_temp_det_2d_mode_1',\
                np.vstack((detuning,intesity_of_first_peak['mode1']\
                            /max(intesity_of_first_peak['mode1']))).T)


plt.xlabel('Detuning, (eV)', fontsize=15)


plt.ylabel(r'$I_1(Z)/I_1(Z=0)$', fontsize=15)

plt.legend()

plt.show()
