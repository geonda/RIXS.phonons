from lib import *
from tqdm import tqdm
import matplotlib.pyplot as plt
###
##############################
# two curves for beta 0.7 and 1.3
#############################

intesity_of_first_peak={}

beta_list=[0.7,1.3]

cr=['g','r']

al=[1,1]

detuning=np.linspace(0.0,0.5,2) # for beta 1 detuning step can be changed


ws=workspace()

ws.timer_start()

ws.initp(type_calc='dd')

ws.inputp('skip')

for i,beta in enumerate(beta_list):

    ws.dict_total['input']['omega_ph0']=0.1

    ws.dict_total['input']['gamma']=0.2

    ws.dict_total['input']['omega_ph_ex']=ws.dict_total['input']['omega_ph0']*beta*beta

    ws.dict_total['input']['nm']=12

    ws.dict_total['input']['nf']=2                                              # only first peak is calculted to save time

    ws.dict_total['input']['coupling0']=0.125

    intesity_of_first_peak[beta]=[]

    for dE in detuning:

        ws.dict_total['input']['omega_in']\
                =ws.dict_total['input']['energy_ex']+dE
        ws.runp()                                                               # exectute

        file='temp_0_run_'+str(ws.nruns)+'.npy'

        data_temp=np.load(file)                                                 # file with peak positions vs intensities (without broadening)

        intesity_of_first_peak[beta].append(data_temp[1][1])                    # first phonon peak intensity

    plt.plot(detuning,intesity_of_first_peak[beta]/intesity_of_first_peak[beta][0],
                        '-o',label='$beta$='+str(beta),color=cr[i],alpha=al[i])

    np.savetxt('0_temp_det_dd_beta_'+str(beta),np.vstack((detuning,intesity_of_first_peak[beta]/intesity_of_first_peak[beta][0])).T)
ws.timer_total('total [s]: ')
#

##############################
# different couplings for beta = 1
#############################
detuning_beta_1=np.linspace(0.0,0.5,20)

coupling_list=[0.095,0.105,0.115,0.125,0.135,0.145,0.155]

x_coupling=0.125

alpha_list=np.linspace(0,1,len(coupling_list))

# ws=workspace()

ws.initp(type_calc='model')                                                     #ws.initp(vib_space=2)

ws.inputp('skip')                                                               # use ws.inputp('ask') to modify

ws.dict_total['input']['omega_ph0']=0.1

ws.dict_total['input']['gamma']=0.2

ws.dict_total['input']['nm']=20

ws.dict_total['input']['nf']=2

intesity_of_first_peak_sf={}


for i,coupling in enumerate(coupling_list):

    ws.dict_total['input']['coupling0']=coupling

    intesity_of_first_peak_sf[coupling]=[]

    for dE in detuning_beta_1:

        ws.dict_total['input']['omega_in']=ws.dict_total['input']['energy_ex']+dE

        ws.runp()
                                                                                # exectute
        file='temp_0_run_'+str(ws.nruns)+'.npy'
                                                                                # file with peak positions vs intensities (without broadening)
        data_temp=np.load(file)

        intesity_of_first_peak_sf[coupling].append(data_temp[1][1])
    if coupling==x_coupling:
        linestyle='-'
    else:
        linestyle='--'
    plt.plot(detuning_beta_1,intesity_of_first_peak_sf[coupling]/\
                max(intesity_of_first_peak_sf[coupling]),linestyle=linestyle,color='b',\
                                            alpha=alpha_list[i])

    np.savetxt('0_temp_det_dd_beta_1_m_'+str(coupling),\
                np.vstack((detuning_beta_1,intesity_of_first_peak_sf[coupling]\
                            /max(intesity_of_first_peak_sf[coupling]))).T)


plt.xlabel('Detuning, (eV)', fontsize=15)

plt.text(0.4,0.3,'$M='+str(max(coupling_list))+' meV$')

plt.text(0.2,0.15,'$M='+str(min(coupling_list))+' meV$')

plt.ylabel(r'$I_1(Z)/I_1(Z=0)$', fontsize=15)

plt.legend()

plt.show()

# ws.plotp()
ws.clear()
