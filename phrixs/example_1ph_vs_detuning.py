from lib import *
from tqdm import tqdm
ws=workspace()
ws.timer_start()
ws.initp(type_calc='model')#ws.initp(vib_space=2)

ws.inputp('skip') # use ws.inputp('ask') to modify
print(ws.dict_total['input'])


ws.dict_total['input']['coupling0']=0.1
ws.dict_total['input']['omega_ph0']=0.1
ws.dict_total['input']['gamma']=0.1
ws.dict_total['input']['energy_ex']=2.
ws.dict_total['input']['omega_in']=2.
ws.dict_total['input']['nm']=20
ws.dict_total['input']['nf']=2    # only first peak is calculted to save time
intesity_of_first_peak=[]
detuning=np.linspace(-1,1,40)     # detuning from -1 ,1 eV, 40 points


for dE in detuning:
    ws.dict_total['input']['omega_in']=ws.dict_total['input']['energy_ex']+dE
    ws.runp() # exectute
    file='temp_0_run_'+str(ws.nruns)+'.npy' # file with peak positions vs intensities (without broadening)
    data_temp=np.load(file)
    intesity_of_first_peak.append(data_temp[1][1]) # first phonon peak intensity

ws.timer_total('total [s]: ')
# save in txt
np.savetxt('1ph_vs_detuning',np.vstack((detuning,intesity_of_first_peak)).T)
# plot in matplotlib
import matplotlib.pyplot as plt
plt.plot(detuning,intesity_of_first_peak)
plt.xlabel('detuning, eV')
plt.ylabel('1 ph intensity, eV')
plt.show()

# ws.plotp()
ws.clear()
