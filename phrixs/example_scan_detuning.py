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
ws.dict_total['input']['nm']=10

for dE in [0.,0.1,0.2]:
    ws.dict_total['input']['omega_in']=ws.dict_total['input']['energy_ex']+dE
    ws.runp()

ws.timer_total('total [s]: ')
ws.plotp()
ws.clear()
