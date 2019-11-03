from lib import *
from tqdm import tqdm
ws=workspace()
ws.timer_start()

ws.initp()
ws.timer_round('init done [s]: ')
ws.inputp('skip')
ws.timer_round('input done [s]: ')
list_coupling=[0.06,0.06]
list_omega=[0.05,0.1]

for omega,coupling in zip(list_omega,list_coupling):
    ws.dict_total['input']['omega_ph0']=float(omega)
    ws.dict_total['input']['coupling0']=float(coupling)
    print(ws.dict_total)
    ws.runp()
ws.initp(vib_space=2)

ws.inputp('skip')
for i,set in enumerate(zip(list_omega,list_coupling)):
    print(set)
    ws.dict_total['input']['coupling'+str(i)]=float(set[1])
    ws.dict_total['input']['omega_ph'+str(i)]=float(set[0])
    ws.dict_total['input']['nm']=int(20)
    print(ws.dict_total)
ws.runp()

ws.timer_round('run done [s]: ')
ws.timer_total('total [s]: ')
ws.plotp()
ws.clear()
