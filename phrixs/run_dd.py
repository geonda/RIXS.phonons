from lib import *
from tqdm import tqdm
ws=workspace()
ws.timer_start()

ws.initp()
ws.timer_round('init done [s]: ')
ws.inputp('skip')
ws.timer_round('input done [s]: ')
ws.dict_total['input']['nm']=int(15)
ws.runp()
ws.initp(type_calc='dd')
ws.inputp('skip')
list_omega=[0.02,0.15]

for omega in list_omega:
    ws.dict_total['input']['omega_ph_ex']=float(omega)
    ws.dict_total['input']['nm']=int(15)
    print(ws.dict_total)
    ws.runp()

ws.timer_round('run done [s]: ')
ws.timer_total('total [s]: ')
ws.figure_dd()
# ws.clear()
