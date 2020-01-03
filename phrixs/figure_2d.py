from lib import *
from tqdm import tqdm
ws=workspace()
ws.timer_start()

#

################# M2>M1
# two 1d osc
ws.initp(type_calc='2d')
ws.timer_round('init done [s]: ')
ws.inputp('skip') # use ws.inputp('ask') to modify input
ws.timer_round('input done [s]: ')
list_coupling=[0.08,0.05]
list_omega=[0.065,0.1]

for omega,coupling in zip(list_omega,list_coupling):
    ws.dict_total['input']['omega_ph0']=float(omega)
    ws.dict_total['input']['coupling0']=float(coupling)
    ws.dict_total['input']['nm']=int(20)
    print(ws.dict_total)
    ws.runp()

# one 2d osc
ws.initp(type_calc='2d',vib_space=2)

ws.inputp('skip') # use ws.inputp('ask') to modify input
for i,set in enumerate(zip(list_omega,list_coupling)):
    print(set)
    ws.dict_total['input']['coupling'+str(i)]=float(set[1])
    ws.dict_total['input']['omega_ph'+str(i)]=float(set[0])
    ws.dict_total['input']['nm']=int(20)
    print(ws.dict_total)
ws.runp()

ws.timer_round('run done [s]: ')
ws.timer_total('total [s]: ')
ws.figure_2d()


################# M2=M1
ws.initp(type_calc='2d')
ws.timer_round('init done [s]: ')
ws.inputp('skip') # use ws.inputp('ask') to modify input
ws.timer_round('input done [s]: ')
list_coupling=[0.08,0.08]
list_omega=[0.065,0.1]

for omega,coupling in zip(list_omega,list_coupling):
    ws.dict_total['input']['omega_ph0']=float(omega)
    ws.dict_total['input']['coupling0']=float(coupling)
    ws.dict_total['input']['nm']=int(20)
    print(ws.dict_total)
    ws.runp()

# one 2d osc
ws.initp(type_calc='2d',vib_space=2)

ws.inputp('skip') # use ws.inputp('ask') to modify input
for i,set in enumerate(zip(list_omega,list_coupling)):
    print(set)
    ws.dict_total['input']['coupling'+str(i)]=float(set[1])
    ws.dict_total['input']['omega_ph'+str(i)]=float(set[0])
    ws.dict_total['input']['nm']=int(20)
    print(ws.dict_total)
ws.runp()

ws.timer_round('run done [s]: ')
ws.timer_total('total [s]: ')
ws.figure_2d()
# ws.clear()

################# M2<M1
ws.initp(type_calc='2d')
ws.timer_round('init done [s]: ')
ws.inputp('skip') # use ws.inputp('ask') to modify input
ws.timer_round('input done [s]: ')
list_coupling=[0.05,0.08]
list_omega=[0.065,0.1]

for omega,coupling in zip(list_omega,list_coupling):
    ws.dict_total['input']['omega_ph0']=float(omega)
    ws.dict_total['input']['coupling0']=float(coupling)
    ws.dict_total['input']['nm']=int(20)
    print(ws.dict_total)
    ws.runp()

# one 2d osc
ws.initp(type_calc='2d',vib_space=2)

ws.inputp('skip') # use ws.inputp('ask') to modify input
for i,set in enumerate(zip(list_omega,list_coupling)):
    print(set)
    ws.dict_total['input']['coupling'+str(i)]=float(set[1])
    ws.dict_total['input']['omega_ph'+str(i)]=float(set[0])
    ws.dict_total['input']['nm']=int(20)
    print(ws.dict_total)
ws.runp()
ws.timer_round('run done [s]: ')
ws.timer_total('total [s]: ')
ws.figure_2d_x()

ws.clear()
