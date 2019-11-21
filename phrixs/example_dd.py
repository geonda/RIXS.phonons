from lib import *

ws = workspace()
ws.timer_start()

#displaced osc
ws.initp()
ws.inputp('ask') # ws.inputp('skip')
ws.runp()

#distorted and displaced osc
ws.initp(type_calc='dd')
ws.inputp('ask')

#list_omega=[0.02,0.15]
#for omega in list_omega:
#    ws.dict_total['input']['omega_ph_ex']=omega
ws.runp()

ws.timer_total('total [s]: ')
ws.plotp()
# ws.clear()
