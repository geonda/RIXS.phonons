from lib import *

ws = workspace()
ws.timer_start()

#displaced osc
ws.initp(type_calc='model')

ws.inputp('skip') # use ws.inputp('ask') to modify input
ws.dict_total['input']['nf']=10
ws.dict_total['input']['nm']=5
ws.dict_total['input']['coupling0']=0.05
ws.dict_total['input']['omega_ph0']=0.05
ws.runp()

ws.initp(type_calc='dd')
ws.inputp('skip') # use ws.inputp('ask') to modify input or call ws.dict_total['input']['parameter_name']
beta_list=[0.99,1.01]
for beta in beta_list:
    ws.dict_total['input']['omega_ph0']=0.05
    ws.dict_total['input']['omega_ph_ex']=ws.dict_total['input']['omega_ph0']*beta*beta
    ws.dict_total['input']['nm']=10
    ws.dict_total['input']['nf']=10
    ws.dict_total['input']['coupling0']=0.05

    ws.runp()


# ws.initp(type_calc='dd')
# ws.inputp('skip') # use ws.inputp('ask') to modify input or call ws.dict_total['input']['parameter_name']
# ws.dict_total['input']['nm']=5
# # ws.dict_total['input']['omega_ph_ex']=ws.dict_total['input']['omega_ph0']
# ws.dict_total['input']['test']=False
# ws.runp()

ws.timer_total('total [s]: ')
ws.plotp(scale=0)
# ws.clear()
