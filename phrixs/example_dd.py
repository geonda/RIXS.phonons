from lib import *

ws = workspace()
ws.timer_start()

#displaced osc
ws.initp(type_calc='model')

ws.inputp('skip') # use ws.inputp('ask') to modify input
ws.dict_total['input']['nf']=10
ws.dict_total['input']['nm']=10
ws.dict_total['input']['coupling0']=0.15
ws.dict_total['input']['omega_ph0']=0.1
ws.dict_total['input']['gamma']=0.200
ws.dict_total['input']['gamma_ph']=ws.dict_total['input']['omega_ph0']/10
ws.dict_total['input']['alpha_exp']=ws.dict_total['input']['gamma_ph']
ws.runp()

ws.initp(type_calc='dd')
ws.inputp('skip') # use ws.inputp('ask') to modify input or call ws.dict_total['input']['parameter_name']
beta_list=[0.9,1.1]

for beta in beta_list:
    ws.dict_total['input']['omega_ph0']=0.1
    ws.dict_total['input']['coupling0']=0.15
    ws.dict_total['input']['gamma']=0.200
    ws.dict_total['input']['gamma_ph']=ws.dict_total['input']['omega_ph0']/10
    ws.dict_total['input']['alpha_exp']=ws.dict_total['input']['gamma_ph']
    ws.dict_total['input']['omega_ph_ex']=ws.dict_total['input']['omega_ph0']*beta*beta
    ws.dict_total['input']['nm']=7
    ws.dict_total['input']['nf']=7
    ws.runp()

ws.timer_total('total [s]: ')
ws.plotp(scale=0)
# ws.clear()
