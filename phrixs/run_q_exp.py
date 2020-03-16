from lib import *
from tqdm import tqdm
from phonon_info_2d import *
ws=workspace()
ws.timer_start()
ws.initp(type_problem='rixs_q',method='gf')#ws.initp(vib_space=2)

# energy().plot_dispersion()
full_data().plot_dispersion()

ws.timer_round('init done [s]: ')
ws.inputp('skip') # use ws.inputp('ask') to modify

ws.timer_round('input done [s]: ')

ws.timer_round('run done [s]: ')
ws.timer_total('total [s]: ')
ws.dict_total['input']['extra']='x'
# ws.initp(type_problem='model,method='gf')
# ws.plotp_model_and_exp('../../storage/rixs_c.csv')
<<<<<<< HEAD
# ws.runp()
# # ws.plotp_model_and_exp('292')
# ws.dict_total['input']['omega_in']=ws.dict_total['input']['energy_ex']-0.2
# ws.runp()
# ws.plotp_model_and_exp('291.8')
=======

ws.dict_total['input']['omega_in']=ws.dict_total['input']['energy_ex']

ws.runp()


ws.plotp_model_and_exp('292')

#
# ws.dict_total['input']['omega_in']=ws.dict_total['input']['energy_ex']
# ws.runp()
# ws.plotp_model_and_exp('292')
#
# ws.dict_total['input']['omega_in']=ws.dict_total['input']['energy_ex']-0.2
# ws.runp()
# ws.plotp_model_and_exp('292.2')
# # ws.dict_total['input']['omega_in']=ws.dict_total['input']['energy_ex']
# ws.runp()
# ws.plotp(scale=0)

# ws.plotp_model_and_exp('292')
>>>>>>> 67c0fe861b3a07eef6051ba8936ee6f4b217a14c
# ws.plot_model_exp()
# ws.clear()
# ws.plotxasp()
