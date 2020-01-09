from lib import *
from tqdm import tqdm
ws=workspace()
ws.timer_start()
ws.initp(type_calc='model')#ws.initp(vib_space=2)
ws.timer_round('init done [s]: ')
ws.inputp('skip') # use ws.inputp('ask') to modify
ws.timer_round('input done [s]: ')
ws.dict_total['input']['coupling0']=0.1
ws.runp()
ws.timer_round('run done [s]: ')
ws.timer_total('total [s]: ')

# ws.plotp_model_and_exp('../../storage/rixs_c.csv')
# ws.clear()
ws.dict_total['input']['coupling0']=0.19
ws.runp()
ws.timer_round('run done [s]: ')
ws.timer_total('total [s]: ')

ws.dict_total['input']['coupling0']=0.3
ws.runp()
ws.timer_round('run done [s]: ')
ws.timer_total('total [s]: ')
ws.plotp()
