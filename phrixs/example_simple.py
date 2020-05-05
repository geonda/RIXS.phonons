from lib import *
from tqdm import tqdm
ws=workspace()
ws.timer_start()
ws.initp(type_calc='model', method = 'gf')#ws.initp(vib_space=2)
ws.timer_round('init done [s]: ')
ws.inputp('skip') # use ws.inputp('ask') to modify
ws.timer_round('input done [s]: ')



ws.dict_total['input']['nf']=11

ws.dict_total['input']['gamma']=0.105

for coupling in [0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45]:
    ws.dict_total['input']['coupling0']=coupling
    ws.runp()

ws.plotp(scale=10**5)
