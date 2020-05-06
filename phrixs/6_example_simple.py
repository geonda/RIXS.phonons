import sys
sys.path.append("./_src/")
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

for coupling in [0.45,0.85]:
    ws.dict_total['input']['coupling0']=coupling
    ws.runp()

ws.plotp_model_and_exp('291.8',scale=0,names_list=['M=0.45 eV','M=0.85 eV'])
