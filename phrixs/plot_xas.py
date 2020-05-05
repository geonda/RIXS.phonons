from lib import *

from tqdm import tqdm
from phonon_info_2d import *



ws=workspace()
ws.timer_start()

ws.initp(type_problem='rixs_q',method='gf')
ws.inputp('skip')

ws.dict_total['input']['extra']='xas'
ws.dict_total['input']['omega_in']=ws.dict_total['input']['energy_ex']
ws.runp()
ws.plotxasp()
