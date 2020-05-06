import sys
sys.path.append("./_src/")
from lib import *
from tqdm import tqdm
from phonon_info_2d import *

ws = workspace()
ws.timer_start()

# overwrite init problem
ws.initp(type_problem = 'rixs_q', method = 'gf')
ws.inputp('skip')

# overwrite input
ws.dict_total['input']['extra'] = 'no_xas'
ws.dict_total['input']['omega_in'] = ws.dict_total['input']['energy_ex']
ws.dict_total['input']['gamma'] = 0.105
ws.dict_total['input']['nf'] = 5 #
# run
ws.runp()

# visual
ws.figure_q_and_exp('291.8')

ws.timer_total('total [s]: ')
