from lib import *
from tqdm import tqdm
ws=workspace()
ws.timer_start()
ws.initp(type_problem='rixs_q',method='gf')#ws.initp(vib_space=2)
ws.timer_round('init done [s]: ')
ws.inputp('skip') # use ws.inputp('ask') and phonon_info.py to modify input
ws.timer_round('input done [s]: ')
ws.runp()
ws.timer_round('run done [s]: ')
ws.timer_total('total [s]: ')
ws.figure_q()
ws.clear()
