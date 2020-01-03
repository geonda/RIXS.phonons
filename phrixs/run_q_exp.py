from lib import *
from tqdm import tqdm
ws=workspace()
ws.timer_start()
ws.initp(type_problem='rixs_q',method='gf')#ws.initp(vib_space=2)
ws.timer_round('init done [s]: ')
ws.inputp('skip') # use ws.inputp('ask') to modify
ws.timer_round('input done [s]: ')
ws.runp()
ws.timer_round('run done [s]: ')
ws.timer_total('total [s]: ')
ws.figure_q_and_exp('name_rixs_exp_file.csv')
# ws.plot_model_exp()
# ws.clear()
