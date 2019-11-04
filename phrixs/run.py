from lib import *
from tqdm import tqdm
ws=workspace()
ws.timer_start()
ws.initp(type_calc='model')#ws.initp(vib_space=2)
ws.timer_round('init done [s]: ')
ws.inputp('ask')
ws.timer_round('input done [s]: ')
ws.runp()
ws.timer_round('run done [s]: ')
ws.timer_total('total [s]: ')
ws.plotp()
ws.clear()
