from lib import *
from tqdm import tqdm
from phonon_info_2d import *
import json
from lmfit import Minimizer, Parameters, report_fit,Model
from matplotlib import pyplot as plt

ws = workspace()
ws.timer_start()
ws.initp(type_problem = 'rixs_q', method = 'gf')#ws.initp(vib_space=2)
ws.inputp('skip')

ws.dict_total['input']['extra']='x'
ws.dict_total['input']['extra_1']='fit'

xexp,yexp = np.loadtxt('292')

def window(x,y,xmin,xmax):
    x_new=[];y_new=[]
    for xi,yi in zip(x,y):
        if xi<=xmax and xi>=xmin:
            x_new.append(xi)
            y_new.append(yi)
    return np.array(x_new), np.array(y_new)

xexp,yexp = window(xexp,yexp,0.1,0.8)

iter=0

def fcn2min(params,xexp,yexp,iter):
    ti={}
    for i in ['ag','ak','am','rg','rk','rm']:
        ti[i]= params[i].value
    iter+=1

    print('########################### N interations = ', iter)
    with open('temp_input.json', 'w') as fp:
        json.dump(ti,fp)

    ws.runp(x=xexp)

    xfit,yfit = np.load('temp_2_run_'+str(ws.nruns)+'.npy')
    resid = abs(yfit/max(yfit)-yexp/max(yexp))
    ws.timer_round('run done [s]: ')
    plt.plot(xfit,xfit-xexp)
    plt.plot(xfit,resid)
    plt.plot(xfit,yfit/max(yfit))
    plt.plot(xfit,yexp/max(yexp))
    # plt.show()
    return resid

params = Parameters()

with open('temp_input.json') as fp:
    ti_temp=json.load(fp)

params.add('ag', ti_temp['ag'], min = 0.1, max=1.,vary=True)
params.add('ak', ti_temp['ak'], min = 0.1, max=1.,vary=True)
params.add('am', ti_temp['am'], min = 0.0, max=1.,vary=False)

params.add('rg', 0.2, min = 0.05, max=0.5, vary=False)
params.add('rk', 0.2, min = 0.05, max=0.5, vary=False)
params.add('rm', ti_temp['rm'], min = 0.05, max=0.5,vary=False)

minner = Minimizer(fcn2min, params,fcn_args=(xexp,yexp,iter))

result = minner.minimize(method='brute')

report_fit(result)
plt.show()
ws.timer_total('total [s]: ')
