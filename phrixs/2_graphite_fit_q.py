import sys
sys.path.append("./_src/")
from lib import *
from tqdm import tqdm
from phonon_info_2d import *
import json
from lmfit import Minimizer, Parameters, report_fit,Model
from matplotlib import pyplot as plt


# init problem
ws = workspace()
ws.timer_start()
ws.initp(type_problem = 'rixs_q', method = 'gf')#ws.initp(vib_space=2)
ws.inputp('skip')

# overwrite input

ws.dict_total['input']['nf'] = 4
ws.dict_total['input']['gamma'] = 0.105
ws.dict_total['input']['omega_in'] = ws.dict_total['input']['energy_ex']

ws.dict_total['input']['extra']='no_xas'
ws.dict_total['input']['extra_1']='fit'

exp_file_name = './_exp_files/291.8'

fit_xmin = 0.1
fit_xmax = 0.8

# fitting part

xexp,yexp = np.loadtxt(exp_file_name)

def window(x,y,xmin,xmax):
    x_new=[];y_new=[]
    for xi,yi in zip(x,y):
        if xi<=xmax and xi>=xmin:
            x_new.append(xi)
            y_new.append(yi)
    return np.array(x_new), np.array(y_new)
xexp,yexp = window(xexp,yexp,fit_xmin,fit_xmax)

def fcn2min(params,xexp,yexp):
    ti={}
    for i in ['ag','ak','am','rg','rk','rm']:
        ti[i]= params[i].value

    print('########################### N interations = ', iter)
    with open('./inputs/input_phonon_info.json', 'w') as fp:
        json.dump(ti,fp,indent=1)

    ws.runp(x=xexp)

    xfit,yfit = np.load('./_out/temp_2_run_'+str(ws.nruns)+'.npy')
    resid = abs(yfit/max(yfit)-yexp/max(yexp))
    ws.timer_round('run done [s]: ')
    plt.plot(xfit,xfit-xexp)
    plt.plot(xfit,resid)
    plt.plot(xfit,yfit/max(yfit))
    plt.plot(xfit,yexp/max(yexp))
    # plt.show()
    return resid


with open('./inputs/input_phonon_info.json') as fp:
    ti_temp=json.load(fp)

params = Parameters()

params.add('ag', ti_temp['ag'], min = 0.2, max=0.5,vary=True)
params.add('ak', ti_temp['ak'], min = 0.6, max=0.8,vary=True)
params.add('am', ti_temp['am'], min = 0.0, max=1.,vary=False)
# params['ag'].set(brute_step=0.05)
# params['ak'].set(brute_step=0.05)
params.add('rg', 0.2, min = 0.05, max=0.5, vary=False)
params.add('rk', 0.2, min = 0.05, max=0.5, vary=False)
params.add('rm', ti_temp['rm'], min = 0.05, max=0.5,vary=False)

minner = Minimizer(fcn2min, params,fcn_args=(xexp,yexp))
result = minner.minimize(method='brute')

report_fit(result)
ws.clear()
ws.timer_total('total [s]: ')
