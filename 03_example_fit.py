import phlab
from  matplotlib import pyplot as plt

ws=phlab.rixs()

model = ws.model_single_osc(name = '1d')
exp = ws.experiment(file='fake_exp.csv',name = 'exp test')

model.input['omega_ph0'] = 0.195
model.input['gamma_ph'] = 0.05
## fit model 2 experiment
model.param2fit.add(name = 'coupling0', ivalue = model.input['coupling0'] , range=[0.05,0.3])
model.fit(experiment = exp)

print(model.fit_report)

model.run()

plt.figure(figsize = (10,5))

vitem=ws.visual(model_list=[model],exp=exp)
vitem.show(scale = 0)
