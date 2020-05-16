import phlab
from  matplotlib import pyplot as plt

ws=phlab.rixs()

model = ws.model_single_osc(name = '1d')
exp = ws.experiment(file='test_data.csv',name = 'test data')

## fit model 2 experiment
model.param2fit.add(name = 'coupling', ivalue = model.input['coupling'] , range=[0.05,0.3])
model.fit(experiment = exp)

print(model.fit_report)

model.run()

plt.figure(figsize = (10,5))

vitem=ws.visual(model_list=[model],exp=exp)
vitem.show(scale = 0)
