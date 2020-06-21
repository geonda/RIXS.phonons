# fitting

import phlab
from  matplotlib import pyplot as plt

ws=phlab.rixs()

model = ws.model_gq_phonons_2d( name = 'gq')

model.color = 'r'
model.input['nf'] = 6

exp = ws.experiment(file = 'test_data.csv',name = 'test data')


model.run()

plt.figure(figsize = (10,5))
vitem=ws.visual(model_list = [model],exp = exp)
vitem.show(scale = 0)



## fit model 2 experiment
model.param2fit.add(name = 'm_gamma', ivalue = model.input['m_gamma'] , range=[0.1,0.2])
model.fit(experiment = exp)

print(model.fit_report)

model.run()

plt.figure(figsize = (10,5))

vitem=ws.visual(model_list=[model],exp=exp)
vitem.show(scale = 0)