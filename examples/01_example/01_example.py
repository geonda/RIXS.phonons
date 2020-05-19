import phlab
from  matplotlib import pyplot as plt

ws=phlab.rixs()

model = ws.model_single_osc(name = '1d')

model.input['coupling'] = 0.15

model.run()

exp = ws.experiment(file='test_data.csv',name = 'test data')

plt.figure(figsize = (10,5))

vitem=ws.visual(model_list=[model],exp=exp)

vitem.show(scale = 0)
