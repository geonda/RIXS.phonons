import phlab
from  matplotlib import pyplot as plt

ws=phlab.rixs()

model1 = ws.create_model()
model1.run()

exp = ws.create_experiment(file='fake_exp.csv',name = 'exp test')

plt.figure(figsize = (10,5))

vitem=ws.connect_visual(model_list=[model1],exp=exp)
vitem.show(scale = 0)
