import phlab
from  matplotlib import pyplot as plt

ws=phlab.rixs()

model1 = ws.model_single_osc(name = '1d')
model2 = ws.model_double_osc(name = '2d')
model3 = ws.model_dist_disp_osc( name= 'dd')

for model in [model1,model2,model3]:
    model.run()

model2.color = 'b'
model3.color = 'g'

exp = ws.experiment(file='fake_exp.csv',name = 'exp test')

plt.figure(figsize = (10,5))

vitem=ws.visual(model_list=[model1,model2,model3],exp=exp)
vitem.show(scale = 0)
