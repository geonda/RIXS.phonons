import phlab
from  matplotlib import pyplot as plt

ws=phlab.rixs()


model1 = ws.model_single_osc(name = 'first mode')
model2 = ws.model_single_osc(name = 'second mode')
model3 = ws.model_double_osc( name= '2d')

# input key parameters
model1.input['coupling'] = 0.09
model1.input['omega_ph'] = 0.03
model1.input['gamma_ph'] = 0.001

model2.input['coupling'] = 0.1
model2.input['omega_ph'] = 0.08
model2.input['gamma_ph'] = 0.001

model3.input['coupling0'] = 0.09
model3.input['omega_ph0'] = 0.03
model3.input['coupling1'] = 0.1
model3.input['omega_ph1'] = 0.08
model3.input['nm'] = 15
model3.input['gamma'] = 0.105
model3.input['gamma_ph'] = 0.001

for model in [model1,model2,model3]:
    model.run()

model3.color = 'r'

plt.figure(figsize = (10,5))

vitem=ws.visual(model_list=[model3],exp=[])

plt.plot(model1.x, (model1.y)/max(model1.y+model2.y),
                 color = 'skyblue',
                 linewidth = 2,
                 label = 'model1',alpha = 1)
plt.plot(model2.x, (model2.y)/max(model1.y+model2.y),
                 color = 'lightpink',
                 linewidth = 2,
                 label = 'model2',alpha = 1)

plt.plot(model1.x, (model1.y+model2.y)/max(model1.y+model2.y),
                color = 'b',
                linewidth = 2,
                label = 'model1 + model2')
plt.xlim([-0.1,0.6])

vitem.show(scale = 0)
