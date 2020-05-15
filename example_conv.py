import phlab
from  matplotlib import pyplot as plt

ws=phlab.rixs()

model = ws.model_single_osc(name = '1d')
model.input['coupling0'] = 0.2
model.converge(parameter = 'nm', pmin = 0, pmax=10,steps=11)



plt.figure(figsize = (10,5))
plt.plot(model.param_space,model.conv_arr,'-o')
plt.show()
