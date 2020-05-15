import phlab
from  matplotlib import pyplot as plt

ws=phlab.rixs()

model = ws.model_single_osc(name = '1d')
model.input['coupling0'] = 0.2
plt.figure(figsize = (10,5))
for nm in [9,10]:
    model.input['nm'] = nm
    model.run()
    plt.plot(model.x,model.y)
    # plt.plot(model.param_space,model.conv_arr,'-o')

plt.show()
