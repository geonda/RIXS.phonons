# Displaced and Distorted Harmonic Oscillator

import phlab
from  matplotlib import pyplot as plt

ws=phlab.rixs()


model1 = ws.model_single_osc(name = 'displaced osc')
model2 = ws.model_dist_disp_osc(name = 'distorted and displaced osc')

model2.input['omega_ph_ex'] = 0.1

for model in [model1,model2]:
    model.run()

model2.color = 'b'

plt.figure(figsize = (10,5))

vitem=ws.visual(model_list=[model1,model2],exp=[])


vitem.show(scale = 1)
