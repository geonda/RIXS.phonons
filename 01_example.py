import phlab
from  matplotlib import pyplot as plt

ws=phlab.rixs()

model = ws.model_gq_phonons_2d( name = 'gq')
model_single = ws.model_single_osc(name = '1d')

# model.color = 'r'
model.input['nf'] = 6
model.input['nq'] = 7
model.input['r'] = 0.1
model.input['m_gamma'] = 0.2
model.input['m_k'] = 0.1
model.input['maxt'] = 200
model.input['nstep'] = 4000
model.input['gamma_ph'] = 0.02
model.input["energy_ex"] = 5
model.input["omega_in"] = 5
model.color='r'

model_single.input['nf'] = 6

model_single.input['nm'] = 40
model_single.input['gamma_ph'] = 0.02
model_single.input['coupling'] = model.input['m_gamma']

model.run()
model_single.run()

exp = ws.experiment(file='292.2.txt',name = '292.2',col=[0,1])

plt.figure(figsize = (10,5))

vitem=ws.visual(model_list=[model,model_single],exp=exp)
plt.xlim([-0.1,1.])
vitem.show(scale = 0)

# vitem.show_xas()
