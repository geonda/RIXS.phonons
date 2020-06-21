import phlab
from  matplotlib import pyplot as plt

ws=phlab.rixs()

model = ws.model_gq_phonons_2d( name = 'gq')
model_single = ws.model_single_osc(name = '1d')

# model.color = 'r'
model.input['nf'] = 6
model.input['r'] = 1.5
model.input['m_gamma'] = 0.15
model.input['m_k'] = 0.
model.input['maxt'] = 200
model.input['nstep'] = 4000

model.input["energy_ex"] = 5
model.input["omega_in"] = 5
model.color='r'

model_single.input['nf'] = 6

model_single.input['nm'] = 40
model_single.input['coupling'] = 0.15

model.run()
model_single.run()

exp = ws.experiment(file='test_data.csv',name = 'test data')

plt.figure(figsize = (10,5))

vitem=ws.visual(model_list=[model,model_single],exp=exp)
#
vitem.show(scale = 0)

vitem.show_xas()
