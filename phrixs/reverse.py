import numpy as np
data=np.loadtxt('../../storage/TO/gamma-m.csv')
np.savetxt('../../storage/TO/gamma-m_r.csv', np.vstack((1-data.T[0],data.T[1])).T)
