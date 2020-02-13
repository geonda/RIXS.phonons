import numpy as np
import config as cg
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid.inset_locator import inset_axes
data_bet_1=\
        np.load('temp_2_run_1.npy')
data_bet_small=\
        np.load('temp_2_run_2.npy')
data_bet_big=\
        np.load('temp_2_run_3.npy')


fig = plt.figure(figsize=(9, 4),facecolor='white')

ax = fig.add_subplot(121)

plt.plot(data_bet_1[0],data_bet_1[1],linewidth=2,color='blue')

# plt.axis([0, 1, 1.1*np.amin(s), 2*np.amax(s)])

plt.xlabel('$\omega_{loss}/omega_{gs}$')

plt.ylabel('r$\mathrm{RIXS \ Intensity,\ arb. \ units }$')

# this is an inset axes over the main axes
inset_axes = inset_axes(ax,
                    width="50%", # width = 30% of parent_bbox
                    height=1.0, # height : 1 inch
                    loc=1)
plt.plot(data_bet_1[0],data_bet_1[1],linewidth=2,color='blue')
#plt.title('Probability')
plt.xticks([])
plt.yticks([])
plt.tight_layout()
plt.show()
