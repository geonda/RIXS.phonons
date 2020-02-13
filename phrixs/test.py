from mpl_toolkits.axes_grid.inset_locator import inset_axes
import matplotlib.pyplot as plt
import numpy as np

# create some data to use for the plot
dt = 0.001
t = np.arange(0.0, 10.0, dt)
r = np.exp(-t[:1000]/0.05)               # impulse response
x = np.random.randn(len(t))
s = np.convolve(x, r)[:len(x)]*dt  # colored noise

fig = plt.figure(figsize=(9, 4),facecolor='white')
ax = fig.add_subplot(121)
# the main axes is subplot(111) by default
plt.plot(t, s)
plt.axis([0, 1, 1.1*np.amin(s), 2*np.amax(s)])
plt.xlabel('time (s)')
plt.ylabel('current (nA)')
plt.title('Subplot 1: \n Gaussian colored noise')

# this is an inset axes over the main axes
inset_axes = inset_axes(ax,
                    width="50%", # width = 30% of parent_bbox
                    height=1.0, # height : 1 inch
                    loc=1)
n, bins, patches = plt.hist(s, 400, normed=1)
#plt.title('Probability')
plt.xticks([])
plt.yticks([])

ax = fig.add_subplot(122)
# the main axes is subplot(111) by default
plt.plot(t, s)
plt.axis([0, 1, 1.1*np.amin(s), 2*np.amax(s)])
plt.xlabel('time (s)')
plt.ylabel('current (nA)')
plt.title('Subplot 2: \n Gaussian colored noise')

plt.tight_layout()
plt.show()
