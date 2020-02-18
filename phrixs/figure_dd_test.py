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



data_bet_big[1]=data_bet_big[1]/10
data_bet_small[1]=data_bet_small[1]/10
data_bet_1[1]=data_bet_1[1]/10

omega_ph=0.1 ########### !



fig = plt.figure(figsize=(5, 6),facecolor='white')

ax = fig.add_subplot(311)

plt.fill_between(data_bet_1[0]/omega_ph,data_bet_1[1]*max(data_bet_big[1])/max(data_bet_1[1]),data_bet_big[1],alpha=0.5,color='grey')

plt.plot(data_bet_1[0]/omega_ph,data_bet_1[1]*max(data_bet_big[1])/max(data_bet_1[1]),linewidth=2,color='grey')

plt.plot(data_bet_big[0]/omega_ph,data_bet_big[1],linewidth=2,color='green')

plt.xlim([-0.5,7])
ax.set_yticks([0,2,4,6])

# plt.axis([0, 1, 1.1*np.amin(s), 2*np.amax(s)])
plt.text(2.5,4.5,r'$\beta=1.1$')


# plt.ylabel('r$\mathrm{RIXS \ Intensity,\ arb. \ units }$')

# this is an inset axes over the main axes
inset_axes_0 = inset_axes(ax,
                    width="35%", # width = 30% of parent_bbox
                    height="60%", # height : 1 inch
                    loc=1)
x0=np.linspace(-1,1,100)
x1=np.linspace(-1,2.5,100)
beta=2
omega=2.
offset=2.2
y0=omega*(x0)**2
y1=omega*(x1-1)**2+offset
y2=(omega*beta**2)*(x1-1)**2+offset
inset_axes_0.plot(x0,y0,linewidth=2,color='grey')
inset_axes_0.plot(x1,y1,linewidth=2,color='grey')
inset_axes_0.plot(x1,y2,linewidth=2,color='orange')
inset_axes_0.fill_between(x1,y1,y2,alpha=0.5,color='grey')
inset_axes_0.axvline(x=0,linestyle='--', color='grey',linewidth=0.5)
inset_axes_0.axvline(x=1,linestyle='--', color='grey',linewidth=0.5)
inset_axes_0.set_xlim([-1,2.5])
inset_axes_0.set_ylim([0,4.2])
inset_axes_0.set_yticks([])
inset_axes_0.set_xticks([])
inset_axes_0.set_xlabel(r'$R,\ (arb. \ units)$',fontsize=7)
inset_axes_0.set_ylabel(r'$\mathrm{PE},\ (arb.\ units)$',fontsize=7)
for side in ['top','right']:
    inset_axes_0.spines[side].set_visible(False)

inset_axes_0.arrow(-1, 0, 3.5, 0., fc='k', ec='k', lw = 0.1,
         head_width=0.2/(3.5), head_length=0.2/(3.5), overhang = 0.,
         length_includes_head= True, clip_on = False)

inset_axes_0.arrow(-1, 0, 0, 4.2, fc='k', ec='k', lw = 0.1,
         head_width=0.2/(4.2), head_length=0.2/(4.2), overhang = 0.,
         length_includes_head= True, clip_on = False)
# plt.plot(data_bet_1[0],data_bet_1[1],linewidth=2,color='blue')
#plt.title('Probability')

plt.xticks([])

plt.yticks([])


############################

ax_big = fig.add_subplot(312)

plt.plot(data_bet_1[0]/omega_ph,data_bet_1[1],linewidth=2,color='blue')

plt.text(2.5,4,r'$\beta=1$')

plt.xlim([-0.5,7])

# plt.axis([0, 1, 1.1*np.amin(s), 2*np.amax(s)])


plt.ylabel(r'$\mathrm{RIXS \ Intensity,\ (arb. \ units) }$',fontsize=15)

inset_axes_1 = inset_axes(ax_big,width="35%", height="60%",loc=1)

x0=np.linspace(-1,1,100)
x1=np.linspace(-1,2.5,100)
beta=1
omega=2.
offset=2.2
y0=omega*(x0)**2
y1=omega*(x1-1)**2+offset
y2=(omega*beta**2)*(x1-1)**2+offset
inset_axes_1.plot(x0,y0,linewidth=2,color='grey')
inset_axes_1.plot(x1,y1,linewidth=2,color='grey')
inset_axes_1.plot(x1,y2,linewidth=2,color='orange')
inset_axes_1.fill_between(x1,y1,y2,alpha=0.5,color='grey')
inset_axes_1.axvline(x=0,linestyle='--', color='grey',linewidth=0.5)
inset_axes_1.axvline(x=1,linestyle='--', color='grey',linewidth=0.5)
inset_axes_1.set_xlim([-1,2.5])
inset_axes_1.set_ylim([0,4.2])
inset_axes_1.set_yticks([])
inset_axes_1.set_xticks([])
inset_axes_1.set_xlabel(r'$R,\ (arb. \ units)$',fontsize=7)
inset_axes_1.set_ylabel(r'$\mathrm{PE},\ (arb.\ units)$',fontsize=7)
for side in ['top','right']:
    inset_axes_1.spines[side].set_visible(False)

inset_axes_1.arrow(-1, 0, 3.5, 0., fc='k', ec='k', lw = 0.1,
         head_width=0.2/(3.5), head_length=0.2/(3.5), overhang = 0.,
         length_includes_head= True, clip_on = False)

inset_axes_1.arrow(-1, 0, 0, 4.2, fc='k', ec='k', lw = 0.1,
         head_width=0.2/(4.2), head_length=0.2/(4.2), overhang = 0.,
         length_includes_head= True, clip_on = False)
# plt.plot(data_bet_1[0],data_bet_1[1],linewidth=2,color='blue')
#plt.title('Probability')

plt.xticks([])

plt.yticks([])


ax_small = fig.add_subplot(313)

plt.text(2.5,3.5,r'$\beta=0.9$')

plt.fill_between(data_bet_1[0]/omega_ph,data_bet_1[1]*max(data_bet_small[1])/max(data_bet_1[1]),data_bet_small[1],alpha=0.5,color='grey')

plt.plot(data_bet_1[0]/omega_ph,data_bet_1[1]*max(data_bet_small[1])/max(data_bet_1[1]),linewidth=2,color='grey')

plt.plot(data_bet_big[0]/omega_ph,data_bet_small[1],linewidth=2,color='red')

# plt.axis([0, 1, 1.1*np.amin(s), 2*np.amax(s)])

plt.xlabel('$\omega_{loss}/\omega_{0}$',fontsize=15)
plt.xlim([-0.5,7])
# plt.ylabel('r$\mathrm{RIXS \ Intensity,\ arb. \ units }$')

# this is an inset axes over the main axes
inset_axes_2 = inset_axes(ax_small,
                    width="35%", # width = 30% of parent_bbox
                    height="60%", # height : 1 inch
                    loc=1)
x0=np.linspace(-1,1,100)
x1=np.linspace(-1,2.5,100)
beta=0.75
omega=2.
offset=2.2
y0=omega*(x0)**2
y1=omega*(x1-1)**2+offset
y2=(omega*beta**2)*(x1-1)**2+offset
inset_axes_2.plot(x0,y0,linewidth=2,color='grey')
inset_axes_2.plot(x1,y1,linewidth=2,color='grey')
inset_axes_2.plot(x1,y2,linewidth=2,color='orange')
inset_axes_2.fill_between(x1,y1,y2,alpha=0.5,color='grey')
inset_axes_2.axvline(x=0,linestyle='--', color='grey',linewidth=0.5)
inset_axes_2.axvline(x=1,linestyle='--', color='grey',linewidth=0.5)
inset_axes_2.set_xlim([-1,2.5])
inset_axes_2.set_ylim([0,4.2])
inset_axes_2.set_yticks([])
inset_axes_2.set_xticks([])
inset_axes_2.set_xlabel(r'$R,\ (arb. \ units)$',fontsize=7)
inset_axes_2.set_ylabel(r'$\mathrm{PE},\ (arb.\ units)$',fontsize=7)
for side in ['top','right']:
    inset_axes_2.spines[side].set_visible(False)

inset_axes_2.arrow(-1, 0, 3.5, 0., fc='k', ec='k', lw = 0.1,
         head_width=0.2/(3.5), head_length=0.2/(3.5), overhang = 0.,
         length_includes_head= True, clip_on = False)

inset_axes_2.arrow(-1, 0, 0, 4.2, fc='k', ec='k', lw = 0.1,
         head_width=0.2/(4.2), head_length=0.2/(4.2), overhang = 0.,
         length_includes_head= True, clip_on = False)
# plt.plot(data_bet_1[0],data_bet_1[1],linewidth=2,color='blue')
# #plt.title('Probability')
plt.xticks([])
plt.yticks([])
plt.savefig('./fig_4_dd.eps', format='eps')
# fig.tight_layout()
plt.show()

#
