'''
Plot 2D posterior maps as a function of <xHI> and logfX for multiple combinations of these parameters.

Version 14.02.2024
'''

import sys
import glob, os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import AutoMinorLocator
from matplotlib import gridspec
import corner

path = 'MCMC_samples'
spec_res = 8
S147 = 64.2
alphaR = -0.44
Nsteps = 20000

telescope = 'uGMRT'
tint = 50
#xHI_mean = [0.11,0.25,0.25,0.25,0.25,0.39,0.39,0.39,0.52,0.52,0.52,0.52,0.52]
#logfX =    [-3.8,-3.8,-3.0,-2.4,-2.0,-3.8,-3.0,-2.4,-3.8,-3.0,-2.4,-2.0,-1.6]
xHI_mean = [0.11,0.25,0.25,0.39,0.39,0.52,0.52]
logfX =    [-3.8,-3.0,-2.0,-3.8,-2.4,-3.0,-1.6]

'''
telescope = 'SKA1-low'
tint = 50
xHI_mean = [0.11,0.25,0.25,0.25,0.25,0.39,0.39,0.39,0.52,0.52,0.52,0.52,0.52,0.11,0.39,0.39,0.52,0.11]
logfX =    [-3.8,-3.8,-3.0,-2.4,-2.0,-3.8,-3.0,-2.4,-3.8,-3.0,-2.4,-2.0,-1.2,-2.0,-2.0,-1.6,-1.6,-2.4]
'''
fsize = 20
colours  = ['royalblue','fuchsia','forestgreen','darkorange','red','grey','darkviolet','cyan','gold','lightcoral','brown','slateblue','limegreen','magenta','crimson','teal','orangered','navy']
fig = plt.figure(figsize=(5.,5.))
gs = gridspec.GridSpec(1,1)
ax0 = plt.subplot(gs[0,0])

for i in range(len(logfX)):
    data = np.load('%s/flatsamp_xHI%.2f_fX%.1f_%s_%dkHz_Smin%.1fmJy_alphaR%.2f_t%dh_%dsteps.npy' % (path,xHI_mean[i],logfX[i],telescope,spec_res,S147,alphaR,tint,Nsteps))
    corner.hist2d(data[:,0],data[:,1],levels=[1-np.exp(-0.5),1-np.exp(-2.)],plot_datapoints=False,plot_density=False,color=colours[i])

for i in range(len(logfX)):
    plt.scatter(xHI_mean[i],logfX[i],marker='x',s=200,linewidths=2.5,color=colours[i])

ax0.set_xticks(np.arange(0.,0.65,0.1))
ax0.set_yticks(np.arange(-4.,-0.9,0.5))
ax0.set_xlim(0.,0.65)
ax0.set_ylim(-4,-0.8)
ax0.set_xlabel(r'$\langle x_{\rm HI}\rangle$', fontsize=fsize)
ax0.set_ylabel(r'$\log_{10}(f_{\mathrm{X}})$', fontsize=fsize)
ax0.tick_params(axis='x',which='major',direction='in',bottom=True,top=True,left=True,right=True
		,length=10,width=1,labelsize=fsize)
ax0.tick_params(axis='y',which='major',direction='in',bottom=True,top=True,left=True,right=True
		,length=10,width=1,labelsize=fsize)
ax0.tick_params(axis='both',which='minor',direction='in',bottom=True,top=True,left=True,right=True
		,length=5,width=1)

plt.tight_layout()
plt.savefig('MCMC_samples/multiparam_infer_%s_%dhr.png' % (telescope,tint))
plt.show()