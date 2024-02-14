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

logfX =    [-2.0,-2.0,-1.8,-1.6,-2.8,-2.8,-2.8]
xHI_mean = [0.39,0.25,0.25,0.39,0.11,0.25,0.39]

path = 'MCMC_samples'
telescope = 'uGMRT'
spec_res = 8
S147 = 64.2
alphaR = -0.44
tint = 500
Nsteps = 5000



fsize = 20
colours  = ['royalblue','fuchsia','forestgreen','darkorange','red','grey','cyan','darkviolet','gold']
fig = plt.figure(figsize=(10.,10.))
gs = gridspec.GridSpec(1,1)
ax0 = plt.subplot(gs[0,0])

for i in range(len(logfX)):
    data = np.load('%s/flatsamp_xHI%.2f_fX%.1f_%s_%dkHz_Smin%.1fmJy_alphaR%.2f_t%dh_%dsteps.npy' % (path,xHI_mean[i],logfX[i],telescope,spec_res,S147,alphaR,tint,Nsteps))
    corner.hist2d(data[:,0],data[:,1],levels=[1-np.exp(-0.5),1-np.exp(-2.)],plot_datapoints=False,plot_density=False,color=colours[i])

for i in range(len(logfX)):
    plt.scatter(xHI_mean[i],logfX[i],marker='x',s=200,linewidths=5,color=colours[i])

ax0.set_xticks(np.arange(0.,0.51,0.1))
ax0.set_yticks(np.arange(-3.,1.1,0.5))
ax0.set_xlim(0.,0.5)
ax0.set_ylim(-3,1)
ax0.set_xlabel(r'$\langle x_{\rm HI}\rangle$', fontsize=fsize)
ax0.set_ylabel(r'$\log_{10}(f_{\mathrm{X}})$', fontsize=fsize)
ax0.tick_params(axis='x',which='major',direction='in',bottom=True,top=True,left=True,right=True
		,length=10,width=1,labelsize=fsize)
ax0.tick_params(axis='y',which='major',direction='in',bottom=True,top=True,left=True,right=True
		,length=10,width=1,labelsize=fsize)
ax0.tick_params(axis='both',which='minor',direction='in',bottom=True,top=True,left=True,right=True
		,length=5,width=1)

plt.savefig('MCMC_samples/multiparam_infer.png')
plt.show()