"""
Parameter inference code based on Bayesian methods.

Uses 2D interpolator for 1D PS from 21-cm forest.

Version 8.2.2024

"""

import sys
import glob, os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy.signal as scisi
from astropy.convolution import convolve, Box1DKernel
from matplotlib.ticker import AutoMinorLocator
from matplotlib import gridspec
import openpyxl
import time
from scipy import interpolate
import emcee
import corner

z_name = float(sys.argv[1])
dvH = float(sys.argv[2])
spec_res = float(sys.argv[3])
n_los = 100
telescope = 'uGMRT'
Nlos = 1000

#path_LOS = 'data/los/'
#files = os.listdir(path_LOS+'*file0.dat')
path_LOS = '../../datasets/21cmFAST_los/los/'

files = glob.glob(path_LOS+'*.dat')
logfX = np.empty(len(files))
xHI_mean = np.empty(len(files))

d_log_k_bins = 0.5
log_k_bins = np.arange(0.0-d_log_k_bins/2.,3.+d_log_k_bins/2.,d_log_k_bins)
k_bins = np.power(10.,log_k_bins)
k_bins_cent = np.power(10.,log_k_bins+d_log_k_bins/2.)[:-1]
print(k_bins)
print(k_bins_cent)
PS_signal_sim = np.empty((len(logfX),len(k_bins_cent)))
sig_PS_signal_sim = np.empty((len(logfX),len(k_bins_cent)))
#print(files)
for j in range(len(files)):
    
    data = np.fromfile(str(files[j]),dtype=np.float32)
    logfX[j] = data[9]
    xHI_mean[j] = data[11]
    #print('f_X=%.2f, <x_HI,box>=%.8f' % (logfX[j],xHI_mean[j]))

    datafile = str('1DPS_signal/power_spectrum_signal_21cmFAST_50Mpc_z%.1f_fX%.2f_xHI%.2f_%dkHz_%dLOS.dat' % (z_name,logfX[j],xHI_mean[j],spec_res,Nlos))
    data = np.fromfile(str(datafile),dtype=np.float32)
    Nlos = int(data[0])
    n_kbins = int(data[1])
    k = data[2:2+n_kbins]
    PS_signal = data[2+n_kbins+0*n_kbins*Nlos:2+n_kbins+1*n_kbins*Nlos]
    PS_signal = np.reshape(PS_signal,(Nlos,n_kbins))[:n_los,:]

    PS_signal_bin = np.empty((n_los,len(k_bins_cent)))

    for i in range(n_los):
      for l in range(len(k_bins_cent)):
        ind = np.where((k>=k_bins[l]) & (k<k_bins[l+1]))[0]
        PS_signal_bin[i,l] = np.mean(PS_signal[i,ind])

    PS_signal_sim[j,:] = np.median(PS_signal_bin,axis=0)
    sig_PS_signal_sim[j,:] = np.std(PS_signal_bin,axis=0)



xHI_inter = 0.15
fX_inter = -2.
PS_signal_inter = np.empty(len(k_bins_cent))
sig_inter = np.empty(len(k_bins_cent))

for i in range(len(k_bins_cent)):
               
  inter_fun_PS21 = interpolate.LinearNDInterpolator(np.transpose([xHI_mean,logfX]),PS_signal_sim[:,i])
  PS_signal_inter[i] = inter_fun_PS21(xHI_inter,fX_inter)
  inter_fun_sig = interpolate.LinearNDInterpolator(np.transpose([xHI_mean,logfX]),sig_PS_signal_sim[:,i])
  sig_inter[i] = inter_fun_sig(xHI_inter,fX_inter)

'''
fig = plt.figure(figsize=(20.,10.))
gs = gridspec.GridSpec(1,1)
fsize = 16

ax0 = plt.subplot(gs[0,0])

for i in range(len(files)):
  ax0.plot(k_bins_cent,PS_signal_sim[i],'-',label=r'SIM: $<xHI>=%.2f,fX=%.1f$' % (xHI_mean[i],logfX[i]))
ax0.plot(k_bins_cent,PS_signal_inter,'--',c='fuchsia',label=r'INTER: $<xHI>=%.2f,fX=%.1f$' % (xHI_inter,fX_inter))
ax0.legend(frameon=False,loc='upper right',fontsize=fsize,ncol=3)
#ax0.fill_between(k_bins_cent,PS_signal_16,PS_signal_84,alpha=0.25,color='darkorange')

#ax0.set_xlim(z_min,z_max)
ax0.set_ylim(1e-13,1e-6)
#ax0.set_yticks(np.arange(0.97,1.031,0.01))
ax0.set_xscale('log')
ax0.set_yscale('log')
ax0.set_xlabel(r'$k \,\rm [MHz^{-1}]$', fontsize=fsize)
ax0.set_ylabel(r'$P_{\rm 1D}\,\rm [MHz]$', fontsize=fsize)
ax0.tick_params(axis='x',which='major',direction='in',bottom=True,top=True,left=True,right=True
		,length=10,width=1,labelsize=fsize)
ax0.tick_params(axis='y',which='major',direction='in',bottom=True,top=True,left=True,right=True
		,length=10,width=1,labelsize=fsize)
ax0.tick_params(axis='both',which='minor',direction='in',bottom=True,top=True,left=True,right=True
		,length=5,width=1)
#ax0.set_title(r'$\mathrm{log}f_{\rm X}=%.1f,\ x_{\rm HI}=%.2f,\ S_{147\mathrm{MHz}}=%.1f\,\mathrm{mJy},\ \alpha_{\mathrm{R}}=%.2f,\ %s:\ \Delta\nu=%d\,\mathrm{kHz},\ t_{\mathrm{int}}=%d\mathrm{hr},\ %d\rm LOS $' % (fX_name,xHI_mean,S_min_QSO,alpha_R,telescope,spec_res,t_int,n_los),fontsize=fsize-4)

plt.tight_layout()
#plt.subplots_adjust(hspace=.0)
plt.show()
'''

xHI_mean_mock = 0.25
logfX_mock = -2.4

datafile = str('1DPS_signal/power_spectrum_signal_21cmFAST_50Mpc_z%.1f_fX%.2f_xHI%.2f_%dkHz_%dLOS.dat' % (z_name,logfX_mock,xHI_mean_mock,spec_res,Nlos))
data = np.fromfile(str(datafile),dtype=np.float32)
Nlos = int(data[0])
n_kbins = int(data[1])
k = data[2:2+n_kbins]
PS_signal = data[2+n_kbins+0*n_kbins*Nlos:2+n_kbins+1*n_kbins*Nlos]
PS_signal = np.reshape(PS_signal,(Nlos,n_kbins))[:n_los,:]

PS_signal_bin = np.empty((n_los,len(k_bins_cent)))

for i in range(n_los):
  for l in range(len(k_bins_cent)):
    ind = np.where((k>=k_bins[l]) & (k<k_bins[l+1]))[0]
    PS_signal_bin[i,l] = np.mean(PS_signal[i,ind])

PS_signal_mock = np.median(PS_signal_bin,axis=0)
sig_PS_signal_moc = np.std(PS_signal_bin,axis=0)

par_spc=np.array([xHI_mean,logfX])



inter_fun_PS21 = interpolate.LinearNDInterpolator(np.transpose([xHI_mean,logfX]),PS_signal_sim)
PS_signal_inter = inter_fun_PS21([0.15,-2])
#print(PS_signal_inter)

def log_LKHD(para):

    inter_fun_PS21 = interpolate.LinearNDInterpolator(np.transpose([xHI_mean,logfX]),PS_signal_sim)
    PS_signal_inter = inter_fun_PS21(para)
    inter_fun_sig = interpolate.LinearNDInterpolator(np.transpose([xHI_mean,logfX]),sig_PS_signal_sim)
    sig_inter = inter_fun_sig(para)
    return 0.5*np.sum((PS_signal_mock-PS_signal_inter)**2/sig_inter**2+np.log(2*np.pi*sig_inter**2))

def log_Prior(para):
    xHI_mean1, logfX1 = para
    if 0.01<=xHI_mean1<=0.5 and -3.0<=logfX1<=1.0:
        return 0
    return -np.inf

def log_posterior(para):
    LP=log_Prior(para)
    if not np.isfinite(LP):
        return -np.inf
    return LP-log_LKHD(para)


n_walk=64
ndim=2
para0=np.array([0.45,-2.9])+1e-4*np.random.randn(n_walk, ndim)
sampler = emcee.EnsembleSampler(n_walk, ndim, log_posterior)
state=sampler.run_mcmc(para0, 5000, progress=True)
samples = sampler.get_chain()

fsize=16
fig, axes = plt.subplots(ndim, sharex=True)
labels = [r"$<x_{\rm HI}>$",r"$\log_{10}f_{\mathrm{X}}$"]


for i in range(ndim):
    ax = axes[i]
    ax.plot(samples[:, :, i], "k", alpha=0.3)
    ax.set_xlim(0, len(samples))
    ax.set_ylabel(labels[i],fontsize=fsize)
    ax.yaxis.set_label_coords(-0.07, 0.5)

axes[-1].set_xlabel("Step number",fontsize=fsize)

#plt.show()
plt.savefig('MCMCchains.png')
plt.close()

para_mean=np.zeros(ndim)
noflat_samples = sampler.get_chain(discard=200, thin=50)
np.save('noflatsamp_1e12all',noflat_samples)

flat_samples = sampler.get_chain(discard=200, thin=50, flat=True)
np.save('flatsamp_1e12all',flat_samples)

param_label = ['<xHI>', 'logfX']

for j in range(ndim):
        mcmc = np.percentile(flat_samples[:, j], [16, 50, 84])
        q = np.diff(mcmc)
        para_mean[j]=mcmc[1]
        print('%s = %.5f + %.5f - %.5f' % (param_label[j],mcmc[1], q[0], q[1]))
        
sett=dict(fontsize=16)
fig=corner.corner(flat_samples,labels=labels,truths=para_mean,labelpad=0.01,label_kwargs=sett, smooth=True)

axes = np.array(fig.axes).reshape((2,2)); print(axes)
ax = axes[1,0]
ax.plot(logfX_mock,xHI_mean_mock,c='red')

#tau = sampler.get_autocorr_time()
#print(tau)
plt.savefig('infered_param_xHI%.2f_fX%.1f_wholepspace_5000.png' % (xHI_mean_mock,logfX_mock))
plt.show()