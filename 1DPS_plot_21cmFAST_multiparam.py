"""
Creating the power spectrum plot for different fX and xHI while fixing the other parameter.

Version 9.2.2024

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



fsize = 16

z_name = float(sys.argv[1])
dvH = float(sys.argv[2])
spec_res = float(sys.argv[3])
n_los = 1000

fX_fid = float(sys.argv[4])
xHI_fid = float(sys.argv[5])



path_LOS = '../../datasets/21cmFAST_los/los/'
files = glob.glob(path_LOS+'*fX%.1f*' % fX_fid)
xHI_mean = np.empty(len(files))

d_log_k_bins = 0.5
log_k_bins = np.arange(0.0-d_log_k_bins/2.,3.+d_log_k_bins/2.,d_log_k_bins)
k_bins = np.power(10.,log_k_bins)
k_bins_cent = np.power(10.,log_k_bins+d_log_k_bins/2.)[:-1]
PS_signal_sim = np.empty((len(files),len(k_bins_cent)))

for j in range(len(files)):
    
    data = np.fromfile(str(files[j]),dtype=np.float32)
    #logfX[j] = data[9]
    xHI_mean[j] = data[11]
    #print('f_X=%.2f, <x_HI,box>=%.8f' % (logfX[j],xHI_mean[j]))

    datafile = str('1DPS_signal/power_spectrum_signal_21cmFAST_50Mpc_z%.1f_fX%.2f_xHI%.2f_%dkHz_%dLOS.dat' % (z_name,fX_fid,xHI_mean[j],spec_res,1000))
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

print(PS_signal_sim[0,:])

fig = plt.figure(figsize=(10.,5.))
gs = gridspec.GridSpec(1,1)

ax0 = plt.subplot(gs[0,0])

for j in range(len(files)):
   
   ax0.plot(k_bins_cent,PS_signal_sim[j,:],'-',color='darkorange',label=r'Signal')
   ax0.text(1.1*k_bins_cent[-1],PS_signal_sim[j,-1],r'$%.2f$' % (xHI_mean[j]),fontsize=fsize-4)

ax0.text(100,7e-7,r'$\mathrm{log}f_{\mathrm{X}}=%.1f$' % (fX_fid),fontsize=fsize)


ax0.set_xlim(1,6e2)
ax0.set_ylim(5e-11,2e-6)
#ax0.set_yticks(np.arange(0.97,1.031,0.01))
ax0.set_xscale('log')
ax0.set_yscale('log')
ax0.set_xlabel(r'$k \,\rm [MHz^{-1}]$', fontsize=fsize)
ax0.set_ylabel(r'$P_{21}\,\rm [MHz]$', fontsize=fsize)
ax0.tick_params(axis='x',which='major',direction='in',bottom=True,top=True,left=True,right=True
		,length=10,width=1,labelsize=fsize)
ax0.tick_params(axis='y',which='major',direction='in',bottom=True,top=True,left=True,right=True
		,length=10,width=1,labelsize=fsize)
ax0.tick_params(axis='both',which='minor',direction='in',bottom=True,top=True,left=True,right=True
		,length=5,width=1)

plt.tight_layout()
plt.subplots_adjust(hspace=2.0)
plt.savefig('1DPS_plots/power_spectrum_vsfX_21cmFAST_50Mpc_z%.1f_fX%.2f.png' % (z_name,fX_fid))
plt.show()



path_LOS = '../../datasets/21cmFAST_los/los/'
files = glob.glob(path_LOS+'*xHI%.2f*' % xHI_fid)
logfX = np.empty(len(files))

d_log_k_bins = 0.5
log_k_bins = np.arange(0.0-d_log_k_bins/2.,3.+d_log_k_bins/2.,d_log_k_bins)
k_bins = np.power(10.,log_k_bins)
k_bins_cent = np.power(10.,log_k_bins+d_log_k_bins/2.)[:-1]
PS_signal_sim = np.empty((len(files),len(k_bins_cent)))

for j in range(len(files)):
    
    data = np.fromfile(str(files[j]),dtype=np.float32)
    logfX[j] = data[9]
    #xHI_mean[j] = data[11]
    #print('f_X=%.2f, <x_HI,box>=%.8f' % (logfX[j],xHI_mean[j]))

    datafile = str('1DPS_signal/power_spectrum_signal_21cmFAST_50Mpc_z%.1f_fX%.2f_xHI%.2f_%dkHz_%dLOS.dat' % (z_name,logfX[j],xHI_fid,spec_res,1000))
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

print(PS_signal_sim[0,:])

fig = plt.figure(figsize=(10.,5.))
gs = gridspec.GridSpec(1,1)

ax0 = plt.subplot(gs[0,0])

for j in range(len(files)):
   
   ax0.plot(k_bins_cent,PS_signal_sim[j,:],'-',color='darkorange',label=r'Signal')
   ax0.text(1.1*k_bins_cent[-1],PS_signal_sim[j,-1],r'$%.1f$' % (logfX[j]),fontsize=fsize-4)

ax0.text(100,7e-7,r'$x_{\mathrm{HI}}=%.2f$' % (xHI_fid),fontsize=fsize)


ax0.set_xlim(1,6e2)
ax0.set_ylim(5e-11,2e-6)
#ax0.set_yticks(np.arange(0.97,1.031,0.01))
ax0.set_xscale('log')
ax0.set_yscale('log')
ax0.set_xlabel(r'$k \,\rm [MHz^{-1}]$', fontsize=fsize)
ax0.set_ylabel(r'$P_{21}\,\rm [MHz]$', fontsize=fsize)
ax0.tick_params(axis='x',which='major',direction='in',bottom=True,top=True,left=True,right=True
		,length=10,width=1,labelsize=fsize)
ax0.tick_params(axis='y',which='major',direction='in',bottom=True,top=True,left=True,right=True
		,length=10,width=1,labelsize=fsize)
ax0.tick_params(axis='both',which='minor',direction='in',bottom=True,top=True,left=True,right=True
		,length=5,width=1)

plt.tight_layout()
plt.subplots_adjust(hspace=2.0)
plt.savefig('1DPS_plots/power_spectrum_vsxHI_21cmFAST_50Mpc_z%.1f_xHI%.2f.png' % (z_name,xHI_fid))
plt.show()