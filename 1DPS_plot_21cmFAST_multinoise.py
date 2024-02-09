"""
Creating the power spectrum plot.

Version 19.10.2023

"""

import sys
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
fX_name = float(sys.argv[2])
xHI_mean = float(sys.argv[3])
telescope = str(sys.argv[4])
dvH = float(sys.argv[5])
spec_res = float(sys.argv[6])
n_los = 1000

S_min_QSO = [64.2,64.2]#,110.6,110.6]
alpha_R = [-0.44,-0.44]#,-0.89,-0.89]
t_int = [100,500]
#t_int = [10,50,100]
#t_int = [10,10,10]



datafile = str('1DPS_signal/power_spectrum_signal_21cmFAST_50Mpc_z%.1f_fX%.2f_xHI%.2f_%dkHz_%dLOS.dat' % (z_name,fX_name,xHI_mean,spec_res,n_los))
data = np.fromfile(str(datafile),dtype=np.float32)
Nlos = int(data[0])
n_kbins = int(data[1])
k = data[2:2+n_kbins]
PS_signal = data[2+n_kbins+0*n_kbins*Nlos:2+n_kbins+1*n_kbins*Nlos]
PS_signal = np.reshape(PS_signal,(Nlos,n_kbins))[:n_los,:]

d_log_k_bins = 0.5
log_k_bins = np.arange(0.0-d_log_k_bins/2.,3.+d_log_k_bins/2.,d_log_k_bins)
k_bins = np.power(10.,log_k_bins)
k_bins_cent = np.power(10.,log_k_bins+d_log_k_bins/2.)[:-1]
print(k_bins)
print(k_bins_cent)
PS_signal_bin = np.empty((n_los,len(k_bins_cent)))

print('%d bins from %.3fMHz^-1 to %.3fMHz^-1' % (len(k),k[1],k[-1]))
for j in range(len(k_bins_cent)):
  ind = np.where((k>=k_bins[j]) & (k<k_bins[j+1]))[0]
  print(len(ind))

for i in range(n_los):
  for j in range(len(k_bins_cent)):
    ind = np.where((k>=k_bins[j]) & (k<k_bins[j+1]))[0]
    PS_signal_bin[i,j] = np.mean(PS_signal[i,ind])

PS_signal_med = np.median(PS_signal_bin,axis=0)
PS_signal_16 = np.percentile(PS_signal_bin,16,axis=0)
PS_signal_84 = np.percentile(PS_signal_bin,84,axis=0)


Nlos_noise = 500
PS_noise_bin = np.empty((len(t_int),Nlos_noise,len(k_bins_cent)))
PS_noise_med = np.empty((len(t_int),len(k_bins_cent)))

for i in range(len(t_int)):

  datafile = str('1DPS_noise/power_spectrum_noise_50cMpc_z%.1f_%s_%dkHz_Smin%.1fmJy_alphaR%.2f_t%dh.dat' % (z_name,telescope,spec_res,S_min_QSO[i],alpha_R[i],t_int[i]))
  data = np.fromfile(str(datafile),dtype=np.float32)
  n_kbins = int(data[0])
  k = data[1:1+n_kbins]
  PS_noise = np.reshape(data[1+n_kbins+0*n_kbins*Nlos_noise:1+n_kbins+1*n_kbins*Nlos_noise],(Nlos_noise,n_kbins))

  for l in range(Nlos_noise):

    for j in range(len(k_bins_cent)):
      ind = np.where((k>=k_bins[j]) & (k<k_bins[j+1]))[0]
      PS_noise_bin[i,l,j]  = np.mean(PS_noise[l,ind])

  PS_noise_med[i,:] = np.median(PS_noise_bin[i],axis=0)
  print(t_int[i],np.amin(PS_noise_med[i]),np.amax(PS_noise_med[i]))

print(PS_noise_med)

fig = plt.figure(figsize=(10.,5.))
gs = gridspec.GridSpec(1,1)

ax0 = plt.subplot(gs[0,0])

ax0.plot([1,2],[1,2],'-',color='darkorange',label=r'Signal')
ax0.plot([1,2],[1,2],'--',color='fuchsia',label=r'Noise')
ax0.legend(frameon=False,loc='lower left',fontsize=fsize,ncol=2)

ax0.plot(k_bins_cent,PS_signal_med,'-',color='darkorange',label=r'Signal')
ax0.fill_between(k_bins_cent,PS_signal_16,PS_signal_84,alpha=0.25,color='darkorange')
for j in range(1,len(S_min_QSO)):
  #ax0.plot([k_bins_cent[0],k_bins_cent[-1]],[np.amax(PS_noise_bin[j]),np.amax(PS_noise_bin[j])],'--',color='fuchsia',label=r'Noise')
  ax0.plot(k_bins_cent,PS_noise_med[j],'--',color='fuchsia',label=r'Signal')
  #ax0.text(20,1.2*np.amax(PS_noise_med[j]),r'$S_{147\mathrm{MHz}}=%.1f\,\mathrm{mJy},\ \alpha_{\mathrm{R}}=%.2f,\ t_{\mathrm{int}}=%d\mathrm{hr}$' % (S_min_QSO[j],alpha_R[j],t_int[j]),fontsize=fsize-4)
  ax0.text(60,1.3*np.amax(PS_noise_med[j]),r'$\sigma_{\mathrm{Noise}}=0.0027$',fontsize=fsize)



#ax0.set_xlim(z_min,z_max)
ax0.set_ylim(1e-10,1e-5)
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
ax0.set_title(r'$\mathrm{log}(f_{\rm X})=%.1f,\ \langle x_{\rm HI}\rangle =%.1f$' % (fX_name,xHI_mean),fontsize=fsize)

plt.tight_layout()
plt.subplots_adjust(hspace=.0)
plt.savefig('1DPS_plots/power_spectrum_21cmFAST_multinoise_50Mpc_z%.1f_fX%.2f_xHI%.2f_%s_%dkHz_sigmanoise.png' % (z_name,fX_name,xHI_mean,telescope,spec_res))
plt.show()
