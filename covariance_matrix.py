"""
Covariance matrix computation for signal, noise and signal+noise.

Version 1.5.2024

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
from scipy.optimize import minimize
import emcee
import corner
from numpy import random

#Input parameters
z_name = float(sys.argv[1])         #redshift
dvH = float(sys.argv[2])            #used rebinning for line profile convolution in m/s
spec_res = float(sys.argv[3])       #spectral resolution of the telescope in kHz
xHI_mean_mock = float(sys.argv[4])  #mock HI fraction
logfX_mock = float(sys.argv[5])     #mock logfX
telescope = str(sys.argv[6])
t_int = float(sys.argv[7])
S_min_QSO = float(sys.argv[8])
alpha_R = float(sys.argv[9])

Nlos = 1000
n_los = 1000



#Prepare k bins
d_log_k_bins = 0.5
log_k_bins = np.arange(0.0-d_log_k_bins/2.,3.+d_log_k_bins/2.,d_log_k_bins)
k_bins = np.power(10.,log_k_bins)
k_bins_cent = np.power(10.,log_k_bins+d_log_k_bins/2.)[:-1]

#Read the mock data for which we want to estimate parameters for pure signal
datafile = str('1DPS_signal/power_spectrum_signal_21cmFAST_50Mpc_z%.1f_fX%.2f_xHI%.2f_%dkHz_%dLOS.dat' % (z_name,logfX_mock,xHI_mean_mock,spec_res,Nlos))
data = np.fromfile(str(datafile),dtype=np.float32)
Nlos = int(data[0])
n_kbins = int(data[1])
k_signal = data[2:2+n_kbins]
PS_signal = data[2+n_kbins+0*n_kbins*Nlos:2+n_kbins+1*n_kbins*Nlos]
PS_signal = np.reshape(PS_signal,(Nlos,n_kbins))[:n_los,:]

#For noise
datafile = str('1DPS_noise/power_spectrum_noise_50Mpc_z%.1f_%s_%dkHz_Smin%.1fmJy_alphaR%.2f_t%dh.dat' % (z_name,telescope,spec_res,S_min_QSO,alpha_R,t_int))
data = np.fromfile(str(datafile),dtype=np.float32)
n_kbins = int(data[0])
k_noise = data[1:1+n_kbins]
PS_noise = data[1+n_kbins+0*n_kbins*Nlos:1+n_kbins+1*n_kbins*Nlos]
PS_noise = np.reshape(PS_noise,(Nlos,n_kbins))[:n_los,:]

#For noisy signal
datafile = str('1DPS_signalandnoise/power_spectrum_signal_21cmFAST_50Mpc_z%.1f_fX%.2f_xHI%.2f_%s_%dkHz_t%dh_Smin%.1fmJy_alphaR%.2f_%dLOS.dat' % (z_name,logfX_mock,xHI_mean_mock,telescope,spec_res,t_int,S_min_QSO,alpha_R,Nlos))
data = np.fromfile(str(datafile),dtype=np.float32)
n_kbins = int(data[1])
k_ns = data[2:2+n_kbins]
PS_ns = data[2+n_kbins+0*n_kbins*Nlos:2+n_kbins+1*n_kbins*Nlos]
PS_ns = np.reshape(PS_ns,(Nlos,n_kbins))[:n_los,:]

#Bin the PS data
PS_signal_bin = np.empty((n_los,len(k_bins_cent)))
PS_noise_bin = np.empty((n_los,len(k_bins_cent)))
PS_ns_bin = np.empty((n_los,len(k_bins_cent)))

for l in range(len(k_bins_cent)):

  ind = np.where((k_signal>=k_bins[l]) & (k_signal<k_bins[l+1]))[0]

  for i in range(n_los):

    PS_signal_bin[i,l] = np.mean(PS_signal[i,ind])

  ind = np.where((k_noise>=k_bins[l]) & (k_noise<k_bins[l+1]))[0]
 
  for i in range(n_los):

    PS_noise_bin[i,l] = np.mean(PS_noise[i,ind])

  ind = np.where((k_ns>=k_bins[l]) & (k_ns<k_bins[l+1]))[0]
 
  for i in range(n_los):

    PS_ns_bin[i,l] = np.mean(PS_ns[i,ind])

#Take median for each k bin
PS_signal_med = np.median(PS_signal_bin,axis=0)
sig_PS_signal_med = np.std(PS_signal_bin,axis=0)
PS_noise_med = np.median(PS_noise_bin,axis=0)
sig_PS_noise_med = np.std(PS_noise_bin,axis=0)
PS_ns_med = np.median(PS_ns_bin,axis=0)
sig_PS_ns_med = np.std(PS_ns_bin,axis=0)

#Create sample of N_samples of median PS from N_obs LOS
N_samples = 10000
N_obs = 10
PS_ns_obs = np.empty((N_samples,len(k_bins_cent)))

for i in range(N_samples):
  LOS = random.randint(0,Nlos-1,size=N_obs)
  PS_ns_obs[i][:] = np.median(PS_ns_bin[LOS][:],axis=0)



#Calculate the covariance and correllation matrices
def covar_matrix(P21_all,P21_sim):
#def covar_matrix(P21_all,P21_sim,NmockLOS):
  
  Ndim = len(P21_sim)
  covariance = np.empty([Ndim,Ndim])

  for i in range(Ndim):

    for j in range(Ndim):
        
        #LOS = random.randint(0,Nlos-1,size=NmockLOS)
        #covariance[i,j] = np.median((P21_all[LOS][:,i]-P21_sim[i])*(P21_all[LOS][:,j]-P21_sim[j]))
        covariance[i,j] = np.median((P21_all[:][:,i]-P21_sim[i])*(P21_all[:][:,j]-P21_sim[j]))

  return covariance

def corr_matrix(covariance):
  
  Ndim = len(covariance)
  correlation = np.empty([Ndim,Ndim])

  for i in range(Ndim):

    for j in range(Ndim):
        
        correlation[i,j] = covariance[i][j]/np.sqrt(covariance[i][i]*covariance[j][j])

  return correlation

Mcovar_signal = covar_matrix(PS_signal_bin,PS_signal_med)
Mcorr_signal  = corr_matrix(Mcovar_signal)

Mcovar_noise = covar_matrix(PS_noise_bin,PS_noise_med)
Mcorr_noise  = corr_matrix(Mcovar_noise)

Mcovar_ns = covar_matrix(PS_ns_obs,PS_ns_med)
Mcorr_ns  = corr_matrix(Mcovar_ns)



#Plot all three cases of PS correllation matrix
fsize = 16
fig = plt.figure(figsize=(7.5,7.85))
gs = gridspec.GridSpec(1,3,width_ratios=[7.5,0.1,0.25])

ax0= plt.subplot(gs[0,0])
im=ax0.imshow(Mcorr_signal,origin='lower',interpolation='none',cmap=plt.cm.inferno
	      ,extent=[log_k_bins[0],log_k_bins[-1],log_k_bins[0],log_k_bins[-1]]
	      ,aspect='auto',vmin=0,vmax=1)

ax0.set_xlabel(r'$\mathrm{log}_{10}(k/\rm MHz^{-1})$',fontsize=fsize)
ax0.set_ylabel(r'$\mathrm{log}_{10}(k/\rm MHz^{-1})$',fontsize=fsize)
#ax0.set_xticks(np.arange(-1.,3.1,1.))
#ax0.set_yticks(np.arange(0.,6.1,1.))
#ax0.set_xlim(Dmin,3.5)
#ax0.set_ylim(0.5,6.5)
ax0.xaxis.set_minor_locator(AutoMinorLocator())
ax0.yaxis.set_minor_locator(AutoMinorLocator())
ax0.tick_params(axis='y',which='major',direction='in',bottom=True,top=True,left=True,right=True
		,length=10,width=1,labelsize=fsize)
ax0.tick_params(axis='y',which='minor',direction='in',bottom=True,top=True,left=True,right=True
		,length=5,width=1)
ax0.tick_params(axis='x',which='major',direction='in',bottom=True,top=True,left=True,right=True
		,length=10,width=1,labelsize=fsize)
ax0.tick_params(axis='x',which='minor',direction='in',bottom=True,top=True,left=True,right=True
		,length=5,width=1)

axc= plt.subplot(gs[0,2])
cbar=fig.colorbar(im,pad=0.02,cax=axc)
#cbar.set_label(r'$\tau_{21}$',size=fsize)
cbar.ax.tick_params(labelsize=fsize)
fig.gca()

ax0.set_title(r"$\langle x_{\rm HI}\rangle=%.2f,\, \log_{10}(f_{\mathrm{X}})=%.1f$" % (xHI_mean_mock,logfX_mock),fontsize=fsize)

plt.tight_layout()
plt.subplots_adjust(wspace=.0)
#plt.savefig('covariance_matrix/correlation_matrix_signal_z%.1f_fX%.2f_xHI%.2f_2.png' % (z_name,logfX_mock,xHI_mean_mock))
#plt.show()
plt.close()



fig = plt.figure(figsize=(7.5,7.85))
gs = gridspec.GridSpec(1,3,width_ratios=[7.5,0.1,0.25])

ax0= plt.subplot(gs[0,0])
im=ax0.imshow(Mcorr_noise,origin='lower',interpolation='none',cmap=plt.cm.inferno
	      ,extent=[log_k_bins[0],log_k_bins[-1],log_k_bins[0],log_k_bins[-1]]
	      ,aspect='auto',vmin=0,vmax=1)

ax0.set_xlabel(r'$\mathrm{log}_{10}(k/\rm MHz^{-1})$',fontsize=fsize)
ax0.set_ylabel(r'$\mathrm{log}_{10}(k/\rm MHz^{-1})$',fontsize=fsize)
#ax0.set_xticks(np.arange(-1.,3.1,1.))
#ax0.set_yticks(np.arange(0.,6.1,1.))
#ax0.set_xlim(Dmin,3.5)
#ax0.set_ylim(0.5,6.5)
ax0.xaxis.set_minor_locator(AutoMinorLocator())
ax0.yaxis.set_minor_locator(AutoMinorLocator())
ax0.tick_params(axis='y',which='major',direction='in',bottom=True,top=True,left=True,right=True
		,length=10,width=1,labelsize=fsize)
ax0.tick_params(axis='y',which='minor',direction='in',bottom=True,top=True,left=True,right=True
		,length=5,width=1)
ax0.tick_params(axis='x',which='major',direction='in',bottom=True,top=True,left=True,right=True
		,length=10,width=1,labelsize=fsize)
ax0.tick_params(axis='x',which='minor',direction='in',bottom=True,top=True,left=True,right=True
		,length=5,width=1)

axc= plt.subplot(gs[0,2])
cbar=fig.colorbar(im,pad=0.02,cax=axc)
#cbar.set_label(r'$\tau_{21}$',size=fsize)
cbar.ax.tick_params(labelsize=fsize)
fig.gca()

ax0.set_title(r"%s, $S_{\rm 147}=%.1f\,\mathrm{mJy},\, \alpha_{\rm R}=%.2f,\, t_{\rm int}=%d\,\rm hr$" % (telescope,S_min_QSO,alpha_R,t_int),fontsize=fsize)

plt.tight_layout()
plt.subplots_adjust(wspace=.0)
#plt.savefig('covariance_matrix/correlation_matrix_noise_z%.1f__%s_%dkHz_Smin%.1fmJy_alphaR%.2f_t%dh_2.png' % (z_name,telescope,spec_res,S_min_QSO,alpha_R,t_int))
#plt.show()
plt.close()



fig = plt.figure(figsize=(7.5,7.85))
gs = gridspec.GridSpec(1,3,width_ratios=[7.5,0.1,0.25])

ax0= plt.subplot(gs[0,0])
im=ax0.imshow(Mcorr_ns,origin='lower',interpolation='none',cmap=plt.cm.inferno
	      ,extent=[log_k_bins[0],log_k_bins[-1],log_k_bins[0],log_k_bins[-1]]
	      ,aspect='auto',vmin=0,vmax=1)

ax0.set_xlabel(r'$\mathrm{log}_{10}(k/\rm MHz^{-1})$',fontsize=fsize)
ax0.set_ylabel(r'$\mathrm{log}_{10}(k/\rm MHz^{-1})$',fontsize=fsize)
#ax0.set_xticks(np.arange(-1.,3.1,1.))
#ax0.set_yticks(np.arange(0.,6.1,1.))
#ax0.set_xlim(Dmin,3.5)
#ax0.set_ylim(0.5,6.5)
ax0.xaxis.set_minor_locator(AutoMinorLocator())
ax0.yaxis.set_minor_locator(AutoMinorLocator())
ax0.tick_params(axis='y',which='major',direction='in',bottom=True,top=True,left=True,right=True
		,length=10,width=1,labelsize=fsize)
ax0.tick_params(axis='y',which='minor',direction='in',bottom=True,top=True,left=True,right=True
		,length=5,width=1)
ax0.tick_params(axis='x',which='major',direction='in',bottom=True,top=True,left=True,right=True
		,length=10,width=1,labelsize=fsize)
ax0.tick_params(axis='x',which='minor',direction='in',bottom=True,top=True,left=True,right=True
		,length=5,width=1)

axc= plt.subplot(gs[0,2])
cbar=fig.colorbar(im,pad=0.02,cax=axc)
#cbar.set_label(r'$\tau_{21}$',size=fsize)
cbar.ax.tick_params(labelsize=fsize)
fig.gca()

#ax0.set_title(r"$\langle x_{\rm HI}\rangle=%.2f,\, \log_{10}(f_{\mathrm{X}})=%.1f,\, %s,\, t_{\rm int}=%d\,\rm hr,\, S_{\rm 147}=%.1f\,\mathrm{mJy},\, \alpha_{\rm R}=%.2f$" % (xHI_mean_mock,logfX_mock,telescope,t_int,S_min_QSO,alpha_R),fontsize=fsize)
ax0.set_title(r"$\langle x_{\rm HI}\rangle=%.2f,\, \log_{10}(f_{\mathrm{X}})=%.1f,\, %s,\, t_{\rm int}=%d\,\rm hr$" % (xHI_mean_mock,logfX_mock,telescope,t_int),fontsize=fsize)

plt.tight_layout()
plt.subplots_adjust(wspace=.0)
plt.savefig('covariance_matrix/correlation_matrix_ns_z%.1f_fX%.2f_xHI%.2f_%s_%dkHz_t%dh_Smin%.1fmJy_alphaR%.2f_median10LOS.png' % (z_name,logfX_mock,xHI_mean_mock,telescope,spec_res,t_int,S_min_QSO,alpha_R))
plt.show()
plt.close()