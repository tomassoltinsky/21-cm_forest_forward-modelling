'''
Plot the 21-cm forest 1D PS from the mock observation, MCMC inferrence and MCMC posterior draws.

Version 16.04.2024
'''

import sys
import glob, os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import AutoMinorLocator
from matplotlib import gridspec
import corner
from numpy import random
from scipy import interpolate

#Input parameters
z_name = float(sys.argv[1])         #redshift
dvH = float(sys.argv[2])            #used rebinning for line profile convolution in m/s
spec_res = float(sys.argv[3])       #spectral resolution of the telescope in kHz
xHI_mock = float(sys.argv[4])  #mock HI fraction
logfX_mock = float(sys.argv[5])     #mock logfX
telescope = str(sys.argv[6])
S147 = float(sys.argv[7])           #intrinsic flux density of background source at 147MHz in mJy
alphaR = float(sys.argv[8])         #radio spectrum power-law index of background source
tint = float(sys.argv[9])           #intergration time for the observation in h

path = 'MCMC_samples'
Nsteps = 10000
Ndraws = 20
n_los = 1000



#Data for mock observation

d_log_k_bins = 0.5
log_k_bins = np.arange(0.0-d_log_k_bins/2.,3.+d_log_k_bins/2.,d_log_k_bins)
k_bins = np.power(10.,log_k_bins)
k_bins_cent = np.power(10.,log_k_bins+d_log_k_bins/2.)[:-1]

datafile = str('1DPS_signal/power_spectrum_signal_21cmFAST_50Mpc_z%.1f_fX%.2f_xHI%.2f_%dkHz_%dLOS.dat' % (z_name,logfX_mock,xHI_mock,spec_res,1000))
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

#Read data for noise
Nlos_noise = 500
datafile = str('1DPS_noise/power_spectrum_noise_50Mpc_z%.1f_%s_%dkHz_Smin%.1fmJy_alphaR%.2f_t%dh.dat' % (z_name,telescope,spec_res,S147,alphaR,tint))
data = np.fromfile(str(datafile),dtype=np.float32)
n_kbins = int(data[0])
k = data[1:1+n_kbins]
PS_noise = data[1+n_kbins+0*n_kbins*Nlos_noise:1+n_kbins+1*n_kbins*Nlos_noise]
PS_noise = np.reshape(PS_noise,(Nlos_noise,n_kbins))

#Bin the PS data
PS_noise_bin = np.empty((Nlos_noise,len(k_bins_cent)))

for i in range(Nlos_noise):
  for l in range(len(k_bins_cent)):
    ind = np.where((k>=k_bins[l]) & (k<k_bins[l+1]))[0]
    PS_noise_bin[i,l] = np.mean(PS_noise[i,ind])

#Take median for each k bin
#PS_noise = np.mean(PS_noise_bin,axis=0)
sig_PS_noise = np.std(PS_noise_bin,axis=0)

print('Mock data prepared')



#Find all of the datasets for the interpolation
path_LOS = '../../datasets/21cmFAST_los/los/'
files = glob.glob(path_LOS+'*.dat')
files_to_remove = glob.glob(path_LOS+'*fX-10.0*.dat')
for i in range(len(files_to_remove)):
   files.remove(files_to_remove[i])
files_to_remove = glob.glob(path_LOS+'*xHI0.7*.dat')
for i in range(len(files_to_remove)):
   files.remove(files_to_remove[i])
files_to_remove = glob.glob(path_LOS+'*xHI0.69*.dat')
for i in range(len(files_to_remove)):
   files.remove(files_to_remove[i])
files_to_remove = glob.glob(path_LOS+'*xHI0.68*.dat')
for i in range(len(files_to_remove)):
   files.remove(files_to_remove[i])
files_to_remove = glob.glob(path_LOS+'*xHI0.67*.dat')
for i in range(len(files_to_remove)):
   files.remove(files_to_remove[i])
files_to_remove = glob.glob(path_LOS+'*xHI0.66*.dat')
for i in range(len(files_to_remove)):
   files.remove(files_to_remove[i])
files_to_remove = glob.glob(path_LOS+'*xHI0.65*.dat')
for i in range(len(files_to_remove)):
   files.remove(files_to_remove[i])
files_to_remove = glob.glob(path_LOS+'*xHI0.64*.dat')
for i in range(len(files_to_remove)):
   files.remove(files_to_remove[i])

logfX = np.empty(len(files))
xHI_mean = np.empty(len(files))
PS_signal_sim = np.empty((len(files),len(k_bins_cent)))

#Find all of the parameter values in simulated data and read the data for interpolation
for j in range(len(files)):
    
    data = np.fromfile(str(files[j]),dtype=np.float32)
    logfX[j] = data[9]
    xHI_mean[j] = data[11]
    #print('f_X=%.2f, <x_HI,box>=%.8f' % (logfX[j],xHI_mean[j]))

    #Read data for signal
    datafile = str('1DPS_signal/power_spectrum_signal_21cmFAST_50Mpc_z%.1f_fX%.2f_xHI%.2f_%dkHz_%dLOS.dat' % (z_name,logfX[j],xHI_mean[j],spec_res,Nlos))
    data = np.fromfile(str(datafile),dtype=np.float32)
    Nlos = int(data[0])
    n_kbins = int(data[1])
    k = data[2:2+n_kbins]
    PS_signal = data[2+n_kbins+0*n_kbins*Nlos:2+n_kbins+1*n_kbins*Nlos]
    PS_signal = np.reshape(PS_signal,(Nlos,n_kbins))[:n_los,:]

    #Bin the PS data
    PS_signal_bin = np.empty((n_los,len(k_bins_cent)))

    for i in range(n_los):
      for l in range(len(k_bins_cent)):
        ind = np.where((k>=k_bins[l]) & (k<k_bins[l+1]))[0]
        PS_signal_bin[i,l] = np.mean(PS_signal[i,ind])

    #Take median for each k bin
    PS_signal_sim[j,:] = np.median(PS_signal_bin,axis=0)

#Set up N-dimensional linear interpolator for calculating P21 for any parameter values within the range given in the prior function
inter_fun_PS21 = interpolate.LinearNDInterpolator(np.transpose([xHI_mean,logfX]),PS_signal_sim)

print('Interpoaltor prepared')



#Data for inferred and posterior draws PS

data = np.load('%s/flatsamp_xHI%.2f_fX%.1f_%s_%dkHz_Smin%.1fmJy_alphaR%.2f_t%dh_%dsteps.npy' % (path,xHI_mock,logfX_mock,telescope,spec_res,S147,alphaR,tint,Nsteps))
xHI_inf = data[0,0]
logfX_inf = data[0,1]
print(xHI_inf,logfX_inf)

ind_draws = np.random.randint(1,data.shape[0],Ndraws)
data = data[ind_draws][:]
xHI_draws = data[:,0]
logfX_draws = data[:,1]
print(xHI_draws)
print(logfX_draws)

print('Inferred and posterior draws PS prepared')





fsize = 20
fig = plt.figure(figsize=(10.,5.))
gs = gridspec.GridSpec(1,1)
ax0 = plt.subplot(gs[0,0])

ax0.errorbar(k_bins_cent,PS_signal_mock,yerr=sig_PS_noise,fmt=' ',marker='o',capsize=5,color='darkorange',label='Mock data')
ax0.plot([1e-20,1e-20],[1e-20,2e-20],'-',linewidth=2,color='fuchsia',label='Inferred')
ax0.plot([1e-20,1e-20],[1e-20,2e-20],'-',color='royalblue',alpha=0.5,label='Posterior draws')
plt.legend(frameon=False,loc='lower left',fontsize=fsize)

for i in range(len(xHI_draws)):
   ax0.plot(k_bins_cent,inter_fun_PS21(xHI_draws[i],logfX_draws[i]),'-',color='royalblue',alpha=0.5,label='Posterior draws')

ax0.plot(k_bins_cent,inter_fun_PS21(xHI_inf,logfX_inf),'-',linewidth=2,color='fuchsia',label='Inferred')

ax0.set_xlim(0.8,4e2)
ax0.set_ylim(3e-10,2e-6)
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
plt.savefig('1DPS_plots/power_spectrum_mockandinf_50Mpc_z%.1f_xHI%.2f_fX%.1f_%s_%dkHz_Smin%.1fmJy_alphaR%.2f_t%dh.png' % (z_name,xHI_mock,logfX_mock,telescope,spec_res,S147,alphaR,tint))
plt.show()