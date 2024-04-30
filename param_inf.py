"""
Parameter inference code based on Bayesian methods.

Uses 2D interpolator for 1D PS from 21-cm forest.

Version 12.2.2024

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

#Input parameters
z_name = float(sys.argv[1])         #redshift
dvH = float(sys.argv[2])            #used rebinning for line profile convolution in m/s
spec_res = float(sys.argv[3])       #spectral resolution of the telescope in kHz
xHI_mean_mock = float(sys.argv[4])  #mock HI fraction
logfX_mock = float(sys.argv[5])     #mock logfX
path_LOS = '../../datasets/21cmFAST_los/los/'
telescope = str(sys.argv[6])
S147 = float(sys.argv[7])           #intrinsic flux density of background source at 147MHz in mJy
alphaR = float(sys.argv[8])         #radio spectrum power-law index of background source
tint = float(sys.argv[9])           #intergration time for the observation in h

Nlos = 1000
n_los = 1000

min_logfX = -4.
max_logfX = 1.
min_xHI = 0.01
max_xHI = 0.6



#Find all of the datasets for the interpolation
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

#Prepare k bins
d_log_k_bins = 0.5
log_k_bins = np.arange(0.0-d_log_k_bins/2.,3.+d_log_k_bins/2.,d_log_k_bins)
k_bins = np.power(10.,log_k_bins)
k_bins_cent = np.power(10.,log_k_bins+d_log_k_bins/2.)[:-1]
#print(k_bins)
#print(k_bins_cent)
PS_signal_sim = np.empty((len(logfX),len(k_bins_cent)))
sig_PS_signal_sim = np.empty((len(logfX),len(k_bins_cent)))



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
    sig_PS_signal_sim[j,:] = np.std(PS_signal_bin,axis=0)



#Read the mock data for which we want to estimate parameters
datafile = str('1DPS_signal/power_spectrum_signal_21cmFAST_50Mpc_z%.1f_fX%.2f_xHI%.2f_%dkHz_%dLOS.dat' % (z_name,logfX_mock,xHI_mean_mock,spec_res,Nlos))
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
PS_signal_mock = np.median(PS_signal_bin,axis=0)
sig_PS_signal_mock = np.std(PS_signal_bin,axis=0)



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



#Define parameters to be estimated
#The below steps are based on tutarial for emcee package (Foreman-Mackey et al. 2013) at https://emcee.readthedocs.io/en/stable/tutorials/line/
par_spc=np.array([xHI_mean,logfX])

#Define prior distribution as uniform within the range of our simulated data
def log_prior(theta):
    xHI_mean1, logfX1 = theta
    if min_xHI <= xHI_mean1 <= max_xHI and min_logfX <= logfX1 <= max_logfX:
        return 0
    return -np.inf

#Set up N-dimensional linear interpolator for calculating P21 for any parameter values within the range given in the prior function
inter_fun_PS21 = interpolate.LinearNDInterpolator(np.transpose([xHI_mean,logfX]),PS_signal_sim)

#Define (negative) likelihood function
def log_likelihood(theta,P21_mock,sig_P21):
    xHI_mean1, logfX1 = theta 
    PS_signal_inter = inter_fun_PS21(xHI_mean1, logfX1)
    #inter_fun_sig = interpolate.LinearNDInterpolator(np.transpose([xHI_mean,logfX]),sig_PS_signal_sim)
    #sig_inter = inter_fun_sig(para)
    return 0.5*np.sum((P21_mock-PS_signal_inter)**2/sig_P21**2+np.log(2*np.pi*sig_P21**2))

#Define posterior function based on Bayes therom. Note that no normalization is assumed.
def log_posterior(theta,P21_mock,sig_P21):
    LP = log_prior(theta)
    if not np.isfinite(LP):
        return -np.inf
    return LP-log_likelihood(theta,P21_mock,sig_P21)



print('Reading data done')
#Initiate MCMC for the parameter estimation
n_walk=64
ndim=2
Nsteps = 20000

initial = np.array([0.2,-1])# + 0.1 * np.random.randn(2)
soln = minimize(log_likelihood, initial, args=(PS_signal_mock,sig_PS_noise),bounds=([min_xHI,max_xHI],[min_logfX,max_logfX]))
para0 = soln.x+1e-4*np.random.randn(n_walk, ndim)
sampler = emcee.EnsembleSampler(n_walk, ndim, log_posterior, args=(PS_signal_mock,sig_PS_noise))
state=sampler.run_mcmc(para0, Nsteps, progress=True)
samples = sampler.get_chain()

#And plot the chains for each parameter
fsize=16
fig, axes = plt.subplots(ndim,sharex=True, figsize=(5.,5.))
labels = [r"$\langle x_{\rm HI}\rangle$",r"$\log_{10}(f_{\mathrm{X}})$"]

for i in range(ndim):
    ax = axes[i]
    ax.plot(samples[:, :, i], "k", alpha=0.3)
    ax.set_xlim(0, len(samples))
    ax.set_ylabel(labels[i],fontsize=fsize)
    ax.yaxis.set_label_coords(-0.07, 0.5)

axes[-1].set_xlabel("Step number",fontsize=fsize)

#plt.savefig('MCMC_samples/MCMCchains_%dsteps.png' % Nsteps)
#plt.show()
plt.close()

#See how many steps does it take to burn-in for each parameter
tau = sampler.get_autocorr_time()
print('Autocorrelation time for xHI: %d' % tau[0])
print('Autocorrelation time for fX:  %d' % tau[1])



para_mean=np.zeros(ndim)
#Discard first 200 steps from the MCMC which corresponds to 5x the burn-in time
#noflat_samples = sampler.get_chain(discard=20, thin=50)
#np.save('noflatsamp_1e12all',noflat_samples)

#Flatten the MCMC
flat_samples = sampler.get_chain(discard=500, thin=50, flat=True)

#Compute the best estimated value and corresponding uncertainty for each parameter
param_label = ['<xHI>', 'logfX']
for j in range(ndim):
        mcmc = np.percentile(flat_samples[:, j], [16, 50, 84])
        q = np.diff(mcmc)
        para_mean[j]=mcmc[1]
        print('%s = %.5f + %.5f - %.5f' % (param_label[j],mcmc[1], q[0], q[1]))
        
array = np.array([para_mean])
array = np.concatenate((array,flat_samples))
np.save('MCMC_samples/flatsamp_xHI%.2f_fX%.1f_%s_%dkHz_Smin%.1fmJy_alphaR%.2f_t%dh_%dsteps.npy' % (xHI_mean_mock,logfX_mock,telescope,spec_res,S147,alphaR,tint,Nsteps),array)

#Present the result in corner plot using corner package (Foreman-Mackey et al. 2023) at https://corner.readthedocs.io/en/latest/
#fig,axes = plt.subplots(ndim,ndim,sharex=True,figsize=(10.,10.))
sett=dict(fontsize=14)
fig=corner.corner(flat_samples,range=[[0.,max_xHI],[min_logfX,max_logfX]],color='royalblue',smooth=True,labels=labels,label_kwargs=sett
                  ,show_titles=True,title_kwargs=sett,truths=para_mean,truth_color='fuchsia')

corner.overplot_points(fig=fig,xs=[[np.nan,np.nan],[np.nan,np.nan],[xHI_mean_mock,logfX_mock],[np.nan,np.nan]],marker='x',markersize=10,markeredgewidth=3,color='darkorange')

plt.savefig('MCMC_samples/inferred_param_xHI%.2f_fX%.1f_%s_%dkHz_Smin%.1fmJy_alphaR%.2f_t%dh_%dsteps.png' % (xHI_mean_mock,logfX_mock,telescope,spec_res,S147,alphaR,tint,Nsteps))
plt.show()