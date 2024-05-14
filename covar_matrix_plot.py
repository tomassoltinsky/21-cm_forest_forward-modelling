"""
Parameter inference code based on Bayesian methods.

Uses 2D interpolator for 1D PS from 21-cm forest.

Likelihood calculated from the covariance matrix.

Version 6.5.2024

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
from matplotlib.colors import LogNorm

import covariance_matrix

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
Nobs = int(sys.argv[10])            #number of observed sources or LOS

Nlos = 1000
n_los = 1000
Nsamples = 10000

min_logfX = -4.
max_logfX = 1.
min_xHI = 0.01
max_xHI = 0.6

fsize = 16

#Prepare k bins
d_log_k_bins = 0.1
log_k_bins = np.arange(0.0-d_log_k_bins/2.,3.+d_log_k_bins/2.,d_log_k_bins)
#log_k_bins = np.arange(0.3-d_log_k_bins/2.,2.6+d_log_k_bins/2.,d_log_k_bins)

k_bins = np.power(10.,log_k_bins)
k_bins_cent = np.power(10.,log_k_bins+d_log_k_bins/2.)[:-1]

#Read the mock data for which we want to estimate parameters
datafile = str('1DPS_dimensionless/1DPS_signalandnoise/power_spectrum_signal_21cmFAST_50Mpc_z%.1f_fX%.2f_xHI%.2f_%s_%dkHz_t%dh_Smin%.1fmJy_alphaR%.2f_%dLOS.dat' % (z_name,logfX_mock,xHI_mean_mock,telescope,spec_res,tint,S147,alphaR,Nlos))
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
PS_signal_mock = covariance_matrix.multi_obs(PS_signal_bin,Nobs,Nsamples)

print('Mock data prepared')

#PS_signal_mock_med = np.median(PS_signal_bin,axis=0)

#PS_signal_mock = np.array([np.median(PS_signal_bin,axis=0)])
#print(PS_signal_mock.shape)

#Find all of the datasets for the interpolation
#files = glob.glob(path_LOS+'*.dat')
#files = glob.glob(path_LOS+'*xHI0.25*.dat')

files = ['../../datasets/21cmFAST_los/los/los_50Mpc_256_n1000_z6.000_fX-3.0_xHI0.15.dat'
        ,'../../datasets/21cmFAST_los/los/los_50Mpc_256_n1000_z6.000_fX-2.0_xHI0.15.dat'
        ,'../../datasets/21cmFAST_los/los/los_50Mpc_256_n1000_z6.000_fX-1.0_xHI0.15.dat'
        ,'../../datasets/21cmFAST_los/los/los_50Mpc_256_n1000_z6.000_fX-3.0_xHI0.25.dat'
        ,'../../datasets/21cmFAST_los/los/los_50Mpc_256_n1000_z6.000_fX-2.0_xHI0.25.dat'
        ,'../../datasets/21cmFAST_los/los/los_50Mpc_256_n1000_z6.000_fX-1.0_xHI0.25.dat'
        ,'../../datasets/21cmFAST_los/los/los_50Mpc_256_n1000_z6.000_fX-3.0_xHI0.35.dat'
        ,'../../datasets/21cmFAST_los/los/los_50Mpc_256_n1000_z6.000_fX-2.0_xHI0.35.dat'
        ,'../../datasets/21cmFAST_los/los/los_50Mpc_256_n1000_z6.000_fX-1.0_xHI0.34.dat'
        ,'../../datasets/21cmFAST_los/los/los_50Mpc_256_n1000_z6.000_fX-3.0_xHI0.52.dat'
        ,'../../datasets/21cmFAST_los/los/los_50Mpc_256_n1000_z6.000_fX-2.0_xHI0.52.dat'
        ,'../../datasets/21cmFAST_los/los/los_50Mpc_256_n1000_z6.000_fX-1.0_xHI0.52.dat']

files = ['../../datasets/21cmFAST_los/los/los_50Mpc_256_n1000_z6.000_fX-2.0_xHI0.25.dat']


logfX = np.empty(len(files))
xHI_mean = np.empty(len(files))

#Find all of the parameter values in simulated data and read the data for interpolation
for j in range(len(files)):
    
    data = np.fromfile(str(files[j]),dtype=np.float32)
    logfX[j] = data[9]
    xHI_mean[j] = data[11]
    #print('f_X=%.2f, <x_HI,box>=%.8f' % (logfX[j],xHI_mean[j]))

    #Read data for signal
    #datafile = str('1DPS_signal/power_spectrum_signal_21cmFAST_50Mpc_z%.1f_fX%.2f_xHI%.2f_%s_%dkHz_t%dh_Smin%.1fmJy_alphaR%.2f_%dLOS.dat' % (z_name,logfX[j],xHI_mean[j],telescope,spec_res,tint,S147,alphaR,Nlos))
    datafile = str('1DPS_dimensionless/1DPS_signal/power_spectrum_signal_21cmFAST_50Mpc_z%.1f_fX%.2f_xHI%.2f_%dkHz_%dLOS.dat' % (z_name,logfX[j],xHI_mean[j],spec_res,Nlos))
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

    #Take average (median) of the simulated PS
    PS_signal_sim = covariance_matrix.multi_obs(PS_signal_bin,Nobs,Nsamples)



    Mcovar = covariance_matrix.covar_matrix(PS_signal_mock,PS_signal_sim)
    Mcorr = covariance_matrix.corr_matrix(Mcovar)

    print(Mcovar)
    print(Mcorr)
    
    diagonal = np.empty(len(k_bins_cent))
    for i in range(len(k_bins_cent)):
      diagonal[i] = Mcovar[i,i]
    print(diagonal)



    #Plotting the covariance and correlation matrices
    fig = plt.figure(figsize=(7.5,7.85))
    gs = gridspec.GridSpec(1,3,width_ratios=[7.5,0.1,0.25])

    ax0= plt.subplot(gs[0,0])
    im=ax0.imshow(Mcorr,origin='lower',interpolation='none',cmap=plt.cm.inferno
            ,extent=[log_k_bins[0],log_k_bins[-1],log_k_bins[0],log_k_bins[-1]]
            ,aspect='auto',vmin=0,vmax=1)

    ax0.set_xlabel(r'$\mathrm{log}_{10}(k/\rm MHz^{-1})$',fontsize=fsize)
    ax0.set_ylabel(r'$\mathrm{log}_{10}(k/\rm MHz^{-1})$',fontsize=fsize)
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

    #ax0.set_title(r"$\langle x_{\rm HI}\rangle=%.2f,\, \log_{10}(f_{\mathrm{X}})=%.1f,\, %s,\, t_{\rm int}=%d\,\rm hr,\, S_{\rm 147}=%.1f\,\mathrm{mJy},\, \alpha_{\rm R}=%.2f$" % (xHI_mean_mock,logfX_mock,telescope,tint,S147,alphaR),fontsize=fsize)
    ax0.set_title(r"$\langle x_{\rm HI}\rangle=%.2f,\, \log_{10}(f_{\mathrm{X}})=%.1f,\, %s,\, t_{\rm int}=%d\,\mathrm{hr},\, N_{\rm obs}=%d$" % (xHI_mean[j],logfX[j],telescope,tint,Nobs),fontsize=fsize)

    plt.tight_layout()
    plt.subplots_adjust(wspace=.0)
    plt.savefig('covariance_matrix/correlation_dimless_matrix_z%.1f_fX%.2f_xHI%.2f_%s_%dkHz_t%dh_Smin%.1fmJy_alphaR%.2f_median%dLOS.png' % (z_name,logfX[j],xHI_mean[j],telescope,spec_res,tint,S147,alphaR,Nobs))
    plt.show()
    plt.close()



    fig = plt.figure(figsize=(7.5,7.85))
    gs = gridspec.GridSpec(1,3,width_ratios=[7.5,0.1,0.25])

    ax0= plt.subplot(gs[0,0])
    im=ax0.imshow(Mcovar,origin='lower',interpolation='none',cmap=plt.cm.inferno
            ,extent=[log_k_bins[0],log_k_bins[-1],log_k_bins[0],log_k_bins[-1]]
            ,aspect='auto',norm=LogNorm(vmin=5e-16,vmax=2e-9))

    ax0.set_xlabel(r'$\mathrm{log}_{10}(k/\rm MHz^{-1})$',fontsize=fsize)
    ax0.set_ylabel(r'$\mathrm{log}_{10}(k/\rm MHz^{-1})$',fontsize=fsize)
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

    #ax0.set_title(r"$\langle x_{\rm HI}\rangle=%.2f,\, \log_{10}(f_{\mathrm{X}})=%.1f,\, %s,\, t_{\rm int}=%d\,\rm hr,\, S_{\rm 147}=%.1f\,\mathrm{mJy},\, \alpha_{\rm R}=%.2f$" % (xHI_mean_mock,logfX_mock,telescope,tint,S147,alphaR),fontsize=fsize)
    ax0.set_title(r"$\langle x_{\rm HI}\rangle=%.2f,\, \log_{10}(f_{\mathrm{X}})=%.1f,\, %s,\, t_{\rm int}=%d\,\mathrm{hr},\, N_{\rm obs}=%d$" % (xHI_mean[j],logfX[j],telescope,tint,Nobs),fontsize=fsize)

    plt.tight_layout()
    plt.subplots_adjust(wspace=.0)
    plt.savefig('covariance_matrix/covariance_matrix_dimless_z%.1f_fX%.2f_xHI%.2f_%s_%dkHz_t%dh_Smin%.1fmJy_alphaR%.2f_median%dLOS.png' % (z_name,logfX[j],xHI_mean[j],telescope,spec_res,tint,S147,alphaR,Nobs))
    plt.show()
    plt.close()