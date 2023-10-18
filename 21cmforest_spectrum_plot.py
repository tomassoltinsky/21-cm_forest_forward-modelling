'''Plotting 21-cm forest

Version 18.10.2023

Ideal simulated spectrum
Smoothed spectrum
Noisy spectrum

Arguments:
1. Path to datasets
2. Redshift
3. log(fX)
4. Rebinning pixel-width in m/s
5. Telescope
6. Telescope spectral resolution in kHz
7. Background source flux density at 147MHz in mJy
8. Background source radio spectrum power-law index
9. Integration time of the observation in hr
10. Line-of-sight
'''

import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy.signal as scisi
from astropy.convolution import convolve, Box1DKernel
from matplotlib.ticker import AutoMinorLocator
from matplotlib import gridspec
import matplotlib.patches as patches
import openpyxl

#constants

import constants
import instrumental_features

fsize = 16

path = str(sys.argv[1])
z_name = float(sys.argv[2])
fX_name = float(sys.argv[3])
dvH = float(sys.argv[4])
telescope = str(sys.argv[5])
spec_res = float(sys.argv[6])
S_min_QSO = float(sys.argv[7])
alpha_R = float(sys.argv[8])
t_int = float(sys.argv[9])
LOS_ori = int(sys.argv[10])

Nlos = 100
filenum = int(np.floor(LOS_ori/Nlos))
LOS = LOS_ori-Nlos*filenum

datafile = str('%slos_regrid/200cMpclong_los_21cmREBIN_n%d_z%.3f_fX%.1f_dv%d_file%d.dat' % (path,Nlos,z_name,fX_name,dvH,filenum))
data  = np.fromfile(str(datafile),dtype=np.float32)
z     = data[0]	#redshift
Lbox  = data[5]/1e3	#box size
Nbins = int(data[7])					#Number of pixels/cells/bins in one line-of-sight
Nlos = int(data[8])						#Number of lines-of-sight
x_initial = 9
print(Nbins)
vel_axis = data[(x_initial+Nbins):(x_initial+2*Nbins)]#Hubble velocity along LoS in km/s
#lam   = lambda_obs(z,vel_axis*1.e5)
#freq  = freq_obs(z,vel_axis*1.e5)
freq = instrumental_features.freq_obs(z,vel_axis*1e5)
redsh = instrumental_features.z_obs(z,vel_axis*1e5)


datafile = str('%stau/tau_200cMpclong_n%d_z%.3f_fX%.1f_dv%d_file%d.dat' %(path,Nlos,z_name,fX_name,dvH,filenum))
data = np.fromfile(str(datafile),dtype=np.float32)
tau = np.reshape(data,(Nlos,Nbins))[LOS]

signal_ori = instrumental_features.transF(tau)

fig = plt.figure(figsize=(8.,4.75))
gs = gridspec.GridSpec(1,1)
ax0 = plt.subplot(gs[0,0])

ax0.plot(freq/1e6,signal_ori,'-',color='black',label=r'$F_{\rm ori}$')
#ax0.legend(loc='lower left',frameon=False,fontsize=fsize,ncol=3)
ax0.set_xlim(freq[-1]/1e6,freq[0]/1e6)
ax0.set_ylim(0.94,1.005)
ax0.set_yticks(np.arange(0.94,1.005,0.02))
ax0.set_xlabel(r'$\nu_{obs}\ [\rm MHz]$', fontsize=fsize)
ax0.set_ylabel(r'$F=e^{-\tau_{21}}$', fontsize=fsize)
ax0.xaxis.set_minor_locator(AutoMinorLocator())
ax0.yaxis.set_minor_locator(AutoMinorLocator())
ax0.tick_params(axis='x',which='major',direction='in',bottom=True,top=True,left=True,right=True
		,length=10,width=1,labelsize=fsize)
ax0.tick_params(axis='y',which='major',direction='in',bottom=True,top=True,left=True,right=True
		,length=10,width=1,labelsize=fsize)
ax0.tick_params(axis='both',which='minor',direction='in',bottom=True,top=True,left=True,right=True
		,length=5,width=1)

#ax0.set_title(r'$\mathrm{log}f_{\rm X}=%.1f, S_{147\mathrm{MHz}}=%.1f\,\mathrm{mJy},\ \alpha_{\mathrm{R}}=%.2f,\ %s:\ \Delta\nu=%d\,\mathrm{kHz},\ t_{\mathrm{int}}=%d\mathrm{hr},\ \rm LOS-%d$' % (fX_name,S_min_QSO,alpha_R,telescope,spec_res,t_int,LOS_ori),fontsize=fsize)
#ax0.text(5.95,0.965,r'SNR=%.2f' % SNR,fontsize=fsize)
'''
freq_min = constants.c/constants.lambda_0/1.e6/(1.+z_max)
freq_max = constants.c/constants.lambda_0/1.e6/(1.+z_min)
freq_label = np.round(np.arange(np.ceil(freq_min),np.floor(freq_max)+0.01,2.),0)
freq_posit = (constants.c/constants.lambda_0/1.e6/freq_label-1.-z_min)/(z_max-z_min)
ax0up = ax0.twiny()
ax0up.set_xlabel(r'$\nu_{obs}\ [\rm MHz]$', fontsize=fsize)
ax0up.set_xticks(freq_posit)
ax0up.set_xticklabels(freq_label)
ax0up.tick_params(axis='x',which='major',direction='in',bottom=False,top=True,left=True,right=True
		,length=10,width=1,labelsize=fsize)

freq_label = np.arange(np.ceil(freq_min*10)/10+0.4,np.floor(freq_max*10)/10+0.01,0.5)
freq_posit = (constants.c/constants.lambda_0/1.e6/freq_label-1.-z_min)/(z_max-z_min)
ax0upmin = ax0.twiny()
ax0upmin.set_xticks(freq_posit)
ax0upmin.tick_params(axis='x',which='major',direction='in',bottom=False,top=True,left=True,right=True
		,length=5,width=1,labeltop=False)
'''
plt.tight_layout()
#plt.subplots_adjust(hspace=.0)
plt.savefig('../../spectra/method_present/spectrum_ori_%dcMpc_z%.1f_fX%s_LOS%d.png' % (Lbox,z,fX_name,LOS_ori))
plt.show()



freq_smooth,signal_smooth = instrumental_features.smooth_fixedbox(freq,signal_ori,spec_res)

fig = plt.figure(figsize=(8.,4.75))
gs = gridspec.GridSpec(1,1)

ax0 = plt.subplot(gs[0,0])

#ax0.plot(redsh,signal_ori,'-',color='royalblue',label=r'$F_{\rm ori}$')
ax0.plot(freq_smooth/1e6,signal_smooth,'-',color='black',label=r'$F_{\rm sig}$')
#ax0.legend(loc='lower left',frameon=False,fontsize=fsize,ncol=3)
ax0.set_xlim(freq_smooth[-1]/1e6,freq_smooth[0]/1e6)
ax0.set_ylim(0.94,1.005)
ax0.set_yticks(np.arange(0.94,1.005,0.02))
ax0.set_xlabel(r'$\nu_{obs}\ [\rm MHz]$', fontsize=fsize)
ax0.set_ylabel(r'$F=e^{-\tau_{21}}$', fontsize=fsize)
ax0.xaxis.set_minor_locator(AutoMinorLocator())
ax0.yaxis.set_minor_locator(AutoMinorLocator())
ax0.tick_params(axis='x',which='major',direction='in',bottom=True,top=True,left=True,right=True
		,length=10,width=1,labelsize=fsize)
ax0.tick_params(axis='y',which='major',direction='in',bottom=True,top=True,left=True,right=True
		,length=10,width=1,labelsize=fsize)
ax0.tick_params(axis='both',which='minor',direction='in',bottom=True,top=True,left=True,right=True
		,length=5,width=1)

#ax0.set_title(r'$\mathrm{log}f_{\rm X}=%.1f, S_{147\mathrm{MHz}}=%.1f\,\mathrm{mJy},\ \alpha_{\mathrm{R}}=%.2f,\ %s:\ \Delta\nu=%d\,\mathrm{kHz},\ t_{\mathrm{int}}=%d\mathrm{hr},\ \rm LOS-%d$' % (fX_name,S_min_QSO,alpha_R,telescope,spec_res,t_int,LOS_ori),fontsize=fsize)
#ax0.text(5.95,0.965,r'SNR=%.2f' % SNR,fontsize=fsize)
'''
freq_min = c/lambda_0/1.e6/(1.+z_max)
freq_max = c/lambda_0/1.e6/(1.+z_min)
freq_label = np.round(np.arange(np.ceil(freq_min),np.floor(freq_max)+0.01,2.),0)
freq_posit = (c/lambda_0/1.e6/freq_label-1.-z_min)/(z_max-z_min)
ax0up = ax0.twiny()
ax0up.set_xlabel(r'$\nu_{obs}\ [\rm MHz]$', fontsize=fsize)
ax0up.set_xticks(freq_posit)
ax0up.set_xticklabels(freq_label)
ax0up.tick_params(axis='x',which='major',direction='in',bottom=False,top=True,left=True,right=True
		,length=10,width=1,labelsize=fsize)

freq_label = np.arange(np.ceil(freq_min*10)/10+0.4,np.floor(freq_max*10)/10+0.01,0.5)
freq_posit = (c/lambda_0/1.e6/freq_label-1.-z_min)/(z_max-z_min)
ax0upmin = ax0.twiny()
ax0upmin.set_xticks(freq_posit)
ax0upmin.tick_params(axis='x',which='major',direction='in',bottom=False,top=True,left=True,right=True
		,length=5,width=1,labeltop=False)
'''
plt.tight_layout()
#plt.subplots_adjust(hspace=.0)
plt.savefig('../../spectra/method_present/spectrum_smooth_%dcMpc_z%.1f_fX%s_%s_%dkHz_LOS%d.png' % (Lbox,z,fX_name,telescope,spec_res,LOS_ori))
plt.show()



N_d = 26
signal_withnoise = signal_smooth+instrumental_features.add_noise(freq_smooth,telescope,spec_res,S_min_QSO,alpha_R,t_int,N_d)
#print(bleh)
fig = plt.figure(figsize=(8.,4.75))
gs = gridspec.GridSpec(1,1)

z_min = redsh[-1]
v_21cm_z6 = constants.c/constants.lambda_0/7./1e6
S_21cm_z6 = S_min_QSO*np.power(v_21cm_z6/147.,alpha_R)
print('S_21cm(z=6.0) = %.2fmJy' % S_21cm_z6)
v_21cm_z6 = constants.c/constants.lambda_0/(z_min+1)/1e6
S_21cm_z6 = S_min_QSO*np.power(v_21cm_z6/147.,alpha_R)
print('S_21cm(z=%.1f) = %.2fmJy' % (z_min,S_21cm_z6))

ax0 = plt.subplot(gs[0,0])

#ax0.plot(redsh,signal_ori,'-',color='royalblue',label=r'$F_{\rm ori}$')
ax0.plot(freq_smooth/1e6,signal_withnoise,'-',color='black',label=r'$F_{\rm obs}$')
ax0.plot(freq_smooth/1e6,signal_smooth,'--',color='darkorange',label=r'$F_{\rm sig}$')
ax0.legend(loc='lower left',frameon=False,fontsize=fsize,ncol=1)
ax0.set_xlim(freq_smooth[-1]/1e6,freq_smooth[0]/1e6)
ax0.set_ylim(0.96,1.03)
ax0.set_yticks(np.arange(0.96,1.03,0.02))
ax0.set_xlabel(r'$\nu_{obs}\ [\rm MHz]$', fontsize=fsize)
ax0.set_ylabel(r'$F=e^{-\tau_{21}}$', fontsize=fsize)
ax0.xaxis.set_minor_locator(AutoMinorLocator())
ax0.yaxis.set_minor_locator(AutoMinorLocator())
ax0.tick_params(axis='x',which='major',direction='in',bottom=True,top=True,left=True,right=True
		,length=10,width=1,labelsize=fsize)
ax0.tick_params(axis='y',which='major',direction='in',bottom=True,top=True,left=True,right=True
		,length=10,width=1,labelsize=fsize)
ax0.tick_params(axis='both',which='minor',direction='in',bottom=True,top=True,left=True,right=True
		,length=5,width=1)

ax0.set_title(r'$S_{147\mathrm{MHz}}=%.1f\,\mathrm{mJy},\ \alpha_{\mathrm{R}}=%.2f,\ t_{\mathrm{int}}=%d\mathrm{hr}$' % (S_min_QSO,alpha_R,t_int),fontsize=fsize)
#ax0.text(5.95,0.965,r'SNR=%.2f' % SNR,fontsize=fsize)
'''
freq_min = c/lambda_0/1.e6/(1.+z_max)
freq_max = c/lambda_0/1.e6/(1.+z_min)
freq_label = np.round(np.arange(np.ceil(freq_min),np.floor(freq_max)+0.01,2.),0)
freq_posit = (c/lambda_0/1.e6/freq_label-1.-z_min)/(z_max-z_min)
ax0up = ax0.twiny()
ax0up.set_xlabel(r'$\nu_{obs}\ [\rm MHz]$', fontsize=fsize)
ax0up.set_xticks(freq_posit)
ax0up.set_xticklabels(freq_label)
ax0up.tick_params(axis='x',which='major',direction='in',bottom=False,top=True,left=True,right=True
		,length=10,width=1,labelsize=fsize)

freq_label = np.arange(np.ceil(freq_min*10)/10+0.4,np.floor(freq_max*10)/10+0.01,0.5)
freq_posit = (c/lambda_0/1.e6/freq_label-1.-z_min)/(z_max-z_min)
ax0upmin = ax0.twiny()
ax0upmin.set_xticks(freq_posit)
ax0upmin.tick_params(axis='x',which='major',direction='in',bottom=False,top=True,left=True,right=True
		,length=5,width=1,labeltop=False)
'''
plt.tight_layout()
#plt.subplots_adjust(hspace=.0)
plt.savefig('../../spectra/method_present/spectrum_noisy_%dcMpc_z%.1f_fX%s_%s_%dkHz_Smin%.1fmJy_alphaR%.2f_t%dh_LOS%d.png' % (Lbox,z,fX_name,telescope,spec_res,S_min_QSO,alpha_R,t_int,LOS_ori))
plt.show()