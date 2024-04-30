"""
Generating 21cm forest 1D PS data for noise.

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

start_clock = time.perf_counter()

#constants

import instrumental_features
import PS1D

path = 'data/'
z_name = float(sys.argv[1])
dvH = float(sys.argv[2])
telescope = str(sys.argv[3])
spec_res = float(sys.argv[4])
S_min_QSO = float(sys.argv[5])
alpha_R = float(sys.argv[6])
N_d = float(sys.argv[7])
t_int = float(sys.argv[8])
n_los = 1000
fX_name = -2.
mean_xHI = 0.31
Nlos = 200

datafile = str('%slos_regrid/los_50Mpc_n%d_z%.3f_fX%.1f_xHI%.2f_dv%d_file%d.dat' % (path,Nlos,z_name,fX_name,mean_xHI,dvH,0))
data  = np.fromfile(str(datafile),dtype=np.float32)
z     = data[0]	#redshift
Nbins = int(data[7])					#Number of pixels/cells/bins in one line-of-sight
Nlos = int(data[8])						#Number of lines-of-sight
x_initial = 12
vel_axis = data[(x_initial+Nbins):(x_initial+2*Nbins)]#Hubble velocity along LoS in km/s
freq = instrumental_features.freq_obs(z,vel_axis*1e5)

freq_smooth = instrumental_features.smooth_fixedbox(freq,freq,spec_res)[0]
bandwidth = (freq_smooth[-1]-freq_smooth[0])/1e6
print('Bandwidth = %.2fMHz' % bandwidth)

n_kbins = int((len(freq_smooth)/2+1))
PS_noise = np.empty((n_los,n_kbins))

done_perc = 0.1

for j in range(n_los):

  noise = instrumental_features.add_noise(freq_smooth,telescope,spec_res,S_min_QSO,alpha_R,t_int,N_d)
  k,PS_noise[j,:] = PS1D.get_P(1.+noise,bandwidth)

  done_LOS = (j+1)/n_los
  if done_LOS>=done_perc:
    print('Done %.2f' % done_LOS)
    done_perc = done_perc+0.1

array = np.append(n_kbins,k)
array = np.append(array,PS_noise)
array.astype('float32').tofile('1DPS_noise/power_spectrum_noise_50Mpc_z%.1f_%s_%dkHz_Smin%.1fmJy_alphaR%.2f_t%dh.dat' % (z,telescope,spec_res,S_min_QSO,alpha_R,t_int),sep='')

stop_clock = time.perf_counter()
time_taken = (stop_clock-start_clock)
print('This took %.3fs of your life' % time_taken)