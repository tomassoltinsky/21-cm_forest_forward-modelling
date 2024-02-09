"""
Generating 21cm forest 1D PS data for signal.

Version 15.11.2023
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



#path = 'data/'
path = '../../datasets/21cmFAST_los/'
z_name = float(sys.argv[1])
dvH = float(sys.argv[2])
spec_res = float(sys.argv[3])
n_los = int(sys.argv[4])
fX_name = float(sys.argv[5])
xHI_mean = float(sys.argv[6])

Nlos = 200
nfiles = 5

datafile = str('%slos_regrid/los_50Mpc_n%d_z%.3f_fX%.1f_xHI%.2f_dv%d_file%d.dat' % (path,Nlos,z_name,fX_name,xHI_mean,dvH,0))
data  = np.fromfile(str(datafile),dtype=np.float32)
z     = data[0]	#redshift
Nbins = int(data[7])					#Number of pixels/cells/bins in one line-of-sight
Nlos = int(data[8])						#Number of lines-of-sight
x_initial = 12
vel_axis = data[(x_initial+Nbins):(x_initial+2*Nbins)]#Hubble velocity along LoS in km/s
freq = instrumental_features.freq_obs(z,vel_axis*1e5)
tau = np.empty((Nlos*nfiles,Nbins))

for i in range(nfiles):

  datafile = str('%stau/tau_50Mpc_n%d_z%.3f_fX%.1f_xHI%.2f_dv%d_file%d.dat' %(path,Nlos,z_name,fX_name,xHI_mean,dvH,i))
  data = np.fromfile(str(datafile),dtype=np.float32)
  tau[i*Nlos:(i+1)*Nlos,:] = np.reshape(data,(Nlos,Nbins))

#tau = tau[:n_los,:-1]
#freq = freq[:-1]
tau = tau[:n_los,:]
Nbins = len(tau[0])
print('Number of pixels (original): %d' % Nbins)
signal_ori = instrumental_features.transF(tau)



freq_smooth = instrumental_features.smooth_fixedbox(freq,signal_ori[0],spec_res)[0]
bandwidth = (freq_smooth[-1]-freq_smooth[0])/1e6
print('Number of pixels (smoothed): %d' % len(freq_smooth))
print('Bandwidth = %.2fMHz' % bandwidth)

n_kbins = int((len(freq_smooth)/2+1))
PS_signal = np.empty((n_los,n_kbins))

done_perc = 0.1

for j in range(n_los):

  signal_smooth = instrumental_features.smooth_fixedbox(freq,signal_ori[j],spec_res)[1]
  k,PS_signal[j,:] = PS1D.get_P(signal_smooth,bandwidth)

  done_LOS = (j+1)/n_los
  if done_LOS>=done_perc:
    print('Done %.2f' % done_LOS)
    done_perc = done_perc+0.1

array = np.append(n_los,n_kbins)
array = np.append(array,k)
array = np.append(array,PS_signal)
array.astype('float32').tofile('1DPS_signal/power_spectrum_signal_21cmFAST_50Mpc_z%.1f_fX%.2f_xHI%.2f_%dkHz_%dLOS.dat' % (z,fX_name,xHI_mean,spec_res,n_los),sep='')

stop_clock = time.perf_counter()
time_taken = (stop_clock-start_clock)
print('It took %.3fs to run' % time_taken)