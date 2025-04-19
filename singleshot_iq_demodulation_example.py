'''

Title: Single-shot neural decoding with acoustic isolation file for comparison. 
Author: Jean Rintoul
Date: 25.04.04

'''
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import signal
from scipy.signal import iirfilter,sosfiltfilt
from scipy.fft import fft,fftshift
import scipy.stats
# 
# 
# Read in the files. 
fsignal = np.load('ae.npy')
iso_fsignal = np.load('ac_iso.npy')
# 
plt.rc('font', family='serif')
plt.rc('font', serif='Arial')
plt.rcParams['axes.linewidth'] = 2
fonts                          = 18
# 
# Change the folder to where you placed the files. 
Fs              = 2e6
gain            = 2000
hp_cut          = 5
result_lp_cut   = 40
timestep        = 1.0/Fs
duration        = 30
N               = int(Fs*duration)
carrier                     = 500000
specificity_test_carrier    = 270000
t               = np.linspace(0, duration, N, endpoint=False)
# 
# 
rbeta           = 12
raw_timestep    = 1/Fs 
rawN            = len(fsignal)
raw_window      = np.kaiser( (rawN), rbeta )
xf              = np.fft.fftfreq( (rawN), d=raw_timestep)[:(rawN)//2]
frequencies     = xf[1:(rawN)//2]

# filters to be used.  
sos_low_band    = iirfilter(17, [hp_cut,result_lp_cut], rs=60, btype='bandpass',
                       analog=False, ftype='cheby2', fs=Fs,
                       output='sos')
# First, band filter around the carrier. 
h_cut               = 1000
sos_demodulate_band = iirfilter(17, [carrier-h_cut,carrier+h_cut], rs=60, btype='bandpass',
                       analog=False, ftype='cheby2', fs=Fs,
                       output='sos')
# 
sos_demodulate_spec_band = iirfilter(17, [specificity_test_carrier-h_cut,specificity_test_carrier+h_cut], rs=60, btype='bandpass',
                       analog=False, ftype='cheby2', fs=Fs,
                       output='sos')

sos_dc_band = iirfilter(17, [0.5], rs=60, btype='lowpass',
                       analog=False, ftype='cheby2', fs=Fs,
                       output='sos')
# 
dc_offset   = sosfiltfilt(sos_dc_band,fsignal)
raw_data    = sosfiltfilt(sos_demodulate_band,fsignal)

# IQ demodulation
def demodulate(in_signal,carrier_f,t): 
    idown = in_signal*np.cos(2*np.pi*carrier_f*t)
    qdown = in_signal*np.sin(2*np.pi*carrier_f*t)  
    demodulated_signal = np.sqrt(idown*idown + qdown*qdown)
    return demodulated_signal

demod_iq = demodulate(raw_data,carrier,t) 
demod_iq = sosfiltfilt(sos_low_band, demod_iq)

spec_raw_data   = sosfiltfilt(sos_demodulate_spec_band,fsignal)
spec_demod      = demodulate(spec_raw_data,specificity_test_carrier,t)

spec_demod      = spec_demod - np.mean(spec_demod)
spec_demod      = sosfiltfilt(sos_low_band, spec_demod)
# 
#
iso_raw_data    = sosfiltfilt(sos_demodulate_band,iso_fsignal)
iso_demod_iq    = demodulate(iso_raw_data,carrier,t) 
iso_demod_iq    = sosfiltfilt(sos_low_band, iso_demod_iq)
# 
fft_rawk        = fft(fsignal*raw_window)
fft_rawk        = np.abs(2.0/(rawN) * (fft_rawk))[1:(rawN)//2]
# 
# 
# Region of time to plot in the graph. 
min_time = 16.5
max_time = 19.5
# 
# min_time = 0
# max_time = 30
# 
# 
# 5-40Hz. 
low_vep_signal              = sosfiltfilt(sos_low_band, fsignal)
low_vep_iso_signal          = sosfiltfilt(sos_low_band, iso_fsignal) 
# 
start                       = int(min_time*Fs)
end                         = int(max_time*Fs)
single_shot_veps            = low_vep_signal[start:end]  
single_shot_iqdemod         = demod_iq[start:end]
single_shot_t               = t[start:end]
single_shot_spec_demod      = spec_demod[start:end]  
# 
iq_single_shot_iso_demod    = iso_demod_iq[start:end]
single_shot_iso_veps        = low_vep_iso_signal[start:end]  
single_fsignal              = fsignal[start:end]

# 
# half a second rolling correlation window. 
window          = 500000
# 
# demodulation correlation test. 
x               = single_shot_veps - np.mean(single_shot_veps)
z               = single_shot_spec_demod - np.mean(single_shot_spec_demod)
b               = single_shot_iqdemod - np.mean(single_shot_iqdemod)
iso_x           = single_shot_iso_veps - np.mean(single_shot_iso_veps)
iq_iso_y        = iq_single_shot_iso_demod - np.mean(iq_single_shot_iso_demod )
# 
# iq iso correlation
iqisodf              = pd.DataFrame({'x': iso_x, 'y': iq_iso_y })
iq_iso_rolling_corr    = iqisodf['x'].rolling(window).corr(iqisodf['y'])
iso_iq_median_corr = np.nanmean(iq_iso_rolling_corr)
print ('mean iso iq ae neural recording corr: ',iso_iq_median_corr)
# 
# iq correlation
iqdf              = pd.DataFrame({'x': x, 'y': b })
iq_rolling_corr   = iqdf['x'].rolling(window).corr(iqdf['y'])
# What is a good correlation metric in this case? 
iq_median_corr = np.nanmean(iq_rolling_corr)
print ('mean iq ae neural recording corr: ',iq_median_corr)
# 
#   frequency specificity. 
df_spec              = pd.DataFrame({'x': x, 'y': z })
spec_rolling_corr    = df_spec['x'].rolling(window).corr(df_spec['y'])
spec_median_corr = np.nanmean(spec_rolling_corr)
print ('mean specificity test corr: ',spec_median_corr)
#     
#  
fig = plt.figure(figsize=(6,3))
ax  = fig.add_subplot(111)
plt.plot(single_shot_t,x/np.max(x),'r')
plt.plot(single_shot_t,b/np.max(b),'k')
ax.set_xlim([min_time,max_time])
# ax.set_xlim([subsection_start,subsection_end])
# ax.set_ylim([-4,4])
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.legend(['spontaneous','iq'],loc='upper right')
# ax.spines['left'].set_visible(False)
# ax.spines['bottom'].set_visible(False)
# plt.xticks([])
# plt.yticks([])
plt.title('spontaneous activity vs iq demodulated')
plt.tight_layout()
plot_filename = 'comparison.png'
plt.savefig(plot_filename)
plt.show()
#  
#    
subsection_start    = 16.5
subsection_end      = 19.5
#  
#  
# 
fig = plt.figure(figsize=(6,3))
ax  = fig.add_subplot(111)
plt.plot(single_shot_t,iso_x/np.max(iso_x),'r')
# plt.plot(single_shot_t,x,'r')
# plt.plot(single_shot_t,y/np.max(y),'k')
plt.plot(single_shot_t,iq_iso_y /np.max(iq_iso_y),'k')
# ax.set_xlim([min_time,max_time])
ax.set_xlim([subsection_start,subsection_end])
ax.set_ylim([-1,1])
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.legend(['spontaneous activity','ac iso demodulation'],loc='upper right')
# ax.spines['left'].set_visible(False)
# ax.spines['bottom'].set_visible(False)
# plt.xticks([])
# plt.yticks([])
plt.tight_layout()
plot_filename = 'iso_comparison.png'
plt.savefig(plot_filename)
plt.show()
# 
# 
# 
fig = plt.figure(figsize=(6,1.5))
ax  = fig.add_subplot(111)
# plt.plot(single_shot_t,rolling_corr,'k')
plt.plot(single_shot_t-subsection_start,iq_rolling_corr,'k')
# ax.set_xlim([min_time,max_time])
# ax.set_xlim([subsection_start,subsection_end])
ax.set_xlim([0,subsection_end-subsection_start])
ax.set_ylim([-1,1])
plt.yticks([-1,0,1])
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.xticks(fontsize=fonts)
plt.yticks(fontsize=fonts)
# plt.title('rolling correlation')
plt.tight_layout()
# ax.set_xlim([min_time,max_time])
plot_filename = 'rolling_correlation.png'
plt.savefig(plot_filename)
plt.show()
#     
# 
fig = plt.figure(figsize=(6,1.5))
ax  = fig.add_subplot(111)
# plt.plot(single_shot_t,rolling_corr,'k')
plt.plot(single_shot_t-subsection_start,iq_iso_rolling_corr,'k')
# ax.set_xlim([min_time,max_time])
# ax.set_xlim([subsection_start,subsection_end])
ax.set_xlim([0,subsection_end-subsection_start])
ax.set_ylim([-1,1])
plt.yticks([-1,0,1])
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.xticks(fontsize=fonts)
plt.yticks(fontsize=fonts)
plt.tight_layout()
# ax.set_xlim([min_time,max_time])
# plt.legend(['hilbert','iq'],loc='upper right')
plot_filename = 'rolling_iso_correlation.png'
plt.savefig(plot_filename)
plt.show()
#      
#      
#  