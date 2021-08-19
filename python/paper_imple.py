# -*- coding: utf-8 -*-
"""
Created on Fri Jun  4 12:29:12 2021

@author: u6026797
"""

#%% Libraries
import numpy as np
import pandas as pd
import math 
import matplotlib.pyplot as plt
from scipy.fftpack import fft, dct
from scipy import signal
from scipy.fft import fftshift
from scipy.signal import stft 
from statsmodels.tsa.stattools import acf

def autocorrelation(x):
    """
    Compute autocorrelation using FFT
    The idea comes from 
    http://dsp.stackexchange.com/a/1923/4363 (Hilmar)
    """
    x = np.asarray(x)
    N = len(x)
    x = x-x.mean()
    s = np.fft.fft(x, N*2-1)
    result = np.real(np.fft.ifft(s * np.conjugate(s), N*2-1))
    result = result[:N]
    result /= result[0]
    return result

def moving_average(a,n=7):
    m = int(n/2)
    MA = []
    for i in range(0,len(a)):
        pl = i-m
        ph=i+m
        if(pl < 0):
            pl = 0
        if(ph > len(a)-1):
            ph = len(a)-1
        MA.append(np.average(a[pl:ph+1]))
    return np.array(MA)

#%% Sample Data
test_seq = (5*(np.sin((2*math.pi/250)*np.arange(1,1560)))*1)
white_noise_seq = np.random.normal(0, .1, test_seq.shape)
test_seq= np.mean((test_seq*2, ((test_seq*white_noise_seq)*0.2*2)),axis=0)

#this comes from tsdl 271
#Hipel and McLeod (1994), Mean daily temperature, Fisher River near Dallas, Jan 01, 1988 to Dec 31, 1991
test_seq= pd.read_csv('sample_ts.csv')
test_seq = test_seq['x']

#smoothing
test_seq = moving_average(test_seq,90)
plt.plot(test_seq)

#truncation
test_seq = test_seq[60:]
#%% step 1
win=signal.get_window('hann',256)
win=signal.get_window('boxcar',256)
#adjust for true rectangular
#win[0]=0
#win[255]=0
#plt.plot(win)
f,t,Zxx = stft(test_seq,window=win,noverlap=255,padded=False)
#%% plot 1a
#as stft bins
#FFT_Index = np.arange(0,129)
plt.pcolormesh(t, FFT_Index, np.abs(Zxx), vmin=0, vmax=0.1, shading='gouraud')
plt.title('STFT Magnitude')
plt.ylabel('FFT Index')
plt.xlabel('Time [sec]')
#plt.show()

#%% step 2
#output_vec = np.array(0)
output_list = list()
for i in range(1,Zxx.shape[0]):
    #calc discrete cos t to partially decorrelate stft series,
    #de_corr_vec=dct(Zxx[i,:],norm='ortho')
    #V1 decor and auto
    
    #de_corr_vec=dct(Zxx[i,:],norm='ortho')
    #de_corr_vec=dct(Zxx[i,:])
    #ac = autocorrelation(de_corr_vec)
    #output_list.append(ac)
    
    #V2 without dct decor
    ac = autocorrelation(Zxx[i,:])
    output_list.append(ac)
    
output_df = pd.DataFrame(output_list)
output_df = output_df.sum()
output_vec = output_df.to_numpy()

plt.plot(output_vec)
#plt.pcolormesh(t, FFT_Index, output_df, vmin=0, vmax=.001, shading='gouraud')
#plt.title('STFT Magnitude')
#plt.ylabel('FFT Index')
#plt.xlabel('Time [sec]')
#plt.show()

#np.where(output_vec==max(output_vec[128:(len(output_vec)-256)]))
plt.plot(output_vec)
plt.plot(output_vec[128:(len(output_vec)-256)])

output_df = output_df.sum()
output_vec = output_df.to_numpy()


#%% plot 1b
p = np.where(output_vec==max(output_vec[128:(len(output_vec)-256)]))[0][0]

plt.plot(output_vec[1:])

plt.plot(test_seq)
plt.vlines(p,ymin=min(test_seq),ymax=max(test_seq))
plt.vlines(p*2,ymin=min(test_seq),ymax=max(test_seq))
plt.vlines(p*3,ymin=min(test_seq),ymax=max(test_seq))
plt.vlines(p*4,ymin=min(test_seq),ymax=max(test_seq))

plt.plot(test_seq)
plt.vlines(359,ymin=30,ymax=-30)
plt.vlines(718,ymin=30,ymax=-30)
plt.vlines(1077,ymin=30,ymax=-30)
plt.vlines(1436,ymin=30,ymax=-30)


np.where(output_vec == max(output_vec[1:]))
#%% wrap

def pms_stationary_detect(input_seq,buffer=True,window=256,output='vec'):
    win=signal.get_window('boxcar',window)
    win[0]=0
    win[(window-1)]=0
    f,t,Zxx = stft(input_seq,window=win,noverlap=(window-1),
                   padded=False,nperseg=window)

    output_list = list()
    for i in range(0,Zxx.shape[0]):
        ac = autocorrelation(Zxx[i,:])
        output_list.append(ac)

    output_df = pd.DataFrame(output_list)
    output_df = output_df.sum()
    output_vec = output_df.to_numpy()
    
    if buffer==True:
        period=np.where(output_vec==max(output_vec[(int(window/2)):]))[0][0]
    elif buffer==False:
        period=np.where(output_vec==max(output_vec))[0][0]
    
    if output=='vec':
        return output_vec
    elif output=='index':
        return period

pms_stationary_detect(test_seq)
    
#%% Sample Data #20
test_seq_a = (0.5*(np.cos((2*math.pi/128)*np.arange(1,915))))
#plt.plot(test_seq_a)
#pms_stationary_detect(test_seq_a[64:564],window=500,output='vector')
#plt.plot(pms_stationary_detect(test_seq_a[64:564],window=256))
test_seq_b = (0.5*(np.cos((2*math.pi/64)*np.arange(1,915))))
test_seq_c = (0.5*(np.cos((2*math.pi/32)*np.arange(1,915))))
test_seq_2 = np.append(np.append(test_seq_c,test_seq_b),test_seq_a)
plt.plot(test_seq_2)

plt.plot(np.arange(2672)/1000,test_seq_2)
plt.ylabel('Value')
plt.xlabel('Time (s)')

#now we implement a simple window function 
#%%
def naive_pms_window(input_seq,window=256,overlap=32,four_window=130):
    output_list = list()
    iters= np.floor(len(input_seq)/overlap)
    for i in np.arange(0,iters):
        ind = int(i*overlap)
        if (ind+window)<len(input_seq)-1:
            snippet = input_seq[ind:(ind+window)]
            out_value = pms_stationary_detect(snippet,window=four_window,output='index')
            output_list.append((ind,out_value))
            print(ind)
        else:
            snippet = input_seq[ind:(len(input_seq)-1)]
            out_value = pms_stationary_detect(snippet,window=len(snippet-1),output='index')
            output_list.append((ind,out_value))
            
    return(output_list)
        
out_test_data = naive_pms_window(test_seq_2)
out_test_data_df = pd.DataFrame(out_test_data)
plt.plot(out_test_data_df[0],out_test_data_df[1])

#dumb_test
plt.plot(test_seq_2[0:128])


#%%


