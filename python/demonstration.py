# -*- coding: utf-8 -*-
"""
Created on Mon Mar  8 15:43:09 2021

@author: u6026797
"""

#%% libraries
#os.chdir('C:\\Users\\u6026797')
#os.chdir('C:\\Users\\u6026797\\Documents\\GitHub\\synth_covid\\python')
import os
import pandas as pd
import numpy as np
from SIR_model import SIR_generation
from gen_noise_profile import get_hmm_outputs, generate_noise_sequences
import matplotlib.pyplot as plt


obs_Rt_intercept_vec = [1.193625387,1.830699913,2.074106191,2.178497481,3.326645157,
         2.063410566,2.701420497,3.564527643,2.321210991,3.00477709,2.217962438,
         8.182968952,2.197790699,2.229650391,1.905533268,3.322893953,2.996588077,
         4.348178948,2.207955707,2.789676393,1.93297168,2.033708206,3.189707307,
         3.160619937,3.046097364,2.187819404,1.922545614,1.949726977,3.130756605,
         2.429033112,3.023799131,3.252698197,2.424760803,2.691291283,3.481094508,
         3.101700772,1.842497079,2.122283903,2.682096586,2.291348801,3.071327953,
         3.617680439,1.867263927,2.810321986,3.170675356,2.254108549,2.222822926,
         5.927286426,7.629853829,2.216177275,3.78026411,2.561284556]
#%% get covid data to extract noise 
url = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_US.csv'
df = pd.read_csv(url, index_col=0)

#create index
unique_index = df['Province_State'].unique()
#aggregate to desired level
agg_df = df.groupby(['Province_State']).sum()
#create output df
input_data_df = pd.DataFrame()
#change columnar observations to single vector
for loc in unique_index:
    row_ind = agg_df[agg_df.index==loc].index[0]
    row_vals = agg_df[agg_df.index==loc].iloc[0,4:].to_numpy()
    #change from total to daily counts
    row_vals = np.diff(row_vals)
    #coerce negatives to 0s
    row_vals[row_vals<1]=1
    #trim sequences to the first observation above 5
    if len(row_vals[row_vals>5])!=0:
        start_mask = [i for i,v in enumerate(row_vals >=5) if v][0]
        row_vals = row_vals[start_mask:]
        input_data_df=input_data_df.append({'index' : row_ind, \
                                            'noise_seq' : row_vals}, ignore_index=True)

#remove cruiseship data
input_data_df=input_data_df.drop([input_data_df[input_data_df['index']=='Diamond Princess'].index[0],])
input_data_df=input_data_df.drop([input_data_df[input_data_df['index']=='Grand Princess'].index[0],])

#update index
unique_index = input_data_df['index'].unique()

plt.figure(figsize=(10,3))
plt.plot(input_data_df[input_data_df['index']=='Utah']['noise_seq'].to_numpy()[0])
plt.title('Utah Covid Cases')

plt.figure(figsize=(10,3))
plt.plot(np.diff(input_data_df[input_data_df['index']=='Utah']['noise_seq'].to_numpy()[0]))
plt.title("Utah Covid Cases (diff'd)")

fig, ax = plt.subplots(1, 1, figsize=(10, 3))
plot_pacf(np.diff(input_data_df[input_data_df['index']=='Utah']['noise_seq'].to_numpy()[0]),
          ax=ax)
plt.plot(x)
#%% generate SIR curves
#infection rate = 0.26
#recovery rate = 0.20
#asymptomatic infection rate = 0.42
#asymptomatic test rate = 0.12
#symptomatic test rate = 0.58
#asmyptomatic recovery rate = 0.19
#asymptomatic transmission rate = 0.33
#symptomatic transmission rate = 0.53

output_sir_seqs = SIR_generation(n_seqs=100, pop_size=1000000, 
                   best_est_params=[0.26, 0.20, 0.42, 0.12, 0.58, 0.19, 0.33, 0.53],
                   permute_params=False, intercept_dist=obs_Rt_intercept_vec,
                   min_outbreak_threshold=200,rt_method='Rtg',all_compartments=False)
#visualise infection curves
for i in output_sir_seqs:
    plt.plot(i)
plt.title('100 Sample SIR Curves')

model,mag_vec,dir_dict, n_list=get_hmm_outputs(subset_index=unique_index,input_data_df=input_data_df,n_order=0)

post_samp = []
for i in range(100):
    hmm_sample=model.sample(35)
    post_samp.append(hmm_sample[0])
post_samp = np.concatenate(post_samp)

#create two hist plots to compare sampled distribution to obs dist
fig, ax = plt.subplots(2,1,figsize=(12,8))
ax[0]=sns.histplot(post_samp,ax=ax[0],bins=9)
ax[0].set(title='HMM samples of mean length for 56 locations [4th order]')
ax[1]=sns.histplot(emm_vec,ax=ax[1],bins=9)
ax[1].set(title='Observed Periodicity from windowed PACF on all States/Territories [4th order]')
#%% Sample noise generation code
#generate sample sequences
#input_ts_list=[]
#for j in range(0,100,1):
#    input_ts_list.append(np.hstack([i for i in range(0,365,1)]))

output_seq, m, shuffles = generate_noise_sequences(input_ts_list=output_sir_seqs, 
                         subset_index=unique_index,
                         input_data_df=input_data_df,
                         n_periodic_components=1, white_noise=False, 
                         white_noise_weight = 0.05, shuffle_permute=True,
                         shuffle_prob = 30, smooth_transitions=False)

plt.plot(output_seq[420])
for i in range(0,50):
    plt.plot(output_seq[i])
plt.title('50 Sample Noised SIR Curves')
#%% Visualize sample noised curves

output_list_seq_df = pd.DataFrame(columns = ['Country_Region','Dt','NewPat','Oracle_kz','shuffles'])
for i in range(0,len(output_sir_seqs)):    
    output_list_seq_df=output_list_seq_df.append({'Country_Region' : i, \
                              'Dt' : np.array([j for j in range(1,1001)]),\
                              'NewPat' : output_seq[i],\
                              'Oracle_kz' : output_sir_seqs[i],
                              'shuffles' : shuffles[i]}, ignore_index=True)
output_list_seq_df.to_pickle('realistic_sir_curves.pkl',protocol=4)


plt.plot(output_list_seq_df.head(1)['NewPat'].to_numpy()[0])
plt.plot(output_list_seq_df.head(1)['Oracle_kz'].to_numpy()[0])
plt.title('Sample Individiual SIR Curve+Noise')

Tot = 10
Cols = 2
Rows = Tot // Cols 
Rows += Tot % Cols

# Create a Position index
plt.figure(figsize=(10,8))
Position = range(1,Tot + 1)
fig = plt.figure(1)
for k in range(Tot):
  ax = fig.add_subplot(Rows,Cols,Position[k])
  ax.plot(output_seq[k],label='noised sequence')
  ax.plot(output_sir_seqs[k],label='Simulated SIR curve')
ax.legend()
fig.suptitle('Simulated SIR modelled outbreaks with Modelled Noise')
