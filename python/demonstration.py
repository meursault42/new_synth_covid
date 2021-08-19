# -*- coding: utf-8 -*-
"""
Created on Mon Mar  8 15:43:09 2021

@author: u6026797
"""

#%% libraries
import os
import pandas as pd
import numpy as np
from SIR_model import SIR_generation
from gen_noise_final import QuasiPeriod
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

#
date_axis = list(df.columns)[-(len(input_data_df[input_data_df['index']=='Utah']['noise_seq'].to_numpy()[0])):]
plt.figure(figsize=(10,3))
plt.plot(date_axis,input_data_df[input_data_df['index']=='Utah']['noise_seq'].to_numpy()[0])
plt.title('Daily New Utah Covid Cases')
#%% generate SIR curves
output_sir_seqs, r0_seqs = SIR_generation(n_seqs=10, pop_size=1000000, 
                   best_est_params=[0.26, 0.20, 0.42, 0.12, 0.58, 0.19, 0.33, 0.53],
                   permute_params=False, intercept_dist=obs_Rt_intercept_vec,
                   min_outbreak_threshold=300,rt_method='Rtg',all_compartments=False)
#visualise infection curves
for i in output_sir_seqs:
    plt.plot(i)
plt.title('100 Sample SIR Curves')
#%% Sample noise generation code
qp = QuasiPeriod(likely_index=[5,6,7,8,9,11],sig_cutoff=0,verbose=False)

noised_sequences, mc_model, shuffle_indexes = qp.generate_noise_sequences(output_sir_seqs,
                                                                          subset_index=unique_index,
                                                                          input_data_df=input_data_df,
                                                                          white_noise=True)
plt.plot(output_sir_seqs[0])
plt.plot(noised_sequences[0])

mc_samples = mc_model.sample(10000)
import seaborn as sns
sns.histplot(mc_samples[0])
#%% Sample noise generation code with statistical censoring
qp = QuasiPeriod(likely_index=[5,6,7,8,9,11],sig_cutoff=0.09,verbose=False)

noised_sequences_2, mc_model_2, shuffle_indexes_2,iter_example = qp.generate_noise_sequences(output_sir_seqs,
                                                                          subset_index=unique_index,
                                                                          input_data_df=input_data_df,
                                                                          white_noise=True)

plt.plot(output_sir_seqs[1])
plt.plot(noised_sequences_2[1])


plt.plot(output_sir_seqs[4])
plt.plot(iter_example[14])
plt.plot(iter_example[13])
plt.plot(noise)

plt.figure(figsize=(23,3.7),dpi=400)
plt.plot(output_sir_seqs[4],linewidth=3,alpha=.9,label='COVID-19 SIR Simulation',linestyle='dashed',color='#003f72')
plt.plot(iter_example[14],linewidth=2,label='Batch Permutations',color='#0083be')
plt.plot(noise,linewidth=2,label='random_noise',color='#c4262e')
plt.plot(iter_example[13],linewidth=3,label='Quasi-Periodic Noise',alpha=.7,color='#557630')
plt.title('Example Noised Synthetic COVID-19 Curve')
plt.xlabel('Index')
plt.ylabel('Synthetic Daily New Reported Cases')
plt.legend()


#ind_vec = np.array([i for i in range(0,len(iter_example[13]))])
ind_vec = np.ones(len(iter_example[13]))
white_noise_seq = np.random.normal(0, 1, len(iter_example[13]))
noise = np.mean((iter_example[13]*2, ((iter_example[13]*white_noise_seq)*.03*2)),axis=0)

plt.plot(iter_example[13])
plt.plot(noise)

mc_samples = mc_model_2.sample(10000)
sns.histplot(mc_samples[0])
#%%
plt.plot(noised_sequences_2[0])
plt.plot(output_sir_seqs[0])

plt.plot(r0_seqs[0])

s=output_sir_seqs[0][1:]/output_sir_seqs[0][:-1]
s=s[15:]
r0_test = r0_seqs[0][15:]

plt.plot(s)
plt.plot(r0_test)

plt.plot(np.diff(s))
plt.plot(np.diff(r0_test))

plt.plot(np.diff(np.diff(r0_test)))
plt.plot(np.diff(np.diff(s)))

plt.plot(np.diff(np.diff(r0_test))**2)
plt.plot(np.diff(np.diff(s))**2)

plt.plot((np.diff(np.diff(r0_test))**2)*.001)
plt.plot(np.diff(np.diff(s))**2)

max((np.diff(np.diff(r0_test))**2)*.001)
max(np.diff(np.diff(s))**2)
#%% Several methods have been made independantly callable from the large wrapper
test_seq = input_data_df[input_data_df['index']=='Utah']['noise_seq'].to_numpy()[0]

#runs individual assessment on a single ts vector, returns fit information
qpfit = qp.emission_hmm(test_seq)

s_input_data_df = input_data_df.head()
s_input_data_df_ind = s_input_data_df['index'].to_list()
#runs fit assessment on group of vectors without generating noise sequences
model, mag_vec, dir_dict, n_vec = qp.get_hmm_outputs(subset_index=s_input_data_df_ind,
                                                              input_data_df=s_input_data_df)