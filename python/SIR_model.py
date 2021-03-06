# -*- coding: utf-8 -*-
"""
Created on Fri Mar  5 16:10:45 2021

@author: u6026797
"""

#%% libraries
import pandas as pd
import numpy as np
from numpy import exp
#import matplotlib.pyplot as plt
import math 
#import seaborn as sns
import random
from scipy.integrate import odeint
from sklearn import preprocessing
from copy import deepcopy

#%% init params / data
pop_size = 1000000

i_0 = 1e-6
ai_0 = 1e-6
e_0 = 4 * (i_0+ai_0)
s_0 = 1 - (i_0+ai_0) - e_0
d_0 = 0
ot_0 = 0
r_0 = 0

x_0 = s_0, e_0, ai_0, i_0, ot_0, d_0, r_0

t_length = 550
grid_size = 545
t_vec = np.linspace(0, t_length, grid_size)
t_vec = np.array(range(0,t_length))

obs_Rt_intercept_vec = [1.193625387,1.830699913,2.074106191,2.178497481,3.326645157,
         2.063410566,2.701420497,3.564527643,2.321210991,3.00477709,2.217962438,
         8.182968952,2.197790699,2.229650391,1.905533268,3.322893953,2.996588077,
         4.348178948,2.207955707,2.789676393,1.93297168,2.033708206,3.189707307,
         3.160619937,3.046097364,2.187819404,1.922545614,1.949726977,3.130756605,
         2.429033112,3.023799131,3.252698197,2.424760803,2.691291283,3.481094508,
         3.101700772,1.842497079,2.122283903,2.682096586,2.291348801,3.071327953,
         3.617680439,1.867263927,2.810321986,3.170675356,2.254108549,2.222822926,
         5.927286426,7.629853829,2.216177275,3.78026411,2.561284556]
#%% functions
def _F(x, t, init_params, R0=1.6):
    '''
    SIR model used for generating disease outbreak compartments. 

    Parameters
    ----------
    x : list
        Holds components of SIR model for iteration
    t : np array
        Outbreak sequence length.
    init_params : vector of floats
        Coefficients for SIR model
    R0 : float, optional
        value of R0 at time t. Defaults to static R0 value of 1.6.

    Returns
    -------
    ds : TYPE
        DESCRIPTION.
    de : TYPE
        DESCRIPTION.
    dai : TYPE
        DESCRIPTION.
    di : TYPE
        DESCRIPTION.
    dot : TYPE
        DESCRIPTION.
    dd : TYPE
        DESCRIPTION.
    dr : TYPE
        DESCRIPTION.

    '''
    s, e, ai, i, ot, d, r = x
    
    infection_rate = init_params[0]
    recovery_rate = init_params[1]
    mortality_rate = 0.009
    asymptomatic_rate = 0.3
    a_infection_rate = init_params[2]
    a_test_rate = init_params[3]
    test_rate = init_params[4]
    a_recovery_rate = init_params[5]
    a_trans_rate = init_params[6]
    trans_rate = init_params[7]

    # New exposure of susceptibles
    transmission_rate = R0(t) if callable(R0) else R0 
    new_exposed = (transmission_rate * trans_rate * i) + (transmission_rate *a_trans_rate * ai) * s 

    # Time derivatives
    ds = - new_exposed
    de = new_exposed - (infection_rate * e) - (a_infection_rate * e)
    #asymptomatic cases
    dai = asymptomatic_rate * a_infection_rate * e - a_recovery_rate * ai
    #symptomatic cases
    di = (1- asymptomatic_rate) * infection_rate * e - recovery_rate * i    
    #observerable test
    #dai -> ai
    dot = (a_test_rate * ai) + (test_rate * i) - ot
    dd = mortality_rate * i - d
    #dai -> ai
    dr = ((1 - mortality_rate) * i) + (a_recovery_rate * ai) - r
    
    return ds, de, dai, di, dot, dd, dr

def _solve_path(R0, t_vec, init_params = [1/5.2,1/18,1/20], x_init=x_0):
    """
    Solve for i(t) and c(t) via numerical integration,
    given the time path for R0.
    """
    G = lambda x, t: _F(x, t, init_params, R0)
    s_path, e_path, ai_path, i_path, ot_path, d_path, r_path = odeint(G, x_init, t_vec).transpose()

    c_path = 1 - s_path - e_path - r_path      # cumulative cases
    #return i_path, c_path, d_path, ot_path
    return s_path, e_path, ai_path, i_path, c_path, ot_path, d_path, r_path 

def Rtp_mitigating(t, r0=3, intervention=0.007, r_bar=0.8, fpc=8, spc=100, fcw=.05, scw = 0.2):
    '''
    Function for generating Rt sequence with periodic noise.

    Parameters
    ----------
    t : np array
        placeholder vector for sequence length
    r0 : float, optional
        Intercept for start of Rt. The default is 3.
    intervention : float, optional
        Value for changing the slope for the exponential function. 
        The default is 0.007.
    r_bar : float, optional
        Value for tail of exponential function. The default is 0.8.
    fpc : int, optional
        Width of fast periodic component window. The default is 8.
    spc : int, optional
        Width of slow periodic component window. The default is 100.
    fcw : int, optional
        Weight applied to fast periodic component. The default is .05.
    fcw : int, optional
        Weight applied to slow periodic component. The default is .05.

    Returns
    -------
    Rt : np.array
        Vector of simulated Rt.

    '''
    Rt = (r0*exp(-intervention*t)+(1-exp(-intervention*t))*r_bar) * \
        (1+(np.sin((2*math.pi/fpc)*t))*fcw) * \
        (1+(np.sin((2*math.pi/spc)*t))*scw)
    return Rt

def Rtg_mitigating(t,r0=3,intervention=.1,r_bar=.9,n_peaks=2):
    '''
    Function for generating Rt sequence with subsequent peaks of a width 
    and height defined by a gaussian distribution.

    Parameters
    ----------
    t : np array
        place holder vector for sequencel length.
    r0 : float, optional
        Intercept for start of Rt. The default is 3.
    intervention : float, optional
        Value for changing the slope for the exponential function. 
        The default is 0.007.
    r_bar : float, optional
        Value for tail of exponential function. The default is 0.8.
    n_peaks : integer, optional
        Number of desired subsequent peaks in Rt. The default is 2.

    Returns
    -------
    Rt: np.array
        Vector of simulated Rt.
    '''
    trunc_normal_dist=np.random.normal(2,.5,size=1000)
    trunc_normal_dist[trunc_normal_dist<.7]=.7
    trunc_normal_dist[trunc_normal_dist>3]=3
    
    Rt_exp = (r0*np.exp(-intervention*t)+(1-np.exp(-intervention*t))*r_bar)
    peak_dict=dict()
    for i in range(0,n_peaks):
        key_name='center_'+str(i)
        peak_dict[key_name]=random.sample(list(range(50,(len(t)-50))),1)
    #generate N dist centered at position n
    randomNums = np.random.normal(100,12,size=100000)
    #round to discretize
    randomInts = np.round(randomNums)
    #send to pd to aggregate
    randdf = pd.DataFrame({'Vals':randomInts,'count':1})
    randdf.groupby(['Vals']).sum()
    #rescale to make feasible 
    randdf = randdf.groupby(['Vals']).sum()
    x=randdf['count'].to_numpy()
    x=x.reshape((1,-1))
    x=preprocessing.normalize(x)[0]
    
    if len(x)%2!=0:
        start_p=-math.floor(len(x)/2)
        end_p=len(x)+start_p
    else:
        start_p = -len(x)/2
        end_p = len(x)/2
    for key,val in peak_dict.items():
        Rt_exp[int(val[0]+start_p):int(val[0]+end_p)]=Rt_exp[int(val[0]+start_p):int(val[0]+end_p)]+x*random.sample(trunc_normal_dist.tolist(),1)
    return Rt_exp#, peak_dict

def Rto_mitigating(t,Rt_external,Rt_local):
    '''
    Convenience function for extending other method of Rt estimation. Simply
    pass in another np.array with an estimated Rt and an Rt vector created via
    Rtg or Rtp_mitigating and it will combine the two.

    Parameters
    ----------
    t : np array
        place holder vector for sequencel length.
    Rt_external : np array
        A shape(x,) np array containing an external Rt sequence.
    Rt_local : np array
        A shape(x,) np array containing an internally dervied Rt sequence.

    Returns
    -------
    A concatenated Rt sequence.

    '''
    if type(t) == int or type(t) == float:
        if t >= (len(Rt_external)-1):
            return Rt_local
        if t < (len(Rt_external)-1):
            return Rt_external[round(t)]
    if len(t) > 1:
        Rt_local[0:len(Rt_external)]=Rt_external
        return Rt_local
    
def SIR_generation(n_seqs=1000, pop_size=1000000, 
                   best_est_params=[0.26, 0.20, 0.42, 0.12, 0.58, 0.19, 0.33, 0.53],
                   permute_params=True, intercept_dist=obs_Rt_intercept_vec,
                   min_outbreak_threshold=100,rt_method='Rtg',all_compartments=False,
                   **args):
    '''
    Wrapper function to generate SIR curves with realistic Rt.

    Parameters
    ----------
    n_seqs : int, optional
        Desired number of simulated outbreaks. The default is 1000.
    pop_size : int, optional
        Population size to simulate. The default is 1000000.
    best_est_params : list, optional
        List of SIR parameters. These are a combination of estimates from published
        research and estimates from fitting SIR models iteratively across a number of US states.
            infection rate = 0.26
            recovery rate = 0.20
            asymptomatic infection rate = 0.42
            asymptomatic test rate = 0.12
            symptomatic test rate = 0.58
            asmyptomatic recovery rate = 0.19
            asymptomatic transmission rate = 0.33
            symptomatic transmission rate = 0.53
    permute_params : Boolean, optional
        Option to randomly permute input parameters. The default is True.
    intercept_dist : List, optional
        A list of Rt intercept values. Derived from methods defined in Coby on all US states.
        The default is obs_Rt_intercept_vec defined at the top. You may provide a list
        of any other value to substitute these.
    min_outbreak_threshold : Boolean, optional
        Cutoff value for minimum outbreak threshold. Enable to require outbreaks
        to infect more than n patients. The default is 100.
    rt_method : str, optional
        Preferred method of calculating Rt. Options are:
            Rtg:
                Exponential + n randomly selected gaussian bumps
            Rtp:
                Exponential + periodic components generated from sin waves
            Rto:
                Exponential + overwrites to with preferred, precalculated rt estimate
    all_compartments : Boolean, optional
        Option to return all compartments from the SIR model. The default is false.
    **args : values to be passed into any subfunction
    
    Returns
    -------
    List of simulated Infected cases.

    '''
    
    output_list_seqs = []
    rt_test_seqs = []
    iter_val = 0
    while iter_val < n_seqs:    
        local_params = deepcopy(best_est_params)
        if permute_params==True:
            for i in range(0,8):
                local_params[i]=local_params[i]+np.random.normal(0,0.005,1)
        
        #test code
        
        intercept = random.sample(intercept_dist,1)[0]
        if rt_method=='Rtg':
            try: 
                rt_est_i=Rtg_mitigating(t=t_vec, intervention=0.05, 
                                      n_peaks=random.sample([2,3,4],1)[0],r0=intercept)
            except: 
                rt_est_i=Rtg_mitigating(t=t_vec, intervention=0.05, 
                                      n_peaks=random.sample([2,3,4],1)[0],r0=intercept)
        elif rt_method=='Rtp':
            rt_est_i=Rtp_mitigating(t=t_vec,r0=random.sample(intercept_dist,1)[0],
                                  scw=.2,fcw=.05,intervention=0.1,r_bar=.9)
        elif rt_method=='Rto':
            rt_est_i=Rto_mitigating(t=t_vec, Rt_external = args['Rt_external'],
                                  Rt_local = Rtp_mitigating(t=t_vec,r0=random.sample(intercept_dist,1)[0],
                                                               scw=0,fcw=0))
        rt_est = rt_est_i
        #R0(np.array(range(0,550)))
        def _R0_overwrite(t, r0=2, intervention=0.1, r_bar=0.9, fpc=8, spc=100, spcc=0, fpcc=0, rt_est=rt_est):
            R0 = (r0*exp(-intervention*t)+(1-exp(-intervention*t))*r_bar) * \
                (1+(np.sin((2*math.pi/fpc)*t))*fpcc) * \
                (1+(np.sin((2*math.pi/spc)*t))*spcc)
            #force overwrite
            if type(t) == int or type(t) == float:
                if t > (len(rt_est)-1):
                    return R0
                if t <= (len(rt_est)-1):
                    return rt_est[round(t)]
            if len(t) > 1:
                R0[0:len(rt_est)]=rt_est
                return R0
        R0 = lambda t: _R0_overwrite(t, rt_est=rt_est)
        
        s_path, e_path, ai_path, i_path, c_path, ot_path, d_path, r_path = _solve_path(R0, t_vec, init_params=local_params )
        compartment_dict = {}
        compartment_dict = {'susceptible':s_path, 'exposed':e_path, 'asymptomatic_infected':ai_path,
                            'symptomatic_infected':i_path, 'cumulative_infected':c_path,
                            'positive_tests':ot_path, 'deceased':d_path, 'recovered': r_path}
        out_dict = {}
        for key,val in compartment_dict.items():
            out_dict[key]= [path * pop_size for path in val]
            
        cases = np.array(out_dict['symptomatic_infected'])+np.array(out_dict['asymptomatic_infected'])        
        #if max(cases)<min_outbreak_threshold:
        #    print('total i < min_threshold')
        #if np.sum(cases)>pop_size:
            #print('Error in model specification: total i > pop_size {}'.format(local_params ))
        if np.sum(cases)<=pop_size and max(cases)>min_outbreak_threshold:
            if all_compartments==False:
                output_list_seqs.append(np.array(out_dict['positive_tests']))
                rt_test_seqs.append(rt_est)
                iter_val+=1
            if all_compartments==True:
                output_list_seqs.append(out_dict)
                iter_val+=1
        
    return output_list_seqs, rt_test_seqs
#%% generate sequences
'''
infection rate = 0.26
            recovery rate = 0.20
            asymptomatic infection rate = 0.42
            asymptomatic test rate = 0.12
            symptomatic test rate = 0.58
            asmyptomatic recovery rate = 0.19
            asymptomatic transmission rate = 0.33
            symptomatic transmission rate = 0.53

output_sir_seqs = SIR_generation(n_seqs=10, pop_size=1000000, 
                   best_est_params=[0.26, 0.20, 0.42, 0.12, 0.58, 0.19, 0.33, 0.53],
                   permute_params=False, intercept_dist=obs_Rt_intercept_vec,
                   min_outbreak_threshold=100,rt_method='Rtg',all_compartments=False)
#visualise infection curves
for i in output_sir_seqs:
    plt.plot(i)

output_list_seq_df = pd.DataFrame(output_list_seqs)
output_list_seq_df = pd.DataFrame(columns = ['Country_Region','Dt','NewPat','Oracle_kz','shuffles'])
for i in range(0,len(output_sir_seqs)):    
    output_list_seq_df=output_list_seq_df.append({'Country_Region' : i, \
                              'Dt' : np.array([j for j in range(1,1000)]),\
                              'NewPat' : [0],\
                              'Oracle_kz' : output_list_seqs[i],}, ignore_index=True)
output_list_seq_df.to_pickle('realistic_sir_curves.pkl',protocol=4)
'''