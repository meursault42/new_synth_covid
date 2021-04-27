# -*- coding: utf-8 -*-
"""
Created on Tue Apr 20 21:05:37 2021
@author: christopher.wilson@utah.edu
"""
#%% Libraries
import numpy as np
import math 
from hmmlearn import hmm
import random 
from copy import deepcopy
from statsmodels.tsa.seasonal import seasonal_decompose
from scipy.stats import chisquare,chi2_contingency,kstest
import warnings
import pmdarima as pm

#%% functions

class QuasiPeriod():
    
    def __init__(self,sig_cutoff = 0.05, likely_index=[5,6,7,8,9,11], prior_window = 7,
                 aa_bypass=True, verbose=False):
        self.sig_cutoff = sig_cutoff
        self.likely_index=likely_index
        self.prior_window=prior_window
        self.aa_bypass = aa_bypass
        self.verbose = verbose
        
        for ind in self.likely_index:
            nlikely_index = np.array([x for x in self.likely_index if x != ind])
            prime_check = nlikely_index%ind
            if (0 in prime_check)==True:
                warnings.warn('Value {} in likely_index contains a multiple of itself. The quasiperiodicty test may behave irreguarly.'.format(ind))
                    
                 
    def print_parameters(self):
        print('significance cutoff', self.sig_cutoff)
    
    def spiegelhalter(self, input_vec, nrepl = 2000):
        '''
        Function runs spiegelhalter test for normality of the input vec with
        n=nrepl simulated normal distributions. The resulting p_val is adjusted
        for the number of simulations.
        Parameters
        ----------
        input_vec : TYPE
            DESCRIPTION.
        nrepl : TYPE, optional
            DESCRIPTION. The default is 2000.
    
        Returns
        -------
        output_dict : TYPE
            T stat: is the test statistic
            p_value: p value of test
    
        '''
        l = 0.0
        vec_len = len(input_vec)
        sort_vec = np.sort(input_vec)
        vec_mean = sum(sort_vec)/vec_len
        stat_val = np.sqrt(((np.sum((input_vec - vec_mean)**2))/(vec_len - 1)))
        range_stat = (sort_vec[vec_len-1] - sort_vec[0])/stat_val 
        c = math.factorial(vec_len)**(1/(vec_len-1))/(2*vec_len)
        g = (np.sum(abs(sort_vec - vec_mean)))/(stat_val * np.sqrt(vec_len * (vec_len-1)))
        t = ((c*range_stat)**(1-vec_len) + g**(1 - vec_len))**(1/(vec_len-1))
        for i in range(0,nrepl):
            z = np.random.normal(size=vec_len)
            z = np.sort(z)
            z1 = np.sum(z)/vec_len
            sub_s = np.sqrt((np.sum((z-z1)**2))/(vec_len-1))
            u = (z[vec_len-1]-z[0])/sub_s
            sub_g = np.sum(abs(z-z1))/(sub_s * np.sqrt(vec_len * (vec_len-1)))
            sub_t = ((c * u)**(1-vec_len) + sub_g**(1-vec_len))**(1/(vec_len-1))
            if (sub_t > t):
                l += 1
        p_value = l/nrepl
        output_dict = {'T stat':t, 'p_value':p_value}
        return output_dict
    
    def decomp_window(self, input_vec,start_value,adj_window):
        '''
        Workhorse function that iterates over segment with supplied period width
        window and returns the window that minimizes curvature approximation as
        sum absolute second difference of residuals.

        Parameters
        ----------
        input_vec : TYPE
            DESCRIPTION.
        likely_index : TYPE
            DESCRIPTION.
        start_value : TYPE
            DESCRIPTION.
        adj_window : TYPE
            DESCRIPTION.

        Returns
        -------
        min_loss : TYPE
            DESCRIPTION.
        best_p : TYPE
            DESCRIPTION.
        best_pval : TYPE
            DESCRIPTION.
        out_list : TYPE
            DESCRIPTION.
        result_m : TYPE
            DESCRIPTION.
        result_a : TYPE
            DESCRIPTION.
        weight : TYPE
            DESCRIPTION.
        ks_list : TYPE
            DESCRIPTION.
        mod_list : TYPE
            DESCRIPTION.
        spieg_out : TYPE
            DESCRIPTION.

        '''
        min_loss = 9E9
        best_p = 0
        result_m = 0
        out_list = []
        s_weight = 0
        n_weight = 0
        mod_list = []
        p_list = []
        spieg_out = 0
        min_floor = math.floor(max(self.likely_index)/2)
        for i in self.likely_index:
            #subset input vector with minimum padding
            adj_vec = input_vec[(start_value-math.floor(i/2)):(start_value+adj_window+math.floor(i/2))]
            in_result = seasonal_decompose(adj_vec,model='multiplicative',period=i)
            mod_list.append(in_result)
            
            #calculate curvature error
            resid = in_result.resid
            tot_resid = np.sum(abs(np.diff(np.diff(resid[~np.isnan(resid)]))))
            out_list.append(tot_resid)
            
            #calculate partial model without residuals
            x_hat = (in_result.seasonal * in_result.trend)
            #run kolmogorov-smirnov test
            ksout=kstest(x_hat[min_floor:adj_window],adj_vec[min_floor:adj_window])
            #take the complement of the pvalue as we are interested in similarity
            ksout = 1-ksout.pvalue
            p_list.append(ksout)
            
            #calculate the weight of the seasonal component to the observed component
            seas_w = abs((in_result.trend * in_result.resid)-in_result.observed)/in_result.observed
            seas_w[seas_w>1]=1
            seas_w = np.mean(seas_w[~np.isnan(seas_w)])
            #calculate the weight of the residual noise component to the observed component
            noise_w = abs((in_result.seasonal * in_result.trend)-in_result.observed)/in_result.observed
            #correction for poor estimates
            noise_w[noise_w>1]=1
            noise_w = np.mean(noise_w[~np.isnan(noise_w)])
            
            if tot_resid < min_loss:
                min_loss = tot_resid
                best_p = i
                best_pval = ksout
                result_m = in_result
                s_weight = seas_w
                n_weight = noise_w
                #calculate speigelhalter test for regularity of 
                spieg_out = self.spiegelhalter((x_hat[min_floor:adj_window]-adj_vec[min_floor:adj_window]))['p_value']
        return min_loss, best_p, best_pval, out_list, result_m, s_weight, n_weight, mod_list, spieg_out

    def _bidirround(self, value):
        if value<0:
            return(math.floor(value))
        if value>0:
            return(math.ceil(value))
    
    def _shuffle_permute(self, input_vec,shuffle_prob=30):
        input_vec = input_vec.astype(float)
        permute_index = []
        for index, values in np.ndenumerate(input_vec):
            flip_logic = (random.randint(1,shuffle_prob)==1)
            if flip_logic == True:
                permute_index.append(index[0])
                shuffle_mag_dist = np.random.normal(.75,.1,1)
                shuffle_day_dist = self._bidirround(np.random.normal(0,1,1))
                change_value=input_vec[index[0]]-(input_vec[index[0]]*shuffle_mag_dist)
                if (index[0]+shuffle_day_dist)<0 or (index[0]+shuffle_day_dist)>(len(input_vec)-1):
                    continue
                elif abs(shuffle_day_dist)==1:
                    input_vec[index[0]]=input_vec[index[0]]*shuffle_mag_dist
                    input_vec[(index[0]+shuffle_day_dist)]+=change_value
                elif abs(shuffle_day_dist)>1:
                    input_vec[index[0]]=input_vec[index[0]]*shuffle_mag_dist
                    change_value = change_value/abs(shuffle_day_dist)
                    start=min(shuffle_day_dist,0)
                    end=max(shuffle_day_dist,0)
                    if end==0:
                        for i in range(start,end):
                            input_vec[(index[0]+i)]+=change_value
                    elif end>0:
                        for i in range((start+1),(end+1)):
                            input_vec[(index[0]+i)]+=change_value
        return input_vec, permute_index
       
    def _simple_white_noise_gen(self, input_ts, chain, n_w_vec):
        delist_chain = chain.reshape([1,-1])[0]
        out_seq=np.array([])
        start = 0
        end = 0
        for i in delist_chain:
            padding = 0
            end += i
            noise = np.random.normal(loc=10,scale=2,size=i)
            noise = noise/np.sum(noise)
            input_sub = input_ts[start:(min(end,len(input_ts)-1))]
            if len(input_sub)==0:
                return out_seq
            if len(noise)>len(input_sub):
                padding = len(noise)-len(input_sub)
                input_sub = np.append(input_sub,np.ones(padding)*input_sub[-1])
            w_chain_noise = self._weight_chain_avg(input_sub,noise,weight=random.sample(random.sample(n_w_vec,1)[0],1)[0])
            if padding != 0:
                w_chain_noise[:len(w_chain_noise)-padding]
            out_seq = np.append(out_seq,w_chain_noise)
            start+=i
        return out_seq
            
    def _chain_noise_gen(self, chain,n_vec):
        '''
        Genereates a white noise sequence. 
    
        '''
        delist_chain = chain[0]
        flat_list = [item for sublist in n_vec for item in sublist]
        out_seq=[]
        for i in delist_chain:
            out_seq.append(r_arma_noise_gen(i[0],i[0],random.sample(flat_list,1)[0]))
        out_seq = np.concatenate(out_seq)
        out_seq = out_seq/sum(out_seq)
        return(out_seq)
    
    def _weight_chain_avg(self, input_ts,noise,weight=0.1):
        '''
        Calculates weighted average of two (x,) dimensional np arrays
        '''
        w_chain_noise=(np.sum(input_ts*noise))*weight*2
        w_raw_seq=(input_ts)*(2-(weight*2))
        w_c_avg=np.mean((w_chain_noise,w_raw_seq),axis=0)
        return w_c_avg
    
    def _chain_smoother(self, input_vec,chain):
        '''
        Smoothes transitions between chains by averaging edges.
        '''
        input_vec_copy=input_vec
        indexer = -1
        for i in chain:
            indexer += i
            if indexer < (len(input_vec)-3):
                tail = np.mean(input_vec_copy[indexer-1:indexer+2])
                head = np.mean(input_vec_copy[indexer:indexer+3])
                input_vec_copy[indexer]=tail
                input_vec_copy[indexer+1]=head
            
        return input_vec_copy
    
    def _nth_order_sort(self, seq,n_order,sort='low'):
        '''
        Parameters
        ----------
        seq : A Numpy array
            The input vector to be sorted
        n_order : Integer
            The n from top item desired. IE return the second highest = 2
    
        Returns
        -------
        TYPE
            Returns the nth from top item from the input sequence.
    
        '''
        if sort=='high':
            for i in range(n_order):
                seq_max = max(seq)
                seq = seq[seq!=seq_max]
            return max(seq)
        if sort=='low':
            for i in range(n_order):
                    seq_min = min(seq)
                    seq = seq[seq!=seq_min]
            return min(seq)
    
    def emission_hmm(self, input_vec,start_window=0,start_value=0,prior_step=0,prior_index=0,
                 prior_magnitude=0,n_order=0, prior_dir_dict=0, prior_noise=0,
                 prior_significance=0):
        '''
        Function recursively iterates through a given array and returns the 
        nth order periodic component found, along with the magnitude and
        the averaged direction vector. 
    
        Parameters
        ----------
        input_vec : Numpy Array
            Input array from which noise is estimated.
        start_window : Integer, optional
            The window for searching for noise. The default is 40. Function may
            fail if the window is too small.
        start_value : Integer, optional
            Used for managing recursive indexing.
        prior_step : List, optional
            Output list that holds prior results. The default is 0.
        prior_magnitude : List, optional
            Output list that holds prior noise magnitude. The default is 0.
        n_order : Integer, optional
            Desired rank of noise derived by pACF. 0 = max, 1 = 2nd highest component. 
            The default is 0.
        prior_dir_dict : dict, optional
            Holds magnitude of observed noise. The default is 0.
        cutoff : Float, optional
            Cutoff value for determining un/useful noise components by pACF coefficient.
            The default is 0.05. Values below this are presumed to be white noise.
        prior_noise : List, optional
            List used to hold noise shapes after stl decomposition. The default is 0.
        likely_index : list, optional
            List holds subset of pACF of interest. The default is 5:13
    
        Returns
        -------
        Numpy Array
            output_vec: Holds emmission outputs for use in subsequent modelling.
        Numpy Array
            magnitude_vec: Holds coefficients from pACF for use in later noise 
            generation.
        Numpy Array
            shape_dict: Holds magnitudes of periodic noise components for use in 
            later noise generation.
        list
            noise_list: Holds arima specifications for residual noise of sequences 
    
        '''
        if start_value==0:
            start_value=math.floor(max(self.likely_index)/2)
        if start_window==0:
            start_window=start_value*6
        ##check for prior_step else build array
        if isinstance(prior_step, np.ndarray):
            output_vec = prior_step
        else:
            output_vec = np.array([])
        ##check for prior_step else build array
        if isinstance(prior_index, np.ndarray):
            output_index = prior_index
        else:
            output_index = np.array([])        
        ##check for prior_magnitude else build array
        if isinstance(prior_magnitude, np.ndarray):
            magnitude_vec = prior_magnitude
        else:
            magnitude_vec = np.array([])
        ##check for prior_significance else build array
        if isinstance(prior_significance, np.ndarray):
            significance_vec = prior_significance
        else:
            significance_vec = np.array([])
        ##check for prior_dir_array else build array
        if isinstance(prior_dir_dict, dict):
            shape_dict = prior_dir_dict
        else:
            shape_dict = {}
            for index_key in self.likely_index:
                load_key = str(index_key)+'_day'
                shape_dict[load_key] = []
        if isinstance(prior_noise, list):
            noise_list = prior_noise
        else:
            noise_list = []
        if isinstance(n_order, list):
            arma_noise_order = n_order
        else:
            arma_noise_order = []
        adj_window = min(start_window,(len(input_vec[start_value:])))
        adj_vec = input_vec[(start_value-5):(start_value+adj_window+5)]
        
        if len(adj_vec)>=start_window:
            if self.verbose==True:
                print('start:{} end:{}'.format(start_value-5,start_value+adj_window+5))
            b_l, b_val, b_pval, stl_out, res_m, s_w, n_w, p_l, spieg = self.decomp_window(input_vec,
                                                                         start_value=start_value,
                                                                         adj_window=start_window)
            #extract dirction of periodic component as %change from mean
            if b_pval>self.sig_cutoff and self.sig_cutoff != 0:
                output_vec = np.append(output_vec, self.prior_window)
                output_index = np.append(output_index, 0)
                magnitude_vec = np.append(magnitude_vec, s_w)
                significance_vec = np.append(significance_vec, spieg)
                qpc = np.random.normal(0,1,size=(7))
                qpc = abs(qpc)/np.sum(abs(qpc))
                prior_day = str(self.prior_window)+'_day'
                shape_dict[prior_day].append(qpc)
                noise_list.append(n_w)
                if self.aa_bypass == False:
                    residuals= res_m.resid[math.floor(max(self.likely_index)/2):adj_window]
                    #retain aa p,d,q terms for arma generation
                    aa_out = pm.auto_arima(residuals)
                    arma_noise_order.append(aa_out.order)
                return (self.emission_hmm(input_vec=input_vec,\
                          start_value=(7+start_value),\
                          prior_index=output_index,\
                          prior_step=output_vec,\
                          prior_magnitude=magnitude_vec,\
                          prior_significance = significance_vec,\
                          n_order=arma_noise_order,\
                          prior_dir_dict = shape_dict,
                          prior_noise = noise_list,
                          start_window=start_window))
            else:
                ##aa modelling of noise is optional since resid are normal
                #estimate arima order of residuals and weight
                residuals= res_m.resid[math.floor(max(self.likely_index)/2):adj_window]
                
                
                if self.aa_bypass == False:
                    #retain aa p,d,q terms for arma generation
                    aa_out = pm.auto_arima(residuals)
                    arma_noise_order.append(aa_out.order)
                noise_list.append(n_w)
                
                qpc = abs(res_m.seasonal[b_val:b_val*2])/np.sum(res_m.seasonal[b_val:b_val*2])
                key = str(b_val)+'_day'
                shape_dict[key].append(qpc)
                
                output_vec = np.append(output_vec, b_val)
                output_index = np.append(output_index, b_val)
                magnitude_vec = np.append(magnitude_vec, s_w)
                significance_vec = np.append(significance_vec, spieg)
                
                return (self.emission_hmm(input_vec=input_vec,\
                              start_value=(b_val+start_value),\
                              prior_step=output_vec,\
                              prior_index=output_index,\
                              prior_magnitude=magnitude_vec,\
                              prior_significance = significance_vec,\
                              n_order=arma_noise_order,\
                              prior_dir_dict = shape_dict,
                              prior_noise = noise_list,
                              start_window=start_window))
        else:
            return {'best_fit_periods':output_vec, 'period_magnitudes':magnitude_vec, 
                    'period_shapes':shape_dict, 'noise_weights':noise_list, 
                    'confidence_censored_periods':output_index, 'normal_test_vec':significance_vec,
                    'arma_noise_order':n_order}
        
    
    def get_hmm_outputs(self, subset_index,input_data_df,dir_dict=0,n_order=0):
        '''
        Wrapper function that extracts periodic noise from a provided input df
        then generates and trains a hidden markov model on those chained periodic
        noise components.
    
        Parameters
        ----------
        subset_index : list
            A list of unique indexes for the input_data_df. Example: state names, 
            zip codes, etc
        input_data_df : Pandas DataFrame
            A dataframe containing at least two columns: index which lists a index
            value as described above and noise_seq, each cell of which should contain
            an np.array of the sequence of interest. 
            Example: 
                ['index', 'noise_seq']
                'Washington', [1,2,3,...]
        n_order : TYPE, optional
            DESCRIPTION. The default is 0.
    
        Returns
        -------
        None.
    
        '''
        emm_vec = np.array([])
        mag_vec = np.array([])
        window_list = []
        mag_list = []
        ##check for prior_dir_array else build array
        if isinstance(dir_dict, dict)==False:
            dir_dict = {}
            for index_key in self.likely_index:
                load_key = str(index_key)+'_day'
                dir_dict[load_key] = []
        noise_list = []
        def _dict_key_merger(dict1,dict2):
            for key in dir_dict:
                for seq in dict2[key]:
                    dict1[key].append(seq)
        
        for ind in subset_index:
            if self.verbose == True:
                print('working on {}'.format(ind))
            data_vec_loc = input_data_df[input_data_df['index']==ind]
            data_vec_loc = data_vec_loc['noise_seq'].to_numpy()[0]
            ehmm = self.emission_hmm(input_vec=data_vec_loc,n_order=n_order)
            emm=ehmm['best_fit_periods'].astype(int)
            emm=emm.reshape(-1,1)
            window_list.append(emm)
            mag_list.append(ehmm['period_magnitudes'])
            noise_list.append(ehmm['noise_weights'])
            _dict_key_merger(dir_dict, ehmm['period_shapes'])
        
        emm_vec = np.concatenate(window_list)
        a_lens = [len(i)for i in window_list]
        
        mag_vec = np.concatenate(mag_list)
        model = hmm.MultinomialHMM(n_components=len(self.likely_index))
        model.fit(emm_vec, a_lens)
        
        return [model,mag_vec,dir_dict, noise_list]
    
    def _gen_hmm_noise_seq(self, input_ts,model,mag_vec,dir_dict):
        '''
        This function takes the output of get_hmm_outputs and creates a Hidden
        Markov Model then samples from it to generate a noise sequence similar the
        observed sequence.
    
        Parameters
        ----------
        input_ts : Numpy Array
            An array of values that you wish to add noise to.
        model : hmm model output
            the output hmm chain from get_hmm_outputs
        mag_vec : list
            List of pACF coefficients produced by get_hmm_outputs
        dir_dict : dictionary
            Dictionary of magnitudes produced by get_hmm_outputs 
    
        Returns
        -------
        input_ts_c : Numpy Array
            A copy of input_ts with hmm derived noise.
    
        '''
        input_ts_len = len(input_ts)
        start_ind = 0
        stop_ind = 0
        input_ts_c = deepcopy(input_ts)
        local_dir_dict = dict()
        hmm_chain = model[0]
        magnitude_distribution = mag_vec
        
        #randomly select single shape for each length
        for shape in dir_dict:
            local_dir_dict[shape]=random.sample(dir_dict[shape],1)[0]    
        
        for i in hmm_chain.tolist():
            stop_ind += i[0]
            if stop_ind<=len(input_ts_c)-1:
                magnitude=random.sample(magnitude_distribution.tolist(),1)[0]
                shape_key=str(i[0])+'_day'
                #shape=random.sample(shape_dict[shape_key],1)[0]
                shape=local_dir_dict[shape_key]
                w_periodic_component=(np.sum(input_ts_c[start_ind:stop_ind])*shape)*magnitude*2
                w_raw_seq=(input_ts_c[start_ind:stop_ind])*(2-(magnitude*2))
                input_ts_c[start_ind:stop_ind]=np.mean((w_periodic_component,w_raw_seq),axis=0)
                start_ind += i[0]
            elif stop_ind>len(input_ts_c)-1:
                append_seq = np.ones(((abs(stop_ind-len(input_ts_c)))))
                input_ts_c=np.append(input_ts_c,append_seq,0)
                magnitude=random.sample(magnitude_distribution.tolist(),1)[0]
                shape_key=str(i[0])+'_day'
                #shape=random.sample(shape_dict[shape_key],1)[0]
                shape=local_dir_dict[shape_key]
                w_periodic_component=(np.sum(input_ts_c[start_ind:stop_ind])*shape)*magnitude*2
                w_raw_seq=(input_ts_c[start_ind:stop_ind])*(2-(magnitude*2))
                input_ts_c[start_ind:stop_ind]=np.mean((w_periodic_component,w_raw_seq),axis=0)
                start_ind += i[0]
        #test index correction?
        input_ts_c = input_ts_c[0:input_ts_len]
        return input_ts_c
    
    def generate_noise_sequences(self, input_ts_list, subset_index, input_data_df,
                                 n_periodic_components=1, white_noise=False, 
                                 white_noise_weight = 0.05, shuffle_permute=True,
                                 shuffle_prob = 30, smooth_transitions=False):
        '''
        Wrapper function that takes in a list of np arrays, then adds modelled, 
        structured noise to replicate noise seen in the covid-19 pandemic daily
        new cases reported.
    
        Parameters
        ----------
        input_ts_list : list
            A list of np arrays that will have noise added to them.
        subset_index : list
            A list of unique indexes for the input_data_df. Example: state names, 
            zip codes, etc
        input_data_df : Pandas DataFrame
            A dataframe containing at least two columns: index which lists a index
            value as described above and noise_seq, each cell of which should contain
            an np.array of the sequence of interest. 
            Example: 
                ['index', 'noise_seq']
                'Washington', [1,2,3,...]
        n_periodic_components : int, optional
            The number of periodic components desired. 1= only max pACF, 2=top 2 etc.
            The default is 1.
        white_noise : Boolean, optional
            Option to include white noise derived from input sequence. The default is True.
        white_noise_weight : float, optional
            Weight applied to white noise. The default is 0.05
        shuffle_permute : Boolean, optional
            Option to apply a unidirectional shuffle as observed in the covid-19 
            pandemic. This distributes a percent of a days values backwards or 
            forwards in time a random 1-3 days. The defaul is True.
        shuffle_prob: int, optional
            Denominator of shuffle permute occurring on any given day. Default is
            approximately once a month (1/30).    
        smooth_transitions: Boolean, optional
            Option to smooth noise chains between sequences. The default is False.
    
        Returns
        -------
        A input_ts_list with selected, modelled noise.
    
        '''
        out_list=[]
        periodic_comp_dict = {}
        out_shuffles=[]
        #generate nth order periodic noise components and white noise set
        for n in list(range(0,n_periodic_components)):
            if self.verbose==True:
                print('Modelling {} order noise. May take a minute.'.format(n))
            model, mag_vec, dir_dict, n_vec = self.get_hmm_outputs(subset_index=subset_index,
                                                              input_data_df=input_data_df,
                                                              n_order=n)
            periodic_comp_dict[str(n)+'_model']=model
            periodic_comp_dict[str(n)+'_mag_vec']=mag_vec
            periodic_comp_dict[str(n)+'_dir_dict']=dir_dict
            periodic_comp_dict[str(n)+'_n_vec']=n_vec
        
        for seq in input_ts_list:
            #calulate maximum chain length
            chain_len=math.floor(len(seq)/5)
            #add periodic noise
            seq_c = deepcopy(seq)
            for n in list(range(0,n_periodic_components)):
                hmm_noise_sample = periodic_comp_dict[str(n)+'_model'].sample(chain_len)
                seq_c = self._gen_hmm_noise_seq(seq_c,
                                        hmm_noise_sample,
                                        periodic_comp_dict[str(n)+'_mag_vec'],
                                        periodic_comp_dict[str(n)+'_dir_dict'])
                if smooth_transitions==True:
                    seq_c =self._chain_smoother(seq_c,np.reshape(hmm_noise_sample[0],(chain_len,)))
            #add white noise
            if white_noise==True:
                seq_c = self._simple_white_noise_gen(seq_c, hmm_noise_sample[0], periodic_comp_dict['0_n_vec'])
            #add shuffle permutations
            if shuffle_permute==True:
                seq_c, shuffle_indexes = self._shuffle_permute(seq_c,shuffle_prob=shuffle_prob)
            ###return seq
            out_list.append(seq_c)
            out_shuffles.append(shuffle_indexes)
        return(out_list,periodic_comp_dict['0_model'],out_shuffles)
    
