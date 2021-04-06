# -*- coding: utf-8 -*-
"""
Created on Wed Feb 10 21:54:48 2021

@author: u6026797
"""


#%% Libraries
import pandas as pd
import numpy as np
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri
pandas2ri.activate()
#%% functions
#%% function  
rstring_hybrid="""
library(forecast)

r_aa_est <- function(ts,period){
  end_ts<-floor(length(ts)/period)
  eval_ts<-ts(ts,start=c(1),end=c(end_ts),frequency = period)
  stl_out<-stl(eval_ts,s.window = period)
  aa_fit<-auto.arima(stl_out$time.series[,'remainder'],seasonal = FALSE)
  aa_out<-arimaorder(aa_fit)
  return(aa_out)
}
"""
rfunc=robjects.r(rstring_hybrid)

rstring="""
library(forecast)

r_aa_noise_gen <-function(seq_len,period,order){
    model<-Arima(ts(rnorm((seq_len*10),mean = 4),freq=period),order=order,method=('ML'))
    sim_out<-simulate(model,seq_len)
    sim_out<-sim_out[1:seq_len]
    #sim_out<-(sim_out/sum(sim_out))
    return(sim_out)
    }
"""
rfunc2=robjects.r(rstring)

def r_ts_fit(ts, period):
  '''
  Wrapper function for R functions stl and auto.arima model via rpy2. Be sure that you have rpy2 
  and its dependencies installed. Specifically:
  
  import rpy2.robjects.packages as rpackages
  utils = rpackages.importr('utils')
  utils.chooseCRANmirror(ind=1)
  packnames = ('forecast')
  from rpy2.robjects.vectors import StrVector
  names_to_install = [x for x in packnames if not rpackages.isinstalled(x)]
  if len(names_to_install) > 0:
      utils.install_packages(StrVector(names_to_install))
  
  Arguments
  ----------
  ts : int/float numeric vector
    Vector to be processed
  period : integer 
    estimated period of ts
  Returns
  -------
  aa_out : list of arima p,d,q

  '''
  ts= robjects.FloatVector(ts)
  period= robjects.IntVector([period])
  
  aa_out = rfunc(ts=ts, period=period)
  return aa_out

def r_arma_noise_gen(seq_len,period,aa_order):
    seq_len=robjects.IntVector([seq_len])
    period=robjects.IntVector([period])
    aa_order=robjects.IntVector(aa_order)
    noise_out = rfunc2(seq_len=seq_len,
                       period=period,
                       order=aa_order)
    return noise_out
    