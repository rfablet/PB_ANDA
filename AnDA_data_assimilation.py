#!/usr/bin/env python

""" AnDA_data_assimilation.py: Apply stochastic and sequential data assimilation technics using model forecasting or analog forecasting. """

__author__ = "Phi Huynh Viet"
__version__ = "2.0"
__date__ = "2017-08-01"
__email__ = "phi.huynhviet@telecom-bretagne.eu"

import numpy as np
from scipy.stats import multivariate_normal
from AnDA_stat_functions import resampleMultinomial, inv_using_SVD, inv_using_Woodbury
from tqdm import tqdm

def AnDA_data_assimilation(yo, DA):
    """ Apply stochastic and sequential data assimilation technics using model forecasting or analog forecasting. """
    #mkl.set_num_threads(10)
    # initializations
    n = len(DA.xb);
    T = yo.values.shape[0];
    
    class x_hat:
        part = np.empty([T,DA.N,n]);
        weights = np.empty([T,DA.N]);
        values = np.empty([T,n]);
        time = yo.time;

    if (DA.method =='AnEnKF' or DA.method =='AnEnKS'):
        m_xa_part = np.empty([T,DA.N,n]);
        xf_part = np.empty([T,DA.N,n]);
        Pf = np.empty([T,n,n]);
        for k in tqdm(range(0,T)):
            # update step (compute forecasts)       
            #print k
            if k==0:
                xf = np.random.multivariate_normal(DA.xb, DA.B, DA.N);
            else:
                xf, m_xa_part_tmp = DA.m(x_hat.part[k-1,:,:],k);
                m_xa_part[k,:,:] = m_xa_part_tmp;  
            #start_time = time.time()
            xf_part[k,:,:] = xf;
            Ef = np.dot(xf.T,np.eye(DA.N)-np.ones([DA.N,DA.N])/DA.N);
            Pf[k,:,:] = np.dot(Ef,Ef.T)/(DA.N-1);
            # analysis step (correct forecasts with observations)          
            i_var_obs = np.where(~np.isnan(yo.values[k,:]))[0];            
            if (len(i_var_obs)>0):                
                if (np.isscalar(DA.R)): # in case of high dimension
                    eps = [np.random.normal(0,np.sqrt(DA.R),len(i_var_obs)) for i in range(DA.N)]
                    eps = np.array(eps)
                    K_part = inv_using_Woodbury(1/DA.R,DA.H[i_var_obs,:],inv_using_SVD(Pf[k,:,:],0.9999),DA.H[i_var_obs,:].T,1/DA.R)
                else: # normal case
                    cov_R = DA.R[np.ix_(i_var_obs,i_var_obs)]
                    eps = np.random.multivariate_normal(np.zeros(len(i_var_obs)),cov_R,DA.N)
                    K_part = np.linalg.inv(np.dot(np.dot(DA.H[i_var_obs,:],Pf[k,:,:]),DA.H[i_var_obs,:].T)+cov_R)                    
                yf = np.dot(DA.H[i_var_obs,:],xf.T); yf = yf.T                
                K = np.dot(np.dot(Pf[k,:,:],DA.H[i_var_obs,:].T),K_part);                   
                d = np.repeat(yo.values[k,i_var_obs][np.newaxis],DA.N,0)+eps-yf;
                x_hat.part[k,:,:] = xf + np.dot(d,K.T);          
            else:
                x_hat.part[k,:,:] = xf;            
            x_hat.weights[k,:] = np.repeat(1.0/DA.N,DA.N);
            x_hat.values[k,:] = np.sum(x_hat.part[k,:,:]*np.repeat(x_hat.weights[k,:][np.newaxis],n,0).T,0);
            #print("---result  %s seconds ---" % (time.time() - start_time))
            
	# end AnEnKF
        if (DA.method == 'AnEnKS'):
            for k in tqdm(range(T-1,-1,-1)):           
                if k==T-1:
                    x_hat.part[k,:,:] = x_hat.part[T-1,:,:];
                else:
                    m_xa_part_tmp = m_xa_part[k+1,:,:];
                    tej, m_xa_tmp = DA.m(np.mean(x_hat.part[k,:,:],0)[np.newaxis],k+1);
                    tmp_1 =(x_hat.part[k,:,:]-np.repeat(np.mean(x_hat.part[k,:,:],0)[np.newaxis],DA.N,0)).T;
                    tmp_2 = m_xa_part_tmp-np.repeat(m_xa_tmp,DA.N,0);                    
                    Ks = 1.0/(DA.N-1)*np.dot(np.dot(tmp_1,tmp_2),inv_using_SVD(Pf[k+1,:,:],0.9999));                    
                    x_hat.part[k,:,:] = x_hat.part[k,:,:]+np.dot(x_hat.part[k+1,:,:]-xf_part[k+1,:,:],Ks.T);
                x_hat.values[k,:] = np.sum(x_hat.part[k,:,:]*np.repeat(x_hat.weights[k,:][np.newaxis],n,0).T,0);
                
        # end AnEnKS  
    
    elif (DA.method =='AnPF'):
        # special case for k=1
        k=0
        k_count = 0
        m_xa_traj = []
        weights_tmp = np.zeros(DA.N);
        xf = np.random.multivariate_normal(DA.xb, DA.B, DA.N)
        i_var_obs = np.where(~np.isnan(yo.values[k,:]))[0]
        if (len(i_var_obs)>0):
            # weights
            for i_N in range(0,DA.N):
                weights_tmp[i_N] = multivariate_normal.pdf(yo.values[k,i_var_obs].T,np.dot(DA.H[i_var_obs,:],xf[i_N,:].T),DA.R[np.ix_(i_var_obs,i_var_obs)]);
            # normalization
            weights_tmp = weights_tmp/np.sum(weights_tmp);
            # resampling
            indic = resampleMultinomial(weights_tmp);
            x_hat.part[k,:,:] = xf[indic,:];         
            weights_tmp_indic = weights_tmp[indic]/sum(weights_tmp[indic])
            x_hat.values[k,:] = sum(xf[indic,:]*(np.repeat(weights_tmp_indic[np.newaxis],n,0).T),0);
            # find number of iterations before new observation
            k_count_end = np.min(np.where(np.sum(1*~np.isnan(yo.values[k+1:,:]),1)>=1)[0]);
        else:
            # weights
            weights_tmp = np.repeat(1.0/N,N);
            # resampling
            indic = resampleMultinomial(weights_tmp);
        x_hat.weights[k,:] = weights_tmp_indic;
        
        for k in tqdm(range(1,T)):
            # update step (compute forecasts) and add small Gaussian noise
            xf, tej = DA.m(x_hat.part[k-1,:,:]) +np.random.multivariate_normal(np.zeros(xf.shape[1]),DA.B/100.0,xf.shape[0]);        
            if (k_count<len(m_xa_traj)):
                m_xa_traj[k_count] = xf;
            else:
                m_xa_traj.append(xf);
            k_count = k_count+1;
            # analysis step (correct forecasts with observations)
            i_var_obs = np.where(~np.isnan(yo.values[k,:]))[0];
            if len(i_var_obs)>0:
                # weights
                for i_N in range(0,DA.N):
                    weights_tmp[i_N] = multivariate_normal.pdf(yo.values[k,i_var_obs].T,np.dot(DA.H[i_var_obs,:],xf[i_N,:].T),DA.R[np.ix_(i_var_obs,i_var_obs)]);
                # normalization
                weights_tmp = weights_tmp/np.sum(weights_tmp);
                # resampling
                indic = resampleMultinomial(weights_tmp);            
                # stock results
                x_hat.part[k-k_count_end:k+1,:,:] = np.asarray(m_xa_traj)[:,indic,:];
                weights_tmp_indic = weights_tmp[indic]/np.sum(weights_tmp[indic]);            
                x_hat.values[k-k_count_end:k+1,:] = np.sum(np.asarray(m_xa_traj)[:,indic,:]*np.tile(weights_tmp_indic[np.newaxis].T,(k_count_end+1,1,n)),1);
                k_count = 0;
                # find number of iterations  before new observation
                try:
                    k_count_end = np.min(np.where(np.sum(1*~np.isnan(yo.values[k+1:,:]),1)>=1)[0]);
                except ValueError:
                    pass
            else:
                # stock results
                x_hat.part[k,:,:] = xf;
                x_hat.values[k,:] = np.sum(xf*np.repeat(weights_tmp_indic[np.newaxis],n,0).T,0);
            # stock weights
            x_hat.weights[k,:] = weights_tmp_indic;   
        # end AnPF
    else :
        print("Error: choose DA.method between 'AnEnKF', 'AnEnKS', 'AnPF' ")
        quit()
    return x_hat
  
        
            
            
