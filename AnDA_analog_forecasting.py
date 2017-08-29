#!/usr/bin/env python

""" AnDA_analog_forecasting.py: Apply the analog method on catalog of historical data to generate forecasts. """

__author__ = "Phi Huynh Viet"
__version__ = "2.0"
__date__ = "2017-08-01"
__email__ = "phi.huynhviet@telecom-bretagne.eu"

import numpy as np
from AnDA_stat_functions import mk_stochastic, sample_discrete, AnDA_RMSE, AnDA_correlate
from scipy.sparse import diags
from sklearn.cluster import KMeans
def AnDA_analog_forecasting(x, in_x, AF):
    """ Apply the analog method on catalog of historical data to generate forecasts. """

    # initializations
    N, n = x.shape
    xf = np.zeros([N,n])
    xf_mean = np.zeros([N,n])
       
    # local or global analog forecasting
    stop_condition = 0
    i_var = np.array([0])
    while (stop_condition!=1):        
        # in case of global approach
        if np.array_equal(AF.neighborhood, np.ones([n,n])):
            i_var_neighboor = np.arange(0,n)
            i_var = np.arange(0,n)
            stop_condition = 1
            kdt = AF.list_kdtree[0]
        # in case of local approach
        else:
            i_var_neighboor = np.where(AF.neighborhood[int(i_var),:]==1)[0]
            kdt = AF.list_kdtree[int(i_var)]
        # find the indices and distances of the k-nearest neighbors (knn) 
        if (AF.flag_cond):
            index_knn, dist_knn = kdt.nn_index(x[:,i_var_neighboor],AF.k_initial)    
        else:
            index_knn, dist_knn = kdt.nn_index(x[:,i_var_neighboor],AF.k)              
        dist_knn = np.sqrt(dist_knn/len(i_var_neighboor))
        index_knn[np.in1d(index_knn,AF.check_indices).reshape(index_knn.shape)] = index_knn[np.in1d(index_knn,AF.check_indices).reshape(index_knn.shape)]-AF.lag
        # using condition to retrive AF.k analogs from AF.k_initial nearest neighbors   
        if (AF.flag_cond):
            mask_tmp = AF.obs_mask[in_x,:]
            if (np.sum(~np.isnan(mask_tmp))>0):
                x_cond_pca = np.copy(mask_tmp)
                x_cond_pca[np.isnan(x_cond_pca)] = AF.x_cond[in_x,np.isnan(x_cond_pca)]
                x_cond_pca = np.dot(x_cond_pca-AF.mu_dX,AF.coeff_dX)    
                x_cond_pca_tmp =np.zeros(n)[None]
                x_cond_pca_tmp[0,i_var_neighboor] = x_cond_pca[i_var_neighboor]
                x_cond_tmp_res = np.dot(x_cond_pca_tmp,AF.coeff_dX.T)+AF.mu_dX
                x_cond_tmp_res = x_cond_tmp_res[:,~np.isnan(mask_tmp)]            
                index_tmp = np.zeros([N,AF.k],dtype=np.int32)
                dist_tmp = np.zeros([N,AF.k])
                for i_N in range(N):
                    successors_reduced = AF.catalogs[index_knn[i_N,:]+AF.lag,:]
                    tmp_1 = np.zeros([AF.k_initial,n])  
                    tmp_1[:,i_var_neighboor] = successors_reduced[:,i_var_neighboor]
                    tmp_1 = np.dot(tmp_1,AF.coeff_dX.T)+AF.mu_dX
                    tmp_1 = tmp_1[:,~np.isnan(mask_tmp)]
                    tmp_3 = np.dot(successors_reduced, AF.coeff_dX.T) + AF.mu_dX
                    tmp_3 = tmp_3[:,~np.isnan(mask_tmp)]
                    dis_next = AnDA_RMSE(x_cond_tmp_res,tmp_1)
                    dis_next_g = AnDA_RMSE(mask_tmp[~np.isnan(mask_tmp)],tmp_3)
                    #corr_next_g = AnDA_correlate(mask_tmp[~np.isnan(mask_tmp)],tmp_3)
                    #corr_next_g = corr2(x[~np.isnan(mask)],tmp_4[~np.isnan(mask)])  
                    if (stop_condition==0):
                        #dis_final = dis_next*dis_next_g*dist_knn[i_N,:]
                        dis_final = dis_next*dist_knn[i_N,:]
                    else:
                        dis_final = dis_next_g*dist_knn[i_N,:]
                    sort_dis = np.argsort(dis_final)
                    index_tmp[i_N,:] = index_knn[i_N,sort_dis[:AF.k]]   
                    dist_tmp[i_N,:] = dis_final[sort_dis[:AF.k]]                      
                index_knn = index_tmp
                dist_knn = dist_tmp
            else:
                index_knn = index_knn[:,:AF.k]
                dist_knn = dist_knn[:,:AF.k]                                
        else:
            index_knn = index_knn[:,:AF.k]
            dist_knn = dist_knn[:,:AF.k]
          
        # normalisation parameter for the kernels
        lambdaa = np.median(dist_knn);
        # compute weights
        if AF.k == 1:
            weights = np.ones([N,1]);
        else:
            weights = mk_stochastic(np.exp(-np.power(dist_knn,2)/lambdaa));
        # reduce 2 array: analogs and successors because EnKF members have many identical nearest neighbors
        index_unique, mask_indices = np.unique(index_knn, return_inverse=True)
        index_knn = mask_indices.reshape(index_knn.shape)
        analogs = AF.catalogs[np.ix_(index_unique,i_var_neighboor)]
        if (stop_condition==0):
            analogs_single = AF.catalogs[np.ix_(index_unique,i_var)]
        #analogs = AF.catalogs[np.ix_(index_unique,i_var)]
        successors = AF.catalogs[np.ix_(index_unique+AF.lag,i_var)]
        # reduce catalog of physic model 
        if (AF.flag_model):
            cata_model = AF.cata_model_full[index_unique+AF.lag,:]
                   
        # reduced version of local linear   
        if (AF.flag_reduced):
            if (AF.regression == 'local_linear'):    
                if (N>=AF.cluster):
                    kmeans = KMeans(n_clusters=AF.cluster, random_state=0).fit(x[:,i_var_neighboor])
                else:
                    kmeans = KMeans(n_clusters=1, random_state=0).fit(x[:,i_var_neighboor])
                for i_cluster in range(kmeans.n_clusters):
                    cluster_x = np.where(kmeans.labels_== i_cluster)[0]
                    index_i_cluster, mask_cluster = np.unique(index_knn[cluster_x,:],return_inverse=True)
                    mask_cluster = mask_cluster.reshape(index_knn[cluster_x,:].shape)
                    if (AF.flag_model):
                        cata_model_tmp = np.concatenate((np.ones((len(cata_model[index_i_cluster,:]),1)),cata_model[index_i_cluster,:]),axis=1)
                        S = np.linalg.lstsq(cata_model_tmp,successors[index_i_cluster,:])[0]
                        ytest_A = np.dot(np.insert(AF.x_model[in_x-1,:],0,1),S)
                        tmp1 = np.dot(cata_model_tmp,S)
                        A = np.concatenate((np.ones((len(index_i_cluster),1)),analogs[index_i_cluster,:], tmp1),axis=1)
                        tmp4 = np.linalg.lstsq(A,successors[index_i_cluster,:])[0]
                        xf_mean[np.ix_(cluster_x,i_var)] = np.dot(np.concatenate((np.ones((len(cluster_x),1)),x[np.ix_(cluster_x,i_var_neighboor)],np.tile(ytest_A,(len(cluster_x),1))),axis=1),tmp4)
                        res_full = successors[index_i_cluster,:]-np.dot( A ,tmp4)
                    else:
                        A = np.concatenate((np.ones((len(index_i_cluster),1)),analogs[index_i_cluster,:]),axis=1)
                        tmp4 = np.linalg.lstsq(A,successors[index_i_cluster,:])[0]
                        res_full = successors[index_i_cluster,:]-np.dot( A ,tmp4)
                        xf_mean[np.ix_(cluster_x,i_var)] = np.dot(np.concatenate((np.ones((len(cluster_x),1)),x[np.ix_(cluster_x,i_var_neighboor)]),axis=1),tmp4)    
                    for jj in range(len(cluster_x)):
                        xf_tmp = np.zeros([AF.k,np.max(i_var)+1])
                        res = res_full[mask_cluster[jj,:],:]
                        xf_tmp[:,i_var] = xf_mean[cluster_x[jj],i_var]+res
                        res = res.T
                        if len(i_var)>1:
                            cov_xf = np.cov(res)
                        else:
                            cov_xf = np.cov(res)[np.newaxis][np.newaxis]
                        weights[cluster_x[jj],:] = 1.0/len(weights[cluster_x[jj],:]) 
                        if (AF.sampling =='gaussian'):
                            # random sampling from the multivariate Gaussian distribution
                            xf[cluster_x[jj],i_var] = np.random.multivariate_normal(xf_mean[cluster_x[jj],i_var],cov_xf)
                        elif (AF.sampling =='multinomial'):
                            # random sampling from the multinomial distribution of the weights
                            i_good = sample_discrete(weights[cluster_x[jj],:],1,1)
                            xf[cluster_x[jj],i_var] = xf_tmp[i_good,i_var]
                        else:
                            print("Error: choose AF.sampling between 'gaussian', 'multinomial' ")
                            quit()

            else:   
                print("Error: Clusterized version only for Local Linear.")
                quit()
        else:       
            # for each member/particle
            for i_N in range(0,N):
                xf_tmp = np.zeros([AF.k,np.max(i_var)+1]);         
                # select the regression method
                if (AF.regression == 'locally_constant'):
                    xf_tmp[:,i_var] = successors[index_knn[i_N,:],:];
                    # weighted mean and covariance
                    xf_mean[i_N,i_var] = np.sum(xf_tmp[:,i_var]*np.repeat(weights[i_N,:][np.newaxis].T,len(i_var),1),0)
                    E_xf = (xf_tmp[:,i_var]-np.repeat(xf_mean[i_N,i_var][np.newaxis],AF.k,0)).T;
                    cov_xf = 1.0/(1.0-np.sum(np.power(weights[i_N,:],2)))*np.dot(np.repeat(weights[i_N,:][np.newaxis],len(i_var),0)*E_xf,E_xf.T);
    
                elif (AF.regression == 'increment'):
                    if (stop_condition==0):
                        xf_tmp[:,i_var] = np.repeat(x[i_N,i_var][np.newaxis],AF.k,0) + successors[index_knn[i_N,:],:]-analogs_single[index_knn[i_N,:],:];
                    else:
                        xf_tmp[:,i_var] = np.repeat(x[i_N,i_var][np.newaxis],AF.k,0) + successors[index_knn[i_N,:],:]-analogs[index_knn[i_N,:],:];
                    # weighted mean and covariance
                    xf_mean[i_N,i_var] = np.sum(xf_tmp[:,i_var]*np.repeat(weights[i_N,:][np.newaxis].T,len(i_var),1),0);
                    E_xf = (xf_tmp[:,i_var]-np.repeat(xf_mean[i_N,i_var][np.newaxis],AF.k,0)).T;               
                    cov_xf = 1.0/(1-np.sum(np.power(weights[i_N,:],2)))*np.dot(np.repeat(weights[i_N,:][np.newaxis],len(i_var),0)*E_xf,E_xf.T);
    
                elif (AF.regression == 'local_linear'):   
                    if (AF.flag_model):
                        cata_model_tmp = np.concatenate((np.ones((AF.k,1)),cata_model[index_knn[i_N,:],:]),axis=1)
                        successors_tmp = successors[index_knn[i_N,:],:]
                        analogs_tmp = analogs[index_knn[i_N,:],:]
                        S = np.linalg.lstsq(cata_model_tmp,successors_tmp)[0]
                        ytest_A = np.dot(np.insert(AF.x_model[in_x-1,:],0,1),S)
                        tmp1 = np.dot(cata_model_tmp,S)
                        A = np.concatenate((np.ones((AF.k,1)),analogs_tmp,tmp1),axis=1)
                        tmp4 = np.linalg.lstsq(A,successors_tmp)[0]
                        mu = np.dot(np.hstack((1,x[i_N,i_var_neighboor],ytest_A)),tmp4)
                        res = successors_tmp-np.dot( A ,tmp4)
                        xf_tmp[:,i_var] = mu+res
                        xf_mean[i_N,i_var] = mu
                    else:
                        successors_tmp = successors[index_knn[i_N,:],:]
                        analogs_tmp = analogs[index_knn[i_N,:],:]
                        W = diags(np.sqrt(weights[i_N,:]))
                        A = np.concatenate((np.ones((len(analogs_tmp),1)),analogs_tmp),axis=1)
                        Aw = W.dot(A)	
                        Bw = W.dot(successors_tmp)
                        tmp4 = np.linalg.lstsq(Aw,Bw)[0]
                        tmp5 = np.linalg.lstsq(A,successors_tmp)[0]
                        mu = np.dot(np.hstack((1,x[i_N,i_var_neighboor])),tmp4)
                        #mu = np.dot(np.hstack((1,x[i_N,i_var])),tmp4)
                        #xf_mean[i_N,i_var] = mu
                        res = (successors_tmp- np.dot( A ,tmp5))                                                   
                        xf_tmp[:,i_var] = mu+res
                        # weighted mean and covariance
                        xf_mean[i_N,i_var] = mu
        
                    res = res.T
                    if len(i_var)>1:
                        cov_xf = np.cov(res)
                    else:
                        cov_xf = np.cov(res)[np.newaxis][np.newaxis]
                    # constant weights for local linear
                    weights[i_N,:] = 1.0/len(weights[i_N,:])   
                else:
                    print("Error: choose AF.regression between 'locally_constant', 'increment', 'local_linear' ")
                    quit() 
                # select the sampling method
                if (AF.sampling =='gaussian'):
                    # random sampling from the multivariate Gaussian distribution
                    xf[i_N,i_var] = np.random.multivariate_normal(xf_mean[i_N,i_var],cov_xf);
                elif (AF.sampling =='multinomial'):
                    # random sampling from the multinomial distribution of the weights
                    i_good = sample_discrete(weights[i_N,:],1,1);
                    xf[i_N,i_var] = xf_tmp[i_good,i_var];
                else:
                    print("Error: choose AF.sampling between 'gaussian', 'multinomial' ")
                    quit()
        # stop condition
        if (np.array_equal(i_var,np.array([n-1])) or (len(i_var) == n)):
            stop_condition = 1;            
        else:
            i_var = i_var + 1;
    
    return xf, xf_mean; # end