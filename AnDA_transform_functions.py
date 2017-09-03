#!/usr/bin/env python

""" AnDA_transform_functions.py: Apply PCA to decompose the spatio-temporal variabilities into a multi-scale representation. """

__author__ = "Phi Huynh Viet"
__version__ = "1.0"
__date__ = "2017-08-01"
__email__ = "phi.huynhviet@telecom-bretagne.eu"


import os
import numpy as np
from AnDA_variables import VAR
from sklearn.decomposition import PCA
import pickle
from scipy.ndimage.morphology import distance_transform_edt as bwdist
import cv2

def LR_perform(HR, path_LR, N_eof): 
    """ Perform global PCA retrieving LR product """   
    if not os.path.exists(path_LR):
        lr = np.copy(HR).reshape(HR.shape[0],-1)
        tmp = lr[0,:]
        sea_v2 = np.where(~np.isnan(tmp))[0]
        lr_no_land = lr[:,sea_v2]
        pca = PCA(n_components=N_eof)
        score_global = pca.fit_transform(lr_no_land)
        coeff_global = pca.components_.T
        mu_global = pca.mean_
        print np.sum(pca.explained_variance_ratio_)
        DataReconstructed_global = np.dot(score_global, coeff_global.T) +mu_global
        lr[:,sea_v2] = DataReconstructed_global
        lr = lr.reshape(HR.shape)
        if not (path_LR == ''):
            fid = netCDF4.Dataset(path_LR,'w', format='NETCDF4')
            fid.createDimension('lat', HR.shape[1])        
            fid.createDimension('lon', HR.shape[2])
            fid.createDimension('time', HR.shape[0])
            fid.createVariable("lr", "f8",("time","lat", "lon"),fill_value=-32768,zlib=True)
            fid.variables['lr'][:] = lr
            fid.close()
    else:
        file_tmp = netCDF4.Dataset(path_LR,'r')
        lr = file_tmp.variables.items()[0][1][:]
        file_tmp.close()         
    return lr
    
   
def Patch_define(sample, path_indices, path_neighbor_indices, patch_r, patch_c):
    """ define spatial position of each patch over the whole image """  
    if not os.path.exists(path_indices):
        r = np.arange(0,patch_r)
        c = np.arange(0,patch_c)
        ind = 0
        index_patch = {}
        while (len(r)==patch_r):
            while (len(c)==patch_c):                
                if (np.sum(np.isnan(sample[np.ix_(r,c)]))==0):   
                    index_patch[ind] = [r[0], c[0]]
                    ind = ind+1            
                c = c+5
                c = c[c<sample.shape[1]]         
            r = r+5
            r = r[r<sample.shape[0]]
            c = np.arange(0,patch_c)
    
        neighbor_patches = {}
        for i in range(ind):
            r_test = index_patch[i][0]
            c_test = index_patch[i][1]
            pair_tmp = []
            for j in range(r_test-3*5,r_test+3*5+1,5):
                for k in range(c_test-3*5,c_test+3*5+1,5):
                    if (j>=0) & (k>=0) & (j<=(sample.shape[0]-patch_r)) & (k<=(sample.shape[1]-patch_c) ):
                        if (np.sum(np.isnan(sample[j:j+patch_r,k:k+patch_c]))==0):
                            pair_tmp.append(index_patch.keys()[index_patch.values().index([j,k])])
            neighbor_patches[i] = pair_tmp
        with open(path_neighbor_indices, 'wb') as handle:
            pickle.dump(neighbor_patches, handle, protocol=pickle.HIGHEST_PROTOCOL)            
        with open(path_indices, 'wb') as handle:
            pickle.dump(index_patch, handle, protocol=pickle.HIGHEST_PROTOCOL)   
    else:
        with open(path_neighbor_indices, 'rb') as handle:
            neighbor_patches = pickle.load(handle)               
        with open(path_indices, 'rb') as handle:
            index_patch = pickle.load(handle)
    return index_patch, neighbor_patches
 
def PCA_perform(dX, path_dX_pca, N_eof, N_patches, patch_r, patch_c): 
    """ Perform PCA on dX to retrieve catalog for AnDA """  
    if not os.path.exists(path_dX_pca):
        pca = PCA(n_components=N_eof)
        patch_dx_full = np.zeros((N_patches*dX.shape[0],patch_r,patch_c))
        r = np.arange(0,patch_r)
        c = np.arange(0,patch_c)
        ind = 0
        while (len(r)==patch_r):
            while (len(c)==patch_c):
                tmp = dX[:,r[0]:r[0]+patch_r,c[0]:c[0]+patch_c]
                if (np.sum(np.isnan(tmp))==0):   
                    patch_dx_full[ind*dX.shape[0]:(ind+1)*dX.shape[0],:,:] = tmp
                    ind = ind+1
                c = c+5
                c = c[c<dX.shape[2]]    
            r = r+5
            r = r[r<dX.shape[1]]
            c = np.arange(0,patch_c)         
        patch_dx_full = patch_dx_full.reshape(patch_dx_full.shape[0],-1)
        dX_train = pca.fit_transform(patch_dx_full)
        dX_eof_coeff = pca.components_.T
        dX_eof_mu = pca.mean_
        if not (path_dX_pca == ''):
            fid = netCDF4.Dataset(path_dX_pca,'w', format='NETCDF4')
            fid.createDimension('lat', N_eof)        
            fid.createDimension('lon', patch_r*patch_c)
            fid.createDimension('time', N_patches*dX.shape[0])
            fid.createVariable("dX_train", "f8",("time","lat"),fill_value=-32768,zlib=True)
            fid.variables['dX_train'][:] = dX_train
            fid.createVariable("dX_eof_coeff", "f8",("lon","lat"),fill_value=-32768)
            fid.variables['dX_eof_coeff'][:] = dX_eof_coeff
            fid.createVariable("dX_eof_mu", "f8",("lon",),fill_value=-32768)
            fid.variables['dX_eof_mu'][:] = dX_eof_mu
            fid.close()
    else:
        file_tmp = netCDF4.Dataset(path_dX_pca,'r')
        dX_train = file_tmp.variables['dX_train'][:]
        dX_eof_coeff = file_tmp.variables['dX_eof_coeff'][:]
        dX_eof_mu = file_tmp.variables['dX_eof_mu'][:]
        file_tmp.close()
    return dX_train, dX_eof_coeff, dX_eof_mu

def sum_overlapping(tmp1,tmp2):
    """ calculate overlapping area between patches  """  
    bw = np.zeros(tmp1.shape)
    D = bwdist(bw==0)
    D = np.exp(-D)
    w1 = np.nan*np.zeros(tmp1.shape)
    w2 = np.nan*np.zeros(tmp1.shape)
    w2[:,0:5] = np.fliplr(D[:,0:5])
    w2[0:5,4:] = np.flipud(D[0:5,4:])
    w1[:,0:5] = D[:,0:5]
    w1[0:5,4:] = D[0:5,4:]
    wsum = w1+w2
    w1 = w1/wsum
    w2 = w2/wsum
    tmp2[~np.isnan(tmp2)] = 2*tmp2[~np.isnan(tmp2)]*w2[~np.isnan(tmp2)]
    tmp1[~np.isnan(tmp2)] = 2*tmp1[~np.isnan(tmp2)]*w1[~np.isnan(tmp2)]
    overlap = np.array([tmp2,tmp1])
    overlap = np.nanmean(overlap,axis=0)
    return overlap
            
def Post_process(Pre_filtered, L, size_w, n_eof):
    """ Remove block artifact due to overlapping patches """
    Post_filtered =  np.nan*np.zeros((L,Pre_filtered.shape[1],Pre_filtered.shape[2]))
    r_sub = np.arange(0,size_w)
    c_sub = np.arange(0,size_w)
    ind = 0
    while (len(r_sub)>0):
        while (len(c_sub)>0):
            tmp = Pre_filtered[:,r_sub[0]:r_sub[-1]+1,c_sub[0]:c_sub[-1]+1]
            tmp = tmp.reshape(tmp.shape[0],-1)
            sea_mask = np.where(~np.isnan(tmp[0,:]))[0]
            tmp_no_land = tmp[:,sea_mask]
            if (len(sea_mask)>0):   
                pca = PCA(n_components=len(sea_mask))
                if (len(sea_mask)>n_eof):
                    pca = PCA(n_components=n_eof)
                score_tmp = pca.fit_transform(tmp_no_land)
                coeff_tmp = pca.components_.T
                mu_tmp = pca.mean_
                DataReconstructed_tmp = np.dot(score_tmp, coeff_tmp.T) +mu_tmp
                tmp[:,sea_mask] = DataReconstructed_tmp                 
                for u in range(0,L):
                    tmp2 = Post_filtered[u,r_sub[0]:r_sub[-1]+1,c_sub[0]:c_sub[-1]+1]
                    tmp1 = tmp[Pre_filtered.shape[0]-L+u,:].reshape(tmp2.shape)                 
                    Post_filtered[u,r_sub[0]:r_sub[-1]+1,c_sub[0]:c_sub[-1]+1] = sum_overlapping(tmp1,tmp2)
                ind = ind+1
            c_sub = c_sub+size_w-5
            c_sub = c_sub[c_sub<Pre_filtered.shape[2]]    
        r_sub = r_sub+size_w-5
        r_sub = r_sub[r_sub<Pre_filtered.shape[1]]
        c_sub = np.arange(0,size_w)         
    return Post_filtered

def Gradient(img, order):
    """ calcuate x, y gradient and magnitude """ 
    sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=3)
    sobelx = sobelx/8.0
    sobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=3)
    sobely = sobely/8.0
    sobel_norm = np.sqrt(sobelx*sobelx+sobely*sobely)
    if (order==0):
        return sobelx
    elif (order==1):
        return sobely
    else:
        return sobel_norm

def Imputing_NaN(data, invalid=None):
    """
    Replace the value of invalid 'data' cells (indicated by 'invalid') 
    by the value of the nearest valid data cell
    """
    import scipy.ndimage as nd
    if invalid is None: invalid = np.isnan(data)
    ind = nd.distance_transform_edt(invalid, return_distances=False, return_indices=True)
    return data[tuple(ind)]
    
def VE_Dineof(PR, HR, OI, mask, N_eof, n_iter):
    start_time = time.time()
    lr = np.copy(OI)
    X_initialization = np.copy(HR)
    print np.mean(AnDA_RMSE(HR[PR.training_days:],lr))
    for i in tqdm(range(n_iter)):        
        for j in range(PR.test_days):
            X_initialization[PR.training_days+j,np.isnan(mask[j,:,:])] = lr[j,np.isnan(mask[j,:,:])]
        lr = LR_perform(X_initialization, '', N_eof)
        lr = lr[PR.training_days:,:,:]
        print np.mean(AnDA_RMSE(HR[PR.training_days:],lr))
    print("---Processing time:  %s seconds ---" % (time.time() - start_time))    
    return lr

def MS_VE_Dineof(PR, HR, OI, mask, N_eof, n_iter):
  
    X_initialization = np.copy(HR)    
    Post_filtered =  np.nan*np.zeros(OI.shape)
    r_sub = np.arange(0,20)
    c_sub = np.arange(0,20)
    ind = 0
    while (len(r_sub)>0):
        while (len(c_sub)>0):
            lr = np.copy(OI[:,r_sub[0]:r_sub[-1]+1,c_sub[0]:c_sub[-1]+1])
            sea_mask = np.where(~np.isnan(lr[0,:,:].flatten()))[0]
            if (len(sea_mask)>0):  
                for i in range(n_iter):  
                    tmp = X_initialization[:,r_sub[0]:r_sub[-1]+1,c_sub[0]:c_sub[-1]+1]
                    for j in range(PR.test_days):
                        tmp[PR.training_days+j,np.isnan(mask[j,r_sub[0]:r_sub[-1]+1,c_sub[0]:c_sub[-1]+1])] = lr[j,np.isnan(mask[j,r_sub[0]:r_sub[-1]+1,c_sub[0]:c_sub[-1]+1])]
                    tmp = tmp.reshape(tmp.shape[0],-1)
                    tmp_no_land = tmp[:,sea_mask]                     
                    pca = PCA(n_components=len(sea_mask))
                    if (len(sea_mask)>N_eof):
                        pca = PCA(n_components=N_eof)
                    score_tmp = pca.fit_transform(tmp_no_land)
                    coeff_tmp = pca.components_.T
                    mu_tmp = pca.mean_
                    DataReconstructed_tmp = np.dot(score_tmp, coeff_tmp.T) +mu_tmp
                    tmp[:,sea_mask] = DataReconstructed_tmp         
                    lr = tmp[PR.training_days:,:].reshape(lr.shape)
                for u in range(0,PR.test_days):
                    tmp2 = Post_filtered[u,r_sub[0]:r_sub[-1]+1,c_sub[0]:c_sub[-1]+1]
                    tmp1 = lr[u,:,:]               
                    Post_filtered[u,r_sub[0]:r_sub[-1]+1,c_sub[0]:c_sub[-1]+1] = sum_overlapping(tmp1,tmp2)
                ind = ind+1
                print ind
            c_sub = c_sub+15
            c_sub = c_sub[c_sub<HR.shape[2]]    
        r_sub = r_sub+15
        r_sub = r_sub[r_sub<HR.shape[1]]
        c_sub = np.arange(0,20)         
    return Post_filtered

def Load_data(PR):
    """ Load necessary datasets """
    VAR_ = VAR()
    try:
        file_tmp = netCDF4.Dataset(PR.path_X,'r')
        X = file_tmp.variables.items()[0][1][:].astype('float64')
        X = X[-(PR.training_days+PR.test_days):]
        del file_tmp
    except:
        print "Cannot find dataset: %s" %(PR.path_X)
        quit()
    # Load Optimal Interpolation as LR product
    try:
        file_tmp = netCDF4.Dataset(PR.path_OI,'r')
        OI = file_tmp.variables.items()[0][1][:].astype('float64')
        del file_tmp
    except:
        print "Cannot find dataset: %s" %(PR.path_OI)
        quit()
    # Load Alongtrack SLA as observation mask
    try:
        file_tmp = netCDF4.Dataset(PR.path_mask,'r')
        mask = file_tmp.variables.items()[0][1][:].astype('float64')
        del file_tmp
    except:
        print "Cannot find dataset: %s" %(PR.path_mask)
        quit()
    # First step, filling missing data by OI
    X_initialization = np.copy(X)
    for i in range(PR.test_days):
        X_initialization[PR.training_days+i,np.isnan(mask[i,:,:])] = OI[i,np.isnan(mask[i,:,:])]
    # Perform global PCA to find LR product
    X_lr = LR_perform(X_initialization,PR.path_X_lr,PR.G_PCA)
    VAR_.X_lr = X_lr
    if (PR.flag_scale):
        # Retrieve dSLA: detail product, used for AnDA
        VAR_.dX_orig = X-X_lr
        # Retrieve dX_OI for comparison with AnDA's results
        VAR_.Optimal_itrp = OI - X_lr[PR.training_days:,:,:]
        del X_initialization, OI, X, X_lr
        # used for retrieving specific catalog for each patch position
        VAR_.index_patch, VAR_.neighbor_patchs = Patch_define(VAR_.dX_orig[0,:,:],PR.path_index_patches,PR.path_neighbor_patches, PR.patch_r, PR.patch_c)
        # Retrieve dSLA_patch in PCA space: catalog used for AnDA
        VAR_.dX_train, VAR_.dX_eof_coeff, VAR_.dX_eof_mu = PCA_perform(VAR_.dX_orig[:PR.training_days,:,:],PR.path_dX_PCA,PR.n,len(VAR_.index_patch),PR.patch_r,PR.patch_c)
        # dSLA_GT as reference, dSLA_Obs by applying alongtrack mask
        VAR_.dX_GT_test = np.copy(VAR_.dX_orig[PR.training_days:,:,:])
        VAR_.Obs_test = np.copy(VAR_.dX_GT_test)   
        for i in range(PR.test_days):
            VAR_.Obs_test[i,np.isnan(mask[i,:,:])] = np.nan
        VAR_.Obs_test = VAR_.Obs_test[:PR.test_days:PR.lag,:,:]
        VAR_.dX_GT_test = VAR_.dX_GT_test[:PR.test_days:PR.lag,:,:]
        VAR_.Optimal_itrp = VAR_.Optimal_itrp[:PR.test_days:PR.lag,:,:]
        VAR_.X_lr = np.concatenate((VAR_.X_lr[:PR.training_days,:,:],VAR_.X_lr[PR.training_days::PR.lag,:,:]),axis=0)
        VAR_.dX_orig = np.concatenate((VAR_.dX_orig[:PR.training_days,:,:],VAR_.dX_orig[PR.training_days::PR.lag,:,:]),axis=0)
    else:
        VAR_.dX_GT_test = np.copy(X[PR.training_days:,:,:])
        VAR_.dX_GT_test = VAR_.dX_GT_test[:PR.test_days:PR.lag,:,:]
        VAR_.Obs_test = np.copy(VAR_.dX_GT_test)   
        for i in range(PR.test_days):
            VAR_.Obs_test[i,np.isnan(mask[i,:,:])] = np.nan
        VAR_.Optimal_itrp = np.copy(OI)
        VAR_.Optimal_itrp = VAR_.Optimal_itrp[:PR.test_days:PR.lag,:,:]
        #VAR_.dX_orig = np.concatenate((X[:PR.training_days,:,:],VAR_.Optimal_itrp),axis=0)
        VAR_.dX_orig = np.copy(X_initialization)
        VAR_.dX_orig = np.concatenate((VAR_.dX_orig[:PR.training_days,:,:],VAR_.dX_orig[PR.training_days::PR.lag,:,:]),axis=0)
        
    return VAR_


