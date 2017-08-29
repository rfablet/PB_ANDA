#!/usr/bin/env python

""" test_AnDA_SST.py: Application of MS_AnDA to spatio-temporal interpolation of SST (sea surface temporature). """

__author__ = "Phi Huynh Viet"
__version__ = "2.0"
__date__ = "2017-08-01"
__email__ = "phi.huynhviet@telecom-bretagne.eu"


import numpy as np
import matplotlib.pyplot as plt
from AnDA_variables import PR, VAR, General_AF, AnDA_result
from AnDA_stat_functions import raPsd2dv1
from AnDA_transform_functions import Load_data, Gradient, Post_process, LR_perform
from AnDA_stat_functions import AnDA_RMSE, AnDA_correlate
from Multiscale_Assimilation import Multiscale_Assimilation as MS_AnDA
import pickle
np.random.seed(1)

###### Parameters setting for SST ###########################

PR_sst = PR()
PR_sst.n = 50 # dimension state
PR_sst.patch_r = 20 # size of patch
PR_sst.patch_c = 20 # size of patch
PR_sst.training_days = 2558 # num of training images: 2008-2014 
PR_sst.test_days = 364 # num of test images: 2015
PR_sst.lag = 3 # lag of time series: t -> t+lag
PR_sst.G_PCA = 20 # N_eof for global PCA
# Input dataset
PR_sst.path_X = '/home/phi/test_global_scale/AnDA_github/data/AMSRE/sst.npz'
PR_sst.path_OI = '/home/phi/test_global_scale/AnDA_github/data/AMSRE/OI.npz'
PR_sst.path_mask = '/home/phi/test_global_scale/AnDA_github/data/AMSRE/metop_mask.npz'
# Dataset automatically created during execution
PR_sst.path_X_lr = '/home/phi/test_global_scale/AnDA_github/data/AMSRE/sst_lr_30.npz'
PR_sst.path_dX_PCA = '/home/phi/test_global_scale/AnDA_github/data/AMSRE/dX_pca.npz'
PR_sst.path_index_patches = '/home/phi/test_global_scale/AnDA_github/data/AMSRE/list_pos.pickle'
PR_sst.path_neighbor_patches = '/home/phi/test_global_scale/AnDA_github/data/AMSRE/pair_pos.pickle'

AF_sst = General_AF()
AF_sst.flag_reduced =True # True: Reduced version of Local Linear AF
AF_sst.flag_cond = True # True: use Obs at t+lag as condition to select successors
                  # False: no condition in analog forecasting
AF_sst.flag_model = False # True: Use gradient, velocity as additional regressors in AF
AF_sst.flag_catalog = True # True: each catalog for each patch position
                    # False: only one catalog for all positions
AF_sst.flag_scale = True  # True: multi scale
                    # False: one scale
AF_sst.cluster = 10       # clusterized version AF
AF_sst.k = 10 # number of analogs
AF_sst.k_initial = 100 # retrieving k_initial nearest neighbors, then using condition to retrieve k analogs 
AF_sst.neighborhood = np.ones([PR_sst.n,PR_sst.n]) # global analogs
AF_sst.neighborhood = np.eye(PR_sst.n)+np.diag(np.ones(PR_sst.n-1),1)+ np.diag(np.ones(PR_sst.n-1),-1)+ \
                       np.diag(np.ones(PR_sst.n-2),2)+np.diag(np.ones(PR_sst.n-2),-2)
AF_sst.neighborhood[0:2,:5] = 1
AF_sst.neighborhood[PR_sst.n-2:,PR_sst.n-5:] = 1 # local analogs
AF_sst.neighborhood[PR_sst.n-2:,PR_sst.n-5:] = 1 # local analogs
AF_sst.regression = 'local_linear' 
AF_sst.sampling = 'gaussian' 
AF_sst.B = 0.01
AF_sst.R = 0.05

"""  Loading data  """
VAR_sst = VAR()
VAR_sst = Load_data(PR_sst) 

file_tmp = np.load('/home/phi/test_global_scale/AnDA_github/data/AMSRE/DINEOF.npz')
VAR_sst.dX_cond = file_tmp['sst_lr']
del file_tmp    
VAR_sst.dX_cond = VAR_sst.dX_cond[:PR_sst.test_days:PR_sst.lag,:,:]
VAR_sst.dX_cond = VAR_sst.dX_cond-VAR_sst.X_lr[PR_sst.training_days:,:,:]
for i in range(len(VAR_sst.dX_cond)):
    VAR_sst.dX_cond[i,~np.isnan(VAR_sst.Obs_test[i,:,:])] = VAR_sst.Obs_test[i,~np.isnan(VAR_sst.Obs_test[i,:,:])]   


"""  Assimilation  """
r_start = 55
c_start = 70
r_length = 50
c_length = 50
AnDA_sst_3 = AnDA_result()
MS_AnDA_sst = MS_AnDA(VAR_sst, PR_sst, AF_sst)
AnDA_sst_3 = MS_AnDA_sst.multi_patches_assimilation('/home/phi/test_global_scale/AnDA_github/data/AMSRE/AnDA/AnDA_1.pickle', 9, r_start, r_length, c_start, c_length)

""" Postprocessing step: remove block artifact (do PCA twice gives perfect results) """
Pre_filtered = np.copy(VAR_sst.dX_orig[:PR_sst.training_days,r_start:r_start+r_length,c_start:c_start+c_length]+VAR_sst.X_lr[:PR_sst.training_days,r_start:r_start+r_length,c_start:c_start+c_length])
Pre_filtered = np.concatenate((Pre_filtered,AnDA_sst_3.itrp_AnDA),axis=0)
AnDA_sst_3.itrp_postAnDA = Post_process(Pre_filtered,len(VAR_sst.Obs_test),33,150) 
AnDA_sst_3.rmse_postAnDA = AnDA_RMSE(AnDA_sst_3.itrp_postAnDA,AnDA_sst_3.GT)
AnDA_sst_3.corr_postAnDA = AnDA_correlate(AnDA_sst_3.itrp_postAnDA-AnDA_sst_3.LR,AnDA_sst_3.GT-AnDA_sst_3.LR)

"""  Reload saved AnDA result  """
with open('/home/phi/test_global_scale/AnDA_github/data/AMSRE/AnDA/AnDA_1.pickle', 'rb') as handle:
    AnDA_sst_1 = pickle.load(handle)    


"""  Radial Power Spetral  """
day = 109
resSLA = 0.25
f0, Pf_AnDA  = raPsd2dv1(AnDA_sst_1.itrp_AnDA[day,:,:],resSLA,True)
f1, Pf_postAnDA  = raPsd2dv1(AnDA_sst_1.itrp_postAnDA[day,:,:],resSLA,True)
f2, Pf_GT    = raPsd2dv1(AnDA_sst_1.GT[day,:,:],resSLA,True)
f3, Pf_OI    = raPsd2dv1(AnDA_sst_1.itrp_OI[day,:,:],resSLA,True)
wf1         = 1/f1
wf2         = 1/f2
wf3         = 1/f3
plt.figure()
plt.loglog(wf2,Pf_GT,label='GT')
plt.loglog(wf3,Pf_OI,label='OI')
plt.loglog(wf1,Pf_AnDA,label='AnDA')
plt.loglog(wf2,Pf_postAnDA,label='postAnDA')
plt.gca().invert_xaxis()
plt.legend()


# display
colormap='jet'
plt.figure()
plt.ion()
for day in range(165,len(AnDA_sst_1.GT)):
    plt.clf()
    gt = AnDA_sst_1.GT[day,:,:]
    obs = AnDA_sst_1.Obs[day,:,:]
    AnDA =  AnDA_sst_1.itrp_AnDA[day,:,:]
    OI = AnDA_sst_1.itrp_OI[day,:,:]
    Post_AnDA = AnDA_sst_1.itrp_postAnDA[day,:,:]
    Grad_gt = Gradient(gt,2)
    Grad_AnDA = Gradient(AnDA,2)
    Grad_Post_AnDA = Gradient(Post_AnDA,2)
    Grad_OI = Gradient(OI,2)
    vmin = np.nanmin(gt)
    vmax = np.nanmax(gt)
    vmin_2 = np.nanmin(Grad_gt)
    vmax_2 = np.nanmax(Grad_gt)
    plt.subplot(3,3,1)
    plt.imshow(gt,aspect='auto',cmap=colormap,vmin=vmin,vmax=vmax)
    plt.colorbar()
    plt.title('GT')
    plt.subplot(3,3,2)
    plt.imshow(obs,aspect='auto',cmap=colormap,vmin=vmin,vmax=vmax)
    plt.colorbar()
    plt.title('Obs')
    plt.subplot(3,3,3)
    plt.imshow(Grad_gt,aspect='auto',cmap=colormap,vmin=vmin_2,vmax=vmax_2)
    plt.colorbar()
    plt.title(r"$\nabla_{GT}$")
    plt.subplot(3,3,4)
    plt.imshow(OI,aspect='auto',cmap=colormap,vmin=vmin,vmax=vmax)
    plt.colorbar()  
    plt.title('OI')
    plt.subplot(3,3,5)
    plt.imshow(AnDA,aspect='auto',cmap=colormap,vmin=vmin,vmax=vmax)
    plt.colorbar()  
    plt.title('AnDA')
    plt.subplot(3,3,6)
    plt.imshow(Post_AnDA,aspect='auto',cmap=colormap,vmin=vmin,vmax=vmax)
    plt.colorbar() 
    plt.title('Post_AnDA')
    plt.subplot(3,3,8)
    plt.imshow(Grad_AnDA,aspect='auto',cmap=colormap,vmin=vmin_2,vmax=vmax_2)
    plt.colorbar()
    plt.title(r"$\nabla_{AnDA}$")
    plt.subplot(3,3,7)
    plt.imshow(Grad_OI,aspect='auto',cmap=colormap,vmin=vmin_2,vmax=vmax_2)
    plt.colorbar()
    plt.title(r"$\nabla_{OI}$")
    plt.subplot(3,3,9)
    plt.imshow(Grad_Post_AnDA,aspect='auto',cmap=colormap,vmin=vmin_2,vmax=vmax_2)
    plt.colorbar()
    plt.title(r"$\nabla_{Post_AnDA}$")
    plt.draw()
    plt.waitforbuttonpress()
    
colormap='jet'
plt.ion()
for day in range(50,len(VAR_sst.dX_GT_test)):
    plt.clf()
    gt = VAR_sst.dX_GT_test[day,:,:]   
    obs = VAR_sst.Obs_test[day,:,:]    
    itrp = VAR_sst.Optimal_itrp[day,:,:]   

    vmin = np.nanmin(gt)
    vmax = np.nanmax(gt)
    plt.subplot(1,3,1)
    plt.imshow(gt,aspect='auto',cmap=colormap,vmin=vmin,vmax=vmax)
    plt.colorbar()
    plt.title('GT')
    plt.subplot(1,3,2)
    plt.imshow(obs,aspect='auto',cmap=colormap,vmin=vmin,vmax=vmax)
    plt.colorbar() 
    plt.title('Obs')
    plt.subplot(1,3,3)
    plt.imshow(itrp,aspect='auto',cmap=colormap,vmin=vmin,vmax=vmax)
    plt.colorbar()  
    plt.title('OI')
    plt.draw()
    plt.waitforbuttonpress()


# 



