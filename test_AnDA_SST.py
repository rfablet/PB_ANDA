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
from AnDA_transform_functions import Load_data, Gradient, Post_process, LR_perform, VE_Dineof, MS_VE_Dineof, Imputing_NaN, PCA_perform
from AnDA_stat_functions import AnDA_RMSE, AnDA_correlate
from AnDA_Multiscale_Assimilation import Multiscale_Assimilation as MS_AnDA
import pickle
np.random.seed(1)

###### Parameters setting for SST ###########################

PR_sst = PR()
PR_sst.flag_scale = True  # True: multi scale, False: one scale                    
PR_sst.n = 50 # dimension state
PR_sst.patch_r = 20 # size of patch
PR_sst.patch_c = 20 # size of patch
PR_sst.training_days = 2558 # num of training images: 2008-2014 
PR_sst.test_days = 364 # num of test images: 2015
PR_sst.lag = 3 # lag of time series: t -> t+lag
PR_sst.G_PCA = 20 # N_eof for global PCA
# Input dataset
PR_sst.path_X = './sst.nc'
PR_sst.path_OI = './OI.nc'
PR_sst.path_mask = './metop_mask.nc'
# Dataset automatically created during execution
PR_sst.path_X_lr = './sst_lr.nc'
PR_sst.path_dX_PCA = './dX_pca.nc'
PR_sst.path_index_patches = './list_pos.pickle'
PR_sst.path_neighbor_patches = './pair_pos.pickle'

AF_sst = General_AF()
AF_sst.flag_reduced =True # True: Reduced version of Local Linear AF
AF_sst.flag_cond = False # True: use Obs at t+lag as condition to select successors
                  # False: no condition in analog forecasting
AF_sst.flag_model = False # True: Use gradient, velocity as additional regressors in AF
AF_sst.flag_catalog = True # True: each catalog for each patch position
                    # False: only one catalog for all positions
AF_sst.cluster = 1       # clusterized version AF
AF_sst.k = 200 # number of analogs
AF_sst.k_initial = 200 # retrieving k_initial nearest neighbors, then using condition to retrieve k analogs 
AF_sst.neighborhood = np.ones([PR_sst.n,PR_sst.n]) # global analogs
AF_sst.regression = 'local_linear' 
AF_sst.sampling = 'gaussian' 
AF_sst.B = 0.01
AF_sst.R = 0.05

"""  Loading data  """
VAR_sst = VAR()
VAR_sst = Load_data(PR_sst) 

"""---------- Additional constraints for SST ------------ """
""" Loading condition """
VAR_sst.dX_cond = np.copy(VAR_sst.Optimal_itrp)
for i in range(len(VAR_sst.dX_cond)):
    VAR_sst.dX_cond[i,~np.isnan(VAR_sst.Obs_test[i,:,:])] = VAR_sst.Obs_test[i,~np.isnan(VAR_sst.Obs_test[i,:,:])]   
""" Loading and calculating physical variables (velocities, gradient of LR"""
import scipy.io
from tqdm import tqdm
file_tmp = netCDF4.Dataset('./velocity_u.nc','r')
v_x = file_tmp.variables.items()[0][1][:]
file_tmp.close()
v_x = np.concatenate((v_x[:PR_sst.training_days,:,:],v_x[PR_sst.training_days::PR_sst.lag,:,:]),axis=0)
for i in tqdm(range(len(v_x))):
    v_x[i,:,:] = Imputing_NaN(v_x[i,:,:])
v_x[np.isnan(VAR_sst.dX_orig)] = np.nan
vx_train, vx_eof_coeff, vx_eof_mu = PCA_perform(v_x[:PR_sst.training_days,:,:],'/home/phi/test_global_scale/AnDA_github/data/AMSRE/vx_pca.nc',PR_sst.n,len(VAR_sst.index_patch),PR_sst.patch_r,PR_sst.patch_c)

file_tmp = netCDF4.Dataset('./velocity_v.nc','r')
v_y = file_tmp.variables.items()[0][1][:]
file_tmp.close()
v_y = np.concatenate((v_y[:PR_sst.training_days,:,:],v_y[PR_sst.training_days::PR_sst.lag,:,:]),axis=0)
for i in tqdm(range(len(v_y))):
    v_y[i,:,:] = Imputing_NaN(v_y[i,:,:])
v_y[np.isnan(VAR_sst.dX_orig)] = np.nan
vy_train, vy_eof_coeff, vy_eof_mu = PCA_perform(v_y[:PR_sst.training_days,:,:],'/home/phi/test_global_scale/AnDA_github/data/AMSRE/vy_pca.nc',PR_sst.n,len(VAR_sst.index_patch),PR_sst.patch_r,PR_sst.patch_c) 

g_x = np.copy(VAR_sst.X_lr)
for i in tqdm(range(len(g_x))):    
    tmp = Gradient(g_x[i,:,:],0)
    g_x[i,:,:] = Imputing_NaN(tmp)
g_x[np.isnan(VAR_sst.X_lr)] = np.nan   
gx_train, gx_eof_coeff, gx_eof_mu = PCA_perform(g_x[:PR_sst.training_days,:,:],'/home/phi/test_global_scale/AnDA_github/data/AMSRE/gx_pca.nc',PR_sst.n,len(VAR_sst.index_patch),PR_sst.patch_r,PR_sst.patch_c)

g_y = np.copy(VAR_sst.X_lr)
for i in tqdm(range(len(g_y))):    
    tmp = Gradient(g_y[i,:,:],1)
    g_y[i,:,:] = Imputing_NaN(tmp)
g_y[np.isnan(VAR_sst.X_lr)] = np.nan
gy_train, gy_eof_coeff, gy_eof_mu = PCA_perform(g_y[:PR_sst.training_days,:,:],'/home/phi/test_global_scale/AnDA_github/data/AMSRE/gy_pca.nc',PR_sst.n,len(VAR_sst.index_patch),PR_sst.patch_r,PR_sst.patch_c)

constraint_1 = [vx_train,v_x[PR_sst.training_days:],vx_eof_coeff,vx_eof_mu]
constraint_2 = [vy_train,v_y[PR_sst.training_days:],vy_eof_coeff,vy_eof_mu]
constraint_3 = [gx_train,g_x[PR_sst.training_days:],gx_eof_coeff,gx_eof_mu]
constraint_4 = [gy_train,g_y[PR_sst.training_days:],gy_eof_coeff,gy_eof_mu]
VAR_sst.model_constraint = []
VAR_sst.model_constraint.append(constraint_1)
VAR_sst.model_constraint.append(constraint_2)

"""  Assimilation  """
r_start = 55
c_start = 70
r_length = 50
c_length = 50
level = 20 # 22 patches executed simultaneously

saved_path =  'saved_path.pickle'
AnDA_sst_1 = AnDA_result()
MS_AnDA_sst = MS_AnDA(VAR_sst, PR_sst, AF_sst)
AnDA_sst_1 = MS_AnDA_sst.multi_patches_assimilation(level, r_start, r_length, c_start, c_length)

""" Postprocessing step: remove block artifact (do PCA twice gives perfect results) """
Pre_filtered = np.copy(VAR_sst.dX_orig[:PR_sst.training_days,r_start:r_start+r_length,c_start:c_start+c_length]+VAR_sst.X_lr[:PR_sst.training_days,r_start:r_start+r_length,c_start:c_start+c_length])
Pre_filtered = np.concatenate((Pre_filtered,AnDA_sst_1.itrp_AnDA),axis=0)
AnDA_sst_1.itrp_postAnDA = Post_process(Pre_filtered,len(VAR_sst.Obs_test),33,150) 
AnDA_sst_1.rmse_postAnDA = AnDA_RMSE(AnDA_sst_1.itrp_postAnDA,AnDA_sst_1.GT)
AnDA_sst_1.corr_postAnDA = AnDA_correlate(AnDA_sst_1.itrp_postAnDA-AnDA_sst_1.LR,AnDA_sst_1.GT-AnDA_sst_1.LR)

""" Save AnDA result """         
with open(saved_path, 'wb') as handle:
    pickle.dump(AnDA_sst_1, handle)
    
"""  Reload saved AnDA result  """
with open(saved_path, 'rb') as handle:
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



