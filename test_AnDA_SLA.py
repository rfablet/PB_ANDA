#!/usr/bin/env python

""" test_AnDA_SLA.py: Application of MS_AnDA to spatio-temporal interpolation of SLA (sea level anomaly). """

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
PR_sla = PR()
PR_sla.n = 15 # dimension state
PR_sla.patch_r = 20 # size of patch
PR_sla.patch_c = 20 # size of patch
PR_sla.training_days = 6209-122 # num of training images: 2008-2014 
PR_sla.test_days = 122 # num of test images: 2015
PR_sla.lag = 1 # lag of time series: t -> t+lag
PR_sla.G_PCA = 12 # N_eof for global PCA
# Input dataset
PR_sla.path_X = '/home/phi/test_global_scale/AnDA_github/data/SLA/SLA.npz' 
PR_sla.path_OI = '/home/phi/test_global_scale/AnDA_github/data/SLA/OI.npz'
PR_sla.path_mask = '/home/phi/test_global_scale/AnDA_github/data/SLA/alongtrack_mask.npz'
# Dataset automatically created during execution
PR_sla.path_X_lr = '/home/phi/test_global_scale/AnDA_github/data/SLA/sla_lr.npz'
PR_sla.path_dX_PCA = '/home/phi/test_global_scale/AnDA_github/data/SLA/dX_pca.npz'
PR_sla.path_index_patches = '/home/phi/test_global_scale/AnDA_github/data/SLA/list_pos.pickle'
PR_sla.path_neighbor_patches = '/home/phi/test_global_scale/AnDA_github/data/SLA/pair_pos.pickle'

AF_sla = General_AF()
AF_sla.flag_reduced = False # True: Reduced version of Local Linear AF
AF_sla.flag_cond = False # True: use Obs at t+lag as condition to select successors
                     # False: no condition in analog forecasting
AF_sla.flag_model = False # True: Use gradient, velocity as additional regressors in AF
AF_sla.flag_catalog = True # True: each catalog for each patch position
                    # False: only one catalog for all positions
AF_sla.flag_scale = True  # True: multi scale
                    # False: one scale
AF_sla.cluster = 1       # clusterized version AF
AF_sla.k = 50 # number of analogs
AF_sla.k_initial = 200 # retrieving k_initial nearest neighbors, then using condition to retrieve k analogs 
AF_sla.neighborhood = np.ones([PR_sla.n,PR_sla.n]) # global analogs
AF_sla.regression = 'local_linear' 
AF_sla.sampling = 'gaussian' 
AF_sla.B = 0.0001
AF_sla.R = 0.0001

"""  Loading data  """
VAR_sla = VAR()
VAR_sla = Load_data(PR_sla) 

"""  Assimilation  """
r_start = 75
c_start = 35
r_length = 65
c_length = 65
AnDA_sla_1 = AnDA_result()
MS_AnDA_sla = MS_AnDA(VAR_sla, PR_sla, AF_sla)
AnDA_sla_1 = MS_AnDA_sla.multi_patches_assimilation('/home/phi/test_global_scale/AnDA_github/data/SLA/AnDA/AnDA_1.pickle', 8, r_start, r_length, c_start, c_length)

""" Postprocessing step: remove block artifact (do PCA twice gives perfect results) """
Pre_filtered = np.copy(VAR_sla.dX_orig[:PR_sla.training_days,r_start:r_start+r_length,c_start:c_start+c_length]+VAR_sla.X_lr[:PR_sla.training_days,r_start:r_start+r_length,c_start:c_start+c_length])
Pre_filtered = np.concatenate((Pre_filtered,AnDA_sla_1.itrp_AnDA),axis=0)
AnDA_sla_1.itrp_postAnDA = Post_process(Pre_filtered,len(VAR_sla.Obs_test),17) 
Pre_filtered = np.copy(VAR_sla.dX_orig[:PR_sla.training_days,r_start:r_start+r_length,c_start:c_start+c_length]+VAR_sla.X_lr[:PR_sla.training_days,r_start:r_start+r_length,c_start:c_start+c_length])
Pre_filtered = np.concatenate((Pre_filtered,AnDA_sla_1.itrp_postAnDA),axis=0)
AnDA_sla_1.itrp_postAnDA = Post_process(Pre_filtered,len(VAR_sla.Obs_test),13)          
X_initialization = np.copy(VAR_sla.X_lr[:,r_start:r_start+r_length,c_start:c_start+c_length]+VAR_sla.dX_orig[:,r_start:r_start+r_length,c_start:c_start+c_length])
X_initialization[PR_sla.training_days:,:,:] = AnDA_sla_1.itrp_postAnDA 
X_lr = LR_perform(X_initialization,'',100)
AnDA_sla_1.itrp_postAnDA = X_lr[PR_sla.training_days:,:,:]

"""  Reload saved AnDA result  """
with open('./data/AMSRE/AnDA/AnDA_1.pickle', 'rb') as handle:
    AnDA_sla_1 = pickle.load(handle)    

"""  Radial Power Spetral  """
day = 85
resSLA = 0.25
f0, Pf_AnDA  = raPsd2dv1(AnDA_sla_1.itrp_AnDA[day,:,:],resSLA,True)
f1, Pf_postAnDA  = raPsd2dv1(AnDA_sla_1.itrp_postAnDA[day,:,:],resSLA,True)
f2, Pf_GT    = raPsd2dv1(AnDA_sla_1.GT[day,:,:],resSLA,True)
f3, Pf_OI    = raPsd2dv1(AnDA_sla_1.itrp_OI[day,:,:],resSLA,True)
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
colormap='nipy_spectral'
plt.figure()
plt.ion()
for day in range(13,len(AnDA_sla_1.GT)):
    plt.clf()
    gt = AnDA_sla_1.GT[day,:,:]
    obs = AnDA_sla_1.Obs[day,:,:]
    AnDA =  AnDA_sla_1.itrp_AnDA[day,:,:]
    OI = AnDA_sla_1.itrp_OI[day,:,:]
    Post_AnDA = AnDA_sla_1.itrp_postAnDA[day,:,:]
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
for day in range(50,len(VAR_sla.dX_GT_test)):
    plt.clf()
    gt = VAR_sla.dX_GT_test[day,:,:]   
    obs = VAR_sla.Obs_test[day,:,:]    
    itrp = VAR_sla.Optimal_itrp[day,:,:]   
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






