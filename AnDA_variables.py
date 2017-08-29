import numpy as np

###### Parameters setting for SST ###########################
class PR:
    flag_scale = []  # True: multi scale
                        # False: one scale
    n = [] # dimension state
    patch_r = [] # size of patch
    patch_c = [] # size of patch
    training_days = [] # num of training images: 2008-2014 
    test_days = [] # num of test images: 2015
    lag = [] # lag of time series: t -> t+lag
    G_PCA = [] # N_eof for global PCA
    # Input dataset
    path_X = [] # dir of dataset X ( SST or SLA )
    path_OI = [] # dir of OI product of test years 
    path_mask = [] # dir of observation mask
    # Dataset automatically created during execution
    path_X_lr = [] # dir to store LR product of X, created by PCA
    path_dX_PCA = [] # dir to store PCA transformation of dX
    path_index_patches = [] # dir to store all position of each patch over image
    path_neighbor_patches = [] # dir to store position of each path's neighbors

class General_AF:
    def __init__(self):
        self.flag_reduced = [] # True: Reduced version of Local Linear AF
        self.flag_cond = [] # True: use Obs at t+lag as condition to select successors
                          # False: no condition in analog forecasting
        self.flag_model = [] # True: Use gradient, velocity as additional regressors in AF
        self.flag_catalog = [] # True: each catalog for each patch position
                            # False: only one catalog for all positions
        self.lag = [] # equal to PR.lag
        self.cluster = []       # clusterized version AF
        self.k = [] # number of analogs
        self.k_initial = [] # retrieving k_initial nearest neighbors, then using condition to retrieve k analogs 
        self.neighborhood = [] # global analogs
    #        AF.neighborhood = np.eye(PR.n)+np.diag(np.ones(PR.n-1),1)+ np.diag(np.ones(PR.n-1),-1)+ \
    #                   np.diag(np.ones(PR.n-2),2)+np.diag(np.ones(PR.n-2),-2)
    #        AF.neighborhood[0:2,:5] = 1 
    #        AF.neighborhood[PR.n-2:,PR.n-5:] = 1 # local analogs
        self.catalogs = []
        self.regression = []
        self.sampling = []
        self.list_kdtree = [] # store kdtree, nearest neighbor searcher
        self.B = [] # variance of initial state error 
        self.R = [] # variance of observation error
    
        self.check_indices = [] 
        self.x_cond = [] # conditional state to retrieve analogs
        self.coeff_dX = [] # EOF space of state
        self.mu_dX = [] # EOF mean vector
        self.x_model = [] # physical model : velocity & gradient
        self.obs_mask = [] # mask of observation
        self.cata_model_full = [] # training set of physical model

    def copy(self,AF):
        self.flag_reduced = AF.flag_reduced
        self.flag_cond = AF.flag_cond
        self.flag_model = AF.flag_model
        self.flag_catalog = AF.flag_catalog
        self.lag = AF.lag
        self.cluster = AF.cluster     
        self.k = AF.k
        self.k_initial = AF.k_initial
        self.neighborhood = AF.neighborhood
        self.catalogs = AF.catalogs
        self.regression = AF.regression
        self.sampling = AF.sampling
        self.list_kdtree = AF.list_kdtree
        self.B = AF.B
        self.R = AF.R
        self.check_indices = AF.check_indices
        self.x_cond = AF.x_cond
        self.x_model = AF.x_model
        self.obs_mask = AF.obs_mask
        self.cata_model_full = AF.cata_model_full
        self.coeff_dX = AF.coeff_dX
        self.mu_dX = AF.mu_dX

        
###### Parameters setting ############################

###### Datasets Definition ###########################
class VAR:
    X_lr = []
    dX_orig = []
    Optimal_itrp = []    
    dX_train = [] # training catalogs  for dX in EOF space
    dX_eof_coeff = [] # EOF base vector
    dX_eof_mu = [] # EOF mean vector    
    dX_GT_test = [] # dX GT in test year
    Obs_test = [] # Observation in test year, by applying mask to dX GT    
    dX_cond = [] # condition used for AF
    gr_vl_train = [] # gradient, velocity used as physical condition
    gr_vl_test = {}  
    gr_vl_coeff = {}        
    index_patch = [] # store order of every image patch: 0, 1,..total_patchs
    neighbor_patchs = [] # store order of neighbors of every image patch
   
class AnDA_result:
    itrp_AnDA = []
    itrp_OI = []
    itrp_postAnDA = []
    GT = []
    Obs = []
    LR = []
    # stats: rmse & correlation 
    rmse_AnDA = []
    corr_AnDA = []
    rmse_OI = []
    corr_OI = []
    rmse_postAnDA = []
    corr_postAnDA = []


###### Datasets Definition ###########################   
