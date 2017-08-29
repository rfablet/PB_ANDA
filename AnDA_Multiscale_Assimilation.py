import numpy as np
from pyflann import *
from sklearn.decomposition import PCA
from AnDA_analog_forecasting import AnDA_analog_forecasting as AnDA_AF
from AnDA_data_assimilation import AnDA_data_assimilation
from AnDA_variables import General_AF, AnDA_result
from AnDA_stat_functions import AnDA_RMSE, AnDA_correlate
from AnDA_transform_functions import sum_overlapping
import time
import multiprocessing
import pickle
import mkl


global VAR

def unwrap_self_f(arg, **kwarg):
    """
    Use function outside to unpack the self from the Multiscale_Assililation 
    for calling single_patch_assimilation
    """    
    return Multiscale_Assimilation.single_patch_assimilation(*arg,**kwarg)
    

class Multiscale_Assimilation:
    def __init__(self, _VAR, _PR, _AF):
        global VAR
        VAR = _VAR
        self.PR = _PR
        self.AF = _AF
    def single_patch_assimilation(self,coordinate):
        """  single patch assimilation """
        
        global VAR
        mkl.set_num_threads(1)
        # position of patch
        r = coordinate[0]
        c = coordinate[1]    
        # Specify Analog Forecasting Module
        AF = General_AF()
        AF.copy(self.AF)
        AF.lag = self.PR.lag
        # find sea_mask to remove land pixel
        sea_mask = VAR.dX_GT_test[0,r[0]:r[-1]+1,c[0]:c[-1]+1].flatten()
        sea_mask = np.where(~np.isnan(sea_mask))[0]    
            
        # use classic assmilation (without any condition) at border    
        if ((len(sea_mask)!=self.PR.patch_r*self.PR.patch_c) or self.PR.flag_scale): # bordering patches, reset to classic AF
            AF.flag_catalog = False

        # observation for this patch    
        obs_p = VAR.Obs_test[:,r[0]:r[-1]+1,c[0]:c[-1]+1]
        obs_p = obs_p.reshape(obs_p.shape[0],-1)
        obs_p_no_land = obs_p[:,sea_mask]   
        AF.obs_mask = obs_p_no_land
        if (AF.flag_catalog):
            try:
                # retrieving neighbor patchs
                k_patch = VAR.index_patch.keys()[VAR.index_patch.values().index([r[0],c[0]])]
                listpos = VAR.neighbor_patchs[k_patch] 
                listpos = sum(list(map((lambda x:range(x*self.PR.training_days,(x+1)*self.PR.training_days)),listpos)),[])      
                # remove last patch at each position out of retrieving analogs
                jet = len(listpos)/self.PR.training_days
            except ValueError:
                print "Cannot define position of patch!!!"
                quit()
        else:
            k_patch = 0
            jet = 1                  
        index_rem = []
        for i_lag in range(self.PR.lag):
            index_rem.append(np.arange(self.PR.training_days-i_lag-1,self.PR.training_days*(jet+1)-1,self.PR.training_days))
        AF.check_indices = np.array(index_rem).flatten()    
        # specify kind of AF depending on the position of patch (at border or not)  
        if ((len(sea_mask)==self.PR.patch_r*self.PR.patch_c) and self.PR.flag_scale): # not bordering patches
            if (AF.flag_catalog):
                AF.catalogs = VAR.dX_train[listpos,:]
            else:
                AF.catalogs = VAR.dX_train
            AF.coeff_dX = VAR.dX_eof_coeff
            AF.mu_dX = VAR.dX_eof_mu         
        else: # bordering patches
            #patch_dX = VAR.dX_orig[:self.PR.training_days,r[0]:r[-1]+1,c[0]:c[-1]+1]
            patch_dX = VAR.dX_orig[:,r[0]:r[-1]+1,c[0]:c[-1]+1]
            patch_dX = patch_dX.reshape(patch_dX.shape[0],-1)
            patch_dX_no_land = patch_dX[:,sea_mask]
            if (len(sea_mask)>=self.PR.n):
                AF.neighborhood = np.ones([self.PR.n,self.PR.n])
                pca = PCA(n_components=self.PR.n)               
            else:
                AF.neighborhood = np.ones([len(sea_mask),len(sea_mask)])
                pca = PCA(n_components=len(sea_mask))  
            AF.catalogs = pca.fit_transform(patch_dX_no_land)
            AF.coeff_dX = pca.components_.T
            AF.mu_dX = pca.mean_
            if (self.PR.flag_scale):
                AF.flag_model = False
        # Specify Observation
        class yo:
            time = np.arange(0,len(obs_p_no_land))
            values = obs_p_no_land-AF.mu_dX       
        # list of kdtree: 1 kdtree for all patch position in global analogs; each kdtree for each patch position in local analogs 
        list_kdtree = []
        if np.array_equal(AF.neighborhood, np.ones(AF.neighborhood.shape)):
            neigh = FLANN() 
            neigh.build_index(AF.catalogs[0:-self.PR.lag,:], algorithm="kdtree", target_precision=0.99,cores=1,sample_fraction=1,log_level = "info");
            list_kdtree.append(neigh)
        else:
            for i_var in range(self.PR.n):
                i_var_neighboor = np.where(AF.neighborhood[int(i_var),:]==1)[0]
                neigh = FLANN() 
                neigh.build_index(AF.catalogs[0:-self.PR.lag,i_var_neighboor], algorithm="kdtree", target_precision=0.99,cores=1,sample_fraction=1,log_level = "info");
                list_kdtree.append(neigh)   
        AF.list_kdtree = list_kdtree
        # Specify physical model as conditions for AF
        if (AF.flag_model):
            try:
                vx_test_p = VAR.gr_vl_test['vx'][:,r[0]:r[-1]+1,c[0]:c[-1]+1]
                vx_test_p = np.dot(vx_test_p.reshape(vx_test_p.shape[0],-1)-VAR.gr_vl_coeff['vx_eof_mu'],VAR.gr_vl_coeff['vx_eof_coeff'])
                vy_test_p = VAR.gr_vl_test['vy'][:,r[0]:r[-1]+1,c[0]:c[-1]+1]
                vy_test_p = np.dot(vy_test_p.reshape(vy_test_p.shape[0],-1)-VAR.gr_vl_coeff['vy_eof_mu'],VAR.gr_vl_coeff['vy_eof_coeff'])
                gx_test_p = VAR.gr_vl_test['gx'][:,r[0]:r[-1]+1,c[0]:c[-1]+1]
                gx_test_p = np.dot(gx_test_p.reshape(gx_test_p.shape[0],-1)-VAR.gr_vl_coeff['gx_eof_mu'],VAR.gr_vl_coeff['gx_eof_coeff'])
                gy_test_p = VAR.gr_vl_test['gy'][:,r[0]:r[-1]+1,c[0]:c[-1]+1]
                gy_test_p = np.dot(gy_test_p.reshape(gy_test_p.shape[0],-1)-VAR.gr_vl_coeff['gy_eof_mu'],VAR.gr_vl_coeff['gy_eof_coeff'])
                gr_vl_test_p = np.concatenate((vx_test_p[:,:], vy_test_p[:,:]),axis=1)
                if (AF.flag_catalog):
                    AF.cata_model_full = VAR.gr_vl_train[listpos,:] 
                else:
                    AF.cata_model_full = VAR.gr_vl_train 
            except ValueError:
                print "Cannot find physical model for AF !!!"
                quit()
        else:
            AF.cata_model_full = []
            gr_vl_test_p = None
        AF.x_model = gr_vl_test_p  
                
        # Specify dX condition for retrieving analogs
        if (AF.flag_cond):
            try:
                dX_cond_p = VAR.dX_cond[:,r[0]:r[-1]+1,c[0]:c[-1]+1]
                dX_cond_p = dX_cond_p.reshape(dX_cond_p.shape[0],-1)[:,sea_mask]  
            except ValueError:
                print "Cannot find dX condition for AF !!!"
                quit()
        else:
            dX_cond_p = None
        AF.x_cond = dX_cond_p      
        # Assimilation   
        class DA:
            method = 'AnEnKS' 
            N = 100
            xb = np.dot(VAR.dX_GT_test[0,r[0]:r[-1]+1,c[0]:c[-1]+1].flatten()[sea_mask]-AF.mu_dX,AF.coeff_dX)
            B = AF.B * np.eye(AF.coeff_dX.shape[1])
            H = AF.coeff_dX
            if (self.PR.flag_scale):
                R = AF.R
            else:
                R = AF.R * np.eye(len(sea_mask))   
                #R = AF.R
            @staticmethod
            def m(x,in_x): # x: query point at time t, in_x: index of condition at time t+lag
                return AnDA_AF(x, in_x, AF)        
        # AnDA results
        dX_interpolated = np.nan*np.zeros([len(yo.values),self.PR.patch_r, self.PR.patch_c]) 
        x_hat = AnDA_data_assimilation(yo, DA)
        x_hat = np.dot(x_hat.values,AF.coeff_dX.T)+ AF.mu_dX
        res_sst = np.nan*np.zeros(obs_p.shape)
        res_sst[:,sea_mask] = x_hat    
        res_sst = res_sst.reshape(len(yo.values),len(r),len(c))
        dX_interpolated[:,:res_sst.shape[1],:res_sst.shape[2]] = res_sst
        return dX_interpolated
 
    def multi_patches_assimilation(self, level, r_start, r_length, c_start, c_length):
        """ multi patches assimilation       
        level: 1 for series assimilation; >1 for parallel assimilation 
        """
        global VAR
        AnDA_result_test = AnDA_result()  
        AnDA_result_test.LR = VAR.X_lr[self.PR.training_days:,r_start:r_start+r_length,c_start:c_start+c_length]
        AnDA_result_test.GT = VAR.dX_GT_test[:,r_start:r_start+r_length,c_start:c_start+c_length] + AnDA_result_test.LR
        AnDA_result_test.Obs = VAR.Obs_test[:,r_start:r_start+r_length,c_start:c_start+c_length] + AnDA_result_test.LR
        AnDA_result_test.itrp_OI = VAR.Optimal_itrp[:,r_start:r_start+r_length,c_start:c_start+c_length] + AnDA_result_test.LR
        AnDA_result_test.corr_OI = AnDA_correlate(AnDA_result_test.itrp_OI-AnDA_result_test.LR,AnDA_result_test.GT-AnDA_result_test.LR)
        AnDA_result_test.itrp_AnDA = np.nan*np.zeros((len(VAR.Obs_test),r_length,c_length)) 
        ###########
        mask_sample = VAR.dX_GT_test[0,:,:]  
        r_sub = np.arange(r_start,r_start+self.PR.patch_r)
        c_sub = np.arange(c_start,c_start+self.PR.patch_c)
        ind = 0
        # Choosing 5 as overlapping width
        all_patches = []
        while (len(r_sub)>5):
            while (len(c_sub)>5): 
                if (np.sum(~np.isnan(mask_sample[np.ix_(r_sub,c_sub)]))>0):
                    all_patches.append([r_sub,c_sub])
                    ind = ind+1
                c_sub = c_sub+self.PR.patch_c-5
                c_sub = c_sub[c_sub<c_length+c_start]        
            r_sub = r_sub+self.PR.patch_r-5
            r_sub = r_sub[r_sub<r_length+r_start]
            c_sub  = np.arange(c_start,c_start+self.PR.patch_c)    
        start_time = time.time()
        print("---Processing  %s patches ---" % (ind))   
        pool = multiprocessing.Pool(level)
        result_tmp = pool.map(unwrap_self_f,zip([self]*len(all_patches),all_patches))
        pool.close()
        pool.join()
        print("---Processing time:  %s seconds ---" % (time.time() - start_time))    
        result_tmp = np.array(result_tmp)
        
        r_sub = np.arange(r_start,r_start+self.PR.patch_r)
        c_sub = np.arange(c_start,c_start+self.PR.patch_c)
        ind = 0
        # Choosing 5 as overlapping width
        while (len(r_sub)>5):
            while (len(c_sub)>5): 
                if (np.sum(~np.isnan(mask_sample[np.ix_(r_sub,c_sub)]))>0):
                    itrp_field = result_tmp[ind,:,:len(r_sub),:len(c_sub)]
                    for u in range(0,len(VAR.Obs_test)):
                        tmp1 = itrp_field[u,:,:]
                        tmp2 = AnDA_result_test.itrp_AnDA[u,(r_sub[0]-r_start):(r_sub[-1]+1-r_start),(c_sub[0]-c_start):(c_sub[-1]+1-c_start)]
                        AnDA_result_test.itrp_AnDA[u,(r_sub[0]-r_start):(r_sub[-1]+1-r_start),(c_sub[0]-c_start):(c_sub[-1]+1-c_start)] = sum_overlapping(tmp1,tmp2)
                    ind = ind+1
                c_sub = c_sub+self.PR.patch_c-5
                c_sub = c_sub[c_sub<c_length+c_start]        
            r_sub = r_sub+self.PR.patch_r-5
            r_sub = r_sub[r_sub<r_length+r_start]
            c_sub  = np.arange(c_start,c_start+self.PR.patch_c) 
            
        AnDA_result_test.corr_AnDA = AnDA_correlate(AnDA_result_test.itrp_AnDA,AnDA_result_test.GT-AnDA_result_test.LR)    
        AnDA_result_test.itrp_AnDA = AnDA_result_test.itrp_AnDA + AnDA_result_test.LR
        AnDA_result_test.rmse_AnDA = AnDA_RMSE(AnDA_result_test.itrp_AnDA,AnDA_result_test.GT)        
        AnDA_result_test.rmse_OI = AnDA_RMSE(AnDA_result_test.itrp_OI,AnDA_result_test.GT)

        return AnDA_result_test
