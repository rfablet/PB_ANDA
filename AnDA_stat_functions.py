#!/usr/bin/env python

""" AnDA_stat_functions.py: Collection of statistical functions used in AnDA. """

__author__ = "Phi Huynh Viet"
__version__ = "2.0"
__date__ = "2017-08-01"
__email__ = "phi.huynhviet@telecom-bretagne.eu"

import numpy as np
    
def AnDA_RMSE(a,b):
    """ Compute the Root Mean Square Error between 2 n-dimensional vectors. """
    if (a.ndim==1):
        a = a[np.newaxis]
    if (a.ndim>2):
        a = a.reshape(a.shape[0],-1)
    if (b.ndim==1):
        b = b[np.newaxis]    
    if (b.ndim>2):
        b = b.reshape(b.shape[0],-1)
    return np.sqrt(np.nanmean((a-b)**2,1))
    
    
def AnDA_correlate(a,b):
    """ Compute the Correlation between 2 n-dimensional vectors. """
    if (a.ndim==1):
        a = a[np.newaxis]
    if (a.ndim>2):
        a = a.reshape(a.shape[0],-1)
    if (b.ndim==1):
        b = b[np.newaxis] 
    if (b.ndim>2):
        b = b.reshape(b.shape[0],-1)
    a = a - np.nanmean(a,1)[np.newaxis].T
    b = b - np.nanmean(b,1)[np.newaxis].T
    r = np.nansum((a*b),1) / np.sqrt(np.nansum((a*a),1) * np.nansum((b*b),1))
    return r  
    

def normalise(M):
    """ Normalize the entries of a multidimensional array sum to 1. """
    c = np.sum(M)
    # Set any zeros to one before dividing
    d = c + 1*(c==0)
    M = M/d
    return M

def mk_stochastic(T):
    """ Ensure the matrix is stochastic, i.e., the sum over the last dimension is 1. """

    if len(T.shape) == 1:
        T = normalise(T)
    else:
        n = len(T.shape)
        # Copy the normaliser plane for each i.
        normaliser = np.sum(T,n-1);
        normaliser = np.dstack([normaliser]*T.shape[n-1])[0]
        # Set zeros to 1 before dividing
        # This is valid since normaliser(i) = 0 iff T(i) = 0

        normaliser = normaliser + 1*(normaliser==0)
        T = T/normaliser.astype(float)
    return T

def sample_discrete(prob, r, c):
    """ Sampling from a non-uniform distribution. """

    cumprob = np.cumsum(prob)
    n = len(cumprob)
    R = np.random.rand(r,c)
    M = np.zeros([r,c])
    for i in range(0,n-1):
        M = M+1*(R>cumprob[i])    
    return int(M)

def resampleMultinomial(w):
    """ Multinomial resampler. """

    M = np.max(w.shape);
    Q = np.cumsum(w,0);
    Q[M-1] = 1; # Just in case...
    i = 0;
    indx = [];
    while (i<=(M-1)):
        sampl = np.random.rand(1,1);
        j = 0;
        while (Q[j]<sampl):
            j = j+1;
        indx.append(j);
        i = i+1
    return indx

def inv_using_SVD(Mat, eigvalMax):
    """ SVD decomposition of Matrix. """
    
    U,S,V = np.linalg.svd(Mat, full_matrices=True);
    eigval = np.cumsum(S)/np.sum(S);
    # search the optimal number of eigen values
    i_cut_tmp = np.where(eigval>=eigvalMax)[0];
    S = np.diag(S);
    V = V.T;
    i_cut = np.min(i_cut_tmp)+1
    U_1 = U[0:i_cut,0:i_cut]
    U_2 = U[0:i_cut,i_cut:]
    U_3 = U[i_cut:,0:i_cut]
    U_4 = U[i_cut:,i_cut:]
    S_1 = S[0:i_cut,0:i_cut]
    S_2 = S[0:i_cut,i_cut:]
    S_3 = S[i_cut:,0:i_cut]
    S_4 = S[i_cut:,i_cut:]
    V_1 = V[0:i_cut,0:i_cut]
    V_2 = V[0:i_cut,i_cut:]
    V_3 = V[i_cut:,0:i_cut]
    V_4 = V[i_cut:,i_cut:]
    tmp1 = np.dot(np.dot(V_1,np.linalg.inv(S_1)),U_1.T);
    tmp2 = np.dot(np.dot(V_1,np.linalg.inv(S_1)),U_3.T);
    tmp3 = np.dot(np.dot(V_3,np.linalg.inv(S_1)),U_1.T);
    tmp4 = np.dot(np.dot(V_3,np.linalg.inv(S_1)),U_3.T);
    inv_Mat = np.concatenate((np.concatenate((tmp1,tmp2),axis=1),np.concatenate((tmp3,tmp4),axis=1)),axis=0);
    tmp1 = np.dot(np.dot(U_1,S_1),V_1.T);
    tmp2 = np.dot(np.dot(U_1,S_1),V_3.T);
    tmp3 = np.dot(np.dot(U_3,S_1),V_1.T);
    tmp4 = np.dot(np.dot(U_3,S_1),V_3.T);
    hat_Mat = np.concatenate((np.concatenate((tmp1,tmp2),axis=1),np.concatenate((tmp3,tmp4),axis=1)),axis=0);
    det_inv_Mat = np.prod(np.diag(S[0:i_cut,0:i_cut]));   
    return inv_Mat;
 
def inv_using_Woodbury(ainv,u,cinv,v,rinv):
    """ inv using Woodbury equation """
    
    tmp = inv_using_SVD(cinv+np.dot(v,u)*rinv,0.9999)
    tmp_inv = -rinv*rinv*np.dot(np.dot(u,tmp),v)
    np.fill_diagonal(tmp_inv,tmp_inv.diagonal()+ainv)
    #return ainv-rinv*rinv*np.dot(np.dot(u,tmp),v)
    return tmp_inv
    
def hanning2d(M, N):
    """
    A 2D hanning window, as per IDL's hanning function.  See numpy.hanning for the 1d description
    """
    
    if N <= 1:
        return np.hanning(M)
    elif M <= 1:
        return np.hanning(N) # scalar unity; don't window if dims are too small
    else:
        return np.outer(np.hanning(M),np.hanning(N))

def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(phi, rho)
    
def raPsd2dv1(img,res,hanning):
    """ Computes and plots radially averaged power spectral density (power
     spectrum) of image IMG with spatial resolution RES.
    """
    
    img = img.copy()
    N, M = img.shape
    if hanning:
        img = hanning2d(*img.shape) * img        
    imgf = np.fft.fftshift(np.fft.fft2(img))
    imgfp = np.power(np.abs(imgf)/(N*M),2)    
    # Adjust PSD size
    dimDiff = np.abs(N-M)
    dimMax = max(N,M)
    if (N>M):
        if ((dimDiff%2)==0):
            imgfp = np.pad(imgfp,((0,0),(dimDiff/2,dimDiff/2)),'constant',constant_values=np.nan)
        else:
            imgfp = np.pad(imgfp,((0,0),(dimDiff/2,1+dimDiff/2)),'constant',constant_values=np.nan)
            
    elif (N<M):
        if ((dimDiff%2)==0):
            imgfp = np.pad(imgfp,((dimDiff/2,dimDiff/2),(0,0)),'constant',constant_values=np.nan)
        else:
            imgfp = np.pad(imgfp,((dimDiff/2,1+dimDiff/2),(0,0)),'constant',constant_values=np.nan)
    halfDim = int(np.ceil(dimMax/2.))
    X, Y = np.meshgrid(np.arange(-dimMax/2.,dimMax/2.-1+0.00001),np.arange(-dimMax/2.,dimMax/2.-1+0.00001))           
    theta, rho = cart2pol(X, Y)                                              
    rho = np.round(rho+0.5)   
    Pf = np.zeros(halfDim)
    f1 = np.zeros(halfDim)
    for r in range(halfDim):
      Pf[r] = np.nansum(imgfp[rho == (r+1)])
      f1[r] = float(r+1)/dimMax
    f1 = f1/res
    return f1, Pf


 
