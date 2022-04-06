'''
Tools and utility functions for neural manifold learning analysis
Paper: "Neural manifold analysis of brain circuit dynamics in neurological disorders"
Author: Giuseppe P Gava, 01/03/2022
'''

import numpy as np


### Utility functions
def smooth(x, window_len=11, window='hanning'):
    """
    Smooth the data using a window with requested size.
    
    Inputs:
        x: Input signal 
        window_len: Dimension of the smoothing window; should be an odd integer
        window: Window type from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    Outputs:
       y: Smoothed signal
    """

    if window_len<3:
        return x

    s = np.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]

    if window=='flat': #moving average
        w = np.ones(window_len,'d')
    else:
        w = eval('np.'+window+'(window_len)')

    y = np.convolve(w/w.sum(),s,mode='valid')
    return y

def centre_scale(x):
    xm = x - np.mean(x)
    return xm/np.max(xm)


### Manifold learning algorithms codes
def cmdscale(D):
    """                                                                                       
    Classical multidimensional scaling (MDS) -- based on the MATLAB cmdscale function                                                
    Inputs:                                                                             
        D: Symmetric distance matrix.                                                            

    Outputs:                                                                                  
    Y : Configuration matrix. Each column represents a dimension. Only the                    
        p dimensions corresponding to positive eigenvalues of B are returned.                 
        Note that each dimension is only determined up to an overall sign,                    
        corresponding to a reflection.                                                        

    evals : Eigenvalues of B.                                                                     
    """
    # Number of points                                                                        
    n = len(D)
    # Centering matrix                                                                        
    H = np.eye(n) - np.ones((n, n))/n
    # YY^T                                                                                    
    B = -H.dot(D**2).dot(H)/2
    # Diagonalize                                                                             
    evals, evecs = np.linalg.eigh(B)
    # Sort by eigenvalue in descending order                                                  
    idx = np.argsort(evals)[::-1]
    evals = evals[idx]
    evecs = evecs[:,idx]
    # Compute the coordinates using positive-eigenvalued components only                      
    w, = np.where(evals>0)
    L = np.diag(np.sqrt(evals[w]))
    V = evecs[:,w]
    Y = V.dot(L)

    return Y, evals


### Linear decoder

def OLE(X, Y):
    # Optimal linear estimator - obtain the estimator's weights
    X = np.c_[np.ones((X.shape[0], 1)), X]
    # obtain the estimator
    return np.dot(np.linalg.inv(np.dot(X.T, X)), np.dot(X.T, Y))

def LinRec(f, X, Y):
    # Obtain the prediction on the test data
    X = np.c_[np.ones((X.shape[0], 1)), X]  # append a row of ones to the input
    Rec = np.dot(X, f)
    # Evaluate the prediction
    RMSE = np.mean(np.sqrt((Y - Rec)**2))
    R = np.corrcoef(Y, Rec)[0, 1]
    return Rec, RMSE, R


### Reconstruction score

def new_LLE_pts(M, M_LLE, k, x, lam=1):
    # Adapted from https://github.com/jakevdp/pyLLE/blob/master/pyLLE/python_only/LLE.py
    """
    inputs:
       - M: a rank [d * N] data-matrix
       - M_LLE: a rank [m * N] matrixwhich is the output of LLE(M,k,m)
       - k: the number of neighbors used to produce M_LLE
       - x: a length d data vector OR a rank [d * Nx] array
    outputs:
       - y: the LLE reconstruction of x
    """
    # make sure inputs are correct
    d,N = M.shape
    m,N2 = M_LLE.shape
    assert N==N2
    Nx = x.shape[1]
    W = np.zeros([Nx,N])

    for i in range(x.shape[1]):
        #  find k nearest neighbors
        M_xi = M-np.atleast_2d(x[:,i]).T
        vec = (M_xi**2).sum(0)
        nbrs = np.argsort(vec)[1:k+1]
        
        #compute covariance matrix of distances
        M_xi = M_xi[:,nbrs]
        Q = M_xi.T @ M_xi

        #singular values of x give the variance:
        # use this to compute intrinsic dimensionality
        sig2 = (np.linalg.svd(M_xi,compute_uv=0))**2
    
        #Covariance matrix may be nearly singular:
        # add a diagonal correction to prevent numerical errors
        Q += (lam) * np.identity(Q.shape[0])

        #solve for weight
        w = np.linalg.solve(Q,np.ones((Q.shape[0],1)))[:,0]
        w /= np.sum(w)

        W[i,nbrs] = w
    
    return np.array( M_LLE  * np.matrix(W).T ).T 


### Intrinsic dimensionality

def intrinsic_dimensionality(points, nstep=10, metric='euclidean', dist_mat=None, ds=1,\
                              plot=1, verbose=0, offset_min=10, win_smooth=7, fit='std',\
                              thr_start=10, thr_fi=1e5):
    '''
    Obtain the intrinsic dimensionality of a point cloud / neural population activity
    by using exponent (slope on log-log plot) ofcumulative neighbours (NN) distrubution
    to estimate intrinsic dimensionality

    Inputs:
         points: NxT point cloud
         n_step: n of distance step to evaluate
         metric: which metric to use to obtain the distance matrix
         dist_mat: if 'precomputed', consider `points` as a distance matrix
         ds: downsampling factor
         plot: visualise curve (boolean)
         verbose: print feedback on code progression
         fit: how to estimate dimewnsionality
           - 'std': use `thr_start` and `thr_fi` to select NN range where to fit exponential
           - 'all': use the full NN range
           - 'diff': find the linear part of the curve (using 2nd diff) and fit line to it
                       (start at offset min at least; `win_smooth` is smoothing parameter to estimate
                        min and max of the 2nd diff)
                        
     Outputs:
         Nneigh: number of neighbours found per point at every radii
         radii: radii steps used
         p: linear fit in log-log axes
         
    Faster than using KD tree
    '''

    from sklearn.metrics.pairwise import euclidean_distances, cosine_distances
    
    Nneigh = np.zeros((points.shape[0],nstep))
    if dist_mat=='precomputed':
        dist_mat = points
    else: # compute distance matrix
        # Find diameter of point cloud
        points[np.isnan(points)] = 0
        if metric=='euclidean': dist_mat = euclidean_distances(points)
        elif metric=='cosine': dist_mat = cosine_distances(points)
    dist_mat[dist_mat==0] = np.nan
    minD = np.nanmin(dist_mat)
    maxD = np.nanmax(dist_mat)
    
    # Define distances to evaluate
    radii = np.logspace(np.log10(minD), np.log10(maxD), nstep)
    # Fing #neigh vs dist
    for n,rad in enumerate(radii):
        Nneigh[:,n] = np.sum((dist_mat<rad), 1)
        if verbose: print(f'{n+1}/{len(radii)}')
    
    # find slope of neighbors increase = dimensionality
    sem = np.std(Nneigh,0)
    mean_ = np.mean(Nneigh,0)
    sem_ = np.log10(sem[1:])
    x2p = radii[1:]; y2p = mean_[1:]
    x2p = x2p / np.max(x2p) # normalise the distance radii
    
    # find indeces where curve is linear for fit
    if fit=='std': # fit line from #thr_start NN until #thre_fi NN
        start = np.argmin(np.abs(mean_ - thr_start))
        fi = np.argmin(np.abs(mean_ - thr_fi))
    elif fit=='diff': # find linear part of the curve using 2nd diff
        diff2nd = np.diff(np.diff(smooth(np.log10(y2p), window_len=win_smooth)))
        fi = np.argmin(diff2nd[offset_min:])+offset_min
        start = np.argmax(diff2nd[:fi])
    elif fit=='all': # use all data
        start = 0
        fi = len(mean_)-1
        
    # line fit
    x2fit = x2p[start:fi]
    y2fit = y2p[start:fi]
    p = np.polyfit(np.log10(x2fit), np.log10(y2fit), deg=1)
    
    # plot
    if plot:
        import matplotlib.pyplot as plt
        import seaborn as sns
        plt.plot(x2p, y2p) # og data
        y_mod = 10**p[1] * np.power(x2fit, p[0]) # best fit power law
        plt.plot(x2fit, y_mod, 'r')
        plt.xlabel('Distance')
        plt.ylabel('# neighbours')
        plt.xscale('log')
        plt.yscale('log')
        plt.title('The dimensionality/slope is %.2f'%p[0])
        sns.despine()

    return Nneigh, radii, p