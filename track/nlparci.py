from __future__ import division # Sets division to be float division
import numpy as np
from scipy.linalg import qr, lstsq, norm
from scipy.stats import t
import pdb

def nlparci(beta,resid,J,alpha=0.05):
    n = len(resid)
    p = len(beta)
    v = n-p
    
    # Approximation when a column is zero vector
    temp = (np.max(abs(J),0)==0).nonzero()[0]
    if len(temp)!=0:
        J[:,temp] = np.sqrt(np.spacing(1).astype(J.dtype))
        
    # Calculate covariance matrix
    R = qr(J,0)[1]
    R = R[0:7,:]
    Rinv = lstsq(R,np.identity(7))[0]
    diag_info = np.sum(Rinv*Rinv,1)
    rmse = norm(resid) / np.sqrt(v)
    se = np.sqrt(diag_info) * rmse
    
    # Calculate confidence interval
    delta = se*t.ppf(1-alpha/2,v)
    col1 = (beta-delta).reshape(7,1)
    col2 = (beta+delta).reshape(7,1)
    return np.hstack([col1,col2])
    