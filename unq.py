from __future__ import division # Sets division to be float division
import numpy as np

def unq(array,idx):  
    s = array.shape
    if not s:
        print('WARNING - array must be an array')
    if list(idx): # check this if conditional later
        q = array[idx]
        qshift = np.roll(q,-1,0)
        indices = (q!=qshift).nonzero()[0]
        if list(indices):
            ret = idx[indices]
        else:
            ret = np.hstack([len(q)-1])
    else:
        arrayshift = np.roll(array,-1)
        indices = (array!=arrayshift).nonzero()[0]
        if list(indices):
            ret = indices
        else:
            ret = np.hstack([len(array)-1])        
    return ret