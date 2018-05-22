from __future__ import division # Sets division to be float division
import numpy as np

def rsqd(w,h):   
    # produce a parabolic mask
    s = np.zeros((w,h))
    xc = (w-1)/2
    yc = (h-1)/2
    x = np.arange(-xc,xc+1)
    x = x**2
    y = np.arange(-yc,yc+1)
    y = y**2
    for j in xrange(0,h):
        s[:,j] = x.T+y[j]     
    return s
    
def thetarr(w):    
    # produce a theta mask 
    theta = np.zeros((w,w))
    xc = (w-1)/2
    yc = (w-1)/2
    x = np.arange(-xc,xc+1)
    y = np.arange(-yc,yc+1)
    for j in xrange(0,w):
        theta[:,j] = np.arctan2(x,y[j])
    return theta