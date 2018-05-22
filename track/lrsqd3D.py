from __future__ import division # Sets division to be float division
import numpy as np
def lrsqd3D(extent,yratio=1,zratio=1):
    if len(extent)==1:
        ext = np.zeros(3)+extent
    else:
        ext = extent
    x=ext[0]
    y=ext[1]
    z=ext[2]
    
    r2 = np.zeros((x,y,z))
    xc = np.double(x-1)/2
    yc = np.double(y-1)/2
    zc = np.double(z-1)/2
    
    yi = np.zeros(x)+1
    xi = np.zeros(y)+1
    
    xa = np.arange(-xc,x-xc)
    xa = xa**2
    ya = np.arange(-yc,y-yc)/yratio
    ya = ya**2
    za = np.arange(-zc,z-zc)/zratio
    za = za**2
    
    xi = xi.reshape(len(xi),1)
    ya = ya.reshape(len(ya),1)
    for k in xrange(0,int(z)):
        r2[:,:,k] = (xi*xa) + (ya*yi) + za[k]
    
    return r2
        
    
    
    
        