from __future__ import division # Sets division to be float division
import numpy as np
import scipy.signal as sps

def bpass(img,Lambda,w):
    #      Written 7-23-03 by Maria Kilfoil
    #      Adapted for Python in June 2013 by Kevin Smith (Maria Kilfoil's Group)
    #
    # 	   Implements a real-space bandpass filter to suppress pixel noise and 
    #      slow-scale image variations while retaining information of a characteristic size.
    # 	   *Works with anisotropic 3d cube data*
    # 
    #  CALLING SEQUENCE:
    # 		res = bpass(img, lambda, w)
    #  INPUTS:
    # 		img:	two-dimensional array to be filtered.
    # 		Lambda: characteristic lengthscale of noise in pixels. Additive noise averaged 
    #                   over this length should vanish. May assume any positive floating value.
    # 			Make it a 3-vector if aspect ratio is not 1:1:1.
    # 		w: A length in pixels somewhat larger than *half* a typical object. Must be an odd valued 
    #                   integer. Make it a 3-vector if aspect ratio is not 1:1:1.
    #  OUTPUTS:
    # 		res:	filtered image.
    #  PROCEDURE:
    # 		simple 'mexican hat' wavelet convolution yields spatial bandpass filtering.
    #  NOTES:
    #		based on "Methods of digital video microscopy for colloidal studies", John Crocker 
    #       and David Grier, J. Colloid Interface Sci. 179, 298 (1996), and on bpass.pro IDL code 
    #       written by John Crocker and David Grier. 
    #

    
    a = np.double(img)
    b = np.double(Lambda)
    w = int(np.around(np.maximum(w,2*b)))
    N = int(2*w+1)
    r = np.arange(-w,w+1)/(2*b)
    xpt = np.exp(-r**2)
    B = (xpt.sum())**2
    xpt = xpt/(xpt.sum())
    factor = (((xpt**2).sum())**2-1/(N**2))*B
    # note: N not N^2 etc since doing 2D conv along each axis separately
    gx=np.zeros((1,len(xpt)))
    gx[0,:] = xpt
    gy = np.reshape(gx[0],(len(gx[0]),1))
    bx = np.zeros([1,N])-1/N
    by = np.reshape(bx[0],(len(bx[0]),1))
    g = sps.convolve2d(a,gx,'valid')
    g = sps.convolve2d(g,gy,'valid')
    b = sps.convolve2d(a,bx,'valid')
    b = sps.convolve2d(b,by,'valid')
    res = g-b
    s = np.maximum(res/factor,0)
    tmp=np.zeros([len(a[:,1]),len(a[1,:])])
    tmp[w:(len(s[:,1])+w),w:(len(s[1,:])+w)]=s;
    s=tmp
    return s
    