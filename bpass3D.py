from __future__ import division # Sets division to be float division
import numpy as np
import scipy.signal as sps
def bpass3D(image, lnoise, lobject, inputv):
    # bpass3d is written by Yongxiang Gao and Maria Kilfoil, based on
    # the IDL code written by John C. Crocker and David G. Grier.
    # the input variable supposed to be contained in varargin is [noclip nopad]
    # inputv is to indicate whether there is input for noclip or nopad by
    # logical number 1 and 0.
    # Translated to Python in June 2013 by Kevin Smith (Maria Kilfoil's Group)
    image = np.double(image)
    lnoise = np.array(lnoise)
    lobject = np.array(lobject)
    try:
        nn = len(lnoise)
    except:
        lnoise = np.array([0]*1)+lnoise
        nn = 1       
    try:
        no = len(lobject)
    except:
        lobject = np.array([0]*1)+lobject
        no = 1
    if ((nn > 1) and (no==1)) or ((nn==1) and (no>1)):
        print('Both length parameters must be scalars of 3-vectors')
        return
    # do x direction masks
    bb = np.double(lnoise[0])
    w = np.round(np.max([lobject[0],2*bb]))
    N = 2*w +1
    r = np.double(np.arange(-w,w+1)/(2*bb))
    gx = np.exp(-r**2)
    gx = gx / np.sum(gx)
    bx = np.zeros(N,np.double) - 1/N
    factor = np.sum(gx**2) - np.fix(1/N)
    if nn==1: # ks untested
        gy = gx
        gz = gx
        by = bx
        bz = bx
    else:
        # do y direction masks
        bb = np.double(lnoise[1])
        w = np.round(np.max([lobject[1],2*bb]))
        N = 2*w +1
        r = np.double(np.arange(-w,w+1)/(2*bb))
        gy = np.exp(-r**2)
        gy = gy / np.sum(gy)
        by = np.zeros(N,np.double)-1/N
        # do z direction masks
        bb = np.double(lnoise[2])
        w = np.round(np.max([lobject[2],2*bb]))
        N = 2*w +1
        r = np.double(np.arange(-w,w+1)/(2*bb))
        gz = np.exp(-r**2)
        gz = gz / np.sum(gz)
        bz = np.zeros(N,np.double) - 1/N
    
    nx,ny,nf = image.shape
    if (N>=nf) and (inputv[1]==1):
        # stack is too thin for any data to survive!
        print('WARNING: data cube thinner than convolution kernel!')
        print('Disabling nopad keyword to compensate')
        pad = 1
    
    if not (inputv[1]==1):
        if len(lobject)==1:
            padxw = np.round(np.max([lobject[0],2 * bb]))
            padyw = padxw
            padzw = padxw
        else:
            padxw = np.round(np.max([lobject[0],2 * bb]))
            padyw = np.round(np.max([lobject[1],2 * bb]))
            padzw = np.round(np.max([lobject[2],2 * bb]))
    # pad out the array with average values of corresponding frames
        ave = np.zeros(nf,np.double)
        for i in xrange(0,nf):
            ave[i] = np.sum(image[:,:,i])/(nx*ny)
        g = np.zeros((nx+2*padxw, ny+(2*padyw), nf+(2*padzw)),np.double)
        g[:,:,0:padzw] = ave[0]
        g[:,:,nf-padzw:] = ave[nf-1]
        for i in xrange(0,nf):
            g[:,:,padzw+i] = ave[i]
        g[padxw:padxw+nx,padyw:padyw+ny,padzw:padzw+nf] = np.double(image)
        nx = nx + (2*padxw)
        ny = ny + (2*padyw)
        nf = nf + (2*padzw)
    else:
        g = image
        padxw = 0
        padyw = 0
        padzw = 0
    
    del image
    b = tuple(g)
    b = np.array(g)
    gx = np.reshape(gx,(1,len(gx)))
    gy = np.reshape(gy,(len(gy),1))
    bx = np.reshape(bx,(1,len(bx)))
    by = np.reshape(by,(len(by),1))
    gz = np.reshape(gz,(len(gz),1))
    bz = np.reshape(bz,(len(bz),1))
    # do x and y convolutions
    for i in xrange(int(padzw),int(nf-padzw)):
        g[:,:,i] = sps.convolve2d(g[:,:,i],gx,'same')
        g[:,:,i] = sps.convolve2d(g[:,:,i],gy,'same')
        b[:,:,i] = sps.convolve2d(b[:,:,i],bx,'same')
        b[:,:,i] = sps.convolve2d(b[:,:,i],by,'same')
    temp = g.transpose((2,0,1))
    
    d = np.zeros((temp.shape[0],temp.shape[1],int(ny-padyw)))
    for i in xrange(int(padyw),int(ny-padyw)):
        d[:,:,i] = sps.convolve2d(temp[:,:,i],gz,'same')
    del temp
    g = d.transpose((1,2,0))
    del d
    temp2 = b.transpose((2,0,1))
    del b
    
    f = np.zeros((temp2.shape[0],temp2.shape[1],int(ny-padyw)))
    for i in xrange(int(padyw),int(ny-padyw)):
        f[:,:,i] = sps.convolve2d(temp2[:,:,i],bz,'same')
    del temp2
    b = f.transpose((1,2,0))
    del f
    
    if not (inputv[1]==1):
        g = g[padxw:nx-padxw,padyw:ny-padyw,padzw:nf-padzw]
        b = b[padxw:nx-padxw,padyw:ny-padyw,padzw:nf-padzw]
    g = g+b
    del b
    
    if inputv[0]==1:
        res = g/factor
    else:
        res = np.maximum([g/factor],0)[0]
        
    return res
    
    
    
        
        
        
        
        
        
    