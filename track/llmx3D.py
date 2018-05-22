from __future__ import division # Sets division to be float division
import numpy as np
import lrsqd3D

def llmx3D(a, sep, pad):
    # a 3d-happy version of DGGs 'local max', which does NOT use
    # the dilation algorithm.  NB: the data MUST be padded, or it
    # can crash!! Best of all, 'sep' is a float 3-vector. 'sep' is
    # the actual minimum distance between two maxima (i.e. the bead
    # diameter or so.  If the image is 'padded', tell the code the
    # padding size to save it a little work.
    # Translated by Yongxiang Gao in 2005 from IDL code developped by John
    # Crocker and David Grier
    # Translated by Kevin Smith (of Maria Kilfoil's Group) in June 2013 from Matlab
    
    sep = np.array(sep)
    pad = np.array(pad)
    allmin = np.min(a)
    allmax = np.max(a)
    a[0,0,0] = allmin
    a = np.fix(((255.0+1)*np.single(a-allmin)-1)/np.single(allmax-allmin))
    allmin = a[0,0,0]
    nx, ny, nz = a.shape
    bignum = nx*ny*nz
    # the diameter of the local max algorithm is 2*sep
    # extent extent is the next biggest odd integer
    extent = np.fix(sep*2) +1 # i.e. mask is diameter 2*sep and extent is an integer
    extent = extent + (extent+1)%2
    rsq = lrsqd3D.lrsqd3D(extent, sep[1]/sep[0], sep[2]/sep[0])
    mask = (rsq<sep[0]**2)
    
    # cast the mask into a one-dimensional form -- imask!
    bmask = np.zeros((nx,ny,extent[2]))
    bmask[0:extent[0],0:extent[1]] = mask
    imask = (bmask>0).ravel('F').nonzero()[0] + bignum - (nx*ny*(extent[2]//2)) - (nx*(extent[1]//2)) - (extent[0]//2)
    percentile = 0.7
    hash1 = np.ones(nx*ny*nz,'uint8')
    aflat = a.ravel('F')
    ww = (a[:,:,pad[2]:nz-pad[2]]>allmin).ravel('F').nonzero()[0] + (nx*ny*pad[2])
    nww = len(ww)
    ss = ((aflat[ww.astype(int)]).argsort(kind='mergesort')).astype(int)
    s = ww[ss[int(np.fix(percentile*nww)):]]
    ww = 0
    s = np.hstack([np.fliplr(s.reshape(1,len(s)))[0],0]).astype(int)
    idx=0
    rr = int(s[idx])
    m = aflat[rr]
    r = -1
    i = -1
    erwidx = len(s)-1
    
    while 1:
        # get the actual local max in a small mask
        indx = ((rr+imask)%(bignum)).astype(int)
        actmax = np.max(aflat[indx])
        # if our friend is a local max, then nuke out the big mask,update r
        if m >= actmax:
            r = np.hstack([r,rr])
            hash1[indx] = 0
        else:
            w = (aflat[indx] < m).nonzero()[0]
            nw = len(w)
            if nw > 0:
                indx2 = ((rr+imask[w])%(bignum)).astype(int)
                hash1[indx2] = 0
        
        # get the next non-nuked id
        while 1:
            idx = idx+1
            if hash1[s[idx]]==1 or idx >= erwidx:
                break
        
        if idx < erwidx:
            rr = s[idx]
            m = aflat[s[idx]]
        else:
            m = allmin
        if m <= allmin:
            break
            
    if len(r) > 1:
        r = r[1:]
    else:
        r = np.zeros((1,1))
        r[0,0] = -1
    
    x = (r%(nx*ny))%nx
    y = (r%(nx*ny))//nx
    z = r//(nx*ny)
    
    xtrue = np.logical_and(x>=pad[0]-1, x<=(nx-pad[0]-1)).nonzero()[0]
    ytrue = np.logical_and(y>=pad[1]-1, y<=(ny-pad[1]-1)).nonzero()[0]
    ztrue = np.logical_and(z>=pad[2]-1, z<=(nz-pad[2]-1)).nonzero()[0]
    w = np.array([i for i in xtrue if (i in ytrue and i in ztrue)])
    del r
    
    nw = len(w)
    if nw > 0:
        numel = len(x)
        x = x.reshape(numel,1)
        y = y.reshape(numel,1)
        z = z.reshape(numel,1)
        r = np.hstack([x[w],y[w],z[w]])
    else:
        r = np.zeros((1,1))
        r[0,0]=-1
    
    return r
    
    
    
    
        
            
                
            
        