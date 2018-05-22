from __future__ import division # Sets division to be float division
import sys
import create_mask
import fieldof
import numpy as np
import scipy.ndimage.morphology as spim

def localmax(image,sep,field,intmin):
    # This function takes in an image and returns the location of its local
    # maxima.
    
    # supports the field argument
    Range = np.double(sep//2)
    a=np.double(image)
    w = int(np.around(2*Range+1))
    s = create_mask.rsqd(w,w)
    mask = (s<=Range**2)
    yrange = Range
    if field==1 or field==0:
        mask = fieldof.fieldof(mask,0)
        yrange = np.double(np.int16(Range/2))+1
    elif field != 2:
        sys.exit('Error:Field parameter in localmax must be 0,1 or 2')
    imd = np.double(spim.grey_dilation(a,None,mask))
    # but don't include pixels from the background which will be too dim
    if intmin==0:
        k=np.zeros((256,len(a[0,:])))
        amin=a.min()
        amax=a.max()
        for ii in xrange(0,len(a[0,:])):
            tmp=np.histogram(a[:,ii],256,(amin,amax))[0]
            k[:,ii]=tmp
        h=k.sum(axis=1)
        h=h.cumsum()
        h=h/(h.max())
        intmin=1+(h<0.70).sum()
        if intmin<5:
            intmin=np.around(len(h)/5)
    r = np.logical_and(a==imd, a>=intmin).nonzero()
    
    # Discard maxima within range of the edge
    sz = a.shape
    nx = sz[1]
    ny = sz[0]
    x=[0]*len(r[0])
    y=[0]*len(r[0])
    for i in xrange(0,len(r[0])):
        x[i]=r[1][i]
        y[i]=r[0][i]
    x0 = x - Range
    x1 = x + Range
    y0 = y - yrange
    y1 = y + yrange
    x=x-np.double(0) #convert to array
    y=y-np.double(0) #convert to array
    good = np.logical_and(np.logical_and(x0>=0,x1<(nx-1)),np.logical_and(y0>=0,y1<(ny-1))).nonzero()
    good=good[0]
    ngood = len(good)
    r=list(r)
    r[0]=list(r[0])
    r[1]=list(r[1])
    r=r-np.double(0) #convert to an array
    
    r0=[0]*ngood-np.double(0)
    r1=[0]*ngood-np.double(0)
    r0 = r[0][good]
    r1 = r[1][good]
    del r
    r=[0]*2; r[0]=[0]*ngood; r[1]=[0]*ngood;
    r[0]=r0; r[1]=r1;
    x = x[good]
    y = y[good]
    x0 = x0[good]
    x1 = x1[good]
    y0 = y0[good]
    y1 = y1[good]
    
    # Find and clear spurious points arising from features which get 
    # found twice or which have flat peaks and thus produce multiple hits.
    c=np.zeros((ny,nx))
    for i in xrange(0,ngood):
        #print i,r[0][i],r[1][i]
        c[int(r[0][i]),int(r[1][i])] = a[int(r[0][i]),int(r[1][i])]
    center = w * Range + Range # position in mask pixel number of the center of the mask
    for i in xrange(0,ngood):
         b = c[int(y0[i]):int(y1[i]+1),int(x0[i]):int(x1[i]+1)]
         b = b*mask # look only in circular region
         Y = np.msort(b)
         I = b.argsort(axis=0)
         g1 = I[-1,:] # array that contains which row was the biggest value for each column
         yi = np.max(Y[-1,:]) # yi is the abs maximum within the mask
         locx = np.argmax(Y[-1,:]) # locx is the column number of the max
         locy = g1[locx] # row number corresponding to maximum
         [d1, d2]=b.shape;
         location = (locx)*(d1)+locy # find the location of remaining maximum in the mask
         if location != center: # compare location of max to position of center of max
             c[int(y[i]),int(x[i])]=0
    r = (c!=0).nonzero() # What's left are valid maxima
    r=list(r)-np.double(0); r[0]=r[0]; r[1]=r[1];
    return r # return their locations