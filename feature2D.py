from __future__ import division # Sets division to be float division
import bpass, localmax, create_mask, fracshift
import numpy as np

def feature2D(img,Lambda,w,masscut=0,Imin=0,field=2,bandpass='raw',
              verbose=True):
    
    #     Written 7-29-03 by Maria Kilfoil
    # 	  Finds and measures roughly circular 'features' within an image.
    #     note: extent should be 2*w+1 in which w is the same as bpass.
    
    #  CALLING SEQUENCE:
    # 	f = feature2D(image,lambda,diameter,masscut,Imin,field)
    #  INPUTS:
    # 	img:	(nx,ny) array which presumably contains some features worth finding
    #   Lambda: length scale of noise to be filtered out, in pixels; typically 1
    # 	w:      a parameter which should be a little greater than the radius of the 
    #           largest features in the image.
    # 	Imin: 	(optional) Set this optional parameter to the minimum allowed value for the peak 
    #           brightness of a feature. Useful for limiting the number of spurious features in
    #       	noisy images. If set to 0 (default), the top 30# of the bright pixels
    #       	will be used.
    # 	masscut: (optional) Setting this parameter saves runtime by reducing the runtime wasted on 
    #           low mass 'noise' features. (default is 0, accept all.)
    # 	field: 	(optional) Set this parameter to 0 or 1 if image is actually just one (odd or even) field of an interlaced 
    #           (e.g. video) image. All the masks will then be constructed with a 2:1 aspect ratio. Otherwise 
    #           set to 2 for progressive scan cameras.
    #   bandpass: (optional) Setting this parameter to 'bp' will run feature finding on the bandpassed image, while setting
    #           it to 'raw' will run feature finding on the raw image. The default is 'raw'.
    #
    # NOT IMPLEMENTED:
    # 	separation: an optional parameter which specifies the minimum allowable separation 
    #           between feature centers. The default value is diameter+1.
    #
    #
    #  OUTPUTS:
    # 		f(:,1):	the x centroid positions, in pixels.
    # 		f(:,2): the y centroid positions, in pixels. 
    # 		f(:,3): integrated brightness of the features. ("mass")
    # 		f(:,4): the square of the radius of gyration of the features.
    # 		    (second moment of the "mass" distribution, where mass=intensity)
    # 		f(:,5): eccentricity, which should be zero for circularly symmetric features and 
    #                   order one for very elongated images.
    #  RESTRICTIONS:
    #       To work properly, the image must consist of bright, circularly symmetric regions 
    #       on a roughly zero-valued background. To find dark features, the image should be 
    #       inverted and the background subtracted. If the image contains a large amount of 
    #       high spatial frequency noise, performance will be improved by first filtering the image.
    #       BPASS will remove high spatial frequency noise, and subtract the image background. 
    #       Individual features should NOT overlap.
    #
    #  MODIFICATION HISTORY:
    # 		This code is inspired by feature_stats2 written by
    # 			David G. Grier, U of Chicago, 			 1992.
    # 		Written by John C. Crocker, U of Chicago, optimizing 
    # 			runtime and measurement error, 			10/93.
    #       	Matlab version written by Maria L. Kilfoil		2003.
    #       10-18-07 Vincent Pelletier, Maria Kilfoil -- added masscut to Matlab version
    #       June 2013 - Adpated for Python by Kevin Smith (Maria Kilfoil's Group)
    
    extent=2.*w+1
    if (bandpass=='bp'):
        image=bpass.bpass(img,Lambda,w)
    elif (bandpass=='raw'):
        image=img
    else:
        image=img
    if extent%2==0:
        print('Requires an odd extent. Adding 1...')
        extent=extent+1
    sz = image.shape
    nx = sz[1]
    ny = sz[0]
    # if n_params() eq 2 then sep = extent+1
    sep = extent
    
    #Put a border around the image to prevent mask out-of-bounds
    #Only use the following 2 lines if you are not using bpass to do spatial filtering first.
    #Otherwise, image returned from bpass already had a border of w width
    a = image
    
    #   Finding the local maxima
    loc = localmax.localmax(image,sep,field,Imin)
    if len(loc[0])==0 or loc[0][0]==-1:
        r = -1
        return r
    x=[0]*len(loc[0])
    y=[0]*len(loc[0])
    for i in xrange(0,len(loc[0])):
        y[i]=int(loc[0][i])
        x[i]=int(loc[1][i])
    x=x-np.double(0) #convert to arrays
    y=y-np.double(0)
    
    nmax = len(loc[0])
    m = np.zeros((1,len(loc[0])))[0]
    xl = x-(extent//2)
    xh = xl+extent-1;
    
    extent=int(extent)
    # Set up some masks
    rsq = create_mask.rsqd(extent,extent)
    t = create_mask.thetarr(extent)
    
    mask  = np.less_equal(rsq,(extent/2)**2)
    mask2 = np.ones((extent,1))*np.arange(1,extent+1,1) # checked
    mask2 = mask2*mask
    mask3 = (rsq*mask)+(1/6)
    cen = int((extent-1)/2)
    cmask = np.cos(2*t)*mask
    smask = np.sin(2*t)*mask
    cmask[cen,cen]=0.0
    smask[cen,cen]=0.0
    
    suba = np.zeros((extent,extent,nmax))
    xmask = mask2
    ymask = mask2.T;
    yl = y-(extent//2)
    yh = yl+extent-1
    yscale = 1
    ycen = cen
    
    # Estimate the mass
    for i in xrange(0,nmax):
        m[i] = (np.double(a[int(yl[i]):int(yh[i])+1,int(xl[i]):int(xh[i])+1])*mask).sum()
    
    # remove features based on 'masscut' parameter
    b = (m>masscut).nonzero()
    nmax = len(b[0])
    if nmax==0:
        print('No feature found!')
        r=[];
        return r
    xl = xl[b]
    xh = xh[b]
    yl = yl[b]
    yh = yh[b]
    x  =  x[b]
    y  =  y[b]
    m  =  m[b]
    
    if verbose:
        print(str(nmax) + ' features found.')
    
    # Setup some result arrays
    xc = [0]*nmax-np.double(0)
    yc = [0]*nmax-np.double(0)
    rg = [0]*nmax-np.double(0)
    e = [0]*nmax-np.double(0)
    
    # Calculate feature centers
    for i in xrange(0,nmax):
        xc[i] = (np.double(a[int(yl[i]):int(yh[i])+1,int(np.fix(xl[i])):int(xh[i])+1])*xmask).sum()
        yc[i] = (np.double(a[int(yl[i]):int(yh[i])+1,int(xl[i]):int(xh[i])+1])*ymask).sum()
    x1=x
    y1=y
    
    # Correct for the 'offset' of the centroid masks
    xc = xc/m-((extent+1)/2)
    yc = (yc/m - (extent+1)/2)/yscale
    
    # Update the positions and correct for the width of the 'border'
    x = x+xc-0*(extent//2)
    y=(y+yc-0*(extent//2))*yscale
    x2=x
    y2=y
    
    # Construct the subarray and calculate the mass, squared radius of gyration, eccentricity
    for i in xrange(0,nmax):
        suba[:,:,i] = fracshift.fracshift(np.double(a[int(yl[i]):int(yh[i])+1,int(xl[i]):int(xh[i])+1]),-xc[i],-yc[i])
        m[i] = (suba[:,:,i]*mask).sum()
        rg[i] = ((suba[:,:,i]*mask3).sum())/m[i]
        tmp = np.sqrt((((suba[:,:,i]*cmask).sum())**2)+(((suba[:,:,i]*smask).sum())**2))
        tmp2 = (m[i]-suba[cen,ycen,i]+1e-6)
        e[i] = tmp/tmp2
    
    for i in xrange(0,nmax):
        xc[i] = np.double(suba[:,:,i]*xmask).sum()
        yc[i] = np.double(suba[:,:,i]*ymask).sum()
        
    xc = xc/m - (extent+1)/2
    yc = (yc/m - (extent+1)/2)/yscale # get mass center
    x3 = x2 + xc - 0*(extent//2)
    y3 = (y2 + yc - 0*(extent//2))*yscale
    
    r_mat=np.zeros((nmax,5)) 
    r_mat[:,0]=x3 # final x position
    r_mat[:,1]=y3 # final y position
    r_mat[:,2]=m  # final integrated intensity
    r_mat[:,3]=rg # final radius of gyration
    r_mat[:,4]=e  # final eccentricity
    # uncomment the following lines if you would like x1, y1, x2, and x3 to be stored
    # (these are the initial position and the position after the first iteration of centroid fitting).
    #r2_mat=np.zeros((nmax,4))   #### change to resize array
    #r_mat=np.hstack([r_mat,r2_mat])
    #r_mat[:,5]=x1
    #r_mat[:,6]=y1  
    #r_mat[:,7]=x2
    #r_mat[:,8]=y2 
    r = r_mat
    
    return r
    