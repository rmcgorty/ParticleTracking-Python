from __future__ import division # Sets division to be float division
import numpy as np
import sys
import llmx3D, lrsqd3D, make_arr, fracshift3D,gaussian3D,isinstack, nlparci
import pdb
reload(llmx3D)
reload(lrsqd3D)
reload(make_arr)
reload(fracshift3D)
reload(gaussian3D)
def feature3D_gaussian(image,diameter,masksz,xyzmax,inputv,sep,masscut=0,bckg=0,threshold=0):
    # this program is written by Yongxiang Gao and Maria Kilfoil based on IDL
    # code written by John C. Crocker and David G. Grier.
    # varargin should contain input for [ separation, masscut, threshold], and
    # whether there is input for it or not indicated in the logical input for
    # inputv. inputv contains 3 logical input, for example [0,1,0].
    # Last revisement is done on June 15, 2005
    # Adapted for Python in June 2013 by Kevin Smith (Maria Kilfoil's Group)
    #
    # diameter is size of features, as sent to llmx3dMB, and previously to this
    # function, to bpass3D. masksz is the mask size over which the fitting is
    # done.
    #
    # 		Feature3d
    # PURPOSE:
    # Finds and measures roughly spheroidal 'features' within 
    # 		a 3d image. Works best with dilute or separate feature,
    # 		i.e. non close-packed structures.
    # CATEGORY:
    # 		Image Processing
    #  CALLING SEQUENCE:
    # 		f = feature3d( image, diameter ,separation, masscut, threshold )
    #  INPUTS:
    # 		image:	(nx,ny,nz) array which presumably contains some
    # 			features worth finding
    # 		diameter: a parameter which should be a little greater than
    # 			the diameter of the largest features in the image.
    # 			May be a single float number if the image is 
    # 			isotropic, a 3-vector otherwise.
    # 		separation: an optional parameter which specifies the 
    # 			minimum allowable separation between feature 
    # 			centers. The default value is diameter-1.
    # 		masscut: Setting this parameter saves runtime by reducing
    # 			the runtime wasted on low mass 'noise' features.
    # 		threshold: Set this parameter to a number less than 1 to
    # 			threshold each particle image by 
    # 			(peak height)*(threshold).  Reduces pixel biasing
    # 			with an particle specific threshold.
    #  OUTPUTS:
    # 		f(:,1):	this contains the x centroid positions, in pixels.
    # 		f(:,2): this contains the y centroid positions, in pixels. 
    # 		f(:,3): this contains the z centroid positions, in pixels.
    # 		f(:,4): this contains the integrated brightness.
    # 		f(:,5): this contains the squared radius of gyration. 
    # 		f(:,6): this contains the peak height of the feature.
    # 		f(:,7): this contains the fraction of voxels above threshold.
    # 
    #  SIDE EFFECTS:
    # 		Displays the number of features found on the screen.
    #  RESTRICTIONS:
    # 		To work properly, the image must consist of bright, 
    # 		smooth regions on a roughly zero-valued background. 
    # 		To find dark features, the image should be 
    # 		inverted and the background subtracted. If the image
    # 		contains a large amount of high spatial frequency noise,
    # 		performance will be improved by first filtering the image.
    # 		'bpass3d' will remove high spatial frequency noise, and 
    # 		subtract the image background and thus may provides a useful 
    # 		complement to using this program. Individual features 
    # 		should NOT overlap or touch.  Furthermore, the maximum
    # 		value of the top of the feature must be in the top 30th
    # 		percentile of brightness in the entire image.
    # 		For images where the particles are close packed, the
    # 		system of bpass3d/feature3d is not ideal, but will give
    # 		rough coordinates.  We often find setting 'sep' to roughly
    # 	diameter/2 seems helpful to avoid particle loss.
    #  PROCEDURE:
    # 		First, identify the positions of all the local maxima in
    # 		the image ( defined in a circular neighborhood with radius
    # 		equal to 'separation' ). Around each of these maxima, place a 
    # 		circular mask, of diameter 'diameter', and calculate the x,y,z
    #     	centroids, the total of all the pixel values.
    # 		If the restrictions above are adhered to, and the features 
    # 		are more than about 5 pixels across, the resulting x 
    # 		and y values will have errors of order 0.1 pixels for 
    # 		reasonably noise free images.
    # 		If 'threshold' is set, then the image within the mask is 
    # 		thresholded to a value of the peak height*threshold.  This
    # 		is useful when sphere images are connected by faint bridges
    # 		which can cause pixel biasing.
    # 
    #  *********	       READ THE FOLLOWING IMPORTANT CAVEAT!        **********
    # 		'feature3d' is capable of finding image features with sub-pixel
    # 		accuracy, but only if used correctly- that is, if the 
    # 		background is subtracted off properly and the centroid mask 
    # 		is larger than the feature, so that clipping does not occur.
    # 		It is an EXCELLENT idea when working with new data to plot
    # 		a histogram of the x-positions mod 1, that is, of the
    # 		fractional part of x in pixels.  If the resulting histogram
    # 		is flat, then you're ok, if its strongly peaked, then you're
    # 		doing something wrong- but probably still getting 'nearest
    # 		pixel' accuracy.
    # 
    # 		For a more quantitative treatment of sub-pixel position 
    # 		resolution see: 
    # 		J.C. Crocker and D.G. Grier, J. Colloid Interface Sci.
    # 		*179*, 298 (1996).
    # 
    #  MODIFICATION HISTORY:
    # 		This code is inspired by feature_stats2 written by
    # 			David G. Grier, U of Chicago, 			 1992.
    # 		Generalized version of feature.pro			 1998.
    # 		Improved local maximum routine, from fcp3d		 1999.
    # 		Added 'threshold' keyword to reduce pixel biasing.	 1999.
    # 		
    # 	This code feature3d.pro is copyright 1999, John C. Crocker and 
    # 	David G. Grier.  It should be considered 'freeware'- and may be
    # 	distributed freely in its original form when properly attributed.
    # -
    # 
    # 	produce a 3d, anisotropic parabolic mask
    # 	anisotropic masks are 'referenced' to the x-axis scale.
    # 	ratios less than one squash the thing relative to 'x'
    # 	using float ratios allows precise 'diameter' settings 
    # 	in an odd-sized mask.
    
    # For anisotropy, make diameter a 3-vector, otherwise one # is ok.
    # Image should consist of smooth well separated peaks on a zero
    # or near zero background, with diameter set somewhat larger
    # than the diameter of the peak!
    diameter = np.array(diameter)
    sep = np.array(sep)
    masksz = np.array(masksz)
    
    if inputv[0]==1:
        seperation = sep
    if inputv[2]==1:
        if threshold <=0 or threshold >=0.9:
            sys.exit('Error: Threshold value must be between 0.0 and 0.9!')
    
    # make extents be the smallest odd integers bigger than diameter        
    try:
        if len(diameter) == 1:
            diameter = np.zeros(3)+diameter
    except:
        diameter = np.zeros(3)+diameter
        
    extent = np.fix(diameter)+1
    extent = extent + (extent+1)%2
    extt = extent
    nx, ny, nz = image.shape
    if not (inputv[0]==1):
        sep = diameter-1
    else:
        sep = np.double(seperation)
    
    try: 
        if len(sep)==1:
            sep = np.zeros(3)+sep
    except:
        sep = np.zeros(3)+sep
    
    # Put a border around the image to prevent mask out-of-bounds
    a = np.zeros((nx+extent[0],ny+extent[1],nz+extent[2]))
    xlow = int(extent[0]//2)
    ylow = int(extent[1]//2)
    zlow = int(extent[2]//2)
    for i in xrange(0,nz):
        a[xlow:xlow+nx,ylow:ylow+ny,zlow+i] = image[:,:,i]
    nx = int(nx + extent[0])
    ny = int(ny + extent[1])
    nz = int(nz + extent[2])
    # Find the local maxima in the filtered image
    loc = llmx3D.llmx3D(a,sep,extent//2)
    if (loc[0,0]==-1):
        print('No local maxima found!')
        r=np.array([])
        return r

    # Set up some stuff...
    nmax = len(loc[:,0])
    x = loc[:,0]
    y = loc[:,1]
    z = loc[:,2]
    
    # Leave some space near the edge to avoid out of border
    xtrue = np.logical_and(x>masksz[0]-3, x<xyzmax[0]+extent[0]-masksz[0]+1).nonzero()[0]
    ytrue = np.logical_and(y>masksz[1]-3, y<xyzmax[1]+extent[1]-masksz[1]+1).nonzero()[0]
    ztrue = np.logical_and(z>masksz[2]-3, z<xyzmax[2]+extent[2]-masksz[2]-1).nonzero()[0]
    id = [i for i in xtrue if (i in ytrue and i in ztrue)]
    x = x[id]
    y = y[id]
    z = z[id]
    extent = np.fix(masksz) +1
    extent = extent + (extent+1)%2
    xl = x - extent[0]//2
    xh = xl + extent[0]
    yl = y -extent[1]//2
    yh = yl + extent[1]
    zl = z - extent[2]//2
    zh = zl + extent[2]
    
    rsq = lrsqd3D.lrsqd3D(extent,masksz[1]/masksz[0],masksz[2]/masksz[0])
    mask = (rsq<((masksz[0]/2)**2)+1)
    shell = (mask == (rsq > (masksz[0]/2 -1)**2))
    nask = np.sum(mask)
    rmask = (rsq*mask)+(1/6)
    
    imask = (make_arr.make_arr(extent[0],extent[1],extent[2])-1) % int(extent[0])+1
    xmask = np.double(mask*imask)
    imask = (make_arr.make_arr(extent[1],extent[0],extent[2])-1) % int(extent[1])+1
    ymask = np.double(mask*imask.transpose(1,0,2))
    imask = (make_arr.make_arr(extent[2],extent[1],extent[0])-1) % int(extent[2])+1
    zmask = np.double(mask*imask.transpose(2,1,0))
    nmax = len(x)
    m = np.zeros(nmax)
    pd = np.zeros(nmax)
    thresh = np.zeros(nmax)
    nthresh = np.zeros(nmax)
    tops = np.zeros(nmax)
    for i in xrange(0,nmax):
        tops[i] = a[x[i],y[i],z[i]]
    if inputv[2]==1:
        thresh = tops*threshold
    if inputv[2]==1:
        for i in xrange(0,nmax):
            bb = a[xl[i]:xh[i],yl[i]:yh[i],zl[i]:zh[i]]
            nthresh[i] = np.sum(bb*mask>thresh[i])/np.sum(bb*mask>0)
    
    # Estimate the mass
    for i in xrange(0,nmax):
        temp=a[xl[i]:xh[i],yl[i]:yh[i],zl[i]:zh[i]]-thresh[i]
        m[i] = np.sum((temp>0)*temp*mask) # mass of each feature
        del temp
    # do a masscut, and prevent dividing by 0 in the centroid calc.
    w = (m>masscut).nonzero()[0]
    nmax = len(w)
    if nmax==0:
        print('No features found!')
        r=np.array([])
        return r
        xl = xl[w]
    xh = xh[w]
    yl = yl[w]
    yh = yh[w]
    zl = zl[w]
    zh = zh[w]
    x = x[w]
    y = y[w]
    z = z[w]
    m = m[w]
    tops = tops[w]
    thresh = thresh[w]
    nthresh = nthresh[w]
    print(str(nmax) + ' features found.')
    
    r=np.zeros((len(x),31))
    w = extent//2
    wshift = extt//2
    
    for jj in xrange(0,len(x)):   # cycle through all the local maxima found
        #if jj%49==0:
        #    print jj
        img_vec=np.zeros(extent[0]*extent[1]*extent[2])
        pos_vec=np.zeros(extent[0]*extent[1]*extent[2])
        # Linearize the xyz positions and the image for mynlnfit
        for k in xrange(0,int(extent[2])):
            for j in xrange(0,int(extent[1])):
                img_vec[int(k*extent[1]*extent[0]+j*extent[0]):int(k*extent[1]*extent[0]+(j+1)*extent[0])] = np.double(a[int(x[jj]-w[0]):int(x[jj]+w[0]+1),int(y[jj]-w[1]+j),int(z[jj]-w[2]+k)])
                pos_vec[int(k*extent[1]*extent[0]+j*extent[0]):int(k*extent[1]*extent[0]+(j+1)*extent[0])] = (z[jj]-w[2]+k)*2000**2 + (y[jj]-w[1]+j)*2000 + np.arange(x[jj]-w[0],(x[jj]+w[0]+1))
        
        if bckg==0:
           beta,R,J,it = gaussian3D.fit(pos_vec,img_vec,[x[jj],y[jj],z[jj]]+list(masksz/3)+[np.double(a[x[jj],y[jj],z[jj]])])
           beta = beta + [0]
        else: #  ks figure out Jacobian and desired error
           beta,R,J,it = gaussian3D.fit(pos_vec,img_vec,[x[jj],y[jj],z[jj]]+list(masksz/3)+[np.double(a[x[jj],y[jj],z[jj]]),bckg])
        r[jj,0:3] = np.hstack([x[jj], y[jj], z[jj]])-wshift
        r[jj,3:6] = np.array(beta[0:3])-wshift
        r[jj,9:13] = np.array(beta[3:7])
        r[jj,20] = it
        r[jj,22] = m[jj]
        r[jj,24:28] = beta[3:7]
        
        if it < 800 and isinstack.isinstack(beta[0:3]-wshift, np.array(xyzmax) - 1):
            x[jj] = round(beta[0])
            y[jj] = round(beta[1])
            z[jj] = round(beta[2])
            for k in xrange(0,int(extent[2])):
                for j in xrange(0,int(extent[1])):
                    img_vec[int(k*extent[1]*extent[0]+j*extent[0]):int(k*extent[1]*extent[0]+(j+1)*extent[0])] = np.double(a[int(x[jj]-w[0]):int(x[jj]+w[0]+1),int(y[jj]-w[1]+j),int(z[jj]-w[2]+k)])
                    pos_vec[int(k*extent[1]*extent[0]+j*extent[0]):int(k*extent[1]*extent[0]+(j+1)*extent[0])] = (z[jj]-w[2]+k)*2000**2 + (y[jj]-w[1]+j)*2000 + np.arange(x[jj]-w[0],(x[jj]+w[0]+1))
            if bckg==0:
                beta2,R,J,it = gaussian3D.fit(pos_vec,img_vec,[x[jj],y[jj],z[jj]]+list(masksz/3)+[np.double(a[x[jj],y[jj],z[jj]])])
                beta2 = beta2 + [0]
            else:
                beta2,R,J,it = gaussian3D.fit(pos_vec,img_vec,[x[jj],y[jj],z[jj]]+list(masksz/3)+[np.double(a[x[jj],y[jj],z[jj]]),bckg])
            r[jj,21] = it
            r[jj,6:9] = beta2[0:3]-wshift
            r[jj,9:13] = beta[3:7]
            if it < 800 and isinstack.isinstack(beta[0:3]-wshift, np.array(xyzmax) - 1):
                ci=nlparci.nlparci(beta2[0:7],R,J[:,0:7])  # computes 2 sigma (95% confidence) values
                mystderr = (ci[:,1]-ci[:,0])/4  # a good approximation for sigma
                r[jj,13:20]=mystderr
            r[jj,28] = r[jj,9]**2+r[jj,10]**2+r[jj,11]**2 # size of feature (sigmax**2+sigmay**2+sigmaz**2)
            r[jj,29] = tops[jj]
            if bckg!=0:
                r[jj,30] = beta2[7]
    rsq = lrsqd3D.lrsqd3D(extent,1,1)
    mask_fs = rsq < (3/2)**2 + 0.5
    for i in xrange(0,nmax):
        if int(r[i,6]-w[0]+1)>=1 and int(r[i,7]-w[1]+1)>=1 and int(r[i,8]-w[2]+1)>=1 and int(r[i,6]+w[0])<=xyzmax[0]-1 and int(r[i,7]+w[1])<=xyzmax[1]-1 and int(r[i,8]+w[2])<=xyzmax[2]-1:
            suba = fracshift3D.fracshift3D(np.double(image[int(r[i,6]-w[0]):int(r[i,6]+w[0]+1),int(r[i,7]-w[1]):int(r[i,7]+w[1]+1),int(r[i,8]-w[2]):int(r[i,8]+w[2]+1)]),int(r[i,6])-r[i,6],int(r[i,7])-r[i,7],int(r[i,8])-r[i,8])
            try:
                r[i,23] = np.sum(suba*mask_fs) # Integrated intensity
            except:
                pdb.set_trace()
        else:
            r[i,23] = 0
    
    return r
            
    
    
     