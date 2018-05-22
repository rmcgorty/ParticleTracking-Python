from __future__ import division # Sets division to be float division
import time
import unq
import numpy as np
import pdb
import luberize
reload(unq)

def trackmem(xyzs,maxdisp,dim,goodenough,memory):
    
    #      Written 02-2005 by Naama Gal and Maria Kilfoil
    #
    # 	Links the positions found in successive frames into trajectories.
    #       The algorithm tries to minimize the sum of the distances between features in two successive frames. 
    # 	*Works with 2d and 3d spatial dimension data*
    
    # The input xyzs must be a matrix of positions and time (ie: MT), where time must
    # be in the last column and positions in the first columns.  The program will
    # sort the data according to the first columns, where the number of columns
    # taken into account is determined by dim. 
    # Columns in between will be ignored.
    
    # Also, the matrix needs to be sorted by time.
    
    # The spatial dimension dim of the data is specified by the user.
    
    dd= len(xyzs[1,:])-1 # the last column of xyzs
    t = (xyzs[:,dd])
    
    #check the input time vector is ok, i.e. sorted and uniform
    st_t=np.roll(t,1)
    st=[0]*(len(st_t)-1)-np.double(0) # intialize st and make it an array
    for i in xrange(1,len(t)):
        st[i-1] = t[i]-st_t[i]
    
    # all elements of st should be equal and positive, OR ZERO
    if ((st<0).sum()!=0):
        print('ERROR - Time vector out of order!')
    w = (st>0).nonzero()[0]
    z = len(w)
    if (z==0):
        print('ERROR - All positions are at the same time!')
    elif ((st[w]-st[w[0]]).sum()!=0):
        print('WARNING - Time vector gapped or not evenly gridded!')
    
    z = z+1
    
    # partition the input data by unique times
    res = unq.unq(t,[])
    res = res+1
    res = np.hstack([0,res,len(t)]) # ks I don't understand the len(t)
    ngood = res[1]-res[0]
    # ngood here is number of features, starting from the beginning of the input
    # features file, which have the same time step
    eyes = np.arange(0,ngood)
    
    pos = xyzs[eyes,0:dim]
    istart = 1 # we don't need to track t=0
    n = ngood
    
    # how long are the 'working' copies of the data?
    zspan = 50
    if n > 200:
        zspan = 20
    if n > 500:
        zspan = 10
    
    resx = np.zeros((zspan,n)) -1
    bigresx = np.zeros((z,n)) -1
    mem = np.zeros((1,n))[0]
    uniqid = np.arange(0,n) # ks not sure about this, might have to change to np.arange(0,n) if this is used for indexing
    maxid = n
    olist = np.hstack([1,1])
    
    if goodenough>0:
        dumphash = np.zeros((1,n))[0]
        nvalid = np.ones((1,n))[0]
        
    resx[0,:] = eyes # put the first set of feature indices in the first row of resx
    
    # set up some nice constants
    maxdisq = maxdisp**2
    notnsqrd = np.logical_and((np.sqrt(n*ngood) > 200),(dim < 7)) # ks why is it n*ngood if they are the same
    
    if notnsqrd:
        cube = np.zeros((3**dim,dim))
        # construct the vertices of a 3x3x3... d-dimensional hypercube
        for d in xrange(1,dim+1):
            numb = 0
            for j in xrange(1,int(3**dim)+1,int(3**(d-1))):
                for jj in xrange(j-1,j+3**(d-1)-1):
                    cube[jj,(d-1)] = numb
                numb = (numb+1)%3
        # calculate a blocksize which may be greater than maxdisp, but which
        # keeps nblocks reasonably small.
        volume = 1 # volume in dimensional space. e.g. dim=2, "volume" is the area
        for d in xrange(0,dim):
            minn = np.min(xyzs[w,d])
            maxx = np.max(xyzs[w,d])
            volume = volume*(maxx-minn)
        
        blocksize = np.max(maxdisp,(volume/(20*ngood))**(1.0/dim)) # Tailor the factor in bottom for the particular system 
    # Start the main loop over the frames.
    for i in xrange(istart+1,z+1): # always starts at 2 (while inipos is not implemented)
        ispan = (i-1) % zspan
        # Get the new particle positions
        m = res[i] - res[i-1] # number of new particles
        eyes = res[i-1] + np.arange(0,m) # points to the lines in the feature file for the new particles
        
        if m>0:
            xyi = xyzs[eyes,0:dim] # positions of new particles
            found = np.zeros((1,m))[0]
            
            # THE TRIVIAL BOND CODE BEGINS
            if notnsqrd:
                # Use the raster metrix code to do trivial bonds
                
                # construct "s", a one dimensional parameterization of the space
                # (which consists of the d-dimensional raster scan of the volume.)
                abi = np.floor(xyi/blocksize)
                abpos = np.floor(pos/blocksize)
                si = np.ones((1,m))[0]
                spos = np.zeros((1,n))[0]
                dimm = np.zeros((1,dim))[0]
                coff = 1
                
                for j in xrange(0,dim):
                    minn = np.min(np.vstack([abi[:,j],abpos[:,j]]))
                    maxx = np.max(np.vstack([abi[:,j],abpos[:,j]]))
                    abi[:,j] = abi[:,j] - minn
                    abpos[:,j] = abpos[:,j] - minn
                    dimm[j] = maxx - minn + 1
                    si = si + abi[:,j]*coff
                    si= si - 1 # ks added in. Not sure
                    spos = spos + abpos[:,j]*coff
                    coff = coff*dimm[j]
                nblocks = coff # the no. of blocks in the volume
                
                # trim down (intersect) the hypercube if its too big to fit in
                # the particle volume. (i.e. if dimm(j) lt 3)
                cub = cube
                deg = (dimm<3).nonzero()[0]
                if deg:
                    for j in xrange(0,len(deg)):
                        cub = cub[(cub[:,deg[j]]<dimm[deg[j]]).nonzero(),:]
                
                # calculate the "s" coordinates of hypercube (with a corner at the origin)
                scube = np.zeros((len(cub[:,0])))
                coff = 1
                for j in xrange(0,dim):
                    scube = scube + cub[:,j]*coff
                    coff = coff*dimm[j]
                
                # shift the hypercube "s" coordinates to be centered around the origin
                coff = 1
                for j in xrange(0,dim):
                    if dimm[j] > 3:
                        scube = scube - coff
                scube = (scube+nblocks) % nblocks
                
                # get the sorting for the particles by their "s" positions.
                isort = si.argsort()
                
                # make a hash table which will allow us to know which new particles
                # are at a given si.
                strt = np.zeros(nblocks)-1
                fnsh = np.zeros(nblocks)
                
                for j in xrange(1,m+1):
                    if strt[si[isort[j-1]]] == -1: # ks how could it be anything else?
                        strt[si[isort[j-1]]] = j
                        fnsh[si[isort[j-1]]] = j
                    else:
                        fnsh[si[isort[j-1]]] = j
                # loops over the old particles, and find those new particles in the 'cube'.
                coltot = np.zeros(m)
                rowtot = np.zeros(n)
                which1 = np.zeros(n)
                for j in xrange(0,n):
                    map1 = -2
                    s = ((scube + spos[j]) % nblocks)
                    s=s.astype(int)
                    w = (strt[s] != -1).nonzero()[0]
                    ngood = len(w)
                    if ngood !=0:
                        s = s[w]
                        for k in xrange(0,ngood):
                            map1 = np.hstack([map1,isort[(strt[s[k]]-1):(fnsh[s[k]])]])
                        map1 = map1[1:len(map1)]
                        
                        # find those trivial bonds
                        distq = np.zeros(len(map1))
                        for d in xrange(0,dim):
                            distq = distq + (xyi[map1,d] - pos[j,d])**2
                        ltmax = distq < maxdisq
                        rowtot[j] = ltmax.sum()
                        
                        if rowtot[j]>=1:
                            w = ltmax.nonzero()[0]
                            coltot[map1[w]] = coltot[map1[w]] +1
                            which1[j] = map1[w[0]]
                
                nrtk = np.floor(n-(rowtot==0).sum())
                w = (rowtot==1).nonzero()[0]
                ngood = len(w)
                if ngood != 0: # ks not tested
                    ww = (coltot[which1[w]]==1).nonzero()[0]
                    ngood = len(w)
                    if ngood != 0:
                        resx[ispan,w[ww]] = eyes[which1[w[ww]]]
                        found[which1[w[ww]]] = 1
                        rowtot[w[ww]] = 0
                        coltot[which1[w[ww]]] = 0
                
                labely = (rowtot>0).nonzero()[0]
                ngood=len(labely)
                if ngood !=0:
                    labelx = (coltot>0).nonzero()[0]
                    nontrivial = 1
                else:
                    nontrivial = 0
                del abi,abpos,fnsh,rowtot,coltot,which1,isort
            else:
                # or: Use simple N**2 time routine to calculate trivial bonds
                
                # let's try a nice, loopeless way! Don't bother tracking perm. lost guys
                wh = (pos[:,0]>0).nonzero()[0]
                ntrack = len(wh)
                if ntrack==0:
                    print('WARNING - No valid particles to track!')
                xmat = np.ones((ntrack,m))
                for ng in xrange(0,m):
                    xmat[:,ng] = ng
                xmat=xmat.astype(int)
                ymat = np.ones((m,ntrack))
                for ng in xrange(0,ntrack):
                    ymat[:,ng] = ng
                ymat = (ymat.astype(int)).T
                
                for d in xrange(0,dim):
                    x = xyi[:,d]
                    y = pos[wh,d]
                    if d==0:
                        dq = (x[xmat]-y[ymat])**2
                    else:
                        dq = dq + (x[xmat]-y[ymat])**2
                ltmax = (dq < maxdisq)
                        
                # figure out which trivial bonds go with which
                rowtot = np.zeros(n)
                rowtot[wh] = ltmax.sum(axis=1)
                if (ntrack>1):
                    coltot = ltmax.sum(axis=0)
                else:
                    coltot = ltmax[0]
                which1 = np.zeros(n)
                for j in xrange(0,ntrack):
                    mx = np.max(ltmax[j,:]) # max is faster than where
                    w = (ltmax[j,:]==mx).nonzero()[0]
                    if len(w)>1:
                        w = w[0]
                    which1[wh[j]] = w
                ntrk = np.floor(n - (rowtot==0).sum())
                w = (rowtot==1).nonzero()[0]
                ngood = len(w)
                which1 = which1.astype(int)
                if (ngood !=0):
                    ww = (coltot[which1[w]]==1).nonzero()[0]
                    ngood = len(ww)
                    if (ngood !=0):
                        resx[ispan,w[ww]] = eyes[which1[w[ww]]]
                        found[which1[w[ww]]] = 1
                        rowtot[w[ww]] = 0
                        coltot[which1[w[ww]]] = 0
                labely = (rowtot>0).nonzero()[0]
                ngood = len(labely)
                if (ngood != 0):
                    labelx = (coltot >0).nonzero()[0]
                    nontrivial = 1
                else:
                    nontrivial = 0
                del rowtot, coltot, which1
            # THE TRIVIAL BOND CODE ENDS
            if nontrivial:
                xdim = len(labelx)
                ydim = len(labely)
                
                # make a list of the non-trivial bonds
                bonds = np.ones(2)
                bondlen = 0
                for j in xrange(0,ydim):
                    distq = np.zeros(xdim)
                    for d in xrange(0,dim):
                        distq = distq + (xyi[labelx,d] - pos[labely[j],d])**2
                    w = (distq < maxdisq).nonzero()[0]
                    ngood = len(w)
                    bonds=np.vstack([bonds,np.hstack([w.reshape(len(w),1),np.zeros((ngood,1))+j])]) # ks not sure about the plus one
                    bondlen = np.hstack([bondlen,distq[w]])
                bonds = bonds[1:,:]
                bondlen = bondlen[1:]
                numbonds = bonds.shape[0]
                mbonds = tuple(bonds)
                mbonds = np.array(mbonds)
                
                if np.max([xdim,ydim]) < 4:
                    nclust = 1
                    maxsz = 0
                    mxsz = xdim
                    mysz = ydim
                    bmap = np.zeros(bonds.shape[0])-1
                else:
                    # THE SUBNETWORK CODE BEGINS
                    
                    lista = np.zeros(numbonds)
                    listb = np.zeros(numbonds)
                    nclust = 0
                    maxsz = 0
                    thru = xdim
                    
                    while (thru != 0):
                        #    the following code extracts connected sub-networks of the non-trivial
                        #    bonds.  NB: lista/b can have redundant entries due to
                        #    multiple-connected subnetworks.
                        w = (bonds[:,1]>=0).nonzero()[0] # ks not sure about this equal to
                        lista[0] = bonds[w[0],1]
                        listb[0] = bonds[w[0],0]
                        bonds[w[0],:] = -(nclust+1) # ks not sure
                        adda = 1
                        addb = 1
                        donea = 0
                        doneb = 0
                        
                        repeat=1
                        while (repeat==1):
                            if donea !=adda:
                                w = (bonds[:,1]==lista[donea]).nonzero()[0]
                                ngood = len(w)
                                if ngood != 0:
                                    listb[addb:addb+ngood] = bonds[w,0]
                                    bonds[w,:] = -(nclust+1)
                                    addb = addb+ngood
                                donea = donea+1
                            if doneb != addb:
                                w = (bonds[:,0]==listb[doneb]).nonzero()[0]
                                ngood = len(w)
                                if ngood !=0:
                                    lista[adda:adda+ngood] = bonds[w,1]
                                    bonds[w,:] = -(nclust+1)
                                    adda = adda+ngood
                                doneb=doneb+1
                            if (donea == adda) and (doneb==addb):
                                repeat=0
                        idxsortb = listb[0:doneb].argsort()
                        idxsorta = lista[0:donea].argsort()
                        xsz = len(unq.unq(listb[0:doneb],idxsortb))
                        ysz = len(unq.unq(lista[0:donea],idxsorta)) # ks test this
                        
                        if xsz*ysz > maxsz:
                            maxsz = xsz*ysz
                            mxsz = xsz
                            mysz = ysz
                        
                        thru = thru - xsz
                        nclust = nclust + 1
                        
                    bmap = bonds[:,0]
                
                # PREMUTATION CODE BEGINS
                for nc in xrange(0,nclust):
                    w = (bmap == -(nc+1)).nonzero()[0]
                    nbonds = len(w)
                    bonds = mbonds[w,:]
                    lensq = bondlen[w]
                    sortidx = bonds[:,0].argsort()
                    uold = bonds[unq.unq(bonds[:,0],sortidx),0]
                    nold = len(uold)
                    unew = bonds[unq.unq(bonds[:,1],[]),1] # ks -1 was in first index
                    nnew = len(unew)
                    
                    # check that runtime is not excessive
                    if nnew > 5:
                        rnsteps = 1
                        for ii in xrange(0,nnew):
                            rnsteps = rnsteps * len((bonds[:,1]==unew[ii]).nonzero()[0])
                            if rnsteps > 5e4:
                                print('WARNING: Difficult combinatorics encountered')
                                print('WARNING: Program may not finish - Try reducing maxdisp')
                            if rnsteps > 2e5:
                                print('Excessive Combinatorics! Try reducing maxdisp')
                    st = np.ones(nnew)
                    fi = np.ones(nnew)
                    h = np.ones(nbonds)
                    ok = np.ones(nold)+1
                    if nnew-nold > 0:
                        nlost = nnew - nold
                    else:
                        nlost = 0
                    
                    for ii in xrange(0,nold):
                        h[(bonds[:,0] == uold[ii]).nonzero()[0]] = ii
                    st[0] = 0
                    fi[nnew-1]=nbonds
                    if nnew > 1:
                        sb = bonds[:,1]
                        sbr = np.roll(sb,1)
                        sbl = np.roll(sb,-1)
                        st[1:] = (sb[1:] != sbr[1:]).nonzero()[0]+1
                        fi[0:nnew-1] = (sb[0:nbonds-1] != sbl[0:nbonds-1]).nonzero()[0]+1
                    
                    checkflag = 0
                    while checkflag != 2:
                        pt = st-1
                        lost = np.zeros(nnew)
                        who = 0
                        losttot = 0
                        mndisq = nnew*maxdisq
                        pt=pt.astype(int)
                        ok=ok.astype(int)
                        fi=fi.astype(int)
                        h=h.astype(int)
                        
                        while who != -1:
                            if pt[who] != fi[who]-1:
                                w = (ok[h[pt[who]+1:fi[who]]]).nonzero()[0]
                                ngood = len(w)
                                if ngood > 0:
                                    if pt[who] != st[who]-1:
                                        ok[h[pt[who]]] = 1
                                    pt[who] = pt[who]+w[0]+1
                                    ok[h[pt[who]]] = 0 # ks not sure if this should be -1
                                    if who == nnew-1:
                                        ww = (lost == 0).nonzero()[0]
                                        dsq = (lensq[pt[ww]]).sum() + losttot*maxdisq
                                        if dsq < mndisq:
                                            minbonds = pt[ww]
                                            mndisq = dsq
                                    else:
                                        who = who +1
                                else:
                                    notlost = -lost[who]-1
                                    if np.logical_and(((notlost % 2) == 1), (losttot != nlost)):
                                        lost[who] = 1
                                        losttot = losttot+1
                                        if pt[who] != st[who]-1:
                                            ok[h[pt[who]]]=1
                                        if who == nnew-1:
                                            ww = (lost==0).nonzero()[0]
                                            dsq = (lensq[pt[ww]]).sum()+losttot*maxdisq
                                            if dsq < mndisq:
                                                minbonds = pt[ww]
                                                mndisq = dsq
                                        else:
                                            who = who + 1
                                    else:
  
                                        if pt[who] != st[who]-1:
                                            ok[h[pt[who]]] = 1
                                        pt[who] = st[who] -1
                                        if lost[who]:
                                            lost[who] = 0
                                            losttot = losttot -1
                                        who = who -1
                            else:
                                notlost = -lost[who]-1
                                if np.logical_and((notlost % 2)==1,(losttot != nlost)):
                                    lost[who] = 1
                                    losttot = losttot +1
                                    if pt[who] != st[who]-1:
                                        ok[h[pt[who]]] = 1
                                    if who == nnew-1:
                                        ww = (lost == 0).nonzero()[0]
                                        dsq = (lensq[pt[ww]]).sum() + losttot*maxdisq
                                        if dsq < mndisq:
                                            minbonds = pt[ww]
                                            mndisq = dsq
                                    else:
                                        who = who +1
                                else:
                                    if pt[who] != st[who]-1:
                                        ok[h[pt[who]]] = 1
                                    pt[who] = st[who] -1
                                    if lost[who]:
                                        lost[who] = 0
                                        losttot = losttot -1
                                    who = who -1
                        checkflag = checkflag + 1
                        if checkflag == 1:
                            # we need to check that our constraint on nlost is not forcing us away from the minimum id's
                            plost = np.min([np.floor(mndisq/maxdisq),nnew-1])
                            if plost > nlost+1:
                                nlost = plost
                            else:
                                checkflag = 2
                    
                    # update resx using the minimum bond configuration
                    bonds = bonds.astype(int)
                    resx[ispan,labely[bonds[minbonds,1]]] = eyes[labelx[bonds[minbonds,0]]]
                    found[labelx[(bonds[minbonds,0]).astype(int)]] = 1                     
                            
                            
                            
            w = (resx[ispan,:]>0).nonzero()[0]
            nww = len(w)
            resx = resx.astype(int)
            if (nww > 0):
                pos[w,:] = xyzs[resx[ispan,w],0:dim]
                if (goodenough > 0):
                     nvalid[w] = nvalid[w]+1
            else:
                 print('WARNING - tracking zero particles!')
             
            # we need to add new guys, as appropriate.
            newguys = (found==0).nonzero()[0]
            nnew = len(newguys)
            if (nnew > 0):
                newarr = np.zeros((zspan,nnew))-1
                resx = np.hstack([resx,newarr])
                resx[ispan,n:] = eyes[newguys]
                pos = np.vstack([pos,xyzs[eyes[newguys],0:dim]])   
                mem = np.hstack([mem,np.zeros(nnew)])
                uniqid = np.hstack([uniqid,np.arange(0,nnew)+maxid])
                maxid = maxid + nnew
                if goodenough > 0:
                    dumphash = np.hstack([dumphash,np.zeros(nnew)])
                    nvalid = np.hstack([nvalid,np.ones(nnew)])
                n = n + nnew
                
        else:
            print('WARNING - No positions found for t=' + str(i))
            
        # update the 'memory' array
        w = (resx[ispan,:] != -1).nonzero()[0]
        nok = len(w)
        if (nok != 0):
            mem[w] = 0    # guys get reset if they're found
        mem = mem + (resx[ispan,:] == -1)
        
        # if a guy has been lost for more than memory times, mark him as permanently lost.
        # For now, set these guys to pos = (-maxdisp,-maxdisp,...), so we can never track
        # them again. It would be better to make a smaller pos, but then we'd have to change
        # 'n', which would be gnarly.
        wlost = (mem == memory+1).nonzero()[0]
        nlost = len(wlost)
        if nlost > 0:
            pos[wlost,:] = -maxdisp
            # check to see if we should 'dump' newly lost guys
            if goodenough > 0:
                wdump = (nvalid[wlost] < goodenough).nonzero()[0]
                ndump = len(wdump)
                if (ndump > 0):
                    dumphash[wlost[wdump]] = 1
        # we need to insert the working copy of resx into the big copy bigresx
        # do our house keeping every zspan time steps (dumping bad lost guys)

        if ((ispan+1) == zspan or i ==z):
            #  if a permanently lost guy has fewer than goodenough valid positions
            #  then we 'dump' him out of the data structure- this largely alleviates
            #  memory problems associated with the 'add' keyword and 'noise' particles
            #  To improve speed- do it infrequently.
            # in case we've added some we need to pad out bigresx too
            nold = len( bigresx[0,:] );
            nnew = n - nold
            if (nnew > 0):
                newarr = np.zeros((z,nnew))-1
                bigresx = np.hstack([bigresx,newarr])
            
            if goodenough > 0:
                if dumphash.sum() > 0:
                    wkeep = (dumphash==0).nonzero()[0]
                    nkeep = len(wkeep)
                    resx = resx[:,wkeep]
                    bigresx = bigresx[:,wkeep] # this really hurts runtime
                    pos = pos[wkeep,:]
                    mem = mem[wkeep]
                    uniqid = uniqid[wkeep]
                    nvalid = nvalid[wkeep]
                    n = nkeep
                    dumphash = np.zeros(nkeep)      
            bigresx[(i-ispan-1):i,:] = resx[0:ispan+1,:] # ks not sure about indexing
            resx = np.zeros((zspan,n)) -1
            
            #  We should pull permanently lost guys, parse them and concat them
            #  onto the 'output list', along with their 'unique id' number to
            #  make scanning the data files a little easier.  Do infrequently.
            wpull = (pos[:,0] == -maxdisp).nonzero()[0]
            npull = len(wpull)
            if npull > 0:
                lillist = np.hstack([1,1])
                for ipull in xrange(0,npull):
                    wpull2 = (bigresx[:,wpull[ipull]] != -1).nonzero()[0]
                    npull2 = len(wpull2)
                    lillist = np.vstack([lillist,np.hstack([(bigresx[wpull2,wpull[ipull]]).reshape(npull2,1),np.zeros((npull2,1))+uniqid[wpull[ipull]]])])
                olist = np.vstack([olist,lillist[1:,:]])
            # now  get rid of the guys we don't need anymore...
            # but watch out for when we have no valid particles to track!
            wkeep = (pos[:,0] > 0).nonzero()[0]
            nkeep = len(wkeep)
            if nkeep ==0:
                print('WARNING - We are going to crash now, no particles...')
            resx = resx[:,wkeep]
            bigresx = bigresx[:,wkeep]
            pos = pos[wkeep,:]
            mem = mem[wkeep]
            uniqid = uniqid[wkeep]
            n = nkeep
            dumphash = np.zeros(nkeep)
            if goodenough > 0:
                nvalid = nvalid[wkeep]             
    # end of the big loop over z time steps...
    # make a final scan for short trajectories that weren't lost at the end
    if goodenough > 0:
        nvalid = (bigresx > -1).sum(axis=0)
        wkeep = (nvalid >= goodenough).nonzero()[0]
        nkeep = len(wkeep)
        if nkeep < n:
            bigresx = bigresx[:,wkeep]
            n = nkeep
            uniqid = uniqid[wkeep]
            pos = pos[wkeep,:]
    
    # make the final scan to 'pull everybody else into the olist
    wpull = (pos[:,0] != -2*maxdisp).nonzero()[0]
    npull = len(wpull)
    if npull > 0:
        lillist = np.hstack([1,1])
        for ipull in xrange(0,npull):
            wpull2 = (bigresx[:,wpull[ipull]] != -1).nonzero()[0]
            npull2 = len(wpull2)
            lillist = np.vstack([lillist,np.hstack([(bigresx[wpull2,wpull[ipull]]).reshape(npull2,1),np.zeros((npull2,1))+uniqid[wpull[ipull]]])])
        olist = np.vstack([olist,lillist[1:,:]])
    try:
        olist = olist[1:,:]
    except:
        olist = np.array([])
    
    # free up a little memory for the final step!
    bigresx = 0
    resx = 0
    
    # need to make up a result array!
    if olist != []:
        nolist = len(olist[:,0])
    else:
        nolist = 0
    res = np.zeros((nolist,dd+2))
    
    olist = olist.astype(int)
    if olist != []:
        for j in xrange(0,dd+1):
            res[:,j] = xyzs[olist[:,0],j]
            res[:,dd+1] = olist[:,1] +1
    else:
        res = np.asarray([])
    
    if res.shape[0]>0:
        lub = luberize.luberize(res)
    else:
        lub = res
    
    stop_time = time.time()
    return lub