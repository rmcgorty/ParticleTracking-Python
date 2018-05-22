import unq
import numpy as np

def luberize(tracks):
    # reassigns the unique ID# to 0,1,2,3...
    # presort will sort on ID# first, then reassign
    # start will begin with that ID#
    
    # function returns a new track array
    
    ndat = len(tracks[0,:])-1
    
    newtracks = tracks
    
    u = unq.unq((newtracks[:,ndat]),[])
    ntracks = len(u)
    u = np.hstack([-1,u])
    for i in xrange(0,ntracks):
        newtracks[u[i]+1:u[i+1]+1,ndat] = i+1
    
    return newtracks