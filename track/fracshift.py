from __future__ import division # Sets division to be float division
import numpy as np

def fracshift(im,shiftx,shifty):
    # barrel "shifts" a floating point arr by a fractional pixel amount
    # using a 'lego' interpolation technique.
    
    ipx = int(shiftx) # integer part
    ipy = int(shifty)
    fpx = shiftx - ipx # decimal part
    fpy = shifty - ipy
    
    # to handle negative shifts:
    if fpx<0:
        fpx = fpx + 1
        ipx = ipx - 1
    
    if fpy<0:
        fpy = fpy + 1
        ipy = ipy - 1
        
    image = np.double(im)
    imagex = np.roll(image,ipy,0)
    imagex = np.roll(imagex,ipx+1,1)
    imagey = np.roll(image,ipy+1,0)
    imagey = np.roll(imagey,ipx,1)
    imagexy = np.roll(image,ipy+1,0)
    imagexy = np.roll(imagexy,ipx+1,1)
    image = np.roll(image,ipy,0)
    image = np.roll(image,ipx,1)
    
    res = ((1-fpx)*(1-fpy)*image)+(fpx*(1-fpy)*imagex)+((1-fpx)*fpy*imagey)+(fpx*fpy*imagexy)
    return res