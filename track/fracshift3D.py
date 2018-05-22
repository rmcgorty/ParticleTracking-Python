from __future__ import division # Sets division to be float division
import numpy as np

def fracshift3D(im,shiftx,shifty,shiftz):
    # This function is aimed to increase
    # the resolution of x, y and z up to sub-pixel.
    # Written by Yongxiang Gao on June 15, 2005
    # Translated to Python by Kevin Smith (Maria Kilfoil's Group) June 2013
    
    ipx = int(shiftx)
    ipy = int(shifty)
    ipz = int(shiftz)
    
    fpx = shiftx - ipx
    fpy = shifty - ipy
    fpz = shiftz - ipz
    
    # to handle negative shifts
    if fpx < 0:
        fpx = fpx +1
        ipx = ipx -1
    if fpy < 0:
        fpy = fpy +1
        ipy = ipy -1
    if fpz <0:
        fpz = fpz +1
        ipz = ipz -1
    
    im = np.double(im)
    
    imagexz = np.roll(im,ipx,0)
    imagexz = np.roll(imagexz,ipy+1,1)
    imagexz = np.roll(imagexz,ipz,2)
    
    imageyz = np.roll(im,ipx+1,0)
    imageyz = np.roll(imageyz,ipy,1)
    imageyz = np.roll(imageyz,ipz,2)
    
    imagez = np.roll(im,ipx+1,0)
    imagez = np.roll(imagez,ipy+1,1)
    imagez = np.roll(imagez,ipz,2)
    
    imagexyz = np.roll(im,ipx,0)
    imagexyz = np.roll(imagexyz,ipy,1)
    imagexyz = np.roll(imagexyz,ipz,2)
    
    imagex = np.roll(im,ipx,0)
    imagex = np.roll(imagex,ipy+1,1)
    imagex = np.roll(imagex,ipz+1,2)
    
    imagey = np.roll(im,ipx+1,0)
    imagey = np.roll(imagey,ipy,1)
    imagey = np.roll(imagey,ipz+1,2)
    
    image = np.roll(im,ipx+1,0)
    image = np.roll(image,ipy+1,1)
    image = np.roll(image,ipz+1,2)
    
    imagexy = np.roll(im,ipx,0)
    imagexy = np.roll(imagexy,ipy,1)
    imagexy = np.roll(imagexy,ipz+1,2)
    
    res = (1-fpz)*((1-fpx)*(1-fpy)*imagexyz+(1-fpx)*fpy*imagexz+fpx*(1-fpy)*imageyz+fpx*fpy*imagez)+fpz*((1-fpx)*(1-fpy)*imagexy+(1-fpx)*fpy*imagex+ fpx*(1-fpy)*imagey+ fpx*fpy*image)
    return res
    
    
    