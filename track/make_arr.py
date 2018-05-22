from __future__ import division # Sets division to be float division
import numpy as np
def make_arr(xr,yr,zr):
    arr = np.arange(1,int(xr*yr*zr)+1)
    matx = arr.reshape((int(xr),int(yr),int(zr)),order='F')
    return matx