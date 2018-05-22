import numpy as np
def fieldof(array,f):
    # returns the even or odd field of an image
    # f=0 for odd fields starting with first,f=1 for even fields
    nd=array.ndim
    sz=array.shape
    if nd != 2:
        print('Error: Argument must be a two-dimensional array!')
    # if keyword_set(odd) then f=1 else f=0
    k=np.around((sz[0]+(1-f))/2)
    if k>0:
        ny2 = np.uint16(k)
    else:
        ny2=np.uint16(0)
    rows=np.arange(0,np.double(ny2)+1)*2 +1 +f
    res=array[rows,:]
    return res