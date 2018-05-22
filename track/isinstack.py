def isinstack(test, stacklim):
    bool1 = test[0] > 0 and test[0] < stacklim[0]
    bool2 = test[1] > 0 and test[1] < stacklim[1]
    bool3 = test[2] > 0 and test[2] < stacklim[2]
    return (bool1 and bool2 and bool3)