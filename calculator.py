# -----------------------------------------------------------------------------
# calculator.py
# ----------------------------------------------------------------------------- 
'''
Before editing:

10014 function calls in 0.030

After editing:

4 function calls in 0.001 seconds

The speed up rate is 0.03/0.001 = 30 

'''
import numpy as np

def hypotenuse(x,y):
    """
    Return sqrt(x**2 + y**2) for two arrays, a and b.
    x and y must be two-dimensional arrays of the same shape.
    """
    xx = np.multiply(x,x)
    yy = np.multiply(y,y)
    zz = np.add(xx, yy)
    return np.sqrt(zz)