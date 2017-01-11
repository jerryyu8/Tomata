import numpy as np

# Tutorial from
# http://iamtrask.github.io/2015/07/12/basic-python-network/

# sigmoid function
# translates from numbers to range of -1 to 1
def sigmoid(x):
    return 1/(1+np.exp(-x))

# deriv function of sigmoid (tells confidence)
# gets derivative of the point from range of -1 to 1
def deriv(y):
    return x * (1-x)



