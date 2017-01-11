import numpy as np

# Tutorial from
# http://iamtrask.github.io/2015/07/12/basic-python-network/

# Sigmoid function
# Translates from numbers to range of 0 to 1
def sigmoid(x):
    return 1/(1+np.exp(-x))

# Deriv function of sigmoid (tells confidence)
# Gets derivative of the point from range of 0 to 1
def deriv(y):
    return y * (1-y)

# Initialization

# Input Data
x = np.array([ [0,0,1],
               [0,1,1],
               [1,0,1],
               [1,1,1] ])

# Output Data
y = np.array([[0,0,1,1]]).T

# Seed RNG
np.random.seed(1)

# Initialize weights
# Have weights be from -1 to 1 in a 3x1 matrix
syn0 = 2 * np.random.random((3,1)) - 1

# Testing
for iter in range(10000):

    # Forward Propagation
    l0 = x # original
    rawInput = np.dot(l0,syn0)
    l1 = sigmoid(rawInput)

    # Error
    l1_error = y - l1
    # Factor in confidence, which is in slope
    # Closer to 0 or 1 > more confident > smaller slope > less change
    # Closer to .5 > less confident > larger slope > more change
    l1_delta = l1_error * deriv(l1)
    # Update Weights
    # delta is how much off each output is
    # flip l0 from each col being input and row being output
    # to col being output and row being input
    # then multiplying by delta shows how much each input needs to change
    # based on weights of all outputs and whether or not if affects it
    syn0 += np.dot(l0.T, l1_delta)

print ("Output After Training:")

print (l1)
