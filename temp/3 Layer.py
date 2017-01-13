import numpy as np

# 3 layer neural net
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
# First Layer
# Have weights be from -1 to 1 in a 3x4 matrix
# Convert raw input 4x3 into 4x4 output
syn0 = 2 * np.random.random((3,4)) - 1
# Second Layer
# Convert raw input 4x4 to a 4x1
syn1 = 2 * np.random.random((4,1)) - 1

# Testing
for iter in range(60000):

    # Forward Propagation
    l0 = x # original

    l1RawInput = np.dot(l0,syn0)
    l1 = sigmoid(l1RawInput)

    l2RawInput = np.dot(l1,syn1)
    l2 = sigmoid(l2RawInput)

    # Second Layer Processing
    # Error
    l2_error = y - l2
    # Factor in confidence, which is in slope
    l2_delta = l2_error * deriv(l2)

    # First Layer Processing
    # Error
    # Error of each result split over all weights from syn1
    l1_error = l2_delta.dot(syn1.T)
    # Factor in confidence, which is in slope
    l1_delta = l1_error * deriv(l1)

    if (iter % 10000) == 0:
        print ("Error:" + str(np.mean(np.abs(l2_error))))

    # Update Weights
    syn1 += l1.T.dot(l2_delta)
    syn0 += l0.T.dot(l1_delta)

testX = np.array([[0,1,0]])
print("Output of [0,1,0]")
testl1 = sigmoid(testX.dot(syn0))
print(sigmoid(testl1.dot(syn1)))

testX = np.array([[1,1,0]])
print("Output of [1,1,0]")
testl1 = sigmoid(testX.dot(syn0))
print(sigmoid(testl1.dot(syn1)))

print ("Output After Training:")
print (l2)

