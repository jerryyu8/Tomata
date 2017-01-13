import numpy as np
from preProcessing import *


# 3 layer neural net

# Alpha values (changing the step of the weights)
# alphas = [0.001,0.01,0.1,1,10,100,1000]
alphas = [.1]

# Hidden size (of the hidden layer)
# Larger helps get to result faster, covers more
# Usually recommended to be between the input and output sizes
# Along with 1 optimized hidden layer
hiddenSize = 500

# Sigmoid function for nonlinearity
# Translates from numbers to range of 0 to 1
def sigmoid(x):
    return 1/(1+np.exp(-x))

# Deriv function of sigmoid (tells confidence)
# Gets derivative of the point from range of 0 to 1
def deriv(y):
    return y * (1-y)

# Initialization

preProcess = preProcess()
preProcess.run()

# Input Data
x = preProcess.input

# Output Data
y = preProcess.output

# Train with all alpha values
for alpha in alphas:
    print("Training With Alpha:" + str(alpha))

    # Seed RNG
    np.random.seed(1)

    # Initialize weights
    # First Layer
    # Have weights be from -1 to 1 in a 3x4 matrix
    # Convert raw input 4x3 into 4x4 output
    # 10x200400
    syn0 = 2 * np.random.random((2000,hiddenSize)) - 1
    # Second Layer
    # Convert raw input 4x4 to a 4x1
    syn1 = 2 * np.random.random((hiddenSize,1)) - 1

    # Testing
    for iter in range(30001):

        # Forward Propagation
        l0 = np.array(x) # original

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
            print ("Error at "+ str(iter)+ ": " + str(np.mean(np.abs(l2_error))))
        # Update Weights
        syn1 += alpha * l1.T.dot(l2_delta)
        syn0 += alpha * l0.T.dot(l1_delta)

print ("Output After Training:")
print (l2)

print("Testing")
l0 = np.array(preProcess.test) # original

l1RawInput = np.dot(l0,syn0)
l1 = sigmoid(l1RawInput)

l2RawInput = np.dot(l1,syn1)
l2 = sigmoid(l2RawInput)

print("Results")
print(l2)

f = open('rawLog.txt', 'w')
np.set_printoptions(precision=3)
f.write(np.array_repr(syn0))
f.write(np.array_repr(syn1))
