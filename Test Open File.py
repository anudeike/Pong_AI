import csv
import numpy as np
import pandas as pd

#============OPEN FILE===============#
pathY = r"C:\Users\berro\Desktop\Python Helpful Materials\Artificial Neural Networks\PongProject\test_YMaster.csv"
pathX = r"C:\Users\berro\Desktop\Python Helpful Materials\Artificial Neural Networks\PongProject\test_XMaster.csv"

#Opening and putting them in arrays
dfX = pd.read_csv(pathX)
dfY = pd.read_csv(pathY)

#alpha
alpha = 0.01
#inputs
X = pd.DataFrame.as_matrix(dfX)
Y = pd.DataFrame.as_matrix(dfY)

def nonlinear(x, deriv=False):
    if (deriv==True):
        return x*(1.0-x)
    return 1.0/(1.0+np.exp(-x))

#seed
np.random.seed(1)

#synapses
syn0 = 2*np.random.random((4,500))-1
syn1 = 2*np.random.random((500,500))-1

test_x = np.array([[251,497,-246],
                  [299,249,50],
                  [194,180,14],
                  [140,148,-8],
                  [210,140,70]])

for j in range(10000):

    # forward propagation
    l0 = X
    l1 = nonlinear(np.dot(l0, syn0))
    l2 = nonlinear(np.dot(l1, syn1))

    # how much did we miss
    l2_error = Y - l2

    # multiply how much missed by the slope of sigmoid at the value in l1
    l2_delta = l2_error * nonlinear(l2, True)

    # how muchh did l1 contribute to l2 error
    # (accroding to the weights)
    l1_error = l2_delta.dot(syn1.T)

    # in what direction is the target l1?
    # were we sure? if so, don't change too much
    l1_delta = l1_error * nonlinear(l1, True)

    # update weight
    syn1 += alpha * (l1.T.dot(l2_delta))
    syn0 += alpha * (l0.T.dot(l1_delta))

    # display error
    if (j % 1000) == 0:
        print("ERROR: " + str(np.mean(np.abs(l2_error))))

# Testing Forward propagation
def foward_test(test_x, syn0, syn1):
    l0_test = test_x
    l1_test = nonlinear(np.dot(l0_test, syn0))
    l2_test = nonlinear(np.dot(l1_test, syn1))

# Dress up the array (make it look nice)
    l2_test_output = []
    for x in range(len(l2_test)):
        l2_test_output.append(l2_test[x][0])

    print("Test Output")
    print(l2_test_output)

# Put all the l2 data in a way I could see it: Just the first probabilites
l2_output = []
for x in range(len(l2)):
    l2_output.append(l2[x][0])

print("Output")
print(l2_output)