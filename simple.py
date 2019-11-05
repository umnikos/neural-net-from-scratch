import numpy as np

# sigmoid function
"""
def nonlin(x,deriv=False):
    if(deriv==True):
        return x*(1-x)
    return 1/(1+np.exp(-x))
"""
def nonlin(x,deriv=False):
    dx = np.ones_like(x)
    dx[x<=0] = 0.01
    if(deriv):
        return dx
    return x*dx
    
# input dataset
X = np.array([  [0,0,1],
                [0,1,1],
                [1,0,1],
                [1,1,1] ])
    
# output dataset            
y = np.array([[0,0,1,1]]).T

# seed random numbers to make calculation
# deterministic (just a good practice)
np.random.seed(10)

# initialize weights randomly with mean 0
syn0 = np.random.random((3,1))
print("Weights Before Training:")
print(syn0)

a = 0.1
b = 0.6
z = np.zeros((3, 1))
for iter in range(100):

    # forward propagation
    l0 = X
    l1 = nonlin(np.dot(l0,syn0))

    # backwards propagation
    l1_error = (l1-y)
    l1_delta = l1_error * nonlin(l1,True)
    z = b*z + np.dot(l0.T,l1_delta)
    syn0 -= a * z

#syn0 = np.array([[-100],[0],[0]])

print("Output After Training:")
print(l1)
print("Weights After Training:")
print(syn0)
print("New Example ([0,0,0])")
print(nonlin(np.dot(np.array([[0,0,0]]),syn0)))
print("New Example ([0,1,0])")
print(nonlin(np.dot(np.array([[0,1,0]]),syn0)))
print("New Example ([1,0,0])")
print(nonlin(np.dot(np.array([[1,0,0]]),syn0)))
print("New Example ([1,1,0])")
print(nonlin(np.dot(np.array([[1,1,0]]),syn0)))
