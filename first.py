#!/usr/bin/python3
import numpy as np
import idx2numpy
import cv2 as cv
np.random.seed(1)
np.set_printoptions(suppress=True)

# 60k x 28 x 28
images = idx2numpy.convert_from_file("training/train-images-idx3-ubyte")
# 10k x 28 x 28
test_images = idx2numpy.convert_from_file("test/t10k-images-idx3-ubyte")
# 60k
labels = idx2numpy.convert_from_file("training/train-labels-idx1-ubyte")
# 10k
test_labels = idx2numpy.convert_from_file("test/t10k-labels-idx1-ubyte")

# 60k x 28*28
inputs = images.reshape((60000, 28*28))
# 10k x 28*28
test_inputs = test_images.reshape((10000, 28*28))
# 60k x 10
outputs = np.zeros((60000,10))
for i in range(60000):
    outputs[i][labels[i]]=1

def relu(x, deriv=False):
    """
    dx = np.ones_like(x)
    dx[x<=0] = 0.01
    if(deriv):
        return dx
    return x*dx
    """
    if(deriv==True):
        return x*(1-x)
    result = np.zeros_like(x)
    result[x < 0] = np.exp(x[x < 0]) / (np.exp(x[x < 0]) + 1)
    result[x >= 0] = 1 / (1 + np.exp(-x[x >= 0]))
    return result


def batch_generator(inputs, outputs, batch_size=1000):
    indices = np.arange(len(inputs))
    batch=[]
    while True:
        np.random.shuffle(indices)
        for i in indices:
            batch.append(i)
            if len(batch) == batch_size:
                yield inputs[batch], outputs[batch]
                batch=[]

n1 = 16
n2 = 10

def random_weights():
    global syn1
    syn1 = 2*np.random.random((28*28, n1))-1
    global syn2
    syn2 = 2*np.random.random((n1, n2))-1


def forward(ins):
    # 1000 x n1
    global a1 
    a1 = relu(np.dot(ins, syn1))
    # 1000 x n2
    global a2 
    a2 = relu(np.dot(a1, syn2))

def train(epochs):
    global syn1
    global syn2
    a = 0.001
    b = 0.5
    gen = batch_generator(inputs, outputs)
    z1 = 0
    z2 = 0
    for e in range(epochs):
        for i in range(60):
            # forward propagation
            ins, outs = next(gen)
            forward(ins)
        #   print("a2:")
        #   print(a2)

            # backwards propagation
            # 1000 x n2
            a2_err = a2 - outs
            # 1000 x n2
            a2_delta = a2_err * relu(a2,True)
            # 1000 x n1
            a1_err = np.dot(a2_delta, syn2.T) 
            a1_delta = a1_err * relu(a1,True)
            z2 = b*z2 + np.dot(a1.T, a2_delta)
        #   print("z2:")
        #   print(z2)
            syn2 -= a * z2
            z1 = b*z1 + np.dot(ins.T, a1_delta)
        #   print("z1:")
        #   print(z1)
            syn1 -= a * z1
        if e % 10 == 0:
            forward(inputs)
            print("epoch {0} error:".format(e+300)) 
            print(np.sum(np.square(a2-outputs)))
            np.save('syn1.npy', syn1)
            np.save('syn2.npy', syn2)

#random_weights()
syn1 = np.load('syn1.npy')
syn2 = np.load('syn2.npy')
#train(1001)
forward(test_inputs)
guesses = np.argmax(a2, axis=1)
print("guesses:")
print(guesses[range(100)])
print("actual values:")
print(test_labels[range(100)])
print("accuracy: {0}".format(np.sum(guesses == test_labels)*100/10000))
"""
import imageio
im = 255-imageio.imread('four.png')[:,:,0]
print(im)
forward(im.flatten())
print(a2)
print(np.argmax(a2))
"""
