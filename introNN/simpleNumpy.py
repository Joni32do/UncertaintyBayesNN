#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 1 21:18:33 2022

@author: Jo_Ni
"""

from telnetlib import DM
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

### Parameter

n = 20
train = True


epoch = 10000
lr = 0.01
loss_collection = []



### Calculated Parameter

def f(x):
    return torch.pow(x,3) - 2 * torch.pow(x,2) + torch.pow(x,1)


x_train = torch.reshape(torch.linspace(-2,2,n),(n,1))
y_train = f(x_train)
#neural network with one hidden layer with n neurons and one input and one output
w_1 = torch.rand((1,n))
b_1 = torch.zeros((1,n))
w_2 = torch.rand((n,1))
b_2 = torch.zeros((1,1))

def forward(x, w_1,b_1,w_2,b_2):
    return sigmoid(x@w_1+b_1)@w_2 + b_2


def sigmoid(x):
    '''
    Implementation of the sigmoid function
    '''
    return 1/(1 + np.exp(-x))

def dSigmoid(x):
    '''
    Derivative of the sigmoid of function
    '''
    return torch.mul((1 - sigmoid(x)),(sigmoid(x)))

#Plotting Test
# x = torch.linspace(-5,5,100)
# yS = sigmoid(x)
# yDs = dSigmoid(x)
# plt.plot(x,yS,x,yDs)
# plt.show()

def mse(y_train, y_pred):
    '''
    calculate the mean square error element wise
    '''
    return 0.5 * torch.sum(np.square(y_train-y_pred),axis=-1)

def dMse(y_train, y_pred):
    return y_pred - y_train




def calcGradient(x_train, y_train, y_pred, w_1, b_1, w_2, b_2):
    ly1 = x_train@w_1 + b_1
    dLdw1 = dMse(y_train, y_pred) * dSigmoid(ly1) * x_train * w_2
    dLdb1 = dMse(y_train, y_pred) * dSigmoid(ly1) * w_2
    dLdw2 = dMse(y_train, y_pred) * sigmoid(ly1)
    dLdb2 = dMse(y_train, y_pred)
    return dLdw1, dLdb1, dLdw2, dLdb2




if train:

    

    # for k in [x_train, y_train, w_1, b_1, w_2, b_2]:
    #     print(k)


    for e in range(epoch):
        y_pred = forward(x_train, w_1, b_1, w_2, b_2)
        loss = torch.mean(mse(y_train,y_pred))


        # calculate gradient
        dLdw1, dLdb1, dLdw2, dLdb2 = calcGradient(x_train, y_train, y_pred, w_1, b_1, w_2, b_2)
        
        # calculate gradient over entire dataset
        w_1 = w_1 - lr * torch.mean(dLdw1, axis=0, keepdims=True)
        b_1 = b_1 - lr * torch.mean(dLdb1, axis=0, keepdims=True)
        w_2 = w_2 - lr * torch.t(torch.mean(dLdw2, axis=0, keepdims=True))
        b_2 = b_2 - lr * torch.t(torch.mean(dLdb2, axis=0))

        # Tracking
        if (e+1) % 200 == 0:
            print("epoch",e+1,"  loss:", loss)
            loss_collection.append(loss)

    plt.plot(loss_collection)
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.show()

    ###Testing
    print(w_1)
    print(b_1)
    print(w_2)
    print(b_2)
    print("\n")
else:
    #load saved values (better do this not this ugly)
    w_1 = torch.tensor([[2.2282, 2.1509, 2.2236, 2.2972, 2.4106, 2.1838, 2.2648, 2.2394, 2.2091,
         2.2673]])
    b_1 = torch.tensor([[-9.0243, -8.6625, -9.0025, -9.3418, -4.9926, -8.8172, -9.1932, -9.0758,
         -8.9353, -9.2047]])
    w_2 = 0.0
    b_2 = 0.0


x_test = torch.reshape(torch.linspace(-2,2,20),(20,1))
y_test = f(x_test)
y_pred = forward(x_test, w_1, b_1, w_2, b_2)

print(torch.Tensor.size(y_pred))
plt.plot(x_test, y_test, label='function')

plt.plot(x_test,y_pred,'b-', label = 'prediction')
plt.xlabel('x-values')
plt.ylabel('y-values')
plt.legend()
plt.show()

# params = [w_1, b_1, w_2, b_2]
# np.savetxt("model.csv",params, delimiter=',')









