from pathlib import Path
import os

import numpy as np
import matplotlib.pyplot as plt


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist
from bnnLayer import BayesLinear, MeanFieldGaussianFeedForward
from matplotlib import cm

'''
Until I have achieved something meaningful I will mainly work in this file and not refactor in main and other sub files.
This might be ugly but better then create a good framework for shit
'''

##################################################################################################################
# Uncertainty
#######
torch.manual_seed(42)






##################################################################################################################
# Dataset
######

n_dim = 1

n_train = 20
n_test = 81 #better if it is a square

in_dis = 2 #Assumes symmetric distance and zero centering
out_dis = 3 #in_dis < out_dis

aleatoric_train = 0 #aleatoric uncertainty
aleatoric_test = 0

#TODO: If else decision is ugly
if n_dim == 1:

    ### 1D Regression Dataset

    #I could also discriminate like in 2D between equidistant und random

    def f(x,aleatoric=0):
        '''
        cubic polynomial R1 -> R1
        '''
        # return torch.pow(x,4) - torch.pow(x,2) + 5 * torch.pow(x,1) + aleatoric*(torch.rand(x.size())-0.5)
        return torch.sin(2*torch.pi *x)/x + aleatoric*(torch.rand(x.size())-0.5)


    x_train = torch.reshape(torch.linspace(-in_dis,in_dis,n_train),(n_train,1))
    y_train = f(x_train,aleatoric_train)

    x_test = torch.reshape(torch.linspace(-out_dis,out_dis,n_test),(n_test,1))
    y_test = f(x_test,aleatoric_test)
else:
    ### 2D Regression Dataset


    def f_2D(X,Y=None,aleatoric=0):
        if Y is not None:
            return torch.sin(torch.sqrt(X**2 + Y **2)) + aleatoric*(torch.rand(X.size())-0.5)
        else:
            return torch.sin(torch.sqrt(X[:,0]**2 + X[:,1]**2)) + aleatoric*(torch.rand(X.size()[0])-0.5)


    # Train
    x_train = in_dis * 2 * (torch.rand((n_train,2))-0.5)
    y_train = torch.reshape(f_2D(x_train,aleatoric=aleatoric_train),(n_train,1))


    # Test
    n_test_dim = int(np.sqrt(n_test))
    x1_test = torch.linspace(-out_dis,out_dis,n_test_dim)
    x2_test = torch.linspace(-out_dis,out_dis,n_test_dim)

    X1,X2 = torch.meshgrid(x1_test, x2_test)
    Z = f_2D(X1,X2,aleatoric_test)

    




################################################################################################################## 
# Architecture
###### 

n_hidden = 5

n_in = x_train.size(dim=1) #0st dim data, 1st dim Dimensions of Vector
n_out = y_train.size(dim=1)

# mu1 = torch.zeros((n_hidden,n_in))
# sigma1 = torch.ones((n_hidden, n_in))

# mu2 = torch.zeros((n_out, n_hidden))
# sigma2 = torch.ones((n_out, n_hidden))
mu1 = 0
sigma1 = 1

mu2 = 0
sigma2 = 1
#self, prior_mu, prior_sigma, in_features, out_features, bias=True

# What kind of Bayes_Layer
# finn: Bayes_Layer
# bnnG: BayesLinear
# mine: LinearBayes
# tuto: MeanFieldGaussianFeedForward
bayes = True

if bayes:
    model = nn.Sequential(MeanFieldGaussianFeedForward(n_in, n_hidden, weightPriorMean= mu1, weightPriorSigma=sigma1),
                        nn.Sigmoid(),
                        MeanFieldGaussianFeedForward(n_hidden, n_out, weightPriorMean = mu2, weightPriorSigma = sigma2))
else:
    model = nn.Sequential(nn.Linear(n_in, n_hidden),
                        nn.Sigmoid(),
                        nn.Linear(n_hidden,n_out))





##################################################################################################################
# Training
######


train_when_exist = False #If model exists already doesn't train it

## Test 
# n_dim
# n_train
# n_test

# in_dis   #Assumes symmetric distance and zero centering
# out_dis  #in_dis < out_dis

# aleatoric_train
# aleatoric_test


## Architecture
# n_hidden = 20

## Hyperparameters
epoch = 10000
lr = 0.01
loss_collection = []


# filename = "E10e6H200Lre-3.pth"
# filename = "bayes_TutoE10e4H20Lr10e-2Sigmoid.pth"
filename = '_'.join([str(n_dim)+'D','E'+str(epoch),'H'+str(n_hidden),'T'+str(n_train)]) + '.pth'


pathFolder = Path("./simpleReg/models/")
pathTo = os.path.join(pathFolder,filename)

if train_when_exist or not os.path.exists(pathTo):


    #############################
    ### V A R I A T I O N A L
    ###    I N F E R E N C E
    #############################


    
    


    mse_loss = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr)


    ## Training

    for step in range(1,epoch+1):
        #TODO: Implement DataLoader, Batches and Shuffle
        pred = model(x_train)
        loss = mse_loss(pred, y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % 100 == 0:
            print("\r",(f'\r Epoch {step}: Loss: {loss.item()}'),end="")
            loss_collection.append(loss.item())


    

    #############################
    ### M  C  M  C  
    #############################






    ### Training Plot
    plt.loglog(loss_collection)
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.show()


    
    torch.save(model.state_dict(), pathTo)
    # torch.save(loss_collection, pathFolder.joinpath("loss_collection.txt")) .txt doesnt work

else:
    model.load_state_dict(torch.load(pathTo))


##################################################################################################################
# Inference
######

model.eval()





if n_dim==1:
    y_pred = model(x_test)
    if bayes:
        draws = 20
        y_pred_avg = y_pred.detach().clone()
        for i in range(draws-1):
            y_pred_iter = model(x_test)
            # plt.plot(x_test, y_pred_iter.detach().numpy())
            y_pred_avg += y_pred_iter
        y_pred_avg = y_pred_avg/draws



    plt.figure(figsize=(10,5))
    plt.subplot(1,2,1)
    plt.plot(x_test, y_test)
    plt.title("Plot of the data")
    plt.subplot(1,2,2)
    plt.plot(x_test, y_test, 'go')
    plt.plot(x_test, f(x_test,0), 'bx')
    plt.plot(x_test, y_pred.detach().numpy(),'r--')
    plt.plot(x_test, y_pred_avg.detach().numpy())
    plt.title("Model Evaluation")
    plt.legend(['data with noise','data','single sample','average of 20 samples']) #20 magic number
    plt.show()

else:
    
    x_test = out_dis * 2 * (torch.rand((n_test,2))-0.5)
    y_test = torch.reshape(f_2D(x_test,aleatoric=aleatoric_test),(n_test,1))
    y_pred = model(x_test)
    X12 = torch.zeros((n_test,2))
    X12[:,0] = torch.flatten(X1)
    X12[:,1] = torch.flatten(X2)
    z_pred = model(X12)
    z_grid = torch.reshape(z_pred,(n_test_dim,n_test_dim)).detach().numpy()
    error = np.abs(z_grid - Z.numpy())

    fig = plt.figure(figsize=plt.figaspect(0.5)) #Plots a figure 2:1
    ax = fig.add_subplot(1,2,1,projection="3d")
    surf = ax.plot_surface(X1,X2,Z, cmap=cm.summer, linewidth=0, alpha = 0.7, label='function')
    scatter_z = ax.scatter(X12[:,0],X12[:,1],z_pred.detach().numpy(), label='prediction')
    surf_error = ax.plot_surface(X1,X2, error, cmap=cm.Reds, linewidth=0, alpha = 0.7, label='error')
    fig.colorbar(surf, shrink=0.5, aspect=5)
    # ax.legend(handles=[surf, scatter_z, surf_error])
    

    #Second Plot
    ax1 = fig.add_subplot(1,2,2,projection='3d')
    scatter_train = ax1.scatter(x_train[:,0],x_train[:,1], y_train, marker='o', label='train_data')
    scatter_test = ax1.scatter(x_test[:,0], x_test[:,1], y_test, marker = '^', label='test_data')
    scatter_pred = ax1.scatter(x_test[:,0], x_test[:,1], y_pred.detach().numpy(), marker = 'o',label='prediction')
    # ax1.legend(handles=[scatter_train,scatter_test,scatter_pred])
    
    plt.show()



##################################################################################################################
# Measure Uncertainty
######

#First mu, then sigma of dist
#Before I get fancy with this I first should handle general architectures and understand bnnLayers better
for param in model.parameters():
    print(param)









#Calibration Curve

# AUC