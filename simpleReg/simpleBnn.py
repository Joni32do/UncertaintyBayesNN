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
This might be ugly but better then create a good framework for bad code
'''

##################################################################################################################
# Uncertainty
#######
torch.manual_seed(42)






##################################################################################################################
# Dataset
######

n_dim = 1
option = 0

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

    def f(x,aleatoric=0,option=0):
        '''
        cubic polynomial R1 -> R1

        '''
        unc = aleatoric*(torch.rand(x.size())-0.5)
        if option == 0:
            return torch.pow(x,4) - torch.pow(x,2) + 5 * torch.pow(x,1) + unc
        elif option == 1:
            return torch.sin(2*torch.pi *x)/x + unc
        else:
            return torch.zeros(x.size())
        


    x_train = torch.reshape(torch.linspace(-in_dis,in_dis,n_train),(n_train,1))
    y_train = f(x_train,aleatoric_train,option)

    x_test = torch.reshape(torch.linspace(-out_dis,out_dis,n_test),(n_test,1))
    y_test = f(x_test,aleatoric_test,option).detach().numpy()
else:
    ### 2D Regression Dataset


    def f_2D(X,Y=None,aleatoric=0):
        if Y is not None:
            return torch.sin(torch.sqrt(X**2 + Y **2)) + aleatoric*(torch.rand(X.size())-0.5)
        else:
            return torch.sin(torch.sqrt(X[:,0]**2 + X[:,1]**2)) + aleatoric*(torch.rand(X.size()[0])-0.5)


    # Train
    x_train = in_dis * 2 * (torch.rand((n_train,2))-0.5)
    y_train = torch.reshape(f_2D(x_train,aleatoric=aleatoric_train),(n_train,1)).detach().numpy()


    # Test
    n_test_dim = int(np.sqrt(n_test))

    # Grid like data
    x1_test = torch.linspace(-out_dis,out_dis,n_test_dim)
    x2_test = torch.linspace(-out_dis,out_dis,n_test_dim)
    X1,X2 = torch.meshgrid(x1_test, x2_test)
    Z = f_2D(X1,X2,aleatoric_test)

    # Random data
    x_test = out_dis * 2 * (torch.rand((n_test,2))-0.5)
    y_test = torch.reshape(f_2D(x_test,aleatoric=aleatoric_test),(n_test,1)).detach().numpy()

    




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
filename = '_'.join([str(option),str(n_dim)+'D','E'+str(epoch),'H'+str(n_hidden),'T'+str(n_train)]) + '.pth'


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


#For stochastic Neural Networks - draws 20 times (pointless for non-stochastic NN)
draws = 20

y_stoch = np.zeros((n_test,n_out,draws))
for i in range(draws-1):
    y_stoch[:,:,i] = model(x_test).detach().numpy()
y_pred_avg = np.mean(y_stoch,axis=-1)

error_pred = np.abs(y_test - y_pred_avg)
print(error_pred)
error_max = np.max(error_pred)
error_mean = np.mean(error_pred)
error_draw = y_stoch - y_pred_avg #Autocast-to Tensor ^3
print(error_pred, error_max, error_mean, error_draw)
#Sigma Hat
# s = 1/(draws - 1) * error_draw@error_draw.T
# sigma_hat = torch.linalg.inv(s)
# nssr = torch.matmul()

if n_dim==1:


    plt.figure(figsize=(10,5))
    plt.subplot(1,2,1)
    plt.plot(x_test, y_test)
    plt.title("Plot of the data")
    plt.subplot(1,2,2)
    plt.plot(x_test, y_test, 'go')
    plt.plot(x_test, f(x_test,0), 'bx')
    plt.plot(x_test, y_stoch[:,:,0],'r--')
    plt.plot(x_test, y_pred_avg)
    plt.title("Model Evaluation")
    plt.legend(['data with noise','data','single sample','average of 20 samples']) #20 magic number
    plt.show()

else:
  
    #TODO: Devide Square_Grid and Random data more clearly 
    X12 = torch.zeros((n_test,2))
    X12[:,0] = torch.flatten(X1)
    X12[:,1] = torch.flatten(X2)
    z_pred = model(X12)
    z_grid = torch.reshape(z_pred,(n_test_dim,n_test_dim)).detach().numpy()
    error_pred = np.abs(z_grid - Z.numpy())

    # First Plot
    fig = plt.figure(figsize=plt.figaspect(0.5)) #Plots a figure 2:1
    ax = fig.add_subplot(1,2,1,projection="3d")
    surf = ax.plot_surface(X1,X2,Z, cmap=cm.summer, linewidth=0, alpha = 0.7, label='function')
    scatter_z = ax.scatter(X12[:,0],X12[:,1],z_pred.detach().numpy(), label='prediction')
    surf_error = ax.plot_surface(X1,X2, error_pred, cmap=cm.Reds, linewidth=0, alpha = 0.7, label='error')
    fig.colorbar(surf, shrink=0.5, aspect=5)
    # ax.legend(handles=[surf, scatter_z, surf_error])
    

    #Second Plot
    ax1 = fig.add_subplot(1,2,2,projection='3d')
    scatter_train = ax1.scatter(x_train[:,0],x_train[:,1], y_train, marker='o', label='train_data')
    scatter_test = ax1.scatter(x_test[:,0], x_test[:,1], y_test, marker = '^', label='test_data')
    scatter_pred = ax1.scatter(x_test[:,0], x_test[:,1], y_stoch[:,:,0].detach().numpy(), marker = 'o',label='prediction')
    # ax1.legend(handles=[scatter_train,scatter_test,scatter_pred])
    
    plt.show()



##################################################################################################################
# Measure Uncertainty
######

#First mu, then sigma of dist
#Before I get fancy with this I first should handle general architectures and understand bnnLayers better

params = [param for param in model.parameters()]
weight_mu_last_layer = params[-4].detach().numpy()
weight_sigma_last_layer = np.exp(params[5].detach().numpy())
print(np.argmax(weight_sigma_last_layer))


# fig = plt.figure()
# x = np.linspace(-10,10,1000)
# for mu, sigma in enumerate((weight_mu_last_layer,weight_sigma_last_layer)):
#     plt.plot(x, np.normal(mu,sigma))











#Calibration Curve

# num_bins = 100
# # the histogram of the actual error distribution
# n, bins, patches = ax.hist(errors_np.flatten(), num_bins, density=True, label='Observed histogram')
# ax.set_xlabel('Actual prediction error')
# ax.set_ylabel('Probability density')

# covs = stochasticPreds - predictions

# covs = torch.matmul(covs, torch.transpose(covs, 1, 2))/(args.nruntests-1.)
# weigths = torch.linalg.inv(covs) #

# nssr = torch.matmul(errors[:,np.newaxis,:], torch.matmul(weigths, errors[:,:,np.newaxis]))
# nssr = nssr.cpu().numpy().flatten()
# nssr = np.sort(nssr)
# p_obs = np.linspace(1./nssr.size,1.0,nssr.size)
# p_pred = Chi2Dist.cdf(nssr, 9);