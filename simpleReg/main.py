'''
Here all jigsaw-pieces shall come together
Until I have achieved something meaningful I will mainly work in this file and not refactor in main and other sub files.
This might be ugly but better then create a good framework for bad code
'''
from pathlib import Path
import os
import datetime

import numpy as np
import matplotlib.pyplot as plt


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist
from bnnLayer import *
from matplotlib import cm



##################################################################################################################
# Uncertainty
#######
torch.manual_seed(42)






##################################################################################################################
# Dataset
######

#I use no DataLoader

n_dim = 1
option = 2

n_train = 20
n_test = 400 #better if it is a square

in_dis = 2 #Assumes symmetric distance and zero centering
out_dis = 2.5 #in_dis < out_dis

aleatoric_train = 0.01 #aleatoric uncertainty
aleatoric_test = 0.1

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
        elif option == 2:
            return torch.pow(x,4) - 3*torch.pow(x,2) + 1 + unc
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
    y_train = torch.reshape(f_2D(x_train,aleatoric=aleatoric_train),(n_train,1))


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

n_hidden = 10

n_in = x_train.size(dim=1) #0st dim data, 1st dim Dimensions of Vector
n_out = y_train.size(dim=1)
y_train.detach().numpy()

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
    model = nn.Sequential(LinearBayes(n_in, n_hidden, mu1, sigma1),
                        nn.Sigmoid(),
                        LinearBayes(n_hidden, n_out, mu2, sigma2))
else:
    model = nn.Sequential(nn.Linear(n_in, n_hidden),
                        nn.Sigmoid(),
                        nn.Linear(n_hidden,n_out))





##################################################################################################################
# Training
######


train_when_exist = False #If model exists already doesn't train it



## Hyperparameters
epoch = 10000
mse_loss = nn.MSELoss()
loss_collection = []
# train = "MCMC" #Or 'VI'
train = "VI"

# filename = "E10e6H200Lre-3.pth"
# filename = "bayes_TutoE10e4H20Lr10e-2Sigmoid.pth"
filename = '_'.join([str(n_dim)+'D','O'+str(option),train,'E'+str(epoch),'H'+str(n_hidden),'T'+str(n_train)])


pathModel = Path("./simpleReg/models/")
pathTo = os.path.join(pathModel,filename+'.pth')

if train_when_exist or not os.path.exists(pathTo):



    #############################
    ### V A R I A T I O N A L
    ###    I N F E R E N C E
    #############################
    if train == 'VI':
        lr = 0.1
        optimizer = torch.optim.Adam(model.parameters(), lr)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

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
                scheduler.step()


    

    #############################
    ### M  C  M  C  
    #############################
    elif train == 'MC':

        def step():
            pass

        sample = []
        theta_0 = torch.nn.utils.parameters_to_vector(model.parameters())
        
        theta_old = theta_0
        loss_old = np.infty
        step_size = 0.1

        theta_new = torch.normal(theta_old, step_size)
        state_dict["2.sigma_b"] = torch.rand((1,1)) 

        # 0.mu_w
        # 0.sigma_w
        # 0.mu_b
        # 0.sigma_b
        # 2.mu_w
        # 2.sigma_w
        # 2.mu_b
        # 2.sigma_b

        pred = model(x_train)
        loss_new = mse_loss(pred, y_train)
        acceptance = loss_old/loss_new
        if acceptance >= 1 or acceptance > np.random.uniform(0,1):
            loss_old = loss_new
            theta_old = theta_new
            sample.append(theta_new)




    ### Training Plot
    plt.loglog(loss_collection)
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.savefig(os.path.join(pathModel,filename+"loss.svg"))

    
    torch.save(model.state_dict(), pathTo)
    # torch.save(loss_collection, pathFolder.joinpath("loss_collection.txt")) .txt doesnt work

else:
    model.load_state_dict(torch.load(pathTo))


##################################################################################################################
# Inference
######

model.eval()


#For stochastic Neural Networks - draws 20 times (pointless for non-stochastic NN)
draws = 10

y_stoch_test = np.zeros((n_test,n_out,draws))
y_stoch_train = np.zeros((n_train,n_out,draws))
for i in range(draws):
    y_stoch_test[:,:,i] = model(x_test).detach().numpy().copy()
    y_stoch_train[:,:,i] = model(x_train).detach().numpy().copy()
y_pred_avg_test = np.mean(y_stoch_test,axis=-1)

y_pred_avg_train = np.mean(y_stoch_train,axis=-1)

#TODO:
# error_pred = np.abs(y_test - y_pred_avg)
# error_max = np.max(error_pred)
# error_mean = np.mean(error_pred)
# error_draw = y_stoch - y_pred_avg #Autocast-to Tensor ^3
#Sigma Hat
# s = 1/(draws - 1) * error_draw@error_draw.T
# sigma_hat = torch.linalg.inv(s)
# nssr = torch.matmul()

if n_dim==1:
    show_examples=5
    pathFigure = os.path.join(Path("./simpleReg/figures/"),filename)
    Path(pathFigure).mkdir(parents=True, exist_ok=True)

    #Prediction Error
    fig = plt.figure(figsize=(5,5))
    ax = fig.add_subplot(1,1,1)
    ax.plot(x_train, np.abs(y_train-y_pred_avg_train), label='train error')
    ax.plot(x_test, np.abs(y_test-y_pred_avg_test), label='test error')
    ax.set_yscale('log')
    plt.title("Prediction Error")

    plt.legend()
    plt.savefig(os.path.join(pathFigure,"predError.svg"))

    #Prediction Curves
    plt.figure(figsize=(5,5))
    plt.plot(x_test, y_test, label="Function with noise")
    for i in range(show_examples):
        plt.plot(x_test, y_stoch_test[:,:,i], label="Prediction "+str(i+1))
    plt.plot(x_test, y_pred_avg_test, label="Average of Predictions")
    plt.legend()
    plt.title("Prediction Curves")
    plt.savefig(os.path.join(pathFigure,"predictionCurves.svg"))

    #Deviation of Prediction Curves
    plt.figure(figsize=(5,5))
    for i in range(draws):
        plt.plot(x_test, y_stoch_test[:,:,i] - y_pred_avg_test, label="Deviation from average " + str(i))
    plt.legend()
    plt.title("Single Draw Deviation")
    plt.savefig(os.path.join(pathFigure,"Deviation.svg"))
    
    #Aleatoric Noise
    plt.figure(figsize=(5,5))
    #aleatoric*(torch.rand(x.size())-0.5)
    plt.scatter(x_test.detach().numpy(),y_test-f(x_test,0,option).detach().numpy(),label='noise test')
    plt.scatter(x_train, y_train-f(x_train,0,option),label='noise train')
    plt.title("noise")
    plt.legend()
    plt.savefig(os.path.join(pathFigure,"noise.svg"))

    #Variance
    fig = plt.figure(figsize=(5,5))
    ax1 = fig.add_subplot(2,1,1)
    params = [param for param in model.parameters()]
    weight_mu_last_layer = params[-4].detach().numpy()
    ax1.hist(weight_mu_last_layer)
    plt.title("Histogram of sigma of last layer")
    ax2 = fig.add_subplot(2,1,2)
    ax2.plot(x_test, np.std(y_stoch_test,axis=-1), label="test std")
    ax2.plot(x_train, np.std(y_stoch_train, axis=-1), label="train std")
    plt.title("Standard deviation")
    plt.legend()
    plt.savefig(os.path.join(pathFigure,"StandardDeviation"))
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
    scatter_pred = ax1.scatter(x_test[:,0], x_test[:,1], y_stoch_test[:,:,0], marker = 'o',label='prediction')
    # ax1.legend(handles=[scatter_train,scatter_test,scatter_pred])
    
    plt.show()



##################################################################################################################
# Measure Uncertainty
######

#First mu, then sigma of dist
#Before I get fancy with this I first should handle general architectures and understand bnnLayers better

params = [param for param in model.parameters()]
weight_mu_last_layer = params[-4].detach().numpy()
print(weight_mu_last_layer)
print(model.parameters())
'''print(model.state_dict())
model.state_dict["2.sigma_b"] = torch.rand((1,1)) 
print(model.state_dict())'''

# weight_sigma_last_layer = np.exp(params[5].detach().numpy())
# print(np.argmax(weight_sigma_last_layer))


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