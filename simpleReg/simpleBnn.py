from pathlib import Path
import os

import numpy as np
import matplotlib.pyplot as plt


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist
from bnnLayer import BayesLinear



## Architecture
n_in = 1
n_hidden = 4
n_out = 1

# mu1 = torch.zeros((n_hidden,n_in))
# sigma1 = torch.ones((n_hidden, n_in))

# mu2 = torch.zeros((n_out, n_hidden))
# sigma2 = torch.ones((n_out, n_hidden))
mu1 = 0
sigma1 = 1

mu2 = 0
sigma2 = 1
#self, prior_mu, prior_sigma, in_features, out_features, bias=True

bayes = True

if bayes:
    model = nn.Sequential(BayesLinear(mu1, sigma1, n_in, n_hidden),
                        nn.ReLU(),
                        BayesLinear(mu2,sigma2, n_hidden, n_out))
else:
    model = nn.Sequential(nn.Linear(n_in, n_hidden),
                        nn.ReLU(),
                        nn.Linear(n_hidden,n_out))



## Create Dataset

n_train = 1000
n_test = 1000

def f(x):
    '''
    cubic polynomial R1 -> R1
    '''
    aleatoric_uncertainty = 0
    return torch.pow(x,4) - torch.pow(x,2) + 2 * torch.pow(x,1) + aleatoric_uncertainty*torch.rand(x.size())


x_train = torch.reshape(torch.linspace(-2,2,n_train),(n_train,1))
y_train = f(x_train)

x_test = torch.reshape(torch.linspace(-2.5,2.5,n_test),(n_test,1))
y_test = f(x_test)

# plt.figure(figsize=(10,5))
# plt.subplot(1,2,1)
# plt.scatter(x_train, y_train)
# plt.title("Scatter plot of the data")
# plt.subplot(1,2,2)
# plt.plot(x_train, y_train)
# plt.title("Training Data linear spline fit")
# plt.show()




train = True

# filename = "E10e6H200Lre-3.pth"
filename = "bayes_tiny.pth"


pathFolder = Path("./simpleReg/models/")
pathTo = os.path.join(pathFolder,filename)

if train:


    #############################
    ### V A R I A T I O N A L
    ###    I N F E R E N C E
    #############################


    ## Hyperparameters
    epoch = 10000
    lr = 0.01
    loss_collection = []


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
            print(f'Epoch {step}: Loss: {loss.item()}')
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

else:
    model.load_state_dict(torch.load(pathTo))


## Inference

model.eval()

y_pred = model(x_test)

if bayes:
    draws = 20
    for i in range(draws-1):
        y_pred += model(x_test)
    y_pred = y_pred/draws




plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.plot(x_test, y_test)
plt.title("Scatter plot of the data")
plt.subplot(1,2,2)
plt.plot(x_test, y_pred.detach().numpy())
plt.title("Model Evaluation")
plt.show()


### Measure Uncertainty











#Calibration Curve

# AUC