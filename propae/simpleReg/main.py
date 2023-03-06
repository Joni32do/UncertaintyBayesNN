'''
Here all jigsaw-pieces shall come together
Until I have achieved something meaningful I will mainly work in this file and not refactor in main and other sub files.
This might be ugly but better then create a good framework for bad code
'''
from pathlib import Path
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.stats import chi2

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist

from bnnLayer import *



##################################################################################################################
# Settings
#######
torch.manual_seed(42)


name = ""  #defaul ""

## Architecture

bayes = True
pretrain = True
train = "MC" #"MC" #Or 'VI'

draws = 100


n_hidden = 3

#Initial Value
mu1 = 0
sigma1 = 1
mu2 = 0
sigma2 = 1 #0.1



## Hyperparameters
epoch = 5000
preTrainingEpoch = 100000
n_chains = 10000


loss_fun = nn.MSELoss()
lr = 0.2




## Dataset

n_in = 1
option = 3

n_train = 100
n_test = 1000 #better if it is a square





in_dis_l = 0
in_dis_r = 1.5
out_dis_l = -0.5
out_dis_r = 2


#Not used for 1D
in_dis = 2 #Assumes symmetric distance and zero centering
out_dis = 2.5 #in_dis < out_dis


aleatoric_train = 0.1 #aleatoric uncertainty
aleatoric_test = 0.1





## File management
train_even_when_exist = True #If model exists already doesn't train it

# filename = '_'.join([str(n_in)+'D','O'+str(option),train,'E'+str(epoch),'H'+str(n_hidden),'T'+str(n_train)])
filename = '_'.join([train,'E'+str(epoch),'H'+str(n_hidden),'T'+str(n_train),name])
if bayes:
    filename = filename + "B"
if pretrain:
    filename = filename + "P"


pathModel = Path("./models/")
pathTo = os.path.join(pathModel,filename+'.pth')
pathTo_pre = os.path.join(pathModel,filename+'_pre.pth')




##################################################################################################################
# Dataset
######

#I use no DataLoader

if n_in == 1:

    ### 1D Regression Dataset

    #I could also discriminate like in 2D between equidistant und random

    def f(x,aleatoric=0,option=0):
        '''
        4th degree polynomial R1 -> R1
        sinc
        4th degree
        3rd degree
        '''
        unc = aleatoric*torch.randn(x.size())
        if option == 0:
            return torch.pow(x,4) - torch.pow(x,2) + 5 * torch.pow(x,1) + unc
        elif option == 1:
            return torch.sin(2*torch.pi *x)/x + unc
        elif option == 2:
            return torch.pow(x,4) - 3*torch.pow(x,2) + 1 + unc
        elif option == 3:
            return -2 * torch.pow(x,3) + 4 * torch.pow(x,2) -  torch.pow(x,1) + unc
        else:
            return torch.zeros(x.size())
        


    x_train = torch.reshape(torch.linspace(in_dis_l,in_dis_r,n_train),(n_train,1))
    y_train = f(x_train,aleatoric_train,option)

    x_test = torch.reshape(torch.linspace(out_dis_l,out_dis_r,n_test),(n_test,1))
    y_test = f(x_test,aleatoric_test,option).detach().numpy()

else:
    ### 2D Regression Dataset


    def f(X,Y=None,aleatoric=0,option=0):
        if Y is not None:
            return -torch.sin(torch.sqrt(X**2 + Y **2)) + aleatoric*(torch.rand(X.size())-0.5)
        else:
            return -torch.sin(torch.sqrt(X[:,0]**2 + X[:,1]**2)) + aleatoric*(torch.rand(X.size()[0])-0.5)


    # Train
    x_train = in_dis * 2 * (torch.rand((n_train,2))-0.5)
    y_train = torch.reshape(f(x_train,aleatoric=aleatoric_train),(n_train,1))


    # Test
    n_test_dim = int(np.sqrt(n_test))

    # Grid like data
    x1_test = torch.linspace(-out_dis,out_dis,n_test_dim)
    x2_test = torch.linspace(-out_dis,out_dis,n_test_dim)
    X1,X2 = torch.meshgrid(x1_test, x2_test)
    Z = f(X1,X2,aleatoric_test)

    # Random data
    x_test = out_dis * 2 * (torch.rand((n_test,2))-0.5)
    y_test = torch.reshape(f(x_test,aleatoric=aleatoric_test),(n_test,1)).detach().numpy()

    




################################################################################################################## 
# Architecture
###### 


n_out = y_train.size(dim=1) #always one
y_train.detach().numpy()






if pretrain or not bayes:
    model = nn.Sequential(nn.Linear(n_in, n_hidden),
                        nn.Sigmoid(),
                        nn.Linear(n_hidden,n_out))
    if os.path.exists(pathTo_pre):
        model.load_state_dict(torch.load(pathTo_pre))
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr)
        for step in range(preTrainingEpoch):
                pred = model(x_train)
                loss = loss_fun(pred, y_train)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                print("\r",f"Pretraining step  {step} / {preTrainingEpoch}")
        torch.save(model.state_dict(), pathTo_pre)
    pretrained_params = torch.nn.utils.parameters_to_vector(model.parameters())
    params = [param for param in model.parameters()]
    
      

if bayes and train == 'VI':
    if pretrain:
        model = nn.Sequential(LinearBayes(n_in, n_hidden, mu_w_init=params[0].detach(),sigma_w_init=sigma1,
                                            mu_b_init=params[1].detach(),sigma_b_init=sigma1),
                        nn.Sigmoid(),
                        LinearBayes(n_hidden, n_out, mu_w_init=params[2].detach(),sigma_w_init=sigma2,
                                            mu_b_init=params[3].detach(),sigma_b_init=sigma2))
    else:
        model = nn.Sequential(LinearBayes(n_in, n_hidden, mu_w_init=mu1,sigma_w_init=sigma1,
                                            mu_b_init=mu1,sigma_b_init=sigma1),
                        nn.Sigmoid(),
                        LinearBayes(n_hidden, n_out, mu_w_init=mu2,sigma_w_init=sigma2,
                                            mu_b_init=mu2,sigma_b_init=sigma2))




##################################################################################################################
# Training
######

markov_chain = None #ugly




doTrain = train_even_when_exist or not os.path.exists(pathTo) or train == 'MC'
print(doTrain)

if bayes:
    if doTrain:

        loss_collection = []
        
        #############################
        ### V A R I A T I O N A L
        ###    I N F E R E N C E
        #############################
        if train == 'VI':
            
            optimizer = torch.optim.Adam(model.parameters(),lr)
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9) #,verbose=True)

            for step in range(epoch):
               
                pred = model(x_train)
                loss = loss_fun(pred, y_train)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if step % 1000 == 0:
                    print("\r",(f'\r Epoch {step+1}: Loss: {loss.item()}'),end="")
                    loss_collection.append(loss.item())
                    scheduler.step()


        

        #############################
        ### M  C  M  C  
        #############################
        elif train == 'MC':
            model.eval()
            # Pretrained 0-1
            # Variational Bayes 0-1
            n_param = n_in * n_hidden + n_hidden + n_hidden * n_out + n_out #10/20B for 3H 
           

            
            
            theta_0 = torch.nn.utils.parameters_to_vector(model.parameters())

            print(theta_0)
            markov_chain = torch.zeros((n_param, n_chains))
            markov_chain[:,0] = theta_0

            step = 0
            tries = 0
            step_size = 0.1 * lr
             

            loss_old = loss_fun(model(x_train),y_train)
            print(loss_old)
            while step < n_chains-1:
                
                proposal = torch.normal(markov_chain[:,step],step_size)
                
                #Create model with proposal vector
                torch.nn.utils.vector_to_parameters(proposal, model.parameters())

                
                pred = model(x_train)
                loss_new = loss_fun(pred, y_train)
                
                #If loss_old and new are close acc is still high - simple approach is to use sqrt as bijective function
                acc = torch.log(loss_new)/torch.log(loss_old)
                print(acc, step/(tries+1), step)
                if acc >= 0.95 + 0.05*torch.rand(1): # torch.rand(1)
                    loss_old = loss_new
                    loss_collection.append(loss_new.detach().numpy())
                    print(step)
                    markov_chain[:,step+1] = proposal
                    step += 1
                tries += 1
                if (step+1)%100==0:
                    print(step+1)

            print(f"The acceptance rate is with {tries} tries and {step+1} steps: {step/tries}",end="")



        
        
        torch.save(model.state_dict(), pathTo)
        # torch.save(loss_collection, pathFolder.joinpath("loss_collection.txt")) .txt doesnt work

    else:
        model.load_state_dict(torch.load(pathTo))



##################################################################################################################
# Inference
######

model.eval()







def draw_from_dist_markov(markov_chain):
    '''with kernel density optimazition'''
    # pdf_KDE = lambda x: 0.0
    # for r in range(torch.size(markov_chain,1)):     #n_chains
    #     pdf_KDE = lambda x: pdf_KDE(x) + torch.norm..........

    #StackOverflow: https://stats.stackexchange.com/questions/43674/simple-sampling-method-for-a-kernel-density-estimator
    variance = 0.001
    index = np.random.randint(0,markov_chain.size(dim = 1))
    return torch.normal(markov_chain[:,index],variance)

def evalBayes(x, n_out, model, draws, train, markov_chain):
    n_data = x.size(dim=0)
    y_stoch = np.zeros((n_data, n_out, draws))
    if train == 'VI':
        for i in range(draws):
            y_stoch[:,:,i] = model(x).detach().numpy().copy()
    if train == 'MC':
        for i in range(draws):
            theta_i = draw_from_dist_markov(markov_chain)
            torch.nn.utils.vector_to_parameters(theta_i, model.parameters())
            y_stoch[:,:,i] = model(x).detach().numpy().copy()
    return y_stoch
    
#For stochastic Neural Networks - draws 10 times (pointless for non-stochastic NN)


# y_stoch_train, y_pred_avg_train = evalBayes(x_train, n_out, model, draws)
# error_train = np.abs(y_train-y_pred_avg_train)
#  std_train = np.std(y_stoch_train, axis=-1)
y_stoch_test = evalBayes(x_test, n_out, model, draws,train,markov_chain)


y_pred_avg_test = np.mean(y_stoch_test,axis=-1)
std_test = np.std(y_stoch_test,axis=-1)
error_test = np.abs(y_test-y_pred_avg_test)
y_test_no_noise = f(x_test,aleatoric=0,option=option).flatten()

#Quantile
quant = 0.025
upper = np.quantile(y_stoch_test,q = 1-quant,axis=-1).flatten()
lower = np.quantile(y_stoch_test,q = quant,axis=-1).flatten()
q_up_test = np.quantile(y_stoch_test - np.reshape(y_pred_avg_test,(1000,1,1)),q = 1-quant,axis=-1).flatten()
q_low_test = np.quantile(y_stoch_test - np.reshape(y_pred_avg_test,(1000,1,1)),q = quant,axis=-1).flatten()




if pretrain and train == 'MC':
        torch.nn.utils.vector_to_parameters(pretrained_params, model.parameters())
        y_pred_pretrain = model(x_test).detach().numpy().copy()


x_test = x_test.detach().numpy().copy().flatten()






    













### P L O T T I N G
pathFigure = os.path.join(Path("./figures/"),filename)
Path(pathFigure).mkdir(parents=True, exist_ok=True)

show_examples= 3 #int(np.ceil(np.log(draws)))



plt.style.use('ggplot')



### Training Plot
if bayes and doTrain:
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.plot(loss_collection)
    ax.set_yscale('log')
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.savefig(os.path.join(pathFigure,"loss.pdf"))


### 1D Plot
if n_in==1:
    #Plot function with data and margins
    """ fig = plt.figure(figsize=(8,5))
    axF = fig.add_subplot(1,1,1)
    axF.fill_between(x_test, 
        y_test_no_noise + 2* aleatoric_test, 
        y_test_no_noise - 2* aleatoric_test, 
        alpha = 0.5, linewidth = 0,color = 'lightsteelblue')
    axF.plot(x_test, y_test_no_noise, color = 'tab:blue',label='$f$')
    axF.scatter(x_test, y_test, s = 3, c = 'tab:green',label='$D_{test}$')
    axF.scatter(x_train, y_train, s = 10, c = 'tab:red',label='$D_{train}$')
   
    plt.legend()
    plt.xlim(-0.2,1.8)
    plt.ylim(-0.5,1.5)
    plt.savefig(os.path.join(pathFigure,"functionWithData.pdf")) """

    #Prediction Curves
    fig = plt.figure(figsize=(8,5))
    ax = fig.add_subplot(1,1,1)
    ax.fill_between(x_test, 
        y_test_no_noise + 2* aleatoric_test, 
        y_test_no_noise - 2* aleatoric_test, 
        alpha = 0.5, linewidth = 0,color = 'lightsteelblue')
    ax.plot(x_test,y_test_no_noise, linewidth = 2, label="$f$",color = 'tab:blue')
    
    ax.fill_between(x_test, (y_pred_avg_test - 2* std_test).flatten(),
                            y_pred_avg_test.flatten() + 2* std_test.flatten(),  
                            linewidth=0,alpha = 0.2,color = 'lightcoral')
    ax.fill_between(x_test, lower, upper, linewidth = 0, alpha = 0.2, color = 'mediumpurple')
    ax.plot(x_test, y_pred_avg_test, linewidth = 2, color = 'tab:red', label="$y_{avg}$")
    colorArr = ["slateblue","mediumpurple", "orchid","plum","lightsteelblue","slateblue","mediumpurple", "orchid","plum","lightsteelblue"]
    if bayes:
        for i in range(show_examples):
            ax.plot(x_test, y_stoch_test[:,:,i], linestyle='dashed',linewidth=1, color=colorArr[i],label="$y_{"+str(i+1)+"}$")
    # if pretrain and train == 'MC':
    #     ax.plot(x_test,y_pred_pretrain,color='tab:purple',label='pretrain')
    plt.legend()
    # plt.title("Prediction Curves")
    plt.xlim(-0.2,1.8)
    plt.ylim(-0.5,1.5)
    plt.savefig(os.path.join(pathFigure,"predictionCurves.pdf"))


    # #Prediction Error
    # fig = plt.figure(figsize=(5,5))
    # ax = fig.add_subplot(1,1,1)
    # ax.plot(x_train, error_train, label='train error')
    # ax.plot(x_test, error_test, label='test error')
    # ax.set_yscale('log')
    # plt.title("Prediction Error")
    # plt.legend()
    # plt.savefig(os.path.join(pathFigure,"predError.svg"))

    
   

    #Deviation of Prediction Curves
    fig = plt.figure(figsize=(8,5))
    ax = fig.add_subplot(1,1,1)
    for i in range(show_examples):
        ax.plot(x_test, y_stoch_test[:,:,i] - y_pred_avg_test,linewidth=1,linestyle='dashed') # label="Deviation from average " + str(i)
   
    ax.fill_between(x_test, 2*std_test.flatten(), -2*std_test.flatten(),alpha = 0.4, 
        linewidth = 0,color = 'lightcoral',label='$2\sigma$ region')
    ax.fill_between(x_test, q_up_test.flatten(), q_low_test.flatten(),alpha = 0.4, 
        linewidth = 0,color = 'plum',label='2.5% - 97.5% quantiles')
    plt.legend()
    # plt.title("Deviation from average for single draw")
    plt.savefig(os.path.join(pathFigure,"Deviation.pdf"))
    # plt.savefig(os.path.join(pathFigure,"Deviation.svg"))
    

    # params = [param for param in model.parameters()]
    # weight_sigma_last_layer = params[-3].detach().numpy()
 
    


    # #Calibration Curve
    # print(np.shape(std_test), np.shape(error_test))
    nssr = np.multiply(1/std_test, np.power(error_test,2))
    nssr = np.sort(nssr,axis=None)
    p_obs = np.linspace(1/n_test,1,n_test)
    p_pred = chi2.cdf(nssr,1)
    
    plt.figure("Calibration curve for sparse measure model")
    plt.plot(p_pred, p_obs, c='#ff7f0e', label='Calibration curve')
    # plt.scatter(p_pred,p_obs,s=2,c='#ff7f0e', label='points')
    plt.plot([0,1],[0,1], 'k--', alpha=0.5, label='Ideal curve')
    plt.xlabel('Predicted probability')
    plt.ylabel('Observed probability')
    plt.title("Calibration curve")
    plt.axis('equal')
    plt.xlim([0,1])
    plt.ylim([0,1])
    plt.legend()
    plt.savefig(os.path.join(pathFigure,'CalibrationCurve.pdf'))

### 2D - Plot
else:
  
    #TODO: Devide Square_Grid and Random data more clearly 
    X12 = torch.zeros((n_test,2))
    X12[:,0] = torch.flatten(X1)
    X12[:,1] = torch.flatten(X2)
    z_pred = model(X12)
    # z_pred = nn_model(X12)
    z_grid = torch.reshape(z_pred,(n_test_dim,n_test_dim)).detach().numpy()
    error_pred = np.abs(z_grid - Z.numpy())

    # Function
    fig = plt.figure(figsize=(5,5)) #Plots a figure 2:1
    ax = fig.add_subplot(1,1,1,projection="3d")
    surf = ax.plot_surface(X1,X2,Z, cmap=cm.summer, linewidth=0, alpha = 0.7, label='function')
    plt.savefig(os.path.join(pathFigure,"Function.svg"))

    # Prediction
    fig = plt.figure(figsize=(5,5)) #Plots a figure 2:1
    ax = fig.add_subplot(1,1,1,projection="3d")
    scatter_z = ax.plot_surface(X1,X2,np.reshape(z_pred.detach().numpy(),np.shape(X1)), label='prediction')
    plt.savefig(os.path.join(pathFigure,"Prediction.svg"))
    

    # Error
    fig = plt.figure(figsize=(5,5)) #Plots a figure 2:1
    ax = fig.add_subplot(1,1,1,projection="3d")
    surf_error = ax.plot_surface(X1,X2, error_pred, linewidth=0, cmap=cm.Reds, alpha = 0.7, label='error') #
    plt.savefig(os.path.join(pathFigure,"Error.svg"))
    # 
   
    
    ###### fig.colorbar(surf, shrink=0.5, aspect=5)
    # # ax.legend(handles=[surf, scatter_z, surf_error])
    plt.savefig(os.path.join(pathFigure,"Error.svg"))
    
 
    # #Second Plot
    # fig = plt.figure(figsize=(5,5))
    # ax1 = fig.add_subplot(1,2,2,projection='3d')
    # scatter_train = ax1.scatter(x_train[:,0],x_train[:,1], y_train, marker='o', label='train_data')
    # scatter_test = ax1.scatter(x_test[:,0], x_test[:,1], y_test, marker = '^', label='test_data')
    # scatter_pred = ax1.scatter(x_test[:,0], x_test[:,1], y_stoch_test[:,:,0], marker = 'o',label='prediction')
    # # ax1.legend(handles=[scatter_train,scatter_test,scatter_pred])
    
    plt.show()



##################################################################################################################
# Measure Uncertainty
######

#First mu, then sigma of dist
#Before I get fancy with this I first should handle general architectures and understand bnnLayers better

# params = [param for param in model.parameters()]
# weight_mu_last_layer = params[-4].detach().numpy()
# weight_sigma_last_layer = params[-3].detach().numpy()
# print(weight_mu_last_layer)
# print(weight_sigma_last_layer)
# print(params[-2].detach().numpy())
# print(params[-1].detach().numpy())




#Trash

    # #Std on x
    # fig = plt.figure(figsize=(5,5))

    # ax2 = fig.add_subplot(1,1,1)
    # ax2.plot(x_test, std_test, label="test std")
    # ax2.plot(x_train, std_train, label="train std")
    # ax2.set_yscale('log')
    # plt.legend()
    # plt.title("Standard deviation")
    # plt.savefig(os.path.join(pathFigure,"StandardDeviation.svg"))
    # # plt.show()




    #Some shit from MCMC approach
    #Find new proposal
                # proposal = model.state_dict()
                # for l in layerNames:
                #     proposal[l] = torch.normal(proposal[l], step_size)