import torch
from torch import nn

import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance

import matplotlib.pyplot as plt


import os
import time
from pathlib import Path


from bnnLayer import LinearBayes
from bnnNet import BayesianNet

'''
Testframework

Doesn't really mind to much about Exception handeling and other because it 
anyways is only a product of time

'''

np.random.seed(42)


def generate_data(bars, samples, noise = 0.001, x_min = 0.001 , x_max = 1):
    '''
    generates data similar to retardation factor
    parameters
        bars - number of locations in input space
        samples - number of data per bar
        std - noise with deviation which scales with x
    '''
    n = bars * samples
    x = torch.linspace(x_min,x_max, bars).reshape((bars,1))

    #Noise
    noise = torch.randn((bars,samples)) * x * noise
    
    #Automatch to shape (n,samples)
    k_d = 1.5
    beta = 0.7
    rho = 1.58
    n_e = 0.4
    f = 0.3

    factor = rho/n_e * f * k_d * beta #
    #Null ausschließen
    y = 1 +factor * x**(beta-1) + noise
    y = torch.reshape(y,(n,1))
    x = x.repeat_interleave(samples,1)
    return x,y


def train_net(net, epochs, x_train, y_train, pretrain_epochs = 0, sort = False):
    '''
    Trains a Bayesian network (def line 16)
        - lr = 0.001
    '''
    
    # Define the loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.01)
    net.set_pretrain(True)

    def closure():
        optimizer.zero_grad()

        # Forward pass
        output = net(x_train)
        
        # Compute the loss
        mse_loss = criterion(output, y_train)
        
        # Compute the KL divergence loss for the Bayesian self.layers TODO:
        kl_weight = 0.0
        kl_divergence_loss = net.kl_loss(kl_weight)

        # Backward pass
        loss = mse_loss + kl_divergence_loss
        loss.backward()
        # (loss + kl_divergence_loss).backward()
        return mse_loss.item()





    # Train the net for 1000 epochs
    for epoch in range(epochs):
        # Change from pretrain to train
        if epoch == pretrain_epochs:
            net.set_pretrain(False)


        mse = optimizer.step(closure)
        if sort:
            net.sort_bias()

        # Print the loss every 100 epochs
        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch + 1}: Loss = {mse:.4f}")

            # for m in net.layers:
            #     print(m.rho_w)


    return mse


def eval_Bayes_net(net, x, samples, quantile = 0.05):
    # Evaluate function using the Bayesian network   
    y_preds = np.zeros((samples,x.size(dim=0)))
    for i in range(samples):
        y_preds[i,:] = net.forward(torch.Tensor(x).unsqueeze(1)).detach().numpy().flatten()
         
    # Calculate mean and quantiles
    mean = np.mean(y_preds, axis=0)
    lower = np.quantile(y_preds, quantile, axis=0)
    upper = np.quantile(y_preds, 1-quantile, axis=0)
    return y_preds,mean,lower,upper


def calc_water(y_train, y_preds, samples):
    '''
    calculates for each bar (single 1D coordinate of input space) with samples the wasserstein_distance 
    
    return
        -average of wasserstein over all bars
    
    '''
    bars = len(y_train)/samples
    water = np.zeros(bars)
    for i in range(bars):
        train_bar = y_train[i * samples, (i+1) * samples]
        pred_bar = y_preds[:,i]
        water[i] = wasserstein_distance(train_bar, pred_bar)
    return np.mean(water)
    
    #wasserstein_distance

def create_fig(x_train, y_train, x_test, y_true,std_train, mean ,lower, upper, path = None):
    # Plot results
    fig = plt.figure(figsize=(8,5))
    plt.scatter(x_train, y_train,s = 2 ,color = 'red',label="Train data")
    plt.plot(x_test,y_true, label='Noiseless function')
    plt.fill_between(x_test.squeeze(), (y_true - x_test * 2 * std_train).squeeze(), (y_true+ x_test * 2 * std_train).squeeze(), 
                     alpha=0.5, label='True distribution')
    plt.plot(x_test, mean, label='Average Prediction')
    plt.fill_between(x_test.squeeze(), lower.squeeze(),upper.squeeze(), alpha=0.5, label='5%-95% Quantile')
    plt.legend()
    if path is not None:
        plt.savefig(path + ".pdf")
    else:
        plt.show()


def experiment(epochs, pretrain_epochs, arc, bayes_arc, 
               t, samples, x_train, y_train, noise, x_test, y_true, path):
    
    #Location and name of file
    
    pathFigure = os.path.join(path, str(t))
    
    #Network    
    net = BayesianNet(arc, bayes_arc)
                
    #Training
    mse = train_net(net,epochs, x_train, y_train, pretrain_epochs, sort)
    
    #Evaluation
    y_preds,mean,lower,upper = eval_Bayes_net(net,x_test,samples)
    water = calc_water(y_train, y_preds)

    #Plotting
    create_fig(x_train, y_train, x_test, y_true, noise, mean, lower, upper, pathFigure)

    return np.around(mse, 4), water



if __name__ == '__main__':
    
    ### Parameters
    #Dataset
    bars = 10
    # Samples from Function and BNN
    samples = 10
    noise = 0.1 #Is increasing with distance from origin see data
    

    #Training
    epochs = 1000
    
    #Extras
    pretrain_epochs = 500
    sort = True




    #Data
    x_train,y_train = generate_data(bars, samples ,noise, 0.1,1)
    x_test, y_true = generate_data(bars= 200,samples = 1,noise = 0, x_min = 0.1, x_max = 1)
    plt.scatter(x_train, y_train, s = 1, c = 'r')
    plt.plot(x_test,y_true)
    plt.show()


    
    #Documentation
    descr = "Bar_" + str(bars) + "Std_" + str(noise) +  "E_" + str(epochs) + "P_" + str(pretrain_epochs) + "Sort" if sort else ""
    '''
    Other influences:
        - Training
            o Learning rate
            o Additional KL - Loss
        - 

    '''   


    ###Meta Analysis

    #Architectures
    architectures =  [[1, 8, 8, 1], #96
                      [1, 8, 8, 1], #97
                      [1, 4, 9, 4, 1], #97
                      [1, 8, 4, 8, 1]] #98
    
    #Bayes Architectur
    '''
    This is the main investigation of my bachelor thesis:

    Description of bayes_arc is given in bnnNet


    Some examples:

        Horizontal:
            b_arc = 0.5

        Vertical:
            b_arc = [0, 0, 1, 0]

        Proportional:
            b_arc = [0, 0.4, 0.6, 1]

        Neurons:
            b_arc = [0, 6, 6, 1]

        List:
            b_arc =  [[0, 0, 0, 1], 
                      [0, 0, 8, 0], 
                      [0, 0, 9, 0, 0], 
                      [0, 0, 4, 0, 0]] 


        #TODO: What is not yet possible is to describe only the last layer, because that is dependent on how many layers an architecture has

        (Note: In principle the input can't be Bayesian therefore it is always ignored)



    If only specific Networks shall be learned and compared
        o all_combinations_possible = False
    	
    
    '''
    bayes_arcs = [[0.5]]
    all_combinations_possible = True

 
    #Multiple trainings with same params
    tries = 100
    


    ###### Experiment
    if all_combinations_possible:

        ### Meassurements
        n_cases = len(bayes_arcs) * len(architectures) * tries
        s_cases = (len(bayes_arcs),len(architectures),tries)
        
        #Time
        training_time = []
        
        #Error meassure
        mse = np.zeros(s_cases)
        wasserstein = np.zeros(s_cases)


        #File managment
        main_path = os.path.join(Path("./meta/"),descr) #If empty plots functions
        Path(main_path).mkdir(parents=True, exist_ok=True)







        ###Experiment loops

        #Stochasticy
        for i, arc in enumerate(architectures):

            #Architectures
            for j,bayes_arc in enumerate(bayes_arcs):
                print("Training architecture: ", arc, bayes_arc)

                #Folder
                path = os.path.join(main_path,"A_" + str(arc) + "B_" + str(bayes_arc))
                Path(path).mkdir(parents=True, exist_ok=True)


                #Multiple experiments 
                start = time.time()
                for t in range(tries):

                    m, w  = experiment(epochs, pretrain_epochs, arc, bayes_arc, 
                                    t, samples, x_train, y_train, noise, x_test, y_true, path)
                    mse[i,j,t] = m
                    wasserstein[i,j,t] = w
                
                
                training_time[i,j] = time.time() - start
                #(f"{((time.time()-start)/tries):.3f}")


    #Not all possible combinations
    else:
        assert len(architectures) == len(bayes_arcs)
        n = len(architectures)

        for i in range(n):
            #I could refactor and put here the same in as before
            pass



    #Final Logging
    print(descr)
    print(f"Average training time per model: {training_time}")
    print(mse)
    print(np.median(mse,axis = 1))


