#System
import os
import sys
import time
from pathlib import Path

#Machine Learning
import torch
from torch import nn
from bnnLayer import LinearBayes
from bnnNet import BayesianNet

#Math
import numpy as np
from scipy.stats import wasserstein_distance

#Plotting
import matplotlib.pyplot as plt

#Data
import pandas as pd
import xlsxwriter

#Import from upper directory
sys.path.append("..")
sys.path.append("..")

#Parameter json
import json



'''
Testframework


x_train[::samples] reverses np.repeat

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
    
    #Calculate 1/R with 
    k_d = 1.5
    beta = 0.7
    rho = 1.58
    n_e = 0.4
    f = 0.3

    a = (1-n_e)/n_e * rho * f * k_d * beta
    y = torch.zeros((bars,1))
    y[x!=0] = 1/(1 + a * x[x!=0]**(beta-1))
    y = y + noise
    y = torch.reshape(y,(n,1))
    x = x.repeat_interleave(samples,0)


    return x,y  


def train_net(net, epochs, x_train, y_train, pretrain_epochs = 0, sort = False, logging = True):
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
        if logging and (epoch + 1) % 100 == 0:
            print(f"\t \t Epoch {str(epoch + 1).rjust(len(str(epochs)),'0')}/{epochs}: Loss = {mse:.4f}")

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
    bars = int(len(y_train)/samples)
    water = np.zeros(bars)
    for i in range(bars):
        train_bar = y_train[i * samples:(i+1) * samples,0]
        pred_bar = y_preds[:,i]
        water[i] = wasserstein_distance(train_bar, pred_bar)
    return np.mean(water)
    
    #wasserstein_distance

def create_fig(x_train, y_train,std_train, mean ,lower, upper, path = None):
    '''
    Plots results
    '''
    x_test, y_true = generate_data(bars=200,samples = 1,noise = 0, x_min = 0, x_max = 1)
    samples = int(len(x_train)/len(mean))
    
    fig = plt.figure(figsize=(8,5))
    plt.scatter(x_train, y_train,s = 2 ,color = 'red',label="Train data")
    plt.plot(x_test,y_true, label='Noiseless function')
    plt.fill_between(x_test.squeeze(), (y_true - x_test * 2 * std_train).squeeze(), (y_true+ x_test * 2 * std_train).squeeze(), 
                     alpha=0.5, label='True distribution')
    plt.plot(x_train[::samples], mean, label='Average Prediction')
    plt.fill_between(x_train[::samples].squeeze(), lower.squeeze(),upper.squeeze(), alpha=0.5, label='5%-95% Quantile')
    plt.legend()
    if path is not None:
        plt.savefig(path + ".pdf")
    else:
        plt.show()
    plt.close()


def experiment(dict, arc, bayes_arc, t,  path):
    ##From dictionary
    #Parameters training
    training = dict["training"]

    epochs = training["epochs"]
    pretrain_epochs = training["pretrain_epochs"]
    rho = training["prior_rho"]
    sort = training["sort"]

    #Parameter data
    data = dict["data"]

    bars = data["n_bars"]
    samples = data["n_samples"]
    noise = data["noise"]
    x_min = data["x_min"]
    x_max = data["x_max"]
    #####



    #Location and name of file
    path_figure = os.path.join(path, str(t))
       
    #Data (cheap - takes 0.0001s)
    x_train,y_train = generate_data(bars, samples ,noise, x_min, x_max)

    #Network    
    net = BayesianNet(arc, bayes_arc, rho)
            

    #Training
    mse = train_net(net,epochs, x_train, y_train, pretrain_epochs, sort)
    #TODO:
    #Sample with 
    

    #Evaluation
    y_preds,mean,lower,upper = eval_Bayes_net(net,x_train[::samples],samples)
    water = calc_water(y_train, y_preds, samples)

    #Plotting
    create_fig(x_train, y_train, noise, mean, lower, upper, path_figure)

    return mse , water



if __name__ == '__main__':
    root = os.path.join(os.path.realpath(__file__), "..", "meta_analysis")
    print("This is the root ", root)

    #Load parameters from json
    with open(os.path.join(root,"meta.json"), 'r') as param:
        meta = json.load(param)
    
    print(meta["training"]["sort"])
    #File managment
    description = "Bar_" + str(meta["data"]["n_bars"]) + \
                  "S_" + str(meta["data"]["n_samples"]) + \
                  "Std_" + str(meta["data"]["noise"])+ \
                  "E_" + str(meta["training"]["epochs"]) + \
                  "P_" + str(meta["training"]["pretrain_epochs"]) + \
                  "Rho_" + str(meta["training"]["prior_rho"]) + \
                  "Sort" if meta["training"]["sort"] else ""
    '''
    Other influences:
        - Training
            o Learning rate
            o Additional KL - Loss
        - bnnLayer
            o Initial Values
                * mu
                * rho_w, rho_b

    '''   
    #
    main_path = os.path.join(root, description) #If empty plots functions
    Path(main_path).mkdir(parents=True, exist_ok=True)

    ###Meta Analysis

    #Architectures
    architectures =  meta["build"]["architectures"]
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
    bayes_arcs = meta["build"]["bayes_arcs"]
    #Combis
    all_combis = meta["build"]["all_combinations_possible"]
    #Multiple trainings with same params
    tries = meta["build"]["tries"]
    

    ###### Experiment
    if all_combis:

        ### Meassurements
        n_cases = len(architectures) * len(bayes_arcs) * tries
        s_cases = (len(architectures),len(bayes_arcs),tries)
        
        #Time
        training_time = np.zeros(s_cases)
        
        #Error meassure
        mse = np.zeros(s_cases)
        wasserstein = np.zeros(s_cases)


        ###Experiment loops

        #Stochasticy
        for i, arc in enumerate(architectures):

            #Architectures
            for j,bayes_arc in enumerate(bayes_arcs):
                print("Training architecture: ", arc, " with stochasticity ", bayes_arc, " model", (i*len(architectures)+j+1),"/",len(architectures)*len(bayes_arcs))

                #Directories
                path = os.path.join(main_path,"A_" + str(arc), "B_" + str(bayes_arc))
                Path(path).mkdir(parents=True, exist_ok=True)


                #Multiple experiments 
                for t in range(tries):
                    print("\t Try: ", t+1, "/", tries)
                    start = time.time()

                    m, w  = experiment(meta, arc, bayes_arc, t, path)
                    
                    mse[i,j,t] = m
                    wasserstein[i,j,t] = w
                    training_time[i,j,t] = time.time() - start
                
                
                print(f"  Architecture took: {(np.sum(training_time[i,j,:])):.3f} seconds")
                #(f"{((time.time()-start)/tries):.3f}")


    #Not all possible combinations
    else:
        assert len(architectures) == len(bayes_arcs)
        n = len(architectures)

        for i in range(n):
            #I could refactor and put here the same in as before
            pass



    #Final Logging
    print("\n \n \n")
    print("#############################")
    print("## T R A I N I N G   E N D ##")
    print("#############################")
    print("\n Trained parameters:")
    print(description,"\n")
    print(f"Average time per model: {np.around(np.mean(training_time,axis=2).flatten(),3)}")
    print(f"Total time per model: {np.sum(training_time,axis=2).flatten()}")
    print()
    print("All Wasserstein errors:")
    print(wasserstein)
    print()

    idx = np.unravel_index(np.argmin(wasserstein), wasserstein.shape)    
    print("The best model is: \n \t Architecture: ",architectures[idx[0]],"\n \t Bayes Archi:", bayes_arcs[idx[1]], "\n at try", idx[2])
    
    idx = np.unravel_index(np.argmin(np.mean(wasserstein, axis=2)), wasserstein.shape) 
    print("The best average model is: \n \t Architecture: ",architectures[idx[0]],"\n \t Bayes Archi:", bayes_arcs[idx[1]])

    print(np.median(mse,axis = 2))


    #Save numpy ndarrays
    np.save(os.path.join(main_path, "mse"), mse)
    np.save(os.path.join(main_path, "wasserstein"), wasserstein)
    
    # Saving the parameters from the meta.json file
    with open(os.path.join(main_path, "meta_params.json"), "w") as fp:
        json.dump(meta, fp)

    # Creating Excel Writer Object from Pandas  
    str_arcs = [str(i) for i in architectures]
    str_bayes_arcs = [str(i) for i in bayes_arcs]
    m_df = pd.DataFrame(data = np.mean(mse,axis=2), index = str_arcs, columns= str_bayes_arcs)
    w_df = pd.DataFrame(np.mean(wasserstein,axis=2), index = str_arcs, columns= str_bayes_arcs)
 
    
    with pd.ExcelWriter(os.path.join(main_path, "results.xlsx"), engine='xlsxwriter') as writer:
        m_df.to_excel(writer,sheet_name='Mean Squared Error')   
        w_df.to_excel(writer,sheet_name='Wasserstein')
