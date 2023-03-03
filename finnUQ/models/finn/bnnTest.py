#System
import os
import sys
import time
import copy
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


def generate_data(dic):
    '''
    generates data similar to retardation factor
    parameters
        bars - number of locations in input space
        samples - number of data per bar
        std - noise with deviation which scales with x
    '''
    #For training
    bars = dic["n_bars"]
    samples = dic["n_samples"]
    #For plotting
    n_plot = dic["n_plot"]
    #Number of training points
    n = bars * samples

    #lin or log
    if dic["bars_log"]:
        val = dic["log"]
        fn = torch.logspace
    else:
        val = dic["lin"]
        fn = torch.linspace
    
    #Calculate x
    x = fn(val["min"], val["max"],bars).reshape((bars,1))
    x_plot = fn(val["min"], val["max"],n_plot).reshape((n_plot,1))

    
    #Parameters 
    params = dic["params"]

    k_d = params["k_d"]
    beta = params["beta"]
    rho = params["rho"]
    n_e = params["n_e"]
    f = params["f"]
    a = (1-n_e)/n_e * rho * f * k_d * beta

    ###Calculate 1/R

    #For training
    y = torch.zeros((bars,1))
    y[x!=0] = 1/(1 + a * x[x!=0]**(beta-1))
    #For plotting
    y_plot = torch.zeros((n_plot,1))
    y_plot[x_plot!=0] = 1/(1 + a * x_plot[x_plot!=0]**(beta-1))

    
    ##Noise
    #TODO: Noise function: x
    noise = torch.randn((bars,samples)) * x * dic["noise"]
    
    #Automatch to shape (n,samples)
    y = y + noise
    y = torch.reshape(y,(n,1))

    #Match x to y
    x_repeat = x.repeat_interleave(samples,0)
    

    dic["x_true"] = x_plot
    dic["y_true"] = y_plot

    dic["x_single"] = x
    dic["x_train"] = x_repeat
    dic["y_train"] = y

    return dic  


def train_net(net, x_train, y_train, epochs, pretrain_epochs = 0, sort = False, logging = False):
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


def calc_water(y_preds, y_train, samples):
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

def create_fig(data, mean ,lower, upper, path = None):
    '''
    Plots results

        - parameters
            o data - Dictionary containing the data
            o mean - Bayes-Net mean
            o lower - Bayes-Net lower quantile (5%)
            o upper - Bayes-Net upper quantile (95%)
            o path - to store figure -> if None plot
    '''    
    fig = plt.figure(figsize=(8,5))

    ###Plot true function
    plt.scatter(data["x_train"],
                data["y_train"],
                s = 2 ,
                color = 'red',
                label="Train data")
    
      #Helper
    x = data["x_true"].squeeze()
    y = data["y_true"].squeeze()
      # 2 * sigma=95% -> more noise per x (see data_generation())
    q = 2 * x * data["noise"]

    plt.plot(x,y, label='Noiseless function')
    
    plt.fill_between(x, y-q, y+q, alpha=0.5, label='True distribution')




    ###Plot BNN
    plt.plot(data["x_single"], mean, label='Average Prediction')
    plt.fill_between(data["x_single"].squeeze(), 
                     lower.squeeze(),
                     upper.squeeze(), 
                     alpha=0.5, 
                     label='5%-95% Quantile')
    
    
    #Finalize
    plt.legend()
    if path is not None:
        plt.savefig(path + ".pdf")
    else:
        plt.show()
    plt.close()


def experiment(arc, bayes_arc, t, data, hyper, path):
    '''
        - Parameter
            o arc - Architecture
            o bayes_arc - Bayesian Architecture
            o t - number of try
            
            o data - dictionary containing data
            o hyper - dictionary containing hyperparameter for training

            o path - Path for storing a figure 
    
    
    '''
    #Location and name of file
    path_figure = os.path.join(path, str(t))

    #Network    
    net = BayesianNet(arc, bayes_arc, hyper["rho"])
            

    #Training
    mse = train_net(net, 
                    data["x_train"], 
                    data["y_train"], 
                    hyper["epochs"], 
                    hyper["pretrain_epochs"], 
                    hyper["sort"])
    #TODO: Sample with MCMC 
        #...


    #Evaluation
    y_preds,mean,lower,upper = eval_Bayes_net(net,
                                              data["x_single"],
                                              data["n_samples"])
    
    water = calc_water(y_preds, 
                       data["y_train"], 
                       data["n_samples"])


    #Plotting
    create_fig(data, mean, lower, upper, path_figure)

    return mse, water



if __name__ == '__main__':
    root = os.path.join(os.path.realpath(__file__), "..", "meta_analysis")
    print("This is the root ", root)

    #Load parameters from json
    with open(os.path.join(root,"meta.json"), 'r') as param:
        meta = json.load(param)
    
    
    #File managment
    description = "X" + str(meta["data"]["n_bars"]) + \
                  "_S" + str(meta["data"]["n_samples"]) + \
                  "_N" + str(meta["data"]["noise"])+ \
                  "_E" + str(meta["training"]["epochs"]) + \
                  "_P" + str(meta["training"]["pretrain_epochs"]) + \
                  "_Rho" + str(meta["training"]["rho"])
    if meta["data"]["bars_log"]:
        description += "Log"

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
    main_path = os.path.join(root, description)
    
    Path(main_path).mkdir(parents=True, exist_ok=True)

    ###Meta Analysis
    build = meta["build"]

    #Architectures
    architectures =  build["architectures"]
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
    bayes_arcs = build["bayes_arcs"]
    #Combis
    all_combis = build["all_combinations_possible"]
    #Multiple trainings with same params
    tries = build["tries"]
    

    #Dataset
    #   Calculate synthetic dataset and stores it in dictionary 
    #   with the parameters
    data_info = copy.deepcopy(meta["data"]) 
    data = generate_data(data_info)

    #Training hyperparams
    hyper = meta["training"]



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
                print("Training architecture: ", arc, " with stochasticity ", bayes_arc, " model", (i*len(bayes_arcs)+j+1),"/",len(architectures)*len(bayes_arcs))

                #Directories
                path = os.path.join(main_path,"A_" + str(arc), "B_" + str(bayes_arc))
                Path(path).mkdir(parents=True, exist_ok=True)


                #Multiple experiments 
                for t in range(tries):
                    sys.stdout.write(f"\r \t Run: {t+1}/{tries}")
                    start = time.time()

                    m, w  = experiment(arc, bayes_arc, t, data, hyper, path)
                    
                    mse[i,j,t] = m
                    wasserstein[i,j,t] = w
                    training_time[i,j,t] = time.time() - start
                
                
                print(f"\n    Architecture took: {(np.sum(training_time[i,j,:])):.3f} seconds")
                #(f"{((time.time()-start)/tries):.3f}")


    #Not all possible combinations
    else:
        assert len(architectures) == len(bayes_arcs)
        n_cases = len(architectures)

        for i in range(n_cases):
            #I could refactor and put here the same in as before
            pass





    
    #Final Logging
    print("\n \n \n")
    print("#############################")
    print("## T R A I N I N G   E N D ##")
    print("############################# \n ")
    print("Trained parameters:")
    print(description,"\n")

    total_time = np.sum(training_time)
    print(f"Total time: {total_time} \n")

    idx = np.unravel_index(np.argmin(wasserstein), wasserstein.shape)    
    print("The best model is: \n \t Architecture: ",architectures[idx[0]],"\n \t Bayes Archi:", bayes_arcs[idx[1]], "\n at try", idx[2])
    
    idx_avg = np.unravel_index(np.argmin(np.mean(wasserstein, axis=2)), wasserstein.shape) 
    print("The best average model is: \n \t Architecture: ",architectures[idx_avg[0]],"\n \t Bayes Archi:", bayes_arcs[idx_avg[1]])

    

    #Save numpy ndarrays
    np.save(os.path.join(main_path, "mse"), mse)
    np.save(os.path.join(main_path, "wasserstein"), wasserstein)
    np.save(os.path.join(main_path, "training_time"), training_time)
    



    # Saving the parameters from the meta.json file
    #TODO: Write number of parameters and final time
    meta["result"] = {"n_net": n_cases,
                      "time": total_time
                      }

    with open(os.path.join(main_path, "meta_params.json"), "w") as fp:
        json.dump(meta, fp)

    # Creating Excel Writer Object from Pandas  
    str_arcs = [str(i) for i in architectures]
    str_bayes_arcs = [str(i) for i in bayes_arcs]
    m_df = pd.DataFrame(np.median(mse,axis=2),        index = str_arcs, columns= str_bayes_arcs)
    w_df = pd.DataFrame(np.median(wasserstein,axis=2),index = str_arcs, columns= str_bayes_arcs)
    t_df = pd.DataFrame(np.sum(training_time,axis=2), index = str_arcs, columns= str_bayes_arcs)
 
    
    with pd.ExcelWriter(os.path.join(main_path, "results.xlsx"), engine='xlsxwriter') as writer:
        m_df.to_excel(writer,sheet_name='Mean Squared Error')   
        w_df.to_excel(writer,sheet_name='Wasserstein')
        t_df.to_excel(writer,sheet_name='Training Time')
        
        ws_mse = writer.sheets['Mean Squared Error']
        ws_mse.write_string(0,0,'Median')


'''

Procrastination: Training von NN mit Progress Bar

def start_progress(title):
    global progress_x
    sys.stdout.write(title + ": [" + "-"*40 + "]" + chr(8)*41)
    sys.stdout.flush()
    progress_x = 0

def progress(x):
    global progress_x
    x = int(x * 40 // 100)
    sys.stdout.write("#" * (x - progress_x))
    sys.stdout.flush()
    progress_x = x

def end_progress():
    sys.stdout.write("#" * (40 - progress_x) + "]\n")
    sys.stdout.flush()


'''