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
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')
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
    generatedata similar to retardation factor to 
    adds
        - training
        - evaluation (with Bayes methods)
        - plotting
    data to dictionary

    parameters
        data distribution
            o lin or log
            o min and max value
        for data types 
            - training
                o n samples
            - evaluation
                o bars - number of x locations
                o samples - number of data per bar
        noise
            o noise factor
                -> is input in noise function which scales with x
    '''
    
    ### Get number of samples
    #For training
    n = dic["n_train"]

    #For evaluating
    bars = dic["n_bars"]
    samples = dic["n_samples"]

    #For plotting
    n_plot = dic["n_plot"]
    

    ###Distribute and calculate
    # lin or log
    if dic["is_log"]:
        val = dic["log"]
        fn = torch.logspace
    else:
        val = dic["lin"]
        fn = torch.linspace
    
    #Calculate x
    x_train = fn(val["min"], val["max"], n).reshape((n,1))
    x_eval = fn(val["min"], val["max"],bars).reshape((bars,1))
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
    y_train = torch.zeros((n,1))
    y_train[x_train!=0] = 1/(1 + a * x_train[x_train!=0]**(beta-1))

    #For evaluating
    y_eval_mean = torch.zeros((bars,1))
    y_eval_mean[x_eval!=0] = 1/(1 + a * x_eval[x_eval!=0]**(beta-1))

    #For plotting
    y_plot = torch.zeros((n_plot,1))
    y_plot[x_plot!=0] = 1/(1 + a * x_plot[x_plot!=0]**(beta-1))

    
    ### Noise
    noise_fn = lambda x: dic["noise"] #x * dic["noise"]

    #Train
    noise_train = torch.randn((n,1)) * noise_fn(x_train)
    y_train = y_train + noise_train
    
    #Eval
    noise_eval = torch.randn((bars,samples)) * noise_fn(x_eval)
    y_eval = y_eval_mean + noise_eval  #Automatch to shape (bars,samples)

    #Saving to dictionary
    dic["x_train"] = x_train
    dic["y_train"] = y_train

    dic["x_eval"] = x_eval
    dic["y_eval_mean"] = y_eval_mean
    dic["y_eval"] = y_eval

    dic["x_true"] = x_plot
    dic["y_true"] = y_plot
    
    return dic  



def train_net(net, x_train, y_train, hyper, path):
    '''
    Trains a Bayesian network (def line 16)
        - lr = 0.001
    '''
    #Logging
    logging = hyper["logging"]

    #Hyperparameters
    
    epochs = hyper["epochs"]
    pretrain_epochs = hyper["pretrain_epochs"]
    lr = hyper["learning_rate"]
    sort = hyper["sort"]



    # Define the loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr)
    errors = []
    
    def closure():
        optimizer.zero_grad()

        # Forward pass
        output = net(x_train)
        
        # Compute the loss
        mse_loss = criterion(output, y_train)
        
        # Compute the KL divergence loss for the Bayesian self.layers TODO:
        kl_weight = 0
        kl_divergence_loss = net.kl_loss() * kl_weight

        # Backward pass
        loss = mse_loss
        loss.backward()
        # (loss + kl_divergence_loss).backward()
        return mse_loss.item()





    # Train the net for 1000 epochs
    for epoch in range(epochs):
        
        # Change from pretrain to train
        if epoch == pretrain_epochs:
            net.set_pretrain(False)


        mse = optimizer.step(closure)
        errors.append(mse)
        if sort:
            net.sort_bias()

        # Print the loss every 100 epochs
        if logging and (epoch + 1) % 100 == 0:
            print(f"\t \t Epoch {str(epoch + 1).rjust(len(str(epochs)),'0')}/{epochs}: Loss = {mse:.4f}")

            # for m in net.layers:
            #     print(m.mu_b.data)
            #     print(m.rho_w.data)

    if logging:
        plt.plot(errors)
        plt.yscale('log')
        plt.savefig(path + "train.pdf")
        # plt.close()



def eval_Bayes_net(net, x_eval, samples, quantile = 0.025):
    '''
    Evaluates the input {x_eval} for {samples} times
    (bars, samples)
    '''  
    bars = x_eval.size(dim = 0)
    y_preds = np.zeros((bars, samples))
    for i in range(samples):
        y_preds[:,i] = net.forward(x_eval).detach().numpy().flatten()
         
    # Calculate mean and quantiles
    mean = np.mean(y_preds, axis=1)
    lower = np.quantile(y_preds, quantile, axis=1)
    upper = np.quantile(y_preds, 1-quantile, axis=1)
    return y_preds,mean,lower,upper

def calc_water(y_preds, y_eval):
    '''
    calculates for each bar (single 1D coordinate of input space) with samples the wasserstein_distance 
    
    return
        -average of wasserstein over all bars
    
    '''
    assert np.shape(y_preds) == np.shape(y_eval)
    
    bars, samples = np.shape(y_preds)
    water = np.zeros(bars)
    for i in range(bars):
        water[i] = wasserstein_distance(y_preds[i,:], y_eval[i,:])
    return np.mean(water)

def create_fig(data, y_preds, mean ,lower, upper, path = None):
    '''
    Plots results

        - parameters
            o data - Dictionary containing the data
            o mean - Bayes-Net mean
            o lower - Bayes-Net lower quantile (5%)
            o upper - Bayes-Net upper quantile (95%)
            o path - to store figure -> if None plot
    '''    
    fig,ax = plt.subplots(1,1,figsize=(8,5))
    if data["is_log"]:
        ax.set_xscale('log')

    ###Plot true function
    
      #Helper
    x = data["x_true"].squeeze()
    y = data["y_true"].squeeze()
      # 2 * sigma=95% -> more noise per x (see data_generation())
    q = 2 * data["noise"]

    plt.plot(x,y, label='Noiseless function')
    
    plt.fill_between(x, y-q, y+q, alpha=0.5, label='True distribution')




    ###Plot BNN
    plt.plot(data["x_eval"], mean, label='Average Prediction')
    plt.fill_between(data["x_eval"].squeeze(), 
                     lower.squeeze(),
                     upper.squeeze(), 
                     alpha=0.5, 
                     label='2.5%-97.5% Quantile')
    
    
    #Scatter
    plt.scatter(data["x_train"],
                data["y_train"],
                s = 2 ,
                color = 'green',
                label="Train data")
    
    
    x_rep = data["x_eval"].repeat_interleave(data["n_samples"],0)
    side_slide = 0.002
    
    if data["is_log"]:
        side_slide = 0.05 * x_rep
    plt.scatter(x_rep,
                torch.reshape(data["y_eval"],(data["n_samples"] * data["n_bars"],1)),
                s = 1 ,
                color = 'blue',
                label="stoch. dist.")
    
    plt.scatter(x_rep + side_slide,
                np.reshape(y_preds,(data["n_samples"] * data["n_bars"],1)),
                s = 1 ,
                color = 'orange',
                label="stoch. dist. BNN")



    #Finalize
    plt.legend()
    if path is not None:
        plt.savefig(path + ".pdf")
    else:
        plt.show()
    plt.close()




def experiment(arc, bayes_arc, t, data, hyper, arc_path):
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
    path = os.path.join(arc_path, str(t))

    #Network    
    net = BayesianNet(arc, bayes_arc, hyper["rho"])
            

    #Training
    train_net(net, data["x_train"], data["y_train"], hyper, path)

    #Save state dict
    torch.save(net.state_dict(), path + "model.pth")


    #Evaluation
    y_preds,mean,lower,upper = eval_Bayes_net(net, data["x_eval"], data["n_samples"])
    loss_fn = nn.MSELoss() 
    mse = loss_fn(torch.tensor(mean),data["y_eval_mean"].squeeze())
    water = calc_water(y_preds, data["y_eval"])

    #Plotting
    create_fig(data, y_preds, mean, lower, upper, path)

    return mse, water



if __name__ == '__main__':
    root = os.path.join(os.path.dirname(os.path.realpath(__file__)), "meta_analysis")
    print("This is the root ", root)

    #Load parameters from json
    with open(os.path.join(root,"meta.json"), 'r') as param:
        meta = json.load(param)
    
    
    #File managment
    description = "X" + str(meta["data"]["n_train"]) + \
                  "_N" + str(meta["data"]["noise"])+ \
                  "_E" + str(meta["training"]["epochs"]) + \
                  "_Rho" + str(meta["training"]["rho"])
    if meta["data"]["is_log"]:
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
                
                #TODO: Logging
                plt.close()
                print(f"\n    Architecture took: {(np.sum(training_time[i,j,:])):.3f} seconds")


    #Not all possible combinations
    else:
        assert len(architectures) == len(bayes_arcs)
        n_cases = len(architectures)
        s_cases = (n_cases, tries)
        
        #Time
        training_time = np.zeros(s_cases)
        
        #Error meassure
        mse = np.zeros(s_cases)
        wasserstein = np.zeros(s_cases)


        for i in range(n_cases):
            arc = architectures[i]
            bayes_arc = bayes_arcs[i]
            print(f"Training architecture: {arc} with stochasticity {bayes_arc} model {i+1}/{n_cases}")

            #Directories
            path = os.path.join(main_path,"A_" + str(arc), "B_" + str(bayes_arc))
            Path(path).mkdir(parents=True, exist_ok=True)


            #Multiple experiments 
            for t in range(tries):
                sys.stdout.write(f"\r \t Run: {t+1}/{tries}")
                start = time.time()

                m, w  = experiment(arc, bayes_arc, t, data, hyper, path)
                
                mse[i,t] = m
                wasserstein[i,t] = w
                training_time[i,t] = time.time() - start
            
            
            print(f"\n    Architecture took: {(np.sum(training_time[i,:])):.3f} seconds")





    
    #Final Logging
    print("\n \n \n")
    print("#############################")
    print("## T R A I N I N G   E N D ##")
    print("############################# \n ")
    print("Trained parameters:")
    print(description,"\n")

    total_time = np.sum(training_time)
    print(f"Total time: {np.around(total_time,0)} seconds -> {np.around(total_time/3600,1)} hours \n")

    
    
    idx = np.unravel_index(np.argmin(wasserstein), wasserstein.shape)    
    idx_avg = np.unravel_index(np.argmin(np.mean(wasserstein, axis=-1)), wasserstein.shape) 
    if all_combis:
        print("The best model is: \n \t Architecture: ",architectures[idx[0]],"\n \t Bayes Archi:", bayes_arcs[idx[1]], "\n at try", idx[2])
        print("The best average model is: \n \t Architecture: ",architectures[idx_avg[0]],"\n \t Bayes Archi:", bayes_arcs[idx_avg[1]])
    else:
        print(f"The best model is {idx_avg[:-1]} at try {idx_avg[-1]}")

    

    #Save numpy ndarrays
    numpy_path = os.path.join(main_path,"results")
    Path(numpy_path).mkdir(parents=True, exist_ok=True)
    np.save(os.path.join(numpy_path, "mse"), mse)
    np.save(os.path.join(numpy_path, "wasserstein"), wasserstein)
    np.save(os.path.join(numpy_path, "training_time"), training_time)
    



    # Saving the parameters from the meta.json file
    #TODO: Write number of parameters
    meta["result"] = {"n_net": n_cases,
                      "time": total_time
                      }

    with open(os.path.join(main_path, "meta_params.json"), "w") as fp:
        json.dump(meta, fp)


    # Creating Excel Writer Object from Pandas  
    str_arcs = [str(i) for i in architectures]
    
    if all_combis:
        str_bayes_arcs = [str(i) for i in bayes_arcs]
    else:
        str_bayes_arcs = ["experiment"]
        for i, b_arc in enumerate(bayes_arcs):
            str_arcs[i] = str_arcs[i] + str(b_arc)


    m_df = pd.DataFrame(np.median(mse,axis=-1),        index = str_arcs, columns= str_bayes_arcs)
    w_df = pd.DataFrame(np.median(wasserstein,axis=-1),index = str_arcs, columns= str_bayes_arcs)
    t_df = pd.DataFrame(np.sum(training_time,axis=-1), index = str_arcs, columns= str_bayes_arcs)
 
    
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
