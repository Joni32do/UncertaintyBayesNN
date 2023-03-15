#System
import os
import sys
import time
import copy
from pathlib import Path
import argparse

#Machine Learning
import torch

#Math
import numpy as np

#Plotting
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')

#Data
import pandas as pd


#Parameter json
import json



####Code
#Import custom function
from bnnNet import BayesianNet
from data import generate_data
from visualize import plot_bayes
from train import train_net
from evaluate import eval_Bayes_net, calc_water
'''
Testframework


'''


def experiment(arc, bayes_arc, trie, data, hyper, arc_path):
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
    path = os.path.join(arc_path, str(trie))

    #Network    
    net = BayesianNet(arc, bayes_arc, hyper["rho"])
            

    #Train
    final_loss = train_net(net, data, hyper, arc_path)

    #Save state dict
    torch.save(net.state_dict(), os.path.join(arc_path,"model.pth"))


    #Evaluation
    y_preds,mean,lower,upper = eval_Bayes_net(net, data["x_eval"], data["n_samples"])
    wasserstein = calc_water(y_preds, data["y_eval"])

    #Plotting
    plot_bayes(data, y_preds, mean, lower, upper, path)

    return final_loss, wasserstein



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-n","--name",default="debug", type=str,help="name of the folder where Analysis according to meta.json")
    args = parser.parse_args()
    #File managment >>>> ENTER  N A M E  <<<<
    description = args.name

    root =os.path.dirname(os.path.realpath(__file__))
    main_path = os.path.join(root,"meta_analysis", description)
    Path(main_path).mkdir(parents=True, exist_ok=True)

    #Load parameters from json
    with open(os.path.join(root,"meta.json"), 'r') as param:
        meta = json.load(param)
    
    
    


    ###Meta Analysis
    build = meta["build"]

    #Architectures
    architectures =  build["architectures"]
    #Bayes Architectur

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
        final_losses = np.zeros(s_cases)
        wasserstein = np.zeros(s_cases)

        


        ###Experiment loops

        #Architectures
        for i, arc in enumerate(architectures):

            #Stochasticy
            for j,bayes_arc in enumerate(bayes_arcs):
                print("Training architecture: ", arc, " with stochasticity ", bayes_arc, " model", (i*len(bayes_arcs)+j+1),"/",len(architectures)*len(bayes_arcs))

                #Directories
                path = os.path.join(main_path,"A_" + str(arc), "B_" + str(bayes_arc))
                Path(path).mkdir(parents=True, exist_ok=True)


                #Multiple experiments 
                for t in range(tries):
                    sys.stdout.write(f"\r \t Run: {t+1}/{tries}")
                    start = time.time()

                    l, w  = experiment(arc, bayes_arc, t, data, hyper, path)
                    
                    final_losses[i,j,t] = l
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
        final_losses = np.zeros(s_cases)
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
                
                final_losses[i,t] = m
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
    np.save(os.path.join(numpy_path, "final_losses"), final_losses)
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


    m_df = pd.DataFrame(np.median(final_losses,axis=-1),        index = str_arcs, columns= str_bayes_arcs)
    w_df = pd.DataFrame(np.median(wasserstein,axis=-1),index = str_arcs, columns= str_bayes_arcs)
    t_df = pd.DataFrame(np.median(training_time,axis=-1), index = str_arcs, columns= str_bayes_arcs)
 
   
    with pd.ExcelWriter(os.path.join(main_path, "results.xlsx"), engine='xlsxwriter') as writer:
        m_df.to_excel(writer,sheet_name='Final Losses')   
        w_df.to_excel(writer,sheet_name='Wasserstein')
        t_df.to_excel(writer,sheet_name='Training Time')
        
        ws_mse = writer.sheets['Final Losses']
        ws_mse.write_string(0,0,'Median')


'''

description = "X" + str(meta["data"]["n_train"]) + \
                  "_N" + str(meta["data"]["noise"])+ \
                  "_E" + str(meta["training"]["epochs"]) + \
                  "_Rho" + str(meta["training"]["rho"])
    if meta["data"]["is_log"]:
        description += "Log"

'''
