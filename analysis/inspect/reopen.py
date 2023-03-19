'''
This script opens a trained network and replots and evaluates it

'''


import numpy as np
import torch

import matplotlib.pyplot as plt

import os
import sys
import json
import argparse
import copy

sys.path.append(os.path.pardir)

from bnnNet import BayesianNet
from data import generate_data
from visualize import plot_bayes, plot_retardation, plot_pretrain, plot_pretrain_retardation
from evaluate import eval_Bayes_net, calc_water

PRETRAIN = False
BAYES_ARC = [1]
# ARCHITECTURE = [1, 32, 1]
ARCHITECTURE = [1, 8,4,8, 1]
# ARCHITECTURE = [1, 4,9,4, 1]
MODEL_NAME = "FINAL.pth"


if __name__ == '__main__':
    #Name
    parser = argparse.ArgumentParser()
    parser.add_argument("-n","--name",default="debug", type=str,help="name of the folder where Analysis according to meta.json")
    args = parser.parse_args()

    #Read Parameters
    root =os.path.dirname(os.path.realpath(__file__))
    with open(os.path.join(root,"meta_params.json"), 'r') as param:
        meta = json.load(param)
    plot_path = os.path.join(root, args.name)
    
    #Load with global parameters
    net = BayesianNet(ARCHITECTURE,BAYES_ARC, meta["training"]["rho"], init_pretrain = PRETRAIN)
    net.load_state_dict(torch.load(MODEL_NAME)) #This can go wrong, if sort Bias is true
    net.eval()
    
    #Dataset
    data = generate_data(copy.deepcopy(meta["data"]) )

    if PRETRAIN:
        pred = net.forward(data["x_train"]).detach().numpy()
        plot_pretrain(data, pred, plot_path)
        plot_pretrain_retardation(data, pred, plot_path)
    else:
        #Evaluation
        y_preds,mean,lower,upper = eval_Bayes_net(net, data["x_eval"], data["n_samples"])
        wasserstein = calc_water(y_preds, data["y_eval"], evaluation_type="mat")
        print(wasserstein)
        plt.plot(np.linspace(0,1,len(wasserstein)),wasserstein)
        plt.show()
        
        plot_bayes(data, y_preds, mean, lower, upper, plot_path)
        plot_retardation(data, y_preds, mean, lower, upper, plot_path)
    

