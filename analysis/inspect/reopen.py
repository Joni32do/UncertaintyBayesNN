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
from visualize import plot_bayes, plot_retardation
from evaluate import eval_Bayes_net, calc_water

BAYES_ARC = [ [0, -1] ]

ARCHITECTURE = [1, 32, 1]


if __name__ == '__main__':
    #Name
    parser = argparse.ArgumentParser()
    parser.add_argument("-n","--name",default="debug", type=str,help="name of the folder where Analysis according to meta.json")
    args = parser.parse_args()

    #Read Parameters
    root =os.path.dirname(os.path.realpath(__file__))
    with open(os.path.join(root,"meta_params.json"), 'r') as param:
        meta = json.load(param)

    
    net = BayesianNet(ARCHITECTURE,BAYES_ARC, meta["training"]["rho"], init_pretrain = False)
    net.load_state_dict(torch.load("model.pth"))

    #Dataset
    #Calculate dataset and stores it in dictionary with the parameters
    data = generate_data(copy.deepcopy(meta["data"]) )

    
    #Evaluation
    y_preds,mean,lower,upper = eval_Bayes_net(net, data["x_eval"], data["n_samples"])
    wasserstein = calc_water(y_preds, data["y_eval"], evaluation_type="mat")
    print(wasserstein)
    plt.plot(np.linspace(0,1,len(wasserstein)),wasserstein)
    plot_path = os.path.join(root, args.name)
    plot_bayes(data, y_preds, mean, lower, upper, plot_path)
    plot_retardation(data, y_preds, mean, lower, upper, plot_path)

