import numpy as np
import torch
from torch import nn

from bnnLayer import LinearBayes


class BayesianNet(nn.Module):
    def __init__(self, arc=[1,10,1], bayes_factor = 0, bayes_arc = None):
        super(BayesianNet, self).__init__()
        '''
        Generates a fully Bayesian network  
        
        Required arguments:
            - architecture: _arc_
        
        Optional arguments:
            - bayes_factor: Can be used to adjust how much percent of each layer is bayesian
                    o   Either takes value between 0 < bayes_factor < 1
                    o   Or integer numbers from 0 to n_out which indicate how many neurons have zero_variance
                    o   If it has value -1 nn.Linear is used
                    Admittingly a bit overused
            
            - bayes_arc: Bayes architecture 
                    o   Directly invoke zero_variance each layer 
        
        
        '''
        self.arc=arc
        self.layers_num = len(arc)
        
        if bayes_arc is None:
            bayes_arc = np.ones(len(arc)) * bayes_factor 
            print(bayes_arc)



        layers = []
        for i in range(self.layers_num-1):
            layers.append(LinearBayes(arc[i],arc[i+1],
                            rho_w_prior = -4, rho_b_prior = -4, 
                            bayes_factor = bayes_arc[i+1], pretrain=True))
                
        self.layers = nn.ModuleList(layers)


    def sort_bias(self):
        previous_sort = torch.arange(0,self.arc[0]) #first sort is the identity
        for layer in self.layers:
            previous_sort = layer.sort_bias(previous_sort)

    def set_pretrain(self, pretrain):
        for layer in self.layers:
            layer.pretrain = pretrain

    def kl_loss(self, kl_weight):
        loss = 0
        for layer in self.layers:
            loss += layer.kl_loss(kl_weight)
        return loss
    
    def forward(self, x):
        
        for idx, layer in enumerate(self.layers):
            if idx < self.layers_num - 2: 
                x = torch.tanh(layer(x))
            else: #last layer
                x = layer(x)

        return x