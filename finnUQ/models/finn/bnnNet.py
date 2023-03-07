import numpy as np
import torch
from torch import nn

from bnnLayer import LinearBayes


class BayesianNet(nn.Module):
    def __init__(self, arc=[1,10,1], bayes_arc = [1], rho = -4, rho_w = None, rho_b = None):
        super(BayesianNet, self).__init__()
        '''
        Generates a fully Bayesian network  
        
        Required arguments:
            - architecture: _arc_
        
        Optional arguments:
                        
            - bayes_arc: Bayes architecture 
                    o  single value -> bayes_factor for all layers

                    o  shape -> speciefied bayes_factor for each layer 

                    - bayes_factor: Can be used to adjust how much percent of each layer is bayesian
                    o   Either takes value between 0 < bayes_factor < 1
                    o   Or integer numbers from 0 to n_out which indicate how many neurons have zero_variance
                    o   If it has value -1 nn.Linear is used
                    Admittingly a bit overused

            - rho: Initial variation of params
                    o TODO: rho_w \neq rho_b

        Always starts training in pretrain and must be turned off for Bayes learning
        
        '''
        #Initialize
        self.arc=arc
        self.layers_n = len(arc)
        self.activation_fn = torch.nn.Tanh()
        
        #For scalar extent to array
        if len(bayes_arc) == 1:
            bayes_arc = np.ones(len(arc)) * bayes_arc[0]

        #Ugly
        if rho_w is None:
            rho_w = rho
        if rho_b is None:
            rho_b = rho


        #creates a list with Linear Bayes layers of specified architectur
        layers = []
        for i in range(self.layers_n-1):
            layers.append(LinearBayes(arc[i],arc[i+1],
                            rho_w_prior = rho_w, rho_b_prior = rho_b, 
                            bayes_factor = bayes_arc[i+1], pretrain=True))
                
        self.layers = nn.ModuleList(layers)
        
        #Always starts in pretrain mode
        self.set_pretrain(True)


    def sort_bias(self):
        previous_sort = torch.arange(0,self.arc[0]) #first sort is the identity
        for layer in self.layers:
            previous_sort = layer.sort_bias(previous_sort)

    def set_pretrain(self, pretrain):
        for layer in self.layers:
            layer.pretrain = pretrain

    def kl_loss(self):
        loss = 0
        for layer in self.layers:
            loss += layer.kl_loss()
        return loss
    
    def forward(self, x):
        
        for idx, layer in enumerate(self.layers):
            if idx < self.layers_n - 2: 
                x = self.activation_fn(layer(x))
            else: #last layer
                x = layer(x)

        return x