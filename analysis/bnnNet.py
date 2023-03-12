import numpy as np
import torch
from torch import nn

from bnnLayer import LinearBayes


class BayesianNet(nn.Module):
    def __init__(self, arc=[1,10,1], bayes_arc = [1], rho_w = -4, rho_b = None, elbo_hyper = None):
        super(BayesianNet, self).__init__()
        '''
        Generates a Bayesian network with customizable stochasticity  
        
        Required arguments:
            - architecture: _arc_
        
        Optional arguments:
                        
            - bayes_arc: Bayes architecture 
                    o  single value -> bayes_factor for all layers

                    o  shape -> speciefied bayes_factor for each layer 

                    - bayes_factor: Can be used to adjust how much percent of each layer is bayesian
                    o   Either takes value between 0 < bayes_factor < 1
                    o   Or integer numbers from 0 to n_out which indicate how many neurons have zero_variance
                    Admittingly a bit overused

            - rho: Initial variation of params
                    o TODO: rho_w \neq rho_b

        Always starts training in pretrain and must be turned off for Bayes learning
        
        '''
        #Initialize
        self.layers_n = len(arc)
        self.elbo_hyper = elbo_hyper
        self.activation_fn = torch.nn.Tanh()


        #Priors can be initialized differently
        if rho_b is None:
            rho_b = rho_w

        #Architectue
        self.arc=arc
        self.bayes_arc = self.assemble_bayes_arc(bayes_arc.copy())

        


        #creates a list with Linear Bayes layers of specified architectur
        layers = []
        for i in range(self.layers_n-1):
            layers.append(LinearBayes(arc[i],arc[i+1],
                            rho_w_prior = rho_w, 
                            rho_b_prior = rho_b, 
                            bayes_factor_w = self.bayes_arc[i][0],
                            bayes_factor_b = self.bayes_arc[i][1], 
                            pretrain=True))
                
        self.layers = nn.ModuleList(layers)
        

    #Functions for Init
    def assemble_bayes_arc(self, bayes_arc):
        '''
        Bayes layer needs to be of the form
        [ [0, 1], [1, 1], [0, 0] ]
            (if Architecture is of length 4 e.g. [1, 8, 8, 1])
        
        Method ensures this form -> Decodes Shortcuts
        [ [1], [1], [0.5] ] -> [ [1, 1], [1, 1], [0.5, 0.5]]

        [ [-1] ]            -> [ [0, 0], [0, 0], [-1, -1] ]

        [ [0.5] ]           -> [ [0.5,0.5], [0.5,0.5], [0.5,0.5] ]

        [ [0, -1] ]         -> [ [0, 0], [0, 0], [-1, -1] ]
        '''
        
        #For scalar extent to array
        if len(bayes_arc) == 1:
            
            #Checks if last layer special (and removes instruction which is the first entry)
            middle, last  = self.get_last_layer(bayes_arc)
                   
            for i in range(self.layers_n - 2):
                bayes_arc.append(middle)

            bayes_arc.append(last)
        
        #If only one entry is given extend it for both weight and bias
        if type(bayes_arc[0]) is not list:
            for i in range(len(bayes_arc)):
                bayes_arc[i] = [bayes_arc[i], bayes_arc[i]]
        
        return bayes_arc
        

    def get_last_layer(self, bayes_arc):
        '''
        If first entry is negative all the intermediate layers wont be Bayes

        Returns intermediate layer and last layer (the same if not neg) 
        
        Ensures: bayes_arc is empty
        '''
        if type(bayes_arc[0]) is not list:
            if bayes_arc[0] < 0:
                middle = 0
                last = - bayes_arc.pop(0)
            else:
                middle = bayes_arc.pop(0)
                last = middle
        else:
            if bayes_arc[0][0] < 0 or bayes_arc[0][1] < 0:
               middle = [0,0]
               last = bayes_arc.pop(0)
               last = [- last[0], - last[1]]
            else:
                middle = bayes_arc.pop(0)
                last = middle
        return middle, last

    #Training routines
    def sort_bias(self):
        previous_sort = torch.arange(0,self.arc[0]) #first sort is the identity
        for layer in self.layers:
            previous_sort = layer.sort_bias(previous_sort)

    def set_pretrain(self, pretrain):
        '''
        Sets a layer to pretrain
        '''
        for layer in self.layers:
            layer.activate_pretrain(pretrain)

    #ELBO loss
    def sample_elbo(self, x, y, samples = 5, noise = 0.1, kl_weight = 0.01):

        n = y.size()[0]
        log_likes = torch.empty((samples,n,1))
        log_priors = torch.empty((samples,))
        log_posts = torch.empty((samples,))
        noise = torch.ones_like(y) * noise
        for i in range(samples):
            pred = self.forward(x)
            log_likes[i] = torch.distributions.Normal(y, noise).log_prob(pred)
            log_priors[i] = self.log_prior()
            log_posts[i] = self.log_post()
        log_post = log_posts.mean()
        log_prior = log_priors.mean()
        log_like = log_likes.mean()

        elbo_loss = kl_weight * (log_post + log_prior) - log_like
        return elbo_loss


    def log_prior(self):
        '''
        Calculates the prior over all variational layers
            
            * sum (but could be average) 
        '''
        log_prior = 0
        for layer in self.layers:
            log_prior += layer.get_log_prior()
        return log_prior
    
    def log_post(self):
        '''
        Calculates the posterior over all variational layers
            
            * sum (but could be average) 
        '''
        log_post = 0
        for layer in self.layers:
            log_post += layer.get_log_post()
        return log_post

    #Forward routines
    def forward(self, x):
        
        for idx, layer in enumerate(self.layers):
            if idx < self.layers_n - 2: 
                x = self.activation_fn(layer(x))
            else: #last layer
                x = layer(x)

        return x