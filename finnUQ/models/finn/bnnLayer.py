import math
import numpy as np
import matplotlib.pyplot as plt


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist
from torch.distributions import Normal
from torch.nn.parameter import Parameter



#My own which does work 
class LinearBayes(nn.Module):
    """
    Implementation of a Bayesian linear layer with Variational Inference (VI)
    Requires:
        - Input size (n_in)
        - Output size (n_out)
        - Initial values for weights and biases with:
   
            - matrices with fitting dimensions (e.g. from a pretrained linear model)
            - scalar values

            - mu is mean value
            - rho is related to std with sigma = log(1 + e^{\rho})

    """
    def __init__(self, n_in, n_out, mu_w_prior=0,rho_w_prior=-1, mu_b_prior=0, rho_b_prior=-1, 
                 bayes_factor_w = 1, bayes_factor_b = 1, pretrain = False):
        
        super(LinearBayes, self).__init__()
        
        #Mode
        self.pretrain = pretrain
        
        #Initialization from scalar
        #   Kaiman init for weights
        init_scale = 1/torch.sqrt(torch.tensor(n_in))
        if not torch.is_tensor(mu_w_prior):
            mu_w_prior = mu_w_prior + init_scale *(torch.rand(n_out, n_in) - 0.5)
        if not torch.is_tensor(rho_w_prior):
            rho_w_prior = rho_w_prior*torch.ones(n_out,n_in)
        if  not torch.is_tensor(mu_b_prior):
            mu_b_prior = mu_b_prior +  torch.zeros(n_out)
        if not torch.is_tensor(rho_b_prior):
            rho_b_prior = rho_b_prior*torch.ones(n_out)

        #Size
        self.n_in = n_in
        self.n_out = n_out

        #Priors
        self.prior_sigma = 0.1
        self.mu_w_prior = Normal(mu_w_prior, self.prior_sigma)
        self.mu_b_prior = Normal(mu_b_prior, self.prior_sigma)
        self.rho_w_prior = rho_w_prior
        self.rho_b_prior = rho_b_prior




        #Initialize distribution as priors
        #Weights
        self.mu_w = nn.Parameter(mu_w_prior)
        self.rho_w = nn.Parameter(rho_w_prior)
        self.weights = None
        #Bias
        self.mu_b = nn.Parameter(mu_b_prior)
        self.rho_b = nn.Parameter(rho_b_prior)
        self.bias = None    

        #Noise
        self.noise_w = Normal(torch.zeros_like(self.mu_w), torch.ones_like(self.mu_w))
        self.noise_b = Normal(torch.zeros_like(self.mu_b), torch.ones_like(self.mu_b))

        
        ##No noise    -->  sparse Bayes  <--
          #Conversion:
        #Proportional to Neuron (special case: 1 Neuron ->  1/n_out [1 is understood as fully bayesian])
        if bayes_factor_w <= 1:
            bayes_factor_w = math.ceil(bayes_factor_w * n_out)
        if bayes_factor_b <= 1:
            bayes_factor_b = math.ceil(bayes_factor_b * n_out)
        
        #Assign logical arrays
        self.is_bayes_w = torch.zeros(n_out,1)
        self.is_bayes_b = torch.zeros(n_out)
        if bayes_factor_w != 0:
            self.is_bayes_w[:bayes_factor_w] = 1
        if bayes_factor_b != 0:
            self.is_bayes_b[:bayes_factor_b] = 1
        self.sampled_params = torch.sum(self.is_bayes_b) + torch.sum(self.is_bayes_w)


         

    def sample(self, stochastic = True):
        '''
        Samples from the neurons where is_bayes
        '''
        self.weights = self.mu_w  + (torch.log(1 + torch.exp(self.rho_w)) * self.noise_w.sample() 
                                     * self.is_bayes_w if stochastic else 0)
        
        self.bias = self.mu_b + (torch.log( 1 + torch.exp(self.rho_b)) * self.noise_b.sample() 
                                 * self.is_bayes_b if stochastic else 0)

    def activate_pretrain(self,activate:bool = False):
        '''
        Activates or deactivates pretrain
        Sets new priors to the pretrained network
        '''
        self.pretrain = activate
        if not activate:
            self.mu_w_prior = Normal(self.mu_w, self.prior_sigma)
            self.mu_b_prior = Normal(self.mu_b, self.prior_sigma)

    def sort_bias(self,previous_weight):
        '''
        Sorts the parameters according to the bias
            - previous_weight: Shuffeling from previous layer propogates
        
            #TODO: In reality only few sorts happen, maybe change this
        '''
        # Sort the weights if previous layer was shuffled
        self.mu_w.data = self.mu_w.data[:,previous_weight]
        self.rho_w.data = self.rho_w.data[:,previous_weight]

        # Sort the bias vector in descending order
        sorted_bias, sorted_indices = torch.sort(self.mu_b, descending=True)

        
        #Sort bias 
        self.mu_b.data = sorted_bias
        self.rho_b.data = self.rho_b.data[sorted_indices]
        #Sort weights
        self.mu_w.data = self.mu_w.data[sorted_indices, :]
        self.rho_w.data = self.rho_w.data[sorted_indices, :]
        #Sort is_bayes
        self.is_bayes_w = self.is_bayes_w[sorted_indices]
        self.is_bayes_b = self.is_bayes_b[sorted_indices]
        
        return sorted_indices
  
    def get_log_prior(self):
        # log_prob_w = self.mu_w_prior.log_prob(self.weights).sum()
        # log_prob_b = self.mu_b_prior.log_prob(self.bias).sum()
        # log_prob =  log_prob_w + log_prob_b 
        if self.sampled_params == 0:
            return 0
        else:
            return (torch.relu(-self.rho_w +self.rho_w_prior).sum() + torch.relu(-self.rho_b + self.rho_b_prior).sum())
        # print(f"Penalize  {penalizing_term} Prob {log_prob}")
        # return penalizing_term

    def get_log_post(self):
        '''
        For sparse Bayes this adds a constant
        '''
        # w_post = torch.distributions.Normal(
        #     self.mu_w, torch.log(1 + torch.exp(self.rho_w)))
        # b_post = torch.distributions.Normal(
        #     self.mu_b, torch.log(1 + torch.exp(self.rho_b)))
        return 0 #w_post.log_prob(self.weights).sum() + b_post.log_prob(self.bias).sum()
    

    
    def forward(self, x):
        if self.pretrain:
            self.sample(stochastic = False)
        else:
             self.sample()
        return F.linear(x, self.weights, self.bias)

