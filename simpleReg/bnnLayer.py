import numpy as np
import matplotlib.pyplot as plt


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist

class BayesLinear(nn.Module):
    r"""
    Applies Bayesian Linear

    Arguments:
        prior_mu (Float): mean of prior normal distribution.
        prior_sigma (Float): sigma of prior normal distribution.

    .. note:: other arguments are following linear of pytorch 1.2.0.
    https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/linear.py
    
    """
    __constants__ = ['prior_mu', 'prior_sigma', 'bias', 'in_features', 'out_features']

    def __init__(self, prior_mu, prior_sigma, in_features, out_features, bias=True):
        super(BayesLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        self.prior_mu = prior_mu
        self.prior_sigma = prior_sigma
        self.prior_log_sigma = np.log(prior_sigma)
        
        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features))
        self.weight_log_sigma = nn.Parameter(torch.Tensor(out_features, in_features))
        self.register_buffer('weight_eps', None)
               
        if bias is None or bias is False :
            self.bias = False
        else :
            self.bias = True
            
        if self.bias:
            self.bias_mu = nn.Parameter(torch.Tensor(out_features))
            self.bias_log_sigma = nn.Parameter(torch.Tensor(out_features))
            self.register_buffer('bias_eps', None)
        else:
            self.register_parameter('bias_mu', None)
            self.register_parameter('bias_log_sigma', None)
            self.register_buffer('bias_eps', None)
            
            
            
        self.reset_parameters()

    def reset_parameters(self):
        # Initialization method of Adv-BNN
        stdv = 1. / np.sqrt(self.weight_mu.size(1))
        self.weight_mu.data.uniform_(-stdv, stdv)
        self.weight_log_sigma.data.fill_(self.prior_log_sigma)
        if self.bias :
            self.bias_mu.data.uniform_(-stdv, stdv)
            self.bias_log_sigma.data.fill_(self.prior_log_sigma)
         
        # Initialization method of the original torch nn.linear.
#         init.kaiming_uniform_(self.weight_mu, a=math.sqrt(5))
#         self.weight_log_sigma.data.fill_(self.prior_log_sigma)
        
#         if self.bias :
#             fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight_mu)
#             bound = 1 / math.sqrt(fan_in)
#             init.uniform_(self.bias_mu, -bound, bound)
            
#             self.bias_log_sigma.data.fill_(self.prior_log_sigma)

    def freeze(self) :
        self.weight_eps = torch.randn_like(self.weight_log_sigma)
        if self.bias :
            self.bias_eps = torch.randn_like(self.bias_log_sigma)
        
    def unfreeze(self) :
        self.weight_eps = None
        if self.bias :
            self.bias_eps = None 
            
    def forward(self, input):
        r"""
        Overriden.
        """
        if self.weight_eps is None :
            weight = self.weight_mu + torch.exp(self.weight_log_sigma) * torch.randn_like(self.weight_log_sigma)
        else :
            weight = self.weight_mu + torch.exp(self.weight_log_sigma) * self.weight_eps
        
        if self.bias:
            if self.bias_eps is None :
                bias = self.bias_mu + torch.exp(self.bias_log_sigma) * torch.randn_like(self.bias_log_sigma)
            else :
                bias = self.bias_mu + torch.exp(self.bias_log_sigma) * self.bias_eps                
        else :
            bias = None
            
        return F.linear(input, weight, bias)
        
    def extra_repr(self):
        r"""
        Overriden.
        """
        return 'prior_mu={}, prior_sigma={}, in_features={}, out_features={}, bias={}'.format(self.prior_mu, self.prior_sigma, self.in_features, self.out_features, self.bias is not None)


class LinearBayes(nn.Module):
    """
    Implementation of a linear layer with Bayesian weights.

    At the moment the initial mu and sigma must already fit the required size of
    the matrices of weight and bias respectively to n_in and n_out 
    torch.Tensor(n_out, n_in)

    TODO: Add bias
    """
    def __init__(self, mu, sigma):
        super(LinearBayes, self).__init__()

        self.mu = nn.Parameter(mu, requires_grad=True)
        self.sigma = nn.Parameter(sigma, requires_grad=True)
        self.weightsDist = dist.normal.Normal(self.mu, self.sigma)
        self.weights = 0.0

    def sample(self):
        self.weights = self.weightsDist.sample()
    
    
    def forward(self, x):
        self.sample()
        return F.linear(x, self.weights)


