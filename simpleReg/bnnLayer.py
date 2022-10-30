import numpy as np
import matplotlib.pyplot as plt


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist
from torch.distributions import Normal
from torch.nn.parameter import Parameter


'''
Not the best practics, but I know just compare all the different approaches and find a mitigate which works the best for me
and my usecase
'''

# Bayesian-neural-network-pytorch
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





####### From FINN
class Bayes_Layer(nn.Module):
    """
        This is the building block layer for bayesian NODE's
    """

    def __init__(self, input_features, output_features, prior_var=1.):

        super().__init__()

        # set input and output dimensions
        self.input_features = input_features
        self.output_features = output_features

        # initialize mu and rho parameters for the weights of the layer
        self.w_mu = nn.Parameter(Normal(0., 0.1).expand(
            (output_features, input_features)).sample())
        self.w_rho = nn.Parameter(Normal(-3., 0.1).expand(
            (output_features, input_features)).sample())

        # initialize mu and rho parameters for the layer's bias
        self.b_mu = nn.Parameter(Normal(0., 0.1).expand(
            (output_features,)).sample())
        self.b_rho = nn.Parameter(Normal(-3., 0.1).expand(
            (output_features,)).sample())

        # initialize weight samples
        # (these will be calculated whenever the layer makes a prediction)
        self.w = None
        self.b = None

        # initialize prior distribution for all of the weights and biases
        self.prior = torch.distributions.Normal(0, prior_var)
        
    def sample_weights(self):
        """This functionality is implemented here in order to assure
        that the weights are sampled before any time forward propagation
        """
        # sample weights
        self.w = Normal(self.w_mu, torch.log(1+torch.exp(self.w_rho))).rsample()
        
        # sample bias
        self.b = Normal(self.b_mu, torch.log(1+torch.exp(self.b_rho))).rsample()
        
        # log prior
        w_log_prior = self.prior.log_prob(self.w)
        b_log_prior = self.prior.log_prob(self.b)
        self.log_prior = torch.sum(w_log_prior) + torch.sum(b_log_prior)

        # log posterior
        self.w_post = Normal(
            self.w_mu.data, torch.log(1+torch.exp(self.w_rho)))
        self.b_post = Normal(
            self.b_mu.data, torch.log(1+torch.exp(self.b_rho)))
        self.log_post = self.w_post.log_prob(
            self.w).sum() + self.b_post.log_prob(self.b).sum()

    def forward(self, input):
        """
          Standard linear forward propagation
        """
        return F.linear(input, self.w, self.b)






class VIModule(nn.Module) :
	"""
	A mixin class to attach loss functions to layer. This is usefull when doing variational inference with deep learning.
	"""
	
	def __init__(self, *args, **kwargs) :
		super().__init__(*args, **kwargs)
		
		self._internalLosses = []
		self.lossScaleFactor = 1
		
	def addLoss(self, func) :
		self._internalLosses.append(func)
		
	def evalLosses(self) :
		t_loss = 0
		
		for l in self._internalLosses :
			t_loss = t_loss + l(self)
			
		return t_loss
	
	def evalAllLosses(self) :
		
		t_loss = self.evalLosses()*self.lossScaleFactor
		
		for m in self.children() :
			if isinstance(m, VIModule) :
				t_loss = t_loss + m.evalAllLosses()*self.lossScaleFactor
				
		return t_loss


class MeanFieldGaussianFeedForward(VIModule) :
	"""
	A feed forward layer with a Gaussian prior distribution and a Gaussian variational posterior.
	"""
	
	def __init__(self, 
			  in_features, 
			  out_features, 
			  bias = True,  
			  groups=1, 
			  weightPriorMean = 0, 
			  weightPriorSigma = 1.,
			  biasPriorMean = 0, 
			  biasPriorSigma = 1.,
			  initMeanZero = False,
			  initBiasMeanZero = False,
			  initPriorSigmaScale = 0.01) :
		
		
		super(MeanFieldGaussianFeedForward, self).__init__()
		
		self.samples = {'weights' : None, 'bias' : None, 'wNoiseState' : None, 'bNoiseState' : None}
		
		self.in_features = in_features
		self.out_features = out_features
		self.has_bias = bias
		#Gaussian Distribution Parameters
		self.weights_mean = Parameter((0. if initMeanZero else 1.)*(torch.rand(out_features, int(in_features/groups))-0.5)) #Values between -.5 and .5
		self.lweights_sigma = Parameter(torch.log(initPriorSigmaScale*weightPriorSigma*torch.ones(out_features, int(in_features/groups))))
		#Noise	- probably $\varepsilon$ in BBP
		self.noiseSourceWeights = Normal(torch.zeros(out_features, int(in_features/groups)), 
								   torch.ones(out_features, int(in_features/groups)))
		#Adding to Loss function
		self.addLoss(lambda s : 0.5*s.getSampledWeights().pow(2).sum()/weightPriorSigma**2)
		self.addLoss(lambda s : -self.out_features/2*np.log(2*np.pi) - 0.5*s.samples['wNoiseState'].pow(2).sum() - s.lweights_sigma.sum())
	
		
		#The same for bias
		if self.has_bias :
			self.bias_mean = Parameter((0. if initBiasMeanZero else 1.)*(torch.rand(out_features)-0.5))
			self.lbias_sigma = Parameter(torch.log(initPriorSigmaScale*biasPriorSigma*torch.ones(out_features)))
			
			self.noiseSourceBias = Normal(torch.zeros(out_features), torch.ones(out_features))
			
			self.addLoss(lambda s : 0.5*s.getSampledBias().pow(2).sum()/biasPriorSigma**2)
			self.addLoss(lambda s : -self.out_features/2*np.log(2*np.pi) - 0.5*s.samples['bNoiseState'].pow(2).sum() - self.lbias_sigma.sum())
			
			
	def sampleTransform(self, stochastic=True) :
		#draws a random sample and some random noise 
		self.samples['wNoiseState'] = self.noiseSourceWeights.sample().to(device=self.weights_mean.device)
		self.samples['weights'] = self.weights_mean + (torch.exp(self.lweights_sigma)*self.samples['wNoiseState'] if stochastic else 0)
		
		if self.has_bias :
			self.samples['bNoiseState'] = self.noiseSourceBias.sample().to(device=self.bias_mean.device)
			self.samples['bias'] = self.bias_mean + (torch.exp(self.lbias_sigma)*self.samples['bNoiseState'] if stochastic else 0)
		
	def getSampledWeights(self) :
		return self.samples['weights']
	
	def getSampledBias(self) :
		return self.samples['bias']
	
	def forward(self, x, stochastic=True) :
		
		self.sampleTransform(stochastic=stochastic)
		
		return nn.functional.linear(x, self.samples['weights'], bias = self.samples['bias'] if self.has_bias else None)






#My own which doesnt work 
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


