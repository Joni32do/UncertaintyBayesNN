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
                 bayes_factor = 1, pretrain = False):
        
        super(LinearBayes, self).__init__()
        
        ###Sparse Bayesian
        #Proportional to Bayes (special case: 1 Neuron ->  1/n_out [1 is understood as fully bayesian])
        if bayes_factor <= 1:
            bayes_factor = int(bayes_factor * n_out)
        #Neurons
        self.is_bayes = torch.zeros(n_out,1)
        # self.is_bayes_w = 
        # self.is_bayes_b = 
        if bayes_factor != 0:
             self.is_bayes[:int(bayes_factor)] = 1


        #Mode
        self.pretrain = pretrain
        
        #Additional support for scalar initialization
        #   Better: reset parameters ...
        if not torch.is_tensor(mu_w_prior):
            mu_w_prior = mu_w_prior + torch.rand(n_out, n_in) - 0.5
        if not torch.is_tensor(rho_w_prior):
            rho_w_prior = rho_w_prior*torch.ones(n_out,n_in)
        if  not torch.is_tensor(mu_b_prior):
            mu_b_prior = mu_b_prior + torch.rand(n_out) - 0.5
        if not torch.is_tensor(rho_b_prior):
            rho_b_prior = rho_b_prior*torch.ones(n_out)

        #Size
        self.n_in = n_in
        self.n_out = n_out

        #Priors
        self.mu_w_prior = mu_w_prior
        self.sigma_w_prior = torch.log(1+torch.exp(rho_w_prior))

        self.mu_b_prior = mu_b_prior
        self.sigma_b_prior = torch.log(1+torch.exp(rho_b_prior))


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
         

    def sample(self, stochastic = True):
        '''
        Samples from the neurons where is_bayes
        '''
        self.weights = self.mu_w  + (torch.log(1 + torch.exp(self.rho_w)) * self.noise_w.sample() 
                                     * self.is_bayes if stochastic else 0)
        
        self.bias = self.mu_b + (torch.log( 1 + torch.exp(self.rho_b)) * self.noise_b.sample() 
                                 * self.is_bayes.squeeze() if stochastic else 0)


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
        self.is_bayes = self.is_bayes[sorted_indices]
        
        return sorted_indices
  
    def kl_loss(self):
        '''
        Evaluates the Kullback-Leibler divergence loss

        TODO:
        !But does it?

        I think I only should penalize if sigma is to low

        '''
        
        sigma_w = torch.log(1 + torch.exp(self.rho_w))
        #Helper
        s_div = self.sigma_w_prior/ sigma_w
        m_dif = self.mu_w_prior - self.mu_w
        loss = 2 * (torch.log(s_div) - 1 + s_div.pow(-2) + ((m_dif)/self.sigma_w_prior).pow(2)).sum()


        #Same calculation for bias
        sigma_b = torch.log(1 + torch.exp(self.rho_b))
        # Helper
        s_div = self.sigma_b_prior/ sigma_b
        m_dif = self.mu_b_prior - self.mu_b
        loss += 2 * (torch.log(s_div) - 1 + s_div.pow(-2) + ((m_dif)/self.sigma_b_prior).pow(2)).sum()
        #          -- Dieser Teil is für Standardabweichung ---- Dieser Teil für Mittelwerte ----- 
        
        #Wenn Neuronen ausgeschalten sind, ist ihr prior ihr rho und damit ist der erste Teil 0 -> passt!
        
        
        


        return loss
    
    def forward(self, x):
        if self.pretrain:
            self.sample(stochastic = False)
        else:
             self.sample()
        return F.linear(x, self.weights, self.bias)



"""       
        self.prior_mu = prior_mu
        self.prior_sigma = prior_sigma
        self.prior_log_sigma = np.log(prior_sigma)
        
        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features))
        self.weight_log_sigma = nn.Parameter(torch.Tensor(out_features, in_features))
        self.register_buffer('weight_eps', None)
  
            
"""
     
        
"""     def sample_weights(self):
self.prior = torch.distributions.Normal(0, prior_var)
        This functionality is implemented here in order to assure
        that the weights are sampled before any time forward propagation
        
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
 """







	
""" 	def evalAllLosses(self) :
		
		t_loss = self.evalLosses()*self.lossScaleFactor
		
		for m in self.children() :
			if isinstance(m, VIModule) :
				t_loss = t_loss + m.evalAllLosses()*self.lossScaleFactor
				
		#Adding to Loss function
		self.addLoss(lambda s : 0.5*s.getSampledWeights().pow(2).sum()/weightPriorSigma**2)
		self.addLoss(lambda s : -self.out_features/2*np.log(2*np.pi) - 0.5*s.samples['wNoiseState'].pow(2).sum() - s.lweights_sigma.sum())

			self.addLoss(lambda s : 0.5*s.getSampledBias().pow(2).sum()/biasPriorSigma**2)##
			self.addLoss(lambda s : -self.out_features/2*np.log(2*np.pi) - 0.5*s.samples['bNoiseState'].pow(2).sum() - self.lbias_sigma.sum()) 
            
            
            
# # kl = 0.5 * (2 * torch.log(sigma_prior / sigma) - 1 + (sigma / sigma_prior).pow(2) + ((mu_prior - mu) / sigma_prior).pow(2)).sum() 
        # # kl_divergence(self.weight, F.softplus(self.log_sigma),
        # #               torch.zeros_like(self.weight).to(self.weight.device), 
        # #               F.softplus(torch.ones_like(self.log_sigma)*self.log_sigma_prior).to(self.weight.device))
        # # Das für weights und biases
        # #Weights



        # loss += 0.5*self.weights.pow(2).sum()/(self.prior_mu**2).sum()
        # loss -= 0.5 * (self.n_out *np.log(2*np.pi) + self.noise_w.sample().pow(2).sum()) - self.rho_w.sum()

        # #Bias
        # loss += 0.5*self.bias.pow(2).sum()/(self.prior_mu**2).sum()
        # loss -= 0.5 * (self.n_out *np.log(2*np.pi) + self.noise_b.sample().pow(2).sum()) - self.rho_b.sum()            
            
            
            
            
            
            
            
"""
	
'''
        print("Before Sort:")
        print(self.mu_b.data)
        print(self.rho_b.data)
        print(self.mu_w.data)
        print(self.rho_w.data)
        print(self.is_bayes)
        print("Sort indicies")
        print(previous_weight)
        print(sorted_indices)
        print("After Sort:")
        print(self.mu_b.data)
        print(self.rho_b.data)
        print(self.mu_w.data)
        print(self.rho_w.data)
        print(self.is_bayes)
        
'''