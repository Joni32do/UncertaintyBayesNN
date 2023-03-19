import numpy as np
import torch
import copy
from torch import nn

from bnnLayer import LinearBayes


#Temporary
import matplotlib as plt


class BayesianNet(nn.Module):
    def __init__(self, arc=[1,10,1], bayes_arc = [1], rho_w = -4, rho_b = None, elbo_hyper = None, init_pretrain = True):
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
        
        Example:

            Network with Architecture:
                [  2  ,                5           ,       1  ]
                [Input,  Neurons in Hidden Layer(s),    Output]
            
            Bayes Architecture:
                [     ,  [      4     ,      5     ],   [0, 1]]
                [     ,  [Neurons with Bayes Weight, Neurons with Bayes Weight],...]


        There are plenty of Shortcuts Explained in the method 
                >> assemble_bayes_arc <<
        '''
        #Initialize
        self.n_layers = len(arc)
        self.elbo_hyper = elbo_hyper
        self.activation_fn = torch.nn.ReLU()
        self.pretrain = init_pretrain

        #Priors can be initialized differently
        if rho_b is None:
            rho_b = rho_w

        #Architectue
        self.arc=arc
        self.bayes_arc = self.decode_shortcuts(bayes_arc.copy())

        


        #creates a list with Linear Bayes layers of specified architectur
        layers = []
        for i in range(self.n_layers-1):
            layers.append(LinearBayes(arc[i],arc[i+1],
                            rho_w_prior = rho_w, 
                            rho_b_prior = rho_b, 
                            bayes_factor_w = self.bayes_arc[i][0],
                            bayes_factor_b = self.bayes_arc[i][1], 
                            pretrain=init_pretrain))
                
        self.layers = nn.ModuleList(layers)
        
        self.n_sampled = 0
        for layer in self.layers:
            self.n_sampled += layer.sampled_params
        #For plotting ELBO
        self.log_like = 0
        self.log_prior = 0


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
        self.pretrain = pretrain

    #ELBO loss
    def sample_elbo(self, x, y, samples, noise, kl_weight):
        if self.pretrain:
            return nn.MSELoss()(self.forward(x),y)
        n = y.size()[0]
        log_likes = torch.empty((samples,n,1))
        log_priors = torch.empty((samples,))
        # log_posts = torch.empty((samples,))

        #This has to be changed, if the true distribution want to be described
        noise = torch.reshape(torch.logspace(-2,0,len(y)), y.shape) * noise
        
        for i in range(samples):
            pred = self.forward(x)
            log_likes[i] = torch.distributions.Normal(y, noise).log_prob(pred)
            log_priors[i] = self.get_log_prior()
            # log_posts[i] = self.log_post()
        
        # log_post = log_posts.mean()
        self.log_prior = kl_weight *log_priors.mean()
        self.log_like = -log_likes.mean()

        elbo_loss = self.log_prior + self.log_like
        return elbo_loss


    def get_log_prior(self):
        '''
        Calculates the prior over all variational layers
            
            * sum (but could be average) 
        '''
        log_prior = 0
        for layer in self.layers:
            log_prior += layer.get_log_prior()
        return log_prior/self.n_sampled
    
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
            if idx < self.n_layers - 2: 
                x = self.activation_fn(layer(x))
            else: #last layer
                x = layer(x)

        return x
    

    '''
    Functions for Initialization
    
    Explains all available Shortcuts for Bayesian Architecture
    
    
    '''

        #Functions for Init
    def decode_shortcuts(self, bayes_arc):
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
                   
            for i in range(self.n_layers - 2):
                bayes_arc.append(middle)

            bayes_arc.append(last)
        
        #If only one entry is given extend it for both weight and bias
        if type(bayes_arc[0]) is not list:
            for i in range(len(bayes_arc)):
                bayes_arc[i] = [bayes_arc[i], bayes_arc[i]]
        
        return bayes_arc
        

    def get_last_layer(self, bayes_arc):
        '''
        Shortcut for only last layer:

            Returns: 
                intermediate layer and last layer (the same if not neg) 
            
            Ensures: 
                bayes_arc is empty
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
    
    '''
    Functions to plot the network
    
    '''
    def get_net_params(self):
        mu_w = []
        mu_b = []
        rho_w = []
        rho_b = []
        for layer in self.layers:
            #Append mean
            mu_w.append(copy.deepcopy(layer.mu_w).detach().numpy())
            mu_b.append(copy.deepcopy(layer.mu_b).detach().numpy())
            
            #Append variance
            rho_w_full = copy.deepcopy(layer.rho_w).detach().numpy()
            rho_b_full = copy.deepcopy(layer.rho_b).detach().numpy()
            # is_not_bayes_w = not copy.deepcopy(layer.is_bayes_w).detach.numpy()
            # is_not_bayes_b = not copy.deepcopy(layer.is_bayes_b).detach.numpy()
            rho_w_sparse = rho_w_full
            rho_b_sparse = rho_b_full
            rho_w.append(rho_w_sparse)
            rho_b.append(rho_b_sparse)
            #Set variance of not sampled points to -100


        return mu_w, mu_b, rho_w, rho_b
    
    def print_net(self):
        mu_w, mu_b, rho_w, rho_b = self.get_net_params() 
        print("The mean of the weights ", mu_w)
        print("The mean of the biases ", mu_b)
        print("The standard deviation of the weights ",rho_w)
        print("The standard deviation of the biases ",rho_b)


    def plot_network(self):
        mu_w, mu_b, rho_w, rho_b = self.get_net_params()
        img_mu = self.draw_weight_bias(mu_w,mu_b)
        img_rho = self.draw_weight_bias(rho_w, rho_b)
        fig,axes = plt.subplots

    def draw_weight_bias(self, weights, biases):
        '''
            Expects an list of the weights and the biases 
        '''
        lcm = np.lcm(self.arc)
        height = lcm
        bias_width = np.ceil(0.2*lcm)
        width = (self.n_layers-1)*(lcm + bias_width)
        img = np.zeros(width, height)
        for i in range(self.n_layers-1):
            img[:,i*lcm:(i+1)*lcm] = self.draw_weight_bias_block(weights[i],biases[i],bias_width)
        return img

    def draw_weight_bias_block(self, weight, bias,bias_width):
        lcm = np.lcm(self.arc)
        height, width = weight.shape
        height_factor = int(lcm/height)
        width_factor = int(lcm/width)
        longer_weight = np.repeat(weight, height_factor, axis = 0)
        longer_weight = np.repeat(longer_weight, width_factor, axis = 1)
        
        bias_factor = int(lcm/len(bias))
        bias = np.reshape(bias, (len(bias),1))
        longer_bias = np.repeat(bias, bias_factor, axis = 0)
        longer_bias = np.repeat(bias, bias_width, axis = 1)
        return np.concatenate((longer_weight,longer_bias),axis = 1)

        


