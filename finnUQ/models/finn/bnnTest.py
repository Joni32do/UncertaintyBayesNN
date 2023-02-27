import numpy as np
import torch
from torch import nn

from bnnLayer import LinearBayes
from torch.distributions import MultivariateNormal as Normal

import os
import time
from pathlib import Path
import matplotlib.pyplot as plt

from bnnNet import BayesianNet

'''
Testframework

Doesn't really mind to much about Exception handeling and other because it 
anyways is only a product of time

'''

np.random.seed(42)


def generate_data(n, std = 0.001, x_min = -1 , x_max = 1):
    # Generate an arbitrary dataset
    x = torch.linspace(x_min,x_max, n).reshape((n,1))
    noise = torch.randn((n,1)) * x * std
    y = torch.sin(x) + noise
    return x,y


def train_net(net, epochs, x_train, y_train, pretrain_epochs = 0, sort = False):
    '''
    Trains a Bayesian network (def line 16)
        - lr = 0.001
    '''
    
    # Define the loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.01)
    net.set_pretrain(True)

    def closure():
        optimizer.zero_grad()

        # Forward pass
        output = net(x_train)
        
        # Compute the loss
        mse_loss = criterion(output, y_train)
        
        # Compute the KL divergence loss for the Bayesian self.layers TODO:
        kl_weight = 0.0
        kl_divergence_loss = net.kl_loss(kl_weight)

        # Backward pass
        loss = mse_loss + kl_divergence_loss
        loss.backward()
        # (loss + kl_divergence_loss).backward()
        return mse_loss.item()





    # Train the net for 1000 epochs
    for epoch in range(epochs):
        # Change from pretrain to train
        if epoch == pretrain_epochs:
            net.set_pretrain(False)


        mse = optimizer.step(closure)
        if sort:
            net.sort_bias()

        # Print the loss every 100 epochs
        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch + 1}: Loss = {mse:.4f}")

            # for m in net.layers:
            #     print(m.rho_w)


    return mse


def eval_Bayes_net(net, x, n_runs, quantile = 0.05):
    # Evaluate function using the Bayesian network   
    y_preds = np.zeros((n_runs,x.size(dim=0)))
    for i in range(n_runs):
        y_pred = net.forward(torch.Tensor(x).unsqueeze(1)).detach().numpy().flatten()
        y_preds[i] = y_pred

    # Calculate mean and quantiles
    mean = np.mean(y_preds, axis=0)
    lower = np.quantile(y_preds, quantile, axis=0)
    upper = np.quantile(y_preds, 1-quantile, axis=0)
    return mean, lower, upper


def create_fig(x_train, y_train, x_test, y_true,std_train, mean ,lower, upper, nameFig = "figure.pdf"):
    # Plot results
    fig = plt.figure(figsize=(8,5))
    plt.scatter(x_train, y_train,s = 3 ,color = 'red',label="Train data")
    plt.plot(x_test,y_true, label='True function')
    plt.fill_between(x_test.squeeze(), (y_true - x_test * std_train).squeeze(), (y_true+ x_test * std_train).squeeze(), alpha=0.5, label='UQ Function')
    plt.plot(x_test, mean, label='Average Prediction')
    plt.fill_between(x_test.squeeze(), (lower).squeeze(), (upper).squeeze(), alpha=0.5, label='5%-95% Quantile')
    plt.legend()
    # plt.savefig(os.path.join(nameFig))
    plt.show()


def experiment(BayesianNet, std_train, epochs, pretrain_epochs, 
               bayes_factor, build, bayes_arc, t, n_runs, 
               plot, x_train, y_train, x_test, y_true, pathFigure):
    
    #Location and name of file
    file = os.path.join(pathFigure,str(build) + str(bayes_factor) + str(t) + ".pdf")
    
    #Network    
    net = BayesianNet(build, bayes_factor, bayes_arc)
                
    #Training
    mse = train_net(net,epochs, x_train, y_train, pretrain_epochs, sort)
    
    #Evaluation
    mean,lower,upper = eval_Bayes_net(net,x_test,n_runs)
    
    #Plotting
    if plot:
        create_fig(x_train, y_train, x_test, y_true, std_train, mean, lower, upper, file)

    return np.around(mse, 4)



if __name__ == '__main__':
    
    ### Parameters
    #Dataset
    n_train = 250
    std_train = 0.02 #Is increasing with distance from origin see data
    
    #Training
    epochs = 1000
    
    #Extras
    pretrain_epochs = 5
    sort = True

    #Evaluation BNN
    n_runs = 5



    #Data
    x_train,y_train = generate_data(n_train,std_train,-3.5,3.5)
    x_test, y_true = generate_data(500,0,-3.5,3.5)


    #Architectures
    architectures =  [[1, 10, 10, 1], #96
                      [1, 8, 8, 1], #97
                      [1, 4, 9, 4, 1], #97
                      [1, 8, 4, 8, 1]] #98
    
    #Bayes Architectur

    #Vertical (or even complete flexibility)^
    #Either proportional or count Neurons (Special case 1 means all)
    bayes_arc =      [[0, 0, 0, 1], 
                      [0, 0, 8, 0], 
                      [0, 0, 9, 0, 0], 
                      [0, 0, 4, 0, 0]] 
    #bayes_arc = None

    #Horizontal approach with bayes_factors
    
    bayes_factors = [0] #[0,0.4,0.8]


    #Documentation
    descr = "Std_" + str(std_train) + "N_" + str(n_train) + "P_" + str(pretrain_epochs) + "Sort" + str(sort) + "Prop" + str(bayes_factors)
    # descr = "Linear"
    tries = 100
    training_time = []
    mse = np.zeros((len(architectures),tries))


    #File managment
    pathFigure = os.path.join(Path("./old_files/"),descr)
    Path(pathFigure).mkdir(parents=True, exist_ok=True)
    plot = True





    #Experiment loop
    #       Improvement: one loop over param_class

    #Loop over sparcity
    for bayes_factor in bayes_factors:

        #Loop over architectures
        for i, build in enumerate(architectures):
            start = time.time()

            #Multiple experiments 
            for t in range(tries):

                print("Training architecture: ", build, bayes_factor, t)
                
                mse[i,t] = experiment(BayesianNet, std_train, epochs, pretrain_epochs, bayes_factor, 
                                      build, bayes_arc[i], t, n_runs, 
                                      plot, x_train, y_train, x_test, y_true, pathFigure)

            #Time
            training_time.append(f"{((time.time()-start)/tries):.3f}")

    

    #Final Logging
    print(descr)
    print(f"Average training time per model: {training_time}")
    print(mse)
    print(np.median(mse,axis = 1))




















####Ugly Aftermath

# Std_0.01N_30P_3000SortTrue
# Average training time per model: ['5.223', '5.135', '6.012', '6.332']
# [[0.023 0.008 0.012 0.01  0.008 0.011 0.016 0.015 0.007 0.014 0.028 0.012
#   0.02  0.011 0.028 0.014 0.007 0.011 0.014 0.045]
#  [0.018 0.01  0.067 0.019 0.011 0.131 0.01  0.002 0.008 0.018 0.014 0.009
#   0.015 0.01  0.015 0.008 0.022 0.014 0.032 0.026]
#  [0.025 0.042 0.007 0.006 0.041 0.041 0.013 0.012 0.022 0.019 0.026 0.01
#   0.025 0.019 0.041 0.022 0.123 0.019 0.017 0.032]
#  [0.02  0.022 0.034 0.02  0.019 0.018 0.013 0.019 0.011 0.052 0.047 0.019
#   0.047 0.019 0.058 0.027 0.042 0.034 0.014 0.043]]
# [0.0157  0.02295 0.0281  0.0289 ]


# Std_0.1N_400P_2500SortTrue
# Average training time per model: ['12.132', '9.381', '10.820', '11.674']
# [[0.051 0.054 0.05  0.074 0.072 0.074 0.061 0.062 0.066 0.055 0.07  0.055
#   0.058 0.058 0.058 0.061 0.062 0.063 0.056 0.073]
#  [0.07  0.061 0.05  0.063 0.095 0.065 0.045 0.046 0.057 0.057 0.095 0.055
#   0.052 0.05  0.073 0.044 0.054 0.104 0.06  0.074]
#  [0.048 0.176 0.121 0.061 0.052 0.107 0.066 0.054 0.049 0.047 0.048 0.049
#   0.056 0.062 0.066 0.051 0.061 0.062 0.061 0.09 ]
#  [0.062 0.076 0.046 0.062 0.072 0.063 0.071 0.066 0.055 0.088 0.049 0.052
#   0.084 0.072 0.053 0.046 0.061 0.056 0.079 0.046]]
# [0.06165 0.0635  0.06935 0.06295]

# Std_0.02N_250P_2500SortTrueProp0.5
# Average training time per model: ['11.415', '8.358', '9.696', '10.436', '12.026', '14.970', '10.778', '8.352', '9.603', '10.435', '11.561', '14.860', '10.312', '8.298', '9.720', '10.548', '11.511', '1686.819', '10.082', '8.333', '10.009', '10.491', '11.498', '14.758', '9.382', '8.358', '9.516', '10.462', '11.513', '14.813', '9.372', '8.029', '9.694', '10.229', '11.432', '14.857', '8.679', '7.809', '9.313', '10.087', '11.328', '14.732', '8.259', '7.345', '9.169', '9.699', '11.384', '14.931', '7.801', '7.482', '8.876', '9.482', '10.406', '14.723', '7.291', '6.892', '8.493', 
# '8.817', '10.408', '14.929']
# [[0.008 0.011 0.01  0.008 0.009 0.013 0.009 0.008 0.013 0.006]
#  [0.003 0.026 0.018 0.005 0.003 0.009 0.031 0.016 0.005 0.026]
#  [0.011 0.008 0.009 0.023 0.011 0.008 0.003 0.013 0.012 0.005]
#  [0.012 0.015 0.016 0.014 0.014 0.022 0.016 0.012 0.012 0.007]
#  [0.04  0.029 0.004 0.06  0.024 0.047 0.04  0.029 0.008 0.03 ]
#  [0.053 0.136 0.075 0.025 0.105 0.022 0.071 0.072 0.04  0.078]]
# [0.0095 0.0142 0.0103 0.014  0.0311 0.0677]


#Average time for training nn.Linear ['1.119']
# Median was [0.002]

#Average training time per model: ['1.258']
# Median was [0.003] and some examples worse

#It is better to use Linear Layer but not substancialy







# [1, 5, 5, 5, 5, 1], #106
# [1, 3, 3, 3, 3, 3, 3, 3, 1]] #104