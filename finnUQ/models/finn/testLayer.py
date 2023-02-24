import numpy as np
import torch
from torch import nn

from bnnLayer import LinearBayes
from torch.distributions import MultivariateNormal as Normal

import os
import time
from pathlib import Path
import matplotlib.pyplot as plt


np.random.seed(42)

class BayesianNet(nn.Module):
    def __init__(self, layer_sizes=[1,10,1], proportional_bayes = 0):
        super(BayesianNet, self).__init__()

        self.layer_sizes=layer_sizes
        self.layers_num = len(layer_sizes)
        
        layers = []
        for i in range(self.layers_num-1):
            layers.append(LinearBayes(layer_sizes[i],layer_sizes[i+1],
                                        rho_w_prior = -5, rho_b_prior = -5, 
                                        zero_variance = proportional_bayes, pretrain=True))
        self.layers = nn.ModuleList(layers)


    def sort_bias(self):
        previous_sort = torch.arange(0,self.layer_sizes[0]) #first sort is the identity
        for layer in self.layers: 
            previous_sort = layer.sort_bias(previous_sort)

    def set_pretrain(self, pretrain):
        for layer in self.layers:
            layer.pretrain = pretrain

    def forward(self, x):
        
        for idx, layer in enumerate(self.layers):
            if idx < self.layers_num - 2: 
                x = torch.tanh(layer(x))
            else: #last layer
                x = layer(x)

        return x
    




def generate_data(n, std = 0.001, x_min = -1 , x_max = 1):
    # Generate an arbitrary dataset
    x = torch.linspace(x_min,x_max, n).reshape((n,1))
    noise = torch.randn((n,1)) * x * std
    y = torch.sin(x) + noise
    return x,y


def train_net(net, epochs, x_train, y_train, pretrain_epochs = 0, sort = False):
    # Define the loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    net.set_pretrain(True)

    def closure():
        optimizer.zero_grad()

        # Forward pass
        output = net(x_train)
        
        # Compute the loss
        loss = criterion(output, y_train)
        # Compute the KL divergence loss for the Bayesian self.layers TODO:
        # kl_divergence_loss = net.layer1.compute_kl_divergence_loss() + net.layer2.compute_kl_divergence_loss()

        # Backward pass
        loss.backward()
        # (loss + kl_divergence_loss).backward()
        return loss.item()





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
    plt.savefig(os.path.join(nameFig))


def experiment(BayesianNet, std_train, epochs, pretrain_epochs, 
               n_runs, x_train, y_train, x_test, y_true, 
               pathFigure, bayes_factor, build, t):
    
    #Location and name of file
    file = os.path.join(pathFigure,str(build) + str(bayes_factor) + str(t) + ".pdf")
    
    #Network    
    net = BayesianNet(build, bayes_factor)
                
    #Training
    mse = train_net(net,epochs, x_train, y_train, pretrain_epochs)
    
    #Evaluation
    mean,lower,upper = eval_Bayes_net(net,x_test,n_runs)
    
    #Plotting
    create_fig(x_train, y_train, x_test, y_true, std_train, mean, lower, upper, file)

    return np.around(mse, 3)



if __name__ == '__main__':
    
    ### Parameters
    #Dataset
    std_train = 0.02
    n_train = 250
    
    #Training
    epochs = 2500
    
    #Extras
    pretrain_epochs = 0
    sort = True
    bayes_factors = 0 #[0,0.4,0.8]
        #TODO:
        #Horizontal and Vertical sparse Bayes

    #Evaluation BNN
    n_runs = 100



    #Data
    x_train,y_train = generate_data(n_train,std_train,-3.5,3.5)
    x_test, y_true = generate_data(500,0,-3.5,3.5)


    #Architectures
    architectures =  [[1, 32, 1], #96
                        [1, 8, 8, 1], #97
                        [1, 4, 9, 4, 1], #97
                        [1, 8, 4, 8, 1]] #98
                        # [1, 5, 5, 5, 5, 1], #106
                        # [1, 3, 3, 3, 3, 3, 3, 3, 1]] #104

    #Documentation
    descr = "Std_" + str(std_train) + "N_" + str(n_train) + "P_" + str(pretrain_epochs) + "Sort" + str(sort) + "Prop" + str(min_bayes)
    tries = 1
    training_time = []
    mse = np.zeros((len(architectures),tries))


    #File managment
    pathFigure = os.path.join(Path("./old_files/"),descr)
    Path(pathFigure).mkdir(parents=True, exist_ok=True)






    #Experiment loop
    #TODO:
    #Kann man das noch schöner aufschreiben?
    #Eventuell neue Klasse mit Testcases schreiben und dann darüber iterieren

    #Loop over sparcity
    for bayes_factor in bayes_factors:

        #Loop over architectures
        for i, build in enumerate(architectures):
            start = time.time()

            #Multiple experiments 
            for t in range(tries):
                print("Training architecture: ", build, bayes_factor, t)
                mse[i,t] = experiment(BayesianNet, std_train, epochs, pretrain_epochs, n_runs, 
                           x_train, y_train, x_test, y_true, pathFigure, bayes_factor, build, t)

            #Time
            training_time.append(f"{((time.time()-start)/tries):.3f}")

    
    print(descr)
    print(f"Average training time per model: {training_time}")
    print(mse)
    print(np.median(mse,axis = 1))

    #Average time [35.42496600151062, 29.087465858459474, 32.65702700614929, 35.68414011001587]
    #             [41.32263898849487, 30.761462450027466, 34.610280227661136, 37.57550983428955]
    # [164.99880924224854, 118.13511543273925, 138.7293704509735, 151.16199288368225] devide by 4
    # [52.89364037513733, 39.832936429977416, 47.22352075576782, 50.13765907287598]   devide by 4
    # Average training time per model: ['12.720', '9.381', '10.854', '11.830']


    # 
    # [20133471116423607, 15106025151908398, 27054928243160248, 2332756482064724, 16894308850169182, 46769317239522934, 3304370865225792, 1674988493323326, 1912361942231655, 26123512536287308, 13175337575376034, 5407398194074631, 4117017984390259, 41945718228816986, 25435183197259903, 18552729859948158, 17231905832886696, 5038640648126602, 3293321281671524, 19826224073767662]
    # [1736282743513584, 23164907470345497, 3120882250368595, 2104974165558815, 25558410212397575, 13111635111272335, 2545033022761345, 14738003723323345, 39215344935655594, 4256651923060417, 15468195080757141, 09874545969069004, 11715863831341267, 19217902794480324, 20155303180217743, 21815728396177292, 1436429750174284, 15021882951259613, 3352915495634079, 45724786818027496]
    # [20846521481871605, 21361052989959717, 21236462518572807, 4215819388628006, 7366295158863068, 11309860274195671, 19387878477573395, 29029840603470802, 1671369932591915, 26124469935894012, 18320558592677116, 1164066232740879, 18822193145751953, 13453301042318344, 2321937493979931, 11851240880787373, 7449352741241455, 3556154668331146, 28258193284273148, 21847521886229515]
    # [25888919830322266, 22623518481850624, 14749597758054733, 2262823097407818, 43576259166002274, 3083065338432789, 4830504208803177, 1321403868496418, 16028545796871185, 2707815170288086, 17219970002770424, 1485853735357523, 2619716338813305, 19749265164136887, 46049341559410095, 23089351132512093, 26051145046949387, 20485857501626015, 21118244156241417, 28392259031534195]

    # Pretraining 1000 [[0.016168026253581047, 0.02578503079712391, 0.016533294692635536, 0.021291028708219528, 0.02304607443511486, 0.02290804125368595, 0.02143917977809906, 0.029293807223439217, 0.017819544300436974, 0.02391926571726799, 0.02131359837949276, 0.032965052872896194, 0.0194984320551157, 0.04183349758386612, 0.01797882840037346, 0.017847711220383644, 0.017408881336450577, 0.019218407571315765, 0.010998682118952274, 0.022906959056854248], [0.012258275412023067, 0.024070139974355698, 0.033813294023275375, 0.021265270188450813, 0.02939278818666935, 0.031587984412908554, 0.02515518292784691, 0.036335911601781845, 0.010197069495916367, 0.014204910956323147, 0.017414169386029243, 0.020592276006937027, 0.018513228744268417, 0.019866062328219414, 0.03974160924553871, 0.023357989266514778, 0.029085805639624596, 0.030779970809817314, 0.013882946223020554, 0.022058958187699318], [0.042827505618333817, 0.016970165073871613, 0.04971015453338623, 0.04536658525466919, 0.04114975780248642, 0.014731847681105137, 0.035673145204782486, 0.07452093064785004, 0.01668245904147625, 0.07398519665002823, 0.033485278487205505, 0.07680857181549072, 0.019883012399077415, 0.028392188251018524, 0.04702213779091835, 0.020187310874462128, 0.02679395116865635, 0.04247790947556496, 0.031651537865400314, 0.03024674952030182], [0.030301876366138458, 0.02935848757624626, 0.02033906802535057, 0.029170339927077293, 0.017497455701231956, 0.014652649872004986, 0.01940433494746685, 0.022295957431197166, 0.014791572466492653, 0.009791702032089233, 0.04010399803519249, 0.02116173878312111, 0.03984319791197777, 0.017222754657268524, 0.04208578169345856, 0.07639366388320923, 0.050331782549619675, 0.018113570287823677, 0.024140827357769012, 0.02538885362446308]]
    # Pretraining 1000 without sort


    # ['0.034', '0.019', '0.040', '0.012', '0.027', '0.029', '0.013', '0.017', '0.026', '0.020', '0.041', '0.017', '0.018', '0.020', '0.020', '0.020', '0.019', '0.014', '0.026', '0.017', 
    # '0.029', '0.030', '0.025', '0.045', '0.011', '0.037', '0.016', '0.027', '0.015', '0.064', '0.013', '0.038', '0.034', '0.013', '0.016', '0.012', '0.012', '0.022', '0.009', '0.027', 
    # '0.020', '0.018', '0.018', '0.021', '0.042', '0.053', '0.035', '0.036', '0.024', '0.043', '0.028', '0.073', '0.025', '0.029', '0.013', '0.048', '0.033', '0.024', '0.050', '0.040', 
    # '0.040', '0.021', '0.026', '0.031', '0.023', '0.032', '0.037', '0.013', '0.014', '0.059', '0.022', '0.024', '0.021', '0.055', '0.018', '0.015', '0.025', '0.031', '0.012', '0.102']

# Average training time per model: ['13.288', '9.733', '11.012', '11.865']
# [[0.026 0.026 0.025 0.019 0.048 0.021 0.055 0.023 0.022 0.021 0.02  0.022
#   0.013 0.021 0.017 0.019 0.025 0.031 0.019 0.012]
#  [0.024 0.024 0.015 0.013 0.017 0.057 0.02  0.029 0.014 0.038 0.017 0.017
#   0.018 0.013 0.032 0.036 0.025 0.043 0.017 0.024]
#  [0.044 0.024 0.037 0.034 0.067 0.052 0.029 0.023 0.07  0.052 0.029 0.042
#   0.043 0.032 0.031 0.042 0.027 0.015 0.012 0.049]
#  [0.022 0.019 0.026 0.032 0.019 0.023 0.034 0.038 0.016 0.019 0.011 0.022
#   0.022 0.019 0.034 0.012 0.019 0.015 0.101 0.025]]
# [0.02425 0.02465 0.0377  0.0264 ]

#More Pretraining
# [[0.02  0.01  0.014 0.014 0.014 0.015 0.011 0.015 0.03  0.019 0.024 0.02
#   0.017 0.023 0.013 0.011 0.018 0.026 0.015 0.01 ]
#  [0.026 0.011 0.012 0.025 0.017 0.017 0.03  0.023 0.04  0.021 0.011 0.016
#   0.01  0.015 0.019 0.015 0.021 0.012 0.026 0.025]
#  [0.009 0.011 0.112 0.071 0.084 0.043 0.035 0.016 0.019 0.009 0.117 0.025
#   0.035 0.018 0.05  0.038 0.029 0.04  0.044 0.011]
#  [0.02  0.021 0.046 0.018 0.018 0.019 0.02  0.092 0.017 0.015 0.03  0.012
#   0.02  0.012 0.022 0.042 0.016 0.032 0.013 0.013]]
# [0.01695 0.0196  0.0408  0.0249 ]


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