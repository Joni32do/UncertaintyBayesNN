import torch
import numpy as np


np.random.seed(42)



def noise_function(noise, description):
    '''
    Returns a function which produces noise
    '''
    if description == "+":
        return lambda x: noise
    
    elif description == "*":
        return lambda x: x * noise
    
    elif description == "beta":
        return lambda x: 0 #not implemented
    else:
        raise NotImplementedError("Choose noise_fn to be + or x")

def generate_data(data):
    '''
    generatedata similar to retardation factor to 
    adds
        - training
        - evaluation (with Bayes methods)
        - plotting
    data to dictionary

    parameters
        data distribution
            o lin or log
            o min and max value
        for data types 
            - training
                o n samples
            - evaluation
                o bars - number of x locations
                o samples - number of data per bar
        noise
            o noise factor
                -> is input in noise function which scales with x
    '''
    
    ### Get number of samples
    #For training
    n = data["n_train"]

    #For evaluating
    bars = data["n_bars"]
    samples = data["n_samples"]

    #For plotting
    n_plot = data["n_plot"]
    


    ###Distribute and calculate
    # lin or log
    if data["is_log"]:
        val = data["log"]
        fn = torch.logspace
    else:
        val = data["lin"]
        fn = torch.linspace
    
    #Calculate x
    x_train = fn(val["min"], val["max"], n).reshape((n,1))
    x_eval = fn(val["min"], val["max"],bars).reshape((bars,1))
    x_plot = fn(val["min"], val["max"],n_plot).reshape((n_plot,1))



    #Noise function
    noise_fn = noise_function(data["noise"], data["noise_fn"])

    
    #Parameters 
    params = data["params"]

    k_d = params["k_d"]
    beta = params["beta"]
    rho = params["rho"]
    n_e = params["n_e"]
    f = params["f"]
    a = (1-n_e)/n_e * rho * f * k_d * beta

    ###Calculate 1/R

    #For training
    y_train = torch.zeros((n,1))
    y_train[x_train!=0] = 1/(1 + a * x_train[x_train!=0]**(beta-1))

    #For evaluating
    y_eval_mean = torch.zeros((bars,1))
    y_eval_mean[x_eval!=0] = 1/(1 + a * x_eval[x_eval!=0]**(beta-1))

    #For plotting
    y_plot = torch.zeros((n_plot,1))
    y_plot[x_plot!=0] = 1/(1 + a * x_plot[x_plot!=0]**(beta-1))

    
    ### Noise
    
    #Train
    noise_train = torch.randn((n,1)) * noise_fn(x_train)
    y_train = y_train + noise_train
    
    #Eval
    noise_eval = torch.randn((bars,samples)) * noise_fn(x_eval)
    y_eval = y_eval_mean + noise_eval  #Automatch to shape (bars,samples)

    #Saving to dictionary
    data["x_train"] = x_train
    data["y_train"] = y_train

    data["x_eval"] = x_eval
    data["y_eval_mean"] = y_eval_mean
    data["y_eval"] = y_eval

    data["x_true"] = x_plot
    data["y_true"] = y_plot
    
    return data  