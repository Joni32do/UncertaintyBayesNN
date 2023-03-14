import matplotlib.pyplot as plt
import torch
import numpy as np

from data import noise_function

'''
Some code duplets, but I don't wanted to be too fancy
'''

def plot_bayes(data, y_preds, mean ,lower, upper, path = None):
    '''
    Plots results

        - parameters
            o data - Dictionary containing the data and related info
            o mean - Bayes-Net mean
            o lower - Bayes-Net lower quantile (2.5%)
            o upper - Bayes-Net upper quantile (97.5%)
            o path - to store figure -> if None plot
    '''    
    fig,ax = plt.subplots(1,1,figsize=(8,5))
    if data["is_log"]:
        ax.set_xscale('log')

    ###Plot true function
    
    x = data["x_true"].squeeze()
    y = data["y_true"].squeeze()
    plt.plot(x,y, label='Noiseless function')

    
    noise_fn = noise_function(data["noise"], data["noise_fn"])
    plt.fill_between(x, y - 2* noise_fn(x), y + 2*noise_fn(x), alpha=0.2, label='True distribution')




    ###Plot BNN
    plt.plot(data["x_eval"], mean, label='Average Prediction')
    plt.fill_between(data["x_eval"].squeeze(), 
                     lower.squeeze(),
                     upper.squeeze(), 
                     alpha=0.2, 
                     label='2.5%-97.5% Quantile')
    
    
    #Scatter
    plt.scatter(data["x_train"],
                data["y_train"],
                s = 1 , alpha = 0.8,
                color = 'blue',
                label="Train data")
    
    
    x_rep = data["x_eval"].repeat_interleave(data["n_samples"],0)
    side_slide = 0.01
    
    if data["is_log"]:
        side_slide = 0.05 * x_rep
    plt.scatter(x_rep,
                torch.reshape(data["y_eval"],(data["n_samples"] * data["n_bars"],1)),
                s = 0.5 , alpha = 0.5,
                color = 'green',
                label="stoch. dist.")
    
    plt.scatter(x_rep + side_slide,
                np.reshape(y_preds,(data["n_samples"] * data["n_bars"],1)),
                s = 1 , alpha = 0.5,
                color = 'orange',
                label="stoch. dist. BNN")



    #Finalize
    plt.legend()
    plt.xlabel(r"Concentration $c(x,t)$ in $\left[\frac{\mu g}{cm^3} \right]$")
    plt.ylabel(r"Retardation factor $\bar{R}(c)$")
    if path is not None:
        plt.savefig(path + ".pdf")
    else:
        plt.show()
    plt.close()





def plot_pretrain(data, mean, path = None):
    fig,ax = plt.subplots(1,1,figsize=(8,5))
    if data["is_log"]:
        ax.set_xscale('log')

    
    ###Plot true function
    x = data["x_true"].squeeze()
    y = data["y_true"].squeeze()
    plt.plot(x,y, label='Noiseless function')

    #Plot 2sigma environment
    noise_fn = noise_function(data["noise"], data["noise_fn"])
    plt.fill_between(x, y - 2* noise_fn(x), y + 2*noise_fn(x), alpha=0.4, label='True distribution')

    ###Plot BNN
    plt.plot(data["x_train"], mean, label='Pretrained NN')


    #Scatter
    plt.scatter(data["x_train"],
                data["y_train"],
                s = 1 , alpha = 0.8,
                color = 'blue',
                label="Train data")
    
    plt.legend()
    plt.xlabel(r"Concentration $c(x,t)$ in $\left[\frac{\mu g}{cm^3} \right]$")
    plt.ylabel(r"Retardation factor $\bar{R}(c)$")
    if path is not None:
        plt.savefig(path + ".pdf")
    else:
        plt.show()
    plt.close()




def plot_losses(losses, path):
        plt.plot(losses)
        plt.yscale('log')
        plt.savefig(path)




def plot_retardation(data, path = None):
    '''
    Plots results

        - parameters
            o data - Dictionary containing the data and related info
            o path - to store figure -> if None plot
    '''    
    fig,ax = plt.subplots(1,1,figsize=(8,5))
    if data["is_log"]:
        ax.set_xscale('log')

    x = data["x_true"].squeeze()
    y = data["y_true"].squeeze()
    plt.plot(x,y, label='Noiseless function')

    
    noise_fn = noise_function(data["noise"], data["noise_fn"])
    plt.fill_between(x, y - 2* noise_fn(x), y + 2*noise_fn(x), alpha=0.4, label='True distribution')

    plt.legend()
    plt.xlabel(r"Concentration $c(x,t)$ in $\left[\frac{\mu g}{cm^3} \right]$")
    plt.ylabel(r"Retardation factor $\bar{R}(c)$")
    if path is not None:
        plt.savefig(path + ".pdf")
    else:
        plt.show()
    plt.close()

