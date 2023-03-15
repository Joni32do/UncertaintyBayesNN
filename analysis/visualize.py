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
    
    
    
    
    
    
    #Scatter only for some bars
    goal_bars = 6
    scatter_sample = int(data["n_bars"]/(goal_bars-1))
    #[0::scatter_sample]

    x_bar_single = data["x_eval"][0::scatter_sample]
    x_bars = x_bar_single.repeat_interleave(data["n_samples"],0)
    y_bars_eval = data["y_eval"][0::scatter_sample,:].flatten()
    y_bars_pred = y_preds[0::scatter_sample,:].flatten()

    #Little shift to the right
    side_slide = 0.01
    if data["is_log"]:
        side_slide = 0.05 * x_bars


    plt.scatter(x_bars,
                y_bars_eval,
                s = 2 , alpha = 0.3,
                color = 'green',
                label="stoch. dist.")
    
    # x_B = (x_rep + side_slide)
    # y_B = np.reshape(y_preds,(data["n_samples"] * data["n_bars"],1))
    plt.scatter(x_bars + side_slide,
                y_bars_pred,
                s = 2 , alpha = 0.3,
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
        shift = abs(np.min(losses)) + 0.0001
        plot_losses = [x + shift for x in losses]
        plt.plot(plot_losses)
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

