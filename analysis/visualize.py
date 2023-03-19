import matplotlib.pyplot as plt
import torch
import numpy as np
import os

import matplotlib.animation as a

from data import noise_function

'''
Some code duplets, but I don't wanted to be too fancy
'''

SCATTER_DOT_SIZE = 6

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
    side_slide = 0.007
    if data["is_log"]:
        side_slide = 0.05 * x_bars


    plt.scatter(x_bars,
                y_bars_eval,
                s = SCATTER_DOT_SIZE , alpha = 0.3,
                color = 'green',
                label="stoch. dist.")
    
    # x_B = (x_rep + side_slide)
    # y_B = np.reshape(y_preds,(data["n_samples"] * data["n_bars"],1))
    plt.scatter(x_bars + side_slide,
                y_bars_pred,
                s = SCATTER_DOT_SIZE , alpha = 0.3,
                color = 'orange',
                label="stoch. dist. BNN")



    #Finalize
    plt.legend()
    plt.xlabel(r"Concentration $c(x,t)$ in $\left[\frac{\mu g}{cm^3} \right]$")
    plt.ylabel(r"Retardation factor $\frac{1}{R(c)}$")
    if path is not None:
        plt.savefig(path + ".pdf")
    else:
        plt.show()
    plt.close()





def plot_retardation(data, y_preds, mean ,lower, upper, path = None):
    '''
    Plots $R(c)$ 
    '''    
    fig,ax = plt.subplots(1,1,figsize=(8,5))
    if data["is_log"]:
        ax.set_xscale('log')
    ax.set_yscale('log')
    ###Plot true function
    
    x = data["x_true"].squeeze()
    y = data["y_true"].squeeze()
    plt.plot(x,1/y, label='Noiseless function')

    
    noise_fn = noise_function(data["noise"], data["noise_fn"])
    
    upper_true = 1/(y + 2*noise_fn(x))
    lower_true = 1/(y - 2* noise_fn(x))
    plt.fill_between(x, lower_true, upper_true, alpha=0.2, label='True distribution')




    ###Plot BNN
    plt.plot(data["x_eval"], 1/mean, label='Average Prediction')

    plt.fill_between(data["x_eval"].squeeze(), 
                     1/lower.squeeze(),
                     1/upper.squeeze(), 
                     alpha=0.2, 
                     label='2.5%-97.5% Quantile')
    
    
    
    
    
    
    #Scatter only for some bars
    goal_bars = 6
    scatter_sample = int(data["n_bars"]/(goal_bars-1))
    #[0::scatter_sample]

    x_bar_single = data["x_eval"][0::scatter_sample]
    x_bars = x_bar_single.repeat_interleave(data["n_samples"],0)
    y_bars_eval = 1/data["y_eval"][0::scatter_sample,:].flatten()
    y_bars_pred = 1/y_preds[0::scatter_sample,:].flatten()

    #Little shift to the right
    side_slide = 0.01
    if data["is_log"]:
        side_slide = 0.05 * x_bars


    plt.scatter(x_bars,
                y_bars_eval,
                s = SCATTER_DOT_SIZE , alpha = 0.3,
                color = 'green',
                label="stoch. dist.")
    
    # x_B = (x_rep + side_slide)
    # y_B = np.reshape(y_preds,(data["n_samples"] * data["n_bars"],1))
    plt.scatter(x_bars + side_slide,
                y_bars_pred,
                s = SCATTER_DOT_SIZE , alpha = 0.3,
                color = 'orange',
                label="stoch. dist. BNN")



    #Finalize
    plt.legend()
    plt.xlabel(r"Concentration $c(x,t)$ in $\left[\frac{\mu g}{cm^3} \right]$")
    plt.ylabel(r"Retardation factor ${R(c)}$")
    if path is not None:
        plt.savefig(path + "_ret.pdf")
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
    plt.plot(x,y, label='$R(c)$')

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
    plt.ylabel(r"Retardation factor $\frac{1}{R(c)}$")
    if path is not None:
        plt.savefig(path + ".pdf")
    else:
        plt.show()
    plt.close()



def plot_pretrain_retardation(data, mean, path = None):
    fig,ax = plt.subplots(1,1,figsize=(8,5))
    if data["is_log"]:
        ax.set_xscale('log')

    # ax.set_yscale('log')
    
    ###Plot true function
    x = data["x_true"].squeeze()
    y = data["y_true"].squeeze()
    plt.plot(x,1/y, label='$R(c)$')

    #Plot 2sigma environment
    noise_fn = noise_function(data["noise"], data["noise_fn"])
    upper_true = 1/(y + 2*noise_fn(x))
    lower_true = 1/(y - 2* noise_fn(x))
    plt.fill_between(x,lower_true, upper_true, alpha=0.4, label='True distribution')

    ###Plot BNN
    plt.plot(data["x_train"], 1/mean, label='Pretrained NN')


    #Scatter
    plt.scatter(data["x_train"],
                1/data["y_train"],
                s = 1 , alpha = 0.8,
                color = 'blue',
                label="Train data")
    
    plt.legend()
    plt.xlabel(r"Concentration $c(x,t)$ in $\left[\frac{\mu g}{cm^3} \right]$")
    plt.ylabel(r"Retardation factor $R(c)$")
    if path is not None:
        plt.savefig(path + "_ret.pdf")
    else:
        plt.show()
    plt.close()



def plot_losses(losses, path):
        shift = abs(np.min(losses)) + 0.0001
        plot_losses = [x + shift for x in losses]
        plt.plot(plot_losses)
        plt.yscale('log')
        plt.savefig(path+".pdf")


def plot_losses_elbo(losses, likes, priors, path):
    shift = abs(min(np.min(priors), np.min(likes), np.min(losses))) + 0.0001
    plot_losses = [x + shift for x in losses]
    plot_likes = [x + shift for x in likes]
    plot_priors = [x + shift for x in priors]
    plt.plot(plot_losses, label = "added loss", linewidth=1)
    plt.plot(plot_likes, label = "likelihoods", linewidth=1)
    plt.plot(plot_priors, label = "priors", linewidth=1)
    plt.yscale('log')
    plt.legend()
    plt.savefig(path+".pdf")



def animate_training(image_collection,save_path):
    '''
    Animates a collection of images
    ________________
    |              | W
    |              | I
    |              | D
    |              | T
    | L E N G T H  | H
    ________________
    '''
    img = np.array(image_collection)
    frames, width, length = np.shape(img)
    abs_length = 3
    rel_width = abs_length * width/length

    fig= plt.figure()
    fig.set_size_inches(abs_length,rel_width)
    ax = plt.Axes(fig, [0.,0.,1.,1.])
    ax.set_axis_off()
    fig.add_axes(ax)


    
    image = ax.imshow(img[0,:,:], cmap = 'viridis')
    def animate(i):
        image.set(data = img[i,:,:])

    anim = a.FuncAnimation(fig, animate, interval = 1000, frames = frames-1)

    writergif = a.PillowWriter(fps=10)
    anim.save(save_path +".gif", writer=writergif)


