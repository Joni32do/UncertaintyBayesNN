# @File          :   vis.py
# @Last modified :   2022/12/05 18:44:26
# @Author        :   Matthias Gueltig

'''
This script was added to provide visualization methods for solutions and predictions
of FINN in the 2SS model'''
import numpy as np
import torch as th
import torch.nn as nn

import matplotlib.pyplot as plt
import matplotlib

from mpl_toolkits.axes_grid1 import make_axes_locatable
import sys

import pickle5 as pickle

sys.path.append("..")
# print(sys.path)

from utils.configuration import Configuration
from finn import *
import pandas as pd
import os



def __add_fig(fig, ax, row:float, column:float, title:str, value:np.ndarray, 
    x:np.ndarray, t:np.ndarray, is_c:bool = False):
    """add subplot to fig

    Args:
        fig (_type_): _description_
        ax (_type_): _description_
        row (float): _description_
        column (float): _description_
        title (str): _description_
        value (np.ndarray): _description_
        x (np.ndarray): _description_
        t (np.ndarray): _description_
    """
    font_size = FONT_SIZE
    if is_c:
        cmap = 'viridis'
    else:
        cmap = 'YlOrBr'
    #, norm=matplotlib.colors.SymLogNorm(linthresh = 0.000001)
    h = ax[row, column].imshow(value, cmap = cmap,  
                               interpolation='nearest', 
                    extent=[t.min(), t.max(),
                            x.min(), x.max()],
                    origin='upper', aspect='auto')
    divider = make_axes_locatable(ax[row,column])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(h, cax=cax)
    cbar.ax.tick_params(labelsize=font_size)

    ax[row, column].set_xlim(0, t.max())
    ax[row, column].set_ylim(x.min(), x.max())
    ax[row, column].set_xlabel('$t [d]$', fontsize=font_size)
    ax[row, column].set_ylabel('$x [cm]$', fontsize=font_size)
    ax[row, column].set_title(title, fontsize = 1.5*font_size)
    for label in (ax[row, column].get_xticklabels() + ax[row, column].get_yticklabels()): 
        label.set_fontsize(font_size)

def load_data(number:float, config_NN:Configuration):
    """Loads selected model

    Args:
        number (float): model number

    Returns:
        u_NN: NN calculated solution
        u_init_NN: c and sk as initialized in params.json before starting the NN
    """
    
    data_path = os.path.join("results",str(number))
    with open(os.path.join(data_path,"model.pkl"), "rb") as inp:
        model = pickle.load(inp)
    # with open(f"results/{number}/model.pkl", "rb") as inp:
    #     model = pickle.load(inp)
   
    u_NN = np.load(os.path.join(data_path,"u_hat.npy"))
    u = np.load(f"results/{number}/u_FD.npy")
    t = np.load(f"results/{number}/t_series.npy")
    x = np.load(f"results/{number}/x_series.npy")
    return model, u, u_NN, t, x

def load_bayes(number):
    path = os.path.join("results", str(number))
    mean = np.load(os.path.join(path, "mean.npy"))
    median = np.load(os.path.join(path, "median.npy"))
    std = np.load(os.path.join(path, "std.npy"))
    lower = np.load(os.path.join(path, "lower.npy"))
    upper = np.load(os.path.join(path, "upper.npy"))
    return mean, median, std, lower, upper


def vis_FD_NN(u_FD:np.ndarray, u_NN:np.ndarray,
    t:np.ndarray, x:np.ndarray, save_path = None):

    fig, ax = plt.subplots(2, 2)
    fig.set_size_inches(17,16)
    
    title_c = r"$c(x,t) \left[\frac{\mu g}{cm^3}\right]$"
    title_s = r"$s_k(x,t) \left[\frac{\mu g}{g}\right]$"

    __add_fig(fig=fig, ax=ax, row=0, column=0, title=r"$D_{synth}$: " + title_c, 
        value=u_FD[...,0], x=x, t=t, is_c=True)
    __add_fig(fig=fig, ax=ax, row=1, column=0, title=r"$D_{synth}$: " + title_s, 
        value=u_FD[...,1], x=x, t=t)
    __add_fig(fig=fig, ax=ax, row=0, column=1, title=r"BNN in FINN: " +title_c, 
        value=u_NN[...,0], x=x, t=t, is_c = True)
    __add_fig(fig=fig, ax=ax, row=1, column=1, title=r"BNN in FINN: " +title_s, 
        value=u_NN[...,1], x=x, t=t)
    
    if save_path is not None:
        plt.savefig(save_path)
    else:
        plt.show()

def vis_diff(u_FD:np.ndarray, u_NN:np.ndarray, t:np.ndarray, 
             x:np.ndarray,squared:bool = False, log_value:bool = False,
             save_path = None):
    """calculates difference of u_NN and u_FD solution

    Args:
        model (_type_): _description_
        u_FD (np.ndarray): _description_
        u_NN (np.ndarray): _description_
        t (np.ndarray): _description_
        x (np.ndarray): _description_

    Log value without square is forbidden
    """
    if squared:
        diff_c = (u_FD[...,0] - u_NN[...,0])**2
        diff_sk = (u_FD[...,1] - u_NN[...,1])**2
        if log_value:
            diff_c = np.log10(diff_c)
            diff_sk = np.log10(diff_sk)
    else:
        diff_c = np.abs(u_FD[...,0] - u_NN[...,0])
        diff_sk = np.abs(u_FD[...,1] - u_NN[...,1])

    fig, ax = plt.subplots(1,2)
    fig.set_size_inches(19,8)
    ax = np.expand_dims(ax, axis=0)
    # print(ax.shape)
    

    #Generate titles for plots
    # title_c, title_s = generate_plot_title(squared, log_value)

    title_c = r"$c_{upper} - c_{lower} \left[\frac{\mu g}{cm^3}\right]$"
    title_s = r"$s_{k,upper} - s_{k,lower} \left[\frac{\mu g}{g}\right]$"

    __add_fig(fig=fig, ax=ax, row=0, column=0, title= title_c,
        value=diff_c, x=x, t=t, is_c = True)
    __add_fig(fig=fig, ax=ax, row=0, column=1, title= title_s,
        value=diff_sk, x=x, t=t, is_c = False)
    
    if save_path is not None:
        plt.savefig(save_path)
    else:
        plt.show()

def generate_plot_title(squared, log_value):
    title_c = r"$"
    title_s = r"$"
    if squared:
        if log_value:
            title_c += r"\log("
            title_s += r"\log("
        title_c += r"("
        title_s += r"("
    title_c += r"c_{D_{synth}} - c_{BNN}"
    title_s += r"s_{k, D_{synth}} - s_{k, BNN}"
    if squared:
        title_c += r")^2 "
        title_s += r")^2 "
        if log_value:
            title_c += r")"
            title_s += r")"
    title_c += r"\left[\frac{\mu g}{cm^3}\right]$"
    title_s += r"\left[\frac{\mu g}{g}\right]$"
    return title_c,title_s

def vis_btc(u, u_hat, t, lower = None, upper = None, save_path = None, experimental_data:bool = False):
    font_size =  FONT_SIZE
    fig, ax = plt.subplots(1,1)
    fig.set_size_inches(8,6)
    # plot BTC
    ax.set_xlabel("t [d]", fontsize=font_size)
    ax.set_ylabel("$c \left[\\frac{\mu g}{cm^3}\\right]$", fontsize=font_size)
    # ax.set_title("Conc. of PFOS at outflow", fontsize=font_size)
    # ax.set_yscale('log')
    
    if experimental_data:
        x = np.array([0,3,6,9,13,16,20,23,27,30,34,37,41,45,48,52,55,69,83,97,142,142])
        x[1:-1] = (x[1:-1] + x[:-2])/2 
        y = np.array([0,0.03,0.023,0.0099,0.0051,0.0017,0.00054,0.00073,0.00051,0.00035,0.00025,0.00019,0.00015,0.00011,0.000093,0.000061,0.000068,0.000052,0.00004,0.000029,0.000031,0.000015])

        # y = [0,30000,23000,9900,5100,1700,540,730,510,350,250,190,150,110,93,61,68,52,40,29,31]
        ax.plot(x,y, '-o', label = 'Experimental data')
    else:
        ax.plot(t, u[-1,:,0], label=r"$D_{synth}$")

    ax.plot(t, u_hat[-1,:,0], label="BNN in FINN")

    if upper is not None and lower is not None:
        print(upper[-1,:,0] - lower[-1,:,0])
        ax.fill_between(t, lower[-1,:,0], upper[-1,:,0], color='orange',label='BNN margin')


    ax.legend(fontsize=font_size)
    plt.locator_params(axis="x", nbins=6)
    for label in (ax.get_xticklabels() + ax.get_yticklabels()): label.set_fontsize(font_size)

    #ax[0].set_yscale("log")

    if save_path is not None:
        plt.savefig(os.path.join(save_path,"BTC.pdf"))
    else:
        plt.show()

def vis_sk_end(u, u_hat, t, x, config_NN, save_path = None):
    font_size = FONT_SIZE
    fig, ax = plt.subplots(1,1)

    # plot sk end
    ax.set_xlabel("$s_k \:\left[\\frac{\mu g}{g}\\right]$", fontsize=font_size)
    ax.set_ylabel("$x [cm]$", fontsize=font_size)
    ax.set_title(f"Kin. sorbed conc. of PFOS at t = {t[-1]}d", fontsize=font_size)
    if config_NN.data.name == "data_ext":
        ax.plot(np.flip(u[:,-1,1]), x, color="b", label="FD")
    elif config_NN.data.name == "data_exp":
        ax.plot(np.flip(u[:,-1,1]), x, color="b", label="Experimental data")
    ax.plot(np.flip(u_hat[:,-1,1]), x, color="y", label="FINN")

    
    ax.legend(fontsize=font_size)
    plt.locator_params(axis="x", nbins=6)
    for label in (ax[1].get_xticklabels() + ax[1].get_yticklabels()): label.set_fontsize(font_size)
    
    if save_path is not None:
        plt.savefig(os.path.join(save_path,"BTC.pdf"))
    else:
        plt.show()

def vis_sorption_isotherms(model, u, u_hat, t, x, config_NN:Configuration):
    font_size = FONT_SIZE
    fig, ax = plt.subplots()
    dt = t[1]-t[0]
    # plot sk over cw
    modulo_timesteps = np.ceil(len(t)/10)
    for timestep in range(1,len(t)):
        if timestep%modulo_timesteps == 0:
            color = next(ax._get_lines.prop_cycler)['color']
            ax.plot(u_hat[model.x_start:model.x_stop,timestep,0], u_hat[model.x_start:model.x_stop,timestep,1], label=f"FINN: {np.round(timestep*dt, 2)}d", color=color)
            if config_NN.data.name == "data_ext":
                ax.plot(u[model.x_start:model.x_stop,timestep,0], u[model.x_start:model.x_stop,timestep,1], "--", label=f"FD: {np.round(timestep*dt, 2)}d",  color=color, alpha=0.9)

    ax.set_xlabel("$c \left[\\frac{\mu g}{cm^3}\\right]$", fontsize = font_size)
    ax.set_ylabel("$s_k \left[\\frac{\mu g}{g}\\right]$", fontsize = font_size)
    ax.set_title("$s_k$ vs. $c$ at different times", fontsize=font_size)
    ax.legend(fontsize=font_size, loc='center left', bbox_to_anchor=(1, 0.5))
    for label in (ax.get_xticklabels() + ax.get_yticklabels()): label.set_fontsize(font_size)
    plt.show()







FONT_SIZE = 12

if __name__ == "__main__":
    config = Configuration("config.json")
    number = config.model.number
    
    # TODO: Hier aufräumen
    # params_NN = Configuration(f"results/{number}/params_NN.json") 
    params_FD = Configuration(f"results/{number}/init_params.json")
    config_NN = Configuration(f"results/{number}/config_NN.json")

    # load NN data
    model, u, u_NN, t, x = load_data(number, config_NN)
    
    # load bayes data
    if config.bayes.is_bayes:
        mean, median, std, lower, upper = load_bayes(number)
        

    
    #Save or print (if path is None)
    save_path = os.path.join(os.path.dirname(__file__), "visualize", str(number))
    os.makedirs(save_path, exist_ok=True)

    # visualize
    IS_EXP = False
    #print model
    # print(model.__dict__)
    vis_FD_NN(u, u_NN, t, x, os.path.join(save_path,"FD_NN.pdf"))
    vis_diff(u, u_NN, t, x, save_path = os.path.join(save_path,"Diff.pdf"))
    # vis_diff(u, u_NN, t, x, squared = True, save_path = os.path.join(save_path,"DiffSq.pdf"))

    # vis_diff(u, u_NN, t, x, squared = True, 
    #          log_value=True, save_path = os.path.join(save_path,"DiffLog.pdf"))
    vis_btc( u, u_NN, t, x, save_path = save_path,experimental_data = IS_EXP)

    if config.bayes.is_bayes:
        if save_path is not None:
            bnn_path = os.path.join(save_path, "bayes")
        else:
            bnn_path = None
        os.makedirs(bnn_path, exist_ok=True)
        vis_diff(lower, upper, t, x, save_path= os.path.join(bnn_path,"Diff_Lower_Upper.pdf"))
        # vis_diff(u, median, t, x, squared = True, save_path = os.path.join(bnn_path,"Bayes_Diff.pdf"))
        # vis_diff(u, lower, t, x, squared = True, save_path = os.path.join(bnn_path,"Bayes_LowerDiffSq.pdf"))
        # vis_diff(u, upper, t, x, squared = True, save_path = os.path.join(bnn_path,"Bayes_UpperDiffSq.pdf"))
        # vis_FD_NN(u,  - , t, x, save_path = os.path.join(bnn_path,"Bayes_DifferenceLowerUpper.pdf"))
        vis_btc(u,mean,t,lower,upper,bnn_path,experimental_data = IS_EXP)



    # vis_sorption_isotherms(model, u, u_NN, t, x, config_NN)

    '''
    Time in days
    [0,3,6,9,13,16,20,23,27,30,34,37,41,45,48,52,55,69,83,97,142]
    Amount PFOS in [\mu g/cm^3]
    [0,30000,23000,9900,5100,1700,540,730,510,350,250,190,150,110,93,61,68,52,40,29,31]

    x = [0,3,6,9,13,16,20,23,27,30,34,37,41,45,48,52,55,69,83,97,142
    y = [0,30000,23000,9900,5100,1700,540,730,510,350,250,190,150,110,93,61,68,52,40,29,31]
    plt.plot(x,y)
    plt.yscale('log')
    plt.show()
    '''