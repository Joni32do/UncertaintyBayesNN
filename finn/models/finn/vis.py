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

from mpl_toolkits.axes_grid1 import make_axes_locatable
import sys

import pickle5 as pickle

sys.path.append("..")
print(sys.path)

from utils.configuration import Configuration
from finn import *
import pandas as pd



def __add_fig(fig, ax, row:float, column:float, title:str, value:np.ndarray, 
    x:np.ndarray, t:np.ndarray):
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
    font_size = 22
    h = ax[row, column].imshow(value, interpolation='nearest', 
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
    ax[row, column].set_title(title, fontsize = font_size)
    for label in (ax[row, column].get_xticklabels() + ax[row, column].get_yticklabels()): label.set_fontsize(font_size)

def init_model(number:float, config_NN:Configuration):
    """Loads selected model

    Args:
        number (float): model number

    Returns:
        u_NN: NN calculated solution
        u_init_NN: c and sk as initialized in params.json before starting the NN
    """
    with open(f"results/{number}/model.pkl", "rb") as inp:
        model = pickle.load(inp)

    u_NN = np.load(f"results/{number}/u_hat.npy")
    u = np.load(f"results/{number}/u_FD.npy")
    t = np.load(f"results/{number}/t_series.npy")
    x = np.load(f"results/{number}/x_series.npy")
    return model, u, u_NN, t, x

def vis_FD_NN(model, u_FD:np.ndarray, u_NN:np.ndarray,
    t:np.ndarray, x:np.ndarray, config_NN):

    fig, ax = plt.subplots(2, 2)

    
    title_c = r"FD: $c(t,x) \left[\frac{\mu g}{cm^3}\right]$"
    title_sk = r"FD: $s_k(t,x) \left[\frac{\mu g}{g}\right]$"
    
    __add_fig(fig=fig, ax=ax, row=0, column=0, title=title_c, 
        value=u_FD[...,0], x=x, t=t)
    __add_fig(fig=fig, ax=ax, row=1, column=0, title=title_sk, 
        value=u_FD[...,1], x=x, t=t)
    __add_fig(fig=fig, ax=ax, row=0, column=1, title=r"FINN: $c(t,x) \left[\frac{\mu g}{cm^3}\right]$", 
        value=u_NN[...,0], x=x, t=t)
    __add_fig(fig=fig, ax=ax, row=1, column=1, title=r"FINN: $s_k(t,x) \left[\frac{\mu g}{g}\right]$", 
        value=u_NN[...,1], x=x, t=t)

    plt.show()

def vis_diff(model, u_FD:np.ndarray, u_NN:np.ndarray, t:np.ndarray, x:np.ndarray, config_NN:Configuration):
    """calculates difference of u_NN and u_FD solution

    Args:
        model (_type_): _description_
        u_FD (np.ndarray): _description_
        u_NN (np.ndarray): _description_
        t (np.ndarray): _description_
        x (np.ndarray): _description_
    """
    diff_c = u_FD[...,0] - u_NN[...,0]
    diff_sk = u_FD[...,1] - u_NN[...,1]

    fig, ax = plt.subplots(1,2)
    ax = np.expand_dims(ax, axis=0)
    print(ax.shape)
    __add_fig(fig=fig, ax=ax, row=0, column=0, title=r"$c_{FD} - c_{FINN} \left[\frac{\mu g}{cm^3}\right]$",
        value=diff_c, x=x, t=t)
    __add_fig(fig=fig, ax=ax, row=0, column=1, title=r"$s_{k, FD} - s_{k, FINN} \left[\frac{\mu g}{g}\right]$",
        value=diff_sk, x=x, t=t)
    plt.show()

def vis_btc(model, u, u_hat, t, x, config_NN:Configuration):
    fig, ax = plt.subplots(1,2)
    font_size = 22

    # plot BTC
    ax[0].set_xlabel("t [d]", fontsize=font_size)
    ax[0].set_ylabel("$c \left[\\frac{\mu g}{cm^3}\\right]$", fontsize=font_size)
    ax[0].set_title("Conc. of PFOS at outflow", fontsize=font_size)
    
    ax[0].plot(t, u[-1,:,0], color="b", label="FD")
    ax[0].plot(t, u_hat[-1,:,0], color="y", label="FINN")

    
    # plot sk end
    ax[1].set_xlabel("$s_k \:\left[\\frac{\mu g}{g}\\right]$", fontsize=font_size)
    ax[1].set_ylabel("$x [cm]$", fontsize=font_size)
    ax[1].set_title(f"Kin. sorbed conc. of PFOS at t = {t[-1]}d", fontsize=font_size)
    if config_NN.data.name == "data_ext":
        ax[1].plot(np.flip(u[:,-1,1]), x, color="b", label="FD")
    elif config_NN.data.name == "data_exp":
        ax[1].plot(np.flip(u[:,-1,1]), x, color="b", label="Experimental data")
    ax[1].plot(np.flip(u_hat[:,-1,1]), x, color="y", label="FINN")

    ax[0].legend(fontsize=font_size)
    plt.locator_params(axis="x", nbins=6)
    ax[1].legend(fontsize=font_size)
    plt.locator_params(axis="x", nbins=6)
    for label in (ax[0].get_xticklabels() + ax[0].get_yticklabels()): label.set_fontsize(font_size)
    for label in (ax[1].get_xticklabels() + ax[1].get_yticklabels()): label.set_fontsize(font_size)
    #ax[0].set_yscale("log")
    plt.show()

def vis_sorption_isotherms(model, u, u_hat, t, x, config_NN:Configuration):
    font_size = 22
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


def vis_data(number):

    # load NN params
    params_NN = Configuration(f"results/{number}/params_NN.json")
    params_FD = Configuration(f"results/{number}/init_params.json")
    config_NN = Configuration(f"results/{number}/config_NN.json")

    # load NN data
    model, u, u_NN, t, x = init_model(number, config_NN)

    # visualize
    print(model.__dict__)
    vis_FD_NN(model, u, u_NN, t, x, config_NN)
    vis_diff(model, u, u_NN, t, x, config_NN)
    vis_btc(model, u, u_NN, t, x, config_NN)
    vis_sorption_isotherms(model, u, u_NN, t, x, config_NN)


if __name__ == "__main__":

    vis_data(number=30)