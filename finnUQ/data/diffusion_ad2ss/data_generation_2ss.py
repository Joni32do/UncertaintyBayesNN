# @File          :   data_generation_2ss.py
# @Last modified :   2022/12/05 11:56:14
# @Author        :   Matthias Gueltig

"""
This script can be used to solve Advection-Diffusion equation with Two-Site
sorption model.
Training data (1/4) of simulation time is stored in the folder data_train
Test data (whole simulation time) is stored in the folder data_test
"""

import os
import shutil
import sys
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable

from simulator_2ss import Simulator

sys.path.append("..")
from utils.configuration import Configuration


def generate_sample(simulator: Simulator, visualize_data: bool,
                    save_data: bool, train_data: bool, root_path: str,
                    samples:int = 100, train_noisy:bool = False):
    """This function generates a data sample, visualizes it if desired and saves
    the data to file if desired.

    Args:
        simulator (Simulator): The simulator object for data creation
        visualize_data (bool): Indicates whether to visualize the data
        save_data (bool): Indicates whether to write the data to file
        train_data (bool): Indicates whether to write (1/4) simulation time
        training data.
        root_path (str): The root path of this script
    """

    print("Generating data...")

    # Generate a data sample
    # sample_c corresponds to the dissolved concentration, sk to the kin.
    # sorbed mass concentrations
    sample_c, sample_sk = simulator.generate_sample()
    if train_noisy:
        sample_c_collection = [sample_c]
        sample_sk_collection = [sample_sk]
        for i in range(samples-1):
            sample_c_temp, sample_sk_temp = simulator.generate_sample()
            sample_c_collection.append(sample_c_temp)
            sample_sk_collection.append(sample_sk_temp)
        write_samples_to_file(root_path, simulator, sample_c_collection, sample_sk_collection, train_data)
    

    if visualize_data:
        visualize_sample(sample_c=sample_c,
                         sample_sk=sample_sk,
                         simulator=simulator)
    if save_data and not train_noisy:
        write_data_to_file(root_path=root_path, simulator=simulator,
                           sample_c=sample_c, sample_sk=sample_sk,
                           train_data=train_data)


def write_data_to_file(root_path: str, simulator: Simulator,
                       sample_c: np.ndarray, sample_sk: np.ndarray,
                       train_data: bool):
    """Writes the given data to the according directory in .npy format.

    Args:
        root_path (str): The root_path of the script.
        simulator (Simulator): The simulataor that created the data.
        sample_c (np.ndarray): The dissolved concentration to be written to
        file.
        sample_sk (np.ndarray): The kinetically sorbed concentration to be
        written to file.
        train_data (bool): Indicates wheter to save (1/4) simulation time
        training data.
    """

    # Make new folder in FINN framework in order to access parameters.
    # If folder already exists, ignore.
    params = Configuration("params.json")
    os.makedirs(f"../../models/finn/results/{params.number}", exist_ok=True)

    # Store parameters.
    shutil.copyfile("params.json",
                    f"../../models/finn/results/{params.number}/init_params.json")

    # Stack solution and store it as .npy file.
    # u_FD.shape: (x, t, 2)
    u_FD = np.stack((sample_c, sample_sk), axis=-1)
    np.save(file=f"../../models/finn/results/{params.number}/u_FD.npy",
            arr=u_FD)

    # Write the t- and x-series data.
    np.save(file=f"../../models/finn/results/{params.number}/t_series.npy",
            arr=simulator.t)
    np.save(file=f"../../models/finn/results/{params.number}/x_series.npy",
            arr=simulator.x)

    # Save if necessary training data.
    if train_data:

        # Create the data directory for the training data if it does not yet
        # exist
        data_path = os.path.join(root_path, "data"+"_train")
        os.makedirs(data_path, exist_ok=True)

        # Write the t- and x-series data along with the sample to file, (1/4)
        # of simulation time.
        np.save(file=os.path.join(data_path, "t_series.npy"),
                arr=simulator.t[:len(simulator.t)//4 + 1])
        np.save(file=os.path.join(data_path, "x_series.npy"), arr=simulator.x)
        np.save(file=os.path.join(data_path, "sample_c.npy"),
                arr=sample_c[:, :len(simulator.t)//4 + 1])
        np.save(file=os.path.join(data_path, "sample_sk.npy"),
                arr=sample_sk[:, :len(simulator.t)//4 + 1])

        # Create the data directory for the extrapolation data if it does not
        # yet exist.
        data_path = os.path.join(root_path, "data"+"_ext")
        os.makedirs(data_path, exist_ok=True)

        # Write the t- and x-series data along with the sample to file, whole
        # simulation time.
        np.save(file=os.path.join(data_path, "t_series.npy"), arr=simulator.t)
        np.save(file=os.path.join(data_path, "x_series.npy"), arr=simulator.x)
        np.save(file=os.path.join(data_path, "sample_c.npy"), arr=sample_c)
        np.save(file=os.path.join(data_path, "sample_sk.npy"), arr=sample_sk)

    # Only save whole simulation time.
    else:

        # Create the data directory if it does not yet exist
        data_path = os.path.join(root_path, "data"+"_test")
        os.makedirs(data_path, exist_ok=True)

        # Write the t- and x-series data along with the sample to file
        np.save(file=os.path.join(data_path, "t_series.npy"), arr=simulator.t)
        np.save(file=os.path.join(data_path, "x_series.npy"), arr=simulator.x)
        np.save(file=os.path.join(data_path, "sample_c.npy"), arr=sample_c)
        np.save(file=os.path.join(data_path, "sample_sk.npy"), arr=sample_sk)


def write_samples_to_file(root_path: str, simulator: Simulator,
                       sample_c_collection: np.ndarray, sample_sk_collection: np.ndarray,
                       train_data: bool):
    """Writes the given collection of samples to the according directory in .npy format.

    Args:
        root_path (str): The root_path of the script.
        simulator (Simulator): The simulataor that created the data.
        sample_c (np.ndarray): The dissolved concentration to be written to
        file.
        sample_sk (np.ndarray): The kinetically sorbed concentration to be
        written to file.
        train_data (bool): Indicates wheter to save (1/4) simulation time
        training data.
    """

    # Make new folder in FINN framework in order to access parameters.
    # If folder already exists, ignore.
    params = Configuration("params.json")
    os.makedirs(f"../../models/finn/results/{params.number}", exist_ok=True)

    # Store parameters.
    shutil.copyfile("params.json",
                    f"../../models/finn/results/{params.number}/init_params.json")

    # Stack solution and store it as .npy file.
    # u_FD.shape: (x, t, 2)
    sample_c = np.array(sample_c_collection)
    sample_sk = np.array(sample_sk_collection)
    u_FD = np.stack((sample_c, sample_sk), axis=-1)
    np.save(file=f"../../models/finn/results/{params.number}/u_FD.npy",
            arr=u_FD)

    # Write the t- and x-series data.
    np.save(file=f"../../models/finn/results/{params.number}/t_series.npy",
            arr=simulator.t)
    np.save(file=f"../../models/finn/results/{params.number}/x_series.npy",
            arr=simulator.x)
    
    ###Noises
    np.save(file=f"../../models/finn/results/{params.number}/x_series.npy",
            arr=simulator.noises)

    # Save if necessary training data.
    if train_data:

        # Create the data directory for the training data if it does not yet
        # exist
        data_path = os.path.join(root_path, "data"+"_train")
        os.makedirs(data_path, exist_ok=True)

        # Write the t- and x-series data along with the sample to file, (1/4)
        # of simulation time.
        np.save(file=os.path.join(data_path, "t_series.npy"),
                arr=simulator.t[:len(simulator.t)//4 + 1])
        np.save(file=os.path.join(data_path, "x_series.npy"), arr=simulator.x)
        np.save(file=os.path.join(data_path, "sample_c.npy"),
                arr=sample_c[:, :len(simulator.t)//4 + 1])
        np.save(file=os.path.join(data_path, "sample_sk.npy"),
                arr=sample_sk[:, :len(simulator.t)//4 + 1])

        # Create the data directory for the extrapolation data if it does not
        # yet exist.
        data_path = os.path.join(root_path, "data"+"_ext")
        os.makedirs(data_path, exist_ok=True)

        # Write the t- and x-series data along with the sample to file, whole
        # simulation time.
        np.save(file=os.path.join(data_path, "t_series.npy"), arr=simulator.t)
        np.save(file=os.path.join(data_path, "x_series.npy"), arr=simulator.x)
        np.save(file=os.path.join(data_path, "sample_c.npy"), arr=sample_c)
        np.save(file=os.path.join(data_path, "sample_sk.npy"), arr=sample_sk)

    # Only save whole simulation time.
    else:

        # Create the data directory if it does not yet exist
        data_path = os.path.join(root_path, "data"+"_test")
        os.makedirs(data_path, exist_ok=True)

        # Write the t- and x-series data along with the sample to file
        np.save(file=os.path.join(data_path, "t_series.npy"), arr=simulator.t)
        np.save(file=os.path.join(data_path, "x_series.npy"), arr=simulator.x)
        np.save(file=os.path.join(data_path, "sample_c.npy"), arr=sample_c)
        np.save(file=os.path.join(data_path, "sample_sk.npy"), arr=sample_sk)


def visualize_sample(sample_c: np.ndarray, sample_sk: np.ndarray,
                     simulator: Simulator):
    """Method to visualize a single sample. Code taken and modified from
    https://github.com/maziarraissi/PINNs

    Args:
        sample_c (np.ndarray): The dissolved conc. for visualization.
        sample_sk (np.ndarray): The kin. sorbed conc. for visualization.
        simulator (Simulator): The simulator used for data generation.
    """
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))

    # c(t, x) space and time
    font_size = 22
    h = ax[0].imshow(sample_c, interpolation="none",
                     extent=[simulator.t.min(), simulator.t.max(),
                             simulator.x.min(), simulator.x.max()],
                     origin='upper', aspect='auto')

    divider = make_axes_locatable(ax[0])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(h, cax=cax)
    cbar.ax.tick_params(labelsize=font_size)
    ax[0].set_xlim(0, simulator.t.max())
    ax[0].set_ylim(simulator.x.min(), simulator.x.max())
    ax[0].legend(loc="upper right", fontsize=font_size)
    ax[0].set_xlabel('$t [d]$', fontsize=font_size)
    ax[0].set_ylabel('$x [cm]$', fontsize=font_size)
    ax[0].set_title(r'$c(t,x) \left[\frac{\mu g}{cm^3}\right]$',
                    fontsize=font_size)
    ax[0].tick_params(axis='x', labelsize=font_size)
    ax[0].tick_params(axis='y', labelsize=font_size)
    plt.yticks(fontsize=font_size)
    for label in (ax[0].get_xticklabels() + ax[0].get_yticklabels()):
        label.set_fontsize(font_size)

    # s_k(t,x) space and time
    h = ax[1].imshow(sample_sk, interpolation='nearest',
                     extent=[simulator.t.min(), simulator.t.max(),
                             simulator.x.min(), simulator.x.max()],
                     origin='upper', aspect='auto')
    divider = make_axes_locatable(ax[1])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(h, cax=cax)
    cbar.ax.tick_params(labelsize=font_size)
    ax[1].set_xlim(0, simulator.t.max())
    ax[1].set_ylim(simulator.x.min(), simulator.x.max())
    ax[1].legend(loc="upper right", fontsize=font_size)
    ax[1].set_xlabel('$t [d]$', fontsize=font_size)
    ax[1].set_title(r'$s_k(t,x) \left[\frac{\mu g}{g}\right]$',
                    fontsize=font_size)
    ax[1].tick_params(axis='x', labelsize=font_size)
    ax[1].tick_params(axis='y', labelsize=font_size)
    plt.yticks(fontsize=font_size)
    for label in (ax[1].get_xticklabels() + ax[1].get_yticklabels()):
        label.set_fontsize(font_size)

    plt.show()


def main():
    """
    Main method used to create the datasets.
    """
    # Determine the root path for this script and set up a path for the data
    root_path = os.path.abspath("")

    # Meta data specification is recommended to be done in params.json, since
    # data is reused in the results folder of FINN 2SS Sorption.
    params = Configuration("params.json")

    # Perform simulation with or without sand layer
    sand = params.sandbool
    noisy = True
    if sand:
        simulator = Simulator(
            d_e=params.D_e,
            n_e=params.porosity,
            rho_s=params.rho_s,
            beta=params.beta,
            f=params.f,
            k_d=params.k_d,
            cw_0=params.init_conc,
            t_max=params.T_MAX,
            t_steps=params.T_STEPS,
            x_right=params.X_LENGTH,
            x_steps=params.X_STEPS,
            v=params.v_e,
            a_k=params.a_k,
            alpha_l=params.alpha_l,
            s_k_0=params.kin_sorb,
            sand=params.sandbool,
            n_e_sand=params.sand.porosity,
            x_start_soil=params.sand.top,
            x_stop_soil=params.sand.bot,
            alpha_l_sand=params.sand.alpha_l,
            v_e_sand=params.sand.v_e,
            is_noisy = noisy
            )
    else:
        simulator = Simulator(
            d_e=params.D_e,
            n_e=params.porosity,
            rho_s=params.rho_s,
            beta=params.beta,
            f=params.f,
            k_d=params.k_d,
            cw_0=params.init_conc,
            t_max=params.T_MAX,
            t_steps=params.T_STEPS,
            x_right=params.X_LENGTH,
            x_steps=params.X_STEPS,
            v=params.v_e,
            a_k=params.a_k,
            alpha_l=params.alpha_l,
            s_k_0=params.kin_sorb,
            sand=params.sandbool
        )

    # Create train and ext data
    generate_sample(simulator=simulator,
            visualize_data=False,
                    save_data=True,
                    train_data=True,
                    root_path=root_path,
                    samples=100, train_noisy=True)


if __name__ == "__main__":
    main()
    print("Done.")
