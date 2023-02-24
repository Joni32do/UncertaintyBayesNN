# @File          :   train.py
# @Last modified :   2022/12/05 17:53:42
# @Author        :   Matthias Gueltig

"""
Main file for training a model with FINN, including the 2SS - Model. For learning,
synthetic data is used
"""

import os
import sys
import time
from threading import Thread

import numpy as np
import torch as th
import torch.nn as nn
from finn import *

sys.path.append("..")
import utils.helper_functions as helpers
from utils.configuration import Configuration


def run_training(print_progress=True, model_number=None):

    # Load the user configurations
    config = Configuration("config.json")

    # Append the model number to the name of the model
    if model_number is None:
        model_number = config.model.number
    config.model.name = config.model.name + "_" + str(model_number).zfill(2)

    # Print some information to console
    print("Model name:", config.model.name)

    # Hide the GPU(s) in case the user specified to use the CPU in the config
    # file
    if config.general.device == "CPU":
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    root_path = os.path.abspath("../../data")
    data_path = os.path.join(root_path, config.data.type, config.data.name)

    # Set device on GPU if specified in the configuration file, else CPU
    # device = helpers.determine_device()
    device = th.device(config.general.device)
    if config.data.type == "burger":
        # Load samples, together with x, y, and t series
        t = th.tensor(np.load(os.path.join(data_path, "t_series.npy")),
                      dtype=th.float).to(device=device)
        x = np.load(os.path.join(data_path, "x_series.npy"))
        u = th.tensor(np.load(os.path.join(data_path, "sample.npy")),
                             dtype=th.float).to(device=device)
        
        u[1:] = u[1:] + th.normal(th.zeros_like(u[1:]),th.ones_like(u[1:])*config.data.noise)
        
        dx = x[1]-x[0]
    
        # Initialize and set up the model
        model = FINN_Burger(
            u = u,
            D = np.array([1.0]),
            BC = np.array([[0.0], [0.0]]),
            dx = dx,
            layer_sizes = config.model.layer_sizes,
            device = device,
            mode="train",
            learn_coeff=True
        ).to(device=device)
        
    elif config.data.type == "diffusion_sorption":
        # Load samples, together with x, y, and t series

        t = th.tensor(np.load(os.path.join(data_path, "t_series.npy")),
                      dtype=th.float).to(device=device)
        x = np.load(os.path.join(data_path, "x_series.npy"))

        # #size: [501, 26]
        sample_c = th.tensor(np.load(os.path.join(data_path, "sample_c.npy")),
                             dtype=th.float).to(device=device)

        # #size: [501, 26]
        sample_ct = th.tensor(np.load(os.path.join(data_path, "sample_ct.npy")),
                             dtype=th.float).to(device=device)

        # #same dx over all x
        dx = x[1]-x[0]

        # #size: [501, 26, 2]
        u = th.stack((sample_c, sample_ct), dim=len(sample_c.shape))
        # #adds noice with mu = 0, std = data.noise
        # #for all rows apart from the firs one
        # # 1 to last in all dimensions
        u[1:] = u[1:] + th.normal(th.zeros_like(u[1:]),th.ones_like(u[1:])*config.data.noise)
        
        # Initialize and set up the model
        model = FINN_DiffSorp(
            u = u,
            D = np.array([0.0005, 0.000145]),
            BC = np.array([[1.0, 1.0], [0.0, 0.0]]),
            dx = dx,
            layer_sizes = config.model.layer_sizes,
            device = device,
            mode="train",
            learn_coeff=False
        ).to(device=device)
    
    elif config.data.type == "diffusion_ad2ss":
        # Load samples, together with x, y, and t series.
        # If parameters are learned, initial guesses should be defined in the
        # init_params.json
        params = Configuration(f"results/{config.model.number}/init_params.json") #Warum hier ohne os.path
        # params = Configuration(os.path.join("results",config.model.number,"init_params.json"))

        # synthetic data is used
        if config.data.name == "data_train":
            u = np.load(f"results/{config.model.number}/u_FD.npy")
            t = np.load(f"results/{config.model.number}/t_series.npy")
            
            # only use (1/4) of synthetic data set
            u = u[:, :len(t) // 4 + 1, :]
            t = t[:len(t) // 4 + 1]
            print(u,t)

            # transform numpy arrays to pytorch tensors
            u = th.tensor(u, dtype=th.float).to(device=device)
            t = th.tensor(t, dtype=th.float).to(device=device)
        else:
            # use whole synthetic data set for training
            u = th.tensor(np.load(f"results/{config.model.number}/u_FD.npy"),
                          dtype=th.float).to(device=device)
            t = th.tensor(np.load(f"results/{config.model.number}/t_series.npy"),
                          dtype=th.float).to(device=device)

        # spatial discretization stays the same in both cases
        x = th.tensor(np.load(f"results/{config.model.number}/x_series.npy"),
                        dtype=th.float).to(device=device)

    	##################### Noise #########
        # adds noice with mu = 0, std = data.noise
        u[1:] = u[1:] + th.normal(th.zeros_like(u[1:]),
                                    th.ones_like(u[1:])*config.data.noise)
        #####################################

        # same dx over all x
        dx = x[1] - x[0]

        # Initialize and set up the Two-Site sorption model
        # Dependeing on which parameter/functional relationship should be learned
        # corresponding boolean variables have to be changed
        if config.model.bayes:
            model = FINN_DiffAD2ssBayes(
            u=u,
            D=np.array(params.alpha_l*params.v_e+params.D_e),
            BC=np.array([0.0]),
            dx=dx,
            layer_sizes=config.model.layer_sizes,
            device=device,
            mode="train",
            learn_coeff=False,
            learn_f=False,
            learn_f_hyd=False,
            learn_g_hyd=False,
            learn_r_hyd=True,
            learn_k_d=False,
            learn_beta=False,
            learn_alpha=False,
            t_steps=len(t),
            rho_s=np.array(params.rho_s),
            f=np.array(params.f),
            k_d=np.array(params.k_d),
            beta=np.array(params.beta),
            n_e=np.array(params.porosity),
            alpha=np.array(params.a_k),
            v_e=np.array(params.v_e),
            sand=params.sandbool,
            D_sand=np.array(params.sand.alpha_l*params.sand.v_e),
            n_e_sand=np.array(params.sand.porosity),
            x_start_soil=np.array(params.sand.top),
            x_stop_soil=np.array(params.sand.bot),
            x_steps_soil=np.array(params.X_STEPS),
            alpha_l_sand=np.array(params.sand.alpha_l),
            v_e_sand=np.array(params.sand.v_e),
            config=None,
            learn_stencil=False,
            bias=True,
            sigmoid=True
        ).to(device=device)
        else:
            model = FINN_DiffAD2ss(
                u=u,
                D=np.array(params.alpha_l*params.v_e+params.D_e),
                BC=np.array([0.0]),
                dx=dx,
                layer_sizes=config.model.layer_sizes,
                device=device,
                mode="train",
                learn_coeff=False,
                learn_f=False,
                learn_f_hyd=False,
                learn_g_hyd=False,
                learn_r_hyd=True,
                learn_k_d=False,
                learn_beta=False,
                learn_alpha=False,
                t_steps=len(t),
                rho_s=np.array(params.rho_s),
                f=np.array(params.f),
                k_d=np.array(params.k_d),
                beta=np.array(params.beta),
                n_e=np.array(params.porosity),
                alpha=np.array(params.a_k),
                v_e=np.array(params.v_e),
                sand=params.sandbool,
                D_sand=np.array(params.sand.alpha_l*params.sand.v_e),
                n_e_sand=np.array(params.sand.porosity),
                x_start_soil=np.array(params.sand.top),
                x_stop_soil=np.array(params.sand.bot),
                x_steps_soil=np.array(params.X_STEPS),
                alpha_l_sand=np.array(params.sand.alpha_l),
                v_e_sand=np.array(params.sand.v_e),
                config=None,
                learn_stencil=False,
                bias=True,
                sigmoid=True
            ).to(device=device)


    elif config.data.type == "diffusion_reaction":
        # Load samples, together with x, y, and t series
        t = th.tensor(np.load(os.path.join(data_path, "t_series.npy")),
                      dtype=th.float).to(device=device)
        x = np.load(os.path.join(data_path, "x_series.npy"))
        y = np.load(os.path.join(data_path, "y_series.npy"))
        sample_u = th.tensor(np.load(os.path.join(data_path, "sample_u.npy")),
                             dtype=th.float).to(device=device)
        sample_v = th.tensor(np.load(os.path.join(data_path, "sample_v.npy")),
                             dtype=th.float).to(device=device)
        
        dx = x[1]-x[0]
        dy = y[1]-y[0]
        
        u = th.stack((sample_u, sample_v), dim=len(sample_u.shape))
        u[1:] = u[1:] + th.normal(th.zeros_like(u[1:]),th.ones_like(u[1:])*config.data.noise)
    
        # Initialize and set up the model
        model = FINN_DiffReact(
            u = u,
            D = np.array([5E-4/(dx**2), 1E-3/(dx**2)]),
            BC = np.zeros((4,2)),
            dx = dx,
            dy = dy,
            layer_sizes = config.model.layer_sizes,
            device = device,
            mode="train",
            learn_coeff=True
        ).to(device=device)
    
    elif config.data.type == "allen_cahn":
        # Load samples, together with x, y, and t series
        t = th.tensor(np.load(os.path.join(data_path, "t_series.npy")),
                      dtype=th.float).to(device=device)
        x = np.load(os.path.join(data_path, "x_series.npy"))
        u = th.tensor(np.load(os.path.join(data_path, "sample.npy")),
                             dtype=th.float).to(device=device)

        u[1:] = u[1:] + th.normal(th.zeros_like(u[1:]),th.ones_like(u[1:])*config.data.noise)
        
        dx = x[1]-x[0]
    
        # Initialize and set up the model
        model = FINN_AllenCahn(
            u = u,
            D = np.array([0.6]),
            BC = np.array([[0.0], [0.0]]),
            dx = dx,
            layer_sizes = config.model.layer_sizes,
            device = device,
            mode="train",
            learn_coeff=True
        ).to(device=device)

    elif config.data.type == "burger_2d":
        # Load samples, together with x, y, and t series
        t = th.tensor(np.load(os.path.join(data_path, "t_series.npy")),
                      dtype=th.float).to(device=device)
        x = np.load(os.path.join(data_path, "x_series.npy"))
        y = np.load(os.path.join(data_path, "y_series.npy"))
        u = th.tensor(np.load(os.path.join(data_path, "sample.npy")),
                             dtype=th.float, requires_grad=False).to(device=device)
        
        u[1:] = u[1:] + th.normal(th.zeros_like(u[1:]),th.ones_like(u[1:])*config.data.noise)
        
        dx = x[1]-x[0]
        dy = y[1]-y[0]
                         
        # Initialize and set up the model
        model = FINN_Burger2D(
            u = u,
            D = np.array([1.75]),
            BC = np.zeros((4,1)),
            dx = dx,
            dy = dy,
            layer_sizes = config.model.layer_sizes,
            device = device,
            mode="train",
            learn_coeff=True
        ).to(device=device)

    # Count number of trainable parameters
    pytorch_total_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )

    if print_progress:
        print("Trainable model parameters:", pytorch_total_params)

    # If desired, restore the network by loading the weights saved in the .pt
    # file
    if config.training.continue_training:
        if print_progress: 
            print('Restoring model (that is the network\'s weights) from file...')
        model.load_state_dict(th.load(os.path.join(os.path.abspath(""),
                                      "checkpoints",
                                      config.model.name,
                                      config.model.name + ".pt")))
        model.train()

    if config.data.type == "diffusion_ad2ss":
        # Set up an optimizer and the criterion (loss)
        optimizer = th.optim.Adam(model.parameters(),
                                  lr=config.training.learning_rate)
    else:
        optimizer = th.optim.LBFGS(model.parameters(),
                                   lr=config.training.learning_rate)

    # Set up lists to save and store the epoch errors
    epoch_errors_train = []
    best_train = np.infty
    """
    TRAINING
    """
    a = time.time()

    if config.model.bayes and config.training.pretrain:
        pass

    # Start the training and iterate over all epochs
    for epoch in range(config.training.epochs):
        epoch_start_time = time.time()

        # Define the closure function that consists of resetting the
        # gradient buffer, loss function calculation, and backpropagation
        # It is necessary for LBFGS optimizer, because it requires multiple
        # function evaluations

        def closure():
            # Set the model to train mode -> store gradients during pass
            model.train()

            # Reset the optimizer to clear data from previous iterations
            optimizer.zero_grad()

            # forward pass
            u_hat = model(t=t, u=u)

            if config.model.bayes and config.model.sort:
                sort_idx = None
                for idx, l in enumerate(model.modules()):
                    if idx == 0:
                        print("Das klappt (hoffentlich)")
                        sort_idx = th.arange(0,l.in_features)
                    sort_idx = l.sort(sort_idx)

            
            mse = nn.MSELoss(reduction="mean")(u_hat, u)
            #TODO: Hier muss ich noch den KL_Loss hinzufügen

            # do backward pass
            mse.backward()

            return mse

        # Perform one optimization step towards direction of gradients
        mse = optimizer.step(closure)


        

        epoch_errors_train.append(mse.item())
        
        # Create a plus or minus sign for the training error
        train_sign = "(-)"
        if epoch_errors_train[-1] < best_train:
            train_sign = "(+)"
            best_train = epoch_errors_train[-1]
            # Save the model to file (if desired)
            if config.training.save_model:
                # Start a separate thread to save the model
                thread = Thread(target=helpers.save_model_to_file(
                    model_src_path=os.path.abspath(""),
                    config=config,
                    epoch=epoch,
                    epoch_errors_train=epoch_errors_train,
                    epoch_errors_valid=epoch_errors_train,
                    net=model))
                thread.start()

        # Print progress to the console
        if print_progress:
            print(f'''Epoch {str(epoch+1).zfill(int(np.log10(config.training.epochs))+1)}\{config.training.epochs} \t Time: {str(np.round(time.time() - epoch_start_time, 2))} \t Error: {train_sign}{str(np.round(epoch_errors_train[-1], 10))}
            ''')
            # print(f'''Epoch {str(epoch+1).zfill(int(np.log10(config.training.epochs))+1)}/{str(config.training.epochs)} 
            #         took {str(np.round(time.time() - epoch_start_time, 2)).ljust(5, '0')} seconds. 
            #         \t\tAverage epoch training error: {train_sign}{str(np.round(epoch_errors_train[-1], 10)).ljust(12, ' ')}''')
    
    b = time.time()
    if print_progress:
        print('\nTraining took ' + str(np.round(b - a, 2)) + ' seconds.\n\n')
    

if __name__ == "__main__":
    th.set_num_threads(1)
    run_training(print_progress=True)

    print("Done.")
