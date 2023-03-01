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
# from finn.finn import *

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

    root_path = os.path.abspath("../../data")
    data_path = os.path.join(root_path, config.data.type, config.data.name)


    # Print some information to console
    print("Model name:", config.model.name)

    # Hide the GPU(s) in case the user specified to use the CPU in the config file
    if config.general.device == "CPU":
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    # Set device on GPU if specified in the configuration file, else CPU # device = helpers.determine_device()
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
        # If parameters learned: initial guesses defined in init_params.json
        params = Configuration(os.path.join("results",str(config.model.number),"init_params.json"))

        #Synthetic data
        if config.data.name == "data_train":
            u = np.load(f"results/{config.model.number}/u_FD.npy")
            t = np.load(f"results/{config.model.number}/t_series.npy")
            
            # only use (1/4) of synthetic data set
            u = u[:, :len(t) // 4 + 1, :]
            t = t[:len(t) // 4 + 1]

            # np to pytorch tensors
            u = th.tensor(u, dtype=th.float).to(device=device)
            t = th.tensor(t, dtype=th.float).to(device=device)
        else:
            # use whole synthetic data set for training
            u = th.tensor(np.load(f"results/{config.model.number}/u_FD.npy"),
                          dtype=th.float).to(device=device)
            t = th.tensor(np.load(f"results/{config.model.number}/t_series.npy"),
                          dtype=th.float).to(device=device)

        #Spatial discretization
        x = th.tensor(np.load(f"results/{config.model.number}/x_series.npy"),
                        dtype=th.float).to(device=device)
        
        # TODO:Add Noise 
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
        if config.bayes.is_bayes:
            model = FINN_DiffAD2ssBayes(
            u=u,
            D=np.array(params.alpha_l*params.v_e+params.D_e),
            BC=np.array([0.0]),
            dx=dx,
            layer_sizes=config.model.layer_sizes,
            device=device,
            mode="train",
            learn_coeff=config.learn.learn_coeff,
            learn_f=config.learn.learn_f,
            learn_f_hyd=config.learn.learn_f_hyd,
            learn_g_hyd=config.learn.learn_g_hyd,
            learn_r_hyd=config.learn.learn_r_hyd,
            learn_k_d=config.learn.learn_k_d,
            learn_beta=config.learn.learn_beta,
            learn_alpha=config.learn.learn_alpha,
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
            sigmoid=True,
            bayes_factor=config.bayes.bayes_factor,
            bayes_arc=config.bayes.bayes_sizes
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
                learn_coeff=config.learn.learn_coeff,
                learn_f=config.learn.learn_f,
                learn_f_hyd=config.learn.learn_f_hyd,
                learn_g_hyd=config.learn.learn_g_hyd,
                learn_r_hyd=config.learn.learn_r_hyd,
                learn_k_d=config.learn.learn_k_d,
                learn_beta=config.learn.learn_beta,
                learn_alpha=config.learn.learn_alpha,
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






    #Continue training .tf
    if config.training.continue_training:
        if print_progress: 
            print('Restoring model (that is the network\'s weights) from file...')
        model.load_state_dict(th.load(os.path.join(os.path.abspath(""),
                                      "checkpoints",
                                      config.model.name,
                                      config.model.name + ".pt")))
        


    #Optimizer --> Alternative th.optim.LBFGS
    optimizer = th.optim.Adam(model.parameters(), lr=config.training.learning_rate)
    
    #Loss
    criterion = nn.MSELoss(reduction="mean")
    kl_weight = 0


    # Set up lists to save and store the epoch errors
    mse_train = []
    best_train = np.infty
    


    """
    TRAINING
    """
    # Set the model to train mode -> store gradients during pass
    model.train()
    start_training = time.time()

    def closure():
            # Perform one optimization step towards direction of gradients
            #--help: Define the closure function that consists of resetting the gradient buffer, loss function calculation, and backpropagation It is necessary for LBFGS optimizer, because it requires multiple function evaluations
            
            # Reset the optimizer to clear data from previous iterations
            optimizer.zero_grad()

            # forward pass
            u_hat = model(t=t, u=u)

            #Losses
            mse = criterion(u_hat, u)

            #TODO: Add this directly to criterion and not in closure
            #TODO: This is only for Finn where R is learned
            if config.model.bayes:
                kl_divergence_loss = model.func_r.kl_loss(kl_weight)
                loss = mse + kl_divergence_loss
            else:
                loss = mse
            
            #backward pass (calc gradients)
            loss.backward()

            return mse

    
    
    
    # Start the training and iterate over all epochs
    for epoch in range(config.training.epochs):
        epoch_start_time = time.time()

        #Does one training step and appends it
        mse = optimizer.step(closure)
        mse_train.append(mse.item())


        #Bayes #TODO:
        if config.bayes.is_bayes:
            # Change from pretrain to train
            if epoch == config.bayes.pretrain_epochs:
                model.set_pretrain(False)

            #Sort bias if enabled
            if config.bayes.sort:
                model.sort_bias()
        

        
        
        # Create a plus or minus sign for the training error
        train_sign = "(-)"
        if mse_train[epoch] < best_train:
            train_sign = "(+)"
            best_train = mse_train[-1]
            # Save the model to file #TODO: A bit too often?
            if config.training.save_model:
                # Start a separate thread to save the model
                thread = Thread(target=helpers.save_model_to_file(model_src_path=os.path.abspath(""),config=config,epoch=epoch,
                    epoch_errors_train=mse_train,epoch_errors_valid=mse_train,net=model))
                thread.start()
        
        


        # Print progress to the console
        if print_progress:
            print(f'''Epoch {str(epoch+1).zfill(int(np.log10(config.training.epochs))+1)}\{config.training.epochs} \t Time: {str(np.round(time.time() - epoch_start_time, 2))} \t Error: {train_sign}{str(np.round(mse_train[-1], 10))}
            ''')
            # print(f'''Epoch {str(epoch+1).zfill(int(np.log10(config.training.epochs))+1)}/{str(config.training.epochs)} 
            #         took {str(np.round(time.time() - epoch_start_time, 2)).ljust(5, '0')} seconds. 
            #         \t\tAverage epoch training error: {train_sign}{str(np.round(mse_train[-1], 10)).ljust(12, ' ')}''')
    
    
            
    if print_progress:
        print('\nTraining took ' + str(np.round(time.time() - start_training, 2)) + ' seconds.\n\n')
    

if __name__ == "__main__":
    th.set_num_threads(1)
    run_training(print_progress=True)

    print("Done.")
