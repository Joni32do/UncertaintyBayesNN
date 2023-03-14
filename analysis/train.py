import os
import torch
from torch import nn

from visualize import plot_losses, plot_pretrain


def train_net(net, data, hyper, path):
    '''
    Trains a Bayesian network
        - input
            o Network 
            o Input 
            o Output
            o Hyperparameters as dictionary
                * learning rate
                * epochs
                * pretrain_epochs (if exists in parent folder .pth, will use this one)
                * sort
                * elbo_dictionary
                    o noise
                    o samples
                    o kl_weight
                * logging
            o Path to save plot of losses

        - output
            o last loss

        Uses Adam-Optimizer for training
    '''
    #Logging
    logging = hyper["logging"]
    plotting = True

    #Data
    x_train = data["x_train"]
    y_train = data["y_train"]

    #Hyperparameters
    epochs = hyper["epochs"]
    pretrain_epochs = hyper["pretrain_epochs"]
    lr = hyper["learning_rate"]
    sort = hyper["sort"]
    elbo = hyper["elbo"]


    # Define the loss function and optimizer
    optimizer = torch.optim.Adam(net.parameters(), lr)
    losses = []

    #Loss functions
    def elbo_loss(net, x_train, y_train):
        return net.sample_elbo(x_train, y_train, 
                               samples = elbo["samples"], 
                               noise = elbo["noise"], 
                               kl_weight = elbo["kl_weight"])

    def mse_loss(net, x_train, y_train):
        pred = net.forward(x_train)
        return nn.MSELoss()(y_train,pred)
    
    if hyper["loss_fn"] == "elbo":
        loss_fn = elbo_loss
    elif hyper["loss_fn"] == "mse":
        loss_fn = mse_loss
    else:
        raise NotImplementedError("Only elbo or mse is implemented")
    
    #Defines single Optimization step
    def closure():
        optimizer.zero_grad()

        # Compute the loss
        loss = loss_fn(net, x_train, y_train)
            
        # Backward pass
        loss.backward()

        return loss.item()

    #Check if pretrained Network is available
    pretrain_path = os.path.join(path, os.pardir,"pretrain")
    skip_pretrain = 0

    if os.path.isfile(pretrain_path):
        net.load_state_dict(torch.load(pretrain_path + ".pth"))
        net.set_pretrain(False)
        skip_pretrain = pretrain_epochs
    
        





    ### Training Loop
    if logging: 
        print(f"\n \t Begin training")
    
    for epoch in range(skip_pretrain, epochs):
        
        loss = optimizer.step(closure)
        losses.append(loss)


        ###Additional If-Loop Stuff

        # Change from pretrain to train
        if epoch == pretrain_epochs-1:
            change_to_training(net, data, pretrain_path, plotting)
                

        if sort:
            net.sort_bias()

        # Print the loss every 100 epochs
        if logging and (epoch + 1) % 100 == 0:
            print(f"\t \t Epoch {str(epoch + 1).rjust(len(str(epochs)),'0')}/{epochs}: Loss = {loss:.4f}")

            # for m in net.layers:
            #     print(m.mu_b.data)
            #     print(m.rho_w.data)

    
    if plotting:
        plot_losses(losses, os.path.join(path,"train.pdf"))
        
    #returns the final loss -> could also use losses
    return loss






def change_to_training(net, data, pretrain_path, plotting):
            '''
            Plots the pretrained function
            '''
            if plotting:
                 y_pred = net(data["x_train"]).detach().numpy()
                 plot_pretrain(data, y_pred, pretrain_path)
            net.set_pretrain(False)
            torch.save(net.state_dict(), pretrain_path + ".pth")
                 