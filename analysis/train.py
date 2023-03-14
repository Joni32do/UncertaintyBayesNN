import os
import torch
from torch import nn

from visualize import plot_losses


def train_net(net, x_train, y_train, hyper, path):
    '''
    Trains a Bayesian network (def line 16)
        - lr = 0.001
    '''
    #Logging
    logging = hyper["logging"]
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
    pretrain_path = os.path.join(path, os.pardir,"pretrain.pth")
    skip_pretrain = 0

    
    if os.path.isfile(pretrain_path):
        net.load_state_dict(torch.load(pretrain_path))
        net.set_pretrain(False)
        skip_pretrain = pretrain_epochs + 1
        


    if logging: 
        print(f"\n \t Begin training")

    # Train the net for 1000 epochs
    for epoch in range(skip_pretrain, epochs):
        
        # Change from pretrain to train
        if epoch == pretrain_epochs:
            net.set_pretrain(False)
            torch.save(net.state_dict(), pretrain_path)


        loss = optimizer.step(closure)
        losses.append(loss)
        if sort:
            net.sort_bias()

        # Print the loss every 100 epochs
        if logging and (epoch + 1) % 100 == 0:
            print(f"\t \t Epoch {str(epoch + 1).rjust(len(str(epochs)),'0')}/{epochs}: Loss = {loss:.4f}")

            # for m in net.layers:
            #     print(m.mu_b.data)
            #     print(m.rho_w.data)

    plotting = True
    if plotting:
        plot_losses(losses, os.path.join(path,"train.pdf"))
        
    #returns the final loss -> could also use losses
    return loss