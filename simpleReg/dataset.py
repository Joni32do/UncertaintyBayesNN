'''
Not used in the moment

I use this to test data but the main thing still happens in simpleBnn

TODO:

 * Make a class which is also iterable
 * More custom design choice

'''
import torch
import numpy as np
import matplotlib.pyplot as plt

#Copied
from matplotlib import cm



n_train = 200
n_test = 400 #better if it is a square

in_dis = 5 #Assumes symmetric distance and zero centering
out_dis = 5 #in_dis < out_dis


# 1D Regression Dataset



def f(x,aleatoric_uncertainty=0):
    '''
    cubic polynomial R1 -> R1
    '''
    return torch.pow(x,4) - torch.pow(x,2) + 5 * torch.pow(x,1) + aleatoric_uncertainty*(torch.rand(x.size())-0.5)


x_train = torch.reshape(torch.linspace(-in_dis,in_dis,n_train),(n_train,1))
y_train = f(x_train)

x_test = torch.reshape(torch.linspace(-out_dis,out_dis,n_test),(n_test,1))
y_test = f(x_test,1)

# 2D Regression Dataset


def f_2D(X,Y=None,aleatoric_uncertainty=0):
    if Y is not None:
        return torch.sin(torch.sqrt(X**2 + Y **2)) + aleatoric_uncertainty*(torch.rand(X.size())-0.5)
    else:
        return torch.sin(torch.sqrt(X[:,0]**2 + X[:,1]**2)) + aleatoric_uncertainty*(torch.rand(X.size()[0])-0.5)

x_train = in_dis * 2 * (torch.rand((n_train,2))-0.5)
z_train = f_2D(x_train)

n_test_dim = int(np.sqrt(n_test))
x1_test = torch.linspace(-out_dis,out_dis,n_test_dim)
x2_test = torch.linspace(-out_dis,out_dis,n_test_dim)
X,Y = torch.meshgrid(x1_test, x2_test)
Z = f_2D(X,Y)

fig, ax = plt.subplots(subplot_kw={"projection":"3d"})
surf = ax.plot_surface(X,Y,Z, cmap=cm.summer, linewidth=0, alpha = 0.7)
scatter = ax.scatter(x_train[:,0],x_train[:,1], z_train, marker='o')


# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)

plt.show()

