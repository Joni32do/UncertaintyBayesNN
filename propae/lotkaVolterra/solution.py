from scipy import integrate
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import os

def lotkaVolterraRHS(x,t=0,params=[1,0.1,1.5,0.075]):
    '''
    Right hand side of the lotka volterra ODE
    @expects np.shape(x) = (2)
    x[0] = amount of prey
    x[1] = amount of predators
    '''
    change_prey = (params[0]-params[1]*x[1])*x[0]
    change_predator = (-params[2]+params[3]*x[0])*x[1]
    return np.array([change_prey,change_predator])

t = np.linspace(0,15,1000)
x_0 = np.array([15,3])
X, infodict = integrate.odeint(lotkaVolterraRHS,x_0, t, full_output=True)

prey,predator = X.T
fig = plt.figure()
plt.plot(t, prey, 'g-', label='prey')
plt.plot(t, predator, 'r-', label='predator')
plt.legend(loc=1)
plt.xlabel('time')
plt.ylabel('population')
plt.title('Evolution')
plt.savefig(os.path.join('lotkaVolterra','evolution.png'))



fig = plt.figure()

plt.plot(X[:,0],X[:,1],label='trajectory in phase space')
n_grid = 10
x = np.linspace(0,45,n_grid)
y = np.linspace(0,25,n_grid)
X1,Y1 = np.meshgrid(x,y)
DX1, DY1 = lotkaVolterraRHS([X1,Y1])

#Normalization
M = np.hypot(DX1,DY1)
M[M==0]=1
DX1 /= M
DY1 /=  M


Q = plt.quiver(X1,Y1,DX1,DY1,M,pivot='mid',cmap=cm.summer)
plt.legend(loc=1)
plt.savefig(os.path.join('lotkaVolterra','phase_space.png'))
plt.show()