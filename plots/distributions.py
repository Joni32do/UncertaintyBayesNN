import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stat

from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)

x = np.random.rand(20,2)
print(x)
def f(x):
    dist = lambda x: 0
    for i in range(np.size(x,0)):
        dist = lambda x: dist(x) + stat.multivariate_normal(x[i,:],np.eye(np.shape(x,1))).pdf(x)
    return dist

def wrapper(f,X,Y):
    Z = np.zeros(np.shape(X))
    for i in range(np.size(X,0)):
        for j in range(np.size(X,1)):
            x = np.array([X[i,j], Y[i,j]])
            Z[i,j] = f(x)
    return Z

print(f([0,0]))
X,Y = np.meshgrid(np.linspace(-1,1,1000), np.linspace(-1,1,1000))
# Z = np.dstack((X,Y))
fig,ax = plt.subplots(subplot_kw={"projection": "3d"})
surf = ax.plot_surface(X,Y,wrapper(f,X,Y))
plt.show()