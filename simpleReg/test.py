import numpy as np

x = np.reshape(np.arange(8),(4,1,2))
print(x)
print(x[:,:,0])
print(x[:,:,1])

print(np.std(x,axis=-1))