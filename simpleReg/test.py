import numpy as np
import torch

y = torch.ones((4,4))
print(y.size())
print(y.size() == (4,3))
print(y.numel())
print(torch.numel(5))

# x = np.reshape(np.arange(8),(4,1,2))
# print(x)
# print(x[:,:,0])
# print(x[:,:,1])

# print(np.std(x,axis=-1))