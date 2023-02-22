import torch.nn as nn
import torch
from bayesianLinear import BayesianLinear

class BayesianNet(nn.Module):
    def __init__(self):
        super(BayesianNet, self).__init__()
        self.layer1 = BayesianLinear(1, 10)
        self.layer2 = BayesianLinear(10, 1)


    def forward(self, x):
        x = torch.tanh(self.layer1(x))
        x = self.layer2(x)
        return x