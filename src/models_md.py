import torch
import torch.nn as nn
import torch.nn.functional as F

class MD_ReLU(nn.Module):
    def __init__(self, input_dim, weight=None, bias=None):
        super(MD_ReLU, self).__init__()
        self.linear = nn.Linear(input_dim, 2 * input_dim)

        if weight is not None and bias is not None:
            assert weight.shape == self.linear.weight.shape, "Weight shape mismatch"
            assert bias.shape == self.linear.bias.shape, "Bias shape mismatch"
            with torch.no_grad():
                self.linear.weight.copy_(weight)
                self.linear.bias.copy_(bias)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.linear(x)   # (batch_size, 2N)
        x = self.relu(x)
        x = x ** 2
        x = x.sum(dim=1)
        x = torch.sqrt(x)
        return x

class MD_Abs(nn.Module):
    def __init__(self, input_dim, weight=None, bias=None):
        super(MD_Abs, self).__init__()
        self.linear = nn.Linear(input_dim, input_dim)

        if weight is not None and bias is not None:
            assert weight.shape == self.linear.weight.shape
            assert bias.shape == self.linear.bias.shape
            with torch.no_grad():
                self.linear.weight.copy_(weight)
                self.linear.bias.copy_(bias)

    def forward(self, x):
        x = self.linear(x)
        x = torch.abs(x)
        x = x ** 2
        x = x.sum(dim=1)
        x = torch.sqrt(x)
        return x

class MD_Sigmoid(nn.Module):
    def __init__(self, input_dim, weight=None, bias=None):
        super(MD_Sigmoid, self).__init__()
        self.linear = nn.Linear(input_dim, 2 * input_dim)

        if weight is not None and bias is not None:
            assert weight.shape == self.linear.weight.shape
            assert bias.shape == self.linear.bias.shape
            with torch.no_grad():
                self.linear.weight.copy_(weight)
                self.linear.bias.copy_(bias)

        self.relu = nn.Sigmoid()

    def forward(self, x):
        x = self.linear(x)
        x = self.relu(x)
        x = x ** 2
        x = x.sum(dim=1)
        x = torch.sqrt(x)
        return x
