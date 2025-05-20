# xor_models.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class XOR_ReLU(nn.Module):
    def __init__(self):
        super(XOR_ReLU, self).__init__()
        self.linear1 = nn.Linear(2, 2)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(2, 2)
        
        # For ReLU and Abs: Kaiming Normal is appropriate
        nn.init.kaiming_normal_(self.linear1.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.linear2.weight, nonlinearity='relu')
        nn.init.constant_(self.linear1.bias, 0.0)  # Biases initialized to zero to center decision boundaries
        nn.init.constant_(self.linear2.bias, 0.0)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x

class XOR_Abs(nn.Module):
    def __init__(self):
        super(XOR_Abs, self).__init__()
        self.linear1 = nn.Linear(2, 1)
        self.linear2 = nn.Linear(1, 2)
        
        # Abs is similar to ReLU so use Kaiming: Kaiming Normal is appropriate
        nn.init.kaiming_normal_(self.linear1.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.linear2.weight, nonlinearity='relu')
        nn.init.constant_(self.linear1.bias, 0.0)  # Biases initialized to zero to center decision boundaries
        nn.init.constant_(self.linear2.bias, 0.0)
        
    def forward(self, x):
        x = self.linear1(x)
        x = torch.abs(x)
        x = self.linear2(x)
        return x

class XOR_Sigmoid(nn.Module):
    def __init__(self):
        super(XOR_Sigmoid, self).__init__()
        self.linear1 = nn.Linear(2, 2)
        self.sigmoid = nn.Sigmoid()
        self.linear2 = nn.Linear(2, 2)
        
        # For Sigmoid: Xavier Normal is appropriate
        nn.init.xavier_normal_(self.linear1.weight, gain=1.0)
        nn.init.xavier_normal_(self.linear2.weight, gain=1.0)
        nn.init.constant_(self.linear1.bias, 0.0)  # Biases initialized to zero to center decision boundaries
        nn.init.constant_(self.linear2.bias, 0.0)

    def forward(self, x):
        x = self.linear1(x)
        x = self.sigmoid(x)
        x = self.linear2(x)
        return x
