import torch
import torch.nn as nn
import numpy as np

from torch.autograd import Variable

class Dual_Lstm(nn.Module):
    def __init__(self):
        super(Dual_Lstm,self).__init__()
        self.embedding = nn.Embedding(2, 64)
        self.lstm = nn.LSTM(2,128,8)
        self.lstm_2 = nn.LSTM(128,128,12)
        self.mlp = nn.Linear(128,12*2)
    def forward(self, x):
        #input = self.embedding((x))
        x1, _ = self.lstm(x)
        # a,b,c = x1.shape
        # out = self.out(x1.view(-1,c))
        h_nodes, c_nodes = self.lstm_2(x1)

        x3 = self.mlp(h_nodes[:,-1,:])
        x4 = x3.view(12,2)
        #out1 = out.view(a,b,-1)
        return x4




