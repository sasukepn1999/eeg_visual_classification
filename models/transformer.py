#Original model presented in: C. Spampinato, S. Palazzo, I. Kavasidis, D. Giordano, N. Souly, M. Shah, Deep Learning Human Mind for Automated Visual Classification, CVPR 2017 
import sys
import os
import random
import math
import time
import torch; torch.utils.backcompat.broadcast_warning.enabled = True
from torchvision import transforms, datasets
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
import torch.backends.cudnn as cudnn; cudnn.benchmark = True
import numpy as np

class Model(nn.Module):

    def __init__(self, input_size=128, trans_layers=1, output_size=128):
        # Call parent
        super().__init__()
        # Define parameters
        self.input_size = input_size
        #self.d_model = d_model
        self.trans_layers = trans_layers
        self.output_size = output_size

        # Define internal modules
        self.encoderlayer = nn.TransformerEncoderLayer(d_model=128, nhead=8)
        self.encoder = nn.TransformerEncoder(self.encoderlayer, trans_layers)
        #self.output = nn.Linear(input_size, output_size)
        self.classifier = nn.Linear(output_size,40)
        
    def forward(self, x):
        # Prepare LSTM initiale state
        batch_size = x.size(0)
        #lstm_init = (torch.zeros(self.lstm_layers, batch_size, self.lstm_size), torch.zeros(self.lstm_layers, batch_size, self.lstm_size))
        #if x.is_cuda: lstm_init = (lstm_init[0].cuda(), lstm_init[0].cuda())
        #lstm_init = (lstm_init[0], lstm_init[1])
        
        # Forward LSTM and get final state
        x = self.encoder(x)
        # Forward output
        #x = F.relu(self.output(x))
        x = torch.mean(x, dim=1)
        x = self.classifier(x)
        return x

    def inference(self, x):
        # Prepare LSTM initiale state
        batch_size = x.size(0)
        #lstm_init = (torch.zeros(self.lstm_layers, batch_size, self.lstm_size), torch.zeros(self.lstm_layers, batch_size, self.lstm_size))
        #if x.is_cuda: lstm_init = (lstm_init[0].cuda(), lstm_init[0].cuda())
        #lstm_init = (lstm_init[0], lstm_init[1])
        
        # Forward LSTM and get final state
        x = self.encoder(x)
        # Forward output
        #x = F.relu(self.output(x))
        x_f = torch.mean(x, dim=1)
        x = self.classifier(x_f)
        return x_f, x
