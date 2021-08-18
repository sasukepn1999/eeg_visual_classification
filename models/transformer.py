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
        self.classifier = nn.Linear(output_size,40)
        
    def forward(self, x):
        # Batch size
        batch_size = x.size(0)
        
        # Forward 
        x = self.encoder(x)
        x = torch.mean(x, dim=1)
        x = self.classifier(x)
        return x

    def inference(self, x):
        # Batch size
        batch_size = x.size(0)
        
        # Inference (x_f: eeg feature, x: predicted_class)
        x = self.encoder(x)
        x_f = torch.mean(x, dim=1)
        x = self.classifier(x_f)
        return x_f, x
