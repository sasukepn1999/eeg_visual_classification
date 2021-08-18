# Define options
import argparse
parser = argparse.ArgumentParser(description="Template")

# Directory to pretrained model for inference
parser.add_argument('-dm', '--dir-model', default='', help="Directory to pretrained model")

# Directory to Dataset for inference
parser.add_argument('-ed', '--eeg-dataset', default='', help="Directory to Dataset")

# Directory to Feature EEG
parser.add_argument('-df', '--dir-feature', default='', help="Directory to Feature EEG")

# Parse arguments
opt = parser.parse_args()
print(opt)

# Imports
import sys
import os
import random
import math
import time
import torch; torch.utils.backcompat.broadcast_warning.enabled = True
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
import torch.backends.cudnn as cudnn; cudnn.benchmark = True
from scipy.fftpack import fft, rfft, fftfreq, irfft, ifft, rfftfreq
from scipy import signal
import numpy as np
import models
import importlib

# Dataset class.
class EEGDataset:
    
    # Constructor
    def __init__(self, eeg_signals_path):
        # Load EEG signals
        loaded = torch.load(eeg_signals_path)
       
        self.data=loaded['dataset']        
        self.labels = loaded["labels"]
        self.images = loaded["images"]
        self.means = loaded['means']
        self.stds = loaded['stddevs']
        
        # Compute size
        self.size = len(self.data)

    # Get size
    def __len__(self):
        return self.size

    # Get item
    def __getitem__(self, i):
        #eeg = (self.data[i]["eeg"].float() - self.means) / self.stds
        # Process EEG
        eeg = self.data[i]["eeg"].float()
        eeg = eeg.t()
        eeg = eeg[20:460,:]

        # if opt.model_type == "model10":
        #     eeg = eeg.t()
        #     eeg = eeg.view(1,128,460-20)
        # Get label
        label = self.data[i]["label"]
        # Return
        return eeg, label

human_readable_labels = ['sorrel', 'parachute', 'iron', 'anemone_fish', 'espresso_maker', 'coffee_mug', 'mountain_bike', 'revolver', 'giant_panda', 'daisy', 'canoe', 'lycaenid', 'German_shepherd', 'running_shoe', "jack-o'-lantern", 'cellular_telephone', 'golf_ball', 'desktop_computer', 'broom', 'pizza', 'missile', 'capuchin', 'pool_table', 'mailbag', 'convertible', 'folding_chair', 'pajama', 'mitten', 'electric_guitar', 'reflex_camera', 'grand_piano', 'mountain_tent', 'banana', 'bolete', 'digital_watch', 'African_elephant', 'airliner', 'electric_locomotive', 'radio_telescope', 'Egyptian_cat']

# Load model
model = torch.load(opt.dir_model)

# Load dataset
dataset = EEGDataset(opt.eeg_dataset)
eeg_feature = torch.load(opt.eeg_dataset)

# Inference
model.eval()
torch.set_grad_enabled(False)
for i, (input, target) in enumerate(dataset):
  
  input = input.view(1, 440, 128)
  input = input.to("cuda")

  # Forward
  output_ft, _ = model.inference(input)
  output_ft = output_ft.to(device="cpu")

  eeg_feature['dataset'][i]['eeg_feature'] = output_ft
  del eeg_feature['dataset'][i]['eeg']

# Save feature
torch.save(eeg_feature, opt.dir_feature)