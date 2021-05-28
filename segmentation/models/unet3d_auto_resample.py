import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

MEDCOMMON_SEG_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.path.pardir)
sys.path.append(os.path.join(MEDCOMMON_SEG_ROOT, 'external_lib'))

from segmentation.external_lib.MedicalZooPytorch.lib.medzoo.Unet3D import UNet3D
from segmentation.external_lib.MedicalZooPytorch.lib.losses3D.dice import DiceLoss

class Input(nn.Module):
    """Input layer, including re-sample, clip and normalization image."""
    def __init__(self, input_size=(224, 160, 224), clip_window=None):
        '''
        clip_window=(-1200, 1200)
        '''
        super(Input, self).__init__()
        self.input_size = input_size
        self.clip_window = clip_window

    def forward(self, x):
        x = F.interpolate(x, size=self.input_size, mode='trilinear', align_corners=True)
        if self.clip_window is not None:
            x = torch.clamp(x, min=self.clip_window[0], max=self.clip_window[1])
            mean = torch.mean(x)
            std  = torch.std(x)
            x = (x - mean) / (1e-5 + std)

        return x

class Output(nn.Module):
    """Output layer, re-sample image to original size."""
    def __init__(self):
        super(Output, self).__init__()

    def forward(self, x, x_input):
        x = F.interpolate(x, size=(x_input.size(2), x_input.size(3), x_input.size(4)), mode='trilinear', align_corners=True)
        # x = F.sigmoid(x)

        return x

class ResampledUnet3D(nn.Module):
    def __init__(self, in_channels, n_classes, base_n_filter=8, 
        net_args={
            "input_size": [128, 128, 128], 
            "dynamic_resize": False
        }):
        super(ResampledUnet3D, self).__init__()
        
        self.unet_3d = UNet3D(in_channels, n_classes, base_n_filter)
        
        if 'dynamic_resize' in net_args and net_args['dynamic_resize']:
            self.input = Input(input_size=net_args["input_size"])
        else:
            self.input = None
        self.output = Output()

        self.initialize_weights()

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x_input):
        if self.input:
            x = self.input(x_input)
        else:
            x = x_input
        output = self.unet_3d(x)
        if self.input:
            output = self.output(output, x_input)

        return output

