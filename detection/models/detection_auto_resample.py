import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

MEDCOMMON_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.path.pardir, os.path.pardir)
sys.path.append(MEDCOMMON_ROOT)
sys.path.append(os.path.join(MEDCOMMON_ROOT, 'external_lib'))

import external_lib.ResNets3D.models as models3D
import external_lib.ResNets3D.models.resnet
import external_lib.ResNets3D.models.densenet

model_names = sorted(name for name in models3D.__dict__
    if name.islower() and not name.startswith("__"))

print(model_names)

class Input(nn.Module):
    """Input layer, including re-sample, clip and normalization image."""
    def __init__(self, input_size=(224, 160, 224)):
        '''
        clip_window=(-1200, 1200)
        '''
        super(Input, self).__init__()
        self.input_size = input_size

    def forward(self, x):
        x = F.interpolate(x, size=self.input_size, mode='trilinear', align_corners=True)
        return x

class RegressionDetecter(nn.Module):
    def __init__(self, n_objects, 
        net_args={
            'input_size': [128, 128, 128], 
            'arch': 'resnet18'
        }):
        super().__init__()
        self.input = Input(input_size=net_args['input_size'])
        
        self.backbone = None
        print(model_names)
        for name in model_names:
            if net_args['arch'].startswith(name):
                arch = name
                sub_arch = int(net_args['arch'].replace(arch, ''))
                self.backbone = models3D.__dict__[arch].generate_model(sub_arch, n_input_channels=1, widen_factor=0.5, n_classes=n_objects*6)
                break
        
    def forward(self, input):
        x = self.input(input)
        output = self.backbone(x)
        return output


def test_RegressionDetecter():
    net_args = {
        'input_size': [64,64,64], 
        'arch': 'resnet10'
    }
    model = RegressionDetecter(1, net_args)
    input = torch.randn(1,1,80,70,70)
    output = model(input)
    print(output.shape)
    print('====> test_RegressionDetecter')


if __name__ == '__main__':
    test_RegressionDetecter()
