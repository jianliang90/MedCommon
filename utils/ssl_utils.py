import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
sys.path.append(ROOT)

import external_lib.ResNets3D.models as models3D
import external_lib.ResNets3D.models.resnet
import external_lib.ResNets3D.models.densenet
from collections import OrderedDict

import torch

model_names = sorted(name for name in models3D.__dict__
    if name.islower() and not name.startswith("__"))

class SSL_Utils:
    def __init__(self) -> None:
        pass

    @staticmethod
    def load_ssl_model(archname, pretrained_file):
        backbone = None
        for name in model_names:
            if archname.startswith(name):
                arch = name
                sub_arch = int(archname.replace(arch, ''))      
                n_classes_placeholder = 2
                backbone = models3D.__dict__[arch].generate_model(sub_arch, n_input_channels=1, widen_factor=0.5, n_classes=n_classes_placeholder)
        if pretrained_file:
            checkpoint = torch.load(pretrained_file, map_location='cpu')
            state_dict = checkpoint['state_dict']
            new_state_dict = OrderedDict()
            for k in list(state_dict.keys()):
                if k.startswith('module.encoder_q') and not k.startswith('module.encoder_q.fc'):
                    new_state_dict[k[len('module.encoder_q.'):]] = state_dict[k]
            msg = backbone.load_state_dict(new_state_dict, strict=False)
            assert set(msg.missing_keys) == {"fc.weight", "fc.bias"}
            print("=> loaded pre-trained model '{}'".format(pretrained_file))
        backbone.fc.weight.data.normal_(mean=0.0, std=0.01)
        backbone.fc.bias.data.zero_()
        return backbone


          
