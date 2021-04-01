import os
import sys
import time

import torch

sys.path.append(os.path.join(os.path.dirname(__file__), os.path.pardir))

from options.train_options import TrainOptions
import models
from models.pix2pix_3d_model import Pix2Pix3DModel

sys.path.append(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir))

from utils.distributed_utils import DistributedUtils


opt = TrainOptions().parse()

DistributedUtils.init_distributed_mode(opt)

if 'rank' not in opt:
    opt.rank = DistributedUtils.get_rank()

print(opt)

model = models.create_model(opt)
model.setup(opt)

print(model.netD)

# model = Pix2Pix3DModel(opt)

for i in range(50):
    real_a = torch.randn(1,1,32,448,448).float()
    real_b = torch.randn(1,1,32,448,448).float()
    input = {}
    input['A'] = real_a
    input['B'] = real_b
    input['A_paths'] = 'A'
    input['B_paths'] = 'B'
    model.set_input(input)
    model.optimize_parameters()

# time.sleep(30)

print('hello world!')