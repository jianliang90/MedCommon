import os
import sys

import torch

sys.path.append(os.path.join(os.path.dirname(__file__), os.path.pardir))
from options.train_options import TrainOptions

'''
1. export model
'''

# import models
opt = TrainOptions().parse()

opt = TrainOptions().parse()

model = models.create_model(opt)
model.setup(opt)

dummy_input = torch.rand(1, 1, 128, 128, 128)
trace_model = torch.jit.trace(model.netG.module.cpu(), dummy_input)
trace_model.save('cta2mbf_generator_2.pt')


'''
2. load model
'''
# 1. cpu invoke
# loaded_model = torch.jit.load('cta2mbf_generator_2.pt')
# dummy_input = torch.rand(1,1,128,128,128)
# output = loaded_model(dummy_input)
# print(output.shape)
# print(output.dtype)

# # 2. gpu invoke
# loaded_model.cuda(0)
# dummy_input = torch.rand(1, 1, 128, 128, 128)
# output = loaded_model(dummy_input.cuda(0))
# print(output.shape)
# print(output.dtype)

# # 3. change input size
# dummy_input = torch.rand(1, 1, 128, 256, 128)
# output = loaded_model(dummy_input.cuda(0))
# print(output.shape)
# print(output.dtype)



'''
platform: "pytorch_libtorch"
max_batch_size: 1
input [
  {
    name: "input0"
    data_type: TYPE_FP32
    dims: [ 1,-1,-1,-1 ]
  }
]
output [
  {
    name: "output0"
    data_type: TYPE_FP32
    dims: [ 1,-1,-1,-1 ]
  }
]
'''

print('hello world!')
