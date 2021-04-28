import os
import sys

import torch
from torch.utils.data import DataLoader

sys.path.append(os.path.join(os.path.dirname(__file__), os.path.pardir))
from options.train_options import TrainOptions
from datasets.common_ds import GANDataWrapper, get_common_transform, GAN_COMMON_DS
import SimpleITK as sitk
import numpy as np



# load example sample
infile = '/fileser/zhangwd/data/cardiac/cta2mbf/data_114_20210318/5.mbf_myocardium/1069558/cropped_cta.nii.gz'
image_shape = [320,320,160]
transforms = get_common_transform(image_shape, 'GAN_INFERENCE')
img = GANDataWrapper.get_processed_image(transforms,infile)
img = np.array(img.src.data, dtype=np.float32)
img = np.expand_dims(img, 0)
input_tensor = torch.from_numpy(img)
print(input_tensor.shape)


# data_root = '/fileser/zhangwd/data/cardiac/cta2mbf/data_114_20210318/5.mbf_myocardium'
# transform = get_common_transform([320,320,160],'GAN')
# ds = GAN_COMMON_DS(data_root, 'cropped_cta.nii.gz', 'cropped_mbf.nii.gz', [64,64,64], transform)
# dataloader = DataLoader(ds, batch_size=1, num_workers=2, shuffle=True, pin_memory=True)
# dataset_size = len(dataloader)    # get the number of images in the dataset.

# subjects = ds.__getitem__(0)
# real_a = subjects['src']['data'].float()
# real_b = subjects['dst']['data'].float()
# input = {}
# input['A'] = real_a.unsqueeze(0)
# input['B'] = real_b.unsqueeze(0)
# input['A_paths'] = 'A'
# input['B_paths'] = 'B'


'''
1. export model
'''

import models
opt = TrainOptions().parse()

model = models.create_model(opt)
model.setup(opt)

netG_ckpt = '../unit_test/checkpoints/experiment_name/latest_net_G.pth'
netG = model.netG.module
netG.load_state_dict(torch.load(netG_ckpt, map_location='cpu'))
netG.eval()
output = netG(input_tensor.cuda())
output = output.squeeze().detach().cpu().numpy()
output = np.array(output, dtype=np.int16)
sitk_img = sitk.GetImageFromArray(output)
sitk.WriteImage(sitk_img, './results/pytorch_model_test.nii.gz')

dummy_input = torch.rand(1, 1, 128, 128, 128)
trace_model = torch.jit.trace(netG.cpu(), dummy_input)
trace_model.save('cta2mbf_generator_3.pt')


'''
2. load model
'''
loaded_model = torch.jit.load('cta2mbf_generator_3.pt')
loaded_model.cuda()
output = loaded_model(input_tensor.cuda())
output = output.squeeze().detach().cpu().numpy()
output = np.array(output, dtype=np.int16)
sitk_img = sitk.GetImageFromArray(output)
sitk.WriteImage(sitk_img, './results/torchscript_model_test.nii.gz')


'''
3. triton server config.pbtxt
'''

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
