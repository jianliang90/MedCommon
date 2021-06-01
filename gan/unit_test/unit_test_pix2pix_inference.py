import os
import sys
import time

import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

sys.path.append(os.path.join(os.path.dirname(__file__), os.path.pardir))

from options.test_options import TestOptions
from util.visualizer import Visualizer
import models
from models.pix2pix_3d_model import Pix2Pix3DModel

sys.path.append(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir))

from utils.distributed_utils import DistributedUtils


from datasets.common_ds import GAN_COMMON_DS, get_common_transform

import SimpleITK as sitk

opt = TestOptions().parse()
print(opt)

data_root = '/data/medical/cardiac/cta2mbf/data_66_20210517/5.mbf_myocardium'
# transform = get_common_transform([416,416,160],'GAN_INFERENCE')
transform = get_common_transform([384,384,160],'GAN_INFERENCE')
ds = GAN_COMMON_DS(data_root, 'cropped_cta.nii.gz', 'cropped_mbf.nii.gz', [64,64,64], transform)
dataloader = DataLoader(ds, batch_size=1, num_workers=2, shuffle=True, pin_memory=True)
dataset_size = len(dataloader) 

model = models.create_model(opt)
model.setup(opt)

model.netG.eval()
# model.netG.train()

out_dir = "/data/medical/cardiac/cta2mbf/data_66_20210517/6.inference_384x384x160_eval"
os.makedirs(out_dir, exist_ok=True)

for index, (subjects) in tqdm(enumerate(dataloader)):
    real_a = subjects['src']['data'].float()
    real_b = subjects['dst']['data'].float()
    input = {}
    input['A'] = real_a
    input['B'] = real_b
    input['A_paths'] = 'A'
    input['B_paths'] = 'B'
    model.set_input(input)
    fake_b = model.netG(real_a.cuda())
    fake_b = fake_b.detach().squeeze().cpu().numpy()
    real_a = real_a.squeeze().cpu().numpy()
    real_b = real_b.squeeze().cpu().numpy()
    real_a = np.transpose(real_a, [2,1,0])
    real_b = np.transpose(real_b, [2,1,0])
    fake_b = np.transpose(fake_b, [2,1,0])
    pid = subjects['src']['path'][0].split('/')[-2]
    info_img = sitk.ReadImage(subjects['src']['path'][0])
    spacing = info_img.GetSpacing()
    direction = info_img.GetDirection()
    origin = info_img.GetOrigin()
    out_sub_dir = os.path.join(out_dir, pid)
    os.makedirs(out_sub_dir, exist_ok=True)
    real_img_a = sitk.GetImageFromArray(real_a)
    real_img_a.SetSpacing(spacing)
    real_img_a.SetDirection(direction)
    real_img_a.SetOrigin(origin)
    real_img_b = sitk.GetImageFromArray(real_b)
    real_img_b.CopyInformation(real_img_a)
    fake_img_b = sitk.GetImageFromArray(fake_b)
    fake_img_b.CopyInformation(real_img_a)
    sitk.WriteImage(real_img_a, os.path.join(out_sub_dir, 'real_a.nii.gz'))
    sitk.WriteImage(real_img_b, os.path.join(out_sub_dir, 'real_b.nii.gz'))
    sitk.WriteImage(fake_img_b, os.path.join(out_sub_dir, 'fake_b.nii.gz'))
    print('hello world!')       

print('hello world!')
