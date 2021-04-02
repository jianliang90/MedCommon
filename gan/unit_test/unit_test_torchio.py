import os
import sys
sys.path.append('../')
from datasets.common_ds import GAN_COMMON_DS, get_common_transform

import torch
import torchvision
from torch.utils.data import Dataset, DataLoader

from tqdm import tqdm

data_root = '/fileser/zhangwd/data/cardiac/cta2mbf/data_114_20210318/5.mbf_myocardium'

transform = get_common_transform([128,128,160],'GAN')
ds = GAN_COMMON_DS(data_root, 'cropped_cta.nii.gz', 'cropped_mbf.nii.gz', [64,64,64], transform)
dataloader = DataLoader(ds, batch_size=2, num_workers=1, shuffle=True, pin_memory=False)
# one_subject = ds[1]

for index, (subjects) in enumerate(dataloader):
    print('hello world!')


print('hello world!')