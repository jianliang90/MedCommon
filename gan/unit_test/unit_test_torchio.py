import os
import sys
sys.path.append('../')
from datasets.common_ds import GAN_COMMON_DS, get_common_transform

import torch
import torchvision
from torch.utils.data import Dataset, DataLoader

from tqdm import tqdm

data_root = '/fileser/zhangwd/data/cardiac/cta2mbf/data_114_20210318/5.mbf_myocardium'

transform = get_common_transform([320,320,160],'GAN')
ds = GAN_COMMON_DS(data_root, 'cropped_cta.nii.gz', 'cropped_mbf.nii.gz', [64,64,64], transform)

# one_subject = ds[1]
for i in range(ds.__len__()):
    one_subject = ds[i]
    print('hello world')

print('hello world!')