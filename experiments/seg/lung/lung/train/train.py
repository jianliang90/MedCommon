import os
import sys

from tqdm import tqdm
import numpy as np

import shutil

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir, os.path.pardir, os.path.pardir, os.path.pardir))
print(ROOT)
sys.path.append(ROOT)

from utils.datasets_utils import DatasetsUtils
from segmentation.datasets.common_seg_datasets import generate_resampled_pairs_unsame_resolution

from segmentation.runner.train_seg import inference, train

import SimpleITK as sitk
from tqdm import tqdm

# 生成256x256x256大小的数据集
def generate_ds():
    generate_resampled_pairs_unsame_resolution(
        '/data/medical/lung/LUNA/lung_ori', 
        '/data/medical/lung/LUNA/lung_256', 
        [256,256,256]
    )

# 划分数据集
def split_ds():
    image_root = '/data/medical/lung/LUNA/lung_256/images'
    out_config_dir = '/data/medical/lung/LUNA/lung_256/config'
    DatasetsUtils.split_ds(image_root, out_config_dir, 0.8, 0.001)


if __name__ == '__main__':
    # 生成数据集
    # generate_ds()
    # split_ds()
    '''
    ------------------------就是个分界线------------------------
    '''
    train()
    '''
    ------------------------就是个分界线------------------------
    '''
    # 训练数据集中的数据
    # inference(
    #     os.path.join('/data/medical/lung/LUNA/lung_ori/images/1.3.6.1.4.1.14519.5.2.1.6279.6001.640729228179368154416184318668.nii.gz'), 
    #     os.path.join('/data/medical/lung/LUNA/lung_256/inference/exp1'), [256, 256, 256], 
    #     os.path.join('/data/medical/lung/LUNA/lung_256/checkpoints/lung/common_seg_epoch_91_train_0.024')
    # ) 
    # inference(
    #     os.path.join('/data/medical/lung/LUNA/lung_ori/images/1.3.6.1.4.1.14519.5.2.1.6279.6001.608029415915051219877530734559.nii.gz'), 
    #     os.path.join('/data/medical/lung/LUNA/lung_256/inference/exp1'), [256, 256, 256], 
    #     os.path.join('/data/medical/lung/LUNA/lung_256/checkpoints/lung/common_seg_epoch_91_train_0.024')
    # ) 

