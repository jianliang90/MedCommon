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

# 生成256x256x256大小的数据集
def generate_ds():
    generate_resampled_pairs_unsame_resolution(
        '/data/medical/cardiac/chamber/seg/chamber_seg', 
        '/data/medical/cardiac/chamber/seg/chamber_256', 
        [256,256,256]
    )

# 划分数据集
def split_ds():
    image_root = '/data/medical/cardiac/chamber/seg/chamber_256/images'
    out_config_dir = '/data/medical/cardiac/chamber/seg/chamber_256/config'
    DatasetsUtils.split_ds(image_root, out_config_dir, 0.8, 0.001)

if __name__ == '__main__':
    generate_ds()
    split_ds()