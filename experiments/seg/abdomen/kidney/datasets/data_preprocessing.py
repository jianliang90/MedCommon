import os
import sys
from tqdm import tqdm

import shutil

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir, os.path.pardir, os.path.pardir, os.path.pardir))
print(ROOT)
sys.path.append(ROOT)

from utils.datasets_utils import DatasetsUtils
from segmentation.datasets.common_seg_datasets import generate_resampled_pairs_unsame_resolution


data_root = '/data/medical/kidney/kits19'

# 生成分割程序需要的格式
def copy_data(
        src_data_root = '/data/medical/kidney/kits19/data', 
        dst_data_root = '/data/medical/kidney/kits19/ori'
    ):
    os.makedirs(dst_data_root, exist_ok=True)
    dst_img_root = os.path.join(dst_data_root, 'images')
    dst_mask_root = os.path.join(dst_data_root, 'masks')
    os.makedirs(dst_img_root, exist_ok=True)
    os.makedirs(dst_mask_root, exist_ok=True)
    for pid in tqdm(os.listdir(src_data_root)):
        src_sub_root = os.path.join(src_data_root, pid)
        src_img_file = os.path.join(src_sub_root, 'imaging.nii.gz')
        src_mask_file = os.path.join(src_sub_root, 'segmentation.nii.gz')
        if not os.path.isfile(src_img_file):
            continue
        if not os.path.isfile(src_mask_file):
            continue
        dst_img_file = os.path.join(dst_img_root, '{}.nii.gz'.format(pid))
        dst_mask_file = os.path.join(dst_mask_root, '{}.nii.gz'.format(pid))
        shutil.copyfile(src_img_file, dst_img_file)
        shutil.copyfile(src_mask_file, dst_mask_file)


if __name__ == '__main__':
    # copy_data()
    generate_resampled_pairs_unsame_resolution(
        '/data/medical/kidney/kits19/ori', 
        '/data/medical/kidney/kits19/ori_256', 
        [256,256,256]
    )