import os
import sys

from tqdm import tqdm
import numpy as np

import shutil

ROOT = os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir, os.path.pardir, os.path.pardir, os.path.pardir)
sys.path.append(ROOT)

from utils.image_segmentation_preprocessing_utils import ImageSegmentationPreprocessingUtils
from utils.datasets_utils import DatasetsUtils

# 1. copy数据，将数据处理成分割框架需要的形式
def copy_data(in_root='/data/jiapf/NMPA_refine/nii_gz', 
        out_root = '/data/medical/cardiac/seg/coronary/coronary_ori'
    ):
    out_image_root = os.path.join(out_root, 'images')
    out_mask_root = os.path.join(out_root, 'masks')
    os.makedirs(out_image_root, exist_ok=True)
    os.makedirs(out_mask_root, exist_ok=True)
    for suid in tqdm(os.listdir(in_root)):
        sub_in_root = os.path.join(in_root, suid)
        src_image_file = os.path.join(sub_in_root, 'im.nii.gz')
        src_mask_file = os.path.join(sub_in_root, 'mask.nii.gz')
        if not os.path.isfile(src_image_file):
            continue
        if not os.path.isfile(src_mask_file):
            continue
        dst_image_file = os.path.join(out_image_root, '{}.nii.gz'.format(suid))
        dst_mask_file = os.path.join(out_mask_root, '{}.nii.gz'.format(suid))
        shutil.copyfile(src_image_file, dst_image_file)
        shutil.copyfile(src_mask_file, dst_mask_file)


def generate_image_mask_pairs():
    ImageSegmentationPreprocessingUtils.generate_image_mask_pairs(
        ref_mask_root = '/data/medical/cardiac/seg/coronary/coronary_ori/masks', 
        out_root = '/data/medical/cardiac/seg/coronary/coronary_cropped_by_mask', 
        image_root = '/data/medical/cardiac/seg/coronary/coronary_ori/images', 
        mask_root = '/data/medical/cardiac/seg/coronary/coronary_ori/masks', 
        process_num = 12,
        is_dcm = False, 
        mask_pattern = '.nii.gz'
    )

def split_ds():
    image_root = '/data/medical/cardiac/seg/coronary/coronary_cropped_by_mask/images'
    out_config_dir = '/data/medical/cardiac/seg/coronary/coronary_cropped_by_mask/config'
    DatasetsUtils.split_ds(image_root, out_config_dir, 0.8, 0.001)

def analyze_mask_boundary():
    # 1. 分析冠脉mask边界信息
    ImageSegmentationPreprocessingUtils.analyze_mask_boundary(
        '/data/medical/cardiac/seg/coronary/coronary_ori/masks', 
        out_root = '/data/medical/cardiac/seg/coronary/coronary_ori/result/analysis_result',
        out_filename='boundary_info_ori_mask.txt'
    )

if __name__ == '__main__':
    # 1. copy数据，将数据处理成分割框架需要的形式
    # copy_data()
    # 2. 生成数据对
    # generate_image_mask_pairs()
    # 3. 划分数据集
    split_ds()
    # 4. 统计mask边界信息
    analyze_mask_boundary()
        
