import os
import sys
import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir, os.path.pardir, os.path.pardir, os.path.pardir)
sys.path.append(ROOT)

sys.path.append(os.path.join(os.path.dirname(__file__), os.path.pardir))
sys.path.append(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir))
sys.path.append(os.path.join(ROOT, 'segmentation/external_lib/MedicalZooPytorch'))

from segmentation.runner.train_seg import inference, train

import SimpleITK as sitk
from tqdm import tqdm

def fix_mask_sitk_info(root='/fileser/zhangwd/data/lung/airway/segmentation'):
    image_root = os.path.join(root, 'images')
    mask_root = os.path.join(root, 'masks')
    out_mask_root = os.path.join(root, 'masks_x')
    os.makedirs(out_mask_root, exist_ok=True)
    for f in tqdm(os.listdir(mask_root)):
        basename = f.replace('.nii.gz', '')
        image_file = os.path.join(image_root, '{}.nii.gz'.format(basename))
        mask_file = os.path.join(mask_root, '{}.nii.gz'.format(basename))
        out_mask_file = os.path.join(out_mask_root, '{}.nii.gz'.format(basename))
        image = sitk.ReadImage(image_file)
        mask = sitk.ReadImage(mask_file)
        mask.CopyInformation(image)
        sitk.WriteImage(mask, out_mask_file)
    

if __name__ == '__main__':
    # fix_mask_sitk_info()
    # train()
    # inference('/fileser/zhangwd/data/lung/airway/segmentation/images/1.3.6.1.4.1.14519.5.2.1.6279.6001.325164338773720548739146851679.nii.gz', '/fileser/zhangwd/data/lung/airway/segmentation/inference/exp1', [416, 288, 288], '/data/medical/lung/airway/segmentation/checkpoints/airway-bk/common_seg_epoch_128_train_0.052')
    # inference('/fileser/zhangwd/data/lung/airway/segmentation/images/1.2.840.113704.1.111.11692.1420599548.14.nii.gz', '/fileser/zhangwd/data/lung/airway/segmentation/inference/exp1', [416, 288, 288], '/data/medical/lung/airway/segmentation/checkpoints/airway-bk/common_seg_epoch_128_train_0.052')
    inference('/fileser/zhangwd/data/lung/airway/segmentation/images/1.3.6.1.4.1.14519.5.2.1.6279.6001.325164338773720548739146851679.nii.gz', '/fileser/zhangwd/data/lung/airway/segmentation/inference/exp2', [384, 256, 288], '/data/medical/lung/airway/segmentation/checkpoints/airway_f8/common_seg_epoch_144_train_0.047')
    inference('/fileser/zhangwd/data/lung/airway/segmentation/images/1.2.840.113704.1.111.11692.1420599548.14.nii.gz', '/fileser/zhangwd/data/lung/airway/segmentation/inference/exp2', [384, 256, 288], '/data/medical/lung/airway/segmentation/checkpoints/airway_f8/common_seg_epoch_144_train_0.047')
    