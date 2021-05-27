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

data_root = '/data/medical/cardiac/seg/coronary/coronary_cropped_by_mask'    

if __name__ == '__main__':
    # train()
    # inference(
    #     os.path.join(data_root, 'images/1.2.392.200036.9116.2.2054276706.1582589798.9.1347400003.1.nii.gz'), 
    #     os.path.join(data_root, 'inference/exp1'), [384, 384, 256], 
    #     os.path.join(data_root, 'checkpoints/coronary/common_seg_epoch_28_train_0.069')
    # )

    # inference(
    #     os.path.join(data_root, 'images/1.2.392.200036.9116.2.2054276706.1589264256.12.1245900005.1.nii.gz'), 
    #     os.path.join(data_root, 'inference/exp1'), [384, 384, 256], 
    #     os.path.join(data_root, 'checkpoints/coronary/common_seg_epoch_28_train_0.069')
    # )

    inference(
        os.path.join(data_root, 'images/1.2.392.200036.9116.2.2054276706.1582589798.9.1347400003.1.nii.gz'), 
        os.path.join(data_root, 'inference/exp1'), [384, 384, 256], 
        os.path.join(data_root, 'checkpoints/coronary/common_seg_epoch_46_train_0.060')
    )

    inference(
        os.path.join(data_root, 'images/1.2.392.200036.9116.2.2054276706.1589264256.12.1245900005.1.nii.gz'), 
        os.path.join(data_root, 'inference/exp1'), [384, 384, 256], 
        os.path.join(data_root, 'checkpoints/coronary/common_seg_epoch_46_train_0.060')
    )
    