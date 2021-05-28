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

if __name__ == '__main__':
    # train()
    inference(
        os.path.join('/data/medical/cardiac/chamber/seg/chamber_seg/images/1.3.12.2.1107.5.1.4.60320.30000015020202382801500025712.nii.gz'), 
        os.path.join('/data/medical/cardiac/chamber/seg/chamber_256/inference/exp1'), [256, 256, 256], 
        os.path.join('/data/medical/cardiac/chamber/seg/chamber_256/checkpoints/chamber/common_seg_epoch_19_train_0.067')
    )
    inference(
        os.path.join('/data/medical/cardiac/chamber/seg/chamber_seg/images/1.3.12.2.1107.5.1.4.60320.30000018042800095349200029782.nii.gz'), 
        os.path.join('/data/medical/cardiac/chamber/seg/chamber_256/inference/exp1'), [256, 256, 256], 
        os.path.join('/data/medical/cardiac/chamber/seg/chamber_256/checkpoints/chamber/common_seg_epoch_19_train_0.067')
    )