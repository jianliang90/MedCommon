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
        os.path.join('/data/medical/brain/cerebral_parenchyma/exp/cta/images/1.3.12.2.1107.5.1.4.60320.30000012032800280131200018836.nii.gz'), 
        os.path.join('/data/medical/brain/cerebral_parenchyma/exp/ct_256/inference/exp1'), [256, 256, 256], 
        os.path.join('/data/medical/brain/cerebral_parenchyma/exp/ct_256/checkpoints/parenchyma/common_seg_epoch_7_train_0.012')
    )
    inference(
        os.path.join('/data/medical/brain/cerebral_parenchyma/exp/cta/images/1.3.12.2.1107.5.1.4.60320.30000017011523543351200001316.nii.gz'), 
        os.path.join('/data/medical/brain/cerebral_parenchyma/exp/ct_256/inference/exp1'), [256, 256, 256], 
        os.path.join('/data/medical/brain/cerebral_parenchyma/exp/ct_256/checkpoints/parenchyma/common_seg_epoch_7_train_0.012')
    )
