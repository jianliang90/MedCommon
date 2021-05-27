import os
import sys
import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir, os.path.pardir, os.path.pardir, os.path.pardir)
sys.path.append(ROOT)

from detection.datasets.position_detection_common_ds import extract_boundary_info
from detection.runner.regression_detection_trainner import train

if __name__ == '__main__':
    # 提取mask的边界信息，减少后续重复计算
    # extract_boundary_info(mask_root='/data/medical/brain/cerebral_parenchyma/exp/cta/masks', out_file='/data/medical/brain/cerebral_parenchyma/exp/cta/config/mask_boundary_info.json')
    # extract_boundary_info(mask_root='/data/medical/brain/cerebral_parenchyma/exp/cta_256/masks', out_file='/data/medical/brain/cerebral_parenchyma/exp/cta_256/config/mask_boundary_info.json')
    train()