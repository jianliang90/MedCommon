import os
import sys
import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir, os.path.pardir, os.path.pardir, os.path.pardir)
sys.path.append(ROOT)

from detection.datasets.position_detection_common_ds import extract_boundary_info
from detection.runner.regression_detection_trainner import train

if __name__ == '__main__':
#    extract_boundary_info(mask_root='/data/medical/cardiac/seg/coronary/coronary_ori/masks', out_file='/data/medical/cardiac/seg/coronary/coronary_ori/config/mask_boundary_info.json')
    train()