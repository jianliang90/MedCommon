import os
import sys

import SimpleITK as sitk
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from data_io_utils import DataIO

def dice_coef(y_true, y_pred, smooth=1):
  intersection = K.sum(y_true * y_pred, axis=[1,2,3])
  union = K.sum(y_true, axis=[1,2,3]) + K.sum(y_pred, axis=[1,2,3])
  dice = K.mean((2. * intersection + smooth)/(union + smooth), axis=0)
  return dice


class MetricsUtils:
    def __init__(self):
        pass

    @staticmethod
    def dice_coef(y_true, y_pred, smooth=1):
        intersection = K.sum(y_true * y_pred, axis=[1,2,3])
        union = K.sum(y_true, axis=[1,2,3]) + K.sum(y_pred, axis=[1,2,3])
        dice = K.mean((2. * intersection + smooth)/(union + smooth), axis=0)
        return dice

    def eval_dice(pred_mask_file, gt_mask_file):
        pred_data = DataIO.load_nii_image(pred_mask_file)
        gt_data = DataIO.load_nii_image(gt_mask_file)

        dice = MetricsUtils.dice_coef(gt_data['image'], pred_data['image'])

        return dice