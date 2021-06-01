import os
import sys

import numpy as np

import SimpleITK as sitk
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
sys.path.append(root)
from utils.data_io_utils import DataIO
from utils.mask_utils import MaskUtils

def dice_coef(y_true, y_pred, smooth=1):
  intersection = K.sum(y_true * y_pred, axis=[1,2,3])
  union = K.sum(y_true, axis=[1,2,3]) + K.sum(y_pred, axis=[1,2,3])
  dice = K.mean((2. * intersection + smooth)/(union + smooth), axis=0)
  return dice


class MetricsUtils:
    def __init__(self):
        pass

    @staticmethod
    def dice_coef(gt_arr, pred_arr, smooth=1):
        gt_arr[gt_arr>0] = 1
        pred_arr[pred_arr>0] = 1
    
        intersect = (gt_arr*pred_arr).sum()
        denominator = gt_arr.sum() + pred_arr.sum()
        smooth = 1e-8
        dice = 2 * ((intersect + smooth) / (denominator + smooth))
        return dice

    @staticmethod
    def eval_dice(pred_mask_file, gt_mask_file):
        pred_data = DataIO.load_nii_image(pred_mask_file)
        gt_data = DataIO.load_nii_image(gt_mask_file)

        dice = MetricsUtils.dice_coef(gt_data['image'], pred_data['image'])

        return dice

    @staticmethod
    def calc_mae(arr_a, arr_b, mask=True):
        diff_arr = arr_a-arr_b
        mae_mask = np.mean(abs(diff_arr), where=mask)
        mae = np.mean(abs(diff_arr))
        return mae_mask, mae

    @staticmethod
    def calc_mae_with_file(infile_a, infile_b, mask_file=None, mask_label=None):
        image_a = sitk.ReadImage(infile_a)
        image_b = sitk.ReadImage(infile_b)
        image_mask = None
        if mask_file:
            image_mask = sitk.ReadImage(mask_file)
        arr_a = sitk.GetArrayFromImage(image_a)
        arr_b = sitk.GetArrayFromImage(image_b)
        if image_mask:
            mask = sitk.GetArrayFromImage(image_mask)
            mask = MaskUtils.fix_mask_label(mask, mask_label=None)
        else:
            mask = True
        return MetricsUtils.calc_mae(arr_a, arr_b, mask)


def test_calc_mae_with_file():
    infile_a = '/data/medical/cardiac/cta2mbf/data_66_20210517/6.inference/1023293/real_b.nii.gz'
    infile_b = '/data/medical/cardiac/cta2mbf/data_66_20210517/6.inference/1023293/fake_b.nii.gz'
    mask_file = '/data/medical/cardiac/cta2mbf/data_66_20210517/5.mbf_myocardium/1023293/cropped_mask.nii.gz'
    mask_file = None
    mae = MetricsUtils.calc_mae_with_file(infile_a, infile_b, mask_file)
    print(mae)
    print('====> test_calc_mae_with_file finished!')

if __name__ == '__main__':
    test_calc_mae_with_file()