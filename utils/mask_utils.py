import os
import sys

import SimpleITK as sitk

class MaskUtils:
    def __init__(self) -> None:
        pass

    @staticmethod
    def fix_mask_label(mask_arr, mask_label):
        if mask_label is not None:
            if isinstance(mask_label, list):
                mask_mask_arr = mask_arr != mask_label[0]
                for l in range(1, len(mask_label)):
                    mask_mask_arr = mask_mask_arr & (mask_arr != mask_label[l])
                mask_arr[mask_mask_arr] = 0
                mask_arr[~mask_mask_arr] = 1
            else:
                mask_arr[mask_arr != mask_label] = 0
                mask_arr[mask_arr == mask_label] = 1  
        return mask_arr  

    @staticmethod
    def fill_hole(in_mask, radius=8):
        filter = sitk.VotingBinaryHoleFillingImageFilter()
        filter.SetBackgroundValue(0)
        filter.SetForegroundValue(1)
        filter.SetRadius(radius)
        out_mask = filter.Execute(in_mask)
        return out_mask    