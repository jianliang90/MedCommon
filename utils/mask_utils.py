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