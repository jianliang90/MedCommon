import os
import numpy as np
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_io_utils import DataIO

class MaskBoundingUtils:
    def __init__(self):
        print('init MaskBoundingUtils class')

    @staticmethod
    def extract_mask_file_bounding(infile, is_dcm=False, is_print=False):
        if is_dcm:
            data = DataIO.load_dicom_series(infile)
        else:
            data = DataIO.load_nii_image(infile)
        arr = data['image']
        return extract_mask_arr_bounding(arr, is_print)
    
    @staticmethod
    def extract_mask_arr_bounding(in_arr, is_print=False):
        ranges = np.where(in_arr > 0)
        [z_min, y_min, x_min] = np.min(np.array(ranges), axis=1)
        [z_max, y_max, x_max] = np.max(np.array(ranges), axis=1)
        if is_print:
            print('mask shape:\t', mask_arr.shape)
            print('z ranges: [{}\t{}], len:\t{}'.format(z_min, z_max, z_max-z_min))
            print('y ranges: [{}\t{}], len:\t{}'.format(y_min, y_max, y_max-y_min))
            print('x ranges: [{}\t{}], len:\t{}'.format(x_min, x_max, x_max-x_min))
        return z_min, y_min, x_min, z_max, y_max, x_max



if __name__ == '__main__':
    print(__file__)