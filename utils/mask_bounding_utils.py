import os
import numpy as np
import sys

import SimpleITK as sitk

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
        return MaskBoundingUtils.extract_mask_arr_bounding(arr, is_print)
    
    @staticmethod
    def extract_mask_arr_bounding(in_arr, is_print=False):
        ranges = np.where(in_arr > 0)
        [z_min, y_min, x_min] = np.min(np.array(ranges), axis=1)
        [z_max, y_max, x_max] = np.max(np.array(ranges), axis=1)
        if is_print:
            print('mask shape:\t', in_arr.shape)
            print('z ranges: [{}\t{}], len:\t{}'.format(z_min, z_max, z_max-z_min))
            print('y ranges: [{}\t{}], len:\t{}'.format(y_min, y_max, y_max-y_min))
            print('x ranges: [{}\t{}], len:\t{}'.format(x_min, x_max, x_max-x_min))
            print('mask valid bounding shape:\t[{}, {}, {}]'.format(z_max-z_min, y_max-y_min, x_max-x_min))
        return z_min, y_min, x_min, z_max, y_max, x_max

    @staticmethod
    def extract_target_area_by_boundary_info(infile, out_file, boundary_info, is_dcm=False):
        '''
        boudary_info: [min_z, min_y, min_x, max_z, max_y, max_x], make sure, boundary_info is valid!!!!!
        '''
        if is_dcm:
            data = DataIO.load_dicom_series(infile)
        else:
            data = DataIO.load_nii_image(infile)
        arr = data['image']
        [min_z, min_y, min_x, max_z, max_y, max_x] = boundary_info
        target_arr = arr[min_z:max_z+1, min_y:max_y+1, min_x:max_x+1]
        if out_file is not None:
            os.makedirs(os.path.dirname(out_file), exist_ok=True)
            DataIO.save_medical_info_and_data(target_arr, data['origin'], data['spacing'], data['direction'], out_file)

    @staticmethod
    def extract_segmentation_pairs_by_boundary_info(in_image_file, in_mask_file, 
            out_image_file, out_mask_file, boundary_info, is_dcm=False):
        '''
        boudary_info: [min_z, min_y, min_x, max_z, max_y, max_x], make sure, boundary_info is valid!!!!!
        '''
        if is_dcm:
            image_data = DataIO.load_dicom_series(in_image_file)
        else:
            image_data = DataIO.load_nii_image(in_image_file)
        mask_data = DataIO.load_nii_image(in_mask_file)
        
        [min_z, min_y, min_x, max_z, max_y, max_x] = boundary_info
        
        arr = image_data['image']
        image_target_arr = arr[min_z:max_z+1, min_y:max_y+1, min_x:max_x+1]
        out_sitk_image = sitk.GetImageFromArray(image_target_arr)
        out_sitk_image.SetSpacing(image_data['sitk_image'].GetSpacing())
        out_sitk_image.SetOrigin(image_data['sitk_image'].GetOrigin())
        out_sitk_image.SetDirection(image_data['sitk_image'].GetDirection())
        if out_image_file is not None:
            os.makedirs(os.path.dirname(out_image_file), exist_ok=True)
            sitk.WriteImage(out_sitk_image, out_image_file)
        
        arr = mask_data['image']
        mask_target_arr = arr[min_z:max_z+1, min_y:max_y+1, min_x:max_x+1]
        out_sitk_mask = sitk.GetImageFromArray(mask_target_arr)
        out_sitk_mask.CopyInformation(out_sitk_image)
        if out_mask_file is not None:
            os.makedirs(os.path.dirname(out_mask_file), exist_ok=True)
            sitk.WriteImage(out_sitk_mask, out_mask_file)




if __name__ == '__main__':
    print(__file__)