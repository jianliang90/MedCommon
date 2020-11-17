import os
import sys

import numpy as np

class DatasetsUtils:
    def __init__(self):
        pass

    @staticmethod
    def get_random_crop_boundary_3d(crop_size, cropped_boundary):
        '''
        note: crop_size 应该<= 待切割影像的尺寸
        crop_size: [dim_z, dim_y, dim_x], 目标size
        cropped_boundary： [min_z, min_y, min_x, max_z, max_y, max_x]限定区域，若整图都可做crop，则此处cropped_boundary为图片的全区域
        '''
        padding = 1
        [img_d, img_h, img_w] = [cropped_boundary[3]+padding, cropped_boundary[4]+padding, cropped_boundary[5]+padding]
        [input_d, input_h, input_w] = crop_size

        z_min_upper = img_d - input_d
        y_min_upper = img_h - input_h
        x_min_upper = img_w - input_w

        Z_min = np.random.randint(cropped_boundary[0], z_min_upper)
        Y_min = np.random.randint(cropped_boundary[1], y_min_upper)
        X_min = np.random.randint(cropped_boundary[2], x_min_upper)

        Z_max = Z_min + input_d
        Y_max = Y_min + input_h
        X_max = X_min + input_w

        return Z_min, Y_min, X_min, Z_max, Y_max, X_max

    @staticmethod
    def get_center_crop_boundary_3d(crop_size, cropped_boundary):
        '''
        note: crop_size 应该<= 待切割影像的尺寸
        crop_size: [dim_z, dim_y, dim_x], 目标size
        cropped_boundary： [min_z, min_y, min_x, max_z, max_y, max_x]限定区域，若整图都可做crop，则此处cropped_boundary为图片的全区域
        '''
        padding = 1
        [img_d, img_h, img_w] = [cropped_boundary[3]+padding, cropped_boundary[4]+padding, cropped_boundary[5]+padding]
        center_d =  (cropped_boundary[3] + cropped_boundary[0]) // 2
        center_h =  (cropped_boundary[4] + cropped_boundary[1]) // 2
        center_w =  (cropped_boundary[5] + cropped_boundary[2]) // 2
        [input_d, input_h, input_w] = crop_size

        Z_min = center_d-input_d//2
        Y_min = center_h-input_h//2
        X_min = center_w-input_w//2

        Z_max = Z_min + input_d
        Y_max = Y_min + input_h
        X_max = X_min + input_w

        return Z_min, Y_min, X_min, Z_max, Y_max, X_max

    @staticmethod
    def expand_to_multiples_of_n(in_arr, n):
        '''
        n is multiples of 2
        '''
        [d,h,w] = in_arr.shape
        new_d = ((d+n-1)//n) * n
        new_h = ((h+n-1)//n) * n
        new_w = ((w+n-1)//n) * n

        new_arr = np.zeros([new_d, new_h, new_w], dtype=in_arr.dtype)

        beg_d = new_d//2 - d//2
        beg_h = new_h//2 - h//2
        beg_w = new_w//2 - w//2
        
        new_arr[beg_d:beg_d+d, beg_h:beg_h+h, beg_w:beg_w+w] = in_arr

        return new_arr  


    def collapse_multiples_of_n(in_arr, ori_arr):
        [d,h,w] = ori_arr.shape

        new_d = ((d+n-1)//n) * n
        new_h = ((h+n-1)//n) * n
        new_w = ((w+n-1)//n) * n

        assert [new_d, new_h, new_w] = in_arr.shape

        beg_d = new_d//2 - d//2
        beg_h = new_h//2 - h//2
        beg_w = new_w//2 - w//2

        new_arr = np.zeros([d, h, w], dtype=in_arr.dtype)

        new_arr[:, :, :] = ori_arr[beg_d:beg_d+d, beg_h:beg_h+h, beg_w:beg_w+w]

        return new_arr