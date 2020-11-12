import os
import sys

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
        [input_d, input_h, input_w] = size

        z_min_upper = img_d - input_d
        y_min_upper = img_h - input_h
        x_min_upper = img_w - input_w

        Z_min = np.random.randint(cropped_boundary.boundary_d_min, z_min_upper)
        Y_min = np.random.randint(cropped_boundary.boundary_h_min, y_min_upper)
        X_min = np.random.randint(cropped_boundary.boundary_w_min, x_min_upper)

        Z_max = Z_min + input_d
        Y_max = Y_min + input_h
        X_max = X_min + input_w

    @staticmethod
    def get_center_crop_boundary_3d(crop_size, cropped_boundary):
        '''
        note: crop_size 应该<= 待切割影像的尺寸
        crop_size: [dim_z, dim_y, dim_x], 目标size
        cropped_boundary： [min_z, min_y, min_x, max_z, max_y, max_x]限定区域，若整图都可做crop，则此处cropped_boundary为图片的全区域
        '''
        padding = 1
        [img_d, img_h, img_w] = [cropped_boundary[3]+padding, cropped_boundary[4]+padding, cropped_boundary[5]+padding]
        center_d =  (cropped_boundary.[3] + cropped_boundary.[0]) // 2
        center_h =  (cropped_boundary.[4] + cropped_boundary.[1]) // 2
        center_w =  (cropped_boundary.[5] + cropped_boundary.[2]) // 2
        [input_d, input_h, input_w] = size

        Z_min = center_d-input_d//2
        Y_min = center_h-input_h//2
        X_min = center_w-input_w//2

        Z_max = Z_min + input_d
        Y_max = Y_min + input_h
        X_max = X_min + input_w

