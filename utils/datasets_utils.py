import os
import sys

import numpy as np

from tqdm import tqdm
from glob import glob

import SimpleITK as sitk

COMMON_ROOT = os.path.join(os.path.dirname(__file__), os.path.pardir)
sys.path.append(COMMON_ROOT)

from utils.data_io_utils import DataIO

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

        Z_min = np.random.randint(cropped_boundary[0], z_min_upper) if cropped_boundary[0] < z_min_upper else 0
        Y_min = np.random.randint(cropped_boundary[1], y_min_upper) if cropped_boundary[1] < y_min_upper else 0
        X_min = np.random.randint(cropped_boundary[2], x_min_upper) if cropped_boundary[2] < x_min_upper else 0

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
        center_d =  (cropped_boundary[3] + cropped_boundary[0] + padding) // 2
        center_h =  (cropped_boundary[4] + cropped_boundary[1] + padding) // 2
        center_w =  (cropped_boundary[5] + cropped_boundary[2] + padding) // 2
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

    @staticmethod
    def collapse_multiples_of_n(in_arr, ori_arr, n):
        [d,h,w] = ori_arr.shape

        new_d = ((d+n-1)//n) * n
        new_h = ((h+n-1)//n) * n
        new_w = ((w+n-1)//n) * n

        assert (new_d, new_h, new_w) == in_arr.shape

        beg_d = new_d//2 - d//2
        beg_h = new_h//2 - h//2
        beg_w = new_w//2 - w//2

        new_arr = np.zeros([d, h, w], dtype=in_arr.dtype)

        new_arr[:, :, :] = in_arr[beg_d:beg_d+d, beg_h:beg_h+h, beg_w:beg_w+w]

        return new_arr

    @staticmethod
    def resample_image_unified_resolution(image, dst_size, interpolation_mode=sitk.sitkNearestNeighbor):
        img = image
        # print(img.GetSize(), img.GetSpacing())
        
        res_factor = list()
        for s_size, d_size in zip(img.GetSize(), dst_size):
            res_factor.append(s_size / d_size)
        # print('res_factor:{}'.format(res_factor))       
        dst_spacing = list()
        for spacing, factor in zip(img.GetSpacing(), res_factor):
            dst_spacing.append(spacing * factor)   
        # print('dst_spacing:{}'.format(dst_spacing))   

        resampler = sitk.ResampleImageFilter()
        resampler.SetInterpolator(interpolation_mode)
        resampler.SetOutputDirection(img.GetDirection())
        resampler.SetOutputOrigin(img.GetOrigin())
        resampler.SetOutputSpacing(dst_spacing)    
        resampler.SetSize(dst_size)

        img_res = resampler.Execute(img)
        # print(img_res.GetSize(), img_res.GetSpacing())
        img_res_arr = sitk.GetArrayFromImage(img_res)
        # print(img_res_arr.shape, np.max(img_res_arr), np.min(img_res_arr))

        new_img_sitk = sitk.GetImageFromArray(img_res_arr)

        return new_img_sitk

    @staticmethod
    def resample_image_mask_unified_resolution_onecase(image_file, mask_file, dst_image_file, dst_mask_file, dst_size, is_dcm=False):
        if is_dcm:
            image_data = DataIO.load_dicom_series(image_file)
        else:
            image_data = DataIO.load_nii_image(image_file)

        mask_data = DataIO.load_nii_image(mask_file)
        
        resampled_image = DatasetsUtils.resample_image_unified_resolution(image_data['sitk_image'], dst_size)
        resampled_mask = DatasetsUtils.resample_image_unified_resolution(mask_data['sitk_image'], dst_size)

        os.makedirs(os.path.dirname(dst_image_file), exist_ok=True)
        os.makedirs(os.path.dirname(dst_mask_file), exist_ok=True)
        sitk.WriteImage(resampled_image, dst_image_file)
        sitk.WriteImage(resampled_mask, dst_mask_file)

    @staticmethod
    def resample_image_mask_unified_resolution_singletask(series_uids, image_root, mask_root, 
            dst_image_root, dst_mask_root, dst_size, 
            image_postfix='', mask_postfix='', 
            is_dcm=False):
        for series_uid in tqdm(series_uids):
            if is_dcm:
                image_file = os.path.join(image_root, '{}'.format(series_uid))
            else:
                image_file = os.path.join(image_root, '{}{}'.format(series_uid, image_postfix))

            mask_file = os.path.join(mask_root, '{}{}'.format(series_uid, mask_postfix))
            dst_image_file = os.path.join(dst_image_root, '{}.nii.gz'.format(series_uid))
            dst_mask_file = os.path.join(dst_mask_root, '{}.nii.gz'.format(series_uid))

            DatasetsUtils.resample_image_mask_unified_resolution_onecase(image_file, mask_file, dst_image_file, dst_mask_file, dst_size, is_dcm)
            

    @staticmethod
    def resample_image_mask_unified_resolution_multiprocess(image_root, mask_root, 
            dst_image_root, dst_mask_root, dst_size, 
            image_postfix='', mask_postfix='', 
            process_num=12, is_dcm=False):


        series_uids = []
        if is_dcm:
            series_uids = os.listdir(image_root)
        else:
            series_uids = glob(os.path.join(image_root, '*{}'.format(image_postfix)))
            series_uids = [os.path.basename(i).replace(image_postfix, '') for i in series_uids]

        # print(series_uids)
        num_per_process = (len(series_uids) + process_num - 1)//process_num

        # this for single thread to debug
        # DatasetsUtils.resample_image_mask_unified_resolution_singletask(series_uids, image_root, mask_root, 
        #                 dst_image_root, dst_mask_root, dst_size, 
        #                 image_postfix, mask_postfix, 
        #                 is_dcm)

        # this for run 
        import multiprocessing
        from multiprocessing import Process
        multiprocessing.freeze_support()

        pool = multiprocessing.Pool()

        results = []

        print(len(series_uids))
        for i in range(process_num):
            sub_series_uids = series_uids[num_per_process*i:min(num_per_process*(i+1), len(series_uids))]
            print(len(sub_series_uids))
            result = pool.apply_async(DatasetsUtils.resample_image_mask_unified_resolution_singletask, 
                args=(sub_series_uids, image_root, mask_root, 
                        dst_image_root, dst_mask_root, dst_size, 
                        image_postfix, mask_postfix, 
                        is_dcm))
            results.append(result)

        pool.close()
        pool.join()


    @staticmethod
    def split_ds(image_root, out_config_dir, train_ratio=0.7, val_ratio=0.2):
        '''
        debug cmd: split_ds('/fileser/zhangwd/data/lung/changzheng/airway/airway_20201030/images', '/fileser/zhangwd/data/lung/changzheng/airway/airway_20201030/config')

        out_config_dir listed as follows:
            tree -L 1
            .
            ├── test.txt
            ├── train.txt
            └── val.txt

        
        'less train.txt' as follows:
            1.2.840.113704.1.111.5624.1392092458.10
            1.2.840.113704.1.111.6896.1389252289.9
            1.3.46.670589.33.1.63725405821017542900002.4919856832254375598
            1.2.840.113704.1.111.2452.1387439529.10
            1.2.840.113704.1.111.6756.1592183917.11
            1.2.840.113704.1.111.8660.1421889850.10
            1.2.840.113704.1.111.11692.1420599548.14
            1.3.46.670589.33.1.63722560084727458900002.4851763629495772847
            1.2.840.113704.1.111.13172.1389599763.7
            1.3.46.670589.33.1.63700781943575774800001.5142437508376053996
            1.2.840.113704.1.111.10192.1571886399.11
            1.2.840.113704.1.111.9536.1577060319.15
            1.2.840.113704.1.111.1384.1392885868.9
            train.txt (END)

        '''
        series_uids = os.listdir(image_root)
        series_uids = [i.replace('.nii.gz', '') for i in series_uids]
        np.random.shuffle(series_uids)
        train_pos = int(len(series_uids)*train_ratio)
        val_pos = int(len(series_uids)*(train_ratio+val_ratio))
        
        train_series_uids = series_uids[:train_pos]
        val_series_uids = series_uids[train_pos:val_pos]
        test_series_uids = series_uids[val_pos:]

        os.makedirs(out_config_dir, exist_ok=True)
        with open(os.path.join(out_config_dir, 'train.txt'), 'w') as f:
            f.write('\n'.join(train_series_uids))

        with open(os.path.join(out_config_dir, 'val.txt'), 'w') as f:
            f.write('\n'.join(val_series_uids))

        with open(os.path.join(out_config_dir, 'test.txt'), 'w') as f:
            f.write('\n'.join(test_series_uids))   



def test_resample_image_mask_unified_resolution_multiprocess():
    
    '''
    我就是个分割线，下面的是心脏腔室的处理
    '''

    # image_root = '/fileser/zhangwd/data/cardiac/chamber/seg/chamber_seg/images'
    # mask_root = '/fileser/zhangwd/data/cardiac/chamber/seg/chamber_seg/masks'

    # dst_image_root = '/fileser/zhangwd/data/cardiac/chamber/seg/chamber_seg_resampled_unified/images'
    # dst_mask_root = '/fileser/zhangwd/data/cardiac/chamber/seg/chamber_seg_resampled_unified/masks'

    # dst_size = [128, 128, 128]

    # image_postfix = '.nii.gz'
    # mask_postfix = '.nii.gz'

    # process_num=12

    # is_dcm = False

    # DatasetsUtils.resample_image_mask_unified_resolution_multiprocess(
    #     image_root, mask_root, 
    #     dst_image_root, dst_mask_root, dst_size, 
    #     image_postfix, mask_postfix, process_num, is_dcm)


    '''
    我就是个分割线，下面的是心包的处理
    '''

    image_root = '/fileser/zhangwd/data/cardiac/seg/heart_hub/images'
    mask_root = '/fileser/zhangwd/data/cardiac/seg/heart_hub/renamed_masks'

    dst_image_root = '/fileser/zhangwd/data/cardiac/seg/heart_hub/resampled_unified_128/images'
    dst_mask_root = '/fileser/zhangwd/data/cardiac/seg/heart_hub/resampled_unified_128/masks'

    dst_size = [128, 128, 128]

    image_postfix = ''
    mask_postfix = '.mha'

    process_num=12

    is_dcm = True

    DatasetsUtils.resample_image_mask_unified_resolution_multiprocess(
        image_root, mask_root, 
        dst_image_root, dst_mask_root, dst_size, 
        image_postfix, mask_postfix, process_num, is_dcm)



if __name__ == '__main__':
    test_resample_image_mask_unified_resolution_multiprocess()

