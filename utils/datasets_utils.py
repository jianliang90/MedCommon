import os
import sys

import numpy as np

from tqdm import tqdm
from glob import glob

import SimpleITK as sitk

import time

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
    def expand_to_multiples_of_n(in_arr, n, full_val=0):
        '''
        n is multiples of 2
        '''
        [d,h,w] = in_arr.shape
        new_d = ((d+n-1)//n) * n
        new_h = ((h+n-1)//n) * n
        new_w = ((w+n-1)//n) * n

        new_arr = np.full([new_d, new_h, new_w], full_val, dtype=in_arr.dtype)

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
    def extend_image_mask_boundary_for_seg(image_arr, mask_arr, dst_size, boundary_value=0):
        '''
        1. 确保图像的边界(image_arr.shape)<=要扩展的边界（dst_size）
        '''
        assert image_arr.shape == mask_arr.shape
        if image_arr.shape == dst_size:
            return image_arr, mask_arr
        
        z_min_upper = dst_size[0] - image_arr.shape[0]
        y_min_upper = dst_size[1] - image_arr.shape[1]
        x_min_upper = dst_size[2] - image_arr.shape[2]

        z_min = np.random.randint(0, z_min_upper) if 0 < z_min_upper else 0
        y_min = np.random.randint(0, y_min_upper) if 0 < y_min_upper else 0
        x_min = np.random.randint(0, x_min_upper) if 0 < x_min_upper else 0

        z_max = z_min + image_arr.shape[0]
        y_max = y_min + image_arr.shape[1]
        x_max = x_min + image_arr.shape[2]

        image_arr_new = np.full(dst_size, boundary_value, dtype=image_arr.dtype)
        mask_arr_new = np.full(dst_size, 0, dtype=mask_arr.dtype)

        image_arr_new[z_min:z_max, y_min:y_max, x_min:x_max] = image_arr[:,:,:]
        mask_arr_new[z_min:z_max, y_min:y_max, x_min:x_max] = mask_arr[:,:,:]

        return image_arr_new, mask_arr_new

    @staticmethod
    def crop_image_mask_with_padding(image_arr, mask_arr, dst_size, boundary_value=0):
        '''
        1.将图像扩展成dst_size的倍数，再进行随机crop操作
        '''
        assert image_arr.shape == mask_arr.shape
        if image_arr.shape == dst_size:
            return image_arr, mask_arr

        new_size = []
        for i in range(3):
            new_size.append(((image_arr.shape[i]+dst_size[i]-1)//dst_size[i])*dst_size[i])

        new_image_arr = np.full(new_size, boundary_value, dtype=image_arr.dtype)
        new_mask_arr = np.full(new_size, 0, dtype=mask_arr.dtype)

        new_d, new_h, new_w = new_size
        d,h,w = image_arr.shape

        beg_d = new_d//2 - d//2
        beg_h = new_h//2 - h//2
        beg_w = new_w//2 - w//2
        
        new_image_arr[beg_d:beg_d+d, beg_h:beg_h+h, beg_w:beg_w+w] = image_arr
        new_mask_arr[beg_d:beg_d+d, beg_h:beg_h+h, beg_w:beg_w+w] = mask_arr

        z_lower_min = np.random.randint(0, beg_d) if 0 < beg_d else 0
        y_lower_min = np.random.randint(0, beg_h) if 0 < beg_h else 0
        x_lower_min = np.random.randint(0, beg_w) if 0 < beg_w else 0

        z_upper_min = np.random.randint(beg_d+d-dst_size[0], new_d-dst_size[0]) if beg_d+d-dst_size[0] < new_d-dst_size[0] else beg_d+d-dst_size[0]
        y_upper_min = np.random.randint(beg_h+h-dst_size[1], new_h-dst_size[1]) if beg_h+h-dst_size[1] < new_h-dst_size[1] else beg_h+h-dst_size[1]
        x_upper_min = np.random.randint(beg_w+w-dst_size[2], new_w-dst_size[2]) if beg_w+w-dst_size[2] < new_w-dst_size[2] else beg_w+w-dst_size[2]

        z_upper_min = max(z_lower_min, 0)
        y_upper_min = max(y_lower_min, 0)
        x_upper_min = max(x_lower_min, 0)

        z_min = np.random.choice([z_lower_min, z_upper_min])
        y_min = np.random.choice([y_lower_min, y_upper_min])
        x_min = np.random.choice([x_lower_min, x_upper_min])

        z_min = z_min if d > dst_size[0] else 0
        y_min = y_min if h > dst_size[1] else 0
        x_min = x_min if w > dst_size[2] else 0

        z_max = z_min + dst_size[0]
        y_max = y_min + dst_size[1]
        x_max = x_min + dst_size[2]

        # cropped_image = new_image_arr[z_min:z_max, y_min:y_max, x_min:x_max]
        # cropped_mask = new_mask_arr[z_min:z_max, y_min:y_max, x_min:x_max]
        # print(dst_size)
        # assert list(cropped_image.shape) == dst_size
        # print('{}\t{}\t{}\t{}\t{}\t{}'.format(z_min, z_max, y_min, y_max, x_min, x_max))

        return new_image_arr[z_min:z_max, y_min:y_max, x_min:x_max], new_mask_arr[z_min:z_max, y_min:y_max, x_min:x_max]
        


    @staticmethod
    def cut_image_into_blocks_by_sliding_window(image_arr, crop_size, overlap=[0,0,0]):
        '''
        将3d图像按照滑窗的方式，切割成crop_size的大小
        todo: 暂时未提供overlap的版本
        '''      
        # src_data = sitk.GetArrayFromImage(sitk_image)
        src_data = image_arr

        # padding to 32xn/Nxn
        padding = 32
        [pd, ph, pw] = crop_size
        [d,h,w] = src_data.shape
        new_d = ((d+pd-1)//pd)*pd
        new_h = ((h+ph-1)//ph)*ph
        new_w = ((w+pw-1)//pw)*pw

        if not np.all([d,h,w]==np.array([new_d, new_h, new_w])):
            new_arr = np.zeros([new_d, new_h, new_w])
            new_arr[:d,:h,:w] = src_data
        else:
            new_arr = src_data

        cropped_srcs = []
        d_cnt = (d+pd-1)//pd
        h_cnt = (h+ph-1)//ph
        w_cnt = (w+pw-1)//pw
        for iz in range(d_cnt):
            for iy in range(h_cnt):
                for ix in range(w_cnt):
                    cropped_src = new_arr[iz*pd:(iz+1)*pd, iy*ph:(iy+1)*ph, ix*pw:(ix+1)*pw]
                    # cropped_src = torch.from_numpy(cropped_src).float()
                    # cropped_src = torch.unsqueeze(cropped_src, axis=0)
                    # cropped_src = torch.unsqueeze(cropped_src, axis=0)
                    cropped_srcs.append(cropped_src)
        
        return cropped_srcs, d_cnt, h_cnt, w_cnt


    @staticmethod
    def compose_blocks_cutted_by_sliding_window_into_image(arr, blocks_dim, crop_size, ori_size, overlay=[0, 0, 0]):
        '''
        将3d图像按照滑窗的方式，切割成crop_size的大小后，组装成完整的图像
        todo: 暂时未提供overlap的版本
        '''
        assert len(arr) == blocks_dim[0] * blocks_dim[1] * blocks_dim[2]
        dim = np.array(blocks_dim)*np.array(crop_size)
        dst_arr = np.zeros(dim)
        [d_cnt, h_cnt, w_cnt] = blocks_dim
        [pd, ph, pw] = crop_size
        for iz in range(d_cnt):
            for iy in range(h_cnt):
                for ix in range(w_cnt):
                    dst_arr[iz*pd:(iz+1)*pd, iy*ph:(iy+1)*ph, ix*pw:(ix+1)*pw] = arr[iz*h_cnt*w_cnt+iy*w_cnt+ix]
        return dst_arr[:ori_size[0], :ori_size[1], :ori_size[2]]
    


    @staticmethod
    def resample_image_unsame_resolution(image, dst_size, interpolation_mode=sitk.sitkNearestNeighbor):
        '''
        该函数并没有统一分辨率。。。
        '''
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

        return img_res


    @staticmethod
    def restore_ori_image_from_resampled_image(resampled_image, ori_ref_image, interpolation_mode=sitk.sitkNearestNeighbor):
        '''

        '''
        resampler = sitk.ResampleImageFilter()
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)
        resampler.SetOutputDirection(ori_ref_image.GetDirection())
        resampler.SetOutputOrigin(ori_ref_image.GetOrigin())
        resampler.SetOutputSpacing(ori_ref_image.GetSpacing())
        resampler.SetSize(ori_ref_image.GetSize())
        img_res = resampler.Execute(resampled_image)

        return img_res

        

    @staticmethod
    def resample_image_mask_unsame_resolution_onecase(image_file, mask_file, dst_image_file, dst_mask_file, dst_size, is_dcm=False):
        if is_dcm:
            image_data = DataIO.load_dicom_series(image_file)
        else:
            image_data = DataIO.load_nii_image(image_file)

        mask_data = DataIO.load_nii_image(mask_file)
        
        resampled_image = DatasetsUtils.resample_image_unsame_resolution(image_data['sitk_image'], dst_size)
        resampled_mask = DatasetsUtils.resample_image_unsame_resolution(mask_data['sitk_image'], dst_size)

        os.makedirs(os.path.dirname(dst_image_file), exist_ok=True)
        os.makedirs(os.path.dirname(dst_mask_file), exist_ok=True)
        sitk.WriteImage(resampled_image, dst_image_file)
        sitk.WriteImage(resampled_mask, dst_mask_file)

    @staticmethod
    def resample_image_mask_unsame_resolution_singletask(series_uids, image_root, mask_root, 
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

            DatasetsUtils.resample_image_mask_unsame_resolution_onecase(image_file, mask_file, dst_image_file, dst_mask_file, dst_size, is_dcm)
            

    @staticmethod
    def resample_image_mask_unsame_resolution_multiprocess(image_root, mask_root, 
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
        # DatasetsUtils.resample_image_mask_unsame_resolution_singletask(series_uids, image_root, mask_root, 
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
            result = pool.apply_async(DatasetsUtils.resample_image_mask_unsame_resolution_singletask, 
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



def test_resample_image_mask_unsame_resolution_multiprocess():
    
    '''
    我就是个分割线，下面的是心脏腔室的处理
    '''

    image_root = '/fileser/zhangwd/data/cardiac/chamber/seg/chamber_seg/images'
    mask_root = '/fileser/zhangwd/data/cardiac/chamber/seg/chamber_seg/masks'

    dst_image_root = '/fileser/zhangwd/data/cardiac/chamber/seg/chamber_seg_resampled_unified/images'
    dst_mask_root = '/fileser/zhangwd/data/cardiac/chamber/seg/chamber_seg_resampled_unified/masks'

    dst_size = [128, 128, 128]

    image_postfix = '.nii.gz'
    mask_postfix = '.nii.gz'

    process_num=12

    is_dcm = False

    DatasetsUtils.resample_image_mask_unsame_resolution_multiprocess(
        image_root, mask_root, 
        dst_image_root, dst_mask_root, dst_size, 
        image_postfix, mask_postfix, process_num, is_dcm)


    '''
    我就是个分割线，下面的是心包的处理
    '''

    # image_root = '/fileser/zhangwd/data/cardiac/seg/heart_hub/images'
    # mask_root = '/fileser/zhangwd/data/cardiac/seg/heart_hub/renamed_masks'

    # dst_image_root = '/fileser/zhangwd/data/cardiac/seg/heart_hub/resampled_unified_128/images'
    # dst_mask_root = '/fileser/zhangwd/data/cardiac/seg/heart_hub/resampled_unified_128/masks'

    # dst_size = [128, 128, 128]

    # image_postfix = ''
    # mask_postfix = '.mha'

    # process_num=12

    # is_dcm = True

    # DatasetsUtils.resample_image_mask_unsame_resolution_multiprocess(
    #     image_root, mask_root, 
    #     dst_image_root, dst_mask_root, dst_size, 
    #     image_postfix, mask_postfix, process_num, is_dcm)

def test_restore_ori_image_from_resampled_image():
    '''
    我就是个分割线，下面的是: 利用小分辨率的（等分辨率）模型将心脏进行分割，
    并将风格后的心脏mask恢复到原图一样的大小
    '''
    mask_file = '/fileser/zhangwd/data/cardiac/chamber/seg/chamber_seg_resampled_unified/masks/1.3.12.2.1107.5.1.4.60320.30000015020300202700000017926.nii.gz'
    ref_image_file = '/fileser/zhangwd/data/cardiac/chamber/seg/chamber_seg/images/1.3.12.2.1107.5.1.4.60320.30000015020300202700000017926.nii.gz'
    ref_mask_file = '/fileser/zhangwd/data/cardiac/chamber/seg/chamber_seg/masks/1.3.12.2.1107.5.1.4.60320.30000015020300202700000017926.nii.gz'

    resample_mask = sitk.ReadImage(mask_file)
    ori_ref_image = sitk.ReadImage(ref_image_file)
    ori_ref_mask = sitk.ReadImage(ref_mask_file)

    # resample_mask = DatasetsUtils.resample_image_unsame_resolution(ori_ref_mask, [128, 128, 128])

    restored_mask = DatasetsUtils.restore_ori_image_from_resampled_image(resample_mask, ori_ref_image)

    tmp_out_dir = './tmp_out'
    os.makedirs(tmp_out_dir, exist_ok=True)
    out_restored_mask_file = os.path.join(tmp_out_dir, 'restored_mask.nii.gz')
    out_ref_image_file = os.path.join(tmp_out_dir, 'ref_image.nii.gz')
    out_ref_mask_file = os.path.join(tmp_out_dir, 'ref_mask.nii.gz')
    sitk.WriteImage(restored_mask, out_restored_mask_file)
    sitk.WriteImage(ori_ref_image, out_ref_image_file)
    sitk.WriteImage(ori_ref_mask, out_ref_mask_file)

def test_cut_image_into_blocks_by_sliding_window():
    beg = time.time()
    infile = '/fileser/zhangwd/data/lung/changzheng/airway/airway_20201030/pred_masks/1.2.840.113704.1.111.10192.1571886399.11/coarse_lung/cropped_image.nii.gz'
    image = sitk.ReadImage(infile)
    image_arr = sitk.GetArrayFromImage(image)
    crop_size = [128, 128, 128]
    cropped_arrs, d_cnt, h_cnt, w_cnt = DatasetsUtils.cut_image_into_blocks_by_sliding_window(image_arr, crop_size)
    ori_size = list(image.GetSize())[::-1]
    composed_arr = DatasetsUtils.compose_blocks_cutted_by_sliding_window_into_image(cropped_arrs, [d_cnt, h_cnt, w_cnt], crop_size, ori_size)
    composed_image = sitk.GetImageFromArray(composed_arr)
    composed_image.CopyInformation(image)
    
    out_dir = './tmp_out'
    os.makedirs(out_dir, exist_ok=True)
    outfile = os.path.join(out_dir, 'test_cut_image_into_blocks_by_sliding_window.nii.gz')

    sitk.WriteImage(composed_image, outfile)

    end = time.time()
    print('====> test_cut_image_into_blocks_by_sliding_window time elapsed:\t{:.3f}s'.format(end-beg))
    
def test_extend_image_mask_boundary_for_seg():
    beg = time.time()
    image_file = '/fileser/zhangwd/data/lung/changzheng/airway/airway_20201030/paires_croped_by_coarse_lung_seg/images/1.2.840.113704.1.111.10192.1571886399.11.nii.gz'
    mask_file = '/fileser/zhangwd/data/lung/changzheng/airway/airway_20201030/paires_croped_by_coarse_lung_seg/masks/1.2.840.113704.1.111.10192.1571886399.11.nii.gz'

    out_dir = './tmp_out/test_extend_image_mask_boundary_for_seg'
    os.makedirs(out_dir, exist_ok=True)

    sitk_image = sitk.ReadImage(image_file)
    image_arr = sitk.GetArrayFromImage(sitk_image)
    sitk_mask = sitk.ReadImage(mask_file)
    mask_arr = sitk.GetArrayFromImage(sitk_mask)
    for i in tqdm(range(10)):
        out_image_file = os.path.join(out_dir, 'image_{}.nii.gz'.format(i))
        out_mask_file = os.path.join(out_dir, 'mask_{}.nii.gz'.format(i))
        dst_size = [128, 128, 128]
        padding = [np.random.randint(0,5), np.random.randint(0,5), np.random.randint(0,5)]
        crop_size = []
        for i in range(3):
            crop_size.append(dst_size[i] - padding[i])
        
        # 随机取数据
        cropped_boundary = [0,0,0, image_arr.shape[0]-1, image_arr.shape[1]-1, image_arr.shape[2]-1]
        boundary = DatasetsUtils.get_random_crop_boundary_3d(crop_size, cropped_boundary)

        Z_min, Y_min, X_min, Z_max, Y_max, X_max = boundary

        cropped_image = image_arr[Z_min:Z_max, Y_min:Y_max, X_min:X_max]
        cropped_mask = mask_arr[Z_min:Z_max, Y_min:Y_max, X_min:X_max]

        cropped_image, cropped_mask = DatasetsUtils.extend_image_mask_boundary_for_seg(cropped_image, cropped_mask, dst_size)

        out_sitk_image = sitk.GetImageFromArray(cropped_image)
        out_sitk_mask = sitk.GetImageFromArray(cropped_mask)

        sitk.WriteImage(out_sitk_image, out_image_file)
        sitk.WriteImage(out_sitk_mask, out_mask_file)

    end = time.time()
    print('====> test_extend_image_mask_boundary_for_seg time elapsed:\t{:.3f}s'.format(end-beg))

def test_crop_image_mask_with_padding():
    beg = time.time()
    image_file = '/fileser/zhangwd/data/lung/changzheng/airway/airway_20201030/paires_croped_by_coarse_lung_seg/images/1.2.840.113704.1.111.10192.1571886399.11.nii.gz'
    mask_file = '/fileser/zhangwd/data/lung/changzheng/airway/airway_20201030/paires_croped_by_coarse_lung_seg/masks/1.2.840.113704.1.111.10192.1571886399.11.nii.gz'

    out_dir = './tmp_out/test_crop_image_mask_with_padding'
    os.makedirs(out_dir, exist_ok=True)

    sitk_image = sitk.ReadImage(image_file)
    image_arr = sitk.GetArrayFromImage(sitk_image)
    sitk_mask = sitk.ReadImage(mask_file)
    mask_arr = sitk.GetArrayFromImage(sitk_mask)

    # 1. 验证是否出错
    dst_size = [128, 128, 128]
    # for i in tqdm(range(200)):
    #     DatasetsUtils.crop_image_mask_with_padding(image_arr, mask_arr, dst_size, boundary_value=0)

    # 2. check 随机抽样
    # for i in tqdm(range(2000)):
    #     dst_size = [np.random.randint(100, 300), np.random.randint(100, 300), np.random.randint(100, 300)]
    #     cropped_image, cropped_mask = DatasetsUtils.crop_image_mask_with_padding(image_arr, mask_arr, dst_size, boundary_value=0)
    #     assert list(cropped_image.shape) == dst_size

    # 3. 保存查看增强数据
    for i in tqdm(range(10)):
        out_image_file = os.path.join(out_dir, 'image_{}.nii.gz'.format(i))
        out_mask_file = os.path.join(out_dir, 'mask_{}.nii.gz'.format(i))
        dst_size = [np.random.randint(100, 300), np.random.randint(100, 300), np.random.randint(100, 300)]
        cropped_image, cropped_mask = DatasetsUtils.crop_image_mask_with_padding(image_arr, mask_arr, dst_size, boundary_value=0)
        
        out_sitk_image = sitk.GetImageFromArray(cropped_image)
        out_sitk_mask = sitk.GetImageFromArray(cropped_mask)

        sitk.WriteImage(out_sitk_image, out_image_file)
        sitk.WriteImage(out_sitk_mask, out_mask_file)


    end = time.time()
    print('====> test_crop_image_mask_with_padding time elapsed:\t{:.3f}s'.format(end-beg))

if __name__ == '__main__':
    # test_resample_image_mask_unsame_resolution_multiprocess()
    # test_restore_ori_image_from_resampled_image()
    # test_cut_image_into_blocks_by_sliding_window()
    # test_extend_image_mask_boundary_for_seg()
    test_crop_image_mask_with_padding()

