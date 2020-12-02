import os
import numpy as np
import SimpleITK as sitk
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


def resample_cardiac_one_case(img_path, dst_size):
    '''
    in_file: nii.gz file
    return: SimpleITK object
    '''
    img = sitk.ReadImage(img_path)
    print(img.GetSize(), img.GetSpacing())
    res_factor = list()
    for s_size, d_size in zip(img.GetSize(), dst_size):
        res_factor.append(s_size / d_size)
    print('res_factor:{}'.format(res_factor))       
    dst_spacing = list()
    for spacing, factor in zip(img.GetSpacing(), res_factor):
        dst_spacing.append(spacing * factor)   
    print('dst_spacing:{}'.format(dst_spacing)) 

    resampler = sitk.ResampleImageFilter()
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    resampler.SetOutputDirection(img.GetDirection())
    resampler.SetOutputOrigin(img.GetOrigin())
    resampler.SetOutputSpacing(dst_spacing)    
    resampler.SetSize(dst_size)  

    img_res = resampler.Execute(img)
    print(img_res.GetSize(), img_res.GetSpacing())
    img_res_arr = sitk.GetArrayFromImage(img_res)
    print(img_res_arr.shape, np.max(img_res_arr), np.min(img_res_arr))

    # new_img_sitk = sitk.GetImageFromArray(img_res_arr)

    return img, img_res, img_res_arr, img.GetSpacing(), dst_spacing


def resample_cardiac_to_raw_size(img, raw_spc, dst_spacing, direct, origin):
    '''
    '''
    resampler = sitk.ResampleImageFilter()
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    resampler.SetOutputDirection(direct)
    resampler.SetOutputOrigin(origin)
    resampler.SetOutputSpacing(raw_spc)    
    
    # to do: 
    dst_size = []
    for i in range(3):
        dst_size.append(int(dst_spacing[i]/raw_spc[i]*img.GetSize()[i]))

    resampler.SetSize(dst_size)
    img_res = resampler.Execute(img)
    print(img_res.GetSize(), img_res.GetSpacing())
    img_res_arr = sitk.GetArrayFromImage(img_res)
    print(img_res_arr.shape, np.max(img_res_arr), np.min(img_res_arr))
    return img_res, img_res_arr


def resample_cardiac_batch(data_root, save_root, dst_size):
    seriesID_ = os.listdir(data_root)
    for case, seriesID in tqdm(enumerate(seriesID_)):
        print('case:{} seriesID:{}'.format(case, seriesID)) 
        img_path = os.path.join(data_root, seriesID, 'img.nii.gz')
        mask_path = os.path.join(data_root, seriesID, 'mask.nii.gz')
        img = sitk.ReadImage(img_path)
        mask = sitk.ReadImage(mask_path)
        print(img.GetSize(), img.GetSpacing())

        res_factor = list()
        for s_size, d_size in zip(img.GetSize(), dst_size):
            res_factor.append(s_size / d_size)
        print('res_factor:{}'.format(res_factor))       
        dst_spacing = list()
        for spacing, factor in zip(img.GetSpacing(), res_factor):
            dst_spacing.append(spacing * factor)   
        print('dst_spacing:{}'.format(dst_spacing))        

        resampler = sitk.ResampleImageFilter()
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)
        resampler.SetOutputDirection(img.GetDirection())
        resampler.SetOutputOrigin(img.GetOrigin())
        resampler.SetOutputSpacing(dst_spacing)    
        resampler.SetSize(dst_size)

        img_res = resampler.Execute(img)
        print(img_res.GetSize(), img_res.GetSpacing())
        img_res_arr = sitk.GetArrayFromImage(img_res)
        print(img_res_arr.shape, np.max(img_res_arr), np.min(img_res_arr))

        mask_res = resampler.Execute(mask)
        print(mask_res.GetSize(), mask_res.GetSpacing())
        mask_res_arr = sitk.GetArrayFromImage(mask_res)
        print(mask_res_arr.shape, np.unique(mask_res_arr))

        mask_res_arr[mask_res_arr!=6] = 0
        mask_res_arr[mask_res_arr==6] = 1
        
        save_dir = os.path.join(save_root, seriesID)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        np.save(os.path.join(save_dir, 'img.npy'), img_res_arr)
        np.save(os.path.join(save_dir, 'mask.npy'), mask_res_arr)
        new_img_sitk = sitk.GetImageFromArray(img_res_arr)
        new_mask_sitk = sitk.GetImageFromArray(mask_res_arr)
        sitk.WriteImage(new_img_sitk, os.path.join(save_dir, 'img.nii'))
        sitk.WriteImage(new_mask_sitk, os.path.join(save_dir, 'mask.nii'))


def test_resample_cardiac_batch():
    data_root = '../data/seg/nii_file'
    save_root = '../data/seg/np_file'
    dst_size = [128, 128, 128]
    os.makedirs(save_root, exist_ok=True)
    resample_cardiac_batch(data_root, save_root, dst_size)

def split_train_val_set(data_root, out_config_dir, train_ratio=0.9):
    '''
    split_train_val_set('../data/seg/np_file', '../data/seg/config')
    '''
    series_uids = os.listdir(data_root)
    series_uids.sort()
    train_pos = int(len(series_uids)*train_ratio)
    train_series_uids = series_uids[:train_pos]
    val_series_uids = series_uids[train_pos:]
    os.makedirs(out_config_dir, exist_ok=True)
    out_train_file = os.path.join(out_config_dir, 'train.config')
    out_val_file = os.path.join(out_config_dir, 'val.config')
    with open(out_train_file, 'w') as f:
        f.write('\n'.join(train_series_uids))
    with open(out_val_file, 'w') as f:
        f.write('\n'.join(val_series_uids))


class CardiacSeg_DS(Dataset):
    def __init__(self, data_root, config_file):
        series_uids = os.listdir(data_root)
        self.img_files = []
        self.mask_files = []
        self.series_uids = []
        self.config_uids = []
        with open(config_file, 'r') as f:
            for line in f.readlines():
                line = line.strip()
                if line is None or len(line) == 0:
                    continue
                self.config_uids.append(line)
        for sid in series_uids:
            if sid not in self.config_uids:
                continue
            series_path = os.path.join(data_root, sid)
            if not os.path.isdir(series_path):
                continue
            image_file = os.path.join(series_path, 'img.nii')
            mask_file = os.path.join(series_path, 'mask.nii')
            if not os.path.isfile(image_file):
                continue
            if not os.path.isfile(mask_file):
                continue
            self.series_uids.append(series_path)
            self.img_files.append(image_file)
            self.mask_files.append(mask_file)
        
    def __getitem__(self, item):
        image_file = self.img_files[item]
        mask_file = self.mask_files[item]
        sitk_image = sitk.ReadImage(image_file)
        sitk_mask = sitk.ReadImage(mask_file)
        image_arr = sitk.GetArrayFromImage(sitk_image)
        mask_arr = sitk.GetArrayFromImage(sitk_mask)
        image_tensor = torch.from_numpy(image_arr).unsqueeze(0).float()
        mask_tensor = torch.from_numpy(mask_arr).long()
        return image_tensor, mask_tensor, image_file, mask_file

    def __len__(self):
        return len(self.img_files)



def test_CardiacSeg_DS():
    data_root = '../data/seg/np_file'
    ds = CardiacSeg_DS(data_root)
    dataloader = DataLoader(ds, batch_size=2, pin_memory=True, num_workers=2, drop_last=True)
    for index, (images, masks, _, _) in tqdm(enumerate(dataloader)):
        print('images shape:\t', images.shape)





if __name__ == '__main__':
    # test_resample_cardiac_batch()
    # test_CardiacSeg_DS()
    # split_train_val_set('../data/seg/np_file', '../data/seg/config')