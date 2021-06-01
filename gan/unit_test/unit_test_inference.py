import os
import sys
import time

import numpy as np

import torch
from torch.utils.data import DataLoader

from tqdm import tqdm
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), os.path.pardir))

from options.test_options import TestOptions
from util.visualizer import Visualizer
import models
from models.pix2pix_3d_model import Pix2Pix3DModel

sys.path.append(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir))

from utils.distributed_utils import DistributedUtils


from datasets.common_ds import GAN_COMMON_DS, get_common_transform

import SimpleITK as sitk

root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir))
# print(root)
sys.path.append(root)
from utils.image_show_utils import ImageShowUtils
from utils.metrics_utils import MetricsUtils

def inference():
    opt = TestOptions().parse()
    print(opt)

    data_root = '/data/medical/cardiac/cta2mbf/data_66_20210517/5.mbf_myocardium'
    # transform = get_common_transform([416,416,160],'GAN_INFERENCE')
    transform = get_common_transform([384,384,160],'GAN_INFERENCE')
    ds = GAN_COMMON_DS(data_root, 'cropped_cta.nii.gz', 'cropped_mbf.nii.gz', [64,64,64], transform)
    dataloader = DataLoader(ds, batch_size=1, num_workers=2, shuffle=True, pin_memory=True)
    dataset_size = len(dataloader) 

    model = models.create_model(opt)
    model.setup(opt)

    model.netG.eval()

    out_dir = "/data/medical/cardiac/cta2mbf/data_66_20210517/6.inference_384x384x160"
    os.makedirs(out_dir, exist_ok=True)

    for index, (subjects) in tqdm(enumerate(dataloader)):
        real_a = subjects['src']['data'].float()
        real_b = subjects['dst']['data'].float()
        input = {}
        input['A'] = real_a
        input['B'] = real_b
        input['A_paths'] = 'A'
        input['B_paths'] = 'B'
        model.set_input(input)
        fake_b = model.netG(real_a.cuda())
        fake_b = fake_b.detach().squeeze().cpu().numpy()
        real_a = real_a.squeeze().cpu().numpy()
        real_b = real_b.squeeze().cpu().numpy()
        real_a = np.transpose(real_a, [2,1,0])
        real_b = np.transpose(real_b, [2,1,0])
        fake_b = np.transpose(fake_b, [2,1,0])
        pid = subjects['src']['path'][0].split('/')[-2]
        info_img = sitk.ReadImage(subjects['src']['path'][0])
        spacing = info_img.GetSpacing()
        direction = info_img.GetDirection()
        origin = info_img.GetOrigin()
        out_sub_dir = os.path.join(out_dir, pid)
        os.makedirs(out_sub_dir, exist_ok=True)
        real_img_a = sitk.GetImageFromArray(real_a)
        real_img_a.SetSpacing(spacing)
        real_img_a.SetDirection(direction)
        real_img_a.SetOrigin(origin)
        real_img_b = sitk.GetImageFromArray(real_b)
        real_img_b.CopyInformation(real_img_a)
        fake_img_b = sitk.GetImageFromArray(fake_b)
        fake_img_b.CopyInformation(real_img_a)
        sitk.WriteImage(real_img_a, os.path.join(out_sub_dir, 'real_a.nii.gz'))
        sitk.WriteImage(real_img_b, os.path.join(out_sub_dir, 'real_b.nii.gz'))
        sitk.WriteImage(fake_img_b, os.path.join(out_sub_dir, 'fake_b.nii.gz'))
        print('hello world!')       

def export_slicemap_onecase(data_root, out_root):
    ww = 150
    wl = 75
    real_a_file = os.path.join(data_root, 'real_a.nii.gz')
    real_b_file = os.path.join(data_root, 'real_b.nii.gz')
    fake_b_file = os.path.join(data_root, 'fake_b.nii.gz')

    real_a_img = sitk.ReadImage(real_a_file)
    real_b_img = sitk.ReadImage(real_b_file)
    fake_b_img = sitk.ReadImage(fake_b_file)

    real_a_arr = sitk.GetArrayFromImage(real_a_img)
    real_b_arr = sitk.GetArrayFromImage(real_b_img)
    fake_b_arr = sitk.GetArrayFromImage(fake_b_img)

    ImageShowUtils.save_volume_to_jpg(real_a_arr, os.path.join(out_root, 'real_a'), ww, wl, axis=0, file_prefix='x', reverse=False, lut_name='jet')
    ImageShowUtils.save_volume_to_jpg(real_b_arr, os.path.join(out_root, 'real_b'), ww, wl, axis=0, file_prefix='x', reverse=False, lut_name='jet')
    ImageShowUtils.save_volume_to_jpg(fake_b_arr, os.path.join(out_root, 'fake_b'), ww, wl, axis=0, file_prefix='x', reverse=False, lut_name='jet')
    print('hello world!')

def export_slicemap(
        data_root='/data/medical/cardiac/cta2mbf/data_66_20210517/6.inference', 
        out_root = '/data/medical/cardiac/cta2mbf/data_66_20210517/6.inference_slicemap'
    ):
    for suid in tqdm(os.listdir(data_root)):
        sub_data_root = os.path.join(data_root, suid)
        sub_out_root = os.path.join(out_root, suid)
        export_slicemap_onecase(sub_data_root, sub_out_root)

def calc_mae(
        data_root='/data/medical/cardiac/cta2mbf/data_66_20210517/6.inference_384x384x160_eval', 
        out_dir = '/data/medical/cardiac/cta2mbf/data_66_20210517/7.analysis_result'
    ):
    row_elems = []
    for suid in tqdm(os.listdir(data_root)):
        sub_data_root = os.path.join(data_root, suid)
        real_b_file = os.path.join(sub_data_root, 'real_b.nii.gz')
        fake_b_file = os.path.join(sub_data_root, 'fake_b.nii.gz') 
        _, mae = MetricsUtils.calc_mae_with_file(real_b_file, fake_b_file)
        row_elems.append(np.array([suid, mae]))
    df = pd.DataFrame(np.array(row_elems), columns=['inhale_suid', 'mae'])
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, 'mae_384x384x160_eval.csv')
    df.to_csv(out_file)


if __name__ == '__main__':
    # inference()
    calc_mae()
