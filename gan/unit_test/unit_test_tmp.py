# 测试itk snap能否显示彩图

import numpy as np
import matplotlib.pyplot as plt

import os
import sys

from tqdm import tqdm

import SimpleITK as sitk
root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir))
print(root)
sys.path.append(root)
from utils.image_show_utils import ImageShowUtils

data_root = '/data/medical/cardiac/cta2mbf/data_66_20210517/6.inference/1865503'
out_root = '/data/medical/cardiac/cta2mbf/data_66_20210517/6.inference_slicemap/1865503'
os.makedirs(out_root, exist_ok=True)




real_a_file = os.path.join(data_root, 'real_a.nii.gz')
real_b_file = os.path.join(data_root, 'real_b.nii.gz')
fake_b_file = os.path.join(data_root, 'fake_b.nii.gz')


def generate_lut3d():
    real_b_img = sitk.ReadImage(real_b_file)
    real_b_arr = sitk.GetArrayFromImage(real_b_img)
    real_b_arr = np.clip(real_b_arr, a_min=0, a_max=150)
    real_b_arr /= 150

    lut = plt.get_cmap('jet')
    real_b_rgb = lut(real_b_arr)

    rgb_img = sitk.GetImageFromArray(real_b_rgb)
    rgb_img.CopyInformation(real_b_img)

    sitk.WriteImage(rgb_img, os.path.join(data_root, 'real_b_rgb.nii.gz'))

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
    

if __name__ == '__main__':
    # export_slicemap_onecase(data_root, out_root)
    export_slicemap()