import os
import sys

import numpy as np
from tqdm import tqdm
from glob import glob

import SimpleITK as sitk

import pandas as pd

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_io_utils import DataIO

class AirflowDynamicsAnalysisUtils:
    def __init__(self):
        pass

    @staticmethod
    def generate_mask_from_analysis_csv(in_csv_file, dcm_file, out_dir):
        print('csv file:\t{}'.format(in_csv_file))
        # df = pd.read_csv(in_csv_file)
        basename = os.path.basename(dcm_file)
        fev1_basename = '.'.join(os.path.basename(in_csv_file).split('.')[:-1])
        out_series_dir = os.path.join(out_dir, basename, fev1_basename)
        os.makedirs(out_series_dir, exist_ok=True)
        out_image_file = os.path.join(out_series_dir, 'image.nii.gz')
        out_mask_file = os.path.join(out_series_dir, 'mask.nii.gz')
        out_pressure_file = os.path.join(out_series_dir, 'pressure.nii.gz')
        out_velocity_file = os.path.join(out_series_dir, 'velocity.nii.gz')

        image_data = DataIO.load_dicom_series(dcm_file)
        image_arr = image_data['image']
        
        mask_arr = np.zeros(image_arr.shape, dtype=np.uint8)
        pressure_arr = np.zeros(mask_arr.shape, dtype=np.uint32)
        velocity_arr = np.zeros(mask_arr.shape, dtype=np.uint32)

        spc = image_data['spacing']

        cnt = 0
        cnt_points = 0
        with open(in_csv_file, 'r') as f:
            for line in f.readlines():
                cnt += 1
                if cnt < 7:
                    continue
                line = line.strip()
                if line is None or len(line) == 0:
                    continue
                cnt_points += 0
                ss = line.split(',')
                x = float(ss[0]) * 1000
                y = float(ss[1]) * 1000
                z = float(ss[2]) * 1000

                pressure = float(ss[3])
                velocity = float(ss[4]) * 100
                # velocity_u = float(ss[5])
                # velocity_v = float(ss[6])
                # velocity_w = float(ss[7])

                coord_z = int(z/spc[2])
                coord_y = int(y/spc[1])
                coord_x = int(x/spc[0])

                # print('{}\t{}\t{}'.format(coord_z, coord_y, coord_x))
                mask_arr[coord_z, coord_y, coord_x] = 1
                pressure_arr[coord_z, coord_y, coord_x] = pressure
                velocity_arr[coord_z, coord_y, coord_x] = velocity

        sitk_mask = sitk.GetImageFromArray(mask_arr)

        tmp_img = sitk.GetImageFromArray(mask_arr)
        # tmp_img = sitk.Cast(tmp_img, sitk.sitkInt16)
        dilation_filter = sitk.BinaryDilateImageFilter()
        dilation_filter.SetForegroundValue(1)
        dilation_filter.SetBackgroundValue(0)
        dilation_filter.SetKernelRadius(3)
        tmp_img = dilation_filter.Execute(tmp_img)

        erode_filter = sitk.BinaryErodeImageFilter()
        erode_filter.SetForegroundValue(1)
        erode_filter.SetBackgroundValue(0)
        erode_filter.SetKernelRadius(3)
        tmp_img = erode_filter.Execute(tmp_img)       

        sitk_mask = sitk.Cast(tmp_img, sitk.sitkUInt8)
        sitk_mask.CopyInformation(image_data['sitk_image'])
        sitk_pressure = sitk.GetImageFromArray(pressure_arr)
        sitk_pressure.CopyInformation(image_data['sitk_image']) 
        sitk_velocity = sitk.GetImageFromArray(velocity_arr)
        sitk_velocity.CopyInformation(image_data['sitk_image'])

        gray_dilation_filter = sitk.GrayscaleDilateImageFilter()
        gray_dilation_filter.SetKernelRadius(3)
        gray_erode_filter = sitk.GrayscaleErodeImageFilter()
        gray_erode_filter.SetKernelRadius(3)

        sitk_pressure = gray_dilation_filter.Execute(sitk_pressure)
        sitk_pressure = gray_erode_filter.Execute(sitk_pressure)

        sitk_velocity = gray_dilation_filter.Execute(sitk_velocity)
        sitk_velocity = gray_erode_filter.Execute(sitk_velocity)


        sitk.WriteImage(image_data['sitk_image'], out_image_file)
        sitk.WriteImage(sitk_mask, out_mask_file)
        sitk.WriteImage(sitk_pressure, out_pressure_file)
        sitk.WriteImage(sitk_velocity, out_velocity_file)



def test_generate_mask_from_analysis_csv():
    # in_csv_file = '/data/medical/lung/changzheng/airway/airflow_dynamics_analysis/exp_20201202/COPD气流动力学分析分析_gongfengying_test/FEV1_1.5.csv'
    dcm_file = '/data/medical/lung/changzheng/airway/airway_20201030/images/1.2.840.113704.1.111.2452.1387439529.10'
    in_csv_root = '/data/medical/lung/changzheng/airway/airflow_dynamics_analysis/exp_20201202/COPD气流动力学分析分析_gongfengying_test'
    in_csv_files = glob(os.path.join(in_csv_root, '*FEV1*.csv'))
    out_dir = '/data/medical/lung/changzheng/airway/airflow_dynamics_analysis/exp_20201202/preprocessed_data'
    
    for in_csv_file in tqdm(in_csv_files):
        AirflowDynamicsAnalysisUtils.generate_mask_from_analysis_csv(in_csv_file, dcm_file, out_dir)




if __name__ == '__main__':
    test_generate_mask_from_analysis_csv()

    