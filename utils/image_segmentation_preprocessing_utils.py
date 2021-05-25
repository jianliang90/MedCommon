import os
import sys

root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
sys.path.append(root)

from utils.mask_bounding_utils import MaskBoundingUtils
from utils.data_io_utils import DataIO
from utils.datasets_utils import DatasetsUtils
from utils.image_postprocessing_utils import ImagePostProcessingUtils

import numpy as np
from tqdm import tqdm

class ImageSegmentationPreprocessingUtils:
    def __init__(self) -> None:
        pass

    '''
    分析分割mask的边界信息，并输出文件
    '''
    @staticmethod
    def analyze_mask_boundary(mask_root = '/data/medical/brain/Cerebrovascular/segmenation/renamed_masks', 
        out_root = '/data/medical/brain/Cerebrovascular/segmenation/result/analysis_result',
        out_filename='boundary_info_ori_mask.txt'):
        
        # mask_root = '/data/medical/brain/Cerebrovascular/segmenation/renamed_masks'
        max_depth = 0
        max_height = 0
        max_width = 0
        logs = []

        depths = []
        heights = []
        widths = []
        for mask_name in tqdm(os.listdir(mask_root)):
            mask_file = os.path.join(mask_root, mask_name)
            z_min, y_min, x_min, z_max, y_max, x_max = MaskBoundingUtils.extract_mask_file_bounding(mask_file)
            depth = z_max - z_min
            height = y_max - y_min
            width = x_max - x_min
            if depth > max_depth:
                max_depth = depth
            if height > max_height:
                max_height = height
            if width > max_width:
                max_width = width
            depths.append(depth)
            heights.append(height)
            widths.append(width)

            log_info = 'depth:\t{}\theight:\t{}\twidth:\t{}\t{}'.format(depth, height, width, mask_name)
            logs.append(log_info)
        log_info = 'depth:\t{}\theight:\t{}\twidth:\t{}\t{}'.format(max_depth, max_height, max_width, 'max info')
        logs.append(log_info)
        log_info = 'depth:\t{}\theight:\t{}\twidth:\t{}\t{}'.format(
            np.array(depths).sum()/len(depths), np.array(heights).sum()/len(heights), np.array(widths).sum()/len(widths), 'mean info')
        logs.append(log_info)
        log_info = 'depth:\t{}\theight:\t{}\twidth:\t{}\t{}'.format(
            depths[len(depths)//2], heights[len(heights)//2], widths[len(widths)//2], 'middle info')
        logs.append(log_info)    

        os.makedirs(out_root, exist_ok=True)
        out_file = os.path.join(out_root, out_filename)
        with open(out_file, 'w') as f:
            f.write('\n'.join(logs))

        for log in logs:
            print(log)

    '''
    分析影像的边界信息并，输出文件
    '''
    @staticmethod
    def analyze_image_boundary(image_root = '/data/medical/brain/Cerebrovascular/segmenation/images', 
            out_root = '/data/medical/brain/Cerebrovascular/segmenation/result/analysis_result'
        ):
        max_depth = 0
        max_height = 0
        max_width = 0
        logs = []

        depths = []
        heights = []
        widths = []    
        for suid in tqdm(os.listdir(image_root)):
            image_file = os.path.join(image_root, suid)
            image = DataIO.load_dicom_series(image_file)['sitk_image']
            width,height,depth = image.GetSize()
            if depth > max_depth:
                max_depth = depth
            if height > max_height:
                max_height = height
            if width > max_width:
                max_width = width

            depths.append(depth)
            heights.append(height)
            widths.append(width)            
            log_info = 'depth:\t{}\theight:\t{}\twidth:\t{}\t{}'.format(depth, height, width, suid)
            logs.append(log_info)
            print(log_info)
        log_info = 'depth:\t{}\theight:\t{}\twidth:\t{}\t{}'.format(max_depth, max_height, max_width, 'image info')
        logs.append(log_info)
        log_info = 'depth:\t{}\theight:\t{}\twidth:\t{}\t{}'.format(
            np.array(depths).sum()/len(depths), np.array(heights).sum()/len(heights), np.array(widths).sum()/len(widths), 'mean info')
        logs.append(log_info)
        log_info = 'depth:\t{}\theight:\t{}\twidth:\t{}\t{}'.format(
            depths[len(depths)//2], heights[len(heights)//2], widths[len(widths)//2], 'middle info')    
        os.makedirs(out_root, exist_ok=True)
        out_file = os.path.join(out_root, 'boundary_info_ori_images.txt')
        with open(out_file, 'w') as f:
            f.write('\n'.join(logs))

        for log in logs:
            print(log) 

    @staticmethod
    def generate_image_mask_pairs_onecase(ref_mask_root, out_root, image_root, mask_root, suid, is_dcm=True, mask_pattern='.mha'):
        # suid = '1.2.392.200036.9116.2.2054276706.1588060001.16.1155800007.1.nii.gz'
        # suid = '1.2.392.200036.9116.2.2054276706.1588550054.6.1177300002.1.nii.gz'
        suid = suid.replace('.nii.gz', '')
        out_image_root = os.path.join(out_root, 'images')
        os.makedirs(out_image_root, exist_ok=True)
        out_mask_root = os.path.join(out_root, 'masks')
        os.makedirs(out_mask_root, exist_ok=True)
        ref_mask_file = os.path.join(ref_mask_root, '{}.nii.gz'.format(suid))
        boundary_info = MaskBoundingUtils.extract_mask_file_bounding(ref_mask_file)
        in_image_file = os.path.join(image_root, '{}'.format(suid))
        in_mask_file = os.path.join(mask_root, '{}{}'.format(suid, mask_pattern))
        out_image_file = os.path.join(out_image_root, '{}.nii.gz'.format(suid))
        out_mask_file = os.path.join(out_mask_root, '{}.nii.gz'.format(suid))
        # MaskBoundingUtils.extract_target_area_by_boundary_info(in_image_file, out_image_file, boundary_info, True)
        # MaskBoundingUtils.extract_target_area_by_boundary_info(in_mask_file, out_mask_file, boundary_info, False)        
        padding=2
        MaskBoundingUtils.extract_segmentation_pairs_by_boundary_info(in_image_file, in_mask_file, out_image_file, out_mask_file, boundary_info, is_dcm, padding=padding)

    @staticmethod
    def generate_image_mask_pairs_singletask(ref_mask_root, out_root, image_root, mask_root, suids, is_dcm=True, mask_pattern='.mha'):
        for suid in tqdm(suids):
            try:
                ImageSegmentationPreprocessingUtils.generate_image_mask_pairs_onecase(ref_mask_root, out_root, image_root, mask_root, suid, is_dcm, mask_pattern)
            except Exception as e:
                print('====> Error case:\t', suid)
                print(e)

    @staticmethod
    def generate_image_mask_pairs(ref_mask_root, 
            out_root, 
            image_root = '/data/medical/brain/Cerebrovascular/segmenation/images', 
            mask_root = '/data/medical/brain/Cerebrovascular/segmenation/renamed_masks', 
            process_num=12, 
            is_dcm=True, 
            mask_pattern='.mha'
        ):
        series_uids = []
        series_uids = os.listdir(image_root)
        num_per_process = (len(series_uids) + process_num - 1)//process_num

        # this for single thread to debug
        # ImageSegmentationPreprocessingUtils.generate_image_mask_pairs_singletask(
        #                 ref_mask_root, out_root, image_root, mask_root, series_uids, is_dcm, mask_pattern)

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
            result = pool.apply_async(ImageSegmentationPreprocessingUtils.generate_image_mask_pairs_singletask, 
                args=(ref_mask_root, out_root, image_root, mask_root, sub_series_uids, is_dcm, mask_pattern))
            results.append(result)

        pool.close()
        pool.join()    



def test_analyze_mask_boundary():
    # 1. 分析冠脉mask边界信息
    ImageSegmentationPreprocessingUtils.analyze_mask_boundary(
        '/data/medical/cardiac/seg/coronary/coronary_ori/masks', 
        out_root = '/data/medical/cardiac/seg/coronary/coronary_ori/result/analysis_result',
        out_filename='boundary_info_ori_mask.txt'
    )


if __name__ == '__main__':
    test_analyze_mask_boundary()