import os
import SimpleITK as sitk

import numpy as np
from tqdm import tqdm

import time

class ImagePostProcessingUtils:
    def __init__(self):
        pass

    @staticmethod
    def get_maximal_connected_region_singlelabel(sitk_mask):
        """
        only support for one class label output (0 and 1 for back/fore-ground
        :param sitk_mask:
        :return:
        """
        sitk_mask = sitk.Cast(sitk_mask, sitk.sitkUInt8)
        # get the connected components
        sitk_connect_componet = sitk.ConnectedComponent(sitk_mask)

        # statistic analysis of these components
        statsFilter = sitk.LabelIntensityStatisticsImageFilter()
        statsFilter.Execute(sitk_connect_componet, sitk_mask)
        areas = []
        labels = []
        for label in statsFilter.GetLabels():
            bbox = statsFilter.GetBoundingBox(label)
            area = statsFilter.GetNumberOfPixels(label)
            labels.append(label)
            areas.append(area)

        # get the max region
        id = areas.index(max(areas))
        label_value = labels[id]
        np_connecte_component = sitk.GetArrayFromImage(sitk_connect_componet)

        # 0 for background and 1 for forground
        a = np_connecte_component
        a[a != label_value] = 0
        a[a == label_value] = 1

        sitk_maximal_region = sitk.GetImageFromArray(a)
        # sitk_maximal_region.SetSpacing(sitk_mask.GetSpacing())
        # sitk_maximal_region.SetDirection(sitk_mask.GetDirection())
        # sitk_maximal_region.SetOrigin(sitk_mask.GetOrigin())
        sitk_maximal_region.CopyInformation(sitk_mask)

        return sitk_maximal_region, a


    @staticmethod
    def get_maximal_connected_region_multilabel(in_mask, mask_labels=[1]):
        in_mask_arr = sitk.GetArrayFromImage(in_mask)

        out_mask_arr = np.zeros(in_mask_arr.shape, dtype=np.uint8)

        beg = time.time()
        for mask_label in tqdm(mask_labels):
            sub_mask_arr = in_mask_arr.copy()
            sub_mask_arr[sub_mask_arr != mask_label] = 0
            sub_mask_arr[sub_mask_arr == mask_label] = 1
            sub_mask = sitk.GetImageFromArray(sub_mask_arr)
            sub_mask.CopyInformation(in_mask)
            sitk_maximal_region, tmp_arr = ImagePostProcessingUtils.get_maximal_connected_region_singlelabel(sub_mask)
            out_mask_arr[tmp_arr == 1] = mask_label
        out_mask_sitk = sitk.GetImageFromArray(out_mask_arr)
        out_mask_sitk.CopyInformation(in_mask)
        return out_mask_sitk




def test_get_maximal_connected_region_multilabel():
    in_mask_file = '/home/zhangwd/code/work/Lung_COPD/data/copd_400/registried_exp/1.3.12.2.1107.5.1.4.73793.30000017062123413576900064037/mask_pred.nii.gz'
    in_mask = sitk.ReadImage(in_mask_file)

    beg = time.time()
    out_mask_sitk = ImagePostProcessingUtils.get_maximal_connected_region_multilabel(in_mask, mask_labels=[1, 2, 3])

    out_dir = './tmp_out'
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, 'test_get_maximal_connected_region_multilabel.nii.gz')
    sitk.WriteImage(out_mask_sitk, out_file)

    end = time.time()

    print('====> test_get_maximal_connected_region_multilabel time elapsed:\t{:.3f}s'.format(end-beg))


if __name__ == '__main__':
    test_get_maximal_connected_region_multilabel()
    

