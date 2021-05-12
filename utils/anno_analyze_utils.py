import os
import SimpleITK as sitk

import sys
root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
print(root)
sys.path.append(root)
from utils.data_io_utils import DataIO

class DetectionAnnotationAnalyzeUtils:
    def __init__(self):
        pass

    def PysicalCoordinate2PixelCoordinate(pysical_coord, image):
        spacing = image.GetSpacing()
        origin = image.GetOrigin()
        derect = image.GetDirection()
        pixel_coord = [int((pysical_coord[i] - origin[i])/spacing[i]) for i in range(3)]
        return pixel_coord

    def extract_block_around_center(arr, center_coord, block_size):
        boundary_min = [max(0, center_coord[i]-block_size//2) for i in range(3)]
        boundary_max = [min(center_coord[i]+block_size, arr.shape[2-i]) for i in range(3)]

        return arr[boundary_min[2]:boundary_max[2], boundary_min[1]:boundary_max[1], boundary_min[0]:boundary_max[0]]

    def extract_block_around_center_image(image, center_coord, block_size):
        arr = sitk.GetArrayFromImage(image)
        new_arr = DetectionAnnotationAnalyzeUtils.extract_block_around_center(arr, center_coord, block_size)
        new_image = sitk.GetImageFromArray(new_arr)
        new_image.SetOrigin([0,0,0])
        new_image.SetSpacing(image.GetSpacing())
        new_image.SetDirection(image.GetDirection())
        return new_image


def test_DetectionAnnotationAnalyzeUtils():
    import pandas as pd
    anno_file = '/fileser/zhangwd/data/hospital/cz/ggo/annotation/标注结果-长征非测试集(阳性）/01回流数据信息表（UID)-DD1209V6- 阳（阴）性含测试集数据数量表格.xlsx'
    df = pd.read_excel(anno_file, sheet_name=1)

    center_col_name = ['coordX', 'coordY', 'coordZ']

    pt1_col_name = ['x1', 'y1', 'z1']
    pt2_col_name = ['x2', 'y2', 'z2']

    # index = df['series uid'].to_list().index('1.3.46.670589.33.1.63602016093241541400001.5555111662490526388')
    index = df['series uid'].to_list().index('1.2.392.200036.9116.2.2.2.1762671075.1411351131.360464')

    coord_center = df.iloc[index][center_col_name]
    coord_pt1 = df.iloc[index][pt1_col_name]
    coord_pt2 = df.iloc[index][pt2_col_name]

    image_file = '/data/zhaokeyang/Lung_GGO/{}'.format(df.iloc[index]['series uid'])
    image = DataIO.load_dicom_series(image_file)['sitk_image']

    pix_coord_center = DetectionAnnotationAnalyzeUtils.PysicalCoordinate2PixelCoordinate(coord_center, image)
    pix_coord_pt1 = DetectionAnnotationAnalyzeUtils.PysicalCoordinate2PixelCoordinate(coord_pt1, image)
    pix_coord_pt2 = DetectionAnnotationAnalyzeUtils.PysicalCoordinate2PixelCoordinate(coord_pt2, image)

    block_size = 68

    image_block = DetectionAnnotationAnalyzeUtils.extract_block_around_center_image(image, pix_coord_center, block_size)

    out_file = os.path.join('/data/medical/tmp/tmp.nii.gz')
    sitk.WriteImage(image_block, out_file)

    print('====> test_DetectionAnnotationAnalyzeUtils')


if __name__ == '__main__':
    test_DetectionAnnotationAnalyzeUtils()