import os
import sys
import numpy as np
import cv2

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
sys.path.append(ROOT)

from utils.datasets_utils import DatasetsUtils
import SimpleITK as sitk

class ImageShowUtils:
    def __init__(self):
        pass

    @staticmethod
    def save_img(in_arr, out_file, ww=150, wl=50, lut=None):
        min_v = wl-ww//2
        max_v = wl+ww//2
        out_arr = np.clip(in_arr, min_v, max_v)
        out_arr = (out_arr-min_v)/ww
        if lut:
            out_arr = lut(out_arr)*255
            cv2.imwrite(out_file, cv2.cvtColor(np.array(out_arr, dtype=np.uint8), cv2.COLOR_RGB2BGR))
        else:
            out_arr = out_arr*255
            cv2.imwrite(out_file, out_arr)

    @staticmethod
    def save_volume_to_jpg(in_arr, out_root, ww, wl, axis=0, file_prefix=None, reverse=False, lut_name=None):
        '''
        in_arr = sitk.GetArrayFromImage(sitk_image)

        h-f: axis=0
        a-p: axis=1
        l-r: axis=2
        '''
        os.makedirs(out_root, exist_ok=True)
        n = in_arr.shape[axis]
        lut = None
        if lut_name:
            import matplotlib.pyplot as plt
            lut = plt.get_cmap(lut_name)
        for i in range(n):
            if file_prefix:
                file_name = '{}_{}.jpg'.format(file_prefix, i)
            else:
                file_name = '{}.jpg'.format(i)
            sub_file_name = os.path.join(out_root, file_name)
            if axis == 0:
                tmp_arr = in_arr[i,:,:]
            elif axis == 1:
                tmp_arr = in_arr[:,i,:]
            else:
                tmp_arr = in_arr[:,:,i]
            if reverse:
                tmp_arr = tmp_arr[::-1,:]
            ImageShowUtils.save_img(tmp_arr, sub_file_name, ww, wl, lut)

    @staticmethod
    def save_volume_to_mpr_jpg(in_image, out_dir, ww=150, wl=50, prefix='xxx'):
        resampled_img = DatasetsUtils.resample_unified_spacing_x_default_min(in_image)
        arr = sitk.GetArrayFromImage(resampled_img)
        
        [z,y,x] = arr.shape
        z_plane = arr[z//2, :, :]
        y_plane = arr[:, y//2, :]
        x_plane = arr[:,:,x//2]

        os.makedirs(out_dir, exist_ok=True)
        z_file = os.path.join(out_dir, '{}_z.jpg'.format(prefix))
        y_file = os.path.join(out_dir, '{}_y.jpg'.format(prefix))
        x_file = os.path.join(out_dir, '{}_x.jpg'.format(prefix))

        cv2.imwrite(z_file, z_plane)
        cv2.imwrite(y_file, y_plane)
        cv2.imwrite(x_file, x_plane)


def test_save_volume_to_jpg():
    print('todo')

def test_save_volume_to_mpr_jpg():
    infile = '/data/medical/brain/gan/cta2dwi_history_pos/5.train_batch/1014186/fixed_cta.nii.gz'
    image = sitk.ReadImage(infile)
    ImageShowUtils.save_volume_to_mpr_jpg(image, '/data/medical/tmp/mpr')

if __name__ == '__main__':
    # test_save_volume_to_jpg()
    test_save_volume_to_mpr_jpg()

