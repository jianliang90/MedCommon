import os
import sys

class ImageShowUtils:
    def __init__(self):
        pass

    @staticmethod
    def save_img(in_arr, out_file, ww=150, wl=50):
        min_v = wl-ww//2
        max_v = wl+ww//2
        out_arr = np.clip(in_arr, min_v, max_v)
        out_arr = (out_arr-min_v)/ww*255
        cv2.imwrite(out_file, out_arr)

    def save_volume_to_jpg(in_arr, out_root, ww, wl, axis=0, file_prefix=None):
        '''
        in_arr = sitk.GetArrayFromImage(sitk_image)

        h-f: axis=0
        a-p: axis=1
        l-r: axis=2
        '''
        n = in_arr.shape[axis]
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
            ImageShowUtils.save_img(tmp_arr, sub_file_name, ww, wl)



def test_save_volume_to_jpg():
    print('todo')

if __name__ == '__main__':
    test_save_volume_to_jpg()

