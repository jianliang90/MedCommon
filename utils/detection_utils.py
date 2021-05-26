import os
import sys

import numpy as np

import torch

class DETECTION_UTILS:
    def __init__(self) -> None:
        pass

    @staticmethod
    def calc_brick_volume(bricks):
        '''
        bricks: [N, x0, y0, z0, x1, y1, z1]
        '''
        delta_bricks = bricks[:, 3:] - bricks[:,:3]
        # assert np.all(delta_bricks > 0)
        delta_bricks.clip(min = 0)
        volumes = delta_bricks[:,0] * delta_bricks[:,1] * delta_bricks[:,2]
        return volumes
        
    @staticmethod
    def calc_brick_iou(bricks1, bricks2):
        '''
        boxes: [N, x0,y0,z0,x1,y1,z1]
        '''
        v1 = DETECTION_UTILS.calc_brick_volume(bricks1)
        v2 = DETECTION_UTILS.calc_brick_volume(bricks2)

        pt_min = np.maximum(bricks1[:, :3], bricks2[:, :3])
        pt_max = np.minimum(bricks1[:, 3:], bricks2[:, 3:])
        
        whd = (pt_max - pt_min).clip(min=0)
        inter = whd[:, 0] * whd[:, 1] * whd[:, 2]
        union = v1 + v2 - inter

        iou = inter / union

        return iou, union

    @staticmethod
    def point_coordinate_resampled(in_shape, out_shape, in_coord):
        '''
        '''
        out_coord = [out_shape[i]/in_shape[i]*in_coord[i] for i in range(3)]
        return out_coord

    @staticmethod
    def point_coordinate_resampled_normalized(in_shape, out_shape, in_coord):
        '''
        '''
        out_coord = [1/in_shape[i]*in_coord[i] for i in range(3)]
        return out_coord

    @staticmethod
    def restore_normalized_coordinate(in_normalized_coord, in_shape):
        out_coord = [in_normalized_coord[i] * in_shape[i] for i in range(3)]
        return out_coord


    @staticmethod
    def generate_test_bricks(n):
        bricks = np.zeros([n,6])
        for i in range(bricks.shape[0]):
            bricks[i, :3] = np.random.randint(0,4,[3])
            delta_edge = np.random.randint(4,7)
            bricks[i, 3:] = bricks[i, :3] + delta_edge
        return bricks


class PYTORCH_TENSOR_DETECTION_UTILS:
    def __init__(self) -> None:
        pass
    @staticmethod
    def calc_brick_volume(bricks):
        '''
        bricks: [N, x0, y0, z0, x1, y1, z1]
        '''
        delta_bricks = bricks[:, 3:] - bricks[:,:3]
        # assert torch.all(delta_bricks > 0)
        delta_bricks.clip(min = 0)
        volumes = delta_bricks[:,0] * delta_bricks[:,1] * delta_bricks[:,2]
        return volumes

    @staticmethod
    def calc_brick_iou(bricks1, bricks2):
        '''
        boxes: [N, x0,y0,z0,x1,y1,z1]
        '''
        v1 = PYTORCH_TENSOR_DETECTION_UTILS.calc_brick_volume(bricks1)
        v2 = PYTORCH_TENSOR_DETECTION_UTILS.calc_brick_volume(bricks2)

        pt_min = torch.max(bricks1[:, :3], bricks2[:, :3])
        pt_max = torch.min(bricks1[:, 3:], bricks2[:, 3:])
        
        whd = (pt_max - pt_min).clip(min=0)
        inter = whd[:, 0] * whd[:, 1] * whd[:, 2]
        union = v1 + v2 - inter

        iou = inter / union

        return iou, union


def test_brick_volume():
    bricks = DETECTION_UTILS.generate_test_bricks(6)
    print('====> begin to test DETECTION_UTILS.calc_brick_volume')
    volumes = DETECTION_UTILS.calc_brick_volume(bricks)
    for i in range(bricks.shape[0]):
        print('bricks:\t', bricks[i], '\tvolume:\t{}'.format(volumes[i]))
    print('====> end to test DETECTION_UTILS.calc_brick_volume')

    print('====> begin to test PYTORCH_TENSOR_DETECTION_UTILS.calc_brick_volume')
    bricks = torch.from_numpy(bricks)
    volumes = PYTORCH_TENSOR_DETECTION_UTILS.calc_brick_volume(bricks)
    for i in range(bricks.shape[0]):
        print('bricks:\t', bricks[i], '\tvolume:\t{}'.format(volumes[i]))
    print('====> end to test PYTORCH_TENSOR_DETECTION_UTILS.calc_brick_volume')


def test_calc_brick_iou():
    bricks1 = DETECTION_UTILS.generate_test_bricks(6)
    bricks2 = DETECTION_UTILS.generate_test_bricks(6)
    print('====>begin to test DETECTION_UTILS.calc_brick_iou')
    ious, unions = DETECTION_UTILS.calc_brick_iou(bricks1, bricks2)
    for i in range(bricks1.shape[0]):
        print('bricks:\t', bricks1[i], '\tiou:\t{}'.format(ious[i]))
    print('====>end to test DETECTION_UTILS.calc_brick_iou')

    print('====>begin to test PYTORCH_TENSOR_DETECTION_UTILS.calc_brick_iou')
    bricks1 = torch.from_numpy(bricks1)
    bricks2 = torch.from_numpy(bricks2)
    ious, unions = PYTORCH_TENSOR_DETECTION_UTILS.calc_brick_iou(bricks1, bricks2)
    for i in range(bricks1.shape[0]):
        print('bricks:\t', bricks1[i], '\tiou:\t{}'.format(ious[i]))
    print('====>end to test PYTORCH_TENSOR_DETECTION_UTILS.calc_brick_iou')    


def test_point_coordinate_resampled():
    inshape = [512, 512, 243]
    outshape = [128, 128, 128]
    in_pt = [18, 23, 24]
    out_pt = DETECTION_UTILS.point_coordinate_resampled(inshape, outshape, in_pt)
    print('test_point_coordinate_resampled')

if __name__ == '__main__':
    # test_brick_volume()
    # test_calc_brick_iou()
    test_point_coordinate_resampled()


    
        