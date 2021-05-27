import os
import sys
import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir, os.path.pardir, os.path.pardir, os.path.pardir)
sys.path.append(ROOT)

sys.path.append(os.path.join(os.path.dirname(__file__), os.path.pardir))
sys.path.append(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir))
sys.path.append(os.path.join(ROOT, 'segmentation/external_lib/MedicalZooPytorch'))

from segmentation.runner.train_seg import train, inference


if __name__ == '__main__':
    # train()
    # inference('/data/medical/brain/Cerebrovascular/segmenation_erode_k1/images/1.3.12.2.1107.5.1.4.60320.30000018112600105502700009814.nii.gz', '/data/medical/brain/Cerebrovascular/segmenation_erode_k1/inference/exp1', [288, 288, 320], '/data/medical/brain/Cerebrovascular/segmenation_erode_k1/checkpoints/cerebravascular/common_seg_epoch_207_train_0.037')
    # inference('/data/medical/brain/Cerebrovascular/segmenation_erode_k1/images/1.3.12.2.1107.5.1.4.60320.30000018081400152828100052071.nii.gz', '/data/medical/brain/Cerebrovascular/segmenation_erode_k1/inference/exp1', [288, 288, 320], '/data/medical/brain/Cerebrovascular/segmenation_erode_k1/checkpoints/cerebravascular/common_seg_epoch_207_train_0.037')
    inference('/data/medical/brain/Cerebrovascular/segmenation_erode_k1/images/1.3.12.2.1107.5.1.4.60320.30000018112600105502700009814.nii.gz', '/data/medical/brain/Cerebrovascular/segmenation_erode_k1/inference/exp1', [288, 288, 320], '/data/medical/brain/Cerebrovascular/segmenation_erode_k1/checkpoints/cerebravascular/common_seg_epoch_278_train_0.035')
    inference('/data/medical/brain/Cerebrovascular/segmenation_erode_k1/images/1.3.12.2.1107.5.1.4.60320.30000018081400152828100052071.nii.gz', '/data/medical/brain/Cerebrovascular/segmenation_erode_k1/inference/exp1', [288, 288, 320], '/data/medical/brain/Cerebrovascular/segmenation_erode_k1/checkpoints/cerebravascular/common_seg_epoch_278_train_0.035')
    inference('/data/medical/brain/Cerebrovascular/segmenation_erode_k1/images/1.3.12.2.1107.5.1.4.60320.30000016032600140635800002350.nii.gz', '/data/medical/brain/Cerebrovascular/segmenation_erode_k1/inference/exp1', [288, 288, 320], '/data/medical/brain/Cerebrovascular/segmenation_erode_k1/checkpoints/cerebravascular/common_seg_epoch_278_train_0.035')
