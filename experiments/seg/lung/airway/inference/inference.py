import os
import sys
import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir, os.path.pardir, os.path.pardir, os.path.pardir)
sys.path.append(ROOT)

from segmentation.runner.train_seg import inference, train
from segmentation.options.train_options import TrainOptions

def load_inference_opts():
    opts = TrainOptions().parse()
    opts.num_classes = 2
    opts.base_n_filter = 8
    # opts.dynamic_size = [256, 256, 256]
    opts.dynamic_size = None
    opts.weights = './checkpoints/chamber/common_seg_epoch_144_train_0.047'

    return opts
