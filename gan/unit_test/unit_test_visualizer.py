import os
import sys
import time

import numpy as np
from collections import OrderedDict

import torch

sys.path.append(os.path.join(os.path.dirname(__file__), os.path.pardir))

from util.visualizer import Visualizer
from options.train_options import TrainOptions

opt = TrainOptions().parse()

visualizer = Visualizer(opt)


for i in range(100):
    visualizer.reset()
    images = OrderedDict()
    images['0'] = np.random.randint(0, 255, size=(224, 224, 3))
    images['1'] = np.random.randint(0, 255, size=(224, 224, 3))

    visualizer.display_current_results(images, 0, 0)
    print(i)
    time.sleep(1)


print('hello world!')