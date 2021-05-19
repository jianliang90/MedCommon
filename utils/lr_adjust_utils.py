import os
import sys
import math

COMMON_ROOT = os.path.join(os.path.dirname(__file__), os.path.pardir)
sys.path.append(COMMON_ROOT)

class LR_ADJUST_UTILS:
    def __init__(self) -> None:
        pass

    @staticmethod
    def adjust_learning_rate(optimizer, epoch, lr, cos, epochs, schedule=None):
        """Decay the learning rate based on schedule"""
        lr = lr
        if cos:  # cosine lr schedule
            lr *= 0.5 * (1. + math.cos(math.pi * epoch / epochs))
        else:  # stepwise lr schedule
            for milestone in schedule:
                lr *= 0.1 if epoch >= milestone else 1.
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr