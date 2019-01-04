# global values used in model/train/test
import os
from enum import Enum

__all__ = ['model_cfg', 'train_cfg', 'test_cfg', 'TrainMode']

class TrainMode(Enum):
    FEATURE_EXTRACT = 0
    FINE_TUNE = 1
    TWO_STEP_TRAIN = 2

data_dir = "/home/linux_fhb/data/cat_vs_dog"

# model parameters
model_cfg = {
    'num_classes': 2
}

# train parameters
train_cfg = {
    'dir': os.path.join(data_dir, 'train'),
    'batch_size': 32,
    'num_epochs': 30,
    'train_mode': TrainMode.FINE_TUNE,
}

# test parameters
test_dir = {
    'dir': os.path.join(data_dir, 'test'),
}

