# global values used in model/train/test
import os
from enum import Enum

__all__ = ['model_cfg', 'train_cfg', 'test_cfg', 'val_cfg',
            'TrainMode']

class TrainMode(Enum):
    FEATURE_EXTRACT = 0
    FINE_TUNE = 1
    TWO_STEP_TRAIN = 2

data_dir = "/home/linux_fhb/data/hymenoptera_data"

# model parameters
model_cfg = {
    'model_name': 'resnet50',
    'classify_layers': [2],
    'pretrained': True,
    'cuda': True,
    'load_path':None,
}

# train parameters
train_cfg = {
    'dir': os.path.join(data_dir, 'train'),
    'batch_size': 32,
    'num_epochs': 3,
    'train_mode': TrainMode.FEATURE_EXTRACT,
    'train_rate': 1.0,
    'shuffle': False,
    'lr': 1e-3,
    'train_data_dir': 'train_data',
    'save_best': True,
}

# val parameters
val_cfg = {
    'dir': os.path.join(data_dir, 'val'),
    'batch_size': 32,
}

# test parameters
test_cfg = {
    'dir': None,
}

