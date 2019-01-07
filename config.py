# global values used in model/train/test
import os
from enum import Enum

__all__ = ['model_cfg', 'train_cfg', 'test_cfg', 'val_cfg',
            'TrainMode', 'imagenet_mean', 'imagenet_std']

class TrainMode(Enum):
    FEATURE_EXTRACT = 0
    FINE_TUNE = 1
    TWO_STEP_TRAIN = 2

# data_dir = "/home/linux_fhb/data/hymenoptera_data"
data_dir = "/home/linux_fhb/data/cat_vs_dog"

# model parameters
model_cfg = {
    'model_name': 'resnet50',
    'classify_layers': [2],
    'pretrained': True,
    'cuda': True,
    'load_path': None,#"/home/linux_fhb/NfsEdit/CatVsDog/train_result/catvsdog/best_model.pt",
}

# train parameters
train_cfg = {
    'dir': os.path.join(data_dir, 'train'),
    'class_to_idx': {'cat': 0, 'dog':1},
    'train_mode': TrainMode.FEATURE_EXTRACT,
    'train_rate': 0.8,
    'shuffle': False,
    'batch_size': 32,
    'num_epochs': 3,
    'lr': 1e-3,
    'train_result_dir': 'train_result',
    'train_subdir': "train_5epochs",
    'save_best': True,
}
# test parameters
test_cfg = {
    'dir': "/home/linux_fhb/data/cat_vs_dog/test",
    'model_dir': "/home/linux_fhb/NfsEdit/CatVsDog/train_result/train_5epochs",
    'batch_size': 32,
    'num_works': 8,
    'result_dir':'test_result',
    'result_filename':None,
}

# val parameters
val_cfg = {
    'dir': None, #os.path.join(data_dir, 'val'),
    'batch_size': 32,
}

imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std = [0.229, 0.224, 0.225]

