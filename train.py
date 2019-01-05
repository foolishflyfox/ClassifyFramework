from model_cfg import model_cfg, train_cfg
from torchclassify import models
import torch

if model_cfg['cuda'] and not torch.cuda.is_available():
    model_cfg['cuda'] = False

