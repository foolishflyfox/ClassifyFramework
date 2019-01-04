import torch
from torch import nn
from torchvision.models import resnet34, resnet50, resnet101, resnet152

resnet_dict = {
    'resnet34': resnet34,
    'resnet50': resnet50,
    'resnet101': resnet101,
    'resnet152': resnet152,
}

class Resnet(nn.Module):
    def __init__(self, model_name, post_layers, pretrained=True):
        assert model_name in resnet_dict, f"resnet name should in {list(resnet_dict.keys())}"
        self.model_name = model_name
        self.num_cls = num_cls
        


def resnet50():

