import torch
from torch import nn
from torchvision import models

resnet_dict = {
    'resnet34': models.resnet34,
    'resnet50': models.resnet50,
    'resnet101': models.resnet101,
    'resnet152': models.resnet152,
}

def initialize_resnet(model_name, classify_layers, pretrained):
    assert model_name in resnet_dict, f"{model_name} not in {list(resnet_dict)}"
    model_ft = resnet_dict[model_name](pretrained)
    num_ftrs = model_ft.fc.in_features
    post_layers = [nn.Linear(num_ftrs, classify_layers[0])]
    for i in range(1, len(classify_layers)):
        post_layers.append(nn.Linear(classify_layers[i-1], classify_layers[i]))
    model_ft.fc = nn.Sequential(*post_layers)
    model_ft.custom_layers = model_ft.fc
    return model_ft, (224, 224)

