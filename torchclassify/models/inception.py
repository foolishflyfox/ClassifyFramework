import torch
from torch import nn
from torchvision import models

inception_dict = {
    'inception_v3': models.inception_v3,
}

def initialize_inception(model_name, classify_layers, pretrained):
    assert model_name in inception_dict, f"{model_name} not in {list(inception_dict)}"
    model_ft = inception_dict[model_name](pretrained)
    num_ftrs = model_ft.fc.in_features
    post_layers = [nn.Linear(num_ftrs, classify_layers[0])]
    for i in range(1, len(classify_layers)):
        post_layers.append(nn.Linear(classify_layers[i-1], classify_layers[i]))
    model_ft.fc = nn.Sequential(*post_layers)
    num_ftrs = model_ft.AuxLogits.fc.in_features
    post_layers = [nn.Linear(num_ftrs, classify_layers[0])]
    for i in range(1, len(classify_layers)):
        post_layers.append(nn.Linear(classify_layers[i-1], classify_layers[i]))
    model_ft.AuxLogits.fc = nn.Sequential(*post_layers)
    model_ft.custom_layers = nn.ModuleList([model_ft.AuxLogits.fc, model_ft.fc])
    return model_ft, (299, 299)

