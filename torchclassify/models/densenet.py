import torch
from torch import nn
from torchvision import models

densenet_dict = {
    'densenet121': models.densenet121,
    'densenet161': models.densenet161,
    'densenet169': models.densenet169,
    'densenet201': models.densenet201,
}

def initialize_densenet(model_name, classify_layers, pretrained):
    assert model_name in densenet_dict, f"{model_name} not in {list(resnet_dict)}"
    model_ft = densenet_dict[model_name](pretrained)
    num_ftrs = model_ft.classifier.in_features
    post_layers = [nn.Linear(num_ftrs, classify_layers[0])]
    for i in range(1, len(classify_layers)):
        post_layers.append(nn.Linear(classify_layers[i-1], classify_layers[i]))
    model_ft.classifier = nn.Sequential(*post_layers)
    model_ft.custom_layers = model_ft.classifier
    return model_ft, (224, 224)

