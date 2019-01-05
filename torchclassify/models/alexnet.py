import torch
from torch import nn
from torchvision import models

alex_dict = {
    'alexnet': models.alexnet,
}

def initialize_alex(model_name, classify_layers, pretrained):
    assert model_name in alex_dict, f"{model_name} not in {list(alex_dict)}"
    model_ft = alex_dict[model_name](pretrained)
    num_ftrs = model_ft.classifier[-1].in_features
    post_layers = [nn.Linear(num_ftrs, classify_layers[0])]
    for i in range(1, len(classify_layers)):
        post_layers.append(nn.Linear(classify_layers[i-1], classify_layers[i]))
    model_ft.classifier[-1] = nn.Sequential(*post_layers)
    model_ft.custom_layers = model_ft.classifier[-1]
    return model_ft, (224, 224)

