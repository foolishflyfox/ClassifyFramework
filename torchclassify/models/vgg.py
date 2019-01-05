import torch
from torch import nn
from torchvision import models

vgg_dict = {
    'vgg11': models.vgg11,
    'vgg11_bn': models.vgg11_bn,
    'vgg13': models.vgg13,
    'vgg13_bn': models.vgg13_bn,
    'vgg16': models.vgg16,
    'vgg16_bn': models.vgg16_bn,
    'vgg19': models.vgg19,
    'vgg19_bn': models.vgg19_bn,
}

def initialize_vgg(model_name, classify_layers, pretrained):
    assert model_name in vgg_dict, f"{model_name} not in {list(vgg_dict)}"
    model_ft = vgg_dict[model_name](pretrained)
    num_ftrs = model_ft.classifier[-1].in_features
    post_layers = [nn.Linear(num_ftrs, classify_layers[0])]
    for i in range(1, len(classify_layers)):
        post_layers.append(nn.Linear(classify_layers[i-1], classify_layers[i]))
    model_ft.classifier[-1] = nn.Sequential(*post_layers)
    model_ft.custom_layers = model_ft.classifier[-1]
    return model_ft, (224, 224)

