import torch
from torch import nn
from torchvision import models

squeezenet_dict = {
    'squeezenet1_0': models.squeezenet1_0,
    'squeezenet1_1': models.squeezenet1_1,
}

def initialize_squeezenet(model_name, classify_layers, pretrained):
    assert model_name in squeezenet_dict, f"{model_name} not in {list(squeezenet_dict)}"
    assert len(classify_layers)==1, f"squeezenet classify layers is 1 (conv2d),"\
            f" given {len(classify_layers)}"
    model_ft = squeezenet_dict[model_name](pretrained)
    num_ftrs = model_ft.classifier[1].in_channels
    model_ft.classifier[1] = nn.Conv2d(num_ftrs, classify_layers[0], 
                                    kernel_size=(1,1), stride=(1,1))
    model_ft.custom_layers = model_ft.classifier[1]
    model_ft.num_classes = classify_layers[0]
    return model_ft, (224, 224)

