from .resnet import *
from .alexnet import *
from .vgg import *
from .squeezenet import *
from .densenet import *
from .inception import *

model_dicts = [resnet_dict, alex_dict, vgg_dict, squeezenet_dict, densenet_dict, inception_dict]
model_names = []
for model_dict in model_dicts:
    model_names += list(model_dict)

def initialize_model(model_name, classify_layers, pretrained=True):
    assert model_name in model_names, f"model name Error, {model_name} not in {model_names}"

    kwargs = {
        'model_name': model_name,
        'classify_layers': classify_layers,
        'pretrained': pretrained,
    }
    if model_name.startswith('resnet'):
        return initialize_resnet(**kwargs)
    elif model_name.startswith('alexnet'):
        return initialize_alex(**kwargs)
    elif model_name.startswith('vgg'):
        return initialize_vgg(**kwargs)
    elif model_name.startswith('squeezenet'):
        return initialize_squeezenet(**kwargs)
    elif model_name.startswith('densenet'):
        return initialize_densenet(**kwargs)
    elif model_name.startswith('inception'):
        return initialize_inception(**kwargs)
